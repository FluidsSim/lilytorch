"""Shared re-entrancy flag for whole-step CUDA-graph capture.

The per-kernel graph runners (semi-Lagrangian advection, diffusion accumulate,
``bdim_forcing``, ``apply_bcs``) each normally manage their OWN captured graph
and replay it with ``wp.capture_launch``.  The whole-step pre-projection runner
captures the entire pre-Poisson region as ONE graph, by re-issuing those
runners' RAW ``wp.launch`` calls inside a single ``wp.ScopedCapture``.

Nesting is the reason this flag exists: calling ``wp.capture_launch`` (a graph
*launch*) while a stream is already capturing is illegal — Warp raises CUDA
error 900 ("operation not permitted when stream is capturing").  Verified on
Warp 1.14.  So while the whole-step runner builds (or re-runs) its region, it
sets this flag; each per-kernel runner sees it and issues its raw launch
(recorded into the outer graph) instead of its own capture/replay.

Depth-counted so nested whole-step regions (should never happen, but cheap to
be safe) compose correctly.
"""

import warp as wp

_DEPTH = 0


def in_capture() -> bool:
    """True while a whole-step capture region is being built or re-run — the
    per-kernel runners must then issue raw ``wp.launch`` calls, not
    ``wp.capture_launch``."""
    return _DEPTH > 0


class capturing:
    """Context manager marking the enclosed block as part of a whole-step
    capture region (see :func:`in_capture`)."""

    def __enter__(self):
        global _DEPTH
        _DEPTH += 1
        return self

    def __exit__(self, *exc):
        global _DEPTH
        _DEPTH -= 1
        return False


class WholeStepGraphRunner:
    """Capture-and-replay a multi-kernel region as ONE CUDA graph.

    Collapses ~5 per-kernel ``wp.capture_launch`` dispatches (semi-Lagrangian
    advection, diffusion accumulate, ``bdim_forcing``, ``apply_bcs``) into a
    SINGLE captured graph — ~3 µs replay instead of ~130 µs of per-kernel
    graph-replay host overhead.

    Follows the same staging + capture pattern as :class:`ForcesPostGraph`:

    1. ``stage()`` runs OUTSIDE the graph: stages per-step data (bdim dirty
       rect) into persistent device buffers and synchronises (so the graph
       reads fresh data on every replay and no ``torch.cuda.synchronize``
       lives inside the capture — the source of CUDA error 900).
    2. ``issue()`` runs INSIDE ``wp.ScopedCapture`` with the :func:`capturing`
       re-entrancy flag set, so every per-kernel runner sees
       ``_gc.in_capture() == True`` and issues its RAW ``wp.launch`` instead
       of its own ``wp.capture_launch`` (a nested capture_launch is illegal).
    3. Graphs are keyed on the pointer signature of all live tensors; growth
       or reallocation silently drops the cache and the next sighting
       recaptures.  Up to ``max_graphs`` distinct signatures are cached.
    4. Replays: ``stage()`` copies the new per-step data, then a single
       ``wp.capture_launch(graph)`` replays the whole pre-Poisson region.

    Counters ``replays``/``captures``/``eager`` let benchmarks verify the
    fast path engaged.
    """

    __slots__ = ("replays", "captures", "eager",
                 "_graphs", "_seen", "_max_graphs")

    def __init__(self, max_graphs: int = 4):
        self.replays = 0
        self.captures = 0
        self.eager = 0
        self._graphs: dict = {}          # key -> wp.Graph
        self._seen: dict = {}            # key -> sighting count
        self._max_graphs = int(max_graphs)

    def run(self, key, device, issue, stage=None):
        """Capture (on 2nd sighting) or replay (subsequent) the region.

        Parameters
        ----------
        key : tuple
            Pointer + scalar signature — usually ``(field_ptrs..., scalars...)``.
            Graph replayed when ``key`` matches a cached graph; dropped on
            mismatch (reallocation / growth).
        device : str
            CUDA device string, e.g. ``"cuda:0"``, for ``wp.ScopedCapture``.
        issue : callable
            Zero-argument closure that performs the region's work (pure Warp
            launches when ``_gc.in_capture()`` is active).
        stage : callable or None
            Zero-argument closure that stages per-step data into persistent
            device buffers and synchronises — called OUTSIDE the capture
            (both during capture and on every replay).
        """
        # ---- stage per-step data OUTSIDE the graph ----
        if stage is not None:
            stage()

        # ---- replay ----
        graph = self._graphs.get(key)
        if graph is not None:
            self.replays += 1
            wp.capture_launch(graph)
            return

        # ---- eager (first sighting = JIT / module warm-up) ----
        n = self._seen.get(key, 0) + 1
        self._seen[key] = n
        if n < 2:
            self.eager += 1
            issue()
            return

        # ---- capture (second sighting) ----
        # This step's REAL work first — also JIT/module warm-up.
        # Stream capture RECORDS without executing, so the pre-capture
        # ``issue()`` is this step's sole correct output.  Future steps
        # replay the captured graph.
        issue()
        with capturing():
            with wp.ScopedCapture(device=device) as cap:
                issue()
        if len(self._graphs) < self._max_graphs:
            self._graphs[key] = cap.graph
            self.captures += 1
        else:
            # Cache full — stay eager for this signature (correct, not
            # accelerated).  The graph we just built is discarded.
            pass
