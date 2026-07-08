"""Whole-step CUDA-graph capture for the pre-Poisson fluid region.

The whole-step pre-projection runner captures the entire pre-Poisson region
(semi-Lagrangian advection, diffusion accumulate, ``bdim_forcing``, ``apply_bcs``)
as ONE CUDA graph via a single ``wp.ScopedCapture``.  Per-kernel runners are
pure launch wrappers — their raw ``wp.launch`` calls are recorded by the outer
``wp.ScopedCapture``; outside the graph they run standalone.

This module provides only the :class:`WholeStepGraphRunner` itself.  The
former re-entrancy flag (``in_capture`` / ``capturing``) was removed in
Task 7 of ``unified_graph_capture_plan.md`` — per-kernel runners no longer
have their own graph capture/replay, so there is no nested-``capture_launch``
hazard to guard against.
"""

import warp as wp


class WholeStepGraphRunner:
    """Capture-and-replay a multi-kernel region as ONE CUDA graph.

    Collapses ~5 per-kernel ``wp.launch`` dispatches (semi-Lagrangian
    advection, diffusion accumulate, ``bdim_forcing``, ``apply_bcs``) into a
    SINGLE captured graph — ~3 µs replay instead of ~130 µs of per-kernel
    graph-replay host overhead.

    Follows the same staging + capture pattern as :class:`ForcesPostGraph`:

    1. ``stage()`` runs OUTSIDE the graph: stages per-step data (bdim dirty
       rect) into persistent device buffers and synchronises (so the graph
       reads fresh data on every replay and no ``torch.cuda.synchronize``
       lives inside the capture — the source of CUDA error 900).
    2. ``issue()`` runs INSIDE ``wp.ScopedCapture``.  The sub-runners are
       pure launch wrappers that always issue raw ``wp.launch`` — these are
       recorded into the outer graph.
    3. Graphs are keyed on the pointer signature of all live tensors; growth
       or reallocation silently drops the cache and the next sighting
       recaptures.  Up to ``max_graphs`` distinct signatures are cached.
    4. Replays: ``stage()`` copies the new per-step data, then a single
       ``wp.capture_launch(graph)`` replays the whole pre-Poisson region.

    Counters ``replays``/``captures``/``eager`` let benchmarks verify the
    fast path engaged.
    """

    __slots__ = ("replays", "captures", "eager",
                 "_graphs", "_seen", "_max_graphs",
                 "_use_cuda_graph")

    def __init__(self, max_graphs: int = 4, use_cuda_graph: bool = True):
        self.replays = 0
        self.captures = 0
        self.eager = 0
        self._graphs: dict = {}          # key -> wp.Graph
        self._seen: dict = {}            # key -> sighting count
        self._max_graphs = int(max_graphs)
        self._use_cuda_graph = use_cuda_graph

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
            Ignored when ``_use_cuda_graph`` is False.
        issue : callable
            Zero-argument closure that performs the region's work (must use
            only raw ``wp.launch`` calls — no ``wp.capture_launch``).
        stage : callable or None
            Zero-argument closure that stages per-step data into persistent
            device buffers and synchronises — called OUTSIDE the capture
            (both during capture and on every replay).  Ignored when
            ``_use_cuda_graph`` is False.
        """
        # ---- eager path (CPU, or CUDA graphs disabled) ----
        if not self._use_cuda_graph:
            self.eager += 1
            issue()
            return

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
        with wp.ScopedCapture(device=device) as cap:
            issue()
        if len(self._graphs) < self._max_graphs:
            self._graphs[key] = cap.graph
            self.captures += 1
        else:
            # Cache full — stay eager for this signature (correct, not
            # accelerated).  The graph we just built is discarded.
            pass
