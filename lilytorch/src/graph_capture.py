"""Whole-step CUDA-graph capture for the pre-Poisson fluid region.

The whole-step pre-projection runner captures the entire pre-Poisson region
(semi-Lagrangian advection, diffusion accumulate, ``bdim_forcing``, ``apply_bcs``)
as ONE CUDA graph and replays it with a single host launch.

Two backends live here during the ``cuda_native_port`` transition:

* :class:`WholeStepGraphRunner` — the Warp backend (``wp.ScopedCapture`` /
  ``wp.capture_launch``).  Records raw ``wp.launch`` calls only; still the
  production runner while the pre-Poisson region is Warp kernels.
* :class:`NativeWholeStepGraphRunner` — the ``torch.cuda.CUDAGraph`` backend
  (Phase 1.2 of ``milestones/cuda_native_port_plan.md``).  Records anything
  issued on torch's CURRENT stream: native ``TORCH_LIBRARY`` extension ops
  and plain torch ops alike.  Becomes the production runner once the region
  ops are native; the Warp class is deleted then.

The former re-entrancy flag (``in_capture`` / ``capturing``) was removed in
Task 7 of ``unified_graph_capture_plan.md`` — per-kernel runners no longer
have their own graph capture/replay, so there is no nested-``capture_launch``
hazard to guard against.
"""

from collections import OrderedDict

import torch
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


class NativeWholeStepGraphRunner:
    """``torch.cuda.CUDAGraph`` port of :class:`WholeStepGraphRunner`.

    Phase 1.2 of ``milestones/cuda_native_port_plan.md``: same
    ``run(key, device, issue, stage)`` contract and capture-on-second-sighting
    / replay life-cycle as the Warp runner, so the solver swap is a
    constructor change.  Differences from the Warp backend:

    * **What is recorded.**  Stream capture records whatever ``issue()``
      enqueues on torch's CURRENT CUDA stream — native ``TORCH_LIBRARY``
      extension ops (they launch via ``at::cuda::getCurrentCUDAStream()``)
      and plain torch ops.  Raw ``wp.launch`` calls go to Warp's own stream
      and are NOT recorded — mixing Warp kernels into ``issue()`` silently
      drops them from the replay (the ``use_cuda_graphs`` warp_port lesson).
    * **Eviction, never pinning.**  The keyed cache is LRU: when full, the
      least-recently-replayed graph is ``reset()`` (its private memory pool
      is freed) to admit the new capture.  The Warp runner instead kept old
      graphs and left new signatures eager — under allocator reshuffles that
      pins one dead graph per stale pointer signature (the salamander OOM).
    * **Staging.**  ``stage()`` still runs outside the graph, but replay is
      ordered on the SAME torch stream as ``stage()``'s ``copy_``, so torch
      callers need no hard ``torch.cuda.synchronize`` in ``stage()`` (the
      Warp backend needed one because replay ran on Warp's stream).
    * Temporaries allocated inside ``issue()`` during capture come from the
      graph's private memory pool and are reused verbatim on every replay —
      per-step *outputs* must therefore land in persistent buffers whose
      pointers are part of ``key``.

    Any host-side branch taken inside ``issue()`` (e.g. the Phase-2
    per-body-buffer regime A/B selection) is frozen into the captured graph:
    fold its discriminator into ``key`` so each branch gets its own graph.

    Counters ``replays``/``captures``/``eager``/``evictions`` let benchmarks
    verify the fast path engaged.
    """

    __slots__ = ("replays", "captures", "eager", "evictions",
                 "_graphs", "_seen", "_max_graphs",
                 "_use_cuda_graph")

    def __init__(self, max_graphs: int = 4, use_cuda_graph: bool = True):
        self.replays = 0
        self.captures = 0
        self.eager = 0
        self.evictions = 0
        self._graphs: OrderedDict = OrderedDict()  # key -> CUDAGraph (LRU)
        self._seen: dict = {}                      # key -> sighting count
        self._max_graphs = int(max_graphs)
        self._use_cuda_graph = use_cuda_graph

    def run(self, key, device, issue, stage=None):
        """Capture (on 2nd sighting) or replay (subsequent) the region.

        Parameters
        ----------
        key : tuple
            Pointer + scalar signature — usually ``(field_ptrs..., scalars...)``.
            Graph replayed when ``key`` matches a cached graph; a new key
            (reallocation / growth) recaptures, evicting the LRU graph when
            the cache is full.
        device : str
            CUDA device string, e.g. ``"cuda:0"``.  Ignored when
            ``_use_cuda_graph`` is False.
        issue : callable
            Zero-argument closure that performs the region's work on torch's
            current stream (native extension ops + torch ops; no raw
            ``wp.launch``).
        stage : callable or None
            Zero-argument closure that stages per-step data into the
            persistent buffers ``issue()`` reads — called OUTSIDE the capture,
            on EVERY path: eager, capture and replay.  ``issue()`` consumes
            those buffers regardless of graph mode (e.g. ``bdim_forcing`` reads
            the staged dirty rect), so skipping the staging on the eager path
            leaves it reading an uninitialised ``torch.empty`` buffer — silently
            wrong physics on CPU and under ``graph_capture_debug``.
        """
        # ---- stage per-step data OUTSIDE the graph (all paths) ----
        if stage is not None:
            stage()

        # ---- eager path (CPU, or CUDA graphs disabled) ----
        if not self._use_cuda_graph:
            self.eager += 1
            issue()
            return

        # ---- replay ----
        graph = self._graphs.get(key)
        if graph is not None:
            self._graphs.move_to_end(key)
            self.replays += 1
            graph.replay()
            return

        # ---- eager (first sighting = warm-up) ----
        n = self._seen.get(key, 0) + 1
        self._seen[key] = n
        if n < 2:
            self.eager += 1
            issue()
            return

        # ---- capture (second sighting) ----
        dev = torch.device(device)
        with torch.cuda.device(dev):
            # This step's REAL work first — torch requires pre-capture
            # warm-up on a side stream, and stream capture RECORDS without
            # executing, so this eager issue() is this step's sole correct
            # output.  Future steps replay the captured graph.
            side = torch.cuda.Stream(dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                issue()
            torch.cuda.current_stream(dev).wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                issue()

        # ---- LRU eviction: never pin (salamander OOM lesson) ----
        while len(self._graphs) >= self._max_graphs:
            _, old = self._graphs.popitem(last=False)
            old.reset()          # frees the graph's private memory pool
            self.evictions += 1
        self._graphs[key] = graph
        self.captures += 1
        # Bound the sighting book-keeping under pointer churn; a cleared
        # entry only costs one extra eager warm-up sighting.
        if len(self._seen) > 64 * max(self._max_graphs, 1):
            self._seen.clear()
