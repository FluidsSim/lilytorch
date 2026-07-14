"""Whole-step CUDA-graph capture for the pre-Poisson fluid region.

:class:`NativeWholeStepGraphRunner` captures the entire pre-Poisson region
(semi-Lagrangian advection, diffusion accumulate, ``bdim_forcing``,
``apply_bcs``) as ONE ``torch.cuda.CUDAGraph`` and replays it with a single
host launch.

Stream capture records whatever ``issue()`` enqueues on torch's CURRENT CUDA
stream — native ``TORCH_LIBRARY`` extension ops (they launch via
``at::cuda::getCurrentCUDAStream()``) and plain torch ops alike.  A kernel
launched on any OTHER stream is silently dropped from every replay, which is
why the whole pre-Poisson region has to stay native end-to-end.
"""

from collections import OrderedDict

import torch


class NativeWholeStepGraphRunner:
    """Keyed ``torch.cuda.CUDAGraph`` runner for the pre-Poisson region.

    ``run(key, device, issue, stage)``: capture on the second sighting of a
    key, replay thereafter.

    * **What is recorded.**  Stream capture records whatever ``issue()``
      enqueues on torch's CURRENT CUDA stream — native ``TORCH_LIBRARY``
      extension ops (they launch via ``at::cuda::getCurrentCUDAStream()``)
      and plain torch ops.  A kernel launched on any other stream is NOT
      recorded and is silently dropped from every replay.
    * **Eviction, never pinning.**  The keyed cache is LRU: when full, the
      least-recently-replayed graph is ``reset()`` (its private memory pool
      is freed) to admit the new capture.  Keeping old graphs instead would
      pin one dead graph per stale pointer signature (the salamander OOM).
    * **Staging.**  ``stage()`` runs outside the graph, but replay is ordered
      on the SAME torch stream as ``stage()``'s ``copy_``, so callers need no
      hard ``torch.cuda.synchronize`` in ``stage()``.
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
            current stream (native extension ops + torch ops only).
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
