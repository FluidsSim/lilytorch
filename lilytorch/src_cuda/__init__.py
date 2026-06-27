"""``src_cuda`` — the hand-written CUDA/C++ kernel backend tree.

This package is the *analogue* of :mod:`lilytorch.src` whose kernel-dispatching
modules call the hand-written ``.cu`` / ``.cpp`` kernels (registered as
``torch.ops.lilytorch_kernels.*`` and surfaced through
:mod:`lilytorch.src.kernels.ops`) via the unified backend API in
:mod:`lilytorch.src_cuda.kernel`.

Because :mod:`lilytorch.src` already *is* the native/CUDA path, the modules
here are thin re-exports of their ``lilytorch.src`` counterparts — the value of
this tree is the explicit ``kernel/`` backend boundary, mirrored one-for-one by
:mod:`lilytorch.src_warp` (the Warp single-source backend).  Non-kernel modules
(``body``, ``plotting``, ``poisson_fft``, ``operations``, ``diagnostics``,
``video_postprocess`` …) are kernel-agnostic and stay shared in
:mod:`lilytorch.src`; both backends import them from there.

See ``src_cuda/README.md`` and ``warp_poc/VALIDATION_STATUS.md`` (§F) for the
structure rationale.
"""

BACKEND = "cuda"
