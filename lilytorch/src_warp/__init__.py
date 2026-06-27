"""``src_warp`` — the Warp single-source kernel backend tree.

Analogue of :mod:`lilytorch.src` whose kernel-dispatching modules call the
single-source ``@wp.kernel`` ports (in :mod:`lilytorch.warp_poc`) through the
unified backend API in :mod:`lilytorch.src_warp.kernel`.  One kernel source runs
on **CPU and GPU**, retiring the hand-written ``.cu``/``.cpp`` twins for the ops
that are wired in.

Status (see ``src_warp/README.md`` and ``warp_poc/VALIDATION_STATUS.md`` §F):
the Warp kernels are all individually ported + parity-clean (147/147 tests).
This tree wires the **signature-clean drop-in** ops into the live solver —
``advect_flux_add`` (advection flux) and ``cvof_sweep`` (two-phase VOF) — which
have identical call conventions to their native ops.  The remaining ops
(Kernel A/B streaming-SDF/BDIM, the Poisson driver, ``apply_bcs``, the Eulerian
force readout) fall back to native pending a marshalling/driver-assembly bridge;
:data:`lilytorch.src_warp.kernel.WARP_BACKED` lists exactly which ops run on Warp.

Non-kernel modules (``body``, ``plotting``, ``poisson_fft``, ``operations`` …)
are shared from :mod:`lilytorch.src`.
"""

BACKEND = "warp"
