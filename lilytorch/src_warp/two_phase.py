"""``src_warp.two_phase`` — two-phase VOF field with the sweep on Warp.

Subclasses :class:`lilytorch.src.two_phase.TwoPhase` and overrides
``_cvof_sweep`` so the directional-split Weymouth–Yue VOF sweep runs through the
single-source Warp ``cvof_sweep`` kernel (:mod:`lilytorch.src_warp.kernel`),
which is a signature-identical drop-in, bit-exact vs native (2-D+3-D), and runs
on **CPU and GPU** from one source.
"""
from lilytorch.src.two_phase import TwoPhase as _BaseTwoPhase  # noqa: F401
from lilytorch.src_warp import kernel


class TwoPhase(_BaseTwoPhase):
    """``TwoPhase`` whose VOF sweep dispatches to the Warp ``cvof_sweep``."""

    def _cvof_sweep(self, a, u_d, d, dt):
        if "cvof_sweep" in kernel.WARP_BACKED:
            out = a.clone()
            # Warp single-source kernel: CPU and GPU from the same @wp.kernel.
            kernel.cvof_sweep(a, u_d, float(dt) / self.h, d, out)
            return out
        return super()._cvof_sweep(a, u_d, d, dt)
