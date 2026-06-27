"""``src_warp.advection`` — advection-diffusion with the flux on Warp.

Subclasses :class:`lilytorch.src.advection.AdvDiffSolver` and overrides the
fused-flux fast path of ``_solve_convective`` so the per-(component, direction)
accumulation ``rhs += dt_dh·(F_L−F_R)`` runs through the single-source Warp
``advect_flux_add`` kernel (:mod:`lilytorch.src_warp.kernel`) instead of
``torch.ops.lilytorch_kernels.advect_flux_add``.  The Warp op is a
signature-identical drop-in and bit-exact vs native (all 5 schemes, 2-D+3-D),
so the only change is the dispatch target; all other code paths (multi-stream,
python fallback) delegate to the base class unchanged.
"""
from lilytorch.src.advection import (  # noqa: F401
    AdvDiffSolver as _BaseAdvDiffSolver,
    SCHEMES,
    _CUDA_SCHEME_IDS,
    _face_vel,
    _field_for_flux,
    _inner,
    diffusion,
)
from lilytorch.src_warp import kernel


class AdvDiffSolver(_BaseAdvDiffSolver):
    """``AdvDiffSolver`` whose CUDA fused-flux path dispatches to Warp."""

    def _solve_convective(self, *vel, nu_t=None, iteration=0):
        ndim = self.ndim
        use_cuda_kernel = (
            self._is_cuda
            and self._scheme_name in _CUDA_SCHEME_IDS
            and not (self._use_streams and ndim > 1)
        )
        if not use_cuda_kernel:
            # multi-stream / python fallback paths are unchanged.
            return super()._solve_convective(*vel, nu_t=nu_t, iteration=iteration)

        # Mirror the native fused-flux fast path, swapping the op for Warp.
        vel_new = list(vel)
        inner = _inner(ndim)
        scheme_id = _CUDA_SCHEME_IDS[self._scheme_name]
        if self._scheme_name == 'abdquickest':
            h_min = min(self.dh)
            umax = float(max(v.abs().amax() for v in vel))
            C_courant = float(min(max(umax * self.dt / h_min, 0.1), 0.99))
        else:
            C_courant = 0.0
        for i in range(ndim):
            rhs = diffusion.diffuse(
                vel[i], self.dt, nu=self.nu, nu_t=nu_t,
                inv_dh2=self._inv_dh2, dh=self.dh,
            )
            for d in range(ndim):
                fv = _face_vel(vel, i, d, ndim)
                p = _field_for_flux(vel[i], d, ndim)
                kernel.advect_flux_add(
                    fv, p, rhs,
                    float(self._dt_dh[d]), C_courant,
                    scheme_id, d,
                )
                del fv, p
            vel_new[i] = vel[i].clone()
            vel_new[i][inner] += rhs
            del rhs
        return tuple(vel_new)
