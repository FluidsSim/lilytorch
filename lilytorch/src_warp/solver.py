"""``src_warp.solver`` — fluid solver wired to the Warp backend.

Subclasses :class:`lilytorch.src.solver.FluidSolver` and injects the Warp
sub-solvers (advection with the Warp flux, Poisson) by **temporarily** swapping
the ``AdvDiffSolver`` / ``PoissonSolver`` names in the base solver module for the
duration of ``__init__`` only — a localized dependency injection that needs no
edits to ``lilytorch.src`` and leaves no global state behind (so importing the
native ``FluidSolver`` elsewhere is unaffected).

What runs on Warp through this path:

* **advection flux** — via :class:`lilytorch.src_warp.advection.AdvDiffSolver`
  (single-source ``advect_flux_add``).
* **two-phase VOF** — when the two-phase field is built from
  :class:`lilytorch.src_warp.two_phase.TwoPhase` (single-source ``cvof_sweep``).
* **Kernel A/B (2-D)** — ``_fluid_step_kernel_2d`` (overridden below) routes the
  ``streaming_sdf_stag_2d_multi`` + ``bdim_coeff_2d`` calls to the Warp ports via
  the :mod:`lilytorch.src_warp.kernel` marshalling bridge (dtype-generic Kernel
  A; f64 Kernel B — a float32 solver falls back to native Kernel B).  The σ path
  stays native (the Warp streaming bridge does not emit the packed ``key_*``).

What still falls back to native (inherited unchanged; see §F):

* **Kernel A/B (3-D)** ``_fluid_step_kernel_3d`` — needs the 3-D marshalling
  bridge + an f32 Kernel B Warp variant.
* **Poisson driver** — needs the mgcg/multigrid outer-driver assembly.
* **forces** — see :mod:`lilytorch.src_warp.forces`.

``WARP_BACKED`` (from :mod:`lilytorch.src_warp.kernel`) is the authoritative list
of which ops actually execute on Warp.
"""
import torch

import lilytorch.src.solver as _solver_mod
from lilytorch.src.solver import FluidSolver as _BaseFluidSolver  # noqa: F401

from lilytorch.src_warp.advection import AdvDiffSolver as _WarpAdvDiffSolver
from lilytorch.src_warp.poisson_mult import PoissonSolver as _WarpPoissonSolver
from lilytorch.src_warp import kernel

BACKEND = "warp"


class FluidSolver(_BaseFluidSolver):
    """``FluidSolver`` with Warp sub-solvers injected at construction."""

    def __init__(self, *args, **kwargs):
        _save_adv = _solver_mod.AdvDiffSolver
        _save_poi = _solver_mod.PoissonSolver
        _solver_mod.AdvDiffSolver = _WarpAdvDiffSolver
        _solver_mod.PoissonSolver = _WarpPoissonSolver
        try:
            super().__init__(*args, **kwargs)
        finally:
            _solver_mod.AdvDiffSolver = _save_adv
            _solver_mod.PoissonSolver = _save_poi

    # ── Kernel A/B (2-D) on Warp ─────────────────────────────────────────────
    def _fluid_step_kernel_2d(self, u, v, p, timestep):
        """2-D kernel fluid step with Kernel A (streaming SDF) and Kernel B
        (fused BDIM2 + variable-density Poisson coefficients) routed to the Warp
        single-source ports via :mod:`lilytorch.src_warp.kernel`.

        Body copied verbatim from ``lilytorch.src.solver.FluidSolver``; the only
        changes are: (1) the two ``streaming_sdf_stag_2d_multi`` /
        ``bdim_coeff_2d`` calls dispatch through ``kernel.*``; (2) the σ path
        falls back to the native step (the Warp streaming bridge does not emit
        the packed ``key_*`` arrays the σ Kernel B reads); (3) ``comp.sdf_val``
        is pre-filled to ``+FAR`` (the Warp ``atomic_min`` needs it; the native
        op initialised it internally)."""
        if self.apply_bdim_sigma:
            return super()._fluid_step_kernel_2d(u, v, p, timestep)

        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_2d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_2d

        # 1-2. eddy viscosity + advection-diffusion.
        nu_t   = self._compute_nu_t(u, v)
        primes = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
        self.u0.copy_(primes[0])
        self.v0.copy_(primes[1])

        # 3. Init persistent var-dens coefficients (once / on resize).
        self._init_bdim_coeff_persist_2d(timestep)

        # 4. Per-step temporaries for Kernel A -> Kernel B.
        _opts = dict(device=self.device, dtype=self.dtype)
        _FAR  = 1e4
        gs    = self.grid_shape
        sdf_u_tmp = torch.full(gs, _FAR, **_opts)
        sdf_v_tmp = torch.full(gs, _FAR, **_opts)
        bU_tmp    = torch.zeros(gs, **_opts)
        bV_tmp    = torch.zeros(gs, **_opts)
        Ngrid = int(gs[0]) * int(gs[1])
        _key_opts = dict(dtype=torch.int64, device=self.device)
        key_cc_t  = torch.empty(Ngrid, **_key_opts)
        key_u_t   = torch.empty(Ngrid, **_key_opts)
        key_v_t   = torch.empty(Ngrid, **_key_opts)
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        if blend_eps > 0.0:
            num_u_t = torch.zeros(Ngrid, **_opts); num_v_t = torch.zeros(Ngrid, **_opts)
            den_u_t = torch.zeros(Ngrid, **_opts); den_v_t = torch.zeros(Ngrid, **_opts)
        else:
            num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
            den_u_t = torch.empty(1, **_opts); den_v_t = torch.empty(1, **_opts)

        # Warp Kernel A's atomic-min needs the CC SDF pre-filled to +FAR.
        comp.sdf_val.fill_(_FAR)

        # 5. Kernel A (Warp).
        kernel.streaming_sdf_stag_2d_multi(
            sm['F_flat'], sm['F_offsets'],
            sm['body_shapes'], sm['body_meta'], ks['kin'],
            ks['aabb_lo'], ks['aabb_dim'],
            ks['gx'], ks['gy'],
            float(comp.h), int(ks['max_vol']),
            comp.sdf_val, sdf_u_tmp, sdf_v_tmp,
            bU_tmp, bV_tmp,
            key_cc_t, key_u_t, key_v_t,
            int(getattr(self, '_sdf_interp_method', 0)),
            int(ks['dirty_i0']), int(ks['dirty_j0']),
            int(ks['dirty_Ai']), int(ks['dirty_Aj']),
            num_u_t, num_v_t, den_u_t, den_v_t, float(blend_eps),
        )

        # 6. Kernel B (Warp): fused BDIM2 + variable-density coefficients.
        kernel.bdim_coeff_2d(
            primes[0], primes[1],
            sdf_u_tmp, sdf_v_tmp,
            bU_tmp, bV_tmp,
            self.u0, self.v0,
            self._ch_persist, self._cv_persist,
            float(comp.eps), float(self.rho),
            float(timestep), float(comp.h),
            int(ks['dirty_i0']), int(ks['dirty_j0']),
            int(ks['dirty_Ai']), int(ks['dirty_Aj']),
            int(self.bdim_mu0_projection),
        )

        # 6b. Maertens–Weymouth body-divergence RHS correction (before free).
        _body_div_corr = (
            self._mw_body_div_correction(bU_tmp, bV_tmp)
            if self._bdim_body_div_correction else None)

        # 7. Free per-step temporaries before the pressure projection.
        del sdf_u_tmp, sdf_v_tmp, bU_tmp, bV_tmp, primes
        del key_cc_t, key_u_t, key_v_t

        # 8. Boundary conditions on the BDIM-corrected velocity.
        self.adv_diff_solver.set_BCs(self.u0, self.v0)

        # 9. Pressure projection.
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
            body_div_corr=_body_div_corr,
        )
        vels_out = out[:-1]
        p_out    = out[-1]

        # 10. Optional sponge / yield damping + final BC pass.
        if self.use_sponge:
            vels_out = self.apply_sponge_damping(*vels_out)
        if self.use_yield_damping:
            vels_out = self.apply_yield_damping(*vels_out)
        self.adv_diff_solver.set_BCs(*vels_out)

        return (*vels_out, p_out)
