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
  also runs on Warp (Item 5): the bridge emits the body-id ``key_*`` it reads.

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
import lilytorch.src.forces as _forces_mod
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

        # Opt-in CUDA-graph capture of the Warp Kernel-A streaming step (mirrors
        # the existing ``poisson_cuda_graph`` flag).  When True, the kernel-step
        # streaming temporaries are reused as PERSISTENT buffers (stable pointers)
        # so the bridge can capture + replay the fanned launches in one graph,
        # eliminating the eager per-launch host floor (Kernel A → native parity at
        # all grid sizes).  Costs the streaming buffers' steady-state residency
        # (they are no longer freed before the pressure projection), a few % of
        # peak memory at 128³ — hence opt-in.  Default False = current eager path.
        pars = args[0] if args else kwargs.get("pars", {})
        try:
            self._kernel_cuda_graph = bool(
                pars["solver"].get("kernel_cuda_graph", False))
        except Exception:
            self._kernel_cuda_graph = False
        # C1: periodic MGCG convergence check (1 = native every-iter behaviour).
        # The native solver constructor takes no such kwarg, so thread it onto the
        # (Warp) Poisson sub-solver here, after construction.
        try:
            cge = int(pars["solver"].get("poisson_cg_check_every", 1))
            if getattr(self, "poisson_solver", None) is not None:
                self.poisson_solver.cg_check_every = cge
        except Exception:
            pass
        self._kbuf2d = None
        self._kbuf3d = None

    # ── persistent kernel-step buffers (CUDA-graph fast path) ─────────────────
    def _kernel_bufs_2d(self, gs, Ngrid, blend_on):
        """Lazily-allocated persistent streaming temporaries for the 2-D graph
        path (sdf_u/v, body_u/v, key_*, num/den), reused every step."""
        c = self._kbuf2d
        if c is None or c["gs"] != gs or c["dtype"] != self.dtype \
                or c["blend"] != blend_on:
            o = dict(device=self.device, dtype=self.dtype)
            ki = dict(dtype=torch.int64, device=self.device)
            nb = Ngrid if blend_on else 1
            c = dict(gs=gs, dtype=self.dtype, blend=blend_on,
                     sdf_u=torch.empty(gs, **o), sdf_v=torch.empty(gs, **o),
                     bU=torch.empty(gs, **o), bV=torch.empty(gs, **o),
                     key_cc=torch.empty(Ngrid, **ki), key_u=torch.empty(Ngrid, **ki),
                     key_v=torch.empty(Ngrid, **ki),
                     num_u=torch.empty(nb, **o), num_v=torch.empty(nb, **o),
                     den_u=torch.empty(nb, **o), den_v=torch.empty(nb, **o))
            self._kbuf2d = c
        return c

    def _kernel_bufs_3d(self, gs, blend_on):
        """3-D analogue of :meth:`_kernel_bufs_2d` (adds w-axis buffers).  The
        graph path is non-σ, so the key buffers are unused — they are size-1
        dummies, keeping the cache independent of the per-step dirty_vol (which
        changes as the body moves, and would otherwise thrash the graph)."""
        c = self._kbuf3d
        if c is None or c["gs"] != gs or c["dtype"] != self.dtype \
                or c["blend"] != blend_on:
            o = dict(device=self.device, dtype=self.dtype)
            ki = dict(dtype=torch.int64, device=self.device)
            nb = 1  # blend stays on the eager path → no full-grid num/den here
            c = dict(gs=gs, dtype=self.dtype, blend=blend_on,
                     sdf_u=torch.empty(gs, **o), sdf_v=torch.empty(gs, **o),
                     sdf_w=torch.empty(gs, **o),
                     bU=torch.empty(gs, **o), bV=torch.empty(gs, **o),
                     bW=torch.empty(gs, **o),
                     key_cc=torch.empty(1, **ki), key_u=torch.empty(1, **ki),
                     key_v=torch.empty(1, **ki), key_w=torch.empty(1, **ki),
                     num_u=torch.empty(nb, **o), num_v=torch.empty(nb, **o),
                     num_w=torch.empty(nb, **o), den_u=torch.empty(nb, **o),
                     den_v=torch.empty(nb, **o), den_w=torch.empty(nb, **o))
            self._kbuf3d = c
        return c

    # ── Lagrangian forces on Warp ────────────────────────────────────────────
    # The native ``forces_lagrangian_{2,3}d`` (inherited unchanged) calls the
    # module-global ``_lagrangian_forces_{2,3}d_kernel`` in ``lilytorch.src.forces``.
    # Route that single call to the Warp port by temporarily swapping the module
    # global for the duration of the call — the same localized dependency
    # injection used for the sub-solvers in ``__init__`` (no ``lilytorch.src``
    # edits, no global state left behind).  The Warp shim has the identical
    # ``(... , method=, sample_offset=, out=)`` signature, so the inherited body
    # (AABB crop, nu_rho buffers, persistent out buffer, scatter) is reused as-is.
    def forces_lagrangian_2d(self, u, v, p, iteration):
        _save = _forces_mod._lagrangian_forces_2d_kernel
        _forces_mod._lagrangian_forces_2d_kernel = kernel.lagrangian_forces_2d
        try:
            return super().forces_lagrangian_2d(u, v, p, iteration)
        finally:
            _forces_mod._lagrangian_forces_2d_kernel = _save

    def forces_lagrangian_3d(self, u, v, w, p, iteration):
        _save = _forces_mod._lagrangian_forces_3d_kernel
        _forces_mod._lagrangian_forces_3d_kernel = kernel.lagrangian_forces_3d
        try:
            return super().forces_lagrangian_3d(u, v, w, p, iteration)
        finally:
            _forces_mod._lagrangian_forces_3d_kernel = _save

    # ── Eulerian forces (n·δ band integral) on Warp ──────────────────────────
    # Same localized injection: ``forces_method2{,_3d}`` (inherited) call the
    # module-global ``streaming_sdf_forces_post_{2,3}d``; swap it for the Warp
    # port (identical signature incl. ``force_submethod``/``ph_tau``) for the
    # duration of the call.
    def forces_method2(self, u, v, p, iteration):
        _save = _forces_mod.streaming_sdf_forces_post_2d
        _forces_mod.streaming_sdf_forces_post_2d = kernel.streaming_sdf_forces_post_2d
        try:
            return super().forces_method2(u, v, p, iteration)
        finally:
            _forces_mod.streaming_sdf_forces_post_2d = _save

    def forces_method2_3d(self, u, v, w, p, iteration):
        _save = _forces_mod.streaming_sdf_forces_post_3d
        _forces_mod.streaming_sdf_forces_post_3d = kernel.streaming_sdf_forces_post_3d
        try:
            return super().forces_method2_3d(u, v, w, p, iteration)
        finally:
            _forces_mod.streaming_sdf_forces_post_3d = _save

    # ── Kernel A/B (2-D) on Warp ─────────────────────────────────────────────
    def _fluid_step_kernel_2d(self, u, v, p, timestep):
        """2-D kernel fluid step with Kernel A (streaming SDF) and Kernel B
        (fused BDIM2 + variable-density Poisson coefficients) routed to the Warp
        single-source ports via :mod:`lilytorch.src_warp.kernel`.

        Body copied verbatim from ``lilytorch.src.solver.FluidSolver``; the only
        changes are: (1) the two ``streaming_sdf_stag_2d_multi`` /
        ``bdim_coeff_2d`` calls dispatch through ``kernel.*``; (2) the σ path
        runs on Warp too — the streaming bridge emits the winning body-id into
        ``key_u/key_v`` (``emit_keys``) and the Warp σ Kernel B reads it (Item
        5); (3) ``comp.sdf_val`` is pre-filled to ``+FAR`` (the Warp
        ``atomic_min`` needs it; the native op initialised it internally)."""
        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_2d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_2d

        # BDIM-σ: lazily compute the per-body thin-body shifts on first use,
        # then decide whether the σ Kernel B path is active this step.
        if self.apply_bdim_sigma and self._sigma_shifts is None:
            self._compute_sigma_shifts()
        sigma_active = (self.apply_bdim_sigma
                        and self._sigma_shifts is not None
                        and bool(self._sigma_shifts.any()))

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
        Ngrid = int(gs[0]) * int(gs[1])
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        blend_on = blend_eps > 0.0
        # Opt-in CUDA-graph fast path needs PERSISTENT buffers (stable pointers).
        # σ / blend stay on the eager path (graph capture is non-σ, non-blend).
        graph_mode = (getattr(self, "_kernel_cuda_graph", False)
                      and not sigma_active and not blend_on)
        if graph_mode:
            # Persistent buffers; the SDF→FAR / body→0 resets are folded into the
            # bridge's captured graph (no per-step torch fills here).
            c = self._kernel_bufs_2d(gs, Ngrid, blend_on)
            sdf_u_tmp, sdf_v_tmp = c["sdf_u"], c["sdf_v"]
            bU_tmp, bV_tmp = c["bU"], c["bV"]
            key_cc_t, key_u_t, key_v_t = c["key_cc"], c["key_u"], c["key_v"]
            num_u_t, num_v_t = c["num_u"], c["num_v"]
            den_u_t, den_v_t = c["den_u"], c["den_v"]
        else:
            sdf_u_tmp = torch.full(gs, _FAR, **_opts)
            sdf_v_tmp = torch.full(gs, _FAR, **_opts)
            bU_tmp    = torch.zeros(gs, **_opts)
            bV_tmp    = torch.zeros(gs, **_opts)
            _key_opts = dict(dtype=torch.int64, device=self.device)
            key_cc_t  = torch.empty(Ngrid, **_key_opts)
            key_u_t   = torch.empty(Ngrid, **_key_opts)
            key_v_t   = torch.empty(Ngrid, **_key_opts)
            if blend_on:
                num_u_t = torch.zeros(Ngrid, **_opts); num_v_t = torch.zeros(Ngrid, **_opts)
                den_u_t = torch.zeros(Ngrid, **_opts); den_v_t = torch.zeros(Ngrid, **_opts)
            else:
                num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
                den_u_t = torch.empty(1, **_opts); den_v_t = torch.empty(1, **_opts)

        # Warp Kernel A's atomic-min needs the CC SDF pre-filled to +FAR.  In
        # graph mode the bridge folds this reset into the captured graph.
        if not graph_mode:
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
            emit_keys=sigma_active, use_graph=graph_mode,
        )

        # 6. Kernel B (Warp): fused BDIM2 + variable-density coefficients.
        #    σ variant (thin bodies) reads the body-id keys emitted by Kernel A.
        if sigma_active:
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
                key_u=key_u_t, key_v=key_v_t,
                sigma_shifts=self._sigma_shifts,
            )
        else:
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

    # ── Kernel A/B (3-D) on Warp ─────────────────────────────────────────────
    def _fluid_step_kernel_3d(self, u, v, w_vel, p, timestep):
        """3-D kernel fluid step with Kernel A (streaming SDF) and Kernel B
        (fused BDIM2 + variable-density Poisson coefficients) routed to the Warp
        single-source ports via :mod:`lilytorch.src_warp.kernel`.

        Body copied from ``lilytorch.src.solver.FluidSolver._fluid_step_kernel_3d``;
        the only changes are: (1) the two ``streaming_sdf_stag_3d_multi`` /
        ``bdim_coeff_3d`` calls dispatch through ``kernel.*``; (2) the σ path
        runs on Warp too — the streaming bridge emits the winning body-id into
        the dirty-local ``key_{u,v,w}`` (``emit_keys``) and the Warp σ Kernel B
        reads it (Item 5); (3) ``comp.sdf_val`` is pre-filled to ``+FAR`` (the
        Warp ``atomic_min`` needs it; the native op initialised it internally);
        (4) the verbose ``_chk`` memory-debug instrumentation is dropped (it is
        not part of the kernel contract)."""
        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_3d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_3d

        # BDIM-σ: lazily compute per-body thin-body shifts, then decide whether
        # the σ Kernel B path is active this step.
        if self.apply_bdim_sigma and self._sigma_shifts is None:
            self._compute_sigma_shifts()
        sigma_active = (self.apply_bdim_sigma
                        and self._sigma_shifts is not None
                        and bool(self._sigma_shifts.any()))

        # 1-2. eddy viscosity + advection-diffusion.
        nu_t   = self._compute_nu_t(u, v, w_vel)
        primes = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
        del nu_t
        self.u0.copy_(primes[0])
        self.v0.copy_(primes[1])
        self.w0.copy_(primes[2])

        # 3. Init persistent var-dens coefficients (once / on resize).
        self._init_bdim_coeff_persist_3d(timestep)

        # 4. Per-step temporaries for Kernel A -> Kernel B.
        _opts = dict(device=self.device, dtype=self.dtype)
        _FAR  = 1e4
        gs    = self.grid_shape
        dirty_vol = int(ks['dirty_Ai']) * int(ks['dirty_Aj']) * int(ks['dirty_Ak'])
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        blend_on = blend_eps > 0.0
        graph_mode = (getattr(self, "_kernel_cuda_graph", False)
                      and not sigma_active and not blend_on)
        if graph_mode:
            c = self._kernel_bufs_3d(gs, blend_on)
            sdf_u_tmp, sdf_v_tmp, sdf_w_tmp = c["sdf_u"], c["sdf_v"], c["sdf_w"]
            bU_tmp, bV_tmp, bW_tmp = c["bU"], c["bV"], c["bW"]
            key_cc_t, key_u_t = c["key_cc"], c["key_u"]
            key_v_t, key_w_t = c["key_v"], c["key_w"]
            num_u_t, num_v_t, num_w_t = c["num_u"], c["num_v"], c["num_w"]
            den_u_t, den_v_t, den_w_t = c["den_u"], c["den_v"], c["den_w"]
            # SDF→FAR / body→0 resets are folded into the bridge's captured graph.
        else:
            sdf_u_tmp = torch.full(gs, _FAR, **_opts)
            sdf_v_tmp = torch.full(gs, _FAR, **_opts)
            sdf_w_tmp = torch.full(gs, _FAR, **_opts)
            bU_tmp    = torch.zeros(gs, **_opts)
            bV_tmp    = torch.zeros(gs, **_opts)
            bW_tmp    = torch.zeros(gs, **_opts)
            _key_opts = dict(dtype=torch.int64, device=self.device)
            key_cc_t = torch.empty(dirty_vol, **_key_opts)
            key_u_t  = torch.empty(dirty_vol, **_key_opts)
            key_v_t  = torch.empty(dirty_vol, **_key_opts)
            key_w_t  = torch.empty(dirty_vol, **_key_opts)
            if blend_on:
                num_u_t = torch.zeros(dirty_vol, **_opts)
                num_v_t = torch.zeros(dirty_vol, **_opts)
                num_w_t = torch.zeros(dirty_vol, **_opts)
                den_u_t = torch.zeros(dirty_vol, **_opts)
                den_v_t = torch.zeros(dirty_vol, **_opts)
                den_w_t = torch.zeros(dirty_vol, **_opts)
            else:
                num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
                num_w_t = torch.empty(1, **_opts); den_u_t = torch.empty(1, **_opts)
                den_v_t = torch.empty(1, **_opts); den_w_t = torch.empty(1, **_opts)

        # Warp Kernel A's atomic-min needs the CC SDF pre-filled to +FAR.  In
        # graph mode the bridge folds this reset into the captured graph.
        if not graph_mode:
            comp.sdf_val.fill_(_FAR)

        # 5. Kernel A (Warp).
        kernel.streaming_sdf_stag_3d_multi(
            sm['F_flat'], sm['F_offsets'],
            sm['body_shapes'], sm['body_meta'], ks['kin'],
            ks['aabb_lo'], ks['aabb_dim'],
            ks['gx'], ks['gy'], ks['gz'],
            float(comp.h), int(ks['max_vol']),
            comp.sdf_val, sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
            bU_tmp, bV_tmp, bW_tmp,
            key_cc_t, key_u_t, key_v_t, key_w_t,
            int(getattr(self, '_sdf_interp_method', 0)),
            int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
            int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
            num_u_t, num_v_t, num_w_t, den_u_t, den_v_t, den_w_t,
            float(blend_eps),
            emit_keys=sigma_active, use_graph=graph_mode,
        )

        # 6. Kernel B (Warp): fused BDIM2 + variable-density coefficients.
        #    σ variant (thin bodies) reads the body-id keys emitted by Kernel A.
        if sigma_active:
            kernel.bdim_coeff_3d(
                primes[0], primes[1], primes[2],
                sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
                bU_tmp, bV_tmp, bW_tmp,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                float(comp.eps), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
                int(self.bdim_mu0_projection),
                key_u=key_u_t, key_v=key_v_t, key_w=key_w_t,
                sigma_shifts=self._sigma_shifts,
            )
        else:
            kernel.bdim_coeff_3d(
                primes[0], primes[1], primes[2],
                sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
                bU_tmp, bV_tmp, bW_tmp,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                float(comp.eps), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
                int(self.bdim_mu0_projection),
            )

        # 6b. Maertens–Weymouth body-divergence RHS correction (before free).
        _body_div_corr = (
            self._mw_body_div_correction(bU_tmp, bV_tmp, bW_tmp)
            if self._bdim_body_div_correction else None)

        # 7. Free per-step temporaries before the pressure projection.
        del sdf_u_tmp, sdf_v_tmp, sdf_w_tmp, bU_tmp, bV_tmp, bW_tmp, primes
        del key_cc_t, key_u_t, key_v_t, key_w_t

        # 8. Boundary conditions on the BDIM-corrected velocity.
        self.adv_diff_solver.set_BCs(self.u0, self.v0, self.w0)

        # 9. Pressure projection.
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist, cw=self._cw_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
            w_vel=self.w0,
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
