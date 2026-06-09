"""Two-phase (water + real air) fluid solver.

:class:`TwoPhaseSolver` is a thin **subclass** of
:class:`~lilytorch.src.solver.FluidSolver` that turns the single-fluid solver
into a two-phase variable-density solver for **partially-submerged / floating
bodies** (boats, buoyant swimming robots, paddling animals). The air is a real
(light) fluid, so a body sitting at the waterline gets a genuine pressure
reaction — none of the single-fluid ghost-fluid / wetted-body machinery is used.

**Decoupling.** Nothing in :class:`FluidSolver` is modified. Subclassing only
*overrides* methods on this class, so every existing example (single-phase,
FARMS coupling) runs through the untouched base solver byte-for-byte. The
overrides are:

* ``__init__`` — build the :class:`~lilytorch.src.two_phase.TwoPhase` field.
* ``_compute_bdim_coefficients`` — projection coefficients
  ``c = dt·μ0·(1/ρ)_fluid`` from the VOF water/air density (Weymouth & Yue 2011,
  ``(1−δ^B)/ρ``; the body enters via ``μ0`` only, never its density). Used by
  the **Python** solver path.
* ``project`` — the **kernel** path's fused CUDA Kernel B knows nothing of the
  VOF field and writes a single-density ``dt·μ0/ρ_water`` coefficient; this
  override rescales it by ``ρ_water·(1/ρ)_face`` to recover the identical
  variable-density coefficient (reusing the kernel's ``μ0``). No CUDA/solver
  edits; both paths solve the same Poisson.
* ``finalize_step`` — transport the VOF field once per step.
* ``forces_method2`` / ``forces_method2_3d`` — stack per-body SDFs so the
  emergent (real-pressure) eulerian force loop sees them.

Everything else is inherited and reused unchanged:

* **gravity** — the base ``_apply_gravity_body_force`` adds ``dt·g`` to *all*
  cells (real air has mass), which is exactly what two-phase needs;
* **projection core** — the base ``project`` runs the variable-coefficient MGCG
  path ``∇·(c∇p)=div`` → ``u -= c∇p`` (closed-box all-Neumann, ``dirichlet_mask
  = None``); we just feed it the density-based ``c`` (directly on the Python
  path, via the rescale on the kernel path);
* advection–diffusion, BDIM, the Poisson solver, force integration.

The hydrostatic interface jump and the buoyancy on the body therefore emerge
automatically from the density-weighted pressure (``∇p = ρ_fluid g`` at rest).
"""

import torch

from lilytorch.src.solver import FluidSolver
from lilytorch.src import forces as _forces
from lilytorch.src import diffusion as _diffusion
from lilytorch.src.advection import _sl
from lilytorch.src.two_phase import TwoPhase


class TwoPhaseSolver(FluidSolver):
    """Variable-density two-phase solver (water + real air) via VOF."""

    def __init__(self, pars, dtype=None, custom_update=None, compute_forces=True):
        super().__init__(pars, dtype=dtype, custom_update=custom_update,
                         compute_forces=compute_forces)
        solver = pars["solver"]
        tp_cfg = solver.get("two_phase")
        if tp_cfg is None:
            raise ValueError(
                "TwoPhaseSolver requires a 'solver.two_phase' config block."
            )
        if self.poisson_method not in ("multigrid", "mgcg"):
            raise ValueError(
                "two_phase requires poisson_method 'multigrid' or 'mgcg' "
                "(the FFT solver cannot do a variable-density Poisson)."
            )
        # Kernel-mode two-phase reuses the kernel's per-face coefficient
        # ``dt*mu0/rho_water`` and rescales it by the VOF reciprocal-density
        # blend (see :meth:`project`). That identity needs the kernel to keep
        # ``mu0`` in the coefficient numerator, i.e. ``bdim_mu0_projection``
        # must be True (the default). With it off the kernel writes a bare
        # ``dt/rho_water`` and the rescale would silently drop the body's
        # mu0 masking from the variable-density Poisson.
        if self._use_kernels and not self.bdim_mu0_projection:
            raise ValueError(
                "two_phase kernel mode requires bdim_mu0_projection=True so the "
                "kernel coefficient retains mu0 for the reciprocal-density "
                "rescale; got bdim_mu0_projection=False."
            )
        self._init_two_phase(tp_cfg)
        # Saved 3-D PyVista frames (plotting_and_saving -> plot_field_3d) render
        # the air/water INTERFACE: the VOF field alpha at iso-level 0.5, plus the
        # body SDF. This overrides the default vorticity iso-specs (the interface
        # is the point for two-phase). field_fn signature is (solver,u,v,p,w);
        # plot_field_3d draws a single isosurface at the fixed iso_value.
        if self.ndim == 3:
            # Each entry -> one saved PyVista frame series (plot_field_3d). The
            # interface is the air/water surface (alpha=0.5); omega_mag adds the
            # vorticity structures (auto-thresholded). Add/remove entries to
            # control which 3-D fields are rendered.
            self.iso_3d_specs = [
                ("interface", lambda s, u, v, p, w: s.two_phase.alpha, 0.5),
                ("omega_mag",
                 lambda s, u, v, p, w: s.vorticity_components(u, v, w)["omega_mag"],
                 None),
            ]

    # ------------------------------------------------------------------
    def _init_two_phase(self, cfg):
        alpha_src = cfg.get("alpha_init")
        if alpha_src is None:
            raise ValueError("solver.two_phase.alpha_init is required.")
        alpha_init = (eval(alpha_src, {"torch": torch})
                      if isinstance(alpha_src, str) else alpha_src)
        if not callable(alpha_init):
            raise ValueError(
                "solver.two_phase.alpha_init must be a callable or a "
                "lambda-string evaluating to one."
            )
        # The face material coefficient is the harmonic density mean only
        # (carried as the reciprocal density; see TwoPhase). The legacy
        # ``face_density`` key is accepted for backward-compat but only
        # "harmonic" is supported — reject "arithmetic" loudly rather than
        # silently ignoring it.
        _fd = cfg.get("face_density", "harmonic")
        if _fd != "harmonic":
            raise ValueError(
                f"solver.two_phase.face_density={_fd!r} is no longer supported; "
                "the variable-density projection uses the harmonic density mean "
                "(remove the key or set it to 'harmonic')."
            )
        kwargs = dict(
            rho_water = float(cfg.get("rho_water", 1000.0)),
            rho_air   = float(cfg.get("rho_air", 1.0)),
            nu_water  = float(cfg.get("nu_water", 1.0e-6)),
            nu_air    = float(cfg.get("nu_air", 1.5e-5)),
            device    = self.device,
            dtype     = self.dtype,
        )
        if self.ndim == 2:
            self.two_phase = TwoPhase(self.x, self.y, self.h, alpha_init, **kwargs)
        else:
            self.two_phase = TwoPhase(self.x, self.y, self.h, alpha_init,
                                      z=self.z, **kwargs)
        # Consistent conservative mass/momentum transport (Nangia et al. 2019,
        # Desjardins-Moureau): the cure for the high-density-ratio instability
        # (non-conservative is stable only to ~100:1; water/air is 833:1). When
        # on, ``fluid_step`` transports rho*u conservatively with the SAME mass
        # flux that evolves the density, recovers u = rho*u / rho consistently
        # (no 833x blow-up at the interface), and the interface is advected by
        # that same flux (the Weymouth-Yue VOF in finalize_step is skipped).
        # PYTHON path only (the fused kernel is untouched); default OFF so every
        # existing two-phase case is byte-for-byte unchanged.
        self._consistent_momentum = bool(cfg.get("consistent_momentum", False))
        if self._consistent_momentum and self._use_kernels:
            raise ValueError(
                "two_phase.consistent_momentum requires the python solver path "
                "(solver_method != 'kernel'); the fused kernel is not yet ported."
            )
        # Three-phase density (Nangia 2019 WSI, Eq. 23): treat the body as a
        # smoothed THIRD density phase rho_solid, blended by mu0 (the BDIM fluid
        # fraction IS the body Heaviside), INSIDE the variable-density projection
        # coefficient and the gravity body force -- instead of mu0-EXCLUDING the
        # body. Targets the body-interface band spurious current (the killer).
        # None = off (current mu0-exclusion behaviour). Python path only for now.
        self._rho_solid = cfg.get("rho_solid", None)
        if self._rho_solid is not None:
            self._rho_solid = float(self._rho_solid)
        # Fixed-point iteration cycles per step (Nangia Sec 4.2; n_cycles=2 gives
        # 2nd order + reconciles the body-inclusive projection with the rigid
        # constraint and the density). 1 = single forward-Euler pass (default).
        self._n_cycles = max(1, int(cfg.get("consistent_n_cycles", 1)))
        # Air-transparent body: mask the BDIM fluid fraction mu0 by the water
        # fraction alpha so the body effectively does NOT exist in the air phase.
        # Uses a SHARP threshold at alpha=0.5 (the VOF interface): the body is
        # fully present on the water side (alpha>=0.5 -> mu0_eff=mu0) and fully
        # transparent on the air side (alpha<0.5 -> mu0_eff=1).  This eliminates
        # the air-water-body triple-point singularity without creating a "half-
        # present" body zone at the waterline that can cause its own instability.
        # Physically motivated: air forces on the body are ~1000x smaller than
        # water forces, so ignoring the above-water body is an excellent
        # approximation.  Default ON; set to False to restore legacy behaviour.
        self._air_transparent_body = bool(
            cfg.get("air_transparent_body", True))
        flags = []
        if self._consistent_momentum:
            flags.append("CONSISTENT momentum")
        if self._rho_solid is not None:
            flags.append(f"three-phase rho_solid={self._rho_solid}")
        if self._air_transparent_body:
            flags.append("air-transparent-body")
        print(
            f"Two-phase enabled: rho {kwargs['rho_water']}/{kwargs['rho_air']} "
            f"(ratio {kwargs['rho_water']/kwargs['rho_air']:.0f}:1), "
            f"harmonic face density, "
            + (", ".join(flags) if flags else "standard (no mitigations).")
            + ("." if flags else "")
        )

    # NOTE: gravity uses the inherited uniform ``dt*g`` body force (the base
    # applies it to all cells — correct for two-phase, where the air has mass).
    # Body loads are EMERGENT: the inherited eulerian force routine integrates
    # the REAL variable-density pressure over the BDIM band, so buoyancy comes
    # out of the fluid forces, not an external displaced-volume term (see
    # ``_two_phase_forces``). For a fully-resolved 3-D rigid body the
    # ``force_method: "lagrangian"`` path (watertight surface, Σ(A·n)=0) is the
    # most accurate and gives Cz≈1 on the drop-sphere (Weymouth & Yue 2011). A
    # well-balanced p_rgh solve (cleaner dynamic pressure, fewer parasitic
    # interface currents) is a separate follow-up.

    # ------------------------------------------------------------------
    #  Override: density-based projection coefficients
    # ------------------------------------------------------------------
    def _compute_bdim_coefficients(self, timestep):
        """BDIM2 two-phase projection coefficients ``c = dt·μ0_eff / ρ_fluid`` on
        each staggered face (Python path).

        This is the Weymouth & Yue (2011) form (Eqs 24a/26a): the Poisson
        coefficient is ``(1−δ^B)/ρ`` with ``(1−δ^B) = μ0_eff`` the fluid fraction
        and ``ρ`` the **fluid** density — here the water/air VOF blend. We carry
        the reciprocal directly, ``c = dt·μ0_eff·(1/ρ)_face`` with ``(1/ρ)_face``
        the harmonic face density's reciprocal (``recip_density_face``, their Eq
        33), avoiding the dimensional density field and a separate harmonic blend.

        With ``air_transparent_body`` (default ON), μ0_eff = 1 − α·(1−μ0):
        the body is transparent in the air phase, eliminating the water-air-body
        triple-point singularity.  In water (α=1) μ0_eff = μ0 (normal BDIM); in
        air (α=0) μ0_eff = 1 (body invisible, air flows freely).

        The trailing ``ch_cc`` entry is the FFT RHS divisor, which two-phase
        never uses (the FFT path is forbidden; the MGCG path ignores it). It is
        returned as ``None`` to skip an unused full-grid allocation, keeping the
        ``(ch, cv[, cw], ch_cc)`` tuple shape ``fluid_step`` expects.
        """
        tp  = self.two_phase
        dt  = float(timestep)
        rs  = self._rho_solid
        atb = self._air_transparent_body
        out = []
        for d, ax in enumerate(self._bdim_axis_names):       # 'u','v'[,'w']
            recip_face = tp.recip_density_face(d)            # (1/ρ) water/air blend
            mu0 = getattr(self, f'mu0_all_{ax}')             # (1−δ^B), fluid fraction
            if atb:
                # Air-transparent body: mu0_eff = 1 - alpha_face*(1 - mu0)
                # = alpha_face*mu0 + (1-alpha_face). Body invisible in air.
                a_face = self._alpha_face(d)
                mu0 = a_face * mu0 + (1.0 - a_face)
            if rs is not None:
                # THREE-PHASE: body included as density rho_solid (Nangia Eq. 23),
                # c = dt / (mu0*rho_flow + (1-mu0)*rho_solid). No mu0 exclusion ->
                # the body carries a finite density at the waterline band.
                rho_flow_face = 1.0 / recip_face
                rho_3 = mu0 * rho_flow_face + (1.0 - mu0) * rs
                out.append(dt / rho_3)
            else:
                # current: body EXCLUDED (c -> 0 inside body, preserves BDIM vel)
                out.append(dt * mu0 * recip_face)
        out.append(None)                                     # ch_cc: unused (no FFT)
        return tuple(out)

    def _mu0_cc(self):
        """Cell-centred BDIM fluid fraction mu0 (1 in fluid, 0 in body, smoothed
        over the band) from the union body SDF -- the body Heaviside (= H_body)."""
        phi = self.composite_body.sdf_val
        eps = float(getattr(self, "eps", 2.0 * self.h))
        d = (phi / eps).clamp(-1.0, 1.0)
        return 0.5 * (1.0 + d + torch.sin(torch.pi * d) / torch.pi)

    def _alpha_face(self, d, full_grid=True):
        """Water fraction ``alpha`` interpolated to the staggered *d*-face grid.

        The face value is the arithmetic mean of the two adjacent cell-centred
        alpha values (MAC convention: face *i* sits between cells *i-1* and *i*).

        Parameters
        ----------
        d : int
            Stagger axis (0=x/u, 1=y/v, 2=z/w).
        full_grid : bool
            If True (default), returns the full-grid face values including ghost
            cells, matching ``recip_density_face(d)`` and the Python-path mu0
            tensors.  If False, returns cropped interior-only values matching
            the kernel persistent-buffer shapes (``_ch_persist`` etc.).
        """
        a = self.two_phase.alpha              # cell-centred, includes ghost cells
        nd = self.ndim
        if nd == 3:
            if full_grid:
                # Full-grid: clone and fill face column (like recip_density_face)
                out = a.clone()
                if d == 0:       # x-faces: average along x
                    out[1:, :, :] = 0.5 * (a[:-1, :, :] + a[1:, :, :])
                elif d == 1:     # y-faces
                    out[:, 1:, :] = 0.5 * (a[:, :-1, :] + a[:, 1:, :])
                else:            # z-faces
                    out[:, :, 1:] = 0.5 * (a[:, :, :-1] + a[:, :, 1:])
                return out
            else:
                # Cropped to kernel persistent-buffer shapes
                if d == 0:       # x-faces: (Nx+1, Ny, Nz)
                    return 0.5 * (a[1:,   1:-1, 1:-1] + a[:-1,  1:-1, 1:-1])
                elif d == 1:     # y-faces: (Nx, Ny+1, Nz)
                    return 0.5 * (a[1:-1, 1:,   1:-1] + a[1:-1, :-1,  1:-1])
                else:            # z-faces: (Nx, Ny, Nz+1)
                    return 0.5 * (a[1:-1, 1:-1, 1:  ] + a[1:-1, 1:-1, :-1 ])
        # 2-D: kernel persistent buffers have the same full-grid shape as the
        # cell-centred fields (unlike 3-D, where they are cropped).  So both
        # full_grid=True and full_grid=False return the same full-grid tensor.
        out = a.clone()
        if d == 0:           # x-faces
            out[1:, :] = 0.5 * (a[:-1, :] + a[1:, :])
        else:                # y-faces
            out[:, 1:] = 0.5 * (a[:, :-1] + a[:, 1:])
        return out

    # ------------------------------------------------------------------
    #  Override: BDIM velocity imposition — air-transparent body
    # ------------------------------------------------------------------
    def _apply_bdim_all_axes(self, vels):
        """Apply BDIM to each velocity component, with the body transparent
        in the air phase when ``air_transparent_body`` is on.

        In the air (α=0) the body is invisible: μ0_eff = 1, μ1_eff = 0 →
        the BDIM meta-equation reduces to the identity (no body forcing).
        In the water (α=1) the standard BDIM is applied unchanged.
        This is the Python-path companion to the kernel-path coefficient
        rescaling in :meth:`_rescale_kernel_coeffs_two_phase`.
        """
        if not self._air_transparent_body:
            return super()._apply_bdim_all_axes(vels)
        comp = self.composite_body
        out = []
        for i, ax in enumerate(self._bdim_axis_names):
            mu0 = getattr(self, f'mu0_all_{ax}')       # read-only; arithmetic
            mu1 = getattr(self, f'mu1_all_{ax}')       # creates new tensors
            body_vel = getattr(comp, f'body_{ax}')
            normals = tuple(
                getattr(self, f'normal_{n}_{ax}')
                for n in self._bdim_normal_names
            )
            # Mask mu0, mu1 by the water fraction on this face grid
            a_face = self._alpha_face(i, full_grid=True)
            # mu0_eff = alpha*mu0 + (1-alpha) → 1 in air, mu0 in water
            mu0_eff = a_face * mu0 + (1.0 - a_face)
            # mu1_eff = alpha*mu1 → 0 in air (no normal-derivative correction)
            mu1_eff = a_face * mu1
            out.append(self._bdim_apply(
                vels[i], mu0_eff, body_vel, mu1_eff, *normals))
        return tuple(out)

    # ------------------------------------------------------------------
    #  Override: variable density in the KERNEL-mode projection
    # ------------------------------------------------------------------
    def project(self, *args, ch=None, cv=None, cw=None, ch_cc=None, **kwargs):
        """Inject the VOF variable density into the kernel-mode projection.

        The fused CUDA Kernel B has no notion of the two-phase field: it writes
        a single-density coefficient ``c_kernel = dt·μ0/ρ_water`` per face (and
        the outside-AABB prefill is the same constant). The two-phase target is
        ``c = dt·μ0/ρ_face``, which differs by the pure multiplicative field
        ``ρ_water·(1/ρ)_face``. Rescaling the kernel coefficient by that factor
        reproduces ``_compute_bdim_coefficients`` exactly (μ0 is
        reused straight from ``c_kernel`` — no μ0/SDF/normal recompute), so both
        paths solve the identical variable-density Poisson.

        The Python path passes coefficients that are already two-phase-correct,
        so the rescale is gated on ``_use_kernels``. FFT is forbidden for
        two-phase, so the ``poisson_method != "fft"`` guard is belt-and-braces.
        The rescale is out-of-place: the persistent ``_ch_persist`` buffers must
        stay in their water-normalised form for the next step's prefill /
        Kernel-B overwrite (an in-place rescale would compound on the static
        region every step).
        """
        if (self._use_kernels and isinstance(ch, torch.Tensor)
                and self.poisson_method != "fft"):
            ch, cv, cw = self._rescale_kernel_coeffs_two_phase(ch, cv, cw)
        return super().project(*args, ch=ch, cv=cv, cw=cw, ch_cc=ch_cc, **kwargs)

    def _rescale_kernel_coeffs_two_phase(self, ch, cv, cw):
        """Return fresh ``dt·μ0_eff/ρ_face`` coefficients from the kernel's
        water-normalised ``dt·μ0/ρ_water`` ones.

        The kernel writes ``c_kernel = dt·μ0/ρ_water`` (μ0 = body fluid fraction).
        The two-phase target is ``c = dt·μ0_eff/ρ_face`` where:

        * Without air-transparent body: μ0_eff = μ0 (body present in both phases).
          We multiply by ``ρ_water·(1/ρ)_face`` → ``dt·μ0/ρ_face``.

        * With air-transparent body (default): μ0_eff = 1 − α·(1−μ0) =
          α·μ0 + (1−α).  Reconstructing μ0 from c_kernel = dt·μ0/ρ_water:
            c_target = dt/ρ_face · [α·μ0 + (1−α)]
                     = α · (ρ_water/ρ_face) · c_kernel + (1−α) · dt/ρ_face
          The first term is α times the legacy rescaling; the second term is
          the "free air" contribution where the body is transparent.

        ``(1/ρ)_face`` is computed inline from the cell-centred reciprocal
        density field and cropped to each staggered face-grid shape (3-D) or
        used full-grid (2-D), avoiding the full-grid clone in
        ``recip_density_face``.
        """
        tp = self.two_phase
        rw = tp.rho_water
        dt = float(self.dt)
        atb = self._air_transparent_body
        if self.ndim == 3:
            q   = tp.recip_density_cc()                  # one full-grid 1/ρ field
            # Face reciprocal-density rescaling factors ρ_water/ρ_face
            ru  = rw * 0.5 * (q[1:,   1:-1, 1:-1] + q[:-1,  1:-1, 1:-1])
            rv  = rw * 0.5 * (q[1:-1, 1:,   1:-1] + q[1:-1, :-1,  1:-1])
            rw_ = rw * 0.5 * (q[1:-1, 1:-1, 1:  ] + q[1:-1, 1:-1, :-1 ])
            # Face reciprocal density 1/ρ_face (for the free-air term)
            inv_rho_u = ru / rw
            inv_rho_v = rv / rw
            inv_rho_w = rw_ / rw
            if atb:
                a_u = self._alpha_face(0, full_grid=False)
                a_v = self._alpha_face(1, full_grid=False)
                a_w = self._alpha_face(2, full_grid=False)
                # c_target = α·(ρ_w/ρ_face)·c_kernel + (1−α)·dt/ρ_face
                return (a_u * ch * ru + (1.0 - a_u) * dt * inv_rho_u,
                        a_v * cv * rv + (1.0 - a_v) * dt * inv_rho_v,
                        a_w * cw * rw_ + (1.0 - a_w) * dt * inv_rho_w)
            return ch * ru, cv * rv, cw * rw_
        # 2-D kernel coefficients are stored on the full (padded) grid.
        r0 = rw * tp.recip_density_face(0)
        r1 = rw * tp.recip_density_face(1)
        if atb:
            a0 = self._alpha_face(0, full_grid=False)
            a1 = self._alpha_face(1, full_grid=False)
            inv_rho0 = r0 / rw
            inv_rho1 = r1 / rw
            return (a0 * ch * r0 + (1.0 - a0) * dt * inv_rho0,
                    a1 * cv * r1 + (1.0 - a1) * dt * inv_rho1,
                    cw)
        return ch * r0, cv * r1, cw

    # ------------------------------------------------------------------
    #  Override: EMERGENT body force (viscous + real-pressure integral)
    # ------------------------------------------------------------------
    # Buoyancy is NOT injected as an external term. The variable-density
    # projection already produces the correct hydrostatic pressure (verified:
    # the pressure jump across a body matches ρ_w g·D to ~0.1%), and the
    # eulerian BDIM band quadrature ``Σ -p n δ_ε(φ) hᴰ`` of that REAL pressure
    # recovers the buoyancy emergently: gauge-invariant (a constant pressure
    # integrates to ~0 over the closed band — verified Δ~1e-11 under a +1e5
    # gauge shift) and accurate (floating cylinder 0.91× Archimedes; 3-D
    # submerged sphere +4%). It also carries the dynamic load (form drag /
    # added mass / impact). This supersedes the earlier displaced-volume
    # workaround (which added buoyancy analytically and dropped the dynamic
    # load); the ``force_method: "lagrangian"`` path is still available and is
    # the most accurate for a fully-resolved 3-D body (drop-sphere Cz≈1).

    def _two_phase_forces(self, fn3d, vels, p, iteration):
        """Two-phase body loads = viscous + **emergent** pressure force: the
        inherited eulerian routine integrates the REAL variable-density
        pressure over the BDIM band, so buoyancy (and the dynamic load) emerge
        from the fluid forces rather than an external term."""
        # The eulerian per-body force loop reads each body's own SDF from
        # ``comp.sdf_vals`` / ``comp._sdf_sparse``. A bare ``composite_analytical``
        # body populates neither (only the streaming union + per-body
        # ``body.sdf_val``), and the 3-D loop lacks the 2-D stack fallback, so
        # expose a fresh per-body STACK here (refreshed every call).  Bodies that
        # manage ``sdf_vals`` themselves and whose sub-bodies are lightweight
        # proxies WITHOUT ``sdf_val`` (e.g. JellyfishBody's SimpleNamespace) are
        # left untouched -- they already set ``comp.sdf_vals`` in ``update``.
        cb = self.composite_body
        sparse = (hasattr(cb, '_sdf_sparse') and cb._sdf_sparse
                  and cb._sdf_sparse[0] is not None)
        if not sparse and all(hasattr(b, 'sdf_val') for b in cb.bodies):
            cb.sdf_vals = torch.stack([b.sdf_val for b in cb.bodies])
        base = _forces.forces_method2_3d if fn3d else _forces.forces_method2
        base(self, *vels, p, iteration)                # REAL pressure → emergent

    def forces_method2(self, u, v, p, iteration):
        self._two_phase_forces(False, (u, v), p, iteration)

    def forces_method2_3d(self, u, v, w, p, iteration):
        self._two_phase_forces(True, (u, v, w), p, iteration)

    # ------------------------------------------------------------------
    #  Override: consistent conservative mass/momentum transport
    # ------------------------------------------------------------------
    def _apply_gravity_body_force(self, *vels):
        """In consistent mode gravity is applied as a rho*g body force INSIDE the
        conservative momentum advection (see :meth:`_consistent_advect_2d`), so
        the base velocity pre-kick must be suppressed to avoid double-counting
        (and to keep the density-transport flux on the un-kicked velocity)."""
        if self._consistent_momentum and not self._use_kernels:
            return vels
        return super()._apply_gravity_body_force(*vels)

    def fluid_step(self, *args):
        """As the base (advect-BDIM-project), but in consistent mode the velocity
        advection is replaced by consistent conservative momentum transport.
        BDIM, the variable-density coefficients, and the projection are reused
        unchanged. Dimension-agnostic; python path only."""
        if not self._consistent_momentum or self._use_kernels:
            return super().fluid_step(*args)
        D = self.ndim
        u_n = [a.clone() for a in args[:D]]                  # n-level velocity (start)
        p = args[D]; timestep = args[D + 1]
        nu_t = self._compute_nu_t(*u_n)
        alpha_n = self.two_phase.alpha.clone()               # n-level interface
        iterate = [a.clone() for a in u_n]                   # u^{n+1,k}, k=0 -> u^n
        p_cur = p
        # Fixed-point iteration (Nangia Sec 4.2): each cycle RESTARTS from the
        # n-level state; the advecting velocity is the midpoint of the iterate and
        # u^n, the advected/limited velocity is the iterate. n_cycles=1 reduces to
        # a single forward-Euler pass (all three velocities = u^n).
        for k in range(self._n_cycles):
            if k == 0:
                u_adv = u_n; u_lim = u_n
            else:
                u_adv = [0.5 * (it + un) for it, un in zip(iterate, u_n)]
                u_lim = iterate
            self.two_phase.alpha = alpha_n.clone()           # restore start density
            primes = [pp.clone() for pp in
                      self._consistent_advect(u_n, u_adv, u_lim, nu_t, timestep)]
            self._bdim_union_aabb = None
            primes = list(self._apply_bdim_all_axes(primes))
            self.adv_diff_solver.set_BCs(*primes)
            coeffs = self._compute_bdim_coefficients(timestep)
            ch_cc = coeffs[-1]; face_coeffs = coeffs[:-1]
            proj_kwargs = {'ch': face_coeffs[0], 'cv': face_coeffs[1], 'ch_cc': ch_cc}
            if D == 3:
                proj_kwargs['cw'] = face_coeffs[2]
                proj_kwargs['w_vel'] = primes[2]
            if self._bdim_body_div_correction:
                _cb = self.composite_body
                proj_kwargs['body_div_corr'] = self._mw_body_div_correction(
                    _cb.body_u, _cb.body_v,
                    getattr(_cb, 'body_w', None) if D == 3 else None)
            out = self.project(primes[0], primes[1], p_cur, **proj_kwargs)
            iterate = list(out[:-1]); p_cur = out[-1]
        # free BDIM intermediates once, after the iteration (they are recomputed
        # once per step in advance_and_compute_loads, reused across cycles).
        self.__dict__.update(self._FS_FREE_AFTER_BDIM)
        self.__dict__.update(self._FS_FREE_AFTER_BDIM_COEFF)
        vels_out = iterate; p_out = p_cur
        if self.use_sponge:
            vels_out = list(self.apply_sponge_damping(*vels_out))
        if self.use_yield_damping:
            vels_out = list(self.apply_yield_damping(*vels_out))
        self.adv_diff_solver.set_BCs(*vels_out)
        return (*vels_out, p_out)

    def _consistent_advect(self, u_start, u_adv, u_lim, nu_t, dt):
        """Consistent conservative momentum transport (Nangia 2019,
        Desjardins-Moureau), dimension-agnostic. The mass flux ``F = u_adv *
        rho_upwind`` (advecting velocity) drives BOTH the cell-density update AND
        the momentum convection of ``u_lim`` (the advected/limited velocity); the
        momentum starts from ``u_start`` (the n-level ``rho^n u^n``) and is
        recovered as ``rho*u / rho`` with the SAME flux-evolved density (so the
        833:1 jump cannot blow up the velocity). For a single pass (n_cycles=1)
        all three are ``u^n``; the fixed-point iteration feeds the midpoint
        ``u_adv`` and the iterate ``u_lim``. Gravity is a ``rho*g`` BODY FORCE
        (NOT a pre-kick) -> the density flux is on the ~div-free velocity (mass
        conserving). ``two_phase.alpha`` is synced from the evolved density.
        MAC convention: ``q[k]`` is the face LEFT of cell ``k`` on its axis."""
        tp = self.two_phase
        rw, ra = tp.rho_water, tp.rho_air
        # Momentum/transport/recovery use the AIR/WATER density (NOT three-phase):
        # advecting the solid density with the flow corrupts both the body density
        # (it is rigid, not flow-advected) and the interface. rho_solid enters
        # only the projection coefficient + the gravity (empirically it 61 vs 47).
        r = tp.alpha * rw + (1.0 - tp.alpha) * ra
        nd = self.ndim
        dtdh = float(dt) / self.h
        I = tuple(slice(1, -1) for _ in range(nd))
        def A(d, s):  return _sl(nd, d, s)                               # slice s on axis d, full else
        def J(spec):  return tuple(spec.get(a, slice(1, -1)) for a in range(nd))  # interior else

        # upwind cell-face mass fluxes (advecting velocity): F[d][k] thru face left of cell k
        F = []
        for d in range(nd):
            Fd = torch.zeros_like(u_adv[d])
            vf = u_adv[d][A(d, slice(1, None))]
            Fd[A(d, slice(1, None))] = vf * torch.where(
                vf >= 0, r[A(d, slice(0, -1))], r[A(d, slice(1, None))])
            F.append(Fd)

        # evolve cell density by the cell fluxes (conservative -> mass preserving)
        r_new = r.clone()
        dvr = torch.zeros_like(r[I])
        for d in range(nd):
            dvr = dvr + (F[d][J({d: slice(2, None)})] - F[d][J({})])
        r_new[I] = (r[I] - dtdh * dvr).clamp_min(ra)

        # three-phase body Heaviside for the rho_solid gravity blend (band only)
        mu0cc = self._mu0_cc() if self._rho_solid is not None else None

        out = list(u_start)
        for i in range(nd):
            rfi = r.clone()
            rfi[A(i, slice(1, None))] = 0.5 * (r[A(i, slice(0, -1))] + r[A(i, slice(1, None))])
            mu = rfi * u_start[i]                              # n-level momentum rho^n u^n
            dmu = torch.zeros_like(mu[I])
            for d in range(nd):
                if d == i:                                   # self-advection (cell centres)
                    Mc = 0.5 * (F[i][A(i, slice(0, -1))] + F[i][A(i, slice(1, None))])
                    ui = torch.where(Mc >= 0, u_lim[i][A(i, slice(0, -1))],
                                     u_lim[i][A(i, slice(1, None))])
                    phi = Mc * ui
                    dmu = dmu + (phi[J({i: slice(1, None)})] - phi[J({i: slice(0, -1)})])
                else:                                        # cross-advection (i-edge / d-face)
                    Me = torch.zeros_like(F[d])
                    Me[A(i, slice(1, None))] = 0.5 * (F[d][A(i, slice(0, -1))]
                                                      + F[d][A(i, slice(1, None))])
                    Med = Me[A(d, slice(1, None))]
                    vid = torch.where(Med >= 0, u_lim[i][A(d, slice(0, -1))],
                                      u_lim[i][A(d, slice(1, None))])
                    psi = Med * vid
                    dmu = dmu + (psi[J({d: slice(1, None)})] - psi[J({d: slice(0, -1)})])
            mu[I] = mu[I] - dtdh * dmu
            if self.use_gravity and float(self._gravity[i]) != 0.0:
                rfg = rfi
                if mu0cc is not None:                          # three-phase rho*g at the band
                    m0 = rfi.clone()
                    m0[A(i, slice(1, None))] = 0.5 * (mu0cc[A(i, slice(0, -1))]
                                                      + mu0cc[A(i, slice(1, None))])
                    rfg = m0 * rfi + (1.0 - m0) * self._rho_solid
                mu[I] = mu[I] + float(dt) * float(self._gravity[i]) * rfg[I]
            # recover velocity with the evolved face density (consistent -> bounded)
            rfi2 = r_new.clone()
            rfi2[A(i, slice(1, None))] = 0.5 * (r_new[A(i, slice(0, -1))]
                                                + r_new[A(i, slice(1, None))])
            vi = u_start[i].clone()
            vi[I] = mu[I] / rfi2[I]
            ads = self.adv_diff_solver
            vi[I] = vi[I] + _diffusion.diffuse(u_start[i], dt, nu=self.nu, nu_t=nu_t,
                                               inv_dh2=ads._inv_dh2, dh=ads.dh)
            out[i] = vi

        tp.alpha = ((r_new - ra) / (rw - ra)).clamp(0.0, 1.0)
        return tuple(out)

    # ------------------------------------------------------------------
    #  Override: VOF transport in the once-per-step tail
    # ------------------------------------------------------------------
    def finalize_step(self, u, v, p, iteration, w_vel=None):
        """Once-per-step tail: stability check, VOF transport, BDIM-field
        release, optional allocator flush, and plotting/saving."""
        self.check_explosion(iteration)
        # In consistent (evolve) mode the interface already rode the shared mass
        # flux inside fluid_step (alpha synced there); skip the standalone VOF.
        if not self._consistent_momentum:
            if self.ndim == 2:
                self.two_phase.advect(u, v, dt=self.dt)
            else:
                self.two_phase.advect(u, v, w_vel, dt=self.dt)
        self._release_bdim_fields()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.plotting_and_saving(u, v, p, iteration, w_vel=w_vel)


def build_two_phase_solver(config_path, dtype=torch.float32):
    """Build a :class:`TwoPhaseSolver` from a YAML config path (parallels the
    per-example ``build_solver`` factories, which keep returning a plain
    :class:`FluidSolver`)."""
    from lilytorch.util.yaml_operations import yaml2pyobject
    pars = yaml2pyobject(config_path)
    return TwoPhaseSolver(pars, dtype=dtype)
