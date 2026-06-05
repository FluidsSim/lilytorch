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
        print(
            f"Two-phase enabled: rho {kwargs['rho_water']}/{kwargs['rho_air']} "
            f"(ratio {kwargs['rho_water']/kwargs['rho_air']:.0f}:1), "
            f"harmonic face density, conservative Weymouth-Yue VOF."
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
        """BDIM2 two-phase projection coefficients ``c = dt·μ0 / ρ_fluid`` on
        each staggered face (Python path).

        This is the Weymouth & Yue (2011) form (Eqs 24a/26a): the Poisson
        coefficient is ``(1−δ^B)/ρ`` with ``(1−δ^B) = μ0`` the fluid fraction
        and ``ρ`` the **fluid** density — here the water/air VOF blend. We carry
        the reciprocal directly, ``c = dt·μ0·(1/ρ)_face`` with ``(1/ρ)_face`` the
        harmonic face density's reciprocal (``recip_density_face``, their Eq 33),
        avoiding the dimensional density field and a separate harmonic blend.
        The body enters ONLY through ``μ0`` (the geometry), NOT through its
        density: ``μ0`` makes the velocity correction vanish inside the body
        (``μ0=0`` ⇒ ``c=0``), preserving the BDIM-imposed body velocity. The
        body's density/inertia is the rigid-body coupling's concern (MuJoCo /
        external Archimedes), not a fluid property — so ``rho_body`` does not
        appear here.

        The trailing ``ch_cc`` entry is the FFT RHS divisor, which two-phase
        never uses (the FFT path is forbidden; the MGCG path ignores it). It is
        returned as ``None`` to skip an unused full-grid allocation, keeping the
        ``(ch, cv[, cw], ch_cc)`` tuple shape ``fluid_step`` expects.
        """
        tp  = self.two_phase
        dt  = float(timestep)
        out = []
        for d, ax in enumerate(self._bdim_axis_names):       # 'u','v'[,'w']
            recip_face = tp.recip_density_face(d)            # (1/ρ) water/air blend
            mu0 = getattr(self, f'mu0_all_{ax}')             # (1−δ^B), fluid fraction
            out.append(dt * mu0 * recip_face)
        out.append(None)                                     # ch_cc: unused (no FFT)
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
        """Return fresh ``dt·μ0/ρ_face`` coefficients from the kernel's
        water-normalised ``dt·μ0/ρ_water`` ones, multiplying by the face
        reciprocal-density factor ``ρ_water·(1/ρ)_face``.

        ``(1/ρ)_face`` is the arithmetic mean of the cell-centred reciprocal
        density (= reciprocal of the harmonic face density). It is computed
        inline from the single full-grid ``recip_density_cc`` field and cropped
        to each staggered face-grid shape (3-D) or used full-grid (2-D, where
        ``_ch_persist`` is padded), avoiding the full-grid clone in
        ``recip_density_face``.
        """
        tp = self.two_phase
        rw = tp.rho_water
        if self.ndim == 3:
            # Crop the cell-centred 1/ρ field to each face-grid shape and
            # average the two adjacent cells inline (avoids the full-grid clone
            # in recip_density_face; the 3-D grids are the memory-sensitive case).
            q   = tp.recip_density_cc()                  # one full-grid 1/ρ field
            ru  = rw * 0.5 * (q[1:,   1:-1, 1:-1] + q[:-1,  1:-1, 1:-1])
            rv  = rw * 0.5 * (q[1:-1, 1:,   1:-1] + q[1:-1, :-1,  1:-1])
            rw_ = rw * 0.5 * (q[1:-1, 1:-1, 1:  ] + q[1:-1, 1:-1, :-1 ])
            return ch * ru, cv * rv, cw * rw_
        # 2-D kernel coefficients are stored on the full (padded) grid, so the
        # full-grid recip_density_face (boundary face = adjacent cell) aligns
        # directly with the elementwise multiply.
        return (ch * (rw * tp.recip_density_face(0)),
                cv * (rw * tp.recip_density_face(1)),
                cw)

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
    #  Override: VOF transport in the once-per-step tail
    # ------------------------------------------------------------------
    def finalize_step(self, u, v, p, iteration, w_vel=None):
        """Once-per-step tail: stability check, VOF transport, BDIM-field
        release, optional allocator flush, and plotting/saving."""
        self.check_explosion(iteration)
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
