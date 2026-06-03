"""Two-phase (water + real air) fluid solver.

:class:`TwoPhaseSolver` is a thin **subclass** of
:class:`~lilytorch.src.solver.FluidSolver` that turns the single-fluid solver
into a two-phase variable-density solver for **partially-submerged / floating
bodies** (boats, buoyant swimming robots, paddling animals). The air is a real
(light) fluid, so a body sitting at the waterline gets a genuine pressure
reaction — none of the single-fluid ghost-fluid / wetted-body machinery is used.

**Decoupling.** Nothing in :class:`FluidSolver` is modified. Subclassing only
*overrides* methods on this class, so every existing example (single-phase,
FARMS coupling) runs through the untouched base solver byte-for-byte. Only three
methods are overridden:

* ``__init__`` — build the :class:`~lilytorch.src.two_phase.TwoPhase` field.
* ``_compute_variable_density_coefficients`` — projection coefficients
  ``c = dt/ρ_eff`` from the VOF density (× the body via ``μ0``).
* ``finalize_step`` — transport the VOF field once per step.

Everything else is inherited and reused unchanged:

* **gravity** — the base ``_apply_gravity_body_force`` adds ``dt·g`` to *all*
  cells (real air has mass), which is exactly what two-phase needs;
* **projection** — the base ``project`` runs the variable-coefficient MGCG path
  ``∇·(c∇p)=div`` → ``u -= c∇p`` (closed-box all-Neumann, ``dirichlet_mask =
  None``); we just feed it the density-based ``c``;
* advection–diffusion, BDIM, the Poisson solver, force integration.

The hydrostatic interface jump and the buoyancy on the body therefore emerge
automatically from the density-weighted pressure (``∇p = ρ_eff g`` at rest).
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
        self._init_two_phase(tp_cfg)

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
        kwargs = dict(
            rho_water    = float(cfg.get("rho_water", 1000.0)),
            rho_air      = float(cfg.get("rho_air", 1.0)),
            nu_water     = float(cfg.get("nu_water", 1.0e-6)),
            nu_air       = float(cfg.get("nu_air", 1.5e-5)),
            advection    = cfg.get("advection", "cubista"),
            compression  = float(cfg.get("compression", 1.0)),
            face_density = cfg.get("face_density", "harmonic"),
            device       = self.device,
            dtype        = self.dtype,
        )
        if self.ndim == 2:
            self.two_phase = TwoPhase(self.x, self.y, self.h, alpha_init, **kwargs)
        else:
            self.two_phase = TwoPhase(self.x, self.y, self.h, alpha_init,
                                      z=self.z, **kwargs)
        print(
            f"Two-phase enabled: rho {kwargs['rho_water']}/{kwargs['rho_air']} "
            f"(ratio {kwargs['rho_water']/kwargs['rho_air']:.0f}:1), "
            f"advection={kwargs['advection']}, compression={kwargs['compression']}, "
            f"face_density={kwargs['face_density']}."
        )

    # NOTE: gravity uses the inherited uniform ``dt*g`` body force (the base
    # applies it to all cells — correct for two-phase, where the air has
    # mass). A reduced-pressure (p_rgh) formulation would remove the
    # residual parasitic interface currents and enable a clean dynamic
    # pressure force, but it needs a *well-balanced* discretisation
    # (interFoam ``ghf·snGrad(rho)``); a naive version drives growing
    # currents, so it is left as a follow-up. Buoyancy is computed
    # gauge-robustly from the displaced volume (see ``_two_phase_forces``),
    # which does not depend on the pressure formulation.

    # ------------------------------------------------------------------
    #  Override: density-based projection coefficients
    # ------------------------------------------------------------------
    def _compute_variable_density_coefficients(self, timestep):
        """``c = dt / ρ_eff`` on each staggered face grid (+ a cc entry).

        ``ρ_eff = μ0·ρ_fluid + (1-μ0)·ρ_body`` blends the two-phase fluid
        density (water/air from the VOF field) with the immersed body density
        via the BDIM ``μ0`` Heaviside. With no body (placeholder far away)
        ``μ0 ≡ 1`` so ``ρ_eff = ρ_fluid``. Returns the same
        ``(ch, cv[, cw], ch_cc)`` tuple shape the base ``fluid_step`` expects;
        ``ch_cc`` is the FFT RHS divisor (unused on the MGCG path).
        """
        tp       = self.two_phase
        dt       = float(timestep)
        rho_body = float(self.rho_body)
        out      = []
        for d, ax in enumerate(self._bdim_axis_names):       # 'u','v'[,'w']
            rho_fluid_face = tp.density_face(d)
            mu0     = getattr(self, f'mu0_all_{ax}')         # body on this face
            rho_eff = mu0 * rho_fluid_face + (1.0 - mu0) * rho_body
            out.append(dt / rho_eff)
        # cell-centred entry (FFT RHS divisor; tuple-shape only on MGCG path)
        rho_eff_cc = self.mu0_all * tp.density_cc() + (1.0 - self.mu0_all) * rho_body
        out.append(dt / rho_eff_cc)
        return tuple(out)

    # ------------------------------------------------------------------
    #  Override: gauge-robust body force (displaced-volume buoyancy + viscous)
    # ------------------------------------------------------------------
    # The base surface-pressure integral ``∮ -p n dS`` cannot extract the
    # small buoyancy from the LARGE hydrostatic pressure (rho_w g H): the
    # smoothed-delta discretisation suffers catastrophic cancellation (the
    # raw force is gauge/discretisation noise — ~0 when submerged, nonzero in
    # air). So buoyancy is taken analytically from the displaced fluid volume
    # (gauge-robust) and the pressure-based DYNAMIC load is omitted for now
    # (it needs the well-balanced p_rgh formulation; see the gravity note
    # above). Buoyancy + viscous is correct for quasi-static floating.

    def _displaced_buoyancy(self):
        """Per-component buoyancy ``F_i = -g_i * ∫ rho_fluid (1-mu0) dV`` —
        the weight of the fluid displaced by the body (smoothed body
        indicator ``1-mu0`` from the union SDF). Returns a length-ndim list."""
        cb  = self.composite_body
        phi = cb.sdf_val                       # CC union SDF (positive in fluid)
        eps = float(self.eps)
        d   = (phi / eps).clamp(-1.0, 1.0)
        mu0 = 0.5 * (1.0 + d + torch.sin(torch.pi * d) / torch.pi)
        body_ind = 1.0 - mu0
        rho_f = self.two_phase.density_cc()
        inner = tuple(slice(1, -1) for _ in range(self.ndim))
        disp  = float((rho_f * body_ind)[inner].sum().item()) * (self.h ** self.ndim)
        return [-(float(g)) * disp for g in self._gravity]

    def _two_phase_forces(self, fn3d, vels, p, iteration):
        """Two-phase body loads = analytic displaced-fluid **buoyancy** +
        **viscous** stress.

        Buoyancy is computed from the displaced fluid volume (gauge-robust),
        NOT from the surface-pressure integral, which suffers catastrophic
        hydrostatic cancellation (``rho_w g H`` >> buoyancy). The viscous
        stress is taken from the inherited routine.

        NOTE: the pressure-based *hydrodynamic* load (form drag, wave
        radiation, added mass) is intentionally omitted here — recovering it
        cleanly needs the interFoam-style **reduced-pressure (p_rgh) solve**
        so the dynamic pressure is a solve variable rather than a post-hoc
        subtraction (a contaminated analytic ``p - p_hydro`` over- or
        under-shoots). Tracked as a follow-up; buoyancy + viscous is correct
        for quasi-static floating and the right base to build on.
        """
        zero  = torch.zeros_like(p)
        base  = _forces.forces_method2_3d if fn3d else _forces.forces_method2
        base(self, *vels, zero, iteration)             # viscous only (p=0)
        Fb = self._displaced_buoyancy()                # per-component buoyancy
        self.pressure_force_x = self.pressure_force_x + Fb[0]
        self.pressure_force_y = self.pressure_force_y + Fb[1]
        if self.ndim == 3:
            self.pressure_force_z = self.pressure_force_z + Fb[2]

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
