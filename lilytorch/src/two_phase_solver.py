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
* ``_apply_bdim_all_axes`` — air-transparent mu0/mu1 masking of the
  velocity BDIM (python path).
* ``project`` — **fused path** (the default): between the fused single-phase
  bdim_apply and the Poisson solve, repair the velocity with the exact
  air-transparent identity ``a·S + (1−a)·u′`` (u′ captured by wrapping
  ``adv_diff_solver.solve``) and rescale the kernel's water-normalised
  coefficients to the two-phase formulas via
  ``_rescale_kernel_coeffs_two_phase``. No CUDA or solver.py changes.
* ``finalize_step`` — transport the VOF field once per step.
* ``forces_method2`` / ``forces_method2_3d`` — stack per-body SDFs so the
  emergent (real-pressure) eulerian force loop sees them.

Everything else is inherited and reused unchanged:

* **gravity** — the base ``_apply_gravity_body_force`` adds ``dt·g`` to *all*
  cells (real air has mass), which is exactly what two-phase needs;
* **projection core** — the base ``project`` runs the variable-coefficient MGCG
  path ``∇·(c∇p)=div`` → ``u -= c∇p`` (closed-box all-Neumann, ``dirichlet_mask
  = None``); we just feed it the density-based ``c`` (built by the python
  coefficient override or written by the two-phase bdim_apply);
* advection–diffusion, BDIM, the Poisson solver, force integration.

The hydrostatic interface jump and the buoyancy on the body therefore emerge
automatically from the density-weighted pressure (``∇p = ρ_fluid g`` at rest).
"""

import torch

from lilytorch.src.solver import FluidSolver
from lilytorch.src import forces as _forces
from lilytorch.src.advection import _sl
from lilytorch.src.two_phase import TwoPhase, _neumann_pad
from lilytorch.src import native as _native


def body_aware_alpha_init(alpha_init, sdf_cc, eps, h, *,
                          compensate=True, verbose=True):
    """Make an ``alpha_init`` body-aware: carve the body interior out of the
    initial water and (optionally) raise the free surface so the total initial
    water volume still matches the uncarved init.

    The plain flat-level init (``alpha = (Z < zlevel)``) puts water *inside*
    any body straddling or below the interface. That phantom water is mostly
    inert (BDIM imposes the body velocity there, so it rides along with the
    body), but it offsets the mass diagnostic, puts the alpha=0.5 iso-surface
    through the body, and can be shed as spurious blobs by a fast-moving body.

    Carve
        ``alpha *= m`` with ``m`` a smoothed Heaviside of the cell-centred
        union SDF whose transition band lies strictly INSIDE the body
        (``m = 1`` for ``sdf >= 0``, smooth over ``(-eps, 0)``, ``m = 0`` for
        ``sdf <= -eps``).  The wetted outer BDIM band therefore stays fully
        wet: carving it (the historical ``m = mu0``, band centred ON the
        surface) stored an alpha deficit in cells whose velocity is only
        partially body-forced, and any perturbation advected that deficit
        into open water where it reads as buoyant air — erupting as spurious
        rising plumes anchored to the wetted hull (the amphibious-ramp "pipe
        jets"; sharp corners leak first).  Inside the shifted band the
        velocity is strongly body-forced (``mu0 < 0.5``), so the remaining
        deficit rides with the body.  The cut is still smooth (no 0/1 jump)
        and the dryness contract is unchanged: ``alpha[sdf <= -eps] == 0``.

    Compensation
        Carving deletes the submerged body volume from the water. With
        ``compensate`` the wrapped init shifts the vertical coordinate fed to
        ``alpha_init`` (raising the surface) until the carved volume matches
        the uncarved target, by bracketing + bisection on the shift. Sharp
        (Heaviside) inits make the volume a staircase in the shift, so the
        final field is the exact-volume linear blend of the two bracketing
        fields — total volume matches the target to round-off. Body poses are
        NEVER touched: the water level moves, the initial draft does not.

    Parameters
    ----------
    alpha_init : callable -- ``alpha_init(X, Y[, Z])`` on the cell-centred grid.
    sdf_cc : tensor -- cell-centred union body SDF on the same grid
        (``composite_body.sdf_val``), evaluated at the initial body poses.
    eps : float -- BDIM band half-width (``solver.eps``).
    h : float -- uniform cell size (sets the bisection scales).
    compensate : bool -- restore the uncarved total water volume (default True).

    Returns
    -------
    callable with the same ``(X, Y[, Z])`` signature, suitable to pass to
    :class:`~lilytorch.src.two_phase.TwoPhase`.
    """
    eps = float(eps)
    h = float(h)
    # Transition band shifted INSIDE the body: half-width eps/2 centred at
    # sdf = -eps/2, i.e. m(sdf>=0)=1 (wetted band fully wet, nothing for the
    # flow to advect out) and m(sdf<=-eps)=0 (interior dry) — see docstring.
    d = (2.0 * sdf_cc / eps + 1.0).clamp(-1.0, 1.0)
    mu0 = 0.5 * (1.0 + d + torch.sin(torch.pi * d) / torch.pi)

    def wrapped(*coords):
        if mu0.shape != coords[0].shape:
            raise ValueError(
                "body_aware_alpha_init: composite_body.sdf_val shape "
                f"{tuple(mu0.shape)} does not match the cell-centred grid "
                f"{tuple(coords[0].shape)}."
            )
        m = mu0.to(device=coords[0].device)

        def carved(dz):
            cs = list(coords)
            cs[-1] = coords[-1] - dz          # vertical = last axis (y/2D, z/3D)
            return alpha_init(*cs).clamp(0.0, 1.0) * m

        inner = tuple(slice(1, -1) for _ in coords)
        vol = lambda a: float(a[inner].sum())

        a_lo = carved(0.0)
        if not compensate:
            return a_lo
        target = vol(alpha_init(*coords).clamp(0.0, 1.0))
        v_lo = vol(a_lo)
        deficit = target - v_lo               # submerged body volume (in cells)
        if deficit <= 0.25:                   # < a quarter-cell: nothing to do
            return a_lo
        # Bracket the shift: grow hi until the carved volume reaches the target.
        zc = coords[-1]
        max_dz = 0.5 * float(zc.max() - zc.min())
        lo, hi = 0.0, h
        while vol(carved(hi)) < target and hi < max_dz:
            hi *= 2.0
        a_hi = carved(hi)
        v_hi = vol(a_hi)
        if v_hi < target:
            print(f"body_aware_alpha_init: WARNING -- could not restore the "
                  f"displaced volume ({deficit:.1f} cells) within a surface "
                  f"rise of {hi:.3g}; short by {target - v_hi:.1f} cells.")
        else:
            # Bisection (volume is a staircase for sharp inits, so converge
            # the bracket, not the residual), then exact-volume blend.
            for _ in range(60):
                if hi - lo < 1e-3 * h:
                    break
                mid = 0.5 * (lo + hi)
                a_mid = carved(mid)
                v_mid = vol(a_mid)
                if v_mid < target:
                    lo, a_lo, v_lo = mid, a_mid, v_mid
                else:
                    hi, a_hi, v_hi = mid, a_mid, v_mid
            theta = 0.0 if v_hi <= v_lo else (target - v_lo) / (v_hi - v_lo)
            a_lo = (a_lo + theta * (a_hi - a_lo)).clamp(0.0, 1.0)
        if verbose:
            print(f"body_aware_alpha_init: carved {deficit:.1f} cells of body "
                  f"interior out of the water; surface raised by "
                  f"~{0.5 * (lo + hi):.4g} to compensate "
                  f"(volume {vol(a_lo):.1f} / target {target:.1f} cells).")
        return a_lo

    return wrapped


class TwoPhaseSolver(FluidSolver):
    """Variable-density two-phase solver (water + real air) via VOF."""

    # Per-step memo for :meth:`_mu0_cc`.  Class-level default so the getter is
    # safe during __init__, before any instance attribute is bound.
    _mu0_cc_cache = None

    def __init__(self, pars, dtype=None, custom_update=None, compute_forces=True):
        # Two-phase default Poisson: mgcg (multigrid-preconditioned CG) is
        # ~2.2x faster than plain multigrid for the variable-density (833:1)
        # two-phase Poisson.  Injected before the base __init__ reads the key,
        # so an explicit poisson_method in the config still wins.
        pars["solver"].setdefault("poisson_method", "mgcg")
        super().__init__(pars, dtype=dtype, custom_update=custom_update,
                         compute_forces=compute_forces)
        solver = pars["solver"]
        tp_cfg = solver.get("two_phase")
        if tp_cfg is None:
            raise ValueError(
                "TwoPhaseSolver requires a 'solver.two_phase' config block."
            )
        if self.poisson_method not in ("multigrid", "mgcg", "rmgcg"):
            raise ValueError(
                "two_phase requires poisson_method 'multigrid', 'mgcg', or 'rmgcg' "
                "(the FFT solver cannot do a variable-density Poisson)."
            )
        # The two-phase step reuses the single-phase fused kernels untouched.
        # The two-phase repairs run in ``project`` (see the fused-path override
        # section below): the BDIM velocity is fixed by the exact identity
        # ``a*S + (1-a)*u'`` (the historical blocker — the kernel imposing the
        # body velocity into the air — needed only u', captured by wrapping
        # ``adv_diff_solver.solve``), and the Poisson coefficient by the
        # out-of-place rescale of the kernel's water-normalised
        # ``dt*mu0/rho_water``.
        # bdim_mu0_projection=True (default): μ₀ recovered from the kernel
        # coefficient (fast, one mul per cell).  False: μ₀ computed from the
        # persistent cell-centred union SDF, matching the old cuda_kernels
        # path — needed when the staggered-face μ₀ in the kernel does not
        # agree with the face-averaged cell-centred μ₀ (e.g. large bodies at
        # an angle like the amphibious ramp).
        self._init_two_phase(tp_cfg)
        self._rescale_jcap_tol(solver_cfg=pars.get("solver", {}))

        # ── Align body eps with the solver's Maertens-Weymouth eps ──────
        # The BDIM kernel reads comp.eps for its mu0 computation; comp.mu_funcs
        # and the two-phase coefficient rescale (via _mu0_cc) also use it.  The
        # alpha carve uses self.eps.  Overwriting comp.eps with self.eps makes
        # EVERY component (kernel velocity enforcement, Poisson coefficients,
        # and the initial alpha carve) use the SAME eps_multiplier * h value.
        self.composite_body.eps = float(self.eps)

        # Fused path: capture the advection output u' — the only input the
        # air-transparent velocity identity needs.  ``solve`` returns
        # adv_diff_solver's PERSISTENT output buffers (``_sl_out`` /
        # ``_conv_out``), so this stash aliases buffers that already live for
        # the whole run: it holds no extra memory, and it stays valid across a
        # pre-Poisson graph REPLAY (where this python wrapper does not run but
        # the captured kernels still rewrite those same buffers in place).
        self._kernel_primes = None
        _orig_solve = self.adv_diff_solver.solve

        def _solve_and_stash(*a, **k):
            out = _orig_solve(*a, **k)
            self._kernel_primes = out
            return out

        self.adv_diff_solver.solve = _solve_and_stash
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
        # Body-aware initial interface: carve the body interior out of the
        # initial water (alpha *= mu0, the BDIM smoothed Heaviside of the union
        # SDF at the INITIAL body poses) and raise the surface to restore the
        # displaced volume (see ``body_aware_alpha_init``). Init-time only — no
        # per-step cost: alpha is advected with the BDIM composite velocity,
        # which inside the body IS the body velocity, so the dry interior rides
        # along with the body. Body poses / initial draft are never touched.
        # Requires the sdf-gated transparency mask below (an air interior would
        # otherwise make the wetted hull transparent under
        # ``air_transparent_body``). Default OFF: every existing case unchanged.
        self._alpha_exclude_body = bool(cfg.get("alpha_exclude_body", False))
        self._alpha_carve_pending = False
        if self._alpha_exclude_body:
            sdf_cc = getattr(self.composite_body, "sdf_val", None)
            if sdf_cc is None:
                raise ValueError(
                    "two_phase.alpha_exclude_body requires a composite body "
                    "with a cell-centred SDF (composite_body.sdf_val)."
                )
            cb = self.composite_body
            farms_coupled = (bool(getattr(cb, "custom_update", None))
                             or type(cb).__name__ == "MultiAnimatBodies")
            # Placeholder SDF (filled with _FAR=1e4): no body stamped on the
            # grid yet — carving now would silently remove nothing.
            no_body_yet = float(sdf_cc.min()) > 100.0
            if farms_coupled or no_body_yet:
                # FARMS-coupled: at construction the composite SDF still holds
                # the YAML/default poses (MultiAnimatBodies: a _FAR placeholder)
                # — the MuJoCo spawn transform (e.g. the boat's roll=pi/2 +
                # SPAWN_Z) is only stamped in by the handler's first update().
                # Carving NOW would dry the WRONG region (or nothing) and leave
                # the real body interior wet.  Defer the carve to the first
                # fluid_step (which runs after that update).
                self._alpha_carve_pending = True
                self._alpha_carve_raw = alpha_init
                self._alpha_carve_compensate = bool(
                    cfg.get("alpha_volume_compensate", True))
            else:
                # Standalone solver: bodies are posed at construction — carve now.
                alpha_init = body_aware_alpha_init(
                    alpha_init, sdf_cc, self.eps, self.h,
                    compensate=bool(cfg.get("alpha_volume_compensate", True)),
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
        # ``consistent_momentum`` (Nangia et al. 2019 conservative rho*u
        # transport) was deleted in the Phase 4.4 cleanup as an unused python
        # path and is now REINSTATED on the fused native path -- see
        # ``_advect_momentum`` below.  It transports rho*u with the same
        # upwind mass flux that evolves rho, which is the standard cure for
        # the high-density-ratio instability (non-conservative transport is
        # stable only to ~100:1; water/air is 816:1).  Default OFF, so every
        # existing case is byte-for-byte unchanged.
        self._consistent_momentum = bool(cfg.get("consistent_momentum", False))
        # Reconstruction of the shared mass flux: 0 = first-order donor cell
        # (the recovered reference), 1 = W&Y Courant-corrected van-Leer donor
        # for the density, 2 = also van-Leer MUSCL for the advected velocity.
        # Default 2: level 0 is stable but its numerical diffusion damps
        # resolved surface waves.
        self._cm_flux_scheme = int(cfg.get("consistent_momentum_flux", 2))
        # ``alpha_fluid_exact`` was a transitional flag while the exact
        # ``min(alpha, mu0)/mu0`` replaced the amplifying ``alpha/max(mu0,1e-3)``
        # in _alpha_fluid_cc.  The exact form is now the ONLY form: it is
        # strictly better measured, mathematically correct where the clamp was
        # not, and it needs no tolerance.  Reject the key rather than silently
        # ignoring it.
        if "alpha_fluid_exact" in cfg:
            raise ValueError(
                "two_phase.alpha_fluid_exact was removed: the exact "
                "min(alpha, mu0)/mu0 it selected is now the only behaviour "
                "(see TwoPhaseSolver._alpha_fluid_cc).  Delete the key."
            )
        # Weight the face density by the fluid volume mu0 either side, so a
        # carved (alpha = 0, i.e. "air") body interior cannot vote on the
        # density of the faces bounding the hull.  Default OFF; see
        # _kernel_face_mean_fluid_weighted.  This is the BDIM-legal way to say
        # "the density at the body is the density of the FLUID there" -- unlike
        # rho_solid, it invents no solid density and touches no physics, it
        # only stops a zero-fluid cell contributing to a fluid average.
        self._face_density_fluid_weighted = bool(
            cfg.get("face_density_fluid_weighted", False))
        # NOTE: consistent momentum + alpha_exclude_body used to be rejected
        # here.  The incompatibility was NOT intrinsic -- it was the
        # ``alpha / max(mu0, 1e-3)`` divisor, which amplified any alpha/mu0
        # drift in the BDIM band by up to 1000x (measured: |p| 1.6 kPa ->
        # 1.1e6 Pa by it 5750, with the flow still healthy).  With the exact
        # ``min(alpha, mu0)/mu0`` the combination runs: 7000 steps, median |p|
        # 1598 Pa, and 925 pressure excursions against the 893 the carve
        # produces on its OWN -- i.e. consistent momentum adds nothing to the
        # carve's own, still-unexplained blow-up (onset it 2738 regardless of
        # transport scheme, timestep or coefficient formulation).
        if self._cm_flux_scheme not in (0, 2):
            extra = ""
            if self._cm_flux_scheme == 1:
                extra = (
                    "\nScheme 1 (van-Leer density + UPWIND velocity) was "
                    "removed.  It existed to answer one question -- is "
                    "consistent momentum stabilising because of the SHARED "
                    "flux, or merely because first-order upwinding is "
                    "diffusive?  If diffusion were the cause, 1 and 2 would "
                    "have reverted toward the stock growth rate of 7.36 /s; "
                    "measured, they are 1.20 and 1.12.  The shared flux is the "
                    "cause, the experiment is finished, and 2 supersedes 1 by "
                    "limiting the velocity as well at the same cost.  Use 2."
                )
            raise ValueError(
                "two_phase.consistent_momentum_flux must be 0 (first-order "
                "donor cell, the reference the CUDA port is validated against) "
                f"or 2 (van-Leer limited, the production default), got "
                f"{self._cm_flux_scheme}.{extra}"
            )
        # Consistent transport REPLACES the momentum advection outright, so
        # ``solver.convection_method`` stops having any effect.  Say so out
        # loud: a config carrying both reads as if the named scheme were
        # running, and this project has already been bitten by settings that
        # were silently ignored (BRIEF §7 -- YAML drops unknown keys without a
        # word, and a whole batch of experiment configs quietly degraded to
        # plain baselines).
        if self._consistent_momentum:
            _named = getattr(self.adv_diff_solver, "_scheme_name", None)
            print(
                f"TwoPhaseSolver: consistent_momentum is ON, so "
                f"solver.convection_method='{_named}' is IGNORED -- momentum is "
                f"transported by the shared mass flux "
                f"(consistent_momentum_flux={self._cm_flux_scheme}), and the "
                f"Weymouth & Yue VOF sweep is skipped (the interface rides on "
                f"the density).", flush=True)
        if self._consistent_momentum and self.ndim != 3:
            raise ValueError(
                "two_phase.consistent_momentum is implemented for the 3-D "
                "fused path only (no 2-D kernel)."
            )
        if cfg.get("consistent_n_cycles", 1) not in (1, None):
            raise ValueError(
                "two_phase.consistent_n_cycles > 1 (Nangia fixed-point "
                "iteration) was not ported to the fused kernel; only the "
                "single forward-Euler pass (n_cycles=1) is available."
            )
        # ``mu0_free_coeff`` stays removed: it was a toy-boat waterline
        # experiment superseded by rho_solid / air_transparent_body.
        if cfg.get("mu0_free_coeff"):
            raise ValueError(
                "solver.two_phase.mu0_free_coeff was removed (Phase 4.4 "
                "cleanup); use rho_solid / air_transparent_body / "
                "alpha_exclude_body to stabilise the waterline band."
            )
        # Three-phase density (Nangia 2019 WSI, Eq. 23): treat the body as a
        # smoothed THIRD density phase rho_solid, blended by mu0 (the BDIM fluid
        # fraction IS the body Heaviside), INSIDE the variable-density projection
        # coefficient -- instead of mu0-EXCLUDING the body. Targets the
        # body-interface band spurious current (the killer).
        # None = off (current mu0-exclusion behaviour). Python path only for now.
        self._rho_solid = cfg.get("rho_solid", None)
        if self._rho_solid is not None:
            self._rho_solid = float(self._rho_solid)
        # Air-transparent body: mask the BDIM fluid fraction mu0 by the water
        # fraction alpha so the body effectively does NOT exist in the air phase.
        # Uses the resolved VOF fraction continuously: the body is fully present
        # in water (alpha=1), transparent in air (alpha=0), and transitions over
        # the same interface band as the material properties.  Keeping this
        # coefficient continuous avoids grid-scale forcing at the contact line.
        # Physically motivated: air forces on the body are ~1000x smaller than
        # water forces, so ignoring the above-water body is an excellent
        # approximation.  Default ON; set to False to restore legacy behaviour.
        self._air_transparent_body = bool(
            cfg.get("air_transparent_body", True))
        # The old python-only partial-Heaviside pressure readout has been
        # removed: two-phase runs now go through the native streaming force op,
        # whose default union-∂H readout (force_link_normal='union') already
        # carries the summation-by-parts gauge fix for BOTH channels.
        flags = []
        if self._consistent_momentum:
            flags.append("CONSISTENT rho*u transport (Nangia 2019; replaces "
                         "velocity advection AND the W&Y VOF sweep; flux "
                         f"scheme {self._cm_flux_scheme})")
        if self._rho_solid is not None:
            flags.append(f"three-phase rho_solid={self._rho_solid}")
        if self._air_transparent_body:
            flags.append("air-transparent-body")
        if self._alpha_exclude_body:
            flags.append("body-aware alpha init (carve"
                         + ("+volume-compensate)" if cfg.get(
                             "alpha_volume_compensate", True) else ")"))
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

    def _rescale_jcap_tol(self, solver_cfg):
        """Carry the degenerate-cell threshold with the coefficient scale.

        ``poisson_jcap_tol`` freezes a Poisson row whose diagonal falls below
        it (WaterLily's ``iszero(D)`` guard).  It is an ABSOLUTE number, and
        its default was calibrated against the old ``dt*mu0/rho`` coefficient:
        healthy water gave ``J = 6*dt/rho_water = 6e-6`` at dt = 1e-3, so the
        1e-7 default sat at ~1.7 % of a healthy diagonal.

        The WaterLily pressure-impulse scaling multiplies every coefficient by
        ``rho_water/dt`` -- a factor of 1e6 at that timestep -- so the same
        1e-7 became 1.7e-8 of healthy and the guard silently stopped firing.
        Near-dead cells inside a body (measured: J = 7e-6 against a healthy
        6.0, with one connected face out of six) were then solved rather than
        frozen, and ``p = h^2*div/J`` turned them into 1e7 Pa.  That pressure
        reaches the band quadrature: peak load on the salamandra went to
        127.9 N with 150.8 N single-step jumps, on a 1.3 kg robot.  Rescaling
        the threshold with the coefficient returns it to 4.7 N / 0.38 N.

        Scaling it here rather than asking configs for a magic number is the
        point: an absolute tolerance left behind by a rescale is exactly how
        this broke, and a hand-set value would break the same way at the next
        change of dt.  An explicit ``poisson_jcap_tol`` still wins.
        """
        if "poisson_jcap_tol" in solver_cfg:
            return                       # user asked for a specific value
        tp = self.two_phase
        if tp.rho_water == tp.rho_air and self._rho_solid is None:
            return                       # single-phase reduction: not rescaled
        ps = getattr(self, "poisson_solver", None)
        if ps is None or not hasattr(ps, "jcap_tol"):
            return
        factor = float(tp.rho_water) / float(self.dt)
        ps.jcap_tol = float(ps.jcap_tol) * factor
        print(f"TwoPhaseSolver: poisson_jcap_tol scaled by rho_water/dt = "
              f"{factor:.3g} to match the WaterLily coefficient scaling -> "
              f"{ps.jcap_tol:.3g} (healthy two-phase diagonal is ~6).",
              flush=True)

    def _alpha_face(self, d):
        """Water fraction ``alpha`` on the staggered *d*-face grid (full grid;
        see :meth:`_face_mean`)."""
        return self._face_mean(self.two_phase.alpha, d)

    def _momentum_viscosity(self, *vels):
        """Return the phase-dependent kinematic viscosity used by momentum.

        The base solver's scalar ``solver.nu`` is only a single-phase default.
        Two-phase momentum must use ``nu_water``/``nu_air`` from the transported
        VOF field.  Keep the result in a pointer-stable buffer so CUDA graph
        replay sees the current alpha field rather than a captured temporary.
        LES/Carreau increments, when enabled, are added to this molecular
        viscosity.
        """
        ref = vels[0]
        if (getattr(self, '_nu_eff_graph', None) is None
                or self._nu_eff_graph.shape != ref.shape
                or self._nu_eff_graph.dtype != ref.dtype
                or self._nu_eff_graph.device != ref.device):
            self._nu_eff_graph = torch.empty_like(ref)
        out = self._nu_eff_graph
        self.two_phase.viscosity_cc(out=out)
        nu_extra = self._compute_nu_t(*vels)
        if nu_extra is not None:
            out.add_(nu_extra)
        return out

    # ------------------------------------------------------------------
    #  Consistent conservative mass/momentum transport (Nangia et al. 2019)
    # ------------------------------------------------------------------
    def _init_cm_buffers(self, *vel):
        """Lazily allocate the pointer-stable scratch the consistent-momentum
        op writes.  Allocation happens on an eager warm-up step (the graph
        runner runs three before it captures), so the capture itself stays
        allocation-free."""
        ref = vel[0]
        def _stale(b):
            return (b is None or b.shape != ref.shape or b.dtype != ref.dtype
                    or b.device != ref.device)
        if _stale(getattr(self, '_cm_rho_new', None)):
            self._cm_rho_new   = torch.empty_like(self.two_phase.alpha)
            self._cm_alpha_out = torch.empty_like(self.two_phase.alpha)
            self._cm_flux      = torch.empty(
                (3,) + tuple(self.two_phase.alpha.shape),
                dtype=self.two_phase.alpha.dtype,
                device=self.two_phase.alpha.device)
        if (getattr(self, '_cm_out', None) is None
                or any(_stale(b) for b in self._cm_out)):
            self._cm_out = tuple(torch.empty_like(x) for x in vel)
            self._cm_diff_copy = tuple(torch.empty_like(x) for x in vel)

    def _advect_momentum(self, *vel, nu_eff=None):
        """Consistent conservative ``rho*u`` transport in place of the
        velocity advection (Nangia et al. 2019).

        Transports ``rho*u`` with the SAME upwind mass flux that evolves
        ``rho`` and recovers ``u = rho*u / rho_new``, so the 816:1 density
        jump cannot amplify at the interface.  This single call replaces
        BOTH the velocity advection AND the Weymouth & Yue VOF sweep: the
        interface rides on the density, and ``two_phase.alpha`` is written
        in place from it (``_advect_vof`` becomes a no-op).

        Gravity is applied as ``rho*g`` on the momentum here, so
        ``_apply_gravity_body_force`` suppresses the predictor ``dt*g``.

        Runs inside the graph capture: launch-only after the first eager
        warm-up step.
        """
        if not self._consistent_momentum:
            return super()._advect_momentum(*vel, nu_eff=nu_eff)
        u, v, w_vel = vel
        tp = self.two_phase
        self._init_cm_buffers(u, v, w_vel)
        # Publish u' for the air-transparent velocity identity.  On the stock
        # path that stash is done by the ``adv_diff_solver.solve`` wrapper
        # installed in __init__ -- which this override never calls, so without
        # this line ``_kernel_primes`` stays None forever and
        # ``_kernel_blend_velocities`` returns early on EVERY step: the BDIM
        # kernel then keeps imposing the body velocity into the air while the
        # rescaled coefficients say the body is not there, which is the
        # historical kernel-mode blow-up.  It is invisible in the fluid probes
        # (pinning the air to the hull is if anything stabilising) and shows up
        # only as corrupt body loads.  Assign after _init_cm_buffers so a
        # reallocation cannot leave the stash pointing at a dead buffer.
        self._kernel_primes = self._cm_out
        g = self._gravity if self.use_gravity else (0.0, 0.0, 0.0)
        # Every scalar goes through _cached_float: dt / h / nu may be 0-d GPU
        # tensors, and float(gpu_tensor) is a host<->device sync -- which is
        # ILLEGAL inside a CUDA graph capture (it fails the whole capture, not
        # just this call).
        cf = self._cached_float
        dt_f = cf('dt', self.dt)
        _native.consistent_momentum_3d(
            tp.alpha, u, v, w_vel,
            self._cm_rho_new, self._cm_flux,
            self._cm_out[0], self._cm_out[1], self._cm_out[2],
            self._cm_alpha_out,
            tp.rho_water, tp.rho_air, dt_f, cf('h', self.h),
            cf('g0', g[0]), cf('g1', g[1]), cf('g2', g[2]),
            self._cm_flux_scheme,
        )
        # The op writes a separate buffer because the schema declares `alpha`
        # an immutable input; commit it, then restore the zero-gradient ghost
        # contract that the W&Y path maintains via _neumann_pad (the density
        # update leaves ghost cells at their previous value).
        tp.alpha.copy_(self._cm_alpha_out)
        _neumann_pad(tp.alpha)
        for i in range(len(vel)):
            _native.diffuse_add(
                self._cm_out[i], self._cm_diff_copy[i], dt_f,
                dh=self.adv_diff_solver.dh, nu_eff=nu_eff,
                nu=cf('nu', self.nu),
            )
        return tuple(self._cm_out)

    def _probe_band_alpha(self, a_f):
        """Opt-in diagnostic: how healthy is ``alpha/mu0`` inside the BDIM band?

        Set ``LILYTORCH_BAND_ALPHA_PROBE=<every_n_steps>``.  Runs on BOTH the
        stock and the consistent-momentum paths (it hangs off
        ``_alpha_fluid_cc``, which both call every step), so the two are
        directly comparable at matched iterations.

        The number that matters is the fraction of WETTED band cells reading as
        partial air (``a_f < 0.5``): per this method's docstring that is the
        buoyant shell which erupts as spurious plumes.  Off by default; the
        reductions sync, so never leave it on in a production run.
        """
        import os
        n = os.environ.get("LILYTORCH_BAND_ALPHA_PROBE")
        if not n:
            return
        it = getattr(self, "_band_probe_it", 0)
        self._band_probe_it = it + 1
        if it % int(n):
            return
        comp = self.composite_body
        sdf = getattr(comp, "sdf_val", None)
        if sdf is None:
            return
        eps = self._cached_float("eps", self.eps)
        band = (sdf > -eps) & (sdf < eps)
        nb = int(band.sum())
        if nb == 0:
            return
        af = a_f[band]
        wet = af > 0.05                      # ignore the dry (air-side) band
        nwet = int(wet.sum())
        dry_ish = int((af < 0.5).sum())
        print(f"[band] it={it:6d} band={nb} wet={nwet} "
              f"a_f<0.5={dry_ish} ({100.0 * dry_ish / nb:5.1f}%) "
              f"min={float(af.min()):.4f} mean={float(af.mean()):.4f} "
              f"alpha_max={float(self.two_phase.alpha[band].max()):.4f}",
              flush=True)

    def _alpha_fluid_cc(self):
        """Water fraction of the cell's FLUID part (cell-centred).

        ``alpha`` is the water fraction of the TOTAL cell volume
        (water + air + body = 1).  With the ``alpha_exclude_body`` carve the
        BDIM band of a wetted hull holds ``alpha ≈ mu0 < 1`` even though its
        fluid part is pure water; feeding that raw alpha into the density
        blend and the air-transparency masks makes the band read as partial
        AIR — a buoyant ~rho/2 shell pinned along the entire wetted surface,
        held only by quiescence, that erupts as spurious rising plumes once
        perturbed (the amphibious-ramp "pipe jets").  Normalising by the BDIM
        fluid fraction removes the body volume from the phase bookkeeping:

            a_f = clamp(alpha / mu0, 0, 1)

        is 1 in the wetted band, 0 in air, equals alpha away from bodies
        (mu0 = 1), and the clamp bounds the mu0→0 interior (where the sdf
        gates / coefficient floor rule anyway).  Identity when the carve is
        off: alpha is then 0/1 straight through the body, so every existing
        uncarved case is bit-unchanged.
        """
        a = self.two_phase.alpha
        if not self._alpha_exclude_body:
            return a
        mu0 = self._mu0_cc()
        # Capping the NUMERATOR at mu0 makes the ratio <= 1 by construction, so
        # it is finite for EVERY mu0 > 0 -- checked down to float32 tiny; the
        # smallest mu0 the band Heaviside produces is ~5e-10.  There is no
        # tolerance to pick: the test is the exact `mu0 > 0`.  Outside the band
        # mu_funcs writes `(d >= 0)`, i.e. exactly 0.0 or exactly 1.0, so `> 0`
        # is a clean partition and not a threshold.  The final clamp is to the
        # PHYSICAL bounds of a volume fraction, not a tuning constant: alpha is
        # observed to go slightly negative (-9e-5 in a real run) and a negative
        # a_f would drive the density blend toward zero.
        #
        # This superseded ``alpha / max(mu0, 1e-3)``, which was not merely a
        # cruder spelling but a DIFFERENT number where it matters: a consistent
        # band cell with mu0 = alpha = 1e-4 has a_f = 1 (its fluid part is pure
        # water), where the clamp returned 1e-4/1e-3 = 0.1 -- reading a wet cell
        # as 90 % AIR.  That is exactly the buoyant shell this method's
        # docstring describes, manufactured by the guard meant to prevent it.
        # Measured on the pool ramp (carve on, 7000 steps): peak |p| 1.9e7 ->
        # 1.2e7, p95 9.3e4 -> 4.3e4, |u|max 2.009 -> 1.845.
        #
        # ⚠ What a_f means at mu0 == 0 is a DEFINITION, not a tolerance: the
        # cell has no fluid part.  0 is used.  It is not fully inert, because
        # ``_rescale_kernel_coeffs_two_phase`` face-averages q(a_f); set
        # ``face_density_fluid_weighted`` to drop zero-fluid cells out of that
        # average and remove the choice entirely.
        a_f = torch.where(mu0 > 0, torch.minimum(a, mu0) / mu0,
                          torch.zeros_like(a)).clamp(0.0, 1.0)
        self._probe_band_alpha(a_f)
        return a_f

    # ------------------------------------------------------------------
    #  Override: BDIM velocity imposition — air-transparent body
    # ------------------------------------------------------------------
    def _apply_bdim_all_axes(self, vels):
        """Apply BDIM to each velocity component, with the body transparent
        in the air phase when ``air_transparent_body`` is on.

        In the air (α=0) the body is invisible: μ0_eff = 1, μ1_eff = 0 →
        the BDIM meta-equation reduces to the identity (no body forcing).
        In the water (α=1) the standard BDIM is applied unchanged.
        The kernel path applies the same masking via the equivalent blend
        identity ``a*S + (1-a)*u'`` in :meth:`_kernel_blend_velocities`.
        """
        if not self._air_transparent_body:
            return super()._apply_bdim_all_axes(vels)
        comp = self.composite_body
        a_f = self._alpha_fluid_cc()   # fluid-part fraction; see _alpha_fluid_cc
        out = []
        for i, ax in enumerate(self._bdim_axis_names):
            mu0 = getattr(self, f'mu0_all_{ax}')       # read-only; arithmetic
            mu1 = getattr(self, f'mu1_all_{ax}')       # creates new tensors
            body_vel = getattr(comp, f'body_{ax}')
            normals = tuple(
                getattr(self, f'normal_{n}_{ax}')
                for n in self._bdim_normal_names
            )
            # Mask mu0, mu1 continuously by the resolved fluid-part water
            # fraction on this face grid.
            a_face = self._face_mean(a_f, i)
            # mu0_eff = alpha*mu0 + (1-alpha) → 1 in air, mu0 in water
            mu0_eff = a_face * mu0 + (1.0 - a_face)
            # mu1_eff = alpha*mu1 → 0 in air (no normal-derivative correction)
            mu1_eff = a_face * mu1
            if self._alpha_exclude_body:
                # Carved init (alpha_exclude_body): the dry body interior is
                # alpha=0, so the transparency mask must be gated on the face
                # SDF or the body stops imposing its velocity inside itself
                # (and the carved air pocket would not ride with the body).
                sdf_face = getattr(comp, f"sdf_val_{ax}")
                inside = sdf_face < 0.0
                mu0_eff = torch.where(inside, mu0, mu0_eff)
                mu1_eff = torch.where(inside, mu1, mu1_eff)
            out.append(self._bdim_apply(
                vels[i], mu0_eff, body_vel, mu1_eff, *normals))
        return tuple(out)

    # ------------------------------------------------------------------
    #  Override: KERNEL-mode two-phase (velocity blend + coefficient rescale)
    # ------------------------------------------------------------------
    # The fused single-phase bdim_apply writes ``S = mu0*(u'-b) + b + mu1*nd``
    # and ``c_kernel = dt*mu0/rho_water`` — wrong for two-phase on BOTH
    # counts.  Both are repaired here in a handful of python tensor ops, with
    # NO custom kernels and NO solver.py changes, thanks to two identities:
    #
    # 1. VELOCITY.  The air-transparent masking (mu0_eff = a*mu0 + (1-a),
    #    mu1_eff = a*mu1) satisfies EXACTLY
    #
    #        T = mu0_eff*(u'-b) + b + mu1_eff*nd  =  a*S + (1-a)*u'
    #
    #    — mu0, mu1, normals and the normal derivative all cancel, so the
    #    masked result is just the alpha-blend of the kernel's BDIM output
    #    with the raw advected velocity u'.  The only kernel temporary needed
    #    is u' (the advection output), which we capture by wrapping
    #    ``adv_diff_solver.solve``; the reference is dropped before the
    #    Poisson solve, exactly where the base step frees it, so peak memory
    #    is unchanged.  (This masking of the VELOCITY is what the historical
    #    kernel-mode blow-up was missing: the coefficient rescale alone left
    #    the kernel imposing the body velocity into the air.)
    #
    # 2. COEFFICIENT.  ``mu0 = c_kernel * rho_water/dt`` recovers the fluid
    #    fraction from the kernel coefficient (requires bdim_mu0_projection,
    #    enforced at init), after which the two-phase formulas (standard /
    #    rho_solid three-phase, with the air-transparent masking and the
    #    carved-init gate) are evaluated directly on the face grids.
    #    Out-of-place: the persistent ``_ch_persist`` buffers must stay
    #    water-normalised for the next step's Kernel-B overwrite.
    #
    # Both fixes run inside ``project`` (between BDIM and the Poisson solve —
    # after the projection would be too late, the wrong velocity would
    # already have seeded the pressure RHS).

    def _face_mean(self, q, d):
        """Cell-centred ``q`` averaged to the staggered *d*-face grid (full
        grid; MAC: face *i* between cells *i-1* and *i*; boundary face copies
        the adjacent cell, like ``recip_density_face``)."""
        out = q.clone()
        lo = _sl(self.ndim, d, slice(None, -1))
        hi = _sl(self.ndim, d, slice(1, None))
        out[hi] = 0.5 * (q[lo] + q[hi])
        return out

    def _kernel_face_mean_fluid_weighted(self, q, w, d):
        """Face mean of *q* weighted by the FLUID VOLUME *w* (= mu0) either side.

            q_face = (w_L q_L + w_R q_R) / (w_L + w_R)

        A cell with no fluid (mu0 = 0) contributes nothing, so the face density
        is set entirely by whichever side actually holds fluid.  That is the
        point: with the ``alpha_exclude_body`` carve the body interior is
        alpha = 0, i.e. it reads as AIR, and a plain arithmetic face mean lets
        that phantom air into every face bounding the hull -- the wetted band
        is water on one side and "air" on the other, so the face comes out at
        ~half air (1/rho 0.408 against water's 0.001).  Weighting by mu0 says
        instead: the density of a face is the density of the FLUID at that
        face, and the body's interior has no vote.

        Falls back to the plain mean where both sides are dry, which is a cell
        pair wholly inside the body -- the coefficient there is mu0*(...) = 0
        regardless, so the value is immaterial and only needs to be finite.
        """
        if self.ndim == 2:
            lo = _sl(2, d, slice(None, -1))
            hi = _sl(2, d, slice(1, None))
        else:
            lo = [slice(1, -1)] * 3
            hi = [slice(1, -1)] * 3
            lo[d] = slice(None, -1)
            hi[d] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
        wl, wr = w[lo], w[hi]
        den = wl + wr
        return torch.where(den > 0,
                           (wl * q[lo] + wr * q[hi]) / den.clamp(min=1e-30),
                           0.5 * (q[lo] + q[hi]))

    def _kernel_face_mean(self, q, d):
        """Like :meth:`_face_mean` but in the KERNEL coefficient-buffer
        layout: full padded grid in 2-D, ghost-cropped staggered face grid in
        3-D (the shapes of ``_ch_persist``/``_cv_persist``/``_cw_persist``)."""
        if self.ndim == 2:
            return self._face_mean(q, d)
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[d] = slice(None, -1)
        hi[d] = slice(1, None)
        return 0.5 * (q[tuple(lo)] + q[tuple(hi)])

    def _mu0_cc(self):
        """Cell-centred BDIM fluid fraction μ₀ from the union body SDF.

        When ``bdim_mu0_projection=False`` the kernel writes a constant
        ``dt/ρ`` coefficient and the Poisson rescale computes μ₀ from the
        persistent cell-centred union SDF instead of recovering it from the
        kernel's staggered-face value.  This matches the old cuda_kernels
        path and is more robust for large bodies at an angle (the
        staggered-face SDF can differ from the face-averaged cell-centred
        SDF near sharp edges).

        Memoised within a step.  ``mu_funcs`` costs ~4 ms per call at 14 M
        cells (its narrow band is boolean-mask indexing, so it pays a
        ``nonzero`` + gather + scatter over the full grid), and the callers
        ask for the SAME tensor 4-5 times per step:
        ``_rescale_kernel_coeffs_two_phase`` alone evaluates it once for
        ``_alpha_fluid_cc`` and again inside its per-direction loop, which is
        3 more identical full-grid evaluations.  The result depends only on
        ``composite_body.sdf_val``, which is fixed for the whole step once
        ``composite_body.update`` has run.

        The cache is guarded twice over, because the streaming path writes
        ``sdf_val`` from a native kernel: ``_invalidate_mu0_cc`` drops it at
        the top of every step, and the stored ``(data_ptr, _version)`` stamp
        drops it again if the buffer is reallocated or mutated in place
        mid-step.  A stale μ₀ would silently mis-scale the Poisson
        coefficients, so neither guard is relied on alone.
        """
        sdf = self.composite_body.sdf_val
        stamp = (sdf.data_ptr(), sdf._version, tuple(sdf.shape))
        cached = self._mu0_cc_cache
        if cached is not None and cached[0] == stamp:
            return cached[1]
        mu0 = self.composite_body.mu_funcs(sdf)[0]
        self._mu0_cc_cache = (stamp, mu0)
        return mu0

    def _invalidate_mu0_cc(self):
        """Drop the per-step ``_mu0_cc`` memo (see that method)."""
        self._mu0_cc_cache = None

    def _kernel_blend_velocities(self, vels):
        """Apply the air-transparent identity ``u0 := a*u0 + (1-a)*u'`` to the
        kernel-path BDIM output (in place), using the u' stashed by the
        ``adv_diff_solver.solve`` wrapper.  With ``alpha_exclude_body`` the
        blend weight is zeroed inside the body (cell-centred union SDF
        averaged to faces — kernel mode keeps no staggered SDFs), keeping the
        carved dry interior rigidly attached.  Re-applies the BCs afterwards
        so ghost cells match the python path."""
        # NOTE: do NOT clear ``_kernel_primes`` here.  ``solve`` runs INSIDE the
        # pre-Poisson CUDA graph, so on a graph REPLAY the ``_solve_and_stash``
        # python assignment never executes.  Clearing the stash each step would
        # therefore leave ``primes is None`` on every replay and silently skip
        # this blend — the kernel would keep imposing the body velocity into the
        # air (exactly the historical kernel-mode blow-up).  The stashed tensors
        # are ``adv_diff_solver``'s PERSISTENT output buffers (``_sl_out`` /
        # ``_conv_out``), which the graph rewrites in place on every replay, so
        # holding the reference costs no extra memory and always reads this
        # step's u'.
        if not self._air_transparent_body:
            return
        primes = getattr(self, "_kernel_primes", None)
        if primes is None:
            # Reaching here means some momentum path ran without publishing u'.
            # Returning quietly is what let consistent_momentum ship a broken
            # force handoff: the fluid stayed healthy and only the body loads
            # were wrong, so nothing in the fluid probe set could see it.
            raise RuntimeError(
                "air_transparent_body is on but no advection output u' was "
                "published, so the velocity blend a*S + (1-a)*u' cannot run "
                "and the BDIM kernel is imposing the body velocity into the "
                "air phase.  Whatever _advect_momentum ran this step must set "
                "self._kernel_primes to its output buffers (the stock path "
                "does it in the adv_diff_solver.solve wrapper; consistent "
                "momentum does it in TwoPhaseSolver._advect_momentum)."
            )
        comp = self.composite_body
        a_f = self._alpha_fluid_cc()   # fluid-part fraction; see _alpha_fluid_cc
        for d, (vel, prime) in enumerate(zip(vels, primes)):
            w = 1.0 - self._face_mean(a_f, d)      # lerp weight toward u' (air)
            if self._alpha_exclude_body:
                w = w * (self._face_mean(comp.sdf_val, d) >= 0).to(w.dtype)
            vel.lerp_(prime, w)
        self.adv_diff_solver.set_BCs(*vels)

    def _rescale_kernel_coeffs_two_phase(self, coeffs):
        """Return WaterLily-scaled coefficients from the kernel buffers.

        The projection solves for the pressure impulse
        ``q = dt*p/rho_water`` with coefficient
        ``mu0*rho_water/rho_face``.  This is algebraically identical to solving
        for physical pressure with ``dt*mu0/rho_face``, but keeps liquid
        coefficients O(1), so the Poisson tolerance and zero-diagonal test are
        not accidentally compared with a 1e-6-scaled operator.

        When ``bdim_mu0_projection`` is True the fluid fraction μ₀ is recovered
        directly from the kernel coefficient (zero extra cost).  When False the
        kernel wrote a constant ``dt/ρ`` coefficient (no μ₀ baked in) and μ₀
        is computed from the persistent cell-centred union SDF instead, making
        the rescale independent of how the kernel wrote the coefficient.
        """
        tp = self.two_phase
        dt = float(self.dt)
        rs = self._rho_solid
        normalized = (tp.rho_water != tp.rho_air or rs is not None)
        # Fluid-part water fraction (identity when the carve is off) — the
        # density blend and the transparency mask must not count the body
        # volume as air; see _alpha_fluid_cc.
        a_f = self._alpha_fluid_cc()
        q = 1.0 / (a_f * tp.rho_water + (1.0 - a_f) * tp.rho_air)
        comp = self.composite_body
        out = []
        for d, c_kernel in enumerate(coeffs):
            if c_kernel is None:                   # cw in 2-D
                out.append(None)
                continue
            if self._face_density_fluid_weighted:
                inv_rho = self._kernel_face_mean_fluid_weighted(
                    q, self._mu0_cc(), d)
            else:
                inv_rho = self._kernel_face_mean(q, d)
            # Recover fluid fraction μ₀: prefer the kernel coefficient
            # (fast, exact match to what the BDIM kernel used) when
            # available; otherwise compute from the persistent cell-centred
            # union SDF (robust for large angled bodies like the ramp).
            if self.bdim_mu0_projection:
                mu0 = c_kernel * (tp.rho_water / dt)
            else:
                mu0 = self._kernel_face_mean(self._mu0_cc(), d)
            if self._air_transparent_body:
                a = self._kernel_face_mean(a_f, d)
                mu0_t = a * mu0 + (1.0 - a)
                if self._alpha_exclude_body:
                    sdf_face = self._kernel_face_mean(comp.sdf_val, d)
                    mu0_t = torch.where(sdf_face >= 0.0, mu0_t, mu0)
                mu0 = mu0_t
            if rs is not None:
                out.append(tp.rho_water /
                           (mu0 / inv_rho + (1.0 - mu0) * rs))
            elif not normalized:
                # Preserve exact single-phase reduction (including its
                # tolerance scaling) when both phases have one density.
                out.append(dt * mu0 * inv_rho)
            else:
                out.append(tp.rho_water * mu0 * inv_rho)
        # ── The degenerate μ₀ → 0 interior is not floored here ───────
        # c = dt·μ₀/ρ → 0 inside a body makes div(c·grad(p)) singular there.
        # That is the Poisson solver's degenerate-cell mask's job
        # (``poisson_jcap_tol``, |J| < tol → row/column dropped, residual
        # masked), the same guard the un-floored single-phase FluidSolver
        # relies on.
        #
        # A coefficient floor CANNOT also do that job, because the open-water
        # coefficient *is* dt/ρ_water: since c = dt·μ₀/ρ_water ≤ dt/ρ_water for
        # every μ₀ ≤ 1, any floor at that scale saturates the whole SUBMERGED
        # BDIM band, not just the deep interior — the projection then sees no
        # wetted geometry at all and drives flux straight through the body.
        # The historical ``min(1e-6, dt/ρ_water)`` default did exactly that
        # (measured on the pool-ramp case: 99.9 % of the submerged band and
        # 100 % of the submerged interior clamped to the open-water value),
        # and it also pre-empted ``poisson_jcap_tol`` entirely — with it on,
        # not one cell was ever frozen.
        #
        return tuple(out)

    def project(self, *args, ch=None, cv=None, cw=None, ch_cc=None, **kwargs):
        """Fused path only: blend the BDIM velocity with u' (air-transparent
        identity) and rescale the coefficients to the two-phase formulas, then
        run the base variable-coefficient projection.  The python path arrives
        here with freshly-built two-phase coefficients (not the solver's
        persistent water-normalised buffers), so the identity check on
        ``_ch_persist`` keeps it untouched."""
        fused = (isinstance(ch, torch.Tensor)
                 and ch is getattr(self, '_ch_persist', None))
        scaled = fused and (self.two_phase.rho_water != self.two_phase.rho_air
                            or self._rho_solid is not None)
        if fused:
            vels = list(args[:2])
            if self.ndim == 3:
                vels.append(kwargs["w_vel"])
            self._kernel_blend_velocities(vels)
            ch, cv, cw = self._rescale_kernel_coeffs_two_phase((ch, cv, cw))
            if scaled:
                # Base project warm-starts from args[2] and returns physical p
                # by convention. Temporarily convert that persistent field to
                # q, then convert it back below for forces and output.
                args = list(args)
                args[2].mul_(float(self.dt) / self.two_phase.rho_water)
                args = tuple(args)

        out = super().project(*args, ch=ch, cv=cv, cw=cw,
                              ch_cc=ch_cc, **kwargs)
        if scaled:
            out = (*out[:-1], out[-1].mul_(self.two_phase.rho_water /
                                           float(self.dt)))
        return out

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

    def advance_and_compute_loads(self, u, v, p, iteration, t, w_vel=None):
        """Override: when zero_pressure_inside is enabled, zero pressure only
        DEEP inside the body (``sdf < -2h``), leaving the BDIM band
        (``|sdf| < 2h``) intact.  The base solver zeros at ``sdf < 0``, which
        for thin bodies (thickness ~ band width) removes the interior half of
        the band and destroys emergent buoyancy."""
        # New step, new body pose: the μ₀ memo from the previous step is stale.
        self._invalidate_mu0_cc()
        self.composite_body.update(t, iteration, dt=self.dt)
        if self._needs_python_mu_normals():
            self._recompute_mu_normals()

        if self.use_gravity:
            if self.ndim == 2:
                u, v = self._apply_gravity_body_force(u, v)
            else:
                u, v, w_vel = self._apply_gravity_body_force(u, v, w_vel)

        if self.ndim == 2:
            (u, v, p) = self.fluid_step(u, v, p, self.dt)
        else:
            (u, v, w_vel, p) = self.fluid_step(u, v, w_vel, p, self.dt)

        # corrected zero_pressure_inside: zero only the DEEP interior (sdf<-2h),
        # leaving the BDIM band intact (see the method docstring).
        if self.zero_pressure_inside:
            deep = self.composite_body.sdf_val < -2.0 * self.h
            p = torch.where(deep, 0.0, p)

        lagrangian = self.force_method == "lagrangian"
        if self.ndim == 2:
            self.u0, self.v0, self.p0 = u, v, p
            if self.compute_forces:
                (self.forces_lagrangian_2d if lagrangian
                 else self.forces_method2)(u, v, p, iteration)
        else:
            self.u0, self.v0, self.w0, self.p0 = u, v, w_vel, p
            if self.compute_forces:
                (self.forces_lagrangian_3d if lagrangian
                 else self.forces_method2_3d)(u, v, w_vel, p, iteration)

        self._maybe_dump_band(p, iteration)
        return u, v, p, w_vel

    def _maybe_dump_band(self, p, iteration):
        """Opt-in one-shot dump of everything the eulerian buoyancy is built
        from, so the band quadrature can be recomputed offline and checked
        against what the kernel actually returned.

        Set ``LILYTORCH_BAND_DUMP=<iteration>`` (and optionally
        ``LILYTORCH_BAND_DUMP_PATH``).  Off by default and untouched otherwise.

        The point of the dump is to split the two candidate causes of the
        vertical-force deficit apart.  The eulerian pressure force is
        ``F_i = sum (-p) d_i H_eps(phi) h^3``; recomputing that sum offline from
        the same p and phi and comparing against the kernel's own answer says
        whether the QUADRATURE is wrong (they disagree) or the PRESSURE FIELD is
        (they agree but neither equals rho*g*V).
        """
        import os
        want = os.environ.get("LILYTORCH_BAND_DUMP")
        if want is None or int(want) != int(iteration):
            return
        import numpy as np
        comp = self.composite_body
        path = os.environ.get("LILYTORCH_BAND_DUMP_PATH", "band_dump.npz")
        out = {
            "p": p.detach().cpu().numpy(),
            "sdf": comp.sdf_val.detach().cpu().numpy(),
            "alpha": self.two_phase.alpha.detach().cpu().numpy(),
            "eps": float(comp.eps),
            "h": float(self.h),
            "iteration": int(iteration),
            "x": np.asarray(comp.x.detach().cpu()),
            "y": np.asarray(comp.y.detach().cpu()),
            "z": np.asarray(comp.z.detach().cpu()),
            "rho_water": float(self.two_phase.rho_water),
            "rho_air": float(self.two_phase.rho_air),
            "gravity": np.asarray(
                [] if self._gravity is None else
                (self._gravity.detach().cpu() if hasattr(self._gravity, "detach")
                 else self._gravity), dtype=float),
            "force_submethod": int(getattr(self, "force_submethod", 0)),
            "eul_off_pressure": float(self.eul_sample_offset_pressure),
        }
        for name in ("pressure_force_x", "pressure_force_y", "pressure_force_z",
                     "friction_force_lin_x", "friction_force_lin_y",
                     "friction_force_lin_z"):
            v = getattr(self, name, None)
            if v is not None:
                out[name] = np.asarray(
                    v.detach().cpu() if hasattr(v, "detach") else v)
        np.savez_compressed(path, **out)
        print(f"[band-dump] iteration {iteration} -> {path} "
              f"({out['sdf'].shape}, eps={out['eps']:.5g}, h={out['h']:.5g})")

    def _apply_gravity_body_force(self, *vels):
        """Suppressed under consistent momentum transport: gravity is applied
        there as a ``rho*g`` body force on the momentum itself (so the mass
        flux rides on the ~divergence-free velocity).  Adding ``dt*g`` here
        too would double-count it."""
        if self._consistent_momentum:
            return vels
        return super()._apply_gravity_body_force(*vels)

    def _build_hydrostatic_reference(self):
        """Two-phase density is variable (water/air VOF blend), so the
        single-phase analytic ``p_h = rho*(g.x)`` does NOT satisfy
        ``∇p_h = rho_face*g`` across the interface.  Returning ``None``
        keeps the base ``_wb_gravity`` flag OFF -> the legacy uniform
        ``dt*g`` body force is used, byte-identical to before.  The
        variable-density hydrostatic split is Stage 2."""
        return None

    def fluid_step(self, *args):
        """As the base (advect-BDIM-project); the only two-phase addition is
        the deferred body-aware alpha carve, which must run after the coupled
        body poses arrive but before this step's transport."""
        self._try_deferred_alpha_carve()
        return super().fluid_step(*args)

    # ------------------------------------------------------------------
    #  Override: VOF transport in the once-per-step tail
    # ------------------------------------------------------------------
    def _umax_probe(self, u, v, w_vel, iteration):
        """Diagnostic: find the max-|u| cell each step and report WHAT is there
        (inside body / body-band / fluid; water / air / interface; nearest body;
        wall distance).  Opt-in via ``LILYTORCH_UMAX_PROBE=1`` (and optional
        ``LILYTORCH_UMAX_THR``); normal runs are untouched.  Run before
        ``check_explosion`` so the blow-up step itself is captured."""
        import os
        comp = self.composite_body
        D = self.ndim
        comps = [u, v] + ([w_vel] if (D == 3 and w_vel is not None) else [])
        speed = comps[0].abs().clone()
        for c in comps[1:]:
            speed = torch.maximum(speed, c.abs())
        nonfin = not bool(torch.isfinite(speed).all())
        speed = torch.nan_to_num(speed, nan=1e30, posinf=1e30, neginf=1e30)
        umax = float(speed.max())
        thr = float(os.environ.get("LILYTORCH_UMAX_THR", "2.0"))
        if umax < thr and iteration % 25 != 0:
            return
        idx = int(torch.argmax(speed)); sh = tuple(speed.shape)
        if D == 3:
            i = idx // (sh[1] * sh[2]); rem = idx % (sh[1] * sh[2])
            j = rem // sh[2]; k = rem % sh[2]; ijk = (i, j, k)
            z = float(comp.z[k])
        else:
            i = idx // sh[1]; j = idx % sh[1]; ijk = (i, j); z = 0.0
        x = float(comp.x[i]); y = float(comp.y[j]); h = float(self.h)
        sdf = (float(comp.sdf_val[ijk]) if getattr(comp, 'sdf_val', None) is not None
               else float('nan'))
        a = float(self.two_phase.alpha[ijk])
        loc = ("INSIDE-body" if sdf < 0 else
               ("body-BAND" if abs(sdf) < 2 * h else "fluid"))
        phase = "water" if a > 0.9 else ("air" if a < 0.1 else "INTERFACE")
        nb = ""
        sv = getattr(comp, 'sdf_vals', None)
        if sv is not None:
            try:
                per = [abs(float(sv[b][ijk])) for b in range(sv.shape[0])]
                bi = min(range(len(per)), key=lambda b: per[b])
                name = getattr(comp.bodies[bi], 'name', None) or f"body{bi}"
                nb = f" nearest={name}"
            except Exception:
                pass
        dwall = min(i, sh[0] - 1 - i, j, sh[1] - 1 - j)
        if D == 3:
            dwall = min(dwall, k, sh[2] - 1 - k)
        print(f"[umax] it={iteration:4d} |u|max={umax:9.2f} @({x:+.3f},{y:+.3f},{z:+.3f}) "
              f"ijk={ijk} {loc}/{phase} sdf={sdf:+.3f} a={a:.2f} wall={dwall}c{nb}"
              f"{' NONFINITE!' if nonfin else ''}", flush=True)

    def _try_deferred_alpha_carve(self, iteration=None):
        """One-shot deferred body-aware carve (FARMS-coupled paths).

        The MuJoCo spawn poses are unknown when ``alpha_init`` is evaluated at
        construction, so the carve waits until the composite ``sdf_val``
        actually CONTAINS the body and retries until then: the python path
        stamps it in the handler update before the first ``fluid_step``; the
        kernel/streaming path only materialises it in the post-step force
        kernel, so the carve fires at the first ``finalize_step`` instead
        (one step with the wet interior — negligible at these timesteps).
        Gives up with a warning if the SDF never arrives.
        """
        if not self._alpha_carve_pending:
            return
        sdf_cc = self.composite_body.sdf_val
        if float(sdf_cc.min()) > 100.0:          # still the _FAR placeholder
            if iteration is not None and iteration >= 3:
                self._alpha_carve_pending = False
                print("body_aware_alpha_init: WARNING -- composite sdf_val "
                      "never received the body within the first steps; the "
                      "alpha_exclude_body carve was SKIPPED on this path.",
                      flush=True)
            return
        self._alpha_carve_pending = False
        self.two_phase.reinit_alpha(body_aware_alpha_init(
            self._alpha_carve_raw, sdf_cc, self.eps, self.h,
            compensate=self._alpha_carve_compensate,
        ))
        del self._alpha_carve_raw
        wet_inside = int(((self.two_phase.alpha > 0.5)
                          & (sdf_cc < -self.eps)).sum())
        interior = int((sdf_cc < -self.eps).sum())
        print(f"body_aware_alpha_init: deferred carve applied "
              f"(body interior cells: {interior}, sdf min "
              f"{float(sdf_cc.min()):+.3f}); wet cells left inside the "
              f"body: {wet_inside}", flush=True)

    def finalize_step(self, u, v, p, iteration, w_vel=None):
        """Once-per-step tail: stability check, VOF transport, BDIM-field
        release, optional allocator flush, and plotting/saving."""
        import os
        if os.environ.get("LILYTORCH_UMAX_PROBE", "0") == "1":
            self._umax_probe(u, v, w_vel, iteration)
        # Throttle the blow-up check to check_explosion_every (default 50), like
        # the base solver: check_explosion reads GPU reductions to the host
        # (.cpu()) to branch in Python, which forces a GPU->CPU sync that stalls
        # the async CUDA pipeline. Per-step it was a needless sync every step.
        if iteration % self.check_explosion_every == 0:
            self.check_explosion(iteration)
        # Deferred body-aware carve: on the kernel/streaming path the composite
        # sdf_val is first materialised by the post-step force kernel, so the
        # carve can only fire here (before this step's VOF transport).
        self._try_deferred_alpha_carve(iteration)
        self._advect_vof(u, v, p, iteration, w_vel=w_vel)
        # Flow diagnostics on the post-projection field, as in the base tail.
        # This override re-implements the base finalize_step and used to drop
        # the block, so diagnostics.h5 came out all-NaN on every two-phase run.
        if self.diagnostics is not None:
            cb = self.composite_body
            self.diagnostics.update(
                iteration, u, v, p, self.dt, self.nu,
                self.divergence, self.vorticity, w=w_vel,
                sdf_cc=getattr(cb, "sdf_val", None),
                mu_fn=getattr(cb, "mu_funcs", None),
                poisson_solver=getattr(self, "poisson_solver", None),
            )
        self._release_bdim_fields()
        # Flush the CUDA allocator cache only at the base solver's throttled
        # cadence (empty_cache_every, default 200), NOT every step: the grids
        # are fixed-size and the only variable-shape churn (the moving AABB
        # force crops) is bounded, so the caching allocator reuses blocks
        # across steps. A per-step empty_cache() returned all cached blocks to
        # the driver and forced a sync every step -- pure overhead here, the
        # dominant cost vs the one-way path (which runs this same lagrangian/
        # AABB machinery at the 200-step cadence). Only nvidia-smi's reserved
        # number rises; true peak working set is unchanged.
        if self.device.type == "cuda" and iteration % self.empty_cache_every == 0:
            torch.cuda.empty_cache()
        return self.plotting_and_saving(u, v, p, iteration, w_vel=w_vel)

    # ------------------------------------------------------------------
    #  cvof graph capture (Phase 4.3): fold VOF transport into a CUDA graph
    # ------------------------------------------------------------------
    def _advect_vof(self, u, v, p, iteration, w_vel=None):
        """VOF transport (cvof sweeps), graph-captured on CUDA.

        No-op under consistent momentum transport: ``_advect_momentum``
        already moved the interface by evolving the density with the same
        mass flux that carried the momentum, and running a second, independent
        W&Y sweep here would transport alpha twice.

        Uses :class:`NativeWholeStepGraphRunner` to capture the per-direction
        W&Y conservative sweeps into a single graph replay.  Each sweep-parity
        variant gets its own graph; the parity is toggled OUTSIDE the graph
        (before key computation) because Python attribute writes do not execute
        during graph replay.
        On CPU, falls back to eager execution.
        """
        if self._consistent_momentum:
            return
        tp = self.two_phase
        ndim = self.ndim
        dt = float(self.dt)

        if not u.is_cuda:
            # CPU: eager path (parity toggled inside advect())
            if ndim == 2:
                tp.advect(u, v, dt=dt)
            else:
                tp.advect(u, v, w_vel, dt=dt)
            return

        # Toggle parity OUTSIDE the graph so each parity variant gets its
        # own captured graph.  advect_graph_aware reads _sweep_parity but
        # does NOT toggle it (toggle-on-replay would be silently dropped).
        tp._sweep_parity = not getattr(tp, '_sweep_parity', False)
        parity = int(tp._sweep_parity)

        # Lazy-init cvof graph runner
        if getattr(self, '_cvof_graph', None) is None:
            from lilytorch.src.graph_capture import NativeWholeStepGraphRunner
            self._cvof_graph = NativeWholeStepGraphRunner(
                use_cuda_graph=not self._graph_capture_debug)
        else:
            self._cvof_graph._use_cuda_graph = not self._graph_capture_debug
        runner = self._cvof_graph

        if ndim == 2:
            def _run_cvof():
                tp.advect_graph_aware(u, v, dt=dt)
            key = (tp.alpha.data_ptr(), u.data_ptr(), v.data_ptr(),
                   parity, dt)
        else:
            def _run_cvof():
                tp.advect_graph_aware(u, v, w_vel, dt=dt)
            key = (tp.alpha.data_ptr(), u.data_ptr(), v.data_ptr(),
                   w_vel.data_ptr(), parity, dt)
        device_str = f"cuda:{u.device.index}"
        runner.run(key, device_str, _run_cvof, stage=None)


def build_two_phase_solver(config_path, dtype=torch.float32):
    """Build a :class:`TwoPhaseSolver` from a YAML config path (parallels the
    per-example ``build_solver`` factories, which keep returning a plain
    :class:`FluidSolver`)."""
    from lilytorch.util.yaml_operations import yaml2pyobject
    pars = yaml2pyobject(config_path)
    return TwoPhaseSolver(pars, dtype=dtype)
