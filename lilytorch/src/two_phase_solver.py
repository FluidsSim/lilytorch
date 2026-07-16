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
* ``_compute_bdim_coefficients`` / ``_apply_bdim_all_axes`` — **python
  path**: projection coefficients
  ``c = dt·μ0_eff·(1/ρ)_fluid`` from the VOF water/air density (Weymouth &
  Yue 2011, ``(1−δ^B)/ρ``; the body enters via ``μ0`` only, never its
  density) and the air-transparent mu0/mu1 masking of the velocity BDIM.
* ``project`` — **fused path** (the default): between the fused single-phase
  bdim_apply and the Poisson solve, repair the velocity with the exact
  air-transparent identity ``a·S + (1−a)·u′`` (u′ captured by wrapping
  ``adv_diff_solver.solve``) and rescale the kernel's water-normalised
  coefficients to the two-phase formulas. No CUDA or solver.py changes;
  both paths solve the identical Poisson.
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
from lilytorch.src.two_phase import TwoPhase


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
        # REMOVED experimental toggles (cuda_native_port Phase 4.4).  Reject
        # loudly when a config still enables them rather than silently running
        # a different scheme.  ``consistent_momentum`` (Nangia 2019 conservative
        # rho*u transport) and ``mu0_free_coeff`` were toy-boat waterline
        # experiments; the adopted stabilisers are ``rho_solid``,
        # ``air_transparent_body`` and ``alpha_exclude_body``.
        for _removed in ("consistent_momentum", "consistent_n_cycles",
                         "mu0_free_coeff"):
            if cfg.get(_removed):
                raise ValueError(
                    f"solver.two_phase.{_removed} was removed (Phase 4.4 "
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
        # Partial-Heaviside pressure force (the ``n·δ -> ∂_iH`` weight change).
        # The eulerian band integral ``F_i = -Σ p n_i δ_ε`` forms the analytic
        # normal n and the delta kernel separately, so the discrete sum does NOT
        # satisfy summation-by-parts and a hydrostatic baseline leaks in (∝depth,
        # see gauge_anchor notes).  Replacing the weight by the DISCRETE gradient
        # of the smooth body Heaviside makes it satisfy discrete SBP
        # ``Σ p ∂_iH = -Σ (∂_i p) H``, so a hydrostatic field (∂_x p = 0) gives
        # Fx -> 0 exactly while buoyancy (∂_z p = ρg) survives.
        #
        # The force density ``f_i = -p ∂_iH_ε(φ_union)`` is taken over the UNION
        # SDF (one closed surface, no interior inter-link seams that would break
        # the SBP cancellation) and split to links by a softmin partition of
        # unity over the per-link SDFs (Σ_b w_b ≡ 1, so Σ_b F[b] == union force,
        # and each link still gets its own force+torque for MuJoCo).
        # ``partial_heaviside_blend_cells`` sets the softmin scale tau =
        # blend_cells * h (default 1.5 cells).  Viscous channels untouched.
        # Needs per-body SDFs (standalone python-style bodies); the streaming
        # equivalent is ``solver.force_submethod = "deltaH"``.  Opt-in;
        # supersedes the gauge anchors when set.
        self._partial_heaviside_forces = bool(
            cfg.get("partial_heaviside_forces", False))
        self._ph_blend_cells = float(cfg.get("partial_heaviside_blend_cells", 1.5))
        flags = []
        if self._rho_solid is not None:
            flags.append(f"three-phase rho_solid={self._rho_solid}")
        if self._air_transparent_body:
            flags.append("air-transparent-body")
        if self._partial_heaviside_forces:
            flags.append(
                "partial-Heaviside (∂H) pressure forces [union+partition]")
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
        # Fluid-part water fraction: with the alpha_exclude_body carve the raw
        # alpha under-counts water in the wetted BDIM band (body volume is
        # missing from it) -> light shell -> spurious buoyant plumes.  Both the
        # density blend and the transparency mask must see the fluid-part
        # fraction.  Identity when the carve is off (see _alpha_fluid_cc).
        a_f = self._alpha_fluid_cc()
        recip_cc = 1.0 / (a_f * tp.rho_water + (1.0 - a_f) * tp.rho_air)
        out = []
        for d, ax in enumerate(self._bdim_axis_names):       # 'u','v'[,'w']
            recip_face = self._face_mean(recip_cc, d)        # (1/ρ) water/air blend
            mu0 = getattr(self, f'mu0_all_{ax}')             # (1−δ^B), fluid fraction
            if atb:
                # Air-transparent body: mu0_eff = 1 - alpha_face*(1 - mu0)
                # = alpha_face*mu0 + (1-alpha_face). Body invisible in air.
                a_face = self._face_mean(a_f, d)
                mu0_t = a_face * mu0 + (1.0 - a_face)
                if self._alpha_exclude_body:
                    # Carved init: the body interior is alpha=0 (dry), so the
                    # alpha proxy can no longer distinguish "above the free
                    # surface" from "inside the hull" — unrestricted it would
                    # make the body transparent in its own interior and let
                    # water flow through the wetted hull. Gate on the face SDF:
                    # transparency only OUTSIDE the body, plain mu0 inside.
                    sdf_face = getattr(self.composite_body, f"sdf_val_{ax}")
                    mu0_t = torch.where(sdf_face >= 0.0, mu0_t, mu0)
                mu0 = mu0_t
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

    def _alpha_face(self, d):
        """Water fraction ``alpha`` on the staggered *d*-face grid (full grid;
        see :meth:`_face_mean`)."""
        return self._face_mean(self.two_phase.alpha, d)

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
        return (a / mu0.clamp(min=1e-3)).clamp(0.0, 1.0)

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
            # Mask mu0, mu1 by the fluid-part water fraction on this face grid
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
    #    enforced at init), after which the python-path formulas of
    #    ``_compute_bdim_coefficients`` (standard / rho_solid three-phase,
    #    with the air-transparent masking and the carved-init gate)
    #    are evaluated directly on the face grids.  Out-of-place: the
    #    persistent ``_ch_persist`` buffers must stay water-normalised for
    #    the next step's Kernel-B overwrite.
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
        SDF near sharp edges)."""
        return self.composite_body.mu_funcs(self.composite_body.sdf_val)[0]

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
        primes = getattr(self, "_kernel_primes", None)
        if primes is None or not self._air_transparent_body:
            return
        comp = self.composite_body
        a_f = self._alpha_fluid_cc()   # fluid-part fraction; see _alpha_fluid_cc
        for d, (vel, prime) in enumerate(zip(vels, primes)):
            w = 1.0 - self._face_mean(a_f, d)      # lerp weight toward u' (air)
            if self._alpha_exclude_body:
                w = w * (self._face_mean(comp.sdf_val, d) >= 0).to(w.dtype)
            vel.lerp_(prime, w)
        self.adv_diff_solver.set_BCs(*vels)

    def _rescale_kernel_coeffs_two_phase(self, coeffs):
        """Return fresh two-phase Poisson coefficients from the kernel's
        water-normalised ``c_kernel = dt*mu0/rho_water`` ones, evaluating the
        same per-mode formulas as ``_compute_bdim_coefficients``.

        When ``bdim_mu0_projection`` is True the fluid fraction μ₀ is recovered
        directly from the kernel coefficient (zero extra cost).  When False the
        kernel wrote a constant ``dt/ρ`` coefficient (no μ₀ baked in) and μ₀
        is computed from the persistent cell-centred union SDF instead, making
        the rescale independent of how the kernel wrote the coefficient.
        """
        tp = self.two_phase
        dt = float(self.dt)
        rs = self._rho_solid
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
                out.append(dt / (mu0 / inv_rho + (1.0 - mu0) * rs))
            else:
                out.append(dt * mu0 * inv_rho)
        # ── Prevent singular operator inside large bodies ────────────
        # μ₀ → 0 inside the body → c = dt·μ₀/ρ → 0 → div(0·grad(p)) is
        # singular.  Multigrid stalls on large connected zero-coefficient
        # regions.  Clamp to a tiny floor (1e-4 × dt/ρ_water) so the
        # operator is stiff but never fully degenerate.  BDIM already
        # overrides velocities inside the body, so the effect on physics
        # is zero.
        import os
        _floor_str = os.environ.get("LILYTORCH_COEFF_FLOOR")
        if _floor_str is not None:
            _floor = float(_floor_str)
        else:
            _floor = 1e-6  # = dt/ρ_water — prevents degenerate op inside bodies
        if _floor > 0.0:
            out = tuple(torch.clamp(c, min=_floor) if isinstance(c, torch.Tensor) else c
                        for c in out)
        return tuple(out)

    def project(self, *args, ch=None, cv=None, cw=None, ch_cc=None, **kwargs):
        """Fused path only: blend the BDIM velocity with u' (air-transparent
        identity) and rescale the coefficients to the two-phase formulas, then
        run the base variable-coefficient projection.  The python path arrives
        here with freshly-built two-phase coefficients (not the solver's
        persistent water-normalised buffers), so the identity check on
        ``_ch_persist`` keeps it untouched."""
        if isinstance(ch, torch.Tensor) and ch is getattr(self, '_ch_persist', None):
            vels = list(args[:2])
            if self.ndim == 3:
                vels.append(kwargs["w_vel"])
            self._kernel_blend_velocities(vels)
            ch, cv, cw = self._rescale_kernel_coeffs_two_phase((ch, cv, cw))

        return super().project(*args, ch=ch, cv=cv, cw=cw, ch_cc=ch_cc, **kwargs)

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
        if self._partial_heaviside_forces:
            # The base routine already computed the (p-independent) viscous loads;
            # REPLACE its pressure force/torque with the union-∂H partition integral.
            self._apply_partition_heaviside(fn3d, p)

    @staticmethod
    def _heaviside_smooth(phi, eps):
        """Smooth Heaviside = exact antiderivative of the cosine delta the base
        force uses (``δ = (1+cos(πφ/ε))/(2ε)``): 0 for φ≤-ε, 1 for φ≥ε, else
        ``½(1 + φ/ε + sin(πφ/ε)/π)``.  H(φ>0)=1 (fluid side), so ∂H points
        outward -> matches the n·δ sign convention (pforce = -p n)."""
        x = (phi / eps).clamp(-1.0, 1.0)
        return 0.5 * (1.0 + x + torch.sin(torch.pi * x) / torch.pi)

    def _apply_partition_heaviside(self, fn3d, p):
        """THE FIX: union-∂H force density distributed to links by a softmin
        partition of unity on the per-link SDFs.

        Force density ``f_i = -p ∂_iH_ε(φ_union)`` is computed from the UNION SDF
        (one closed surface, NO interior inter-link seams -> SBP cancels the
        hydrostatic baseline, as the union diagnostic proved).  It is split to
        links by weights ``w_b = softmax(-φ_b / τ)`` (τ = blend_cells·h): at a
        union-surface cell the nearest link (smallest φ_b) owns the patch; at a
        seam two abutting links share it smoothly.  Σ_b w_b ≡ 1, so
        ``Σ_b F[b] == union force`` exactly (validated in tests/harness).  Each
        link's torque is about its OWN com.  Cropped to the union AABB for speed;
        per-link SDFs from _sdf_sparse / sdf_vals (python path only)."""
        cb = self.composite_body
        if getattr(cb, "_kernel_step", None) is not None:
            raise RuntimeError(
                "partial_heaviside_forces requires per-body SDFs (standalone "
                "python-style bodies; the rigid streaming update keeps only "
                "the union SDF); use solver.force_submethod='deltaH' on the "
                "streaming path.")
        B = len(cb.bodies)
        h = self.h
        hD = self.h3 if fn3d else self.h2
        eps0 = cb.bodies[0].eps
        tau = max(self._ph_blend_cells * h, 1e-9)
        _d = torch.float64
        _FAR = 1e6
        ndim = 3 if fn3d else 2

        # union force density f_i = -p ∂_iH(φ_union) on the full grid
        sdf_u = cb.sdf_val
        H = self._heaviside_smooth(sdf_u, eps0)
        gH = [torch.gradient(H, spacing=h, dim=d, edge_order=2)[0]
              for d in range(ndim)]
        fdens = [-p * g for g in gH]

        # union AABB over all per-link sub-blocks (+halo); crop everything to it
        use_sparse = (hasattr(cb, "_sdf_sparse") and cb._sdf_sparse
                      and cb._sdf_sparse[0] is not None)
        lo = [0] * ndim
        hi = list(sdf_u.shape)
        aabbs = None
        if use_sparse and all(a is not None for a, _ in cb._sdf_sparse):
            aabbs = [a for a, _ in cb._sdf_sparse]
            lo = [min(a[2 * d] for a in aabbs) for d in range(ndim)]
            hi = [max(a[2 * d + 1] for a in aabbs) for d in range(ndim)]
            halo = 2
            lo = [max(0, lo[d] - halo) for d in range(ndim)]
            hi = [min(sdf_u.shape[d], hi[d] + halo) for d in range(ndim)]
        sl = tuple(slice(lo[d], hi[d]) for d in range(ndim))
        cshape = tuple(hi[d] - lo[d] for d in range(ndim))

        # per-link SDF stack on the crop (_FAR outside each link's AABB)
        sdf_stack = torch.full((B, *cshape), _FAR, device=p.device, dtype=p.dtype)
        for b in range(B):
            if aabbs is not None:
                a = aabbs[b]
                sub = cb._sdf_sparse[b][1]
                lsl = tuple(slice(a[2 * d] - lo[d], a[2 * d + 1] - lo[d])
                            for d in range(ndim))
                sdf_stack[b][lsl] = sub
            else:
                sdf_b = (cb.sdf_vals[b] if getattr(cb, "sdf_vals", None) is not None
                         else cb.bodies[b].sdf_val)
                sdf_stack[b] = sdf_b[sl]
        # softmin partition of unity over links (sums to 1 over b)
        w = torch.softmax(-sdf_stack / tau, dim=0)

        fdc = [f[sl] for f in fdens]
        coords = [cb.X[sl], cb.Y[sl]] + ([cb.Z_grid[sl]] if fn3d else [])
        zero = p.new_zeros(())

        def load(t):                       # density field -> reduced scalar load
            return t.to(_d).sum().to(p.dtype) * hD

        for b in range(B):
            wb = w[b]
            com = cb.bodies[b].com_pos
            # weighted force density / lever arm per axis; pad z with 0 in 2-D so
            # the cross product is dimension-agnostic (Tx, Ty then vanish exactly)
            f = [wb * fdc[d] for d in range(ndim)] + [zero] * (3 - ndim)
            r = [coords[d] - com[d] for d in range(ndim)] + [zero] * (3 - ndim)
            self.pressure_force_x[b] = load(f[0])
            self.pressure_force_y[b] = load(f[1])
            self.pressure_force_z[b] = load(f[2])
            # torque density r x f about the link's own com
            self.pressure_force_ang_x[b] = load(r[1] * f[2] - r[2] * f[1])
            self.pressure_force_ang_y[b] = load(r[2] * f[0] - r[0] * f[2])
            self.pressure_force_ang_z[b] = load(r[0] * f[1] - r[1] * f[0])

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

        return u, v, p, w_vel

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

        Uses :class:`NativeWholeStepGraphRunner` to capture the per-direction
        W&Y conservative sweeps into a single graph replay.  Each sweep-parity
        variant gets its own graph; the parity is toggled OUTSIDE the graph
        (before key computation) because Python attribute writes do not execute
        during graph replay.
        On CPU, falls back to eager execution.
        """
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
