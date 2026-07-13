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
* ``_compute_bdim_coefficients`` / ``_apply_bdim_all_axes`` — **consistent-
  momentum (python-style) path**: projection coefficients
  ``c = dt·μ0_eff·(1/ρ)_fluid`` from the VOF water/air density (Weymouth &
  Yue 2011, ``(1−δ^B)/ρ``; the body enters via ``μ0`` only, never its
  density) and the air-transparent mu0/mu1 masking of the velocity BDIM.
* ``project`` — **fused path** (the default): between the fused single-phase
  bdim_forcing and the Poisson solve, repair the velocity with the exact
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
  coefficient override or written by the two-phase bdim_forcing);
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
        ``alpha *= mu0`` with ``mu0`` the BDIM smoothed Heaviside of the
        cell-centred union SDF — the same band kernel as the projection, so no
        sharp 0/1 cut is introduced at the wetted surface.

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
    d = (sdf_cc / eps).clamp(-1.0, 1.0)
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
        if self.poisson_method == "multigrid":
            print(
                "TwoPhaseSolver: poisson_method='multigrid' is functional but "
                "'mgcg' (multigrid-preconditioned CG) is ~2.2× faster for "
                "variable-density (833:1) two-phase Poisson. "
                "Consider setting poisson_method='mgcg'."
            )
        # FUSED PATH: the default (non-consistent) two-phase step reuses the
        # single-phase fused kernels untouched.  The two-phase repairs run in
        # ``project`` (see the fused-path override section below): the BDIM
        # velocity is fixed by the exact identity ``a*S + (1-a)*u'`` (the
        # historical blocker — the kernel imposing the body velocity into the
        # air — needed only u', captured by wrapping ``adv_diff_solver.solve``),
        # and the Poisson coefficient by the out-of-place rescale of the
        # kernel's water-normalised ``dt*mu0/rho_water``.  The consistent-
        # momentum step (python-style conservative transport) bypasses the
        # fused kernels entirely, so the guards below do not apply to it.
        _consistent = bool(tp_cfg.get("consistent_momentum", False))
        if not _consistent and not self.bdim_mu0_projection:
            raise ValueError(
                "TwoPhaseSolver's fused path requires bdim_mu0_projection=True "
                "(the two-phase coefficient rescale reconstructs mu0 from the "
                "kernel's dt*mu0/rho_water coefficient)."
            )
        self._init_two_phase(tp_cfg)
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
        # Consistent conservative mass/momentum transport (Nangia et al. 2019,
        # Desjardins-Moureau): the cure for the high-density-ratio instability
        # (non-conservative is stable only to ~100:1; water/air is 833:1). When
        # on, ``fluid_step`` transports rho*u conservatively with the SAME mass
        # flux that evolves the density, recovers u = rho*u / rho consistently
        # (no 833x blow-up at the interface), and the interface is advected by
        # that same flux (the Weymouth-Yue VOF in finalize_step is skipped).
        # Runs a python-style step (the fused kernel is bypassed); default OFF
        # so every existing two-phase case is byte-for-byte unchanged.
        self._consistent_momentum = bool(cfg.get("consistent_momentum", False))
        # Three-phase density (Nangia 2019 WSI, Eq. 23): treat the body as a
        # smoothed THIRD density phase rho_solid, blended by mu0 (the BDIM fluid
        # fraction IS the body Heaviside), INSIDE the variable-density projection
        # coefficient and the gravity body force -- instead of mu0-EXCLUDING the
        # body. Targets the body-interface band spurious current (the killer).
        # None = off (current mu0-exclusion behaviour). Python path only for now.
        self._rho_solid = cfg.get("rho_solid", None)
        if self._rho_solid is not None:
            self._rho_solid = float(self._rho_solid)
        # mu0-free coefficient: the cleanest statement of "no hole in the Poisson"
        # -- the projection coefficient is simply c = dt/rho_fluid (the VOF
        # water/air mobility) EVERYWHERE, with mu0 dropped, so the body is purely
        # a velocity constraint and never zeros the coefficient. This is the
        # rho_solid limit rho_solid -> rho_flow (the mu0 blend cancels), but uses
        # the LOCAL VOF density inside the body instead of a pinned constant.
        # Works on BOTH paths: the python coefficient builder skips mu0; the
        # kernel-mode projection rescale returns dt/rho_face directly (discarding
        # the kernel's mu0-laden coeff). Mutually exclusive with rho_solid.
        self._mu0_free_coeff = bool(cfg.get("mu0_free_coeff", False))
        if self._mu0_free_coeff and self._rho_solid is not None:
            raise ValueError(
                "two_phase.mu0_free_coeff and rho_solid both set the projection "
                "coefficient; enable only one (mu0_free_coeff is the rho_solid="
                "rho_flow limit)."
            )
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
        if self._consistent_momentum:
            flags.append("CONSISTENT momentum")
        if self._mu0_free_coeff:
            flags.append("mu0-free coeff (c=dt/rho_fluid everywhere)")
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
        out = []
        for d, ax in enumerate(self._bdim_axis_names):       # 'u','v'[,'w']
            recip_face = tp.recip_density_face(d)            # (1/ρ) water/air blend
            if self._mu0_free_coeff:
                # mu0-FREE: c = dt/rho_fluid everywhere (no body hole); the body
                # is purely a velocity constraint. Uses the LOCAL VOF density,
                # so the body interior carries whatever fluid (water/air) the VOF
                # places there. (rho_solid -> rho_flow limit; mu0 cancels.)
                out.append(dt * recip_face)
                continue
            mu0 = getattr(self, f'mu0_all_{ax}')             # (1−δ^B), fluid fraction
            if atb:
                # Air-transparent body: mu0_eff = 1 - alpha_face*(1 - mu0)
                # = alpha_face*mu0 + (1-alpha_face). Body invisible in air.
                a_face = self._alpha_face(d)
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

    def _mu0_cc(self):
        """Cell-centred BDIM fluid fraction mu0 (1 in fluid, 0 in body, smoothed
        over the band) from the union body SDF -- the body Heaviside (= H_body)."""
        return self.composite_body.mu_funcs(self.composite_body.sdf_val)[0]

    def _alpha_face(self, d):
        """Water fraction ``alpha`` on the staggered *d*-face grid (full grid;
        see :meth:`_face_mean`)."""
        return self._face_mean(self.two_phase.alpha, d)

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
            a_face = self._alpha_face(i)
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
    # The fused single-phase bdim_forcing writes ``S = mu0*(u'-b) + b + mu1*nd``
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
    #    ``_compute_bdim_coefficients`` (standard / rho_solid three-phase /
    #    mu0-free, with the air-transparent masking and the carved-init gate)
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
        for d, (vel, prime) in enumerate(zip(vels, primes)):
            w = 1.0 - self._alpha_face(d)          # lerp weight toward u' (air)
            if self._alpha_exclude_body:
                w = w * (self._face_mean(comp.sdf_val, d) >= 0).to(w.dtype)
            vel.lerp_(prime, w)
        self.adv_diff_solver.set_BCs(*vels)

    def _rescale_kernel_coeffs_two_phase(self, coeffs):
        """Return fresh two-phase Poisson coefficients from the kernel's
        water-normalised ``c_kernel = dt*mu0/rho_water`` ones, evaluating the
        same per-mode formulas as ``_compute_bdim_coefficients``."""
        tp = self.two_phase
        dt = float(self.dt)
        rs = self._rho_solid
        q = tp.recip_density_cc()                  # one full-grid 1/rho field
        comp = self.composite_body
        out = []
        for d, c_kernel in enumerate(coeffs):
            if c_kernel is None:                   # cw in 2-D
                out.append(None)
                continue
            inv_rho = self._kernel_face_mean(q, d)
            if self._mu0_free_coeff:
                out.append(dt * inv_rho)
                continue
            mu0 = c_kernel * (tp.rho_water / dt)
            if self._air_transparent_body:
                a = self._kernel_face_mean(tp.alpha, d)
                mu0_t = a * mu0 + (1.0 - a)
                if self._alpha_exclude_body:
                    sdf_face = self._kernel_face_mean(comp.sdf_val, d)
                    mu0_t = torch.where(sdf_face >= 0.0, mu0_t, mu0)
                mu0 = mu0_t
            if rs is not None:
                out.append(dt / (mu0 / inv_rho + (1.0 - mu0) * rs))
            else:
                out.append(dt * mu0 * inv_rho)
        return tuple(out)

    def project(self, *args, ch=None, cv=None, cw=None, ch_cc=None, **kwargs):
        """Fused path only: blend the BDIM velocity with u' (air-transparent
        identity) and rescale the coefficients to the two-phase formulas, then
        run the base variable-coefficient projection.  The consistent-momentum
        path arrives here with freshly-built two-phase coefficients (not the
        solver's persistent water-normalised buffers), so the identity check
        on ``_ch_persist`` keeps it untouched."""
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

    def _needs_python_mu_normals(self):
        """The consistent-momentum step applies the velocity BDIM and builds
        the two-phase coefficients in python, both of which read the
        staggered mu/normal pack — so it always needs the recompute (the
        streaming update publishes the staggered SDFs it consumes)."""
        return self._consistent_momentum or super()._needs_python_mu_normals()

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

    def _apply_gravity_body_force(self, *vels):
        """In consistent mode gravity is applied as a rho*g body force INSIDE the
        conservative momentum advection (see :meth:`_consistent_advect_2d`), so
        the base velocity pre-kick must be suppressed to avoid double-counting
        (and to keep the density-transport flux on the un-kicked velocity)."""
        if self._consistent_momentum:
            return vels
        return super()._apply_gravity_body_force(*vels)

    def fluid_step(self, *args):
        """As the base (advect-BDIM-project), but in consistent mode the velocity
        advection is replaced by consistent conservative momentum transport.
        BDIM, the variable-density coefficients, and the projection are reused
        unchanged. Dimension-agnostic."""
        self._try_deferred_alpha_carve()
        if not self._consistent_momentum:
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
            _nu_eff = (self.nu + nu_t) if nu_t is not None else None
            inv_dh2 = [1.0 / (h * h) for h in ads.dh]
            scale = float(dt) if _nu_eff is not None else float(self.nu) * float(dt)
            # diffuse_add_ pattern with separate src (u_start[i]) / dst (vi):
            # copy src → temp, then fused Laplacian from temp into dst.
            _tmp = torch.empty_like(u_start[i])
            _diffusion._copy_full_grid_eager(u_start[i], _tmp)
            _diffusion._fused_diffuse_add_eager(_tmp, vi, scale, inv_dh2,
                                                nu_eff_t=_nu_eff)
            out[i] = vi

        tp.alpha = ((r_new - ra) / (rw - ra)).clamp(0.0, 1.0)
        return tuple(out)

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
        # In consistent (evolve) mode the interface already rode the shared mass
        # flux inside fluid_step (alpha synced there); skip the standalone VOF.
        if not self._consistent_momentum:
            self._advect_vof(u, v, p, iteration, w_vel=w_vel)
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
