"""Band-treatment validation: which eulerian pressure-force readout recovers
the EXACT Archimedes buoyancy on a static fully-submerged sphere?

A sphere of radius R sits motionless, fully submerged in uniform-density water
(rho_air = rho_water, so no interface near the body) under gravity.  The exact
vertical pressure force is Archimedes buoyancy  F_z = rho * g * (4/3 pi R^3).

We settle the TwoPhaseSolver's hydrostatic pressure, then compute F_z from the
SAME settled field three ways, to isolate the force-band quadrature:

  * FULL-band n.delta   : -Sum p n_z delta_eps(phi) h^3    over the whole band
  * sdf<0-truncated     : same, but p zeroed where sdf<0   (single-phase zpi)
  * partial-Heaviside dH: -Sum p (d_z H_eps(phi)) h^3      (SBP weight)

Run:  python -m lilytorch.validation.two_phase_3d.band_treatment_check
"""
import math
import torch

from lilytorch.src.two_phase_solver import TwoPhaseSolver


def build_pars(D=0.10, pts_per_D=16):
    R = 0.5 * D
    g = 9.81
    L = 4.0 * D                                  # cubic domain, sphere centred
    N = int(round(4.0 * pts_per_D))
    h = L / N
    cx = cy = cz = L / 2.0
    nu = 1e-5
    dt = 0.02 * h / math.sqrt(g * D)
    rho = 1000.0
    pars = {
        "solver": {
            "use_gpu": torch.cuda.is_available(), "nthreads": 1,
            "Nx": N, "Ny": N, "Nz": N,
            "xmin": 0.0, "xmax": L, "ymin": 0.0, "ymax": L, "zmin": 0.0, "zmax": L,
            "nt": 10000, "nu": nu, "rho": rho, "dt": dt,
            "convection_method": "quick",
            "poisson_tol": 1e-7, "jacobi_weight": 1.0,
            "poisson_max_cycles": 50, "poisson_max_mgcg_cycles": 50,
            "poisson_nsmoothing": 6, "poisson_verbose": False,
            "poisson_folder": "lilytorch/data/",
            "poisson_method": "mgcg", "poisson_smoother": "rbgs",
            "dtype": "float64", "solver_method": "python",
            "force_method": "eulerian",
            "gravity": [0.0, 0.0, -g],
            "two_phase": {                       # uniform density -> pure water
                "alpha_init": "lambda X, Y, Z: torch.ones_like(X).double()",
                "rho_water": rho, "rho_air": rho,
                "nu_water": nu, "nu_air": nu,
                "face_density": "harmonic",
            },
        },
        "boundary_conditions": {
            "BC_type_u": ["D"] * 6, "BC_values_u": [0.0] * 6,
            "BC_type_v": ["D"] * 6, "BC_values_v": [0.0] * 6,
            "BC_type_w": ["D"] * 6, "BC_values_w": [0.0] * 6,
        },
        "body": {
            "type": "composite_analytical", "plotting": False,
            "sdf": [f"lambda x, y, z: sphere(x,y,z,xt=0.0,yt=0.0,zt=0.0,r={R})"],
            "update_maps": [{
                "rotation": "lambda t: 0.0*t",
                "translation": [f"lambda t: {cx} + 0.0*t",
                                f"lambda t: {cy} + 0.0*t",
                                f"lambda t: {cz} + 0.0*t"],
            }],
        },
        "output": {"save_path": "/tmp/lilytorch_band_check/", "save_frames": False,
                   "save_every": 99999, "save": False, "save_drags": False,
                   "vmin": -3.0, "vmax": 3.0},
    }
    return pars, dict(R=R, g=g, h=h, N=N, rho=rho, dt=dt,
                      V=4.0 / 3.0 * math.pi * R ** 3)


def _heaviside(phi, eps):
    x = (phi / eps).clamp(-1.0, 1.0)
    return 0.5 * (1.0 + x + torch.sin(math.pi * x) / math.pi)


def fz_readouts(solver, h):
    """Compute the vertical pressure force three ways from the settled field."""
    p = solver.p0
    cb = solver.composite_body
    sdf = cb.sdf_val
    eps = float(cb.bodies[0].eps)
    h3 = h ** 3
    # unit normal from the SDF gradient (z-component) + cosine delta
    nz = torch.gradient(sdf, spacing=h, dim=2, edge_order=2)[0]
    gmag = torch.sqrt(sum(torch.gradient(sdf, spacing=h, dim=d, edge_order=2)[0] ** 2
                          for d in range(3))).clamp(min=1e-6)
    nz = nz / gmag
    delta = torch.where(sdf.abs() < eps,
                        (1.0 + torch.cos(math.pi * sdf / eps)) / (2.0 * eps),
                        torch.zeros_like(sdf))
    full = float(-(p * nz * delta).sum() * h3)
    p_trunc = torch.where(sdf < 0, torch.zeros_like(p), p)
    trunc = float(-(p_trunc * nz * delta).sum() * h3)
    H = _heaviside(sdf, eps)
    dHz = torch.gradient(H, spacing=h, dim=2, edge_order=2)[0]
    dH = float(-(p * dHz).sum() * h3)
    return full, trunc, dH


def leak_isolation(N=64, L=0.4, theta_deg=30.0):
    """Decisive: does the n·δ horizontal HYDROSTATIC leak come from the
    QUADRATURE or from zero_pressure_inside?  Build an ASYMMETRIC body (a box
    tilted in x-z, so its surface correlates z with n_x) and a PURELY ANALYTIC
    hydrostatic field p = rho g (z_top - z) -- NO solver, NO Poisson, NO pressure
    zeroing anywhere.  A static body's true horizontal force is 0.  If full-band
    n·δ gives F_x != 0 here, the leak is the discrete n·δ quadrature (Σ z n_x δ ≠ 0
    for an asymmetric body), and zero_pressure_inside is exonerated."""
    g, rho = 9.81, 1000.0
    h = L / N
    xs = (torch.arange(N, dtype=torch.float64) + 0.5) * h
    X, Y, Z = torch.meshgrid(xs, xs, xs, indexing="ij")
    cx = cy = cz = L / 2.0
    half = (0.07, 0.045, 0.035)
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    x, y, z = X - cx, Y - cy, Z - cz
    xr = c * x + s * z
    zr = -s * x + c * z
    qx, qy, qz = xr.abs() - half[0], y.abs() - half[1], zr.abs() - half[2]
    outside = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + qz.clamp(min=0) ** 2)
    inside = torch.minimum(torch.maximum(qx, torch.maximum(qy, qz)),
                           torch.zeros_like(qx))
    sdf = outside + inside                       # exact box SDF (|grad|=1)
    p = rho * g * (L - Z)                         # analytic hydrostatic, z_top=L
    eps = 2.0 * h
    h3 = h ** 3
    gx = torch.gradient(sdf, spacing=h, dim=0, edge_order=2)[0]
    gz = torch.gradient(sdf, spacing=h, dim=2, edge_order=2)[0]
    gmag = torch.sqrt(gx ** 2
                      + torch.gradient(sdf, spacing=h, dim=1, edge_order=2)[0] ** 2
                      + gz ** 2).clamp(min=1e-6)
    nx, nz = gx / gmag, gz / gmag
    delta = torch.where(sdf.abs() < eps,
                        (1.0 + torch.cos(math.pi * sdf / eps)) / (2.0 * eps),
                        torch.zeros_like(sdf))
    V = 8 * half[0] * half[1] * half[2]
    F_arch = rho * g * V

    def rd(weight_x, weight_z, pp):
        return (float(-(pp * weight_x).sum() * h3),
                float(-(pp * weight_z).sum() * h3))
    full = rd(nx * delta, nz * delta, p)
    p_tr = torch.where(sdf < 0, torch.zeros_like(p), p)
    trunc = rd(nx * delta, nz * delta, p_tr)
    H = _heaviside(sdf, eps)
    dHx = torch.gradient(H, spacing=h, dim=0, edge_order=2)[0]
    dHz = torch.gradient(H, spacing=h, dim=2, edge_order=2)[0]
    dH = rd(dHx, dHz, p)
    print("=" * 70)
    print("  LEAK ISOLATION: tilted box, ANALYTIC hydrostatic p, NO solver")
    print(f"  (no Poisson, no zero_pressure_inside)  tilt={theta_deg}°  grid {N}^3")
    print(f"  TRUE: F_x = 0 (static body) ;  F_z = rho g V = {F_arch:+.4f} N")
    print("=" * 70)
    print(f"  {'readout':22s}{'F_x [N]':>12s}{'F_z [N]':>12s}{'F_z/Arch':>10s}")
    for lbl, (fx, fz) in [("FULL-band n·δ", full), ("sdf<0-truncated", trunc),
                          ("partial-Heaviside ∂H", dH)]:
        print(f"  {lbl:22s}{fx:+12.5f}{fz:+12.4f}{fz / F_arch:10.4f}")
    print(f"  => if FULL-band n·δ F_x != 0 with NO solver, the leak is the "
          f"QUADRATURE,\n     not zero_pressure_inside.")
    print()


def asym_lagrangian_test(pts_per_D=16, n_settle=250):
    """Empirical close-out: does the LAGRANGIAN watertight integral give F_x≈0 on
    an ASYMMETRIC SOLVER body (tilted box), where eulerian n·δ leaks?  Static box,
    uniform-ρ water + gravity; true F_x=0, F_z=ρgV.  Runs the solver's own readout
    (eulerian vs lagrangian) so it includes the real Poisson pressure."""
    g, rho = 9.81, 1000.0
    D = 0.10
    L = 4.0 * D
    N = int(round(4.0 * pts_per_D))
    h = L / N
    cx = cy = cz = L / 2.0
    nu = 1e-5
    dt = 0.02 * h / math.sqrt(g * D)
    # SMOOTH asymmetric body: a TAPERED capsule (r1≠r2), axis tilted 30° in x-z.
    # Smooth surface -> marching-cubes triangulation is clean (unlike the box),
    # so this fairly tests whether the Lagrangian integral is leak-free.  Tapered
    # + tilted => Σ z n_x δ ≠ 0 (asymmetric), so n·δ should leak F_x.
    # TRUE-SDF asymmetric body: union (min) of two offset spheres of different
    # radii, centres displaced diagonally in x-z -> a tilted "dumbbell".  min of
    # two true SDFs keeps |∇|≈1 (clean band quadrature + clean marching-cubes),
    # and the diagonal offset + unequal radii make Σ z n_x δ ≠ 0 (it should leak).
    ra, rb = 0.045, 0.032
    ax, az = cx - 0.035, cz - 0.035
    bx, bz = cx + 0.045, cz + 0.045
    sa = f"sphere(x,y,z,xt={ax},yt={cy},zt={az},r={ra})"
    sb = f"sphere(x,y,z,xt={bx},yt={cy},zt={bz},r={rb})"
    sdf = f"lambda x, y, z: torch.minimum({sa}, {sb})"
    F_arch = None  # computed numerically from the settled SDF (V = #interior * h^3)

    def build(force_method):
        return {
            "solver": {
                "use_gpu": torch.cuda.is_available(), "nthreads": 1,
                "Nx": N, "Ny": N, "Nz": N,
                "xmin": 0.0, "xmax": L, "ymin": 0.0, "ymax": L, "zmin": 0.0, "zmax": L,
                "nt": 10000, "nu": nu, "rho": rho, "dt": dt,
                "convection_method": "quick", "poisson_tol": 1e-7,
                "jacobi_weight": 1.0, "poisson_max_cycles": 50,
                "poisson_max_mgcg_cycles": 50, "poisson_nsmoothing": 6,
                "poisson_verbose": False, "poisson_folder": "lilytorch/data/",
                "poisson_method": "mgcg", "poisson_smoother": "rbgs",
                "dtype": "float64", "solver_method": "python",
                "force_method": force_method, "gravity": [0.0, 0.0, -g],
                "two_phase": {
                    "alpha_init": "lambda X, Y, Z: torch.ones_like(X).double()",
                    "rho_water": rho, "rho_air": rho, "nu_water": nu, "nu_air": nu,
                    "face_density": "harmonic",
                },
            },
            "boundary_conditions": {
                "BC_type_u": ["D"] * 6, "BC_values_u": [0.0] * 6,
                "BC_type_v": ["D"] * 6, "BC_values_v": [0.0] * 6,
                "BC_type_w": ["D"] * 6, "BC_values_w": [0.0] * 6,
            },
            "body": {"type": "composite_analytical", "plotting": False, "sdf": [sdf],
                     "update_maps": [{"rotation": "lambda t: 0.0*t",
                                      "translation": ["lambda t: 0.0*t"] * 3}]},
            "output": {"save_path": "/tmp/lilytorch_band_box/", "save_frames": False,
                       "save_every": 99999, "save": False, "save_drags": False,
                       "vmin": -3.0, "vmax": 3.0},
        }

    print("=" * 70)
    print("  ASYMMETRIC SMOOTH SOLVER body (tilted dumbbell, 2 offset spheres):")
    print("  does LAGRANGIAN give F_x≈0 where eulerian n·δ leaks?  (TRUE F_x=0)")
    print("=" * 70)
    print(f"  {'force_method':18s}{'F_x [N]':>12s}{'F_z [N]':>12s}{'F_z/Arch':>10s}")
    for fm in ("eulerian", "lagrangian"):
        solver = TwoPhaseSolver(build(fm), dtype=torch.float64, compute_forces=True)
        solver.inside = lambda *a, **k: True
        cb = solver.composite_body
        _o = cb.update
        def _u(*a, **k):
            _o(*a, **k); cb.sdf_vals = [b.sdf_val for b in cb.bodies]
        cb.update = _u
        solver.set_initial_conditions()
        u, v, w, p = solver.u0, solver.v0, solver.w0, solver.p0
        for it in range(n_settle):
            u, v, p, w = solver.advance_and_compute_loads(u, v, p, it, it * dt, w_vel=w)
            solver.finalize_step(u, v, p, it, w_vel=w)
        V = float((cb.sdf_val < 0).sum()) * h ** 3
        F_arch = rho * g * V
        F = solver.get_loads()[0][0]
        fx, fz = float(F[0]), float(F[2])
        print(f"  {fm:18s}{fx:+12.5f}{fz:+12.4f}{fz / F_arch:10.4f}  (V={V:.2e})")
    print()


def _run_sphere(pars, m, n_settle, force_method):
    """Settle the static sphere and return (solver, |u|max). force_method picks
    the solver's own readout (eulerian or lagrangian)."""
    pars = {**pars, "solver": {**pars["solver"], "force_method": force_method}}
    solver = TwoPhaseSolver(pars, dtype=torch.float64, compute_forces=True)
    solver.inside = lambda *a, **k: True
    cb = solver.composite_body
    _orig = cb.update
    def _upd(*a, **k):
        _orig(*a, **k); cb.sdf_vals = [b.sdf_val for b in cb.bodies]
    cb.update = _upd
    solver.set_initial_conditions()
    u, v, w, p = solver.u0, solver.v0, solver.w0, solver.p0
    dt = m["dt"]
    for it in range(n_settle):
        u, v, p, w = solver.advance_and_compute_loads(u, v, p, it, it * dt, w_vel=w)
        solver.finalize_step(u, v, p, it, w_vel=w)
    umax = max(u.abs().max().item(), v.abs().max().item(), w.abs().max().item())
    return solver, umax


def _heaviside_t(phi, eps):
    x = (phi / eps).clamp(-1.0, 1.0)
    return 0.5 * (1.0 + x + torch.sin(math.pi * x) / math.pi)


def kernel_ndelta_gauge(grid_n=56, extent=0.4):
    """Gauge property of the union ndelta readout (the reason deltaH existed).

    Builds the streaming-kernel force state directly (real sphere SDF textures,
    no FARMS) and runs ``streaming_sdf_forces_post_3d`` with the default union
    ndelta submethod (0) on single / dumbbell / 3-link-chain bodies.  The union
    band measure is summation-by-parts, so:
      * constant p      -> Σ_b F_p == 0  (no spurious net force / gauge leak)
      * hydrostatic p   -> horizontal net == 0, vertical net != 0 (buoyancy)
    This is the frozen-field proxy for the live two-phase acceptance test;
    union-∇H subsumes the retired deltaH readout everywhere deltaH was used.
    """
    import types
    import numpy as np
    from lilytorch.integration.BDIMhandler import BDIMhandler
    from lilytorch.src.forces import streaming_sdf_forces_post_3d
    from lilytorch.src.native import interp_3d

    DT = torch.float64
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    class _B: pass
    class _C: pass

    def mk_body(center, R, half, M, h):
        b = _B(); b.h = h
        axc = [torch.linspace(-half, half, M, dtype=DT) for _ in range(3)]
        LX, LY, LZ = torch.meshgrid(*axc, indexing="ij")
        m = {"F": torch.sqrt(LX**2 + LY**2 + LZ**2) - R}
        for ax, a in enumerate("xyz"):
            coord = axc[ax]
            m[f"b{a}"] = coord; m[f"b{a}0"] = float(coord[0])
            m[f"b{a}_last"] = float(coord[-1]); m[f"inv_d{a}"] = float((M - 1) / (2 * half))
        m["inv_vol"] = m["inv_dx"] * m["inv_dy"] * m["inv_dz"]
        b._stream_meta = m
        b.local_aabb = (torch.tensor([-half]*3, dtype=DT), torch.tensor([half]*3, dtype=DT))
        b.eps = 2.0 * h; b._center = center; b._R = R
        return b

    def mk_comp(n, bodies):
        c = _C(); c.h = extent / (n - 1)
        for a in "xyz":
            coord = torch.linspace(0.0, extent, n, dtype=DT)
            setattr(c, a, coord); setattr(c, f"g{a}_1d", coord.contiguous())
        gs = (n, n, n); B = len(bodies)
        c.sdf_val = torch.full(gs, 1e6, dtype=DT)
        c.bodies = bodies; c.nbodies = B; c.body_ids = [(0, i) for i in range(B)]
        c.com_pos = torch.stack([torch.tensor(b._center, dtype=DT) for b in bodies])
        c._body_aabbs = [None]*B; c._sdf_sparse = [None]*B
        return c, gs

    def kin_id(bodies):
        B = len(bodies)
        com = np.array([b._center for b in bodies]); urdf = com.copy()
        return ([com], [urdf], [np.stack([np.eye(3)]*B)],
                [np.zeros((B, 3))], [np.zeros((B, 3))])

    def case(bodies, label):
        comp, gs = mk_comp(grid_n, bodies); h = comp.h; B = len(bodies)
        hd = BDIMhandler.__new__(BDIMhandler)
        # The mock scene tensors (comp.sdf_val, comp.g*_1d, body SDF tables) live
        # on CPU, so the streaming marshalling runs on CPU (eager path); the force
        # op below is then handed CPU→DEV copies.  fs must carry device/dtype and
        # the blend width that _launch_body_update now reads.
        hd.ndim = 3; hd.device = "cpu"; hd.dtype = DT; hd.dtype_np = np.float64
        hd._sim_axes = [0, 1, 2]
        hd.fluid_solver = types.SimpleNamespace(
            composite_body=comp, grid_shape=gs,
            device="cpu", dtype=DT, _body_vel_blend_cells=0.0,
            _sdf_interp_method=0)
        hd.force_method = "eulerian"
        hd.gather_data = types.MethodType(lambda self, it, k=kin_id(bodies): k, hd)
        hd._update_streaming_multi(0.0, 0)

        # analytic union SDF (streaming rasteriser's world->local convention
        # differs from the force kernel's; the op reads sdf_cc explicitly, so a
        # real, consistent union field is all that is needed for op parity).
        Xg, Yg, Zg = torch.meshgrid(comp.x, comp.y, comp.z, indexing="ij")
        su = torch.full(gs, 1e6, dtype=DT)
        for b in bodies:
            cx, cy, cz = b._center
            su = torch.minimum(su, torch.sqrt((Xg-cx)**2 + (Yg-cy)**2 + (Zg-cz)**2) - b._R)

        st = comp._kernel_static_3d; sp = comp._kernel_step
        to = lambda t: t.to(DEV)
        eps_body = float(bodies[0].eps); h3 = h**3; ph_tau = 1.5 * h
        sdf_val = to(su).contiguous()
        gx, gy, gz = to(sp["gx"]), to(sp["gy"]), to(sp["gz"])
        F_flat = to(st["F_flat"]); F_off = st["F_offsets"].to(DEV)
        shapes = st["body_shapes"].to(DEV); meta = to(st["body_meta"])
        kin_t = to(sp["kin"]); a_lo = sp["aabb_lo"].to(DEV); a_dim = sp["aabb_dim"].to(DEV)
        mv = int(sp["max_vol"])
        # ---- constant-p gauge: the union net force must vanish (SBP) --------
        p_const = torch.full(gs, 1234.0, dtype=DT, device=DEV).contiguous()
        # ---- hydrostatic p: horizontal net vanishes, vertical = buoyancy ----
        p_hydro = to((1000.0 * 9.81 * (extent - Zg)).to(DT)).contiguous()
        z = torch.zeros(gs, dtype=DT, device=DEV).contiguous()
        nu_rho = torch.zeros(1, dtype=DT, device=DEV)

        def run_op(pf):
            out = torch.zeros((B, 12), dtype=torch.float64, device=DEV)
            # submethod 0 = union ndelta; nu_rho=0 so only the pressure channel
            # is exercised (the viscous channel is trivially zero).
            streaming_sdf_forces_post_3d(
                F_flat, F_off, shapes, meta, kin_t, a_lo, a_dim, gx, gy, gz,
                h, mv, sdf_val, 0, z, z, z, pf, nu_rho,
                eps_body, 0.0, 0.0, h3, 1, out, 0, ph_tau)
            if DEV == "cuda": torch.cuda.synchronize()
            return out

        # ---- hydrostatic p FIRST: it gives the physical force scale ----------
        fp_h = run_op(p_hydro)[:, 6:9].sum(0)          # union net, hydrostatic p
        horiz = max(abs(float(fp_h[0])), abs(float(fp_h[1])))
        vert = abs(float(fp_h[2]))                      # buoyancy scale (O(rho g V))
        hstat_ok = (vert > 100.0 * horiz)

        # ---- constant-p gauge: the union net force must vanish (SBP) ---------
        # Normalise by the buoyancy scale, NOT the per-body force.  For a SINGLE
        # closed body the ENTIRE per-body force is ~0 under constant p (the
        # discrete ∮ ∂_iH dV = 0), so the per-body magnitude is itself machine
        # noise and cannot serve as a scale (it would trip a clamp and read as a
        # false leak).  The multi-body partition cancellation is still exercised:
        # a real gauge leak scales with the constant-p magnitude, not with eps.
        fp_c = run_op(p_const)[:, 6:9]
        net = fp_c.sum(0).abs().max().item()           # union net -> ~0 by SBP
        rel = net / max(vert, 1e-30)
        gauge_ok = rel < 1e-9

        ok = gauge_ok and hstat_ok
        print(f"  {label:20s} const-p net/buoy={rel:.1e}"
              f"  hydro |Fh|={horiz:.2e} Fz={vert:.2e}"
              f"  [{'OK' if ok else 'FAIL'}]")
        return ok

    print("=" * 70)
    print(f"  UNION ndelta gauge property  (dev={DEV}, {grid_n}^3)")
    print("  constant p -> union net force == 0 (SBP) ; hydrostatic p -> vertical only")
    print("=" * 70)
    cc = 0.2; hh = extent / (grid_n - 1)
    ok = True
    ok &= case([mk_body([cc, cc, cc], 0.05, 0.08, 33, hh)], "single sphere")
    ok &= case([mk_body([cc-0.03, cc, cc-0.03], 0.045, 0.075, 33, hh),
                mk_body([cc+0.035, cc, cc+0.035], 0.035, 0.065, 33, hh)], "two-sphere dumbbell")
    ok &= case([mk_body([cc-0.06, cc, cc], 0.035, 0.06, 29, hh),
                mk_body([cc, cc, cc], 0.04, 0.065, 29, hh),
                mk_body([cc+0.06, cc, cc], 0.035, 0.06, 29, hh)], "three-link chain (seam)")
    print(f"  => UNION ndelta GAUGE: {'PASSED' if ok else 'FAILED'}")
    print()
    return ok


def main(pts_per_D=16, n_settle=300):
    # kernel_ndelta_gauge is the frozen proxy for the live two-phase acceptance
    # test: it drives the native union-∇H readout directly (no fluid solve) and
    # asserts the SBP gauge property that union-∇H inherited from the retired
    # deltaH readout.  leak_isolation is a pure-analytic ∂H diagnostic.
    #
    # The former standalone-solver sections (asym_lagrangian_test and the
    # _run_sphere / build_pars band-treatment sweep) drove the LEGACY python
    # eulerian force path on a non-FARMS TwoPhaseSolver.  That path has been
    # removed (eulerian forces now require the native streaming/BDIMhandler
    # path), so those sections are retired along with it; the gauge proxy below
    # is their replacement.
    ok = kernel_ndelta_gauge()
    leak_isolation()
    if not ok:
        raise SystemExit("UNION ndelta GAUGE FAILED")


if __name__ == "__main__":
    main()
