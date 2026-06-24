"""
3-D sphere WATER-ENTRY (TWO-PHASE VOF + BDIM), standalone validation.

Reproduces the canonical Weymouth & Yue (2011, *JCP* **230** 6233, §4.2)
"impact of a sphere through a free surface": a rigid sphere is driven
**downward at a prescribed constant speed** ``U`` across an air/water
interface, opening a cavity and throwing a splash crown while the simulation
stays bounded and mass-conserving and the vertical hydrodynamic load rises
from ~0 (in air) to a positive impact/buoyancy load as the wetted area grows.

Physics matched to the paper (§4.2):

* **Domain** ``3 x 3 x 6`` diameters (here z is the vertical / gravity axis),
  uniform Cartesian grid, ``dx == dy == dz`` (solver requirement).  The paper
  uses 80 points/diameter and quarter-symmetry; this standalone runs the FULL
  domain at a coarser, configurable ``pts_per_D`` so it finishes in ~1 min on
  one GPU.  Refine ``pts_per_D`` toward 80 to approach the paper's accuracy.
* **Froude number** ``Fr = U / sqrt(g D) = 3`` — the *governing* similarity
  number for water entry (inertia vs gravity), matched EXACTLY (paper Fig. 5).
  ``U`` is derived from ``Fr`` and ``D``.
* **Reynolds number** ``Re = U D / nu`` — set explicitly via ``Re`` and the
  solver kinematic viscosity ``nu = U D / Re`` (the momentum diffusion uses the
  constant solver ``nu``; the VOF only carries the *density* jump).  NOTE: the
  paper's physical water entry is ``Re ~ 1e5`` (real water, ``nu = 1e-6``),
  which is unresolvable on a standalone laminar grid; we therefore run a
  resolved, laminar ``Re`` (default 500, the WaterLily/jellyfish regime) and
  keep ``Re`` a single knob so it can be dialed up with the grid.  Fr (not Re)
  controls the cavity/splash kinematics, so the qualitative water-entry physics
  is reproduced at the resolved Re.
* **Density / viscosity jump** real ``1000 : 1`` water:air (paper Eq. 33),
  carried by the conservative Weymouth & Yue VOF (bounded, mass-conserving,
  no clamping / reconstruction).
* **No-slip sphere**, prescribed motion (the free-body force coupling — release
  the sphere and integrate under the hydrodynamic load, eventually through
  MuJoCo — is a deliberate follow-up; here we validate the FLUID response to a
  known body motion, exactly as the paper's prescribed-``U`` test does).

CFL: the impact SPLASH JET reaches several times the entry speed ``U`` at the
moment of impact / cavity collapse, so ``dt`` is sized for the jet
(``dt ~ 0.04 h / max(U, c)``, ``c = sqrt(g D)``), NOT for ``U`` — sizing on
``U`` looks safe but blows up once ``|u|_jet dt / h`` crosses ~1 (the same
lesson as the 2-D ``run_drop_sphere.py``).

Boundary conditions: a CLOSED box (Dirichlet ``u=0`` on all 6 walls, all-Neumann
pressure) — the validated two-phase path.  The paper uses free-slip far-field
walls; the box walls sit 1.5 D from the sphere, so wall blockage is mild and
the entry physics near the interface is unaffected for this qualitative test.

Checks: (1) stable through entry (no blow-up, peak ``|u|`` bounded),
(2) VOF mass conserved, (3) ``F_z`` rises positive as the wetted area grows.

Each save step writes TWO figures so the cavity / splash can be inspected:
  * ``slice_NNNNN.png`` -- x-z mid-plane (interface ``alpha``, speed ``|u|``,
    in-plane vorticity ``omega_y``);
  * ``iso_NNNNN.png``   -- 3-D isosurfaces: the air/water interface (``alpha =
    0.5`` -> open cavity + splash crown) + the sphere, and a
    velocity-magnitude isosurface + the sphere.  Axes fixed to the full domain.
"""

import math
import os

import numpy as np
import torch
from tqdm import tqdm

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# Standalone validation: skip torch.compile (avoids first-step compile latency).
torch.compile = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))

from lilytorch.src.two_phase_solver import TwoPhaseSolver
from lilytorch.src.poisson_mult import PoissonSolver

# CPU fallback: the standalone multigrid CUDA v-cycle is GPU-only; on CPU
# (no CUDA) dispatch to the plain python v-cycle so the script still runs.
_orig = PoissonSolver._dispatch_vcycle
PoissonSolver._dispatch_vcycle = (
    lambda self, f, p, fa: (_orig(self, f, p, fa) if f.is_cuda
                            else self._vcycle(f, p, fa)))


def build_pars(D=0.10, pts_per_D=16, Fr=3.0, Re=500.0):
    """Assemble the solver config for the sphere water-entry.

    Parameters
    ----------
    D : sphere diameter (m).  R = D/2.
    pts_per_D : grid resolution (cells per diameter).  Paper uses 80.
    Fr : Froude number U/sqrt(g D) (paper §4.2 uses 3).
    Re : Reynolds number U D / nu (sets the solver kinematic viscosity).
    """
    R   = 0.5 * D
    g   = 9.81
    c   = math.sqrt(g * D)                 # gravity-wave / sqrt(gD) scale
    U   = Fr * c                           # entry speed from the Froude number
    nu  = U * D / Re                       # solver viscosity from the Reynolds number

    # 3 x 3 x 6 diameter domain (z vertical), uniform h.
    Lx = Ly = 3.0 * D
    Lz       = 6.0 * D
    Nx = Ny  = int(round(3.0 * pts_per_D))
    Nz       = int(round(6.0 * pts_per_D))
    h  = Lx / Nx                           # = Lz/Nz (dx==dy==dz)

    # Interface 2 D below the top (water below z=Hint, air above); sphere starts
    # just above the surface so it impacts almost immediately.
    Hint = Lz - 2.0 * D
    z0   = Hint + R + 3.0 * h              # initial sphere-centre height (in air)

    # CFL sized for the impact splash JET, not the entry speed (see docstring).
    dt = 0.04 * h / max(U, c)

    rho_w, rho_a = 1000.0, 1.0
    pars = {
        "solver": {
            "use_gpu": torch.cuda.is_available(), "nthreads": 1,
            "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "xmin": 0.0, "xmax": Lx, "ymin": 0.0, "ymax": Ly,
            "zmin": 0.0, "zmax": Lz,
            "nt": 10000, "nu": nu, "rho": rho_w, "dt": dt,
            "convection_method": "quick",          # upwind-biased (robust at high cell-Re)
            "poisson_tol": 1e-5, "jacobi_weight": 1.0,
            "poisson_max_cycles": 3, "poisson_max_mgcg_cycles": 30,
            "poisson_nsmoothing": 6, "poisson_verbose": False,
            "poisson_folder": "lilytorch/data/",
            "poisson_method": "multigrid", "poisson_smoother": "rbgs",
            "dtype": "float32", "solver_method": "python",
            # Lagrangian surface-integral forces on the REAL pressure: the
            # watertight sphere triangulation gives Σ(A·n)=0, so the integral is
            # gauge-invariant and recovers buoyancy + the dynamic impact load
            # (Cz≈1, matching Weymouth & Yue 2011 Fig. 5). The TwoPhaseSolver
            # default (displaced-volume buoyancy) drops the impact load.
            "force_method": "lagrangian",
            "gravity": [0.0, 0.0, -g],
            "two_phase": {
                "alpha_init": f"lambda X, Y, Z: (Z < {Hint}).double()",
                "rho_water": rho_w, "rho_air": rho_a,
                "nu_water": nu, "nu_air": nu,
                "advection": "cubista", "compression": 1.0,
                "face_density": "harmonic",
            },
        },
        "boundary_conditions": {
            "BC_type_u": ["D"] * 6, "BC_values_u": [0.0] * 6,
            "BC_type_v": ["D"] * 6, "BC_values_v": [0.0] * 6,
            "BC_type_w": ["D"] * 6, "BC_values_w": [0.0] * 6,
        },
        # Base sphere at the centred-grid ORIGIN (xt=yt=zt=0) so the 3-D AABB /
        # surface marching-cubes finds its zero set at the domain centre; the
        # body is then PLACED and DRIVEN by the translation: world centre =
        # (Lx/2, Ly/2, z0 - U t).  Translation lambdas must stay differentiable
        # in t (body velocity = autograd d/dt) -> NEVER wrap in torch.tensor():
        # the z-lambda's slope is the -U entry speed, the x/y are constant.
        "body": {
            "type": "composite_analytical", "plotting": False,
            "sdf": [f"lambda x, y, z: sphere(x,y,z,xt=0.0,yt=0.0,zt=0.0,r={R})"],
            "update_maps": [{
                "rotation": "lambda t: 0.0*t",
                "translation": [f"lambda t: {Lx/2} + 0.0*t",
                                f"lambda t: {Ly/2} + 0.0*t",
                                f"lambda t: {z0} - {U}*t"],
            }],
        },
        "output": {"save_path": "/tmp/lilytorch_tp_drop3d/", "save_frames": False,
                   "save_every": 19999, "save": False, "save_drags": False,
                   "vmin": -3.0, "vmax": 3.0},
    }
    meta = dict(R=R, D=D, g=g, c=c, U=U, nu=nu, Fr=Fr, Re=Re,
                Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, h=h,
                Hint=Hint, z0=z0, dt=dt, rho_w=rho_w, rho_a=rho_a)
    return pars, meta


def save_slice(solver, it, t, m, outdir):
    """Save an x-z mid-plane (y = Ly/2) figure: interface alpha, speed |u|,
    and in-plane vorticity omega_y = du/dz - dw/dx."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jmid = solver.v0.shape[1] // 2
    # interior cell-centred fields on the slice -> (Nx, Nz)
    sl = (slice(1, -1), jmid, slice(1, -1))
    alpha = solver.two_phase.alpha[sl].detach().float().cpu().numpy()
    u = solver.u0[sl].detach().float().cpu().numpy()
    w = solver.w0[sl].detach().float().cpu().numpy()
    speed = np.sqrt(u * u + w * w)
    h = float(m["h"])
    dudz = np.gradient(u, h, axis=1)
    dwdx = np.gradient(w, h, axis=0)
    omega_y = dudz - dwdx

    ext = [0.0, m["Lx"], 0.0, m["Lz"]]           # x (horizontal), z (vertical)
    zc = m["z0"] - m["U"] * t                    # prescribed sphere centre z
    fig, axes = plt.subplots(1, 3, figsize=(11, 6.2), constrained_layout=True)
    for ax, field, title, cmap, kw in (
        (axes[0], alpha,   "alpha (1=water,0=air)", "Blues",  dict(vmin=0, vmax=1)),
        (axes[1], speed,   "|u|  (m/s)",            "viridis", dict(vmin=0, vmax=max(3.0*m["U"], 1e-6))),
        (axes[2], omega_y, "omega_y (1/s)",         "RdBu_r",  dict(vmin=-200, vmax=200)),
    ):
        im = ax.imshow(field.T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, **kw)
        ax.axhline(m["Hint"], color="0.4", lw=0.8, ls="--")          # rest interface
        thc = plt.Circle((m["Lx"] / 2, zc), m["R"], fill=False,
                         color="k", lw=1.0)                           # sphere outline
        ax.add_patch(thc)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        fig.colorbar(im, ax=ax, shrink=0.7)
    pen = (m["Hint"] - zc) / m["R"]
    fig.suptitle(f"sphere water-entry  it={it}  t={t:.4f}s  "
                 f"z_c/R below surf = {pen:+.2f}  (Fr={m['Fr']}, Re={m['Re']:.0f})",
                 fontsize=11)
    fig.savefig(os.path.join(outdir, f"slice_{it:05d}.png"), dpi=90)
    plt.close(fig)


def _iso3d(vol, level, h, origin):
    """Marching-cubes isosurface of ``vol`` at ``level`` in PHYSICAL coords
    (``verts`` shifted to the grid origin, spacing ``h``).  Returns
    ``(verts, faces)`` or ``None`` when there is no level crossing."""
    from skimage import measure
    vmin, vmax = float(vol.min()), float(vol.max())
    if not (vmin < level < vmax):
        return None
    try:
        verts, faces, _n, _v = measure.marching_cubes(vol, level=level,
                                                       spacing=(h, h, h))
    except (ValueError, RuntimeError):
        return None
    return verts + np.asarray(origin, dtype=verts.dtype), faces


def save_isosurface(solver, it, t, m, outdir):
    """Save a 3-D isosurface figure with axes fixed to the full domain:
    (left) the air/water interface ``alpha = 0.5`` (open cavity + splash crown)
    with the sphere inside; (right) a velocity-magnitude isosurface (the jet /
    wake) with the sphere.  The sphere is the SDF zero-level set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    inn = (slice(1, -1),) * 3                          # strip ghost layer
    alpha = solver.two_phase.alpha[inn].detach().float().cpu().numpy()
    u = solver.u0[inn].detach().float().cpu().numpy()
    v = solver.v0[inn].detach().float().cpu().numpy()
    w = solver.w0[inn].detach().float().cpu().numpy()
    speed = np.sqrt(u * u + v * v + w * w)
    sdf = solver.composite_body.sdf_val[inn].detach().float().cpu().numpy()
    h = float(m["h"])
    origin = (float(solver.x[1]), float(solver.y[1]), float(solver.z[1]))
    Lx, Ly, Lz, R, Hint, U = (m["Lx"], m["Ly"], m["Lz"], m["R"], m["Hint"], m["U"])
    zc = m["z0"] - U * t
    # adaptive: a shell just below the peak so the near-body flow / jet is
    # always captured (flow around the body ~ U, impact jet > U).
    spd_level = max(0.8 * U, 0.55 * float(speed.max()))

    iso_sph = _iso3d(sdf, 0.0, h, origin)
    panels = [("air/water interface + cavity", _iso3d(alpha, 0.5, h, origin),
               "#1f77b4", 0.18),
              (f"|u| = {spd_level:.1f} m/s isosurface", _iso3d(speed, spd_level, h, origin),
               "#ff7f0e", 0.35)]
    fig = plt.figure(figsize=(12, 6.4))
    for col, (title, iso_field, fc, op) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        if iso_field is not None:
            vF, fF = iso_field
            ax.add_collection3d(Poly3DCollection(vF[fF], alpha=op,
                                facecolor=fc, edgecolor="none"))
        if iso_sph is not None:
            vF, fF = iso_sph
            ax.add_collection3d(Poly3DCollection(vF[fF], alpha=0.9,
                                facecolor="#555555", edgecolor="none"))
        xx, yy = np.meshgrid([0, Lx], [0, Ly])         # rest waterline plane
        ax.plot_surface(xx, yy, np.full_like(xx, Hint), alpha=0.06, color="#1f77b4")
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
        ax.set_box_aspect((Lx, Ly, Lz))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=12, azim=-60)
    fig.suptitle(f"3-D sphere water-entry  it={it}  t={t:.4f}s  "
                 f"z_c/R below surf = {(Hint-zc)/R:+.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"iso_{it:05d}.png"), dpi=95)
    plt.close(fig)


def main(pts_per_D=16, Fr=3.0, Re=500.0, save_every=25,
         penetration_target=15.0):
    pars, m = build_pars(pts_per_D=pts_per_D, Fr=Fr, Re=Re)
    outdir = pars["output"]["save_path"]
    os.makedirs(outdir, exist_ok=True)
    dt, U, R, Hint, z0 = m["dt"], m["U"], m["R"], m["Hint"], m["z0"]

    # run until the sphere centre reaches penetration_target * R below the
    # interface.
    z_target = Hint - penetration_target * R
    n_by_depth = int((z0 - z_target) / U / dt) + 1
    n_steps = min(pars["solver"]["nt"], n_by_depth)

    print("=" * 74)
    print("  3-D sphere WATER-ENTRY  (TWO-PHASE VOF + BDIM, prescribed entry)")
    print(f"  D={m['D']}  grid {m['Nx']}x{m['Ny']}x{m['Nz']} "
          f"({pts_per_D} pts/D, {m['Nx']*m['Ny']*m['Nz']/1e6:.2f}M cells)  h={m['h']:.4e}")
    print(f"  Fr={Fr}  ->  U={U:.3f} m/s   |   Re={Re:.0f}  ->  nu={m['nu']:.3e}")
    print(f"  interface z={Hint:.3f}, sphere start z0={z0:.3f}, dt={dt:.3e}")
    print(f"  {n_steps} steps  (~{penetration_target:.0f}R penetration below surface)")
    print(f"  frames -> {outdir}")
    print("=" * 74)

    solver = TwoPhaseSolver(pars, dtype=torch.float32, compute_forces=True)
    solver.inside = lambda *a, **k: True

    # --- expose per-body SDFs for the eulerian 3-D force integrator ----------
    # The 3-D per-body force loop reads each body's own SDF from
    # ``comp._sdf_sparse`` (mesh / kernel / BDIMhandler bodies) or the legacy
    # dense ``comp.sdf_vals`` stack.  A bare analytical body
    # (``composite_analytical``) on the plain (non-kernel) eulerian path
    # populates NEITHER -- it keeps only the streaming UNION sdf plus each
    # sub-body's current ``body.sdf_val``.  Rather than patch core ``forces.py``,
    # we expose a fresh per-body ``sdf_vals`` list each step here, in the
    # example, by wrapping the composite-body update.  (For a single sphere the
    # union sdf IS the body's sdf, so this is just ``[sdf_val]``.)
    cb = solver.composite_body
    _orig_update = cb.update
    def _update_with_sdf_vals(*a, **k):
        _orig_update(*a, **k)
        cb.sdf_vals = [b.sdf_val for b in cb.bodies]
    cb.update = _update_with_sdf_vals

    solver.set_initial_conditions()
    u, v, w, p = solver.u0, solver.v0, solver.w0, solver.p0
    V0 = solver.two_phase.water_volume()

    Fz_max = 0.0; umax_max = 0.0; blew = False; last = {}
    save_slice(solver, 0, 0.0, m, outdir)
    save_isosurface(solver, 0, 0.0, m, outdir)
    for it in tqdm(range(n_steps)):
        try:
            u, v, p, w = solver.advance_and_compute_loads(u, v, p, it, it * dt, w_vel=w)
            solver.finalize_step(u, v, p, it, w_vel=w)
        except RuntimeError as e:
            print(f"  BLEW it={it}: {str(e)[:60]}"); blew = True; break
        umax = max(u.abs().max().item(), v.abs().max().item(), w.abs().max().item())
        umax_max = max(umax_max, umax)
        if (it + 1) % save_every == 0 or it == n_steps - 1:
            save_slice(solver, it + 1, (it + 1) * dt, m, outdir)
            save_isosurface(solver, it + 1, (it + 1) * dt, m, outdir)
        if (it + 1) % 25 == 0 or it == n_steps - 1:
            zc = z0 - U * (it + 1) * dt
            Fz = float(solver.get_loads()[0][0, 2])
            Fz_max = max(Fz_max, Fz)
            drift = (solver.two_phase.water_volume() - V0) / V0
            last = dict(zc=zc, Fz=Fz, umax=umax, drift=drift)
            tqdm.write(f"  it={it+1:4d} z_c={zc:6.3f} (pen {(Hint-zc)/R:+5.2f}R)  "
                       f"F_z={Fz:+9.3f}  |u|max={umax:.3e}  vol_drift={drift:+.2e}")

    print("\nFinal diagnostics:")
    print(f"  sphere centre z = {last.get('zc', 0):.3f}  "
          f"(interface {Hint:.3f}; penetration {(Hint-last.get('zc',Hint))/R:+.2f} R)")
    print(f"  peak |u|max     = {umax_max:.3e}  (entry U={U:.3f}, sqrt(gD)={m['c']:.3f})")
    print(f"  peak F_z        = {Fz_max:+.3f} N  (rises as wetted volume grows)")
    print(f"  vol drift       = {last.get('drift', 0):+.3e}")

    ok_stable = (not blew) and umax_max < 20.0 * U      # jet ~ few x U; bounded
    ok_mass   = abs(last.get("drift", 1)) < 5e-3
    ok_force  = Fz_max > 0.0                             # net upward load on entry
    print(f"\n  stable OK : {ok_stable}\n  mass OK   : {ok_mass}\n"
          f"  force OK  : {ok_force}")
    print("  ===> 3-D DROP-SPHERE WATER-ENTRY "
          + ("PASSED" if (ok_stable and ok_mass and ok_force) else "NEEDS REVIEW"))
    return ok_stable and ok_mass and ok_force


if __name__ == "__main__":
    main()
