"""
Full closed-loop CPU vs GPU parity test for the Gazzola sphere drop.

Exactly mirrors the controller.py + FARMS interaction, replacing MuJoCo
with a simple Euler integrator for the sphere dynamics:

    m * a = F_friction + F_pressure + F_gravity + F_buoyancy

This creates the closed feedback loop:
    fluid forces  →  body acceleration  →  body position
        ↑                                       |
        └────── SDF update  ←──────────────────┘

Run on a machine with the full lilytorch env (pytorch_interpolation, etc.):
    python run_gazzola_cpu_gpu_comparison.py

To test the full-resolution grid set Nx=256, Ny=2048 and N_STEPS=50.
"""

import sys
import os

# ── Gracefully mock farms_core if not installed (only needed for mesh bodies) ──
try:
    import farms_core  # noqa: F401
except ImportError:
    from unittest.mock import MagicMock
    for _m in ['farms_core', 'farms_core.io', 'farms_core.io.sdf']:
        sys.modules[_m] = MagicMock()

import matplotlib
matplotlib.use('Agg')  # headless

import torch
import numpy as np

# ── point at local source ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lilytorch.src.solver import FluidSolver

# ═══════════════════════════════════════════════════════════════════════════
# Parameters  ── match simulation_config.yaml exactly
# ═══════════════════════════════════════════════════════════════════════════
Nx, Ny      = 64, 512           # quick test; set 256, 2048 for full resolution
xmin, xmax  = -0.02, 0.02
ymin, ymax  = 0.0, 0.32
dt_val      = 0.001             # s
N_STEPS     = 40                # fluid steps to compare
ALARM_TOL   = 1e-5              # L-inf diff threshold for alarm
dtype       = torch.float32

# Sphere physical properties (from controller.py)
radius      = 0.0025            # m
rho_body    = 1010.0            # kg/m³
rho_fluid   = 996.0             # kg/m³
gravity     = -9.81             # m/s²   (MuJoCo convention: z is up → negative)
sphere_mass = np.pi * radius**2 * rho_body          # 2-D mass [kg/m]
sphere_inertia = 0.5 * sphere_mass * radius**2      # rotational inertia
BRINKMANN_K = 1.0e5

# ═══════════════════════════════════════════════════════════════════════════
# FluidSolver parameter dict
# ═══════════════════════════════════════════════════════════════════════════
def make_pars(use_gpu: bool) -> dict:
    dx = (xmax - xmin) / Nx
    return {
        "solver": {
            "use_gpu":                  use_gpu,
            "nthreads":                 4,
            "Nx":                       Nx,
            "Ny":                       Ny,
            "xmin":                     xmin,
            "xmax":                     xmax,
            "ymin":                     ymin,
            "ymax":                     ymax,
            "convection_method":        "abdquickest",
            "dt":                       dt_val,
            "nt":                       N_STEPS + 10,   # keep drag_record small
            "nu":                       8.0e-7,
            "rho":                      rho_fluid,
            "poisson_tol":              1.0e-4,
            "poisson_max_cycles":       3,
            "poisson_max_mgcg_cycles":  3,
            "jacobi_weight":            0.7,
            "poisson_nsmoothing":       5,
            "poisson_verbose":          False,
            "poisson_folder":           "/tmp/",
            "starting_iteration":       0,
            "starting_iteration_path":  None,
        },
        "boundary_conditions": {
            "BC_type_u":   ["N", "N", "N", "N"],
            "BC_values_u": [0, 0, 0, 0],
            "BC_type_v":   ["N", "N", "N", "N"],
            "BC_values_v": [0, 0, 0, 0],
        },
        "output": {
            "save_path":   "/tmp/lilytorch_test/",
            "save_frames": False,
            "save_every":  99999,
            "vmin":        -1,
            "vmax":        1,
            "save_uv":     False,
        },
        # composite_analytical = one circle SDF, initially at y=0.3
        # The update lambdas depend on t so autograd can compute velocities.
        # Actual position is overridden each step in our update() below.
        "body": {
            "type":     "composite_analytical",
            "sdf":      [f"lambda x,y: torch.sqrt(x**2+y**2)-{radius}"],
            "plotting": False,
            "update_maps": [
                {
                    "rotation":    "lambda t: t*0.0",
                    "translation": [
                        "lambda t: t*0.0",
                        f"lambda t: t*0.0+{0.3}",
                    ],
                }
            ],
            "eps":              2 * dx,
            "suit":             0.0,
            "convexify":        False,
            "scale":            1,
            "save_folder":      None,
            "n_samples":        [100, 100],
            "compute_interp":   False,
            "plotting_meshes":  False,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Body-state update  ── mirrors controller.py update() exactly
# Sets SDF, body_u/v, sdf_vals_u/v on the FluidSolver's composite body
# ═══════════════════════════════════════════════════════════════════════════
def update_body_state(fs: FluidSolver, pos_x: float, pos_y: float,
                      vel_x: float, vel_y: float, ang_vel: float = 0.0):
    """
    Replicates the manual SDF+velocity update from controller.py.
    The rotation matrix R = I (no rotation for a sphere), so
    pos_trans = stacked_xy - urdf_pos.
    """
    comp   = fs.composite_body
    body   = comp.bodies[0]
    device = fs.device
    nx, ny = fs.nx, fs.ny

    pos_tensor = torch.tensor([pos_x, pos_y], dtype=dtype, device=device)

    # ── SDF at cell-centre nodes ──────────────────────────────────────────
    ex_cc = comp.stacked_xy[0].reshape(nx, ny) - pos_x
    ey_cc = comp.stacked_xy[1].reshape(nx, ny) - pos_y
    sdf_cc = body.sdf(ex_cc, ey_cc)
    comp.sdf_vals[0] = sdf_cc

    # ── SDF at u-staggered nodes ──────────────────────────────────────────
    ex_us = comp.stacked_xy_u[0].reshape(nx, ny) - pos_x
    ey_us = comp.stacked_xy_u[1].reshape(nx, ny) - pos_y
    sdf_us = body.sdf(ex_us, ey_us)
    comp.sdf_vals_u[0] = sdf_us

    # ── SDF at v-staggered nodes ──────────────────────────────────────────
    ex_vs = comp.stacked_xy_v[0].reshape(nx, ny) - pos_x
    ey_vs = comp.stacked_xy_v[1].reshape(nx, ny) - pos_y
    sdf_vs = body.sdf(ex_vs, ey_vs)
    comp.sdf_vals_v[0] = sdf_vs

    # ── Body velocity fields  (v = v_lin + ω × r)  ───────────────────────
    # y-coords of u-staggered points and x-coords of v-staggered points
    comp.u_vals[0] = (vel_x - ang_vel * (comp.stacked_xy_u[1].reshape(nx, ny) - pos_y))
    comp.v_vals[0] = (vel_y + ang_vel * (comp.stacked_xy_v[0].reshape(nx, ny) - pos_x))

    # ── Reduce over bodies (only one body here) ───────────────────────────
    idx = comp.sdf_vals.argmin(0).unsqueeze(0).expand(comp.sdf_vals.shape)
    comp.sdf_val = comp.sdf_vals.gather(0, idx)[0].reshape(nx, ny)

    idx_u = comp.sdf_vals_u.argmin(0).unsqueeze(0).expand(comp.sdf_vals_u.shape)
    comp.sdf_val_u = comp.sdf_vals_u.gather(0, idx_u)[0].reshape(nx, ny)
    comp.body_u    = comp.u_vals.gather(0, idx_u)[0].reshape(nx, ny)

    idx_v = comp.sdf_vals_v.argmin(0).unsqueeze(0).expand(comp.sdf_vals_v.shape)
    comp.sdf_val_v = comp.sdf_vals_v.gather(0, idx_v)[0].reshape(nx, ny)
    comp.body_v    = comp.v_vals.gather(0, idx_v)[0].reshape(nx, ny)

    # ── Store com_pos (used in forces_method2 moment arm) ────────────────
    comp.com_pos[0] = pos_tensor
    body.com_pos    = pos_tensor

    # ── Update contour for moment arm (body.r_com) ────────────────────────
    body.cnt_update = body.cnt + pos_tensor[:, None]
    body.r_com      = body.cnt_update - pos_tensor[:, None]

    # ── mu_funcs and normals (mirrors controller.py step()) ───────────────
    (fs.mu0_all,   fs.mu1_all)   = comp.mu_funcs(comp.sdf_val)
    fs.m_m0_all                  = 1.0 - fs.mu0_all
    (_, fs.normal_x, fs.normal_y, _) = comp.compute_sdf_properties(comp.sdf_val)

    (fs.mu0_all_u, fs.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
    fs.m_m0_all_u                = 1.0 - fs.mu0_all_u
    (_, fs.normal_x_u, fs.normal_y_u, _) = comp.compute_sdf_properties(comp.sdf_val_u)

    (fs.mu0_all_v, fs.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
    fs.m_m0_all_v                = 1.0 - fs.mu0_all_v
    (_, fs.normal_x_v, fs.normal_y_v, _) = comp.compute_sdf_properties(comp.sdf_val_v)

    # Override solver density (constant fluid density — no hydrostatic term)
    fs.rho = torch.tensor(rho_fluid, dtype=dtype, device=device)


# ═══════════════════════════════════════════════════════════════════════════
# fluid_step  ── exact copy of controller.py fluid_step
# ═══════════════════════════════════════════════════════════════════════════
def fluid_step(fs: FluidSolver, u, v, p, timestep: float):
    ts = torch.tensor(timestep, dtype=dtype, device=fs.device)

    # 1. Advection-diffusion
    uprime, vprime = fs.adv_diff_solver.solve(u, v)

    # 2. Boundary conditions (from controller.py set_BC)
    for i in [1, -1]:
        uprime[i, :] = 0;  uprime[:, i] = 0
        vprime[i, :] = 0;  vprime[:, i] = 0

    # 3. Brinkmann penalization
    bk = BRINKMANN_K * ts
    uprime = (uprime + bk * fs.m_m0_all_u * fs.composite_body.body_u) / \
             (1.0   + bk * fs.m_m0_all_u)
    vprime = (vprime + bk * fs.m_m0_all_v * fs.composite_body.body_v) / \
             (1.0   + bk * fs.m_m0_all_v)

    # 4. Divergence
    fs.div = fs.divergence(uprime, vprime)

    # 5. Poisson solve
    c     = torch.ones_like(u)
    coeff = ts / fs.rho
    ch    = ts * fs.mu0_all_u / fs.rho
    cv    = ts * fs.mu0_all_v / fs.rho
    p, _  = fs.poisson_solver.solve_multigrid(
        fs.div[1:-1, 1:-1],
        p,
        coeff * c,
        ch=ch[1:, 1:-1],
        cv=cv[1:-1, 1:],
    )

    # 6. Pressure projection
    p_x, p_y = fs.gradient(p)
    u_new = uprime - ch * p_x
    v_new = vprime - cv * p_y

    return u_new, v_new, p


# ═══════════════════════════════════════════════════════════════════════════
# Euler body dynamics  ── replaces MuJoCo
# Inputs are scalar forces in N (per unit depth in 2-D)
# ═══════════════════════════════════════════════════════════════════════════
def euler_body_step(pos_x, pos_y, vel_x, vel_y,
                    fx, fy, dt):
    """
    Simple forward-Euler integrator for a free sphere in 2-D.
    Gravity and buoyancy are added here (same formula as controller.py).
    """
    # net external force = fluid + gravity + buoyancy
    Fg   = sphere_mass * gravity              # gravity (negative = downward)
    Fb   = -rho_fluid * np.pi * radius**2 * gravity  # buoyancy (upward)
    ax   = fx / sphere_mass
    ay   = (fy + Fg + Fb) / sphere_mass

    new_vel_x = vel_x + ax * dt
    new_vel_y = vel_y + ay * dt
    new_pos_x = pos_x + vel_x * dt
    new_pos_y = pos_y + vel_y * dt

    return new_pos_x, new_pos_y, new_vel_x, new_vel_y


# ═══════════════════════════════════════════════════════════════════════════
# Per-step substep diagnostics (called on ALARM)
# ═══════════════════════════════════════════════════════════════════════════
def substep_diagnostics(fs_cpu, fs_gpu, u_cpu, v_cpu, p_cpu, u_gpu, v_gpu, p_gpu):
    print("\n  ── Sub-step diagnostics ──")

    # Advection-diffusion
    up_c, vp_c = fs_cpu.adv_diff_solver.solve(u_cpu, v_cpu)
    up_g, vp_g = fs_gpu.adv_diff_solver.solve(u_gpu, v_gpu)
    print(f"    1. adv-diff:   |Δu'|={torch.max(torch.abs(up_c - up_g.cpu())):.3e}"
          f"  |Δv'|={torch.max(torch.abs(vp_c - vp_g.cpu())):.3e}")

    # BCs
    for i in [1, -1]:
        up_c[i,:]=0; up_c[:,i]=0; vp_c[i,:]=0; vp_c[:,i]=0
        up_g[i,:]=0; up_g[:,i]=0; vp_g[i,:]=0; vp_g[:,i]=0

    # Brinkmann
    bk = BRINKMANN_K * dt_val
    up_c2 = (up_c + bk*fs_cpu.m_m0_all_u*fs_cpu.composite_body.body_u)/(1+bk*fs_cpu.m_m0_all_u)
    vp_c2 = (vp_c + bk*fs_cpu.m_m0_all_v*fs_cpu.composite_body.body_v)/(1+bk*fs_cpu.m_m0_all_v)
    up_g2 = (up_g + bk*fs_gpu.m_m0_all_u*fs_gpu.composite_body.body_u)/(1+bk*fs_gpu.m_m0_all_u)
    vp_g2 = (vp_g + bk*fs_gpu.m_m0_all_v*fs_gpu.composite_body.body_v)/(1+bk*fs_gpu.m_m0_all_v)
    print(f"    2. Brinkmann:  |Δu'|={torch.max(torch.abs(up_c2 - up_g2.cpu())):.3e}"
          f"  |Δv'|={torch.max(torch.abs(vp_c2 - vp_g2.cpu())):.3e}")

    # divergence
    div_c = fs_cpu.divergence(up_c2, vp_c2)
    div_g = fs_gpu.divergence(up_g2, vp_g2)
    print(f"    3. divergence: |Δdiv|={torch.max(torch.abs(div_c - div_g.cpu())):.3e}")

    # Poisson
    rho_c = torch.tensor(rho_fluid, dtype=dtype, device=fs_cpu.device)
    rho_g = torch.tensor(rho_fluid, dtype=dtype, device=fs_gpu.device)
    ts    = torch.tensor(dt_val, dtype=dtype)
    cc    = torch.ones_like(u_cpu)
    ch_c  = ts.to(fs_cpu.device) * fs_cpu.mu0_all_u / rho_c
    cv_c  = ts.to(fs_cpu.device) * fs_cpu.mu0_all_v / rho_c
    ch_g  = ts.to(fs_gpu.device) * fs_gpu.mu0_all_u / rho_g
    cv_g  = ts.to(fs_gpu.device) * fs_gpu.mu0_all_v / rho_g
    p_c2, _ = fs_cpu.poisson_solver.solve_multigrid(
        div_c[1:-1,1:-1], p_cpu,
        (ts.to(fs_cpu.device)/rho_c)*cc, ch=ch_c[1:,1:-1], cv=cv_c[1:-1,1:])
    p_g2, _ = fs_gpu.poisson_solver.solve_multigrid(
        div_g[1:-1,1:-1], p_gpu,
        (ts.to(fs_gpu.device)/rho_g)*cc.to(fs_gpu.device),
        ch=ch_g[1:,1:-1], cv=cv_g[1:-1,1:])
    print(f"    4. Poisson:    |Δp|={torch.max(torch.abs(p_c2 - p_g2.cpu())):.3e}")

    # forces
    fs_cpu.forces_method2(up_c2, vp_c2, p_c2, 0)
    fs_gpu.forces_method2(up_g2, vp_g2, p_g2, 0)
    print(f"    5a. Ffric_x:   CPU={float(fs_cpu.friction_force_lin_x[0]):.4e}  "
          f"GPU={float(fs_gpu.friction_force_lin_x[0]):.4e}  "
          f"diff={abs(float(fs_cpu.friction_force_lin_x[0])-float(fs_gpu.friction_force_lin_x[0].cpu())):.3e}")
    print(f"    5b. Ffric_y:   CPU={float(fs_cpu.friction_force_lin_y[0]):.4e}  "
          f"GPU={float(fs_gpu.friction_force_lin_y[0]):.4e}  "
          f"diff={abs(float(fs_cpu.friction_force_lin_y[0])-float(fs_gpu.friction_force_lin_y[0].cpu())):.3e}")
    print(f"    5c. Fpres_x:   CPU={float(fs_cpu.pressure_force_x[0]):.4e}  "
          f"GPU={float(fs_gpu.pressure_force_x[0]):.4e}  "
          f"diff={abs(float(fs_cpu.pressure_force_x[0])-float(fs_gpu.pressure_force_x[0].cpu())):.3e}")
    print(f"    5d. Fpres_y:   CPU={float(fs_cpu.pressure_force_y[0]):.4e}  "
          f"GPU={float(fs_gpu.pressure_force_y[0]):.4e}  "
          f"diff={abs(float(fs_cpu.pressure_force_y[0])-float(fs_gpu.pressure_force_y[0].cpu())):.3e}")

    # SDF / mu check
    sdf_diff = torch.max(torch.abs(fs_cpu.composite_body.sdf_val -
                                    fs_gpu.composite_body.sdf_val.cpu()))
    mu0_diff = torch.max(torch.abs(fs_cpu.mu0_all - fs_gpu.mu0_all.cpu()))
    print(f"    body SDF diff: {sdf_diff:.3e}   mu0 diff: {mu0_diff:.3e}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("Gazzola sphere drop — full closed-loop CPU vs GPU parity test")
    print("  (Euler body dynamics replace MuJoCo; fluid_step = controller.py)")
    print("=" * 72)
    print(f"Grid: {Nx}×{Ny}  dtype: {dtype}  N_STEPS: {N_STEPS}  ALARM_TOL: {ALARM_TOL:.0e}")
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.  Run on a machine with a GPU.")
        return

    # ── Initialise both solvers ──────────────────────────────────────────
    print("Initialising CPU solver …")
    fs_cpu = FluidSolver(make_pars(use_gpu=False), dtype=dtype,
                         costum_update=True, compute_forces=True)
    print("Initialising GPU solver …")
    fs_gpu = FluidSolver(make_pars(use_gpu=True),  dtype=dtype,
                         costum_update=True, compute_forces=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Override the composite body update (like controller.py) ─────────
    # Nothing to do — we call update_body_state() manually each step.

    # ── Initial state ────────────────────────────────────────────────────
    pos_x_cpu, pos_y_cpu = 0.0, 0.3
    vel_x_cpu, vel_y_cpu = 0.0, 0.0
    pos_x_gpu, pos_y_gpu = 0.0, 0.3
    vel_x_gpu, vel_y_gpu = 0.0, 0.0

    u_cpu, v_cpu, p_cpu = fs_cpu.u0.clone(), fs_cpu.v0.clone(), fs_cpu.p0.clone()
    u_gpu, v_gpu, p_gpu = fs_gpu.u0.clone(), fs_gpu.v0.clone(), fs_gpu.p0.clone()

    # ── Header ──────────────────────────────────────────────────────────
    fmt = ("{:>4s}  {:>9s}  {:>9s}  {:>9s}  {:>9s}  "
           "{:>9s}  {:>9s}  {:>9s}  {:>9s}  {:>7s}")
    hdr = fmt.format("step",
                     "|Δu|∞", "|Δv|∞", "|Δp|∞", "|Δpos_y|",
                     "|ΔFfx|", "|ΔFfy|", "|ΔFpx|", "|ΔFpy|",
                     "status")
    print(hdr)
    print("─" * len(hdr))

    alarm = False
    for step in range(N_STEPS):

        # ── Update body state on each solver ────────────────────────────
        update_body_state(fs_cpu, pos_x_cpu, pos_y_cpu, vel_x_cpu, vel_y_cpu)
        update_body_state(fs_gpu, pos_x_gpu, pos_y_gpu, vel_x_gpu, vel_y_gpu)

        # ── One fluid step ───────────────────────────────────────────────
        u_cpu, v_cpu, p_cpu = fluid_step(fs_cpu, u_cpu, v_cpu, p_cpu, dt_val)
        u_gpu, v_gpu, p_gpu = fluid_step(fs_gpu, u_gpu, v_gpu, p_gpu, dt_val)

        # ── Compute forces via forces_method2 (exact controller path) ───
        fs_cpu.forces_method2(u_cpu, v_cpu, p_cpu, step)
        fs_gpu.forces_method2(u_gpu, v_gpu, p_gpu, step)

        fx_cpu = float(fs_cpu.friction_force_lin_x[0] + fs_cpu.pressure_force_x[0])
        fy_cpu = float(fs_cpu.friction_force_lin_y[0] + fs_cpu.pressure_force_y[0])
        fx_gpu = float((fs_gpu.friction_force_lin_x[0] + fs_gpu.pressure_force_x[0]).cpu())
        fy_gpu = float((fs_gpu.friction_force_lin_y[0] + fs_gpu.pressure_force_y[0]).cpu())

        ffx_cpu = float(fs_cpu.friction_force_lin_x[0]);  ffx_gpu = float(fs_gpu.friction_force_lin_x[0].cpu())
        ffy_cpu = float(fs_cpu.friction_force_lin_y[0]);  ffy_gpu = float(fs_gpu.friction_force_lin_y[0].cpu())
        fpx_cpu = float(fs_cpu.pressure_force_x[0]);      fpx_gpu = float(fs_gpu.pressure_force_x[0].cpu())
        fpy_cpu = float(fs_cpu.pressure_force_y[0]);      fpy_gpu = float(fs_gpu.pressure_force_y[0].cpu())

        # ── Euler body integration ───────────────────────────────────────
        pos_x_cpu, pos_y_cpu, vel_x_cpu, vel_y_cpu = euler_body_step(
            pos_x_cpu, pos_y_cpu, vel_x_cpu, vel_y_cpu, fx_cpu, fy_cpu, dt_val)
        pos_x_gpu, pos_y_gpu, vel_x_gpu, vel_y_gpu = euler_body_step(
            pos_x_gpu, pos_y_gpu, vel_x_gpu, vel_y_gpu, fx_gpu, fy_gpu, dt_val)

        # ── Differences ─────────────────────────────────────────────────
        diff_u    = torch.max(torch.abs(u_cpu - u_gpu.cpu())).item()
        diff_v    = torch.max(torch.abs(v_cpu - v_gpu.cpu())).item()
        diff_p    = torch.max(torch.abs(p_cpu - p_gpu.cpu())).item()
        diff_py   = abs(pos_y_cpu - pos_y_gpu)
        diff_ffx  = abs(ffx_cpu - ffx_gpu)
        diff_ffy  = abs(ffy_cpu - ffy_gpu)
        diff_fpx  = abs(fpx_cpu - fpx_gpu)
        diff_fpy  = abs(fpy_cpu - fpy_gpu)

        step_alarm = any(d > ALARM_TOL for d in [diff_u, diff_v, diff_p])
        if step_alarm:
            alarm = True

        status = "ALARM" if step_alarm else "ok"
        print(fmt.format(
            str(step + 1),
            f"{diff_u:.3e}", f"{diff_v:.3e}", f"{diff_p:.3e}",
            f"{diff_py:.3e}",
            f"{diff_ffx:.3e}", f"{diff_ffy:.3e}",
            f"{diff_fpx:.3e}", f"{diff_fpy:.3e}",
            status,
        ))

        if step_alarm:
            print(f"\n  CPU pos_y={pos_y_cpu:.8f}  GPU pos_y={pos_y_gpu:.8f}")
            print(f"  CPU vy   ={vel_y_cpu:.8f}  GPU vy   ={vel_y_gpu:.8f}")
            substep_diagnostics(fs_cpu, fs_gpu,
                                u_cpu, v_cpu, p_cpu, u_gpu, v_gpu, p_gpu)
            print("\n  Stopping after first alarm step for diagnosis.")
            break

    print()
    if not alarm:
        print(f"✓ PASS  —  all {N_STEPS} steps within alarm tolerance {ALARM_TOL:.0e}")
        print(f"  Final CPU pos_y = {pos_y_cpu:.8f} m")
        print(f"  Final GPU pos_y = {pos_y_gpu:.8f} m")
        print(f"  Trajectory diff = {abs(pos_y_cpu - pos_y_gpu):.3e} m")
    else:
        print("✗ FAIL  —  divergence detected; see ALARM rows above.")

    print()
    print("Final fluid field norms:")
    print(f"  ||u||∞  CPU={u_cpu.abs().max():.5e}  GPU={u_gpu.abs().max():.5e}")
    print(f"  ||v||∞  CPU={v_cpu.abs().max():.5e}  GPU={v_gpu.abs().max():.5e}")
    print(f"  ||p||∞  CPU={p_cpu.abs().max():.5e}  GPU={p_gpu.abs().max():.5e}")


if __name__ == "__main__":
    main()
