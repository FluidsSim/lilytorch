"""
Standalone CPU vs GPU parity test for the Gazzola sphere drop fluid solver.

Usage:
    python run_gazzola_cpu_gpu_comparison.py

Requires: torch (with CUDA), numpy, scipy, scikit-image, scikit-fmm,
          pytorch_interpolation, cv2, open3d.
Does NOT require: farms_core, farms_mujoco (mocked internally).

What it does:
  1. Builds the grid (matching simulation_config.yaml: 256×2048).
  2. Creates a BodyAnalytical (circle SDF) for the sphere at (0, 0.3).
  3. Runs N_STEPS of the exact fluid_step from controller.py on both CPU and GPU.
  4. After every step records L-inf differences in u, v, p, divergence, and
     pressure forces.  Prints a table.  Stops early on ALARM.
  5. If divergence is found, prints diagnostics to identify which sub-step
     first introduces the difference.
"""

import sys
import os

# ---- Mock unavailable optional deps (farms_core is only needed for mesh bodies)
from unittest.mock import MagicMock
for _mod in ['farms_core', 'farms_core.io', 'farms_core.io.sdf']:
    sys.modules[_mod] = MagicMock()

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lilytorch.src.adv_diff   import AdvDiffSolver
from lilytorch.src.poisson_mult import PoissonSolver
from lilytorch.src.body        import BodyAnalytical

# ---------------------------------------------------------------------------
# Parameters  (matching simulation_config.yaml)
# ---------------------------------------------------------------------------
Nx, Ny      = 256, 2048          # full grid; reduce to 128,1024 for speed
xmin, xmax  = -0.02, 0.02
ymin, ymax  = 0.0, 0.32
dt_val      = 0.001
nu_val      = 8.0e-7
rho_val     = 996.0
radius      = 0.0025
N_STEPS     = 30
ALARM_TOL   = 1e-5               # L-inf tolerance for alarm
dtype       = torch.float32
BRINKMANN_K = 1.0e5

# Free-fall kinematics for the sphere (open-loop, same value on CPU and GPU)
gravity    = -9.81
rho_body   = 1010.0
g_eff      = gravity * (1.0 - rho_val / rho_body)
sphere_x0, sphere_y0   = 0.0, 0.3
sphere_vy0 = 0.0


# ---------------------------------------------------------------------------
# Helper: build all solver components for one device
# ---------------------------------------------------------------------------
def build_solver(device_str):
    device = torch.device(device_str)
    dx     = (xmax - xmin) / Nx
    dy     = (ymax - ymin) / Ny
    assert abs(dx - dy) < 1e-12, "Non-square grid!"

    x = torch.linspace(xmin - dx/2, xmax + dx/2, Nx+2, dtype=dtype, device=device)
    y = torch.linspace(ymin - dx/2, ymax + dx/2, Ny+2, dtype=dtype, device=device)

    h  = torch.tensor(dx, dtype=dtype, device=device)
    dt = torch.tensor(dt_val, dtype=dtype, device=device)
    nu = torch.tensor(nu_val, dtype=dtype, device=device)

    adv  = AdvDiffSolver(
        device, dt, x, y, nu,
        BC_type_u  = ["N","N","N","N"], BC_values_u = [0,0,0,0],
        BC_type_v  = ["N","N","N","N"], BC_values_v = [0,0,0,0],
        method     = "abdquickest",
    )

    poisson = PoissonSolver(
        dtype, device, h,
        tol         = 1e-4,
        max_cycles  = 3,
        max_vcycles = 3,
        nsmoothing  = 5,
        w           = 0.7,
        verbose     = False,
    )

    # Circle SDF: sqrt(x^2 + y^2) - radius
    sdf_fun = lambda px, py: torch.sqrt(px**2 + py**2) - radius

    # The update lambdas must depend on t for autograd to compute velocities.
    # Velocity = d(translation)/dt.  For a stationary sphere at (0, 0.3):
    #   x(t) = 0.0 * t  → vx = 0
    #   y(t) = 0.3 + 0.0 * t  → vy = 0
    # The body position is overridden each step in update_body(),
    # so the initial position from these lambdas is only used for
    # constructing the contour during __init__.
    body = BodyAnalytical(
        device, x, y,
        sdf_fun,
        update_maps=(
            lambda t: t * 0.0,                              # rotation (0 deg)
            (lambda t: t * 0.0,                             # translation x = 0
             lambda t: torch.tensor(0.3, dtype=dtype, device=device) + t * 0.0),   # y = 0.3
        ),
        plotting=False,
        eps=2*dx,
    )

    nx, ny = Nx+2, Ny+2
    u = torch.zeros((nx, ny), dtype=dtype, device=device)
    v = torch.zeros((nx, ny), dtype=dtype, device=device)
    p = torch.zeros((nx, ny), dtype=dtype, device=device)

    # Precomputed reciprocals (matching solver.py)
    inv_h  = h.reciprocal()
    inv_dx = h.reciprocal()
    inv_dy = h.reciprocal()
    inv_2h = (2.0 * h).reciprocal()
    h2     = h * h

    return dict(
        device=device, x=x, y=y, h=h, dt=dt, nu=nu,
        adv=adv, poisson=poisson, body=body,
        u=u, v=v, p=p,
        inv_h=inv_h, inv_dx=inv_dx, inv_dy=inv_dy, inv_2h=inv_2h, h2=h2,
        nx=nx, ny=ny,
    )


# ---------------------------------------------------------------------------
# Gradient operator (matches solver.py:gradient)
# ---------------------------------------------------------------------------
def gradient(p, inv_h, nx, ny):
    dx_p = torch.zeros_like(p)
    dy_p = torch.zeros_like(p)
    dx_p[1:-1, 1:-1] = (p[1:-1, 1:-1] - p[:-2, 1:-1]) * inv_h
    dy_p[1:-1, 1:-1] = (p[1:-1, 1:-1] - p[1:-1, :-2]) * inv_h
    return dx_p, dy_p


# ---------------------------------------------------------------------------
# Divergence operator (matches solver.py:divergence, staggered MAC grid)
# ---------------------------------------------------------------------------
def divergence(u, v, inv_dx, inv_dy, nx, ny):
    d = torch.zeros_like(u)
    d[1:-1, 1:-1] = (
        (u[2:, 1:-1] - u[1:-1, 1:-1]) * inv_dx +
        (v[1:-1, 2:] - v[1:-1, 1:-1]) * inv_dy
    )
    return d


# ---------------------------------------------------------------------------
# Boundary conditions (matches controller.py:set_BC)
# ---------------------------------------------------------------------------
def set_BC(u, v):
    for i in [1, -1]:
        u[i, :] = 0; u[:, i] = 0
        v[i, :] = 0; v[:, i] = 0


# ---------------------------------------------------------------------------
# Update body at sphere position pos=(px,py) with velocity vy_sphere
# Returns sdf_val, sdf_val_u, sdf_val_v, body_u, body_v, mu0, normal_x, normal_y
# ---------------------------------------------------------------------------
def update_body(sol, px, py, vy_sphere):
    body   = sol['body']
    device = sol['device']
    nx, ny = sol['nx'], sol['ny']
    dtype_ = dtype

    sdf_fun = lambda ex, ey: torch.sqrt(ex**2 + ey**2) - radius

    # cc nodes
    sx = body.stacked_xy[0].reshape(nx, ny) - px
    sy = body.stacked_xy[1].reshape(nx, ny) - py
    sdf_val = sdf_fun(sx, sy)

    # u-staggered nodes
    sxu = body.stacked_xy_u[0].reshape(nx, ny) - px
    syu = body.stacked_xy_u[1].reshape(nx, ny) - py
    sdf_val_u = sdf_fun(sxu, syu)

    # v-staggered nodes
    sxv = body.stacked_xy_v[0].reshape(nx, ny) - px
    syv = body.stacked_xy_v[1].reshape(nx, ny) - py
    sdf_val_v = sdf_fun(sxv, syv)

    # Body velocity (x=0, y=vy_sphere)
    body_u = torch.zeros((nx, ny), dtype=dtype_, device=device)
    body_v = torch.full((nx, ny), vy_sphere, dtype=dtype_, device=device)

    # Heaviside and normals
    (mu0, mu1) = body.mu_funcs(sdf_val)
    (_, nx_f, ny_f, _) = body.compute_sdf_properties(sdf_val)
    (mu0_u, _) = body.mu_funcs(sdf_val_u)
    (_, nx_u, ny_u, _) = body.compute_sdf_properties(sdf_val_u)
    (mu0_v, _) = body.mu_funcs(sdf_val_v)
    (_, nx_v, ny_v, _) = body.compute_sdf_properties(sdf_val_v)

    return dict(
        sdf_val=sdf_val, sdf_val_u=sdf_val_u, sdf_val_v=sdf_val_v,
        body_u=body_u, body_v=body_v,
        mu0=mu0, mu0_u=mu0_u, mu0_v=mu0_v,
        normal_x=nx_f, normal_y=ny_f,
        normal_x_u=nx_u, normal_y_u=ny_u,
        normal_x_v=nx_v, normal_y_v=ny_v,
    )


# ---------------------------------------------------------------------------
# One fluid step (mirrors controller.py:fluid_step exactly)
# ---------------------------------------------------------------------------
def fluid_step(sol, bstate, u, v, p):
    adv     = sol['adv']
    poisson = sol['poisson']
    h2      = sol['h2']
    inv_h   = sol['inv_h']
    inv_dx  = sol['inv_dx']
    inv_dy  = sol['inv_dy']
    nx, ny  = sol['nx'], sol['ny']
    dt      = sol['dt']
    rho     = torch.tensor(rho_val, dtype=dtype, device=sol['device'])

    mu0_u    = bstate['mu0_u']
    m_m0_u   = 1.0 - mu0_u
    mu0_v    = bstate['mu0_v']
    m_m0_v   = 1.0 - mu0_v
    body_u   = bstate['body_u']
    body_v   = bstate['body_v']

    # 1. Advection-diffusion
    uprime, vprime = adv.solve(u, v)

    # 2. BCs
    set_BC(uprime, vprime)

    # 3. Brinkmann penalization
    bk = BRINKMANN_K * float(dt)
    uprime = (uprime + bk * m_m0_u * body_u) / (1.0 + bk * m_m0_u)
    vprime = (vprime + bk * m_m0_v * body_v) / (1.0 + bk * m_m0_v)

    # 4. Divergence
    div = divergence(uprime, vprime, inv_dx, inv_dy, nx, ny)

    # 5. Poisson solve
    c     = torch.ones_like(u)
    coeff = dt / rho
    ch    = dt * mu0_u / rho
    cv    = dt * mu0_v / rho

    p, _ = poisson.solve_multigrid(
        div[1:-1, 1:-1],
        p,
        coeff * c,
        ch = ch[1:, 1:-1],
        cv = cv[1:-1, 1:],
    )

    # 6. Pressure projection
    p_x, p_y = gradient(p, inv_h, nx, ny)
    u_new = uprime - ch * p_x
    v_new = vprime - cv * p_y

    return u_new, v_new, p, div


# ---------------------------------------------------------------------------
# Pressure force integral (no interpolation needed)
# ---------------------------------------------------------------------------
def pressure_forces(p, sdf_val, normal_x, normal_y, h2):
    p_outer = torch.where(sdf_val < 0, torch.zeros_like(p), p)
    pforce_x = -p_outer * normal_x
    pforce_y = -p_outer * normal_y
    fx = pforce_x.to(torch.float64).sum().to(dtype) * h2
    fy = pforce_y.to(torch.float64).sum().to(dtype) * h2
    return fx, fy


# ---------------------------------------------------------------------------
# Per-substep diagnostic: find which substep first shows a difference
# ---------------------------------------------------------------------------
def substep_diagnostics(sol_cpu, sol_gpu, bstate_cpu, bstate_gpu, u_cpu, v_cpu, p_cpu, u_gpu, v_gpu, p_gpu):
    print("\n  [Sub-step diagnostics]")
    adv_cpu, adv_gpu = sol_cpu['adv'], sol_gpu['adv']

    # Step 1: advection-diffusion
    up_cpu, vp_cpu = adv_cpu.solve(u_cpu, v_cpu)
    up_gpu, vp_gpu = adv_gpu.solve(u_gpu, v_gpu)
    d_up = torch.max(torch.abs(up_cpu - up_gpu.cpu())).item()
    d_vp = torch.max(torch.abs(vp_cpu - vp_gpu.cpu())).item()
    print(f"    After adv-diff:       |Δu'|={d_up:.3e}  |Δv'|={d_vp:.3e}")

    # Step 2: BCs
    set_BC(up_cpu, vp_cpu); set_BC(up_gpu, vp_gpu)

    # Step 3: Brinkmann
    dt = float(sol_cpu['dt'])
    bk = BRINKMANN_K * dt
    up_cpu = (up_cpu + bk*(1-bstate_cpu['mu0_u'])*bstate_cpu['body_u']) / (1+bk*(1-bstate_cpu['mu0_u']))
    vp_cpu = (vp_cpu + bk*(1-bstate_cpu['mu0_v'])*bstate_cpu['body_v']) / (1+bk*(1-bstate_cpu['mu0_v']))
    up_gpu = (up_gpu + bk*(1-bstate_gpu['mu0_u'])*bstate_gpu['body_u']) / (1+bk*(1-bstate_gpu['mu0_u']))
    vp_gpu = (vp_gpu + bk*(1-bstate_gpu['mu0_v'])*bstate_gpu['body_v']) / (1+bk*(1-bstate_gpu['mu0_v']))
    d_up2 = torch.max(torch.abs(up_cpu - up_gpu.cpu())).item()
    d_vp2 = torch.max(torch.abs(vp_cpu - vp_gpu.cpu())).item()
    print(f"    After Brinkmann:      |Δu'|={d_up2:.3e}  |Δv'|={d_vp2:.3e}")

    # Step 4: divergence
    inv_dx, inv_dy = sol_cpu['inv_dx'], sol_cpu['inv_dy']
    nx, ny = sol_cpu['nx'], sol_cpu['ny']
    div_cpu = divergence(up_cpu, vp_cpu, inv_dx, inv_dy, nx, ny)
    div_gpu = divergence(up_gpu, vp_gpu, sol_gpu['inv_dx'], sol_gpu['inv_dy'], nx, ny)
    d_div = torch.max(torch.abs(div_cpu - div_gpu.cpu())).item()
    print(f"    After divergence:     |Δdiv|={d_div:.3e}")

    # Step 5: Poisson
    rho_t = torch.tensor(rho_val, dtype=dtype, device=sol_cpu['device'])
    dt_t  = sol_cpu['dt']
    c     = torch.ones_like(u_cpu)
    coeff = dt_t / rho_t
    ch_cpu = dt_t * bstate_cpu['mu0_u'] / rho_t
    cv_cpu = dt_t * bstate_cpu['mu0_v'] / rho_t
    ch_gpu = dt_t * bstate_gpu['mu0_u'] / sol_gpu['dt'].to(torch.device('cuda')) * rho_t  # recompute on gpu
    # simpler: use cpu solver with same div
    rho_g  = torch.tensor(rho_val, dtype=dtype, device=sol_gpu['device'])
    dt_g   = sol_gpu['dt']
    ch_g   = dt_g * bstate_gpu['mu0_u'] / rho_g
    cv_g   = dt_g * bstate_gpu['mu0_v'] / rho_g
    p_new_cpu, _ = sol_cpu['poisson'].solve_multigrid(
        div_cpu[1:-1,1:-1], p_cpu, coeff*c, ch=ch_cpu[1:,1:-1], cv=cv_cpu[1:-1,1:])
    p_new_gpu, _ = sol_gpu['poisson'].solve_multigrid(
        div_gpu[1:-1,1:-1], p_gpu, coeff.to(sol_gpu['device'])*c.to(sol_gpu['device']),
        ch=ch_g[1:,1:-1], cv=cv_g[1:-1,1:])
    d_p = torch.max(torch.abs(p_new_cpu - p_new_gpu.cpu())).item()
    print(f"    After Poisson:        |Δp|={d_p:.3e}")

    # Step 6: projection
    inv_h = sol_cpu['inv_h']
    px_cpu, py_cpu = gradient(p_new_cpu, inv_h, nx, ny)
    px_gpu, py_gpu = gradient(p_new_gpu, sol_gpu['inv_h'], nx, ny)
    u_proj_cpu = up_cpu - ch_cpu * px_cpu
    v_proj_cpu = vp_cpu - cv_cpu * py_cpu
    u_proj_gpu = up_gpu - ch_g   * px_gpu
    v_proj_gpu = vp_gpu - cv_g   * py_gpu
    d_u_proj = torch.max(torch.abs(u_proj_cpu - u_proj_gpu.cpu())).item()
    d_v_proj = torch.max(torch.abs(v_proj_cpu - v_proj_gpu.cpu())).item()
    print(f"    After projection:     |Δu|={d_u_proj:.3e}  |Δv|={d_v_proj:.3e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Gazzola sphere drop — CPU vs GPU fluid solver parity test")
    print("=" * 72)
    print(f"Grid: {Nx}×{Ny}  dtype: {dtype}  N_STEPS: {N_STEPS}  ALARM_TOL: {ALARM_TOL:.0e}")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available — cannot run GPU comparison.")
        print("Run this script on a machine with a CUDA-capable GPU.")
        return

    print("\nBuilding CPU solver...")
    sol_cpu = build_solver('cpu')
    print("Building GPU solver...")
    sol_gpu = build_solver('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    u_cpu, v_cpu, p_cpu = sol_cpu['u'], sol_cpu['v'], sol_cpu['p']
    u_gpu, v_gpu, p_gpu = sol_gpu['u'], sol_gpu['v'], sol_gpu['p']
    h2_cpu, h2_gpu = sol_cpu['h2'], sol_gpu['h2']

    fmt = "{:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>6s}"
    hdr = fmt.format("step", "|Δu|∞", "|Δv|∞", "|Δp|∞", "|Δdiv|∞", "|ΔFpx|", "|ΔFpy|", "status")
    print(hdr)
    print("-" * len(hdr))

    alarm = False
    for step in range(N_STEPS):
        t = step * dt_val

        # Sphere kinematics (same float on both devices)
        vy_sphere = sphere_vy0 + g_eff * t
        pos_y     = sphere_y0 + sphere_vy0 * t + 0.5 * g_eff * t * t
        pos_x     = sphere_x0

        # Update body state on each device
        bstate_cpu = update_body(sol_cpu, pos_x, pos_y, vy_sphere)
        bstate_gpu = update_body(sol_gpu, pos_x, pos_y, vy_sphere)

        # Fluid step
        u_cpu, v_cpu, p_cpu, div_cpu = fluid_step(sol_cpu, bstate_cpu, u_cpu, v_cpu, p_cpu)
        u_gpu, v_gpu, p_gpu, div_gpu = fluid_step(sol_gpu, bstate_gpu, u_gpu, v_gpu, p_gpu)

        # Forces
        fpx_cpu, fpy_cpu = pressure_forces(
            p_cpu, bstate_cpu['sdf_val'], bstate_cpu['normal_x'], bstate_cpu['normal_y'], h2_cpu)
        fpx_gpu, fpy_gpu = pressure_forces(
            p_gpu, bstate_gpu['sdf_val'], bstate_gpu['normal_x'], bstate_gpu['normal_y'], h2_gpu)

        # Diffs
        diff_u   = torch.max(torch.abs(u_cpu   - u_gpu.cpu())).item()
        diff_v   = torch.max(torch.abs(v_cpu   - v_gpu.cpu())).item()
        diff_p   = torch.max(torch.abs(p_cpu   - p_gpu.cpu())).item()
        diff_div = torch.max(torch.abs(div_cpu  - div_gpu.cpu())).item()
        diff_fpx = abs(float(fpx_cpu) - float(fpx_gpu.cpu()))
        diff_fpy = abs(float(fpy_cpu) - float(fpy_gpu.cpu()))

        step_alarm = any(d > ALARM_TOL for d in [diff_u, diff_v, diff_p])
        if step_alarm:
            alarm = True

        status = "ALARM" if step_alarm else "ok"
        print(fmt.format(
            str(step+1),
            f"{diff_u:.3e}", f"{diff_v:.3e}", f"{diff_p:.3e}", f"{diff_div:.3e}",
            f"{diff_fpx:.3e}", f"{diff_fpy:.3e}", status
        ))

        if step_alarm:
            print(f"\n  SDF  CPU/GPU max-diff: "
                  f"{torch.max(torch.abs(bstate_cpu['sdf_val'] - bstate_gpu['sdf_val'].cpu())):.4e}")
            print(f"  mu0  CPU/GPU max-diff: "
                  f"{torch.max(torch.abs(bstate_cpu['mu0']    - bstate_gpu['mu0'].cpu())):.4e}")
            print(f"  n_x  CPU/GPU max-diff: "
                  f"{torch.max(torch.abs(bstate_cpu['normal_x'] - bstate_gpu['normal_x'].cpu())):.4e}")
            substep_diagnostics(
                sol_cpu, sol_gpu, bstate_cpu, bstate_gpu,
                u_cpu, v_cpu, p_cpu, u_gpu, v_gpu, p_gpu
            )
            print()
            print("  Stopping after first alarm step for diagnosis.")
            break

    print()
    if not alarm:
        print(f"✓ PASS: all {N_STEPS} steps within alarm tolerance {ALARM_TOL:.0e}")
    else:
        print(f"✗ FAIL: divergence detected — see ALARM rows above.")

    print()
    print("Final field norms:")
    print(f"  ||u||∞  CPU={u_cpu.abs().max():.6e}  GPU={u_gpu.abs().max():.6e}")
    print(f"  ||v||∞  CPU={v_cpu.abs().max():.6e}  GPU={v_gpu.abs().max():.6e}")
    print(f"  ||p||∞  CPU={p_cpu.abs().max():.6e}  GPU={p_gpu.abs().max():.6e}")


if __name__ == "__main__":
    main()
