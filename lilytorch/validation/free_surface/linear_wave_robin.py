"""Clean LINEARIZED standing gravity wave with the IMPLICIT free-surface
(Robin) boundary condition — the convergent core of the HP4 free surface.

Motivation
----------
The GFM moving-interface probes (`gfm_wave_standalone.py`,
methods `implicit_robin` / `implicit_robin_full`) were resolution-FRAGILE.
A von-Neumann analysis of the implicit Robin BC on a fixed linear domain shows
it is in fact unconditionally stable and convergent:

  mode e^{ikx}:  η^{n+1} = η_pred / (1 + μ) ,   μ = g·dt²·k·tanh(kH)
  → discrete map eigenvalues |λ|² = 1/(1+μ) < 1 (stable, decays ~1/√(1+μ)/step),
    phase atan(√μ) → ω_num = √(gk·tanh kH) as μ→0  (EXACT dispersion).

At RES=75, μ≈3e-4, so the scheme should reproduce the analytic period almost
exactly and CONVERGE under refinement.  This script validates exactly that on a
FIXED domain (water column 0..H, free surface linearised at z=H) with NO GFM,
NO VOF, NO narrow-band velocity zeroing — isolating the implicit coupling that
the moving-interface machinery had been masking.

Layout (MAC, fixed domain, walls on all four sides except the free top):
  p  : cell-centred            (Nx, Nz)
  u  : interior x-faces        (Nx-1, Nz)   (u=0 on side walls)
  v  : interior y-faces        (Nx, Nz-1)   (v=0 on rigid bottom)
  vtop : surface y-face vel    (Nx,)        (the free surface at z=H)
  eta : surface elevation      (Nx,)
Side walls & bottom: homogeneous Neumann pressure (no-flux).  Top: Robin.
"""
import math, os, torch
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)
from lilytorch.src.poisson_gfm import _div_of_faces_2d

g, rho = 9.81, 1000.0
L, H = 0.5, 0.17
a0 = 0.01                                   # small amplitude (linear regime)
RES = int(os.environ.get("FS_RES", "75"))
h = L / RES
Nx, Nz = RES, round(H / h)
k = 2 * math.pi / L
omega = math.sqrt(g * k * math.tanh(k * H))
T_an = 2 * math.pi / omega
dt = 0.3 * h / math.sqrt(g * H)
nt = int(3.0 * T_an / dt)
c = dt / rho
gamma = 2.0 * g * dt * dt / h               # Robin coefficient (∂p/∂z over 0.5h)
mu = g * dt * dt * k * math.tanh(k * H)      # analytic stability parameter

print(f"linear Robin standing wave  {Nx}x{Nz}  dx={h*1e3:.2f}mm"
      f"  T_an={T_an:.4f}s dt={dt:.2e} nt={nt}  mu={mu:.2e}")

xc = (torch.arange(Nx) + 0.5) * h
eta = a0 * torch.cos(k * xc)                 # (Nx,)
u = torch.zeros(Nx - 1, Nz)
v = torch.zeros(Nx, Nz - 1)
vtop = torch.zeros(Nx)


def apply_A(p):
    """Cell-centred pressure Laplacian with Neumann side/bottom walls and the
    Robin free-surface BC on the top face (operator part only)."""
    gx = (p[1:, :] - p[:-1, :]) / h
    gy = (p[:, 1:] - p[:, :-1]) / h
    Ap = _div_of_faces_2d(gx, gy, h)         # interior faces; all walls no-flux
    # top face (z=H, above cell Nz-1): Robin operator part of ∂p/∂z|_H.
    Gop = -2.0 * p[:, Nz - 1] / ((1.0 + gamma) * h)
    Ap[:, Nz - 1] = Ap[:, Nz - 1] - Gop / h  # lower cell gets -flux/h
    return Ap


def cg(rhs, n_iter=500, tol=1e-12):
    p = torch.zeros_like(rhs)
    r = rhs - apply_A(p)
    pdir = r.clone()
    rz = (r * r).sum()
    b2 = (rhs * rhs).sum().clamp(min=1e-30)
    for _ in range(n_iter):
        Ap = apply_A(pdir)
        denom = (pdir * Ap).sum()
        if denom.abs() < 1e-30:
            break
        a = rz / denom
        p = p + a * pdir
        r = r - a * Ap
        if (r * r).sum() / b2 < tol * tol:
            break
        rz_new = (r * r).sum()
        pdir = r + (rz_new / rz) * pdir
        rz = rz_new
    return p


ts, hs = [], []
for it in range(nt):
    # 1. Predictor (no body force: dynamic-pressure form) + predicted elevation.
    vtop_star = vtop
    eta_pred = eta + dt * vtop_star
    src = 2.0 * rho * g * eta_pred / ((1.0 + gamma) * h)     # Robin source (Nx,)
    # 2. RHS = (div(u*) − vtop*/h)/c + src/h at the top row.  The predictor
    #    surface-face flux vtop* MUST enter the surface cell's continuity (it is
    #    omitted by the interior-only _div_of_faces); leaving it out is stable
    #    only while vtop≈0 and then drives a blow-up.
    rhs = _div_of_faces_2d(u, v, h) / c
    rhs[:, Nz - 1] = rhs[:, Nz - 1] - vtop_star / (c * h) + src / h
    # 3. Solve the (SPD) Robin Poisson.
    p = cg(rhs)
    # 4. Velocity correction.
    u = u - c * (p[1:, :] - p[:-1, :]) / h
    v = v - c * (p[:, 1:] - p[:, :-1]) / h
    G_top = -2.0 * p[:, Nz - 1] / ((1.0 + gamma) * h) + src   # full top gradient
    vtop = vtop_star - c * G_top
    # 5. Kinematic surface update.
    eta = eta + dt * vtop

    ts.append(it * dt)
    hs.append(float(eta[0]))
    if it % 100 == 0 or it < 3:
        umax = float(torch.maximum(u.abs().max(), v.abs().max()))
        print(f"  it={it}/{nt} eta0={hs[-1]:+.5f} |u|max={umax:.3e}", flush=True)

# ── analysis ──
import numpy as np
ts_np, hs_np = np.array(ts), np.array(hs)
eta_d = hs_np - np.polyval(np.polyfit(ts_np, hs_np, 1), ts_np)
zc = np.where((eta_d[:-1] < 0) & (eta_d[1:] >= 0))[0]
T_sim = float(np.mean(np.diff(ts_np[zc]))) if len(zc) >= 2 else float("nan")
A1 = np.abs(eta_d[ts_np < T_an]).max()
A2 = np.abs(eta_d[ts_np > ts_np[-1] - T_an]).max()
print(f"\nRESULT T_sim={T_sim:.4f}s  T_an={T_an:.4f}s  err={100*(T_sim-T_an)/T_an:+.1f}%")
print(f"amplitude retained {100*A2/A1:.0f}%  (A1={A1:.4f} A2={A2:.4f})")
