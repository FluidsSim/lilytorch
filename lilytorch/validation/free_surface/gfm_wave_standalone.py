"""Self-contained 2-D MAC free-surface standing wave using the GFM Poisson
(lilytorch/src/poisson_gfm.py). Validates that the sub-cell p=0 BC produces
surface gravity waves at the analytic period omega^2 = g k tanh(kH).

MAC layout (matches poisson_gfm face conventions):
  p, alpha : cell-centred (Nx, Ny)
  u        : interior x-faces (Nx-1, Ny)   [between cells i,i+1]
  v        : interior y-faces (Nx, Ny-1)
Walls all around (boundary-normal velocity = 0). Free surface is interior.
"""
import math, torch
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)
from lilytorch.src.poisson_gfm import level_set_height_2d, gfm_grad_2d, _div_of_faces_2d, gfm_solve_cg_2d

g, rho = 9.81, 1000.0
L, H, Hbox = 0.5, 0.17, 0.34
a0 = 0.02
h = 1.0/150                       # coarse 6.67 mm
Nx, Ny = round(L/h), round(Hbox/h)
k = 2*math.pi/L
omega = math.sqrt(g*k*math.tanh(k*H)); T_an = 2*math.pi/omega
dt = 0.3*h/math.sqrt(g*H)
nt = int(3.0*T_an/dt)
c = dt/rho
print(f"GFM standing wave {Nx}x{Ny} dx={h*1e3:.2f}mm  T_an={T_an:.4f}s dt={dt:.2e} nt={nt}")

xc = (torch.arange(Nx)+0.5)*h
yc = (torch.arange(Ny)+0.5)*h
X, Y = torch.meshgrid(xc, yc, indexing="ij")
alpha = (Y < H + a0*torch.cos(2*math.pi*X/L)).double()
u = torch.zeros(Nx-1, Ny); v = torch.zeros(Nx, Ny-1)

def water_yface(a):            # y-face water mask (avg of the two cells)
    return 0.5*(a[:, :-1] + a[:, 1:]) > 0.5
def water_xface(a):
    return 0.5*(a[:-1, :] + a[1:, :]) > 0.5

def advect_alpha(a, u, v):
    # simple 1st-order upwind with the interior face velocities; walls no-flux
    fx = torch.zeros(Nx-1, Ny)
    aL, aR = a[:-1, :], a[1:, :]
    fx = torch.where(u > 0, u*aL, u*aR)
    fy = torch.zeros(Nx, Ny-1)
    aD, aU = a[:, :-1], a[:, 1:]
    fy = torch.where(v > 0, v*aD, v*aU)
    da = torch.zeros_like(a)
    da[1:, :]  -= fx/h; da[:-1, :] += fx/h
    da[:, 1:]  -= fy/h; da[:, :-1] += fy/h
    return (a + dt*da).clamp(0.0, 1.0)

ts, hs = [], []
for it in range(nt):
    # 1. gravity on water y-faces (air is a void: no fluid there)
    wv = water_yface(alpha)
    v = v + torch.where(wv, torch.full_like(v, -g*dt), torch.zeros_like(v))
    # 2. level set + projection (p=0 at sub-cell surface)
    phi = level_set_height_2d(alpha, h, 0.0)
    div = _div_of_faces_2d(u, v, h)
    p = gfm_solve_cg_2d(div/c, phi, h, n_iter=300, tol=1e-8)
    gx, gy = gfm_grad_2d(p, phi, h)
    u = u - c*gx; v = v - c*gy
    # 3. kill any velocity in the air void (faces not in water)
    u = torch.where(water_xface(alpha), u, torch.zeros_like(u))
    v = torch.where(water_yface(alpha), v, torch.zeros_like(v))
    # 4. advect interface
    alpha = advect_alpha(alpha, u, v)
    ts.append(it*dt); hs.append(float((0.0 + h*alpha[1, :].sum())))   # antinode col height
    if it % 100 == 0:
        print(f"  it={it}/{nt} h={hs[-1]:.4f} umax={float(torch.maximum(u.abs().max(),v.abs().max())):.3e}", flush=True)

import numpy as np
ts, hs = np.array(ts), np.array(hs)
eta = hs - hs.mean()
zc = np.where((eta[:-1] < 0) & (eta[1:] >= 0))[0]
T_sim = float(np.mean(np.diff(ts[zc]))) if len(zc) >= 2 else float('nan')
A1 = np.abs(eta[ts < T_an]).max(); A2 = np.abs(eta[ts > ts[-1]-T_an]).max()
print(f"\nRESULT T_sim={T_sim:.4f}s  T_an={T_an:.4f}s  err={100*(T_sim-T_an)/T_an:+.1f}%")
print(f"amplitude retained {100*A2/A1:.0f}%  (A1={A1:.4f} A2={A2:.4f})")
