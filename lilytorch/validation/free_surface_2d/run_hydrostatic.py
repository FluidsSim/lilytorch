"""
2-D hydrostatic free-surface validation.

A column of water under gravity with an air interface at ``y = 0.5``
should reach a steady state with:

  * velocity ``≈ 0`` everywhere
  * pressure ``p ≈ ρ g (h − y)``  in water  (linear, decreasing with y)
  * pressure ``p ≈ 0``             in air

This script runs the FluidSolver in pure-CPU mode (no farms, no CUDA),
with no real body (a tiny circle is placed in the corner so the existing
BDIM pipeline has something to chew on; it does not interact with the
fluid or the free surface), and reports the L∞ velocity, the max
pressure deviation from the hydrostatic reference inside water, and the
max pressure in air.

Reference solution (gauge ``p_atm = 0``):
    p_ref(y) = ρ g (H − y)   for y < H = 0.5
    p_ref(y) = 0             for y > H
"""

import math
import os
import numpy as np
import torch
from tqdm import tqdm

# Force CPU.
torch.set_default_device("cpu")

# ---- CPU work-arounds for the optimize_speed_memory branch:
# (1) torch.compile is broken in this sandbox (torch 2.12 / py3.12), and is
#     anyway not needed on CPU — disable it before importing solver modules.
# (2) PoissonSolver._dispatch_vcycle hard-routes to CUDA-only native ops,
#     so on CPU we transparently fall back to the pure-PyTorch ``_vcycle``.
torch.compile = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))

from lilytorch.src.solver import FluidSolver
from lilytorch.src.poisson_mult import PoissonSolver

_orig_dispatch = PoissonSolver._dispatch_vcycle
def _cpu_dispatch_vcycle(self, f, p, face_arrs):
    if f.is_cuda:
        return _orig_dispatch(self, f, p, face_arrs)
    return self._vcycle(f, p, face_arrs)
PoissonSolver._dispatch_vcycle = _cpu_dispatch_vcycle


# =====================================================================
# Setup
# =====================================================================
def build_pars():
    rho   = 1000.0
    nu    = 1e-3
    g_y   = -9.81
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    Nx = Ny = 32
    h   = (xmax - xmin) / Nx
    # CFL on gravity-driven wave c = sqrt(g H) ~ 2.2 m/s.  dt = 0.2 h / c.
    dt = 0.2 * h / math.sqrt(abs(g_y) * 0.5)

    pars = {
        "solver": {
            "use_gpu": False,
            "nthreads": 1,
            "Nx": Nx, "Ny": Ny,
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "nt": 600,
            "nu": nu,
            "rho": rho,
            "dt": dt,
            "convection_method": "cds",
            "poisson_tol": 1e-9,
            "jacobi_weight": 1.0,
            "poisson_max_cycles": 30,
            "poisson_max_mgcg_cycles": 30,
            "poisson_nsmoothing": 6,
            "poisson_verbose": False,
            "poisson_folder": "lilytorch/data/",
            "poisson_method": "multigrid",
            "poisson_smoother": "rbgs",
            "dtype": "float64",
            "solver_method": "python",       # avoid kernel-mode (C++ ext not built)
            "rho_body": rho,
            # Gravity (vertical, downward).
            "gravity": [0.0, g_y],
            # Free-surface block (level-set, interface at y = 0.5).
            "free_surface": {
                "phi_init":    "lambda X, Y: Y - 0.5",
                "theta_min":   0.01,
                "band_cells":  4,
                "reinit_iters": 4,
                "reinit_every": 10,
                "extend_iters": 2,
                "extend_every": 1,
            },
        },
        "boundary_conditions": {
            # Closed box (no-slip walls everywhere).  This is consistent
            # with a hydrostatic test: nothing should move.
            "BC_type_u":   ["D", "D", "D", "D"],
            "BC_values_u": [0.0, 0.0, 0.0, 0.0],
            "BC_type_v":   ["D", "D", "D", "D"],
            "BC_values_v": [0.0, 0.0, 0.0, 0.0],
        },
        "body": {
            # A tiny circle parked in the corner so the BDIM pipeline has
            # something to consume.  It does not intersect the column.
            "type": "composite_analytical",
            "plotting": False,
            "sdf": ["lambda x, y: circle(x,y,xt=0.05,yt=0.05,r=0.02)"],
            "update_maps": [{
                "rotation":    "lambda t: torch.tensor(0.0)",
                "translation": [
                    "lambda t: torch.tensor(0.0)",
                    "lambda t: torch.tensor(0.0)",
                ],
            }],
        },
        "output": {
            "save_path": "/tmp/lilytorch_hydrostatic/",
            "save_frames": False,
            "save_every": 1000000,
            "save": False,
            "save_drags": False,
            "vmin": "auto", "vmax": "auto",
        },
    }
    return pars, rho, abs(g_y)


def main():
    pars, rho, g = build_pars()
    print("=" * 70)
    print("  2-D hydrostatic free-surface validation (CPU, float64)")
    print(f"  Nx={pars['solver']['Nx']}, dt={pars['solver']['dt']:.4e}, "
          f"nt={pars['solver']['nt']}")
    print(f"  rho={rho}, g={g}, interface @ y=0.5")
    print("=" * 70)

    solver = FluidSolver(pars, dtype=torch.float64, compute_forces=False)
    # Disable the "body exited domain" early-stop: our placeholder body
    # is intentionally tiny and may sit near a corner.
    solver.inside = lambda *a, **k: True
    solver.set_initial_conditions()

    u = solver.u0
    v = solver.v0
    p = solver.p0

    u_history = []
    p_err_history = []
    for it in tqdm(range(solver.nt)):
        t = it * solver.dt
        (u, v, p, stop) = solver.step_(u, v, p, it, t)
        if (it + 1) % 50 == 0 or it == solver.nt - 1:
            with torch.no_grad():
                umax = max(u.abs().max().item(), v.abs().max().item())
                Y = torch.meshgrid(solver.x, solver.y, indexing="ij")[1]
                p_ref = torch.where(Y < 0.5, rho * g * (0.5 - Y),
                                    torch.zeros_like(Y))
                water_mask = (Y < 0.5 - 1.5 * solver.h).bool()
                water_mask[0, :] = False; water_mask[-1, :] = False
                water_mask[:, 0] = False; water_mask[:, -1] = False
                p_err = (p[water_mask] - p_ref[water_mask]).abs().max().item()
                p_water_mean = p[water_mask].mean().item()
                p_ref_mean   = p_ref[water_mask].mean().item()
                air_mask = (Y > 0.5 + 1.5 * solver.h).bool()
                air_mask[0, :] = False; air_mask[-1, :] = False
                air_mask[:, 0] = False; air_mask[:, -1] = False
                p_air_max = p[air_mask].abs().max().item() if air_mask.any() else 0.0
                u_history.append(umax)
                p_err_history.append(p_err)
                print(f"  it={it+1:4d}  |u|max={umax:.3e}  "
                      f"|p-p_ref|max(water)={p_err:.3e}  "
                      f"p_air_max={p_air_max:.3e}  "
                      f"mean(p)water={p_water_mean:.3e} vs ref {p_ref_mean:.3e}")
        if stop:
            break

    print()
    print("Final diagnostics:")
    with torch.no_grad():
        umax_final = max(u.abs().max().item(), v.abs().max().item())
        Y = torch.meshgrid(solver.x, solver.y, indexing="ij")[1]
        p_ref = torch.where(Y < 0.5, rho * g * (0.5 - Y), torch.zeros_like(Y))
        water = (Y < 0.5 - 1.5 * solver.h).bool()
        water[0, :] = False; water[-1, :] = False
        water[:, 0] = False; water[:, -1] = False
        p_err = (p[water] - p_ref[water]).abs().max().item()
        p_ref_max = p_ref[water].max().item()
        rel_err = p_err / p_ref_max if p_ref_max > 0 else float("nan")
        print(f"  |u|_max                             = {umax_final:.3e}")
        print(f"  max|p - rho*g*(H-y)| inside water   = {p_err:.3e}  "
              f"(relative to rho*g*H = {p_ref_max:.3e} -> {rel_err*100:.2f} %)")

        ok_p   = rel_err < 0.05
        ok_u   = umax_final < 0.05 * math.sqrt(g * 0.5)
        print()
        print(f"  pressure-error OK     : {ok_p}")
        print(f"  velocity-magnitude OK : {ok_u}")
        if ok_p and ok_u:
            print("  ===> HYDROSTATIC TEST PASSED")
        else:
            print("  ===> HYDROSTATIC TEST FAILED")

    return p, Y, p_ref


if __name__ == "__main__":
    main()
