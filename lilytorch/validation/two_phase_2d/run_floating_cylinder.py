"""
2-D floating-cylinder TWO-PHASE validation (P2: immersed body buoyancy).

The central physics check for partially-submerged bodies: a fixed cylinder
straddling the waterline must feel the correct **Archimedes buoyancy**, which
in the two-phase model emerges purely from the density-weighted hydrostatic
pressure acting on the wetted hull (no special buoyancy term).

Part A (this script): a cylinder of radius R is held FIXED with its centre on
the interface (half submerged). At steady state the net vertical fluid force
should equal the weight of the displaced fluid::

    F_y ≈ g * (rho_water * A_below + rho_air * A_above)
        = g * (rho_water + rho_air) * (pi R^2 / 2)      (half-submerged)

A free-floating (released) cylinder settling to its Archimedes draft via the
IQN-ILS coupler is the next step (Part B).

CPU float64.
"""

import math
import torch
from tqdm import tqdm

torch.set_default_device("cpu")
torch.compile = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))

from lilytorch.src.two_phase_solver import TwoPhaseSolver
from lilytorch.src.poisson_mult import PoissonSolver

_orig = PoissonSolver._dispatch_vcycle
PoissonSolver._dispatch_vcycle = (
    lambda self, f, p, fa: (_orig(self, f, p, fa) if f.is_cuda
                            else self._vcycle(f, p, fa)))


def build_pars(R=0.15, Hint=0.5):
    rho_w, rho_a = 1000.0, 1.0
    g = 9.81
    Nx = Ny = 48
    h = 1.0 / Nx
    dt = 0.15 * h / math.sqrt(g * 0.5)
    pars = {
        "solver": {
            "use_gpu": True, "nthreads": 1,
            "Nx": Nx, "Ny": Ny,
            "xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0,
            "nt": 800, "nu": 1e-3, "rho": rho_w, "dt": dt,
            "convection_method": "cds",
            "poisson_tol": 1e-9, "jacobi_weight": 1.0,
            "poisson_max_cycles": 40, "poisson_max_mgcg_cycles": 40,
            "poisson_nsmoothing": 6, "poisson_verbose": False,
            "poisson_folder": "lilytorch/data/",
            "poisson_method": "mgcg", "poisson_smoother": "rbgs",
            "dtype": "float64", "solver_method": "python",
            "gravity": [0.0, -g],
            "two_phase": {
                "alpha_init": f"lambda X, Y: (Y < {Hint}).double()",
                "rho_water": rho_w, "rho_air": rho_a,
                "nu_water": 1e-3, "nu_air": 1e-3,
                "advection": "cubista", "compression": 1.0,
                "face_density": "harmonic",
            },
        },
        "boundary_conditions": {
            "BC_type_u": ["D", "D", "D", "D"], "BC_values_u": [0.0]*4,
            "BC_type_v": ["D", "D", "D", "D"], "BC_values_v": [0.0]*4,
        },
        "body": {   # FIXED cylinder, centre on the interface at real (0.5, Hint).
            # Base the circle at the CENTRED-grid origin (xt=yt=0) and place it
            # via a constant translation. Two grids are in play and BOTH must be
            # satisfied: the SOLVE evaluates the SDF on REAL coords (so the body
            # ends up at real (0.5, Hint)); the 2-D contour init samples the base
            # SDF on the DOMAIN-CENTRED grid (x-xmid in [-L/2, L/2]). Basing at
            # the origin keeps the full circle inside the contour window, so the
            # Lagrangian surface contour (cnt_update) is the WHOLE circle.
            # (circle(xt=0.5, yt=0.5) would put the centre at the centred-grid
            # CORNER -> find_contours captures only a quarter-arc -> a broken
            # contour and ~75% Lagrangian buoyancy error. The eulerian/displaced
            # path uses sdf_val on the real grid and is unaffected either way.)
            "type": "composite_analytical", "plotting": False,
            "sdf": [f"lambda x, y: circle(x,y,xt=0.0,yt=0.0,r={R})"],
            "update_maps": [{"rotation": "lambda t: 0.0*t",
                             "translation": ["lambda t: 0.5 + 0.0*t",
                                             f"lambda t: {Hint} + 0.0*t"]}],
        },
        "output": {"save_path": "/tmp/lilytorch_tp_float/", "save_frames": True,
                   "save_every": 50, "save": False, "save_drags": False,
                   "vmin": -0.5, "vmax": 0.5},
    }
    return pars, rho_w, rho_a, g, R, Hint


def main():
    pars, rho_w, rho_a, g, R, Hint = build_pars()
    A_half = math.pi * R**2 / 2.0
    F_arch = g * (rho_w + rho_a) * A_half     # half-submerged buoyancy
    print("=" * 70)
    print("  2-D floating-cylinder TWO-PHASE buoyancy (fixed, half-submerged)")
    print(f"  R={R}, interface y={Hint}, Archimedes F_y = {F_arch:.4f} N/m")
    print("=" * 70)

    solver = TwoPhaseSolver(pars, dtype=torch.float64, compute_forces=True)
    solver.inside = lambda *a, **k: True
    solver.set_initial_conditions()
    u, v, p = solver.u0, solver.v0, solver.p0

    Fy_hist = []
    for it in tqdm(range(solver.nt)):
        u, v, p, _ = solver.advance_and_compute_loads(u, v, p, it, it*solver.dt)
        solver.finalize_step(u, v, p, it)
        if (it + 1) % 100 == 0 or it == solver.nt - 1:
            loads = solver.get_loads()
            Fy = float(loads[0][0, 1])          # body 0, y-component
            Fy_hist.append(Fy)
            umax = max(u.abs().max().item(), v.abs().max().item())
            print(f"  it={it+1:4d}  F_y={Fy:+.4f}  (Archimedes {F_arch:.4f}, "
                  f"ratio {Fy/F_arch:+.3f})  |u|max={umax:.2e}")

    Fy = sum(Fy_hist[-3:]) / len(Fy_hist[-3:])  # average last reports
    rel = abs(Fy - F_arch) / F_arch
    print("\nFinal diagnostics:")
    print(f"  measured F_y (steady) = {Fy:+.4f} N/m")
    print(f"  Archimedes F_y        = {F_arch:+.4f} N/m")
    print(f"  relative error        = {rel*100:.1f} %")
    ok = rel < 0.10
    print(f"\n  buoyancy OK : {ok}")
    print("  ===> FLOATING-CYLINDER (fixed buoyancy) "
          + ("PASSED" if ok else "NEEDS REVIEW"))


if __name__ == "__main__":
    main()
