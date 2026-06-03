"""
2-D hydrostatic TWO-PHASE (water + real air) validation.

A closed box, water below the interface ``y = 0.5`` (alpha = 1) and real
light air above (alpha = 0), under gravity. At steady state:

  * velocity ``≈ 0`` everywhere;
  * pressure is the variable-density hydrostatic integral
        dp/dy = -rho(y) g ,
    i.e. a steep linear slope (rho_water) in the water and a shallow one
    (rho_air) in the air, continuous across the interface.

Unlike the single-fluid test there is no pinned ``p_atm`` gauge — the air is a
real fluid and the closed box is all-Neumann, so the pressure is defined only
up to an additive constant. The check is therefore **gauge-agnostic**: fit the
constant by matching means, then measure the max residual ``|p - p_ref|``.

Runs CPU-only in float64 (same sandbox work-arounds as the single-fluid test)
and exercises the decoupled :class:`TwoPhaseSolver` end-to-end.
"""

import math
import torch
from tqdm import tqdm

torch.set_default_device("cpu")
# torch.compile is broken in this sandbox and unneeded on CPU.
torch.compile = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))

from lilytorch.src.two_phase_solver import TwoPhaseSolver
from lilytorch.src.poisson_mult import PoissonSolver

_orig_dispatch = PoissonSolver._dispatch_vcycle
def _cpu_dispatch_vcycle(self, f, p, face_arrs):
    if f.is_cuda:
        return _orig_dispatch(self, f, p, face_arrs)
    return self._vcycle(f, p, face_arrs)
PoissonSolver._dispatch_vcycle = _cpu_dispatch_vcycle


def build_pars():
    rho_w, rho_a = 1000.0, 1.0
    g_y = -9.81
    Nx = Ny = 32
    h = 1.0 / Nx
    dt = 0.2 * h / math.sqrt(abs(g_y) * 0.5)
    pars = {
        "solver": {
            "use_gpu": True, "nthreads": 1,
            "Nx": Nx, "Ny": Ny,
            "xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0,
            "nt": 600, "nu": 1e-3, "rho": rho_w, "dt": dt,
            "convection_method": "cds",
            "poisson_tol": 1e-9, "jacobi_weight": 1.0,
            "poisson_max_cycles": 40, "poisson_max_mgcg_cycles": 40,
            "poisson_nsmoothing": 6, "poisson_verbose": False,
            "poisson_folder": "lilytorch/data/",
            "poisson_method": "mgcg", "poisson_smoother": "rbgs",
            "dtype": "float64", "solver_method": "python",
            "rho_body": rho_w,
            "gravity": [0.0, g_y],
            "two_phase": {
                "alpha_init": "lambda X, Y: (Y < 0.5).double()",  # 1 water, 0 air
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
        "body": {  # far-away placeholder (mu0 = 1 in-domain)
            "type": "composite_analytical", "plotting": False,
            "sdf": ["lambda x, y: circle(x,y,xt=-0.5,yt=-0.5,r=0.02)"],
            "update_maps": [{
                "rotation": "lambda t: torch.tensor(0.0)",
                "translation": ["lambda t: torch.tensor(0.0)",
                                "lambda t: torch.tensor(0.0)"],
            }],
        },
        "output": {"save_path": "/tmp/lilytorch_tp_hydro/", "save_frames": True,
                   "save_every": 50, "save": False, "save_drags": False,
                   "vmin": -0.05, "vmax": 0.05},
    }
    return pars, rho_w, rho_a, abs(g_y)


def _p_ref(Y, rho_w, rho_a, g, H=0.5):
    """Variable-density hydrostatic reference (additive constant arbitrary)."""
    return torch.where(Y < H,
                       -rho_w * g * Y,
                       -rho_w * g * H - rho_a * g * (Y - H))


def main():
    pars, rho_w, rho_a, g = build_pars()
    print("=" * 70)
    print("  2-D hydrostatic TWO-PHASE validation (CPU, float64)")
    print(f"  Nx={pars['solver']['Nx']}, dt={pars['solver']['dt']:.4e}, "
          f"nt={pars['solver']['nt']}, rho {rho_w}/{rho_a}, interface @ y=0.5")
    print("=" * 70)

    solver = TwoPhaseSolver(pars, dtype=torch.float64, compute_forces=False)
    solver.inside = lambda *a, **k: True
    solver.set_initial_conditions()
    u, v, p = solver.u0, solver.v0, solver.p0
    V0 = solver.two_phase.water_volume()

    Y = torch.meshgrid(solver.x, solver.y, indexing="ij")[1]
    interior = torch.ones_like(Y, dtype=torch.bool)
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False

    def report(it):
        with torch.no_grad():
            umax = max(u.abs().max().item(), v.abs().max().item())
            pr = _p_ref(Y, rho_w, rho_a, g)
            off = (p[interior] - pr[interior]).mean()      # gauge fit
            perr = (p[interior] - pr[interior] - off).abs().max().item()
            vdrift = (solver.two_phase.water_volume() - V0) / V0
            return umax, perr, vdrift

    for it in tqdm(range(solver.nt)):
        (u, v, p, stop) = solver.step_(u, v, p, it, it * solver.dt)
        if (it + 1) % 100 == 0 or it == solver.nt - 1:
            umax, perr, vdrift = report(it)
            print(f"  it={it+1:4d}  |u|max={umax:.3e}  "
                  f"|p-p_ref|max={perr:.3e}  vol_drift={vdrift:+.2e}")
        if stop:
            break

    umax, perr, vdrift = report(solver.nt - 1)
    p_scale = rho_w * g * 0.5
    rel = perr / p_scale
    print("\nFinal diagnostics:")
    print(f"  |u|_max                       = {umax:.3e}")
    print(f"  max|p - p_ref| (gauge-fit)    = {perr:.3e}  "
          f"(rel to rho_w g H = {p_scale:.3e} -> {rel*100:.2f} %)")
    print(f"  water-volume drift            = {vdrift:+.3e}")
    ok_p = rel < 0.05
    ok_u = umax < 0.05 * math.sqrt(g * 0.5)
    ok_m = abs(vdrift) < 1e-3
    print(f"\n  pressure-error OK : {ok_p}\n  velocity OK       : {ok_u}\n"
          f"  mass-conserve OK  : {ok_m}")
    print("  ===> TWO-PHASE HYDROSTATIC "
          + ("PASSED" if (ok_p and ok_u and ok_m) else "FAILED"))


if __name__ == "__main__":
    main()
