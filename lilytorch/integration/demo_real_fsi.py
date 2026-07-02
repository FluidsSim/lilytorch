"""End-to-end: StrongCoupledFSI on a REAL 2-D FluidSolver + free circle.

A light circle (mass < displaced-fluid mass) is pushed by a constant
external force through initially quiescent fluid.  This is the added-mass
regime: explicit coupling (omega=1) should struggle/diverge, while IQN-ILS
converges in a few sweeps per step.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))      # lilytorch/integration
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
sys.path.insert(0, HERE)

from lilytorch.util.yaml_operations import yaml2pyobject
from fsi_coupling import IQNILS, ConstantUnderRelaxation
from fsi_rigid_body import build_rigid_circle_fsi


def make_pars():
    pars = yaml2pyobject(os.path.join(ROOT, "lilytorch", "src", "configs",
                                      "flow_past_circle_2d.yaml"))
    pars["solver"].update(
        nt=12, save=False, save_frames=False, use_gpu=True,
    )
    # quiescent fluid: zero the inflow so the only flow is body-induced
    for key in ("bc_values_u", "bc_values_v"):
        if key in pars["solver"]:
            pars["solver"][key] = [0.0, 0.0, 0.0, 0.0]
    return pars


def run(accelerator, label, n_steps=12):
    pars = make_pars()
    rho = float(pars["solver"]["rho"])
    radius = 0.08
    m_disp = rho * np.pi * radius**2          # displaced-fluid mass (2-D)
    mass = 0.4 * m_disp                        # light body -> added-mass regime
    f_ext = (0.6 * mass, 0.0)                  # constant horizontal push

    driver, coupling, fs = build_rigid_circle_fsi(
        pars, radius=radius, pos0=(0.0, 0.0), vel0=(0.0, 0.0),
        mass=mass, f_ext=f_ext, accelerator=accelerator,
        tol=1e-4, max_iter=40,
    )
    fs.u0 = torch.zeros_like(fs.u0)            # quiescent start
    fs.v0 = torch.zeros_like(fs.v0)

    dt = float(fs.dt)
    ok = True
    for it in range(n_steps):
        conv = driver.step(it, it * dt, dt)
        if not conv or not np.all(np.isfinite(coupling.get_state())):
            ok = False
            print(f"  [{label}] step {it}: DIVERGED / not converged "
                  f"(iters={driver.last_iters}, res={driver.last_residual:.2e})")
            break

    print(f"  [{label}] mass/m_disp={mass/m_disp:.2f}  "
          f"converged_steps={'all' if ok else it}  "
          f"iters/step={driver.iters_history}")
    if ok:
        print(f"  [{label}] final body state [x,y,vx,vy] = "
              f"{np.array2string(coupling.get_state(), precision=5)}")
    return ok, driver.iters_history


if __name__ == "__main__":
    print("=== Real 2-D FSI: light circle pushed through quiescent fluid ===\n")
    print("IQN-ILS:")
    run(IQNILS(omega_init=0.1, reuse=2), "IQN-ILS")
    print("\nExplicit (constant omega=1.0):")
    run(ConstantUnderRelaxation(omega=1.0), "explicit")
