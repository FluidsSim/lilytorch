"""
Error analysis for flow past cylinder in 2D.
Adapted from lilytorch/src/scripts/run_error_analysis_cylinder.py

Runs the flow past cylinder simulation at multiple grid resolutions,
saves the final fields (u, v, p) for each resolution, then the
companion script plot_error_analysis.py can be used to compute
convergence rates.
"""

from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gc

# ---- grid sizes to test (coarse -> fine) ----
nxs = [64, 128, 256, 512, 1024]

# Convection method
convection_method = "abdquickest"

# Output base directory
output_base = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests/" + convection_method

for nx in nxs:

    print(f"\n{'='*60}")
    print(f"  Running flow past cylinder 2D – grid size Nx=Ny={nx}")
    print(f"{'='*60}\n")

    pars = yaml2pyobject("lilytorch/src/scripts/flow_past_cylinder.yaml")

    R = 0.1  # radius of the cylinder
    time_stop = 200  # physical time to reach steady state

    U  = pars["boundary_conditions"]["BC_values_u"][0]
    D  = 2 * R
    nu = pars["solver"]["nu"]
    Re = U * D / nu
    print(f"Reynolds number: {Re:.1f}")

    # Square domain [-1, 1] x [-1, 1]
    pars["solver"]["xmin"] = -1
    pars["solver"]["xmax"] =  1
    pars["solver"]["ymin"] = -1
    pars["solver"]["ymax"] =  1
    pars["solver"]["Nx"]   = nx
    pars["solver"]["Ny"]   = nx

    # Place cylinder near left boundary
    pars["body"]["sdf"] = [
        f"lambda x, y: circle(x,y,xt={pars['solver']['xmin'] + 3*R},yt=0,r={R})"
    ]

    dx    = (pars["solver"]["xmax"] - pars["solver"]["xmin"]) / nx
    t_clf = dx / U  # CFL-based reference time

    # Time stepping
    pars["solver"]["dt"]                = 0.1 * t_clf
    pars["solver"]["convection_method"] = convection_method
    pars["solver"]["nt"]                = int(time_stop / pars["solver"]["dt"]) + 1

    # Output settings
    pars["output"]["save_frames"] = True
    pars["output"]["save_every"]  = max(200, pars["solver"]["nt"] // 20)

    print(f"  dx = {dx:.6f},  dt = {pars['solver']['dt']:.6e},  nt = {pars['solver']['nt']}")

    # =========== Run simulation ===========
    solver = FluidSolver(pars, dtype=torch.float32, compute_forces=True)
    solver.save_path = f'{output_base}/{nx}/'
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    u = solver.u0
    v = solver.v0
    p = solver.p0

    for it in tqdm(range(0, solver.nt), desc=f"nx={nx}"):
        t = it * solver.dt
        (u, v, p, stop_sim) = solver.step_(u, v, p, it, t)

    # Save last iteration fields
    uv_path = f'{solver.save_path}/uv_field'
    os.makedirs(uv_path, exist_ok=True)
    np.save(f'{uv_path}/u', u.cpu().numpy())
    np.save(f'{uv_path}/v', v.cpu().numpy())
    np.save(f'{uv_path}/p', p.cpu().numpy())

    print(f"  Saved fields to {uv_path}")

    # Free GPU memory before next resolution
    del solver, u, v, p
    torch.cuda.empty_cache()
    gc.collect()

print("\n\nAll simulations complete!")
print(f"Results saved under: {output_base}")
