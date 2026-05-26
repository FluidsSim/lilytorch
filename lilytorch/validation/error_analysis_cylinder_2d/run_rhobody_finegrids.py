from lilytorch.src.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch, numpy as np, os, gc
from tqdm import tqdm

D, R, Re, U = 1.0, 0.5, 100, 1.0
nu, rho = U*D/Re, 1.0
half_L = 5*D
xmin, xmax = -half_L, half_L
ymin, ymax = -half_L, half_L
cx, cy = 0.0, 0.0
t_stop = 3.0
domain_size = xmax - xmin
output_base = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/sim_data"

for Nx in [256, 512, 1024, 2048]:
    dx = domain_size/Nx; Ny = Nx
    print(f"\n{'='*60}\n  Nx={Nx}  dx={dx:.6f}  D/dx={D/dx:.1f}\n{'='*60}", flush=True)
    pars = yaml2pyobject("lilytorch/src/configs/flow_past_cylinder.yaml")
    pars["solver"].update(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, Nx=Nx, Ny=Ny,
                          nu=nu, rho=rho, rho_body=rho, solver_method="python",
                          poisson_method="multigrid", poisson_max_cycles=30,
                          poisson_nsmoothing=10, poisson_tol=1e-8)
    pars["body"]["sdf"] = [f"lambda x, y: circle(x,y,xt={cx},yt={cy},r={R})"]
    pars["boundary_conditions"]["BC_values_u"] = [U, U, 0.0, 0.0]
    pars["boundary_conditions"]["BC_values_v"] = [0.0, 0.0, 0.0, 0.0]
    dt = 0.1*dx/U
    pars["solver"]["dt"] = dt
    pars["solver"]["convection_method"] = "abdquickest"
    pars["solver"]["nt"] = int(t_stop/dt) + 1
    pars["output"]["save_frames"] = False
    save_path = f"{output_base}/Nx{Nx}/"
    solver = FluidSolver(pars, dtype=torch.float32, compute_forces=False)
    print(f"  rho_fluid={solver.rho}  rho_body={solver.rho_body}  nt={solver.nt}", flush=True)
    solver.save_path = save_path
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    u, v, p = solver.u0, solver.v0, solver.p0
    for it in tqdm(range(0, solver.nt), desc=f"Nx={Nx}"):
        u, v, p, _ = solver.step_(u, v, p, it, it*solver.dt)
    uv_path = f"{save_path}/uv_field"; os.makedirs(uv_path, exist_ok=True)
    np.save(f"{uv_path}/u", u.cpu().numpy())
    np.save(f"{uv_path}/v", v.cpu().numpy())
    np.save(f"{uv_path}/p", p.cpu().numpy())
    print(f"  saved -> {uv_path}", flush=True)
    del solver, u, v, p
    torch.cuda.empty_cache(); gc.collect()
print("\nFine grids done.", flush=True)
