"""Prototype: BDIM-consistent projection (MW2015).

Standard lilytorch projection: u = u_bdim - c*mu0*grad(p),  solve div(c*mu0 grad p)=div(u_bdim).
This MISSES the BDIM kernel's first-moment pressure term.  The consistent
projection smooths the WHOLE pressure gradient with the kernel:

    u = u_bdim - c * b_eps[grad p],   b_eps[g] = mu0*g + eps*mu1*d/dn(g)

=> div(c*mu0 grad p) = div(u_bdim) - div(c*eps*mu1 * dn(grad p))

Solved by deferred correction: lag the eps*mu1*dn(grad p) term to the RHS,
solve the standard mu0-Poisson each sweep.  The extra term is O(dx) relative
to the main term, so a few sweeps converge.

Run a 128/256/512 sweep and save fields for analyze_dir.py (512 as ref).
"""
from lilytorch.src.solver import FluidSolver
from lilytorch.src import operations as ops
from lilytorch.util.yaml_operations import yaml2pyobject
import torch, numpy as np, os, gc, types
from tqdm import tqdm

D, R, Re, U = 1.0, 0.5, 100, 1.0
nu, rho = U*D/Re, 1.0
half_L = 5*D
cx, cy = 0.0, 0.0
t_stop = 3.0
ds = 2*half_L
N_SWEEPS = 4
output_base = "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/bdimconsistent_abdquickest"

def fluid_step_consistent(self, u, v, p, timestep):
    # 1. advection-diffusion
    nu_t   = self._compute_nu_t(u, v)
    us, vs = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
    us, vs = us.clone(), vs.clone()
    # 2. BDIM meta-equation (mu0/mu1/normals already recomputed in step_)
    u_b, v_b = self._apply_bdim_all_axes((us, vs))
    self.adv_diff_solver.set_BCs(u_b, v_b)

    c   = float(timestep) / float(self.rho)
    eps = self.eps
    h   = self.h
    mu0_u, mu0_v = self.mu0_all_u, self.mu0_all_v
    mu1_u, mu1_v = self.mu1_all_u, self.mu1_all_v
    nx_u, ny_u   = self.normal_x_u, self.normal_y_u
    nx_v, ny_v   = self.normal_x_v, self.normal_y_v
    ch = c * mu0_u
    cv = c * mu0_v

    div_target = self.divergence(u_b, v_b)        # div(u_bdim)
    p = torch.zeros_like(p)
    for k in range(N_SWEEPS):
        px, py = self.gradient(p)
        dn_px = ops.normal_derivative(px, h, 2, nx_u, ny_u)   # d/dn (dp/dx) on u-grid
        dn_py = ops.normal_derivative(py, h, 2, nx_v, ny_v)   # d/dn (dp/dy) on v-grid
        corr_u = (c * eps) * mu1_u * dn_px
        corr_v = (c * eps) * mu1_v * dn_py
        rhs = div_target - self.divergence(corr_u, corr_v)
        p, _ = self.poisson_solver.solve_multigrid(
            rhs[1:-1, 1:-1], p, ch=ch[1:, 1:-1], cv=cv[1:-1, 1:])

    # final correction with BOTH terms (mu0 grad p + eps*mu1*dn(grad p))
    px, py = self.gradient(p)
    dn_px = ops.normal_derivative(px, h, 2, nx_u, ny_u)
    dn_py = ops.normal_derivative(py, h, 2, nx_v, ny_v)
    u_out = u_b - ch * px - (c * eps) * mu1_u * dn_px
    v_out = v_b - cv * py - (c * eps) * mu1_v * dn_py
    if self.use_sponge:
        u_out, v_out = self.apply_sponge_damping(u_out, v_out)
    self.adv_diff_solver.set_BCs(u_out, v_out)
    return (u_out, v_out, p)

for Nx in [128, 256, 512, 1024, 2048]:
    dx = ds/Nx; Ny = Nx
    print(f"\n{'='*60}\n  Nx={Nx} dx={dx:.6f} D/dx={D/dx:.1f}  [BDIM-consistent, {N_SWEEPS} sweeps]\n{'='*60}", flush=True)
    pars = yaml2pyobject("lilytorch/src/configs/flow_past_cylinder.yaml")
    pars["solver"].update(xmin=-half_L, xmax=half_L, ymin=-half_L, ymax=half_L, Nx=Nx, Ny=Ny,
        nu=nu, rho=rho, solver_method="python", poisson_method="multigrid",
        poisson_max_cycles=30, poisson_nsmoothing=10, poisson_tol=1e-8)
    pars["body"]["sdf"] = [f"lambda x, y: circle(x,y,xt={cx},yt={cy},r={R})"]
    pars["boundary_conditions"]["BC_values_u"] = [U, U, 0.0, 0.0]
    pars["boundary_conditions"]["BC_values_v"] = [0.0, 0.0, 0.0, 0.0]
    dt = 0.1*dx/U; pars["solver"]["dt"] = dt
    pars["solver"]["convection_method"] = "abdquickest"; pars["solver"]["nt"] = int(t_stop/dt)+1
    pars["output"]["save_frames"] = False
    save_path = f"{output_base}/Nx{Nx}/"
    s = FluidSolver(pars, dtype=torch.float32, compute_forces=False)
    s.fluid_step = types.MethodType(fluid_step_consistent, s)
    s.save_path = save_path; os.makedirs(save_path, exist_ok=True)
    s.set_initial_conditions(); u, v, p = s.u0, s.v0, s.p0
    for it in tqdm(range(0, s.nt), desc=f"Nx={Nx}"):
        u, v, p, _ = s.step_(u, v, p, it, it*s.dt)
    uvp = f"{save_path}/uv_field"; os.makedirs(uvp, exist_ok=True)
    np.save(f"{uvp}/u", u.cpu().numpy()); np.save(f"{uvp}/v", v.cpu().numpy()); np.save(f"{uvp}/p", p.cpu().numpy())
    print(f"  saved -> {uvp}", flush=True)
    del s, u, v, p; torch.cuda.empty_cache(); gc.collect()
print("\nBDIM-consistent prototype done.", flush=True)
