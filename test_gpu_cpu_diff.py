"""Diagnostic: compare GPU vs CPU for the key operations in the sphere drop solver."""
import torch
import numpy as np

dtype = torch.float64

# Grid params from simulation_config.yaml
Nx, Ny = 256, 2048
xmin, xmax = -0.02, 0.02
ymin, ymax = 0.0, 0.32
dx = (xmax - xmin) / Nx
dy = (ymax - ymin) / Ny

print(f"dx={dx}, dy={dy}, dx==dy: {abs(dx-dy)<1e-15}")
print(f"Expected grid: ({Nx+2}, {Ny+2}) = (258, 2050)")

# 1. Grid construction
x_cpu = torch.linspace(xmin - dx/2, xmax + dx/2, Nx+2, device='cpu', dtype=dtype)
y_cpu = torch.linspace(ymin - dx/2, ymax + dx/2, Ny+2, device='cpu', dtype=dtype)
x_gpu = torch.linspace(xmin - dx/2, xmax + dx/2, Nx+2, device='cuda', dtype=dtype)
y_gpu = torch.linspace(ymin - dx/2, ymax + dx/2, Ny+2, device='cuda', dtype=dtype)

print(f"\n=== 1. Grid construction ===")
print(f"x: CPU len={len(x_cpu)}, GPU len={len(x_gpu)}, max diff={torch.max(torch.abs(x_cpu - x_gpu.cpu())):.2e}")
print(f"y: CPU len={len(y_cpu)}, GPU len={len(y_gpu)}, max diff={torch.max(torch.abs(y_cpu - y_gpu.cpu())):.2e}")

# 2. SDF computation (circle)
X_cpu, Y_cpu = torch.meshgrid(x_cpu, y_cpu, indexing="ij")
X_gpu, Y_gpu = torch.meshgrid(x_gpu, y_gpu, indexing="ij")

radius = 0.0025
# Simulated sphere at position (0, 0.3)
cx, cy = 0.0, 0.3
sdf_cpu = torch.sqrt((X_cpu - cx)**2 + (Y_cpu - cy)**2) - radius
sdf_gpu = torch.sqrt((X_gpu - cx)**2 + (Y_gpu - cy)**2) - radius

diff_sdf = torch.max(torch.abs(sdf_cpu - sdf_gpu.cpu()))
print(f"\n=== 2. SDF computation ===")
print(f"SDF max diff: {diff_sdf:.2e}")

# 3. torch.gradient (used in compute_sdf_properties)
h_float = float(dx)  # Body stores self.h = float(x[1]-x[0])
h = torch.tensor(dx, dtype=dtype)
h_gpu = torch.tensor(dx, dtype=dtype, device='cuda')

gradx_cpu, grady_cpu = torch.gradient(sdf_cpu, spacing=[h_float, h_float], edge_order=2)
gradx_gpu, grady_gpu = torch.gradient(sdf_gpu, spacing=[h_float, h_float], edge_order=2)

diff_gradx = torch.max(torch.abs(gradx_cpu - gradx_gpu.cpu()))
diff_grady = torch.max(torch.abs(grady_cpu - grady_gpu.cpu()))
print(f"\n=== 3. torch.gradient (SDF normals) ===")
print(f"gradx max diff: {diff_gradx:.2e}")
print(f"grady max diff: {diff_grady:.2e}")

# 4. mu_funcs: 0.5 + 0.5 * erf(sdf / (sqrt(2) * eps))
# Body.mu_funcs uses smoothed Heaviside
eps = 2 * h
eps_gpu = 2 * h_gpu

# Check what mu_funcs actually computes
norm_cpu = torch.sqrt(gradx_cpu**2 + grady_cpu**2)
norm_gpu = torch.sqrt(gradx_gpu**2 + grady_gpu**2)

# Also check gradient itself on a constant field (should be zero)
const_cpu = torch.ones_like(sdf_cpu)
const_gpu = torch.ones_like(sdf_gpu)
gc_cpu = torch.gradient(const_cpu, spacing=[h_float, h_float], edge_order=2)
gc_gpu = torch.gradient(const_gpu, spacing=[h_float, h_float], edge_order=2)
print(f"gradient of constant field max: CPU={max(g.abs().max() for g in gc_cpu):.2e}, GPU={max(g.abs().max() for g in gc_gpu):.2e}")
nrm_diff = torch.max(torch.abs(norm_cpu - norm_gpu.cpu()))
print(f"norm max diff: {nrm_diff:.2e}")

# normalized gradients
nx_cpu = torch.where(norm_cpu > 0, gradx_cpu / norm_cpu, torch.tensor(0.0, dtype=dtype))
nx_gpu = torch.where(norm_gpu > 0, gradx_gpu / norm_gpu, torch.tensor(0.0, dtype=dtype, device='cuda'))
diff_nx = torch.max(torch.abs(nx_cpu - nx_gpu.cpu()))
print(f"normal_x max diff: {diff_nx:.2e}")

# 5. Poisson solve comparison
# Create a simple test divergence field
div_cpu = torch.randn(Nx, Ny, dtype=dtype, device='cpu')
div_gpu = div_cpu.clone().to('cuda')
p0_cpu = torch.zeros(Nx+2, Ny+2, dtype=dtype, device='cpu')
p0_gpu = torch.zeros(Nx+2, Ny+2, dtype=dtype, device='cuda')

# Import poisson solver
import sys
sys.path.insert(0, '/data/andreaferrario/lilytorch')
from lilytorch.src.poisson_mult import PoissonSolver

ps_cpu = PoissonSolver(dtype, torch.device('cpu'), h, tol=1e-7, max_cycles=5, max_vcycles=3, nsmoothing=5, w=0.7, verbose=False)
ps_gpu = PoissonSolver(dtype, torch.device('cuda'), h_gpu, tol=1e-7, max_cycles=5, max_vcycles=3, nsmoothing=5, w=0.7, verbose=False)

# Uniform coefficient (like the controller uses with uniform rho)
c_cpu = torch.ones(Nx+2, Ny+2, dtype=dtype, device='cpu')
c_gpu = torch.ones(Nx+2, Ny+2, dtype=dtype, device='cuda')
coeff_val = 0.0001 / 996.0  # timestep / rho
coeff_cpu = coeff_val * c_cpu
coeff_gpu = coeff_val * c_gpu

ch_cpu = coeff_cpu[1:, 1:-1]
cv_cpu = coeff_cpu[1:-1, 1:]
ch_gpu = coeff_gpu[1:, 1:-1]
cv_gpu = coeff_gpu[1:-1, 1:]

print(f"\n=== 5. Poisson solver ===")
p_cpu_out, r_cpu = ps_cpu.solve_multigrid(div_cpu, p0_cpu, coeff_cpu, ch=ch_cpu, cv=cv_cpu)
p_gpu_out, r_gpu = ps_gpu.solve_multigrid(div_gpu, p0_gpu, coeff_gpu, ch=ch_gpu, cv=cv_gpu)

diff_p = torch.max(torch.abs(p_cpu_out - p_gpu_out.cpu()))
diff_r = torch.max(torch.abs(r_cpu - r_gpu.cpu()))
print(f"Pressure max diff: {diff_p:.2e}")
print(f"Residual max diff: {diff_r:.2e}")
print(f"CPU residual norm: {torch.sqrt((r_cpu**2).sum()):.2e}")
print(f"GPU residual norm: {torch.sqrt((r_gpu**2).sum().cpu()):.2e}")

# 6. Advection-diffusion step comparison
from lilytorch.src.adv_diff import AdvDiffSolver

nu = torch.tensor(8e-7, dtype=dtype)
dt_val = torch.tensor(0.0001, dtype=dtype)
nu_gpu = torch.tensor(8e-7, dtype=dtype, device='cuda')
dt_gpu = torch.tensor(0.0001, dtype=dtype, device='cuda')

ad_cpu = AdvDiffSolver('cpu', dt_val, x_cpu, y_cpu, nu,
    BC_type_u=["N","N","N","N"], BC_values_u=[0,0,0,0],
    BC_type_v=["N","N","N","N"], BC_values_v=[0,0,0,0],
    method="abdquickest")
ad_gpu = AdvDiffSolver('cuda', dt_gpu, x_gpu, y_gpu, nu_gpu,
    BC_type_u=["N","N","N","N"], BC_values_u=[0,0,0,0],
    BC_type_v=["N","N","N","N"], BC_values_v=[0,0,0,0],
    method="abdquickest")

# Initialize with a small velocity field (vortex near sphere)
u_init = torch.zeros(Nx+2, Ny+2, dtype=dtype)
v_init = torch.zeros(Nx+2, Ny+2, dtype=dtype)
# Add a small perturbation near the sphere
r_grid = torch.sqrt(X_cpu**2 + (Y_cpu - 0.3)**2)
mask = (r_grid < 0.01) & (r_grid > 0.003)
v_init[mask] = -0.01

u_cpu_in = u_init.clone()
v_cpu_in = v_init.clone()
u_gpu_in = u_init.clone().to('cuda')
v_gpu_in = v_init.clone().to('cuda')

print(f"\n=== 6. Advection-diffusion ===")
u_cpu_out, v_cpu_out = ad_cpu.solve(u_cpu_in, v_cpu_in)
u_gpu_out, v_gpu_out = ad_gpu.solve(u_gpu_in, v_gpu_in)

diff_u = torch.max(torch.abs(u_cpu_out - u_gpu_out.cpu()))
diff_v = torch.max(torch.abs(v_cpu_out - v_gpu_out.cpu()))
print(f"u max diff after advection: {diff_u:.2e}")
print(f"v max diff after advection: {diff_v:.2e}")

# 7. Check curvature computation (known GPU issue with nested torch.gradient)
print(f"\n=== 7. Curvature computation ===")
numerator_cpu = (
    (grady_cpu**2)*torch.gradient(gradx_cpu, spacing=h_float, dim=0, edge_order=2)[0]+
    (gradx_cpu**2)*torch.gradient(grady_cpu, spacing=h_float, dim=1, edge_order=2)[0]+
    -2*gradx_cpu*grady_cpu*torch.gradient(grady_cpu, spacing=h_float, dim=0)[0]
)
numerator_gpu = (
    (grady_gpu**2)*torch.gradient(gradx_gpu, spacing=h_float, dim=0, edge_order=2)[0]+
    (gradx_gpu**2)*torch.gradient(grady_gpu, spacing=h_float, dim=1, edge_order=2)[0]+
    -2*gradx_gpu*grady_gpu*torch.gradient(grady_gpu, spacing=h_float, dim=0)[0]
)
diff_curv_num = torch.max(torch.abs(numerator_cpu - numerator_gpu.cpu()))
print(f"Curvature numerator max diff: {diff_curv_num:.2e}")

denominator_cpu = norm_cpu**3
denominator_gpu = norm_gpu**3
curv_cpu = torch.where(denominator_cpu > 0, numerator_cpu / denominator_cpu, torch.tensor(0.0, dtype=dtype))
curv_gpu = torch.where(denominator_gpu > 0, numerator_gpu / denominator_gpu, torch.tensor(0.0, dtype=dtype, device='cuda'))
diff_curv = torch.max(torch.abs(curv_cpu - curv_gpu.cpu()))
print(f"Curvature max diff: {diff_curv:.2e}")

# Where is the curvature difference largest?
curv_diff_map = torch.abs(curv_cpu - curv_gpu.cpu())
max_idx = torch.argmax(curv_diff_map.flatten())
ix, iy = max_idx // curv_diff_map.shape[1], max_idx % curv_diff_map.shape[1]
print(f"Max curv diff at ({ix},{iy}), SDF={sdf_cpu[ix,iy]:.6f}, norm={norm_cpu[ix,iy]:.6e}")

# 8. forces_method2 pressure masking test
print(f"\n=== 8. Pressure masking ===")
p_test = torch.randn(Nx+2, Ny+2, dtype=dtype)
p_outer_cpu = torch.where(sdf_cpu < 0, torch.tensor(0.0, dtype=dtype), p_test)
p_outer_gpu = torch.where(sdf_gpu < 0, torch.tensor(0.0, dtype=dtype, device='cuda'), p_test.to('cuda'))
diff_pout = torch.max(torch.abs(p_outer_cpu - p_outer_gpu.cpu()))
print(f"p_outer max diff: {diff_pout:.2e}")

print(f"\n=== SUMMARY ===")
print("If any diff > 1e-10, that's the divergence source.")
print("If all diffs < 1e-14 (float64 ULP), the problem is in time integration / accumulation.")
