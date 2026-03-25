"""Diagnostic: compare GPU vs CPU for the key operations in the sphere drop solver.

Tests cover all operations fixed in fix/cpu-gpu-parity:
  1. Grid construction
  2. SDF computation
  3. SDF normals - explicit central FD (NOT torch.gradient)
  4. Norm / inverse-norm determinism
  5. Poisson solver (multigrid + float64 mean subtraction)
  6. Advection-diffusion
  7. Curvature - explicit FD (NOT torch.gradient)
  8. Pressure masking (zeros_like)
  9. Force-sum determinism  (float32 vs float64 accumulation)
 10. Pressure-mean subtraction (float64 accumulation)
 11. Polynomial sin/cos vs torch.sin/cos parity
 12. Multiply-by-reciprocal gradient / divergence vs division
"""
import torch
import numpy as np

dtype = torch.float32   # sphere drop uses float32

# Grid params from simulation_config.yaml
Nx, Ny = 256, 2048
xmin, xmax = -0.02, 0.02
ymin, ymax = 0.0, 0.32
dx = (xmax - xmin) / Nx
dy = (ymax - ymin) / Ny

print(f"dx={dx}, dy={dy}, dx==dy: {abs(dx-dy)<1e-15}")
print(f"Expected grid: ({Nx+2}, {Ny+2}) = (258, 2050)")
print(f"dtype: {dtype}")

h_float = float(dx)
h       = torch.tensor(dx, dtype=dtype)
h_gpu   = torch.tensor(dx, dtype=dtype, device='cuda')

# ---------------------------------------------------------------------------
# 1. Grid construction
# ---------------------------------------------------------------------------
x_cpu = torch.linspace(xmin - dx/2, xmax + dx/2, Nx+2, device='cpu',  dtype=dtype)
y_cpu = torch.linspace(ymin - dx/2, ymax + dx/2, Ny+2, device='cpu',  dtype=dtype)
x_gpu = torch.linspace(xmin - dx/2, xmax + dx/2, Nx+2, device='cuda', dtype=dtype)
y_gpu = torch.linspace(ymin - dx/2, ymax + dx/2, Ny+2, device='cuda', dtype=dtype)

print(f"\n=== 1. Grid construction ===")
print(f"x: CPU len={len(x_cpu)}, GPU len={len(x_gpu)}, max diff={torch.max(torch.abs(x_cpu - x_gpu.cpu())):.2e}")
print(f"y: CPU len={len(y_cpu)}, GPU len={len(y_gpu)}, max diff={torch.max(torch.abs(y_cpu - y_gpu.cpu())):.2e}")

# ---------------------------------------------------------------------------
# 2. SDF computation
# ---------------------------------------------------------------------------
X_cpu, Y_cpu = torch.meshgrid(x_cpu, y_cpu, indexing="ij")
X_gpu, Y_gpu = torch.meshgrid(x_gpu, y_gpu, indexing="ij")

radius = 0.0025
cx, cy = 0.0, 0.3
sdf_cpu = torch.sqrt((X_cpu - cx)**2 + (Y_cpu - cy)**2) - radius
sdf_gpu = torch.sqrt((X_gpu - cx)**2 + (Y_gpu - cy)**2) - radius

diff_sdf = torch.max(torch.abs(sdf_cpu - sdf_gpu.cpu()))
print(f"\n=== 2. SDF computation ===")
print(f"SDF max diff: {diff_sdf:.2e}")

# ---------------------------------------------------------------------------
# 3. SDF normals — explicit central FD matching body.py:compute_sdf_properties
#    (replaces old torch.gradient test)
# ---------------------------------------------------------------------------
def compute_normals_fd(sdf, h_inv):
    """Central-difference SDF gradients — mirrors body.py compute_sdf_properties."""
    gradx = torch.zeros_like(sdf)
    grady = torch.zeros_like(sdf)
    gradx[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) * (0.5 * h_inv)
    grady[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) * (0.5 * h_inv)
    # one-sided at boundaries
    gradx[0,  :] = (sdf[1,  :] - sdf[0,  :]) * h_inv
    gradx[-1, :] = (sdf[-1, :] - sdf[-2, :]) * h_inv
    grady[:, 0 ] = (sdf[:, 1 ] - sdf[:, 0 ]) * h_inv
    grady[:, -1] = (sdf[:, -1] - sdf[:, -2]) * h_inv

    norm    = torch.sqrt(gradx**2 + grady**2)
    inv_norm = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))
    nx = gradx * inv_norm
    ny = grady * inv_norm
    return gradx, grady, norm, nx, ny

h_inv_cpu = h.reciprocal()
h_inv_gpu = h_gpu.reciprocal()

gradx_cpu, grady_cpu, norm_cpu, nx_cpu, ny_cpu = compute_normals_fd(sdf_cpu, h_inv_cpu)
gradx_gpu, grady_gpu, norm_gpu, nx_gpu, ny_gpu = compute_normals_fd(sdf_gpu, h_inv_gpu)

print(f"\n=== 3. SDF normals (explicit central FD) ===")
print(f"gradx  max diff: {torch.max(torch.abs(gradx_cpu - gradx_gpu.cpu())):.2e}")
print(f"grady  max diff: {torch.max(torch.abs(grady_cpu - grady_gpu.cpu())):.2e}")
print(f"norm   max diff: {torch.max(torch.abs(norm_cpu  - norm_gpu.cpu())):.2e}")
print(f"nx     max diff: {torch.max(torch.abs(nx_cpu    - nx_gpu.cpu())):.2e}")
print(f"ny     max diff: {torch.max(torch.abs(ny_cpu    - ny_gpu.cpu())):.2e}")

# ---------------------------------------------------------------------------
# 4. Multiply-by-reciprocal: norm inversion determinism
# ---------------------------------------------------------------------------
print(f"\n=== 4. Reciprocal vs division (norm inversion) ===")
# divide path
nx_div_cpu = torch.where(norm_cpu > 0, gradx_cpu / norm_cpu, torch.tensor(0.0, dtype=dtype))
nx_div_gpu = torch.where(norm_gpu > 0, gradx_gpu / norm_gpu, torch.tensor(0.0, dtype=dtype, device='cuda'))
# reciprocal path (matches body.py fix)
nx_rec_cpu = gradx_cpu * torch.where(norm_cpu > 0, norm_cpu.reciprocal(), torch.zeros_like(norm_cpu))
nx_rec_gpu = gradx_gpu * torch.where(norm_gpu > 0, norm_gpu.reciprocal(), torch.zeros_like(norm_gpu))

print(f"nx  division path:   CPU-GPU max diff = {torch.max(torch.abs(nx_div_cpu - nx_div_gpu.cpu())):.2e}")
print(f"nx  reciprocal path: CPU-GPU max diff = {torch.max(torch.abs(nx_rec_cpu - nx_rec_gpu.cpu())):.2e}")
print(f"(reciprocal path should be <= division path)")

# ---------------------------------------------------------------------------
# 5. Poisson solver (multigrid + float64 mean subtraction)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, '/home/user/lilytorch')
from lilytorch.src.poisson_mult import PoissonSolver

div_cpu = torch.randn(Nx, Ny, dtype=dtype, device='cpu')
div_gpu = div_cpu.clone().to('cuda')
p0_cpu  = torch.zeros(Nx+2, Ny+2, dtype=dtype, device='cpu')
p0_gpu  = torch.zeros(Nx+2, Ny+2, dtype=dtype, device='cuda')

ps_cpu = PoissonSolver(dtype, torch.device('cpu'),  h,     tol=1e-4, max_cycles=5, max_vcycles=3, nsmoothing=5, w=0.7, verbose=False)
ps_gpu = PoissonSolver(dtype, torch.device('cuda'), h_gpu, tol=1e-4, max_cycles=5, max_vcycles=3, nsmoothing=5, w=0.7, verbose=False)

coeff_val = 0.0001 / 996.0
c_cpu  = coeff_val * torch.ones(Nx+2, Ny+2, dtype=dtype, device='cpu')
c_gpu  = coeff_val * torch.ones(Nx+2, Ny+2, dtype=dtype, device='cuda')
ch_cpu = c_cpu[1:, 1:-1];  cv_cpu = c_cpu[1:-1, 1:]
ch_gpu = c_gpu[1:, 1:-1];  cv_gpu = c_gpu[1:-1, 1:]

print(f"\n=== 5. Poisson solver ===")
p_cpu_out, r_cpu = ps_cpu.solve_multigrid(div_cpu, p0_cpu, c_cpu, ch=ch_cpu, cv=cv_cpu)
p_gpu_out, r_gpu = ps_gpu.solve_multigrid(div_gpu, p0_gpu, c_gpu, ch=ch_gpu, cv=cv_gpu)

diff_p = torch.max(torch.abs(p_cpu_out - p_gpu_out.cpu()))
diff_r = torch.max(torch.abs(r_cpu     - r_gpu.cpu()))
print(f"Pressure max diff:  {diff_p:.2e}")
print(f"Residual  max diff: {diff_r:.2e}")
print(f"CPU residual norm:  {torch.sqrt((r_cpu**2).sum()):.2e}")
print(f"GPU residual norm:  {torch.sqrt((r_gpu**2).sum().cpu()):.2e}")

# ---------------------------------------------------------------------------
# 6. Advection-diffusion
# ---------------------------------------------------------------------------
from lilytorch.src.adv_diff import AdvDiffSolver

nu_cpu  = torch.tensor(8e-7, dtype=dtype)
dt_val  = torch.tensor(0.0001, dtype=dtype)
nu_gpu  = torch.tensor(8e-7, dtype=dtype, device='cuda')
dt_gpu  = torch.tensor(0.0001, dtype=dtype, device='cuda')

BC_kw = dict(
    BC_type_u=["N","N","N","N"], BC_values_u=[0,0,0,0],
    BC_type_v=["N","N","N","N"], BC_values_v=[0,0,0,0],
    method="abdquickest"
)
ad_cpu = AdvDiffSolver('cpu',  dt_val, x_cpu, y_cpu, nu_cpu,  **BC_kw)
ad_gpu = AdvDiffSolver('cuda', dt_gpu, x_gpu, y_gpu, nu_gpu,  **BC_kw)

u_init = torch.zeros(Nx+2, Ny+2, dtype=dtype)
v_init = torch.zeros(Nx+2, Ny+2, dtype=dtype)
r_grid = torch.sqrt(X_cpu**2 + (Y_cpu - 0.3)**2)
mask   = (r_grid < 0.01) & (r_grid > 0.003)
v_init[mask] = -0.01

u_cpu_out, v_cpu_out = ad_cpu.solve(u_init.clone(), v_init.clone())
u_gpu_out, v_gpu_out = ad_gpu.solve(u_init.clone().to('cuda'), v_init.clone().to('cuda'))

print(f"\n=== 6. Advection-diffusion ===")
print(f"u max diff: {torch.max(torch.abs(u_cpu_out - u_gpu_out.cpu())):.2e}")
print(f"v max diff: {torch.max(torch.abs(v_cpu_out - v_gpu_out.cpu())):.2e}")

# ---------------------------------------------------------------------------
# 7. Curvature — explicit second-derivative FD matching body.py fix
#    (replaces old torch.gradient curvature test)
# ---------------------------------------------------------------------------
def compute_curvature_fd(sdf, h_inv):
    """Explicit FD curvature matching body.py compute_sdf_properties."""
    gx = torch.zeros_like(sdf)
    gy = torch.zeros_like(sdf)
    gx[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) * (0.5 * h_inv)
    gy[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) * (0.5 * h_inv)
    gx[0,  :] = (sdf[1,  :] - sdf[0,  :]) * h_inv
    gx[-1, :] = (sdf[-1, :] - sdf[-2, :]) * h_inv
    gy[:, 0 ] = (sdf[:, 1 ] - sdf[:, 0 ]) * h_inv
    gy[:, -1] = (sdf[:, -1] - sdf[:, -2]) * h_inv

    # second derivatives
    gxx = torch.zeros_like(sdf)
    gyy = torch.zeros_like(sdf)
    gxy = torch.zeros_like(sdf)
    gxx[1:-1, :] = (gx[2:, :] - gx[:-2, :]) * (0.5 * h_inv)
    gyy[:, 1:-1] = (gy[:, 2:] - gy[:, :-2]) * (0.5 * h_inv)
    gxy[1:-1, :] = (gy[2:, :] - gy[:-2, :]) * (0.5 * h_inv)

    norm = torch.sqrt(gx**2 + gy**2)
    numerator   = gy**2 * gxx - 2*gx*gy*gxy + gx**2 * gyy
    denominator = norm**3
    curvature   = torch.where(denominator > 0, numerator * denominator.reciprocal(), torch.zeros_like(numerator))
    return curvature

curv_cpu = compute_curvature_fd(sdf_cpu, h_inv_cpu)
curv_gpu = compute_curvature_fd(sdf_gpu, h_inv_gpu)

diff_curv = torch.max(torch.abs(curv_cpu - curv_gpu.cpu()))
print(f"\n=== 7. Curvature (explicit FD) ===")
print(f"Curvature max diff: {diff_curv:.2e}")

curv_diff_map = torch.abs(curv_cpu - curv_gpu.cpu())
max_idx = torch.argmax(curv_diff_map.flatten())
ix, iy  = max_idx // curv_diff_map.shape[1], max_idx % curv_diff_map.shape[1]
print(f"Max curv diff at ({ix},{iy}), SDF={sdf_cpu[ix,iy]:.6f}, norm={norm_cpu[ix,iy]:.6e}")

# ---------------------------------------------------------------------------
# 8. Pressure masking (zeros_like vs scalar literal)
# ---------------------------------------------------------------------------
p_test      = torch.randn(Nx+2, Ny+2, dtype=dtype)
p_outer_cpu = torch.where(sdf_cpu < 0, torch.zeros_like(p_test),           p_test)
p_outer_gpu = torch.where(sdf_gpu < 0, torch.zeros_like(p_test.to('cuda')), p_test.to('cuda'))
print(f"\n=== 8. Pressure masking (zeros_like) ===")
print(f"p_outer max diff: {torch.max(torch.abs(p_outer_cpu - p_outer_gpu.cpu())):.2e}")

# ---------------------------------------------------------------------------
# 9. Force-sum determinism: float32 vs float64 accumulation
#    Simulates forces_method2 summation near a body interface.
# ---------------------------------------------------------------------------
print(f"\n=== 9. Force-sum determinism (float32 vs float64 accumulation) ===")

# Build a plausible force field: nonzero only near the sphere surface
interface_mask = (sdf_cpu.abs() < 3 * h_float).float()
# Pressure force density (product of pressure and normal)
pforcex_cpu = p_outer_cpu * nx_cpu * interface_mask
pforcex_gpu = pforcex_cpu.to('cuda')
h2 = h_float ** 2

# float32 summation — may differ CPU vs GPU due to tree-reduction ordering
fsum_f32_cpu = pforcex_cpu.sum() * h2
fsum_f32_gpu = (pforcex_gpu.sum() * h2).cpu()
diff_f32 = torch.abs(fsum_f32_cpu - fsum_f32_gpu)

# float64 summation — matches the fix applied in solver.py:forces_method2
fsum_f64_cpu = pforcex_cpu.to(torch.float64).sum().to(dtype) * h2
fsum_f64_gpu = (pforcex_gpu.to(torch.float64).sum().to(dtype) * h2).cpu()
diff_f64 = torch.abs(fsum_f64_cpu - fsum_f64_gpu)

print(f"float32 sum CPU-GPU diff: {diff_f32:.6e}  (before fix)")
print(f"float64 sum CPU-GPU diff: {diff_f64:.6e}  (after fix)")
print(f"Improvement factor: {(diff_f32 / (diff_f64 + 1e-45)):.1e}x")
print(f"Interface cells: {interface_mask.sum().int()} / {Nx*Ny}")

# ---------------------------------------------------------------------------
# 10. Pressure-mean subtraction: float64 accumulation
# ---------------------------------------------------------------------------
print(f"\n=== 10. Pressure-mean subtraction (float64 accumulation) ===")
p_rand_cpu = torch.randn(Nx+2, Ny+2, dtype=dtype)
p_rand_gpu = p_rand_cpu.clone().to('cuda')

# float32 mean — ordering differs CPU vs GPU
mean_f32_cpu = p_rand_cpu.mean()
mean_f32_gpu = p_rand_gpu.mean().cpu()
diff_mean_f32 = torch.abs(mean_f32_cpu - mean_f32_gpu)

# float64 mean — matches the fix in solve_multigrid
mean_f64_cpu = p_rand_cpu.to(torch.float64).mean().to(dtype)
mean_f64_gpu = p_rand_gpu.to(torch.float64).mean().to(dtype).cpu()
diff_mean_f64 = torch.abs(mean_f64_cpu - mean_f64_gpu)

print(f"float32 mean CPU-GPU diff: {diff_mean_f32:.6e}  (before fix)")
print(f"float64 mean CPU-GPU diff: {diff_mean_f64:.6e}  (after fix)")

p_sub_f32_cpu = p_rand_cpu - mean_f32_cpu
p_sub_f32_gpu = p_rand_gpu - mean_f32_gpu.to('cuda')
p_sub_f64_cpu = p_rand_cpu - mean_f64_cpu
p_sub_f64_gpu = p_rand_gpu - mean_f64_gpu.to('cuda')

diff_sub_f32 = torch.max(torch.abs(p_sub_f32_cpu - p_sub_f32_gpu.cpu()))
diff_sub_f64 = torch.max(torch.abs(p_sub_f64_cpu - p_sub_f64_gpu.cpu()))
print(f"p - mean, float32 max field diff: {diff_sub_f32:.6e}  (before fix)")
print(f"p - mean, float64 max field diff: {diff_sub_f64:.6e}  (after fix)")

# ---------------------------------------------------------------------------
# 11. Polynomial sin/cos vs torch.sin/cos — parity and accuracy
# ---------------------------------------------------------------------------
print(f"\n=== 11. Polynomial sin/cos parity ===")

def poly_sin(theta):
    """Degree-9 Maclaurin series for sin — matches body.py rototranslate_points."""
    t2 = theta * theta
    t3 = theta * t2;  t5 = t3 * t2;  t7 = t5 * t2;  t9 = t7 * t2
    return theta + (-1.0/6.0)*t3 + (1.0/120.0)*t5 + (-1.0/5040.0)*t7 + (1.0/362880.0)*t9

def poly_cos(theta):
    """Degree-8 Maclaurin series for cos — matches body.py rototranslate_points."""
    t2 = theta * theta
    return (1.0 + (-1.0/2.0)*t2 + (1.0/24.0)*t2*t2
            + (-1.0/720.0)*t2*t2*t2 + (1.0/40320.0)*t2*t2*t2*t2)

# Range typical for rigid-body rotation in the simulation: ±0.1 rad
angles = torch.linspace(-0.1, 0.1, 200, dtype=dtype)
angles_gpu = angles.to('cuda')

# GPU vs CPU for polynomial (should be zero — pure arithmetic)
ps_cpu = poly_sin(angles)
ps_gpu = poly_sin(angles_gpu)
pc_cpu = poly_cos(angles)
pc_gpu = poly_cos(angles_gpu)

diff_poly_s = torch.max(torch.abs(ps_cpu - ps_gpu.cpu()))
diff_poly_c = torch.max(torch.abs(pc_cpu - pc_gpu.cpu()))
print(f"poly_sin CPU-GPU max diff: {diff_poly_s:.2e}  (should be 0)")
print(f"poly_cos CPU-GPU max diff: {diff_poly_c:.2e}  (should be 0)")

# Accuracy vs torch.sin/cos reference
ref_s = torch.sin(angles)
ref_c = torch.cos(angles)
print(f"poly_sin vs torch.sin max error: {torch.max(torch.abs(ps_cpu - ref_s)):.2e}")
print(f"poly_cos vs torch.cos max error: {torch.max(torch.abs(pc_cpu - ref_c)):.2e}")

# GPU torch.sin vs CPU torch.sin (non-deterministic path, shown for contrast)
ts_diff = torch.max(torch.abs(torch.sin(angles) - torch.sin(angles_gpu).cpu()))
tc_diff = torch.max(torch.abs(torch.cos(angles) - torch.cos(angles_gpu).cpu()))
print(f"torch.sin CPU-GPU max diff: {ts_diff:.2e}  (may be non-zero)")
print(f"torch.cos CPU-GPU max diff: {tc_diff:.2e}  (may be non-zero)")

# ---------------------------------------------------------------------------
# 12. Multiply-by-reciprocal gradient / divergence vs division
# ---------------------------------------------------------------------------
print(f"\n=== 12. Multiply-by-reciprocal: gradient / divergence ===")

# Simulate gradient computation over the pressure field
def grad_divide(p, h_val):
    """Old approach: division by h."""
    gx = torch.zeros_like(p)
    gy = torch.zeros_like(p)
    gx[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * h_val)
    gy[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * h_val)
    return gx, gy

def grad_reciprocal(p, h_inv_val):
    """New approach: multiply by reciprocal — matches solver.py fix."""
    inv2h = (2.0 * h_inv_val) if isinstance(h_inv_val, float) else h_inv_val * 0.5  # h_inv/2 = 1/(2h)
    # Correct: 1/(2h) = h_inv * 0.5
    scale = h_inv_val * torch.tensor(0.5, dtype=h_inv_val.dtype, device=h_inv_val.device) if isinstance(h_inv_val, torch.Tensor) else 0.5 / float(h_float)
    gx = torch.zeros_like(p)
    gy = torch.zeros_like(p)
    gx[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) * scale
    gy[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) * scale
    return gx, gy

p_field_cpu = p_rand_cpu
p_field_gpu = p_rand_gpu

h_scalar = float(h_float)
h_inv_t_cpu = h.reciprocal()
h_inv_t_gpu = h_gpu.reciprocal()

gx_div_cpu, gy_div_cpu = grad_divide(p_field_cpu, h_scalar)
gx_div_gpu, gy_div_gpu = grad_divide(p_field_gpu, h_scalar)
gx_rec_cpu, gy_rec_cpu = grad_reciprocal(p_field_cpu, h_inv_t_cpu)
gx_rec_gpu, gy_rec_gpu = grad_reciprocal(p_field_gpu, h_inv_t_gpu)

diff_gx_div = torch.max(torch.abs(gx_div_cpu - gx_div_gpu.cpu()))
diff_gx_rec = torch.max(torch.abs(gx_rec_cpu - gx_rec_gpu.cpu()))
print(f"gradient (division)    CPU-GPU max diff: {diff_gx_div:.2e}")
print(f"gradient (reciprocal)  CPU-GPU max diff: {diff_gx_rec:.2e}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print(f"\n=== SUMMARY ===")
results = [
    ("1.  Grid construction",           torch.max(torch.abs(x_cpu - x_gpu.cpu()))),
    ("2.  SDF computation",             diff_sdf),
    ("3.  SDF normals (FD)",            torch.max(torch.abs(nx_cpu - nx_gpu.cpu()))),
    ("4.  Reciprocal vs division",      torch.max(torch.abs(nx_rec_cpu - nx_rec_gpu.cpu()))),
    ("5.  Poisson solver",              diff_p),
    ("6.  Advection-diffusion u",       torch.max(torch.abs(u_cpu_out - u_gpu_out.cpu()))),
    ("7.  Curvature (FD)",              diff_curv),
    ("8.  Pressure masking",            torch.max(torch.abs(p_outer_cpu - p_outer_gpu.cpu()))),
    ("9.  Force sum float64 fix",       diff_f64),
    ("10. Pressure mean float64 fix",   diff_mean_f64),
    ("11. Polynomial sin/cos",          diff_poly_s),
    ("12. Gradient reciprocal",         diff_gx_rec),
]
threshold = 1e-5  # float32 ULP neighbourhood
all_pass = True
for name, val in results:
    status = "PASS" if float(val) < threshold else "WARN"
    if status == "WARN":
        all_pass = False
    print(f"  {status}  {name}: {float(val):.2e}")

print(f"\n{'All checks PASS.' if all_pass else 'Some checks WARN — review above.'}")
print("(WARN for Poisson/adv-diff is expected for float32 due to iteration accumulation.)")
