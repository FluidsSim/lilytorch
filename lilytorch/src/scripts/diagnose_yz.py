"""
Diagnostic: check y-z symmetry of the 3D sphere solution.
For a sphere in uniform x-flow, by axial symmetry we expect:
  - v(x,y,z) == w(x,z,y)   (after swapping y<->z)
  - u(x,y,z) == u(x,z,y)
  - p(x,y,z) == p(x,z,y)
Also check the vorticity slices for each component.
"""
import torch, sys, yaml, os, time
import numpy as np
sys.path.insert(0, "/data/andreaferrario/lilytorch")

with open("/data/andreaferrario/lilytorch/lilytorch/src/scripts/flow_past_sphere_3d.yaml") as f:
    config = yaml.safe_load(f)

# Use small grid for quick diagnosis
config['solver']['Nx']  = 96
config['solver']['Ny']  = 48
config['solver']['Nz']  = 48
config['solver']['nt']  = 100
config['solver']['skip_projection'] = False
config['solver']['use_gpu'] = True
config['output']['save_frames'] = False
config['output']['save_uv']    = False

from lilytorch.src.solver import FluidSolver
solver = FluidSolver(config, dtype=torch.float64, compute_forces=False)

u = solver.u0.clone()
v = solver.v0.clone()
w = solver.w0.clone()
p = solver.p0.clone()

Ns = config['solver']['nt']
for step in range(Ns):
    t = step * solver.dt
    (u, v, p, w, _) = solver.step_(u, v, p, step, t, w_vel=w)

print(f"Completed {Ns} steps.")
print(f"u range: [{u.min():.6f}, {u.max():.6f}]")
print(f"v range: [{v.min():.6f}, {v.max():.6f}]")
print(f"w range: [{w.min():.6f}, {w.max():.6f}]")
print(f"p range: [{p.min():.6f}, {p.max():.6f}]")

# --- symmetry checks ---
# With Ny==Nz and ymin==zmin, ymax==zmax, if the grid + BCs are symmetric
# then v(x,y,z) should equal w(x,z,y).
# torch array indexing: u[i,j,k] where i=x, j=y, k=z

# Check u: u(x,y,z) vs u(x,z,y)
u_yz_swap = u.permute(0, 2, 1)  # swap y<->z dims
asym_u = (u - u_yz_swap).abs()
print(f"\nSymmetry check u(x,y,z) vs u(x,z,y):")
print(f"  max|diff| = {asym_u.max():.6e},  mean|diff| = {asym_u.mean():.6e}")

# Check v vs w: v(x,y,z) should ~ w(x,z,y)
w_yz_swap = w.permute(0, 2, 1)
asym_vw = (v - w_yz_swap).abs()
print(f"Symmetry check v(x,y,z) vs w(x,z,y):")
print(f"  max|diff| = {asym_vw.max():.6e},  mean|diff| = {asym_vw.mean():.6e}")

# Check p: p(x,y,z) vs p(x,z,y)
p_yz_swap = p.permute(0, 2, 1)
asym_p = (p - p_yz_swap).abs()
print(f"Symmetry check p(x,y,z) vs p(x,z,y):")
print(f"  max|diff| = {asym_p.max():.6e},  mean|diff| = {asym_p.mean():.6e}")

# --- vorticity components ---
h = solver.h
ox = torch.zeros_like(u)
ox[1:-1,1:-1,1:-1] = (
    (w[1:-1,1:-1,1:-1] - w[1:-1,:-2,1:-1])/h -
    (v[1:-1,1:-1,1:-1] - v[1:-1,1:-1,:-2])/h
)
oy = torch.zeros_like(u)
oy[1:-1,1:-1,1:-1] = (
    (u[1:-1,1:-1,1:-1] - u[1:-1,1:-1,:-2])/h -
    (w[1:-1,1:-1,1:-1] - w[:-2,1:-1,1:-1])/h
)
oz = torch.zeros_like(u)
oz[1:-1,1:-1,1:-1] = (
    (v[1:-1,1:-1,1:-1] - v[:-2,1:-1,1:-1])/h -
    (u[1:-1,1:-1,1:-1] - u[1:-1,:-2,1:-1])/h
)
omag = torch.sqrt(ox**2 + oy**2 + oz**2)

print(f"\nVorticity components (interior):")
print(f"  omega_x range: [{ox[1:-1,1:-1,1:-1].min():.6e}, {ox[1:-1,1:-1,1:-1].max():.6e}]")
print(f"  omega_y range: [{oy[1:-1,1:-1,1:-1].min():.6e}, {oy[1:-1,1:-1,1:-1].max():.6e}]")
print(f"  omega_z range: [{oz[1:-1,1:-1,1:-1].min():.6e}, {oz[1:-1,1:-1,1:-1].max():.6e}]")
print(f"  |omega| range: [{omag[1:-1,1:-1,1:-1].min():.6e}, {omag[1:-1,1:-1,1:-1].max():.6e}]")

# Mid-plane slices
nx, ny, nz = u.shape
ix = nx // 2
jy = ny // 2
kz = nz // 2
print(f"\nSlice indices: ix={ix}, jy={jy}, kz={kz}")

# XY slice (z=mid): should see omega_z wake pattern
print(f"\nXY slice (z={kz}):")
print(f"  omega_z range: [{oz[:,:,kz].min():.6e}, {oz[:,:,kz].max():.6e}]")
print(f"  |omega| range: [{omag[:,:,kz].min():.6e}, {omag[:,:,kz].max():.6e}]")

# XZ slice (y=mid): should see omega_y wake pattern
print(f"\nXZ slice (y={jy}):")
print(f"  omega_y range: [{oy[:,jy,:].min():.6e}, {oy[:,jy,:].max():.6e}]")
print(f"  |omega| range: [{omag[:,jy,:].min():.6e}, {omag[:,jy,:].max():.6e}]")

# YZ slice (x=downstream): should see roughly axisymmetric ring
print(f"\nYZ slice (x={ix}):")
print(f"  omega_x range: [{ox[ix,:,:].min():.6e}, {ox[ix,:,:].max():.6e}]")
print(f"  |omega| range: [{omag[ix,:,:].min():.6e}, {omag[ix,:,:].max():.6e}]")

# Check what |omega| looks like at x=downstream for the YZ slice
yz_slice = omag[ix,:,:].cpu().numpy()
print(f"\n  YZ |omega| stats: mean={yz_slice.mean():.6e}, std={yz_slice.std():.6e}")
print(f"  top-left corner (j=0:3,k=0:3): {yz_slice[:3,:3]}")
print(f"  center (j={jy-1}:{jy+2},k={kz-1}:{kz+2}): {yz_slice[jy-1:jy+2,kz-1:kz+2]}")

# ** Check divergence field **
div = solver.divergence(u, v, w)
print(f"\nDivergence (interior):")
print(f"  range: [{div[1:-1,1:-1,1:-1].min():.6e}, {div[1:-1,1:-1,1:-1].max():.6e}]")
print(f"  mean : {div[1:-1,1:-1,1:-1].mean():.6e}")
print(f"  L2   : {div[1:-1,1:-1,1:-1].pow(2).mean().sqrt():.6e}")

# Check mu0 symmetry
print(f"\nmu0 symmetry:")
mu0_yz = solver.mu0_all.permute(0, 2, 1)
print(f"  mu0 range: [{solver.mu0_all.min():.4f}, {solver.mu0_all.max():.4f}]")
print(f"  max|mu0(x,y,z)-mu0(x,z,y)| = {(solver.mu0_all - mu0_yz).abs().max():.6e}")

# Similarly for mu0_u, mu0_v vs mu0_w
mu0u_yz = solver.mu0_all_u.permute(0, 2, 1)
mu0v_yz = solver.mu0_all_v.permute(0, 2, 1)
mu0w_yz = solver.mu0_all_w.permute(0, 2, 1)
print(f"  max|mu0_u(x,y,z)-mu0_u(x,z,y)| = {(solver.mu0_all_u - mu0u_yz).abs().max():.6e}")
print(f"  max|mu0_v(x,y,z)-mu0_w(x,z,y)| = {(solver.mu0_all_v - mu0w_yz).abs().max():.6e}")
