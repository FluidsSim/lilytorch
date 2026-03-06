"""Quick re-run with fixed slice positions to verify."""
import torch, sys, yaml, time
sys.path.insert(0, "/data/andreaferrario/lilytorch")

with open("/data/andreaferrario/lilytorch/lilytorch/src/scripts/flow_past_sphere_3d.yaml") as f:
    config = yaml.safe_load(f)

config['solver']['Nx']  = 96
config['solver']['Ny']  = 48
config['solver']['Nz']  = 48
config['solver']['nt']  = 100
config['solver']['skip_projection'] = False
config['solver']['use_gpu'] = True
config['output']['save_frames'] = True
config['output']['save_every']  = 50
config['output']['save_path']   = "/data/andreaferrario/ns_data/test_sphere_3d_fixslice/"
config['output']['save_uv']    = False

from lilytorch.src.solver import FluidSolver
solver = FluidSolver(config, dtype=torch.float64, compute_forces=False)

# Check what slice indices will be used
import numpy as np
x_np = solver.x.cpu().numpy()
y_np = solver.y.cpu().numpy()
z_np = solver.z.cpu().numpy()
ix = int(np.argmin(np.abs(x_np - 0.0)))
jy = int(np.argmin(np.abs(y_np - 0.0)))
kz = int(np.argmin(np.abs(z_np - 0.0)))
print(f"Slice indices: ix={ix} (x={x_np[ix]:.4f}), jy={jy} (y={y_np[jy]:.4f}), kz={kz} (z={z_np[kz]:.4f})")
print(f"Compare: Nx//2={len(x_np)//2} (x={x_np[len(x_np)//2]:.4f})")

solver.run_sim()
print("Done!")
