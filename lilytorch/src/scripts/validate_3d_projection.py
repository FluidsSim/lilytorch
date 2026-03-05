"""Quick validation: run 3D NS with projection enabled (small grid)."""
import torch, sys, yaml, time, copy
sys.path.insert(0, "/data/andreaferrario/lilytorch")

# Load config and override for quick test
with open("/data/andreaferrario/lilytorch/lilytorch/src/scripts/flow_past_sphere_3d.yaml") as f:
    config = yaml.safe_load(f)

config['solver']['Nx'] = 96
config['solver']['Ny'] = 48
config['solver']['Nz'] = 48
config['solver']['nt'] = 50
config['solver']['skip_projection'] = False
config['solver']['use_gpu'] = True
config['output']['save_frames'] = False
config['output']['save_uv'] = False

print("Config: 96x48x48, 50 steps, skip_projection=False")
print(f"dt={config['solver']['dt']}, nu={config['solver']['nu']}")

from lilytorch.src.solver import FluidSolver
solver = FluidSolver(config)
print(f"Device: {solver.device}")

t0 = time.time()
u = solver.u0.clone()
v = solver.v0.clone()
w = solver.w0.clone()
p = solver.p0.clone()

for step in range(config['solver']['nt']):
    t = step * solver.dt
    (u, v, p, w, stop_sim) = solver.step_(u, v, p, step, t, w_vel=w)

    umax = u.abs().max().item()
    vmax = v.abs().max().item()
    wmax = w.abs().max().item()
    pmax = p.abs().max().item()

    if step < 5 or step % 10 == 0 or step == config['solver']['nt']-1:
        print(f"  step {step:3d}: |u|={umax:.6f}  |v|={vmax:.6f}  |w|={wmax:.6f}  |p|={pmax:.6e}")

    if not all(torch.isfinite(f).all() for f in [u, v, w, p]):
        print(f"  *** BLOWUP at step {step}! ***")
        break
else:
    print(f"\nCompleted {config['solver']['nt']} steps in {time.time()-t0:.1f}s — STABLE!")
