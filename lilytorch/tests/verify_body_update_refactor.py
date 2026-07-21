"""End-to-end verification of the single fused fluid path.

Test A — solver_method deprecation: the python/kernel modes are gone; the
  key must be accepted-but-ignored.  Run the same static-cylinder config
  once without the key and once with the deprecated solver_method='python',
  assert a DeprecationWarning fires and the results are bit-identical
  (both runs are the fused step consuming the python-style fields
  published by the standalone composite update — the jellyfish enabler).

Test B — streaming wiring: drive BDIMhandler._update_streaming_multi +
  _launch_body_update with a FARMS-free fake handler, then run the fused
  step, and check the imposed body velocity + scratch-field release.
"""
import types
import warnings
import numpy as np
import torch

from lilytorch.util.yaml_operations import yaml2pyobject
from lilytorch.src.solver import FluidSolver

NSTEPS = 5


def make_pars(method=None):
    pars = yaml2pyobject("lilytorch/src/configs/flow_past_cylinder.yaml")
    s = pars["solver"]
    s["Nx"] = s["Ny"] = 96
    s["xmin"], s["xmax"] = -0.5, 0.5
    s["ymin"], s["ymax"] = -0.5, 0.5
    s["nt"] = NSTEPS
    s["dt"] = 5e-3
    s["convection_method"] = "quick"
    s["poisson_method"] = "mgcg"
    s["poisson_verbose"] = False
    if method is not None:
        s["solver_method"] = method
    pars["output"]["save_frames"] = False
    pars["output"]["save_uv"] = False
    pars["output"]["vmin"] = "auto"
    pars["output"]["vmax"] = "auto"
    return pars


def run_solver(method=None, nsteps=NSTEPS):
    torch.manual_seed(0)
    fs = FluidSolver(make_pars(method), compute_forces=False)
    u, v, p = fs.u0.clone(), fs.v0.clone(), fs.p0.clone()
    for it in range(nsteps):
        t = it * fs.dt
        u, v, p, _ = fs.advance_and_compute_loads(u, v, p, it, t)
    return fs, u.clone(), v.clone(), p.clone()


print("=" * 70)
print("Test A: solver_method deprecated-and-ignored; single fused path")
print("=" * 70)
fs_df, u_df, v_df, p_df = run_solver(None)
with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter("always")
    fs_py, u_py, v_py, p_py = run_solver("python")
assert any(issubclass(w.category, DeprecationWarning)
           and "solver_method" in str(w.message) for w in _w), \
    "deprecated solver_method did not raise a DeprecationWarning"
du = (u_py - u_df).abs().max().item()
dv = (v_py - v_df).abs().max().item()
dp = (p_py - p_df).abs().max().item()
uref = u_df.abs().max().item()
print(f"  max|du|={du:.3e}  max|dv|={dv:.3e}  max|dp|={dp:.3e}  (|u|max={uref:.3e})")
assert du == 0.0 and dv == 0.0 and dp == 0.0, \
    "deprecated solver_method changed the results (must be ignored)"
assert torch.isfinite(u_df).all() and torch.isfinite(p_df).all(), "non-finite fused output"
print("  -> DeprecationWarning fired; results bit-identical; finite output.")

print()
print("=" * 70)
print("Test B: streaming body_update via fake BDIMhandler -> fused step")
print("=" * 70)
from lilytorch.integration.BDIMhandler import BDIMhandler

fs = FluidSolver(make_pars(), compute_forces=False)
comp = fs.composite_body
D = 2

# fake handler with just the state _update_streaming_multi/_launch_body_update need
fake = types.SimpleNamespace(
    fluid_solver=fs, ndim=D, device=fs.device, dtype=fs.dtype,
    dtype_np=np.float32 if fs.dtype == torch.float32 else np.float64,
    force_method="eulerian", _sim_axes=range(D), _bu_bufs=None,
)
for name in ("_update_streaming_multi", "_launch_body_update",
             "_body_update_bufs", "_stream_kin_static", "_stream_static_pack",
             "_init_interp", "_init_static_body_metadata", "_make_stream_meta",
             "_stream_lagrangian_refresh", "_body_local_aabb_2d"):
    raw = BDIMhandler.__dict__[name]
    if isinstance(raw, staticmethod):
        setattr(fake, name, raw.__func__)
    else:
        setattr(fake, name, types.MethodType(raw, fake))

VX = 0.05  # imposed rigid-body x velocity


def gather_data(iteration):
    com = [np.zeros((1, 2), dtype=fake.dtype_np)]
    urdf = [np.zeros((1, 2), dtype=fake.dtype_np)]
    Rs = [np.eye(2, dtype=fake.dtype_np)[None]]
    lin = [np.array([[VX, 0.0]], dtype=fake.dtype_np)]
    ang = [np.zeros((1,), dtype=fake.dtype_np)]
    return com, urdf, Rs, lin, ang


fake.gather_data = gather_data
comp.body_ids = [(0, 0)]

fake._init_interp()
fake._init_static_body_metadata()
fake._update_streaming_multi(0.0, 0)

assert comp.sdf_val_u is not None and comp.body_u is not None, "contract fields missing"
assert comp.bdim_dirty is not None, "dirty rect missing"
print(f"  dirty rect: {comp.bdim_dirty}")
inside = comp.sdf_val < -2 * float(fs.h)
print(f"  cells inside body: {int(inside.sum())}")
bu_in = comp.body_u[comp.sdf_val_u < -2 * float(fs.h)]
print(f"  body_u inside body: min={bu_in.min().item():.4f} max={bu_in.max().item():.4f} (expected {VX})")
assert torch.allclose(bu_in, torch.full_like(bu_in, VX), atol=1e-5), "body velocity not imposed"

u, v, p = fs.u0.clone(), fs.v0.clone(), fs.p0.clone()
u2, v2, p2 = fs.fluid_step(u, v, p, fs.dt)
assert torch.isfinite(u2).all() and torch.isfinite(p2).all(), "non-finite streaming output"
assert comp.sdf_val_u is None and comp.body_u is None, "scratch fields not released"
u_in = u2[inside]
print(f"  post-step u inside body: mean={u_in.mean().item():.4f} (expected ~{VX})")
assert abs(u_in.mean().item() - VX) < 0.2 * abs(VX), "body velocity not imposed by fused step"
print("  -> streaming body_update published contract fields; fused step consumed + released them.")
print()
print("ALL CHECKS PASSED")
