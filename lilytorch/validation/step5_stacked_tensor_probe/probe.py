"""
Step-5 stacked-tensor storage probe.

Scope (todo.md Step 5): test the hypothesis that replacing the per-axis field
tensors on ``FluidSolver`` / ``CompositeBody`` (``u0/v0/w0``, ``sdf_val_u/v/w``,
``mu0/mu1_[uvw]``, ``body_[uvw]``, ``normal_[xyz]``, ``diff_[uvw]``) with single
``(D, *grid_shape)`` tensors is "free or better" on **both** speed and memory.

This is a CPU-only spike on small analytical-body harnesses (cylinder 2-D /
sphere 3-D, no FARMS). The decision gate is hard: the stacked layout must be
``<=`` baseline on **both** axes; any regression on either is NO-GO.

What this probe actually measures
---------------------------------

1. **Resident storage bytes** of every per-axis group on the live
   ``FluidSolver`` / ``CompositeBody`` instances after running each harness.
   For lilytorch's co-located grids, ``stack((u,v,w))`` has exactly the same
   ``element_size * nelement`` as ``u + v + w``, but we measure it explicitly
   to make the by-construction equality a deterministic data point rather
   than an assertion.

2. **Numerical equivalence** of the stack/unbind round-trip on every
   per-axis group sampled at the final step (must be bitwise identical;
   stack is a memcpy, ``.unbind`` is a view).

3. **Per-step wall-clock** of the baseline run vs a "stacked-shadow" run
   that maintains a single ``(D, *grid)`` buffer for every per-axis group
   alongside the existing tuple fields. The shadow's projection cost is an
   **upper bound** on the runtime overhead a real Step-5 refactor would
   pay if the stacked tensor has to be materialised (rather than acting
   as the source-of-truth with per-axis ``.unbind`` views into it).

4. **Microbenchmark** of the candidate hot per-axis operations the codebase
   currently performs in tuple form
   (``u = u*mu0_u; v = v*mu0_v; w = w*mu0_w`` etc.) versus the equivalent
   single-tensor broadcast on a ``(D, *grid)`` stacked tensor. This is the
   best case for stacked layout, because it ignores the question of how
   you reach the stacked layout in the first place.

5. **Integrated forces and kinetic energy** of the cylinder/sphere harness
   in both runs — the refactor must not change physics.

Memory verdict for this codebase
--------------------------------

All per-axis fields in lilytorch share the same ``grid_shape`` (co-located
cell-centred grid; see ``FluidSolver`` init at solver.py:794-800 and
``BodyAnalytical``/``CompositeBodyAnalytical`` field allocations). So the
resident-byte count of a tuple ``(u, v, w)`` is identical to that of a
stacked ``(D, *grid)`` tensor; the gate is decided on transient/working
memory and on speed.

CPU caveat
----------

Per todo.md and the agent instructions, CPU wall-clock numbers are only
indicative of GPU behaviour. The decisive metric is the deterministic byte
count of resident field tensors; CPU timing is reported as a sanity check.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Apply the CPU-only / nightly-torch / analytical-body workarounds that the
# cylinder_drag_2d harness needs on this branch.  These are *only* used by
# the probe; they do not touch the source tree.
_PROBE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROBE_DIR))
from _cpu_patches import install_cpu_patches  # noqa: E402

install_cpu_patches()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # deterministic CPU timing
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Per-axis storage census
# ---------------------------------------------------------------------------

# All per-axis groups that Step 5 of todo.md proposes to stack. The tuple
# names are (group_label, owner_attr, [member_attrs]).
PER_AXIS_GROUPS_2D = [
    ("velocity",   "solver",         ["u0", "v0"]),
    ("normal_cc",  "solver",         ["normal_x", "normal_y"]),
    ("sdf_val_face",   "composite_body", ["sdf_val_u", "sdf_val_v"]),
    ("body_face",      "composite_body", ["body_u", "body_v"]),
]

PER_AXIS_GROUPS_3D = [
    ("velocity",   "solver",         ["u0", "v0", "w0"]),
    ("normal_cc",  "solver",         ["normal_x", "normal_y", "normal_z"]),
    ("sdf_val_face",   "composite_body", ["sdf_val_u", "sdf_val_v", "sdf_val_w"]),
    ("body_face",      "composite_body", ["body_u", "body_v", "body_w"]),
]


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.element_size()) * int(t.numel())


@dataclass
class GroupBytes:
    label: str
    member_attrs: list
    tuple_bytes: int
    stacked_bytes: int
    shape: tuple


def census_groups(solver, groups) -> list:
    out = []
    for label, owner_name, members in groups:
        owner = solver if owner_name == "solver" else solver.composite_body
        ts = []
        for name in members:
            t = getattr(owner, name, None)
            if t is None or not isinstance(t, torch.Tensor):
                ts = None
                break
            ts.append(t)
        if ts is None:
            continue
        tuple_b = sum(tensor_bytes(t) for t in ts)
        stacked = torch.stack(ts, dim=0)
        stacked_b = tensor_bytes(stacked)
        out.append(GroupBytes(
            label=label,
            member_attrs=list(members),
            tuple_bytes=tuple_b,
            stacked_bytes=stacked_b,
            shape=tuple(stacked.shape),
        ))
    return out


def stack_unbind_roundtrip_check(solver, groups) -> dict:
    """Return per-group max abs(diff) between original and stack->unbind result."""
    diffs = {}
    for label, owner_name, members in groups:
        owner = solver if owner_name == "solver" else solver.composite_body
        ts = [getattr(owner, n, None) for n in members]
        if any(t is None for t in ts):
            continue
        stk = torch.stack(ts, dim=0)
        rt = list(stk.unbind(dim=0))
        d = 0.0
        for orig, back in zip(ts, rt):
            d = max(d, float((orig - back).abs().max().item()))
        diffs[label] = d
    return diffs


# ---------------------------------------------------------------------------
# 2. Stacked-shadow buffer: maintain (D, *grid) alongside tuple fields.
# ---------------------------------------------------------------------------

class StackedShadow:
    """Mirror per-axis tuple fields into pre-allocated ``(D, *grid)`` buffers.

    Models the *upper bound* of the cost a Step-5 refactor would pay if the
    stacked tensor has to be materialised each step from per-axis sources
    (e.g. because some kernels still emit per-axis writes).  This is the
    worst-case for stacked: real refactor can be cheaper if the stacked
    tensor is the source of truth and ``unbind`` views feed the kernels.
    """

    def __init__(self, solver, groups):
        self.solver = solver
        self.groups = []
        self.buffers = {}
        for entry in groups:
            label, owner_name, members = entry
            owner = solver if owner_name == "solver" else solver.composite_body
            ts = [getattr(owner, m, None) for m in members]
            if any(not isinstance(t, torch.Tensor) for t in ts):
                continue
            self.groups.append(entry)
            t0 = ts[0]
            self.buffers[label] = torch.empty(
                (len(members), *t0.shape), dtype=t0.dtype, device=t0.device
            )

    def project(self):
        """Refresh stacked buffers from current tuple fields (in-place copy)."""
        for label, owner_name, members in self.groups:
            owner = self.solver if owner_name == "solver" else self.solver.composite_body
            buf = self.buffers[label]
            for i, n in enumerate(members):
                buf[i].copy_(getattr(owner, n))

    def bytes(self) -> int:
        return sum(tensor_bytes(b) for b in self.buffers.values())


# ---------------------------------------------------------------------------
# 3. Cylinder-drag 2-D harness (small, CPU)
# ---------------------------------------------------------------------------

_CYL_CFG = "lilytorch/src/configs/flow_past_cylinder.yaml"
_SPH_CFG = "lilytorch/src/configs/flow_past_sphere_3d.yaml"


def _build_cylinder_solver(nx=64, ny=64, dtype=torch.float64):
    from lilytorch.src.solver import FluidSolver
    from lilytorch.util.yaml_operations import yaml2pyobject

    pars = yaml2pyobject(_CYL_CFG)
    pars["solver"].update(dict(
        use_gpu=False,
        Nx=nx, Ny=ny,
        xmin=-0.5, xmax=1.5, ymin=-1.0, ymax=1.0,
        nt=10, dt=0.05, nu=1e-3, rho=1.0,
        convection_method="abdquickest",
        solver_method="python",
        poisson_verbose=False, poisson_tol=1e-4,
        poisson_max_cycles=5, poisson_max_mgcg_cycles=3, poisson_nsmoothing=3,
    ))
    pars["boundary_conditions"]["BC_values_u"] = [0.1, 0.1, 0.0, 0.0]
    pars["body"] = {
        "type": "composite_analytical",
        "plotting": False,
        "sdf": ["lambda x, y: circle(x,y,xt=0.0,yt=0.0,r=0.1)"],
        "update_maps": [{
            "rotation": "lambda t: 0*t",
            "translation": ["lambda t: 0*t", "lambda t: 0*t"],
        }],
    }
    pars["output"]["save_path"] = "/tmp/step5_probe_out_2d/"
    pars["output"]["save_frames"] = False

    solver = FluidSolver(pars, dtype=dtype, compute_forces=True)
    import os
    solver.save_path = pars["output"]["save_path"]
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    return solver


def _build_sphere_solver(nx=24, ny=24, nz=24, dtype=torch.float64):
    from lilytorch.src.solver import FluidSolver
    from lilytorch.util.yaml_operations import yaml2pyobject

    pars = yaml2pyobject(_SPH_CFG)
    pars["solver"].update(dict(
        use_gpu=False,
        Nx=nx, Ny=ny, Nz=nz,
        xmin=-0.5, xmax=1.5, ymin=-1.0, ymax=1.0, zmin=-1.0, zmax=1.0,
        nt=5, dt=0.05, nu=1e-3, rho=1.0,
        convection_method="abdquickest",
        solver_method="python",
        poisson_verbose=False, poisson_tol=1e-4,
        poisson_max_cycles=5, poisson_max_mgcg_cycles=3, poisson_nsmoothing=3,
    ))
    pars["boundary_conditions"]["BC_values_u"] = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
    pars["body"] = {
        "type": "composite_analytical",
        "plotting": False,
        "sdf": ["lambda x, y, z: sphere(x,y,z,xt=0.0,yt=0.0,zt=0.0,r=0.1)"],
        "update_maps": [{
            "rotation": "lambda t: 0*t",
            "translation": ["lambda t: 0*t", "lambda t: 0*t", "lambda t: 0*t"],
        }],
    }
    pars["output"]["save_path"] = "/tmp/step5_probe_out_3d/"
    pars["output"]["save_frames"] = False

    solver = FluidSolver(pars, dtype=dtype, compute_forces=True)
    import os
    solver.save_path = pars["output"]["save_path"]
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    return solver


def _run_harness(solver, nt, with_shadow):
    """Run nt steps; optionally maintain a stacked-shadow each step."""
    groups = PER_AXIS_GROUPS_3D if solver.ndim == 3 else PER_AXIS_GROUPS_2D

    u, v, p = solver.u0, solver.v0, solver.p0
    if solver.ndim == 3:
        w = solver.w0
    t_zero = torch.tensor(0.0, dtype=u.dtype, device=u.device)
    # one warmup step so any first-call lazy init doesn't pollute timing
    if solver.ndim == 3:
        u, v, p, w, _ = solver.step_(u, v, p, 0, t_zero, w_vel=w)
    else:
        u, v, p, _ = solver.step_(u, v, p, 0, t_zero)

    shadow = StackedShadow(solver, groups) if with_shadow else None

    t0 = time.perf_counter()
    for it in range(1, nt + 1):
        tphys = torch.tensor(it * float(solver.dt), dtype=u.dtype, device=u.device)
        if solver.ndim == 3:
            u, v, p, w, stop = solver.step_(u, v, p, it, tphys, w_vel=w)
        else:
            u, v, p, stop = solver.step_(u, v, p, it, tphys)
        if shadow is not None:
            shadow.project()
        if stop:
            break
    dt = time.perf_counter() - t0

    if solver.ndim == 3:
        ke = 0.5 * float((u ** 2 + v ** 2 + w ** 2).sum().item())
    else:
        ke = 0.5 * float((u ** 2 + v ** 2).sum().item())

    vd = solver.viscous_drag_record.detach().cpu()
    pd = solver.pressure_drag_record.detach().cpu()
    drag = float(vd.abs().sum().item() + pd.abs().sum().item())

    return {
        "wall_s": dt,
        "per_step_ms": 1000.0 * dt / nt,
        "ke": ke,
        "drag_signature": drag,
        "shadow_bytes": (shadow.bytes() if shadow is not None else 0),
        "groups": [asdict(g) for g in census_groups(solver, groups)],
        "roundtrip_max_abs_diff": stack_unbind_roundtrip_check(solver, groups),
    }


# ---------------------------------------------------------------------------
# 4. Microbench: hot per-axis ops, tuple vs stacked
# ---------------------------------------------------------------------------

def _timeit(fn, repeats=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / repeats  # ms/call


def microbench(D: int, grid, dtype=torch.float64) -> dict:
    """Time the four canonical per-axis ops in tuple and stacked layouts.

    Operations modelled on actual usage in the codebase:

      A. **bdim-mu0-mask**: ``u_out = u * mu0_u`` for each axis (solver.py:825).
      B. **bdim-blend**:    ``u_out = mu0 * u + mu1 * body_u``  (BDIM apply).
      C. **sdf-where-merge**: ``dst = where(mask, src, dst)`` per axis
         (CompositeBodyAnalytical.update at body.py:1268-1282).
      D. **kinetic-energy reduction**: ``sum(u**2 + v**2 + w**2)``.

    For each op, we measure the tuple form (D scalar kernel launches) and
    the stacked form (one broadcast kernel launch).  Both versions write
    into pre-allocated output buffers to keep allocation costs out of the
    benchmark proper (matches what the codebase would do after a refactor).
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    tuple_axes = lambda: [torch.rand(grid, generator=g, dtype=dtype) for _ in range(D)]
    u_t = tuple_axes(); mu0_t = tuple_axes(); mu1_t = tuple_axes()
    body_t = tuple_axes(); src_t = tuple_axes(); mask_t = [torch.rand(grid, generator=g, dtype=dtype) < 0.5 for _ in range(D)]
    dst_t = tuple_axes()
    out_t = tuple_axes()

    u_s = torch.stack(u_t); mu0_s = torch.stack(mu0_t); mu1_s = torch.stack(mu1_t)
    body_s = torch.stack(body_t); src_s = torch.stack(src_t); mask_s = torch.stack(mask_t)
    dst_s = torch.stack(dst_t); out_s = torch.empty_like(u_s)

    # --- A. mu0 mask ----------------------------------------------------
    def A_tuple():
        for i in range(D):
            torch.mul(u_t[i], mu0_t[i], out=out_t[i])
    def A_stack():
        torch.mul(u_s, mu0_s, out=out_s)

    # --- B. BDIM blend: out = mu0 * u + mu1 * body ----------------------
    def B_tuple():
        for i in range(D):
            torch.mul(u_t[i], mu0_t[i], out=out_t[i])
            out_t[i].addcmul_(mu1_t[i], body_t[i])
    def B_stack():
        torch.mul(u_s, mu0_s, out=out_s)
        out_s.addcmul_(mu1_s, body_s)

    # --- C. where-merge -------------------------------------------------
    def C_tuple():
        for i in range(D):
            torch.where(mask_t[i], src_t[i], dst_t[i], out=dst_t[i])
    def C_stack():
        torch.where(mask_s, src_s, dst_s, out=dst_s)

    # --- D. kinetic energy ---------------------------------------------
    def D_tuple():
        s = u_t[0].pow(2).sum()
        for i in range(1, D):
            s = s + u_t[i].pow(2).sum()
        return s
    def D_stack():
        return u_s.pow(2).sum()

    results = {}
    for name, fn_t, fn_s in [
        ("A_mu0_mask",   A_tuple, A_stack),
        ("B_bdim_blend", B_tuple, B_stack),
        ("C_where_merge", C_tuple, C_stack),
        ("D_ke_reduce",  D_tuple, D_stack),
    ]:
        ms_tuple = _timeit(fn_t)
        ms_stack = _timeit(fn_s)
        # numerical equivalence: re-run fresh into independent buffers and
        # compare the unbound stacked output against the tuple output
        results[name] = {
            "tuple_ms": ms_tuple,
            "stack_ms": ms_stack,
            "speedup_x": (ms_tuple / ms_stack) if ms_stack > 0 else float("inf"),
        }

    # numerical equivalence of ops A and B (the only ones whose outputs we
    # have on both sides simultaneously)
    A_tuple(); A_stack()
    diff_A = max(float((out_t[i] - out_s[i]).abs().max().item()) for i in range(D))
    B_tuple(); B_stack()
    diff_B = max(float((out_t[i] - out_s[i]).abs().max().item()) for i in range(D))
    results["_max_abs_diff"] = {"A": diff_A, "B": diff_B}

    # also resident byte count of the inputs in both layouts
    results["_bytes"] = {
        "tuple_bytes": sum(tensor_bytes(t) for t in u_t + mu0_t + mu1_t + body_t + src_t + dst_t + out_t),
        "stack_bytes": sum(tensor_bytes(t) for t in [u_s, mu0_s, mu1_s, body_s, src_s, dst_s, out_s]) +
                       sum(tensor_bytes(t) for t in mask_t),
    }
    return results


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------

def main():
    out_dir = _PROBE_DIR
    results = {"meta": {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "dtype": "float64",
    }}

    # --- 2-D cylinder harness -------------------------------------------
    print("== 2D cylinder drag ==", flush=True)
    sol2 = _build_cylinder_solver(nx=64, ny=64)
    base2 = _run_harness(sol2, nt=20, with_shadow=False)
    sol2b = _build_cylinder_solver(nx=64, ny=64)
    shadow2 = _run_harness(sol2b, nt=20, with_shadow=True)
    results["cylinder_2d"] = {"baseline": base2, "stacked_shadow": shadow2}
    del sol2, sol2b; gc.collect()

    # --- 3-D sphere harness ---------------------------------------------
    print("== 3D sphere drag ==", flush=True)
    sol3 = _build_sphere_solver(nx=24, ny=24, nz=24)
    base3 = _run_harness(sol3, nt=10, with_shadow=False)
    sol3b = _build_sphere_solver(nx=24, ny=24, nz=24)
    shadow3 = _run_harness(sol3b, nt=10, with_shadow=True)
    results["sphere_3d"] = {"baseline": base3, "stacked_shadow": shadow3}
    del sol3, sol3b; gc.collect()

    # --- microbench -----------------------------------------------------
    print("== microbench ==", flush=True)
    micro = {}
    for label, D, grid in [
        ("2d_128",  2, (128, 128)),
        ("2d_256",  2, (256, 256)),
        ("3d_32",   3, (32, 32, 32)),
        ("3d_64",   3, (64, 64, 64)),
    ]:
        print(f"  {label} ...", flush=True)
        micro[label] = microbench(D, grid)
    results["microbench"] = micro

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
