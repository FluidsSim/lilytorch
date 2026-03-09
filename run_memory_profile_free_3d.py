#!/usr/bin/env python3
"""
Memory-profiling wrapper for gen_configs_one_free_3d.

Generates the FARMS config files, then runs the simulation IN-PROCESS
(not via subprocess) with monkey-patched hooks that record GPU memory
at each phase of BDIMhandler.step().
"""

import sys, os, gc, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.cuda

# ── helpers ──────────────────────────────────────────────────────────────
def _mb(b):
    return b / (1024 ** 2)

def snap(label=""):
    torch.cuda.synchronize()
    return {
        "label":    label,
        "alloc_mb": _mb(torch.cuda.memory_allocated()),
        "peak_mb":  _mb(torch.cuda.max_memory_allocated()),
        "rsrvd_mb": _mb(torch.cuda.memory_reserved()),
    }

def print_snap(s):
    print(f"  [{s['label']:>45s}]  alloc={s['alloc_mb']:8.1f} MB   "
          f"peak={s['peak_mb']:8.1f} MB   reserved={s['rsrvd_mb']:8.1f} MB",
          flush=True)

records = []
def record(label):
    s = snap(label)
    records.append(s)
    print_snap(s)
    return s


# ═══════════════════════════════════════════════════════════════════════
# 1. Generate the config files (quick, no GPU)
# ═══════════════════════════════════════════════════════════════════════
import lilytorch.farms_examples._1guillasim.gen_configs_one_free_3d as cfg

cfg.n_iterations = 201
cfg.save_every   = 200
cfg.save_frames  = False
cfg.headless     = True     # no GUI

from lilytorch.util.paths import gen_new_folder
output_folder = gen_new_folder(cfg.stack_folder)
os.makedirs(output_folder, exist_ok=True)

cfg.gen_animat_config(output_folder)
cfg.gen_arena_config(output_folder)
cfg.gen_simulation_config(output_folder)
cfg.gen_experiment_config(output_folder)
cfg.gen_sh_config(output_folder)
os.chdir(output_folder)

print("=" * 80)
print(f"Grid: Nx={cfg.Nx} Ny={cfg.Ny} Nz={cfg.Nz}  "
      f"iters={cfg.n_iterations}  save_every={cfg.save_every}")
print(f"Config folder: {output_folder}")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# 2. Install monkey-patch hooks BEFORE creating the simulation
# ═══════════════════════════════════════════════════════════════════════

# --- BDIMhandler.__init__ ---
import lilytorch.integration.BDIMhandler as bmod
_orig_init_bdim = bmod.BDIMhandler.__init__

def _profiled_init_bdim(self, *args, **kwargs):
    record("BDIMhandler.__init__: enter")
    _orig_init_bdim(self, *args, **kwargs)
    record("BDIMhandler.__init__: exit")

bmod.BDIMhandler.__init__ = _profiled_init_bdim


# --- FluidSolver.__init__ ---
from lilytorch.src import solver as smod
_orig_init_fs = smod.FluidSolver.__init__

def _profiled_init_fs(self, *args, **kwargs):
    record("FluidSolver.__init__: enter")
    _orig_init_fs(self, *args, **kwargs)
    record("FluidSolver.__init__: exit")

smod.FluidSolver.__init__ = _profiled_init_fs


# --- BDIMhandler.step ---
_orig_step = bmod.BDIMhandler.step
_step_call = [0]

def _profiled_step(self, task, physics):
    _step_call[0] += 1
    it = _step_call[0]
    do_trace = (it <= 3) or (it % 50 == 0)

    iteration = self.iteration
    timestep  = self.pars["solver"]["dt"]
    if iteration >= self.pars["solver"]["nt"]:
        return

    t  = iteration * timestep
    fs = self.fluid_solver

    if self.terminate:
        self.iteration += 1
        return

    if do_trace:
        torch.cuda.reset_peak_memory_stats()
        record(f"step {it}: before")

    # 1. update
    self.update(t, iteration, dt=timestep)
    if do_trace:
        record(f"step {it}: after update (SDF+vel)")

    # 2. recompute mu/normals
    if self.ndim == 3:
        self._recompute_mu_normals_3d()
    else:
        self._recompute_mu_normals_2d()
    if do_trace:
        record(f"step {it}: after mu/normals")

    # 3. fluid step (Heun)
    if self.ndim == 3:
        (u, v, w, p) = self.fluid_step(fs.u0, fs.v0, fs.w0, fs.p0, timestep)
        if self.zero_pressure_inside:
            p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
        (fs.u0, fs.v0, fs.w0, fs.p0) = (u, v, w, p)
    else:
        (u, v, p) = self.fluid_step(fs.u0, fs.v0, fs.p0, timestep)
        if self.zero_pressure_inside:
            p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
        (fs.u0, fs.v0, fs.p0) = (u, v, p)
    if do_trace:
        record(f"step {it}: after fluid_step (Heun done)")

    # 4. forces
    if self.ndim == 3:
        fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
    elif self.force_method == "method1":
        fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
    else:
        fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)
    if do_trace:
        record(f"step {it}: after forces")

    # 5. plotting/saving
    if self.ndim == 3:
        self.terminate = fs.plotting_and_saving(
            fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0, check_termination=False
        )
    else:
        self.terminate = fs.plotting_and_saving(
            fs.u0, fs.v0, fs.p0, iteration, check_termination=False
        )

    # 6. apply forces
    self.apply_forces(task, physics)

    # 7. release
    fs._release_bdim_fields()
    if do_trace:
        record(f"step {it}: after release")

    self.iteration += 1

bmod.BDIMhandler.step = _profiled_step


# --- solver_iteration_heun (Heun peak measurement) ---
_orig_heun = smod.FluidSolver.solver_iteration_heun

def _profiled_heun(self, *args, **kwargs):
    it = _step_call[0]
    do_trace = (it <= 3) or (it % 50 == 0)
    if do_trace:
        record(f"step {it}:   heun enter")
    result = _orig_heun(self, *args, **kwargs)
    if do_trace:
        record(f"step {it}:   heun exit")
    return result

smod.FluidSolver.solver_iteration_heun = _profiled_heun


# --- project() inside Heun ---
_orig_project = smod.FluidSolver.project

_project_call = [0]
def _profiled_project(self, *args, **kwargs):
    it = _step_call[0]
    do_trace = (it <= 3) or (it % 50 == 0)
    _project_call[0] += 1
    proj_num = _project_call[0]
    result = _orig_project(self, *args, **kwargs)
    if do_trace:
        record(f"step {it}:     after project() #{proj_num}")
    return result

smod.FluidSolver.project = _profiled_project


# ═══════════════════════════════════════════════════════════════════════
# 3. Run the simulation IN-PROCESS
# ═══════════════════════════════════════════════════════════════════════
from farms_core.experiment.options import ExperimentOptions
from farms_core.extensions.extensions import import_item
from farms_sim.simulation import run_simulation

torch.cuda.reset_peak_memory_stats()
record("before simulation setup")

experiment_options = ExperimentOptions.load('experiment_config.yaml')
experiment_data_loader = import_item(experiment_options.loaders.experiment_data)
experiment_data = experiment_data_loader.from_options(experiment_options)

record("after loading experiment options")

sim = run_simulation(
    experiment_data=experiment_data,
    experiment_options=experiment_options,
)

record("after simulation complete")


# ═══════════════════════════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MEMORY PROFILE SUMMARY")
print("=" * 80)

# Group by phase
phases = {}
for r in records:
    lbl = r["label"]
    if ":" in lbl:
        phase = lbl.split(":", 1)[1].strip()
    else:
        phase = lbl
    if phase not in phases:
        phases[phase] = {"alloc": [], "peak": []}
    phases[phase]["alloc"].append(r["alloc_mb"])
    phases[phase]["peak"].append(r["peak_mb"])

print(f"\n{'Phase':<45s}  {'Avg Alloc MB':>12s}  {'Max Alloc MB':>12s}  {'Final Peak MB':>14s}")
print("-" * 90)
for phase, vals in phases.items():
    avg_a = sum(vals["alloc"]) / len(vals["alloc"])
    max_a = max(vals["alloc"])
    peak  = max(vals["peak"])
    print(f"{phase:<45s}  {avg_a:12.1f}  {max_a:12.1f}  {peak:14.1f}")

final_peak = _mb(torch.cuda.max_memory_allocated())
final_rsrv = _mb(torch.cuda.memory_reserved())
print(f"\nFinal peak allocated: {final_peak:.1f} MB")
print(f"Final reserved:       {final_rsrv:.1f} MB")

# ── Detailed GPU tensor census ────────────────────────────────────────
print("\n" + "=" * 80)
print("GPU TENSOR CENSUS (current)")
print("=" * 80)

tensor_info = {}
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            shape = tuple(obj.shape)
            dtype = str(obj.dtype)
            key = (shape, dtype)
            nbytes = obj.nelement() * obj.element_size()
            if key not in tensor_info:
                tensor_info[key] = {"count": 0, "bytes": 0}
            tensor_info[key]["count"] += 1
            tensor_info[key]["bytes"] += nbytes
    except Exception:
        pass

sorted_tensors = sorted(tensor_info.items(), key=lambda x: -x[1]["bytes"])
total_census = 0
print(f"\n{'Shape':<45s}  {'Dtype':<15s}  {'Count':>6s}  {'Total MB':>10s}")
print("-" * 80)
for (shape, dtype), info in sorted_tensors[:40]:
    mb = _mb(info["bytes"])
    total_census += info["bytes"]
    print(f"{str(shape):<45s}  {dtype:<15s}  {info['count']:6d}  {mb:10.1f}")
print(f"\nTotal census: {_mb(total_census):.1f} MB")


# ── Phase-to-phase deltas (step 2 example) ────────────────────────────
print("\n" + "=" * 80)
print("STEP-BY-STEP DELTAS (from first traced step)")
print("=" * 80)
step2_recs = [r for r in records if r["label"].startswith("step 2:")]
if step2_recs:
    prev = step2_recs[0]
    for r in step2_recs[1:]:
        delta = r["alloc_mb"] - prev["alloc_mb"]
        print(f"  {prev['label']:>45s} -> {r['label'].split(':',1)[1].strip():<30s}  "
              f"delta={delta:+8.1f} MB   alloc={r['alloc_mb']:8.1f} MB")
        prev = r
