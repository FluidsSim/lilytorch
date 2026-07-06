# Injected into the FARMS subprocess before main() via _extra_run_patch().
# Instruments the coupled salamander 2-D step:
#   * per-phase wall time, GPU-synchronised (steps 250-329)
#   * per-phase host submit time, no syncs   (steps 330-369)
#   * pure throughput window, untouched      (steps 150-249)
#   * eager wp.launch / graph-replay / host-sync counts per step
#   * torch.profiler CUDA kernel breakdown   (steps 370-376)
import functools
import json
import os
import time

import torch
import warp as wp

# NOTE: this file runs via exec() inside the run.sh subprocess — no __file__.
# The launcher (prof_coupled_step.py) sets PROF_REPORT; fall back to the
# run's output folder (cwd).
REPORT = os.environ.get("PROF_REPORT",
                        os.path.abspath("prof_coupled_step_report.json"))

S = {
    "mode": "off", "step": -1, "prev_t": None,
    "sync": {}, "sync_n": {}, "submit": {}, "submit_n": {},
    "total_step": [], "handler_wall": [],
    "launch_per_step": [], "replay_per_step": [], "itemsync_per_step": [],
    "wall_ms": None, "prof": None, "prof_rows": None,
}
C = {"launch": 0, "replay": 0, "item": 0, "cpu": 0, "numpy": 0}

# ── counters ────────────────────────────────────────────────────────────────
_orig_launch = wp.launch
def _launch_w(*a, **k):
    C["launch"] += 1
    return _orig_launch(*a, **k)
wp.launch = _launch_w

_orig_capture_launch = wp.capture_launch
def _capture_launch_w(*a, **k):
    C["replay"] += 1
    return _orig_capture_launch(*a, **k)
wp.capture_launch = _capture_launch_w

_orig_item = torch.Tensor.item
def _item_w(self):
    if self.is_cuda:
        C["item"] += 1
    return _orig_item(self)
torch.Tensor.item = _item_w

_orig_cpu = torch.Tensor.cpu
def _cpu_w(self, *a, **k):
    if self.is_cuda:
        C["cpu"] += 1
    return _orig_cpu(self, *a, **k)
torch.Tensor.cpu = _cpu_w

# ── phase wrappers ──────────────────────────────────────────────────────────
def _wrap(cls, attr, name):
    orig = getattr(cls, attr)

    @functools.wraps(orig)
    def w(*a, **k):
        m = S["mode"]
        if m == "sync":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            S["sync"][name] = S["sync"].get(name, 0.0) + time.perf_counter() - t0
            S["sync_n"][name] = S["sync_n"].get(name, 0) + 1
            return r
        if m == "submit":
            t0 = time.perf_counter()
            r = orig(*a, **k)
            S["submit"][name] = S["submit"].get(name, 0.0) + time.perf_counter() - t0
            S["submit_n"][name] = S["submit_n"].get(name, 0) + 1
            return r
        return orig(*a, **k)

    setattr(cls, attr, w)

from lilytorch.integration.BDIMhandler import BDIMhandler
from lilytorch.src.solver import FluidSolver
from lilytorch.src.advection import AdvDiffSolver
import lilytorch
print("[prof] lilytorch resolved at:", lilytorch.__file__, flush=True)

_wrap(BDIMhandler, "_launch_body_update", "stream_sdf")
_wrap(BDIMhandler, "_apply_forces", "apply_forces(readback+push)")
_wrap(FluidSolver, "step_", "fluid_step_total")
_wrap(FluidSolver, "project", "project(poisson+correct)")
_wrap(FluidSolver, "forces_method2", "forces_readout")
_wrap(AdvDiffSolver, "_solve_convective", "advdiff_solve")
_wrap(AdvDiffSolver, "_solve_semi_lagrangian", "advdiff_solve_sl")

from lilytorch.src.interpolation import RegularGridInterpolator
from lilytorch.src import diffusion as _diffmod
from lilytorch.src import advection as _advmod
import lilytorch.src.solver as _solvermod

_wrap(RegularGridInterpolator, "__call__", "sl_interp_call")
_wrap(FluidSolver, "_mw_body_div_correction", "mw_body_div_corr")
_wrap(FluidSolver, "_compute_nu_t", "compute_nu_t")
_wrap(FluidSolver, "advance_and_compute_loads", "advance_and_loads_total")

def _wrap_fn(mod, attr, name):
    orig = getattr(mod, attr)
    @functools.wraps(orig)
    def w(*a, **k):
        m = S["mode"]
        if m == "sync":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            S["sync"][name] = S["sync"].get(name, 0.0) + time.perf_counter() - t0
            S["sync_n"][name] = S["sync_n"].get(name, 0) + 1
            return r
        if m == "submit":
            t0 = time.perf_counter()
            r = orig(*a, **k)
            S["submit"][name] = S["submit"].get(name, 0.0) + time.perf_counter() - t0
            S["submit_n"][name] = S["submit_n"].get(name, 0) + 1
            return r
        return orig(*a, **k)
    setattr(mod, attr, w)

_wrap_fn(_solvermod, "bdim_forcing_2d", "bdim_forcing")
_wrap_fn(_advmod.diffusion, "diffuse", "sl_diffuse")
_wrap(AdvDiffSolver, "set_BCs", "advdiff_set_bcs")

def _report():
    out = {
        "throughput_ms_per_step_untouched": S["wall_ms"],
        "sync_attributed_ms": {
            k: 1000.0 * v / max(S["sync_n"][k], 1) for k, v in S["sync"].items()},
        "submit_only_ms": {
            k: 1000.0 * v / max(S["submit_n"][k], 1) for k, v in S["submit"].items()},
        "total_step_ms_wall_window": (
            1000.0 * sum(S["total_step"]) / max(len(S["total_step"]), 1)),
        "handler_step_ms_sync_window": (
            1000.0 * sum(S["handler_wall"]) / max(len(S["handler_wall"]), 1)),
        "eager_wp_launches_per_step": S["launch_per_step"],
        "graph_replays_per_step": S["replay_per_step"],
        "cuda_item_syncs_per_step": S["itemsync_per_step"],
        "profiler_top_cuda": S["prof_rows"],
    }
    with open(REPORT, "w") as f:
        json.dump(out, f, indent=1)
    print("[prof] report written to", REPORT, flush=True)
    print(json.dumps(out, indent=1), flush=True)

_orig_step = BDIMhandler.step
def _step_w(self, task, physics):
    S["step"] += 1
    n = S["step"]
    now = time.perf_counter()
    if S["prev_t"] is not None and 150 <= n <= 249:
        S["total_step"].append(now - S["prev_t"])
    S["prev_t"] = now

    if n == 150:
        S["mode"] = "off"
        S["wall_t0"] = now
    elif n == 250:
        S["wall_ms"] = 1000.0 * (now - S["wall_t0"]) / 100.0
        S["mode"] = "sync"
    elif n == 330:
        S["mode"] = "submit"
    elif n == 370:
        S["mode"] = "off"
        S["prof"] = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
        S["prof"].__enter__()
    elif n == 377:
        S["prof"].__exit__(None, None, None)
        rows = sorted(S["prof"].key_averages(),
                      key=lambda e: e.self_device_time_total, reverse=True)[:30]
        S["prof_rows"] = [
            {"name": e.key[:80], "cuda_us_total": round(e.self_device_time_total, 1),
             "calls": e.count} for e in rows if e.self_device_time_total > 0]
        _report()
        os._exit(0)

    l0, r0, i0 = C["launch"], C["replay"], C["item"]
    if S["mode"] == "sync":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ret = _orig_step(self, task, physics)
        torch.cuda.synchronize()
        S["handler_wall"].append(time.perf_counter() - t0)
    else:
        ret = _orig_step(self, task, physics)
    if S["mode"] == "submit":
        S["launch_per_step"].append(C["launch"] - l0)
        S["replay_per_step"].append(C["replay"] - r0)
        S["itemsync_per_step"].append(C["item"] - i0)
    return ret
BDIMhandler.step = _step_w
print("[prof] instrumentation installed", flush=True)
