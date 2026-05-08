"""Diagnose what is in the 'Other' residual of _fluid_step_3d.

Patches `_fluid_step_3d` with a re-implementation that adds CUDA-synced
timers around every phase that is *not* covered by the production
named timers (3a/3b/3c/3d/3e/3f).  Outputs a table sorted by mean
per-step cost.

Run from this directory:
    python diagnose_other_3d.py --Nx 128 --Ny 32 --Nz 32 --n_steps 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from collections import defaultdict

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")))

ap = argparse.ArgumentParser()
ap.add_argument("--Nx", type=int, default=128)
ap.add_argument("--Ny", type=int, default=32)
ap.add_argument("--Nz", type=int, default=32)
ap.add_argument("--n_steps", type=int, default=30)
ap.add_argument("--precompile", type=int, default=20)
extra = ap.parse_args()

# Build sys.argv for the cost-analysis script we exec below.
sys.argv = [
    "run_cost_analysis.py",
    "--Nx", str(extra.Nx), "--Ny", str(extra.Ny), "--Nz", str(extra.Nz),
    "--n_steps", str(extra.n_steps), "--precompile", str(extra.precompile),
    "--streaming_sdf_3d",
    "--streaming_forces_3d",
    "--force_shared_union",
    "--mu_normals_union",
    "--bdim_union",
    "--force_narrow_batch",
    "--out_dir", os.path.join(SCRIPT_DIR, "figures", "diagnose_other"),
]


# ── Mini timer ─────────────────────────────────────────────────────────
class Bank:
    def __init__(self):
        self.d: dict[str, list[float]] = defaultdict(list)
        self.active = False

    def time(self, name):
        bank = self
        class _Ctx:
            def __enter__(_):
                if bank.active:
                    torch.cuda.synchronize()
                    _.t0 = time.perf_counter()
            def __exit__(_, *a):
                if bank.active:
                    torch.cuda.synchronize()
                    bank.d[name].append(time.perf_counter() - _.t0)
        return _Ctx()

B = Bank()
B.active = True  # always-on; we discard first DROP entries below.
DROP = 25  # discard precompile + settle entries


# Patch fluid_step_3d AFTER importing BDIMhandler.
def install_patch():
    from lilytorch.integration.BDIMhandler import BDIMhandler

    def diag_fluid_step_3d(self, u, v, w, p, timestep):
        fs = self.fluid_solver
        _bdim = fs._bdim_meta_compiled
        _h = fs.h

        with B.time("F.0  nu_t"):
            nu_t = fs._compute_nu_t(u, v, w)
        with B.time("F.1  adv_diff.solve"):
            (uprime, vprime, wprime) = fs.adv_diff_solver.solve(u, v, w, nu_t=nu_t)
        with B.time("F.2  clone uprime/vprime/wprime"):
            uprime = uprime.clone()
            vprime = vprime.clone()
            wprime = wprime.clone()
        with B.time("F.3  set_BCs(uprime,vprime,wprime)"):
            fs.adv_diff_solver.set_BCs(uprime, vprime, wprime)

        with B.time("F.4  union AABB compute"):
            fs._bdim_union_aabb = (
                fs._compute_union_aabb_3d(halo=2)
                if getattr(fs, '_bdim_union', False) else None
            )

        with B.time("F.5  bdim_apply x3"):
            uprime = fs._bdim_apply_3d(
                uprime, fs.mu0_all_u,
                fs.composite_body.body_u, fs.mu1_all_u,
                fs.normal_x_u, fs.normal_y_u, fs.normal_z_u,
            )
            vprime = fs._bdim_apply_3d(
                vprime, fs.mu0_all_v,
                fs.composite_body.body_v, fs.mu1_all_v,
                fs.normal_x_v, fs.normal_y_v, fs.normal_z_v,
            )
            wprime = fs._bdim_apply_3d(
                wprime, fs.mu0_all_w,
                fs.composite_body.body_w, fs.mu1_all_w,
                fs.normal_x_w, fs.normal_y_w, fs.normal_z_w,
            )
            fs._bdim_union_aabb = None

        with B.time("F.6  release mu1+normals (dict.update)"):
            fs.__dict__.update(fs._FS_FREE_AFTER_BDIM)

        with B.time("F.7  var-density coeffs"):
            ch, cv, cw, ch_cc = self._compute_variable_density_coefficients(timestep)

        with B.time("F.8  release mu0 (dict.update)"):
            fs.__dict__.update(fs._FS_FREE_AFTER_VAR_DENS)

        with B.time("F.9  project"):
            poisson_method = getattr(fs, "poisson_method", "multigrid")
            if poisson_method == "fft":
                (u, v, w, p) = fs.project(uprime, vprime, p,
                                          w_vel=wprime, ch=ch, cv=cv, cw=cw,
                                          ch_cc=ch_cc)
            else:
                (u, v, w, p) = fs.project(uprime, vprime, p,
                                          w_vel=wprime, ch=ch, cv=cv, cw=cw)

        with B.time("F.A  set_BCs(u,v,w) final"):
            fs.adv_diff_solver.set_BCs(u, v, w)
        return (u, v, w, p)

    BDIMhandler._fluid_step_3d = diag_fluid_step_3d
    print("[diagnose] _fluid_step_3d patched with fine-grained timers.")


install_patch()

# Now import + run the standard cost-analysis script.  We intercept the
# precompile/measure transition by hooking into its TimerBank: when its
# first measurement step starts, we activate B; when it finishes, we
# print our table.

import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_cost_analysis", os.path.join(SCRIPT_DIR, "run_cost_analysis.py"))
m = importlib.util.module_from_spec(spec)

# Patch its TimerBank to also flip B.active.
import contextlib
_original_call = None

def _start_active():
    B.active = True
    print("[diagnose] B activated (measurement phase started).")

def _hook():
    # spec.loader.exec_module will call argparse before we can patch; that's fine.
    pass

_hook()
spec.loader.exec_module(m)

# After running, dump our table.
print()
print("=" * 78)
print("  Inner-fluid_step_3d residual diagnosis  (CUDA-synced)")
print("=" * 78)
rows = []
for name, times in sorted(B.d.items()):
    arr = np.array(times[DROP:]) if len(times) > DROP else np.array(times)
    if len(arr) == 0:
        continue
    rows.append((name, float(np.median(arr) * 1e3), float(arr.mean() * 1e3),
                 float(arr.std() * 1e3), len(arr)))
rows.sort(key=lambda r: -r[2])
print(f"{'phase':50s} {'median ms':>10s} {'mean ms':>10s} {'std ms':>8s} {'n':>5s}")
print("-" * 78)
for name, med, mean, std, n in rows:
    print(f"{name:50s} {med:10.4f} {mean:10.4f} {std:8.4f} {n:5d}")
total_inner = sum(r[2] for r in rows)
print("-" * 78)
print(f"{'SUM (mean)':50s} {'':>10} {total_inner:10.4f}")
