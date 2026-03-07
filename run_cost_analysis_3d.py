#!/usr/bin/env python3
"""
Detailed computational-cost analysis for 3-D BDIM swimming (1guilla pinned).

Instruments every major + sub-kernel inside BDIMhandler3D with
CUDA-synchronised timers.  Reports per-step and cumulative statistics,
saves CSV and figures.

KEY DESIGN CHOICES:
  - Runs the FARMS simulation **in-process** (not subprocess) so that
    Python-level monkey-patches apply to every hot path.
  - Runs in **headless** mode with all non-essential extensions removed
    (no FlowViewer, CameraRecording, ExperimentLogger, MjcfSaver,
    DataLogger) to measure the pure solver cost.
  - Patches at the **instance** level after FARMS creates objects, so
    dispatch aliases like ``adv.solve`` are correctly captured.

Usage
-----
    source /path/to/venv/bin/activate
    python run_cost_analysis_3d.py                                     # default 128x32x32, 20 steps
    python run_cost_analysis_3d.py --Nx 256 --Ny 64 --Nz 64 --n_steps 50
"""

import argparse
import os
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

# ── CLI args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="3-D BDIM detailed cost profiler")
parser.add_argument("--Nx",        type=int, default=128)
parser.add_argument("--Ny",        type=int, default=32)
parser.add_argument("--Nz",        type=int, default=32)
parser.add_argument("--n_steps",   type=int, default=20,
                    help="Total steps (first --warmup are discarded)")
parser.add_argument("--warmup",    type=int, default=3)
parser.add_argument("--save_every",type=int, default=9999)
parser.add_argument("--device",    type=str, default="cuda",
                    choices=["cuda", "cpu"])
parser.add_argument("--out_dir",   type=str, default="figures/cost_analysis")
args = parser.parse_args()

USE_CUDA = args.device == "cuda" and torch.cuda.is_available()


# ═══════════════════════════════════════════════════════════════════════
# Timer infrastructure
# ═══════════════════════════════════════════════════════════════════════

class TimerBank:
    """CUDA-synchronised timer collection with step-based warm-up."""

    def __init__(self, use_cuda: bool, warmup: int = 0):
        self.use_cuda = use_cuda
        self._data: dict[str, list[float]] = defaultdict(list)
        self._step_count = 0
        self._warmup = warmup

    @contextmanager
    def __call__(self, label: str):
        if self._step_count < self._warmup:
            yield
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.use_cuda:
            torch.cuda.synchronize()
        self._data[label].append(time.perf_counter() - t0)

    def new_step(self):
        self._step_count += 1
        if self._step_count == self._warmup + 1:
            print(f"\n  [profiler] Warm-up done ({self._warmup} steps). "
                  f"Collecting timings from step {self._step_count}...\n",
                  flush=True)

    def summary(self):
        outer_total = sum(self._data.get("TOTAL step", [1.0]))
        rows = {}
        for label, times in sorted(self._data.items()):
            arr = np.array(times)
            rows[label] = {
                "mean":  arr.mean(),
                "std":   arr.std(),
                "min":   arr.min(),
                "max":   arr.max(),
                "total": arr.sum(),
                "count": len(arr),
                "pct":   100.0 * arr.sum() / outer_total if outer_total > 0 else 0,
            }
        return rows


T = TimerBank(USE_CUDA, warmup=args.warmup)


# ═══════════════════════════════════════════════════════════════════════
# Import project modules
# ═══════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(__file__))

from lilytorch.integration.extensions import FluidExtension
from lilytorch.farms_examples._1guillasim.BDIMhandler3D import BDIMhandler3D
from lilytorch.src.poisson_mult import PoissonSolver


# ═══════════════════════════════════════════════════════════════════════
# Deep instrumentation (instance-level)
# ═══════════════════════════════════════════════════════════════════════

def instrument_handler(handler):
    """Wrap every hot-path method on live instances with fine-grained timers."""
    if getattr(handler, "_profiled", False):
        return
    handler._profiled = True

    fs = handler.fluid_solver
    comp = fs.composite_body
    adv  = fs.adv_diff_solver

    # ── 1. Replace handler.step with a fully-timed version ───────
    #    This replaces the entire step body so we can time *each*
    #    sub-section individually (mu, normals, BDIM meta-eq, etc.)
    _orig_update      = type(handler).update
    _orig_fluid_step  = type(handler).fluid_step
    _orig_apply       = type(handler).apply_forces
    _orig_forces      = type(fs).forces_method2_3d
    _orig_plot        = type(fs).plotting_and_saving

    def detailed_step(self, task, physics):
        T.new_step()
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            self.iteration += 1
            return

        t  = iteration * timestep
        ffs = self.fluid_solver

        if not self.terminate:

            # ── 1. SDF update (FARMS kinematics + body SDF eval) ─────
            with T("1  SDF update (kinematics+SDF)"):
                self.update(t, iteration, dt=timestep)

            # ── 2-5. mu + normals on 4 staggered grids ──────────────
            sdf_fields = [
                ("cc", comp.sdf_val),
                ("u",  comp.sdf_val_u),
                ("v",  comp.sdf_val_v),
                ("w",  comp.sdf_val_w),
            ]
            mu_attrs = [
                ("mu0_all",   "mu1_all",   "m_m0_all",   "normal_x",   "normal_y",   "normal_z"),
                ("mu0_all_u", "mu1_all_u", "m_m0_all_u", "normal_x_u", "normal_y_u", "normal_z_u"),
                ("mu0_all_v", "mu1_all_v", "m_m0_all_v", "normal_x_v", "normal_y_v", "normal_z_v"),
                ("mu0_all_w", "mu1_all_w", "m_m0_all_w", "normal_x_w", "normal_y_w", "normal_z_w"),
            ]
            for (grid_name, sdf_field), (a_mu0, a_mu1, a_mm0, a_nx, a_ny, a_nz) in zip(sdf_fields, mu_attrs):
                with T("2  mu_funcs (Heaviside)"):
                    (mu0, mu1) = comp.mu_funcs(sdf_field)
                setattr(ffs, a_mu0, mu0)
                setattr(ffs, a_mu1, mu1)
                setattr(ffs, a_mm0, 1 - mu0)

                with T("3  compute_sdf_properties (normals)"):
                    result = comp.compute_sdf_properties(sdf_field)
                setattr(ffs, a_nx, result[1])
                setattr(ffs, a_ny, result[2])
                setattr(ffs, a_nz, result[3])

            # ── 6. Fluid step (instrumented separately) ──────────────
            with T("4  fluid_step (total PDE)"):
                (u, v, w, p) = self.fluid_step(
                    ffs.u0, ffs.v0, ffs.w0, ffs.p0, timestep)

            (ffs.u0, ffs.v0, ffs.w0, ffs.p0) = (u, v, w, p)

            # ── 7. Forces ────────────────────────────────────────────
            with T("5  forces_method2_3d"):
                _orig_forces(ffs, ffs.u0, ffs.v0, ffs.w0, ffs.p0, iteration)

            # ── 8. Plotting / saving ─────────────────────────────────
            with T("6  plotting_and_saving"):
                self.terminate = _orig_plot(
                    ffs, ffs.u0, ffs.v0, ffs.p0, iteration, w_vel=ffs.w0)

            # ── 9. Apply forces ──────────────────────────────────────
            with T("7  apply_forces (CPU transfer)"):
                _orig_apply(self, task, physics)

        self.iteration += 1

    # Wrap detailed_step with outer timer and replace BOTH instance
    # and class-level step to guarantee it's called regardless of lookup.
    def outer_step(self_h, task, physics):
        with T("TOTAL step"):
            detailed_step(self_h, task, physics)

    handler.step = types.MethodType(outer_step, handler)

    # ── Instrument fluid_step internals ──────────────────────────
    #    Replace fluid_step with a version that times each sub-part.
    _orig_adv_solve = adv.solve           # the dispatch alias
    _orig_set_bcs   = type(adv).set_BCs
    _orig_nd        = type(fs).normal_derivative
    _orig_div       = type(fs).divergence
    _orig_grad      = type(fs).gradient

    # Poisson
    poisson_fft = getattr(fs, "poisson_solverFFT", None)
    poisson_mg  = getattr(fs, "poisson_solver", None)
    if poisson_fft is not None:
        _orig_poisson_fft = type(poisson_fft).solve
    if poisson_mg is not None:
        _orig_poisson_mg = type(poisson_mg).solve_multigrid
        _orig_jacobi     = type(poisson_mg).Jacobi
        _orig_vcycle     = type(poisson_mg)._vcycle

    def detailed_fluid_step(self_h, u, v, w, p, timestep):
        ffs = self_h.fluid_solver

        # ── advection + diffusion ────────────────────────────────
        with T("4a   advection+diffusion"):
            (uprime, vprime, wprime) = _orig_adv_solve(u, v, w)

        with T("4b   set_BCs (post-advection)"):
            _orig_set_bcs(adv, uprime, vprime, wprime)

        # ── BDIM meta-equation (3 components) ────────────────────
        with T("4c   BDIM meta-equation"):
            uprime = (
                ffs.mu0_all_u * uprime
                + ffs.m_m0_all_u * comp.body_u
                + ffs.mu1_all_u * _orig_nd(ffs,
                    uprime - comp.body_u,
                    ffs.normal_x_u, ffs.normal_y_u, ffs.normal_z_u)
            )
            vprime = (
                ffs.mu0_all_v * vprime
                + ffs.m_m0_all_v * comp.body_v
                + ffs.mu1_all_v * _orig_nd(ffs,
                    vprime - comp.body_v,
                    ffs.normal_x_v, ffs.normal_y_v, ffs.normal_z_v)
            )
            wprime = (
                ffs.mu0_all_w * wprime
                + ffs.m_m0_all_w * comp.body_w
                + ffs.mu1_all_w * _orig_nd(ffs,
                    wprime - comp.body_w,
                    ffs.normal_x_w, ffs.normal_y_w, ffs.normal_z_w)
            )

        # ── divergence ───────────────────────────────────────────
        with T("4d   divergence"):
            ffs.div = _orig_div(ffs, uprime, vprime, wprime)

        # ── Poisson solve ────────────────────────────────────────
        if ffs.poisson_method == "fft":
            coeff = timestep / self_h.rho_fluid
            with T("4e   Poisson FFT solve"):
                p = _orig_poisson_fft(poisson_fft, ffs.div / coeff)
            with T("4f   gradient (pressure)"):
                (p_x, p_y, p_z) = _orig_grad(ffs, p)
            with T("4g   projection"):
                u = uprime - coeff * p_x
                v = vprime - coeff * p_y
                w = wprime - coeff * p_z
        else:
            with T("4e0  Poisson MG: rho coefficients"):
                rho_u = self_h.rho_fluid * ffs.mu0_all_u + self_h.rho_body * ffs.m_m0_all_u
                rho_v = self_h.rho_fluid * ffs.mu0_all_v + self_h.rho_body * ffs.m_m0_all_v
                rho_w = self_h.rho_fluid * ffs.mu0_all_w + self_h.rho_body * ffs.m_m0_all_w
                ch = timestep / rho_u
                cv = timestep / rho_v
                cw = timestep / rho_w

            with T("4e   Poisson multigrid solve"):
                p, _ = _orig_poisson_mg(poisson_mg,
                    ffs.div[1:-1, 1:-1, 1:-1],
                    torch.zeros_like(p),
                    (timestep / self_h.rho_body) * torch.ones_like(ffs.div),
                    ch=ch[1:,  1:-1, 1:-1],
                    cv=cv[1:-1, 1:,  1:-1],
                    cw=cw[1:-1, 1:-1, 1:],
                )

            with T("4f   gradient (pressure)"):
                (p_x, p_y, p_z) = _orig_grad(ffs, p)

            with T("4g   projection"):
                u = uprime - ch * p_x
                v = vprime - cv * p_y
                w = wprime - cw * p_z

        with T("4h   set_BCs (post-projection)"):
            _orig_set_bcs(adv, u, v, w)

        return (u, v, w, p)

    handler.fluid_step = types.MethodType(detailed_fluid_step, handler)

    # ── Instrument Poisson sub-components (Jacobi, v-cycle) ──────
    if poisson_mg is not None:
        _orig_jacobi_fn = type(poisson_mg).Jacobi
        def timed_jacobi(self_p, *a, **kw):
            with T("4e.i   Jacobi smoothing"):
                return _orig_jacobi_fn(self_p, *a, **kw)
        poisson_mg.Jacobi = types.MethodType(timed_jacobi, poisson_mg)

        _orig_vcycle_fn = type(poisson_mg)._vcycle
        _in_vcycle = [False]   # mutable flag for re-entrancy guard
        def timed_vcycle(self_p, *a, **kw):
            if _in_vcycle[0]:   # recursive call – skip timer to avoid double-count
                return _orig_vcycle_fn(self_p, *a, **kw)
            _in_vcycle[0] = True
            try:
                with T("4e.ii  V-cycle (top-level)"):
                    return _orig_vcycle_fn(self_p, *a, **kw)
            finally:
                _in_vcycle[0] = False
        poisson_mg._vcycle = types.MethodType(timed_vcycle, poisson_mg)

    # ── Instrument SDF update sub-parts ──────────────────────────
    #    Time per-body SDF evaluation inside update()
    _orig_body_update = type(handler).update
    def timed_update_detailed(self_h, t, iteration, dt=1):
        ffs = self_h.fluid_solver
        cmp = ffs.composite_body
        gs  = ffs.grid_shape

        with T("1a   FARMS kinematics (data gather)"):
            from scipy.spatial.transform import Rotation
            com_poses  = []
            urdf_poses = []
            Rs         = []
            lin_vels   = []
            ang_vels   = []
            for exp_data in self_h.data:
                sen = exp_data.sensors.links
                com_poses.append(self_h.cython2numpy(sen.com_positions()[iteration, :]))
                urdf_poses.append(self_h.cython2numpy(sen.urdf_positions()[iteration, :]))
                Rs.append(self_h.cython2numpy(
                    Rotation.from_quat(sen.urdf_orientations()[iteration, :]).as_matrix().astype(self_h.dtype_np)))
                lin_vels.append(self_h.cython2numpy(sen.com_lin_velocities()[iteration, :]))
                nlinks = len(sen.names)
                ang_vels.append(self_h.cython2numpy(
                    np.stack([sen.com_ang_velocity(iteration, lk) for lk in range(nlinks)])))

        with T("1b   SDF eval (per-body x 4 grids)"):
            for body_i, body in enumerate(cmp.bodies):
                (animat_id, link_id) = cmp.body_ids[body_i]
                com_pos  = com_poses[animat_id][link_id]
                urdf_pos = urdf_poses[animat_id][link_id]
                R        = Rs[animat_id][link_id]
                lin_vel  = lin_vels[animat_id][link_id]
                ang_vel  = ang_vels[animat_id][link_id]

                pos_trans = R.T @ (cmp.stacked_xy - urdf_pos[:, None])
                sdf_cc = body.sdf(pos_trans[0].reshape(gs), pos_trans[1].reshape(gs), pos_trans[2].reshape(gs))

                pos_trans_u = R.T @ (cmp.stacked_xy_u - urdf_pos[:, None])
                sdf_u = body.sdf(pos_trans_u[0].reshape(gs), pos_trans_u[1].reshape(gs), pos_trans_u[2].reshape(gs))
                vel_u = (lin_vel[0] + ang_vel[1] * (cmp.Zu_stag - com_pos[2]) - ang_vel[2] * (cmp.Yu_stag - com_pos[1]))

                pos_trans_v = R.T @ (cmp.stacked_xy_v - urdf_pos[:, None])
                sdf_v = body.sdf(pos_trans_v[0].reshape(gs), pos_trans_v[1].reshape(gs), pos_trans_v[2].reshape(gs))
                vel_v = (lin_vel[1] + ang_vel[2] * (cmp.Xv_stag - com_pos[0]) - ang_vel[0] * (cmp.Zv_stag - com_pos[2]))

                pos_trans_w = R.T @ (cmp.stacked_xy_w - urdf_pos[:, None])
                sdf_w = body.sdf(pos_trans_w[0].reshape(gs), pos_trans_w[1].reshape(gs), pos_trans_w[2].reshape(gs))
                vel_w = (lin_vel[2] + ang_vel[0] * (cmp.Yw_stag - com_pos[1]) - ang_vel[1] * (cmp.Xw_stag - com_pos[0]))

                cmp.sdf_vals[body_i] = sdf_cc

                if body_i == 0:
                    cmp.sdf_val = sdf_cc; cmp.sdf_val_u = sdf_u; cmp.body_u = vel_u
                    cmp.sdf_val_v = sdf_v; cmp.body_v = vel_v
                    cmp.sdf_val_w = sdf_w; cmp.body_w = vel_w
                else:
                    mask   = sdf_cc < cmp.sdf_val;  cmp.sdf_val   = torch.where(mask,   sdf_cc, cmp.sdf_val)
                    mask_u = sdf_u  < cmp.sdf_val_u; cmp.sdf_val_u = torch.where(mask_u, sdf_u,  cmp.sdf_val_u); cmp.body_u = torch.where(mask_u, vel_u, cmp.body_u)
                    mask_v = sdf_v  < cmp.sdf_val_v; cmp.sdf_val_v = torch.where(mask_v, sdf_v,  cmp.sdf_val_v); cmp.body_v = torch.where(mask_v, vel_v, cmp.body_v)
                    mask_w = sdf_w  < cmp.sdf_val_w; cmp.sdf_val_w = torch.where(mask_w, sdf_w,  cmp.sdf_val_w); cmp.body_w = torch.where(mask_w, vel_w, cmp.body_w)

                cmp.com_pos[body_i] = com_pos
                body.com_pos = com_pos

    handler.update = types.MethodType(timed_update_detailed, handler)
    ffs_ref = handler.fluid_solver
    ffs_ref.composite_body.update = handler.update

    print(f"  [profiler] Instrumented handler (grid {fs.grid_shape}, "
          f"device {fs.device}, poisson={fs.poisson_method})", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Patch FluidExtension to instrument handler after creation
# ═══════════════════════════════════════════════════════════════════════
_orig_init_episode = FluidExtension.initialize_episode

def _patched_init_episode(self, task, physics):
    _orig_init_episode(self, task, physics)
    if hasattr(self, "BDIMhandler") and self.BDIMhandler is not None:
        instrument_handler(self.BDIMhandler)

FluidExtension.initialize_episode = _patched_init_episode

# NOTE: Outer step timing is handled via instance-level patch in
# instrument_handler (detailed_step -> outer_step).  We do NOT patch
# BDIMhandler3D.step at the class level because FARMS before_step
# calls self.BDIMhandler.step() which resolves to the instance attr.


# ═══════════════════════════════════════════════════════════════════════
# Generate configs with ONLY FluidExtension, headless=True
# ═══════════════════════════════════════════════════════════════════════

import lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_3d as cfg

cfg.Nx           = args.Nx
cfg.Ny           = args.Ny
cfg.Nz           = args.Nz
cfg.n_iterations = args.n_steps + 1
cfg.save_every   = args.save_every
cfg.save_frames  = False
cfg.headless     = True     # ← headless
cfg.save_uv      = False

# Override gen_simulation_config to STRIP all non-fluid extensions
_orig_gen_sim_config = cfg.gen_simulation_config

def gen_simulation_config_lean(output_folder):
    """Generate simulation config with only the FluidExtension."""
    _orig_gen_sim_config(output_folder)

    # Read the generated YAML and strip unwanted extensions
    import yaml
    yaml_path = os.path.join(output_folder, 'simulation_config.yaml')
    with open(yaml_path, 'r') as f:
        sim_dict = yaml.safe_load(f)

    # Keep only FluidExtension
    keep_loaders = {"lilytorch.integration.extensions.FluidExtension"}
    original_exts = sim_dict.get("extensions", [])
    sim_dict["extensions"] = [
        ext for ext in original_exts
        if ext.get("loader", "") in keep_loaders
    ]
    # Force headless
    sim_dict.setdefault("runtime", {})["headless"] = True

    with open(yaml_path, 'w') as f:
        yaml.dump(sim_dict, f, default_flow_style=False, sort_keys=False)

    stripped = len(original_exts) - len(sim_dict["extensions"])
    print(f"  [profiler] Stripped {stripped} extensions, kept only FluidExtension")
    print(f"  [profiler] headless = True")

cfg.gen_simulation_config = gen_simulation_config_lean

print("=" * 72)
print(f"  3-D BDIM Detailed Cost Analysis  (headless, fluid-only)")
print(f"  Grid:       {cfg.Nx} x {cfg.Ny} x {cfg.Nz}  "
      f"({cfg.Nx*cfg.Ny*cfg.Nz:,} cells)")
print(f"  Steps:      {args.n_steps}  (warm-up: {args.warmup})")
print(f"  Device:     {'CUDA' if USE_CUDA else 'CPU'}")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
# Run FARMS in-process
# ═══════════════════════════════════════════════════════════════════════

from lilytorch.util.paths import gen_new_folder, save_path
output_folder = gen_new_folder(save_path)
os.makedirs(output_folder, exist_ok=True)
print(f"  Config folder: {output_folder}")

cfg.gen_animat_config(output_folder)
cfg.gen_arena_config(output_folder)
cfg.gen_simulation_config(output_folder)   # uses our lean override
cfg.gen_experiment_config(output_folder)
os.chdir(output_folder)

from farms_core.experiment.options import ExperimentOptions
from farms_core.extensions.extensions import import_item
from farms_sim.simulation import run_simulation
from farms_core.simulation.options import Simulator

experiment_options = ExperimentOptions.load("experiment_config.yaml")
experiment_data_loader = import_item(experiment_options.loaders.experiment_data)
experiment_data = experiment_data_loader.from_options(experiment_options)

print("  Starting simulation...\n")
sim = run_simulation(
    experiment_data=experiment_data,
    experiment_options=experiment_options,
    simulator=Simulator.MUJOCO,
)


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/')
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
os.makedirs(args.out_dir, exist_ok=True)

summary = T.summary()
n_meas = max(1, T._step_count - args.warmup)

if not summary:
    print(f"\nERROR: No measurements (step_count={T._step_count}, warmup={args.warmup})")
    sys.exit(1)

grid_N = args.Nx * args.Ny * args.Nz
step_mean = summary.get("TOTAL step", {}).get("mean", 0)

print("\n" + "=" * 108)
print(f"  TIMING RESULTS  ({n_meas} measured steps, "
      f"grid {args.Nx}x{args.Ny}x{args.Nz} = {grid_N:,} cells, "
      f"{'GPU' if USE_CUDA else 'CPU'})")
print("=" * 108)
hdr = (f"  {'Component':<44s} {'Mean':>9s} {'Std':>9s} {'Min':>9s} "
       f"{'Max':>9s} {'Total':>9s} {'%step':>7s} {'Calls':>6s}")
print(hdr)
print(f"  {'':44s} {'(ms)':>9s} {'(ms)':>9s} {'(ms)':>9s} "
       f"{'(ms)':>9s} {'(s)':>9s} {'':>7s}")
print("-" * 108)

for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
    s = summary[label]
    print(f"  {label:<44s} {1e3*s['mean']:9.2f} {1e3*s['std']:9.2f} "
          f"{1e3*s['min']:9.2f} {1e3*s['max']:9.2f} "
          f"{s['total']:9.4f} {s['pct']:7.1f} {s['count']:6d}")

print("=" * 108)
if step_mean > 0:
    print(f"\n  Average step time: {1e3*step_mean:.2f} ms  "
          f"({1.0/step_mean:.1f} steps/s)")
    print(f"  Grid cells: {grid_N:,}  ({1e6*step_mean/grid_N:.3f} us/cell/step)")


# ── Save CSV ─────────────────────────────────────────────────────────
csv_path = os.path.join(args.out_dir,
                        f"cost_breakdown_{args.Nx}x{args.Ny}x{args.Nz}.csv")
with open(csv_path, "w") as f:
    f.write("component,mean_ms,std_ms,min_ms,max_ms,total_s,pct_of_step,calls\n")
    for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
        s = summary[label]
        f.write(f"{label},{1e3*s['mean']:.4f},{1e3*s['std']:.4f},"
                f"{1e3*s['min']:.4f},{1e3*s['max']:.4f},"
                f"{s['total']:.6f},{s['pct']:.2f},{s['count']}\n")
print(f"\n  CSV saved -> {csv_path}")


# ── Figures ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_labels = [l for l in summary if l != "TOTAL step"]
    plot_labels.sort(key=lambda k: -summary[k]["pct"])

    if not plot_labels:
        raise ValueError("No sub-step timers")

    # ── Horizontal bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(6, 0.55 * len(plot_labels))))
    pcts  = [summary[l]["pct"] for l in plot_labels]
    means = [1e3 * summary[l]["mean"] for l in plot_labels]

    # Colour by category
    def _colour(label):
        if "SDF" in label or "sdf" in label:     return "#4caf50"
        if "mu_funcs" in label:                    return "#8bc34a"
        if "normals" in label:                     return "#cddc39"
        if "advection" in label:                   return "#2196f3"
        if "BDIM" in label:                        return "#03a9f4"
        if "Poisson" in label or "poisson" in label: return "#f44336"
        if "Jacobi" in label or "V-cycle" in label:  return "#e57373"
        if "divergence" in label:                  return "#9c27b0"
        if "gradient" in label:                    return "#ce93d8"
        if "projection" in label:                  return "#ba68c8"
        if "forces" in label:                      return "#ff9800"
        if "BCs" in label:                         return "#78909c"
        if "apply" in label:                       return "#ffb74d"
        if "plotting" in label:                    return "#607d8b"
        return "#90a4ae"

    colours = [_colour(l) for l in plot_labels]
    bars = ax.barh(plot_labels, pcts, color=colours,
                   edgecolor="white", linewidth=0.5)

    for bar, m, p in zip(bars, means, pcts):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{m:.2f} ms ({p:.1f}%)", va="center", fontsize=7.5)

    ax.set_xlabel("% of total step time")
    ax.set_title(f"3-D BDIM Cost Breakdown  -  "
                 f"{args.Nx}x{args.Ny}x{args.Nz}  "
                 f"({n_meas} steps, {'GPU' if USE_CUDA else 'CPU'})")
    ax.invert_yaxis()
    ax.set_xlim(0, max(pcts) * 1.45)
    fig.tight_layout()

    fig_path = os.path.join(args.out_dir,
                            f"cost_breakdown_{args.Nx}x{args.Ny}x{args.Nz}.png")
    fig.savefig(fig_path, dpi=180)
    print(f"  Figure saved -> {fig_path}")
    plt.close(fig)

    # ── Pie chart (group small items) ────────────────────────────
    threshold = 2.0
    pie_labels, pie_vals = [], []
    other_pct = 0.0
    for l in plot_labels:
        p = summary[l]["pct"]
        if p >= threshold:
            pie_labels.append(l)
            pie_vals.append(p)
        else:
            other_pct += p
    if other_pct > 0:
        pie_labels.append("other")
        pie_vals.append(other_pct)

    pie_colours = [_colour(l) for l in pie_labels]
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, autopct="%1.1f%%",
        startangle=140, pctdistance=0.80, textprops={"fontsize": 8},
        colors=pie_colours)
    ax2.set_title(f"Step Cost Distribution  -  "
                  f"{args.Nx}x{args.Ny}x{args.Nz}  "
                  f"({'GPU' if USE_CUDA else 'CPU'})")
    pie_path = os.path.join(args.out_dir,
                            f"cost_pie_{args.Nx}x{args.Ny}x{args.Nz}.png")
    fig2.savefig(pie_path, dpi=180)
    print(f"  Figure saved -> {pie_path}")
    plt.close(fig2)

    # ── Stacked summary: group into logical categories ───────────
    categories = {
        "SDF update":        ["1a", "1b"],
        "mu + normals":      ["2 ", "3 "],
        "Advection+Diffusion": ["4a"],
        "BDIM meta-eq":      ["4c"],
        "Poisson solve":     ["4e"],
        "Projection":        ["4d", "4f", "4g"],
        "BCs":               ["4b", "4h"],
        "Forces":            ["5 "],
        "I/O & misc":        ["6 ", "7 "],
    }
    cat_pcts = {}
    for cat_name, prefixes in categories.items():
        total = 0
        for l, s in summary.items():
            if any(l.startswith(pfx) for pfx in prefixes):
                total += s["pct"]
        if total > 0:
            cat_pcts[cat_name] = total

    if cat_pcts:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        cat_names = list(cat_pcts.keys())
        cat_vals  = [cat_pcts[c] for c in cat_names]
        cat_cols  = ["#4caf50", "#cddc39", "#2196f3", "#03a9f4",
                     "#f44336", "#9c27b0", "#78909c", "#ff9800", "#607d8b"]
        bars3 = ax3.bar(cat_names, cat_vals,
                        color=cat_cols[:len(cat_names)],
                        edgecolor="white")
        for bar, v in zip(bars3, cat_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", fontsize=9)
        ax3.set_ylabel("% of total step time")
        ax3.set_title(f"Grouped Cost Summary  -  "
                      f"{args.Nx}x{args.Ny}x{args.Nz}")
        ax3.set_ylim(0, max(cat_vals) * 1.2)
        plt.xticks(rotation=25, ha="right")
        fig3.tight_layout()
        cat_path = os.path.join(args.out_dir,
                                f"cost_grouped_{args.Nx}x{args.Ny}x{args.Nz}.png")
        fig3.savefig(cat_path, dpi=180)
        print(f"  Figure saved -> {cat_path}")
        plt.close(fig3)

except Exception as e:
    import traceback
    print(f"  Figure generation failed: {e}")
    traceback.print_exc()


# ── Suggested tests ──────────────────────────────────────────────────
print(f"""
+----------------------------------------------------------------------+
|  Suggested cost-scaling tests:                                       |
|                                                                      |
|  1.  python run_cost_analysis_3d.py --Nx  64 --Ny  16 --Nz  16      |
|  2.  python run_cost_analysis_3d.py --Nx 128 --Ny  32 --Nz  32      |
|  3.  python run_cost_analysis_3d.py --Nx 256 --Ny  64 --Nz  64      |
|  4.  python run_cost_analysis_3d.py --Nx 512 --Ny 128 --Nz 128      |
|  5.  Add --device cpu to any of the above for CPU baseline           |
|                                                                      |
|  Scaling:                                                            |
|   FFT Poisson: O(N log N)  where N = 2*Nx * 2*Ny * 2*Nz             |
|   MG Poisson:  O(N) per V-cycle, but n_cycles may vary              |
|   Advection:   O(N)   (3 components x 3 dirs)                       |
|   SDF update:  O(n_bodies x 4 x N)                                  |
|   Normals:     O(4 x N)   (torch.gradient)                          |
+----------------------------------------------------------------------+
""")
