#!/usr/bin/env python3
"""
Computational-cost analysis for the 3-D free-swimming 1guilla simulation.

Instruments every major sub-kernel inside BDIMhandler3D with CUDA-synchronised
timers.  Reports per-step statistics, saves CSV and **paper-quality** figures.

Operation groups (user-specified + extras):
  1. FARMS step         – MuJoCo physics + apply_forces (CPU ↔ GPU transfer)
  2. Body update        – SDF evaluation on 4 staggered grids
  3. Body interp. & normals – Heaviside (mu) + normal computation
  4. Convection–diffusion   – advection + diffusion + BCs
  5. Projection         – Poisson solve + gradient + velocity correction
  6. BDIM meta-equation – immersed-boundary forcing (3 components)
  7. Forces             – forces_method2_3d + plotting/saving

KEY DESIGN CHOICES:
  - Runs the FARMS simulation **in-process** so Python-level monkey-patches
    apply to every hot path.
  - All non-fluid extensions (FlowViewer, CameraRecording, ExperimentLogger,
    MjcfSaver, DataLogger) are stripped for pure solver cost.
  - Headless mode.

Usage
-----
    source /path/to/venv/bin/activate
    python run_cost_analysis.py                                      # default 128×32×32, 20 steps
    python run_cost_analysis.py --Nx 256 --Ny 64 --Nz 64 --n_steps 50
    python run_cost_analysis.py --Nx 512 --Ny 128 --Nz 128 --n_steps 30
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

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="3-D free-swimming BDIM cost analysis (paper figures)")
parser.add_argument("--Nx",        type=int, default=128)
parser.add_argument("--Ny",        type=int, default=32)
parser.add_argument("--Nz",        type=int, default=32)
parser.add_argument("--n_steps",   type=int, default=30,
                    help="Total steps (first --warmup are discarded)")
parser.add_argument("--warmup",    type=int, default=10,
                    help="Warm-up steps to discard (>=10 recommended "
                         "when torch.compile is active)")
parser.add_argument("--save_every",type=int, default=9999)
parser.add_argument("--device",    type=str, default="cuda",
                    choices=["cuda", "cpu"])
parser.add_argument("--out_dir",   type=str, default=None,
                    help="Output directory (default: figures/ in this folder)")
args = parser.parse_args()

USE_CUDA = args.device == "cuda" and torch.cuda.is_available()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.out_dir is None:
    args.out_dir = os.path.join(SCRIPT_DIR, "figures")


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
                  f"Collecting from step {self._step_count}…\n", flush=True)

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
# Project imports
# ═══════════════════════════════════════════════════════════════════════
# Ensure the repo root is on the path
repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from lilytorch.integration.extensions import FluidExtension
from lilytorch.farms_examples._1guillasim.BDIMhandler3D import BDIMhandler3D  # noqa: F401
from lilytorch.src.poisson_mult import PoissonSolver  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Deep instrumentation (instance-level monkey-patching)
# ═══════════════════════════════════════════════════════════════════════

def instrument_handler(handler):
    """Wrap every hot-path method on live instances with fine-grained timers."""
    if getattr(handler, "_profiled", False):
        return
    handler._profiled = True

    fs   = handler.fluid_solver
    comp = fs.composite_body
    adv  = fs.adv_diff_solver

    # Keep originals
    _orig_update     = type(handler).update
    _orig_fluid_step = type(handler).fluid_step
    _orig_apply      = type(handler).apply_forces
    _orig_forces     = type(fs).forces_method2_3d
    _orig_plot       = type(fs).plotting_and_saving

    # ── Replacement for handler.step ─────────────────────────────
    def detailed_step(self, task, physics):
        T.new_step()
        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            self.iteration += 1
            return

        t   = iteration * timestep
        ffs = self.fluid_solver

        if not self.terminate:

            # ── 1. SDF update ────────────────────────────────────
            with T("1  SDF update (body kinematics + SDF eval)"):
                self.update(t, iteration, dt=timestep)

            # ── 2-3. mu + normals on 4 staggered grids ──────────
            if getattr(ffs, '_compile_sdf', False):
                # ── Batched + compiled path (all 4 grids in one fused pass) ──
                from lilytorch.src.body import _mu_normals_batched_3d_compiled
                with T("2  mu+normals (batched+compiled)"):
                    mu0, mu1, nx, ny, nz = _mu_normals_batched_3d_compiled(
                        comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                        comp.sdf_val, comp.h, comp.eps,
                    )
                # Clone outputs — CUDA graph buffers are overwritten on replay
                mu0, mu1 = mu0.clone(), mu1.clone()
                nx, ny, nz = nx.clone(), ny.clone(), nz.clone()

                # Unstack [u, v, w, cc] and store on solver
                ffs.mu0_all_u, ffs.mu1_all_u = mu0[0], mu1[0]
                ffs.m_m0_all_u = 1 - mu0[0]
                ffs.normal_x_u, ffs.normal_y_u, ffs.normal_z_u = nx[0], ny[0], nz[0]

                ffs.mu0_all_v, ffs.mu1_all_v = mu0[1], mu1[1]
                ffs.m_m0_all_v = 1 - mu0[1]
                ffs.normal_x_v, ffs.normal_y_v, ffs.normal_z_v = nx[1], ny[1], nz[1]

                ffs.mu0_all_w, ffs.mu1_all_w = mu0[2], mu1[2]
                ffs.m_m0_all_w = 1 - mu0[2]
                ffs.normal_x_w, ffs.normal_y_w, ffs.normal_z_w = nx[2], ny[2], nz[2]

                ffs.mu0_all, ffs.mu1_all = mu0[3], mu1[3]
                ffs.m_m0_all = 1 - mu0[3]
                ffs.normal_x, ffs.normal_y, ffs.normal_z = nx[3], ny[3], nz[3]
            else:
                # ── Eager path: 4 × individual calls ──
                sdf_fields = [
                    ("u",  comp.sdf_val_u),
                    ("v",  comp.sdf_val_v),
                    ("w",  comp.sdf_val_w),
                    ("cc", comp.sdf_val),
                ]
                mu_attrs = [
                    ("mu0_all_u", "mu1_all_u", "m_m0_all_u", "normal_x_u", "normal_y_u", "normal_z_u"),
                    ("mu0_all_v", "mu1_all_v", "m_m0_all_v", "normal_x_v", "normal_y_v", "normal_z_v"),
                    ("mu0_all_w", "mu1_all_w", "m_m0_all_w", "normal_x_w", "normal_y_w", "normal_z_w"),
                    ("mu0_all",   "mu1_all",   "m_m0_all",   "normal_x",   "normal_y",   "normal_z"),
                ]
                for (grid_name, sdf_field), (a_mu0, a_mu1, a_mm0, a_nx, a_ny, a_nz) in zip(sdf_fields, mu_attrs):
                    with T("2  mu_funcs (Heaviside)"):
                        (mu0, mu1) = comp.mu_funcs(sdf_field)
                    setattr(ffs, a_mu0, mu0)
                    setattr(ffs, a_mu1, mu1)
                    setattr(ffs, a_mm0, 1 - mu0)

                    with T("3  compute_normals"):
                        (nx, ny, nz) = comp.compute_normals(sdf_field)
                    setattr(ffs, a_nx, nx)
                    setattr(ffs, a_ny, ny)
                    setattr(ffs, a_nz, nz)

            # ── 4. Fluid step ────────────────────────────────────
            with T("4  fluid_step (total PDE)"):
                (u, v, w, p) = self.fluid_step(
                    ffs.u0, ffs.v0, ffs.w0, ffs.p0, timestep)

            (ffs.u0, ffs.v0, ffs.w0, ffs.p0) = (u, v, w, p)

            # ── 5. Forces ────────────────────────────────────────
            with T("5  forces_method2_3d"):
                _orig_forces(ffs, ffs.u0, ffs.v0, ffs.w0, ffs.p0, iteration)

            # ── 6. Plotting / saving ─────────────────────────────
            with T("6  plotting_and_saving"):
                self.terminate = _orig_plot(
                    ffs, ffs.u0, ffs.v0, ffs.p0, iteration, w_vel=ffs.w0)

            # ── 7. Apply forces (FARMS ← GPU) ───────────────────
            with T("7  apply_forces (FARMS)"):
                _orig_apply(self, task, physics)

        self.iteration += 1

    def outer_step(self_h, task, physics):
        with T("TOTAL step"):
            detailed_step(self_h, task, physics)
    handler.step = types.MethodType(outer_step, handler)

    # ── Instrument fluid_step internals ──────────────────────────
    _orig_adv_solve = adv.solve
    _orig_set_bcs   = type(adv).set_BCs
    _orig_nd        = type(fs).normal_derivative
    _orig_div       = type(fs).divergence
    _orig_grad      = type(fs).gradient

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

        # ── BDIM meta-equation ───────────────────────────────────
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
            with T("4e   Poisson solve"):
                p = _orig_poisson_fft(poisson_fft, ffs.div / coeff)
            with T("4f   gradient (pressure)"):
                (p_x, p_y, p_z) = _orig_grad(ffs, p)
            with T("4g   projection (vel. correction)"):
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

            with T("4e   Poisson solve"):
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

            with T("4g   projection (vel. correction)"):
                u = uprime - ch * p_x
                v = vprime - cv * p_y
                w = wprime - cw * p_z

        with T("4h   set_BCs (post-projection)"):
            _orig_set_bcs(adv, u, v, w)

        return (u, v, w, p)

    handler.fluid_step = types.MethodType(detailed_fluid_step, handler)

    # ── Instrument Poisson sub-components ────────────────────────
    # NOTE: Sub-component timers (Jacobi, V-cycle) each insert a
    # cuda.synchronize() before and after.  At small grids the sync
    # overhead (~50-100 µs per call) dominates the actual kernel time,
    # causing the profiled Poisson cost to appear 5-20× higher than
    # reality.  We therefore ONLY instrument sub-components when the
    # grid is large enough that the sync overhead is negligible.
    _instrument_poisson_internals = (args.Nx * args.Ny * args.Nz) >= 500_000

    if poisson_mg is not None and _instrument_poisson_internals:
        _orig_jacobi_fn = type(poisson_mg).Jacobi
        def timed_jacobi(self_p, *a, **kw):
            with T("4e.i   Jacobi smoothing"):
                return _orig_jacobi_fn(self_p, *a, **kw)
        poisson_mg.Jacobi = types.MethodType(timed_jacobi, poisson_mg)

        _orig_vcycle_fn = type(poisson_mg)._vcycle
        _in_vcycle = [False]
        def timed_vcycle(self_p, *a, **kw):
            if _in_vcycle[0]:
                return _orig_vcycle_fn(self_p, *a, **kw)
            _in_vcycle[0] = True
            try:
                with T("4e.ii  V-cycle (top-level)"):
                    return _orig_vcycle_fn(self_p, *a, **kw)
            finally:
                _in_vcycle[0] = False
        poisson_mg._vcycle = types.MethodType(timed_vcycle, poisson_mg)

    # ── Instrument SDF update sub-parts ──────────────────────────
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

        with T("1b   SDF eval (per-body × 4 grids)"):
            _use_compile = getattr(ffs, '_compile_sdf', False)
            if _use_compile:
                from lilytorch.src.body import _rotate_grid_3d_compiled
                _rotate_fn = _rotate_grid_3d_compiled
            else:
                from lilytorch.src.body import rotate_grid_3d as _rotate_fn

            for body_i, body in enumerate(cmp.bodies):
                (animat_id, link_id) = cmp.body_ids[body_i]
                com_pos  = com_poses[animat_id][link_id]
                urdf_pos = urdf_poses[animat_id][link_id]
                R        = Rs[animat_id][link_id]
                lin_vel  = lin_vels[animat_id][link_id]
                ang_vel  = ang_vels[animat_id][link_id]

                R_T = R.T

                # ── Full-grid SDF evaluation ────────────────────
                px, py, pz = _rotate_fn(cmp.X, cmp.Y, cmp.Z_grid,
                                        R_T, urdf_pos)
                sdf_cc = body.sdf(px, py, pz)
                cmp.sdf_vals[body_i] = sdf_cc

                # Staggered SDFs via averaging of CC neighbours (O(h²))
                # (eager stagger — clone overhead from CUDA graph
                # buffer reuse exceeds the compilation savings)
                from lilytorch.src.body import _stagger_sdf_3d
                sdf_u, sdf_v, sdf_w = _stagger_sdf_3d(sdf_cc)

                vel_u = (lin_vel[0] + ang_vel[1] * (cmp.Zu_stag - com_pos[2]) - ang_vel[2] * (cmp.Yu_stag - com_pos[1]))
                vel_v = (lin_vel[1] + ang_vel[2] * (cmp.Xv_stag - com_pos[0]) - ang_vel[0] * (cmp.Zv_stag - com_pos[2]))
                vel_w = (lin_vel[2] + ang_vel[0] * (cmp.Yw_stag - com_pos[1]) - ang_vel[1] * (cmp.Xw_stag - com_pos[0]))

                if body_i == 0:
                    cmp.sdf_val   = sdf_cc; cmp.sdf_val_u = sdf_u; cmp.body_u = vel_u
                    cmp.sdf_val_v = sdf_v;  cmp.body_v    = vel_v
                    cmp.sdf_val_w = sdf_w;  cmp.body_w    = vel_w
                else:
                    mask   = sdf_cc < cmp.sdf_val;   cmp.sdf_val   = torch.where(mask,   sdf_cc, cmp.sdf_val)
                    mask_u = sdf_u  < cmp.sdf_val_u;  cmp.sdf_val_u = torch.where(mask_u, sdf_u,  cmp.sdf_val_u); cmp.body_u = torch.where(mask_u, vel_u, cmp.body_u)
                    mask_v = sdf_v  < cmp.sdf_val_v;  cmp.sdf_val_v = torch.where(mask_v, sdf_v,  cmp.sdf_val_v); cmp.body_v = torch.where(mask_v, vel_v, cmp.body_v)
                    mask_w = sdf_w  < cmp.sdf_val_w;  cmp.sdf_val_w = torch.where(mask_w, sdf_w,  cmp.sdf_val_w); cmp.body_w = torch.where(mask_w, vel_w, cmp.body_w)

                cmp.com_pos[body_i] = com_pos
                body.com_pos = com_pos

    handler.update = types.MethodType(timed_update_detailed, handler)
    fs.composite_body.update = handler.update

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


# ═══════════════════════════════════════════════════════════════════════
# Configure: free-swimming 1guilla (headless, fluid-only)
# ═══════════════════════════════════════════════════════════════════════

import lilytorch.farms_examples._1guillasim.gen_configs_one_free_3d as cfg

cfg.Nx           = args.Nx
cfg.Ny           = args.Ny
cfg.Nz           = args.Nz
cfg.use_bdim     = True

# Scale domain extents so that dx = dy = dz regardless of grid shape.
# The baseline domain is centred at (+0.3, 0, 0) with extents 2.4×0.6×0.6.
# For grids larger than production, EXTEND the domain to maintain dx.
# For grids ≤ production (4:1:1), keep the original domain (coarser dx).
_dx_ref = 2.4 / 512                   # reference spacing from production grid
_Lx = max(2.4, args.Nx * _dx_ref)     # extend if grid is wider
_Ly = max(0.6, args.Ny * _dx_ref)
_Lz = max(0.6, args.Nz * _dx_ref)
cfg.xmin = 0.3 - 0.5 * _Lx;  cfg.xmax = 0.3 + 0.5 * _Lx
cfg.ymin = -0.5 * _Ly;        cfg.ymax =  0.5 * _Ly
cfg.zmin = -0.5 * _Lz;        cfg.zmax =  0.5 * _Lz

cfg.n_iterations = args.n_steps + 1
cfg.save_every   = args.save_every
cfg.save_frames  = False
cfg.headless     = True
cfg.save_uv      = False

# Override gen_simulation_config to strip non-fluid extensions
_orig_gen_sim_config = cfg.gen_simulation_config

def gen_simulation_config_lean(output_folder):
    """Keep only FluidExtension — strip viewers, loggers, camera."""
    _orig_gen_sim_config(output_folder)
    import yaml
    yaml_path = os.path.join(output_folder, "simulation_config.yaml")
    with open(yaml_path, "r") as f:
        sim_dict = yaml.safe_load(f)

    keep_loaders = {"lilytorch.integration.extensions.FluidExtension"}
    original_exts = sim_dict.get("extensions", [])
    sim_dict["extensions"] = [
        ext for ext in original_exts if ext.get("loader", "") in keep_loaders
    ]
    sim_dict.setdefault("runtime", {})["headless"] = True

    # Enable torch.compile for convection/diffusion + Poisson solver + forces
    # The compile flags live inside the FluidExtension's bdim_yaml.solver dict
    for ext in sim_dict["extensions"]:
        bdim_yaml = ext.get("config", {}).get("bdim_yaml", {})
        solver_cfg = bdim_yaml.get("solver", {})
        if solver_cfg:
            solver_cfg["compile_adv_diff"] = True
            solver_cfg["poisson_compile"]  = True
            solver_cfg["compile_forces"]   = True
            solver_cfg["compile_sdf"]      = True

    with open(yaml_path, "w") as f:
        yaml.dump(sim_dict, f, default_flow_style=False, sort_keys=False)

    stripped = len(original_exts) - len(sim_dict["extensions"])
    print(f"  [profiler] Stripped {stripped} extensions, kept FluidExtension only")
    print(f"  [profiler] Enabled torch.compile for adv-diff + Poisson + forces + SDF")

cfg.gen_simulation_config = gen_simulation_config_lean

grid_N = args.Nx * args.Ny * args.Nz
print("=" * 72)
print("  3-D Free-Swimming 1guilla – Computational Cost Analysis")
print(f"  Grid:   {args.Nx} × {args.Ny} × {args.Nz}  ({grid_N:,} cells)")
print(f"  Steps:  {args.n_steps}  (warm-up: {args.warmup})")
print(f"  Device: {'CUDA' if USE_CUDA else 'CPU'}")
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
cfg.gen_simulation_config(output_folder)
cfg.gen_experiment_config(output_folder)
os.chdir(output_folder)

from farms_core.experiment.options import ExperimentOptions
from farms_core.extensions.extensions import import_item
from farms_sim.simulation import run_simulation
from farms_core.simulation.options import Simulator

experiment_options = ExperimentOptions.load("experiment_config.yaml")
experiment_data_loader = import_item(experiment_options.loaders.experiment_data)
experiment_data = experiment_data_loader.from_options(experiment_options)

print("  Starting simulation…\n")
sim = run_simulation(
    experiment_data=experiment_data,
    experiment_options=experiment_options,
    simulator=Simulator.MUJOCO,
)


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

os.chdir(SCRIPT_DIR)
os.makedirs(args.out_dir, exist_ok=True)

summary = T.summary()
n_meas = max(1, T._step_count - args.warmup)

if not summary:
    print(f"\nERROR: No measurements (step_count={T._step_count}, "
          f"warmup={args.warmup})")
    sys.exit(1)

step_mean = summary.get("TOTAL step", {}).get("mean", 0)

print("\n" + "=" * 108)
print(f"  TIMING RESULTS  ({n_meas} measured steps, "
      f"grid {args.Nx}×{args.Ny}×{args.Nz} = {grid_N:,} cells, "
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
    print(f"  Grid cells: {grid_N:,}  "
          f"({1e6*step_mean/grid_N:.3f} µs/cell/step)")


# ── Save CSV ─────────────────────────────────────────────────────────
tag = f"{args.Nx}x{args.Ny}x{args.Nz}"
csv_path = os.path.join(args.out_dir, f"cost_breakdown_{tag}.csv")
with open(csv_path, "w") as f:
    f.write("component,mean_ms,std_ms,min_ms,max_ms,total_s,pct_of_step,calls\n")
    for label in sorted(summary.keys(), key=lambda k: -summary[k]["total"]):
        s = summary[label]
        f.write(f"{label},{1e3*s['mean']:.4f},{1e3*s['std']:.4f},"
                f"{1e3*s['min']:.4f},{1e3*s['max']:.4f},"
                f"{s['total']:.6f},{s['pct']:.2f},{s['count']}\n")
print(f"\n  CSV saved → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
# Paper-quality figures
# ═══════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Publication style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# ── Aggregate into paper categories ──────────────────────────────────
# Maps category display name → list of timer-label prefixes
CATEGORIES = {
    "Body update (SDF)":           ["1b"],
    "Forces\ncomputation":         ["5 "],
    "Projection\n(pressure)":      ["4d", "4e ", "4f", "4g", "4h"],
    "Convection\n& diffusion":     ["4a", "4b"],
    "Other":                       ["1a", "7 ", "2 ", "3 ", "4c", "6 "],
}

CAT_COLOURS = {
    "Body update (SDF)":           "#26a69a",
    "Forces\ncomputation":         "#ffa726",
    "Projection\n(pressure)":      "#ef5350",
    "Convection\n& diffusion":     "#42a5f5",
    "Other":                       "#90a4ae",
}

cat_means = {}   # ms
cat_pcts  = {}   # %
for cat_name, prefixes in CATEGORIES.items():
    total_s = 0.0
    for l, s in summary.items():
        if any(l.startswith(pfx) for pfx in prefixes):
            total_s += s["total"]
    mean_ms = 1e3 * total_s / n_meas if n_meas > 0 else 0.0
    pct = 100.0 * total_s / sum(summary.get("TOTAL step", {}).get("total", 1.0) for _ in [1])
    outer_total_s = summary.get("TOTAL step", {}).get("total", 1.0)
    pct = 100.0 * total_s / outer_total_s if outer_total_s > 0 else 0.0
    if mean_ms > 0:
        cat_means[cat_name] = mean_ms
        cat_pcts[cat_name]  = pct

# Print grouped summary
print("\n  ── Grouped categories ──")
for cn in CATEGORIES:
    if cn in cat_means:
        print(f"    {cn:<28s}  {cat_means[cn]:8.2f} ms  ({cat_pcts[cn]:5.1f}%)")

# ── Figure 1: Horizontal bar chart ──────────────────────────────────
cat_names_sorted = sorted(cat_means.keys(), key=lambda k: cat_means[k])
fig, ax = plt.subplots(figsize=(5.5, 3.0))
bars = ax.barh(
    range(len(cat_names_sorted)),
    [cat_means[c] for c in cat_names_sorted],
    color=[CAT_COLOURS.get(c, "#90a4ae") for c in cat_names_sorted],
    edgecolor="white", linewidth=0.6, height=0.7,
)
ax.set_yticks(range(len(cat_names_sorted)))
ax.set_yticklabels(cat_names_sorted)
ax.set_xlabel("Time per step (ms)")
ax.set_title(f"Cost breakdown – {args.Nx}×{args.Ny}×{args.Nz} "
             f"({grid_N:,} cells, {'GPU' if USE_CUDA else 'CPU'})")

# Annotate bars
for bar, cn in zip(bars, cat_names_sorted):
    w = bar.get_width()
    pct = cat_pcts[cn]
    ax.text(w + 0.02 * max(cat_means.values()), bar.get_y() + bar.get_height() / 2,
            f"{w:.1f} ms ({pct:.0f}%)", va="center", fontsize=8)

ax.set_xlim(0, max(cat_means.values()) * 1.35)
fig.tight_layout()
bar_path = os.path.join(args.out_dir, f"cost_barh_{tag}.pdf")
fig.savefig(bar_path)
fig.savefig(bar_path.replace(".pdf", ".png"))
print(f"  Figure saved → {bar_path}")
plt.close(fig)


# ── Figure 2: Pie chart ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5, 4))

pie_names = list(cat_pcts.keys())
pie_vals  = [cat_pcts[c] for c in pie_names]
pie_cols  = [CAT_COLOURS.get(c, "#90a4ae") for c in pie_names]

# Clean labels (remove \n for pie)
pie_display = [c.replace("\n", " ") for c in pie_names]

wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_display, autopct="%1.1f%%",
    startangle=140, pctdistance=0.78, textprops={"fontsize": 8},
    colors=pie_cols,
    wedgeprops=dict(edgecolor="white", linewidth=1.0),
)
for at in autotexts:
    at.set_fontsize(7)
ax2.set_title(f"Step cost distribution – {args.Nx}×{args.Ny}×{args.Nz}")
fig2.tight_layout()
pie_path = os.path.join(args.out_dir, f"cost_pie_{tag}.pdf")
fig2.savefig(pie_path)
fig2.savefig(pie_path.replace(".pdf", ".png"))
print(f"  Figure saved → {pie_path}")
plt.close(fig2)


# ── Figure 3: Detailed breakdown (all sub-timers) ───────────────────
detail_labels = [l for l in summary if l != "TOTAL step"]
detail_labels.sort(key=lambda k: -summary[k]["pct"])

if detail_labels:
    fig3, ax3 = plt.subplots(figsize=(6, max(3.5, 0.35 * len(detail_labels))))
    detail_pcts  = [summary[l]["pct"] for l in detail_labels]
    detail_means = [1e3 * summary[l]["mean"] for l in detail_labels]

    def _colour(label):
        if "SDF" in label or "sdf" in label:        return "#26a69a"
        if "mu_funcs" in label:                       return "#66bb6a"
        if "normals" in label:                        return "#a5d6a7"
        if "advection" in label:                      return "#42a5f5"
        if "BDIM" in label:                           return "#29b6f6"
        if "Poisson" in label or "poisson" in label:  return "#ef5350"
        if "Jacobi" in label or "V-cycle" in label:   return "#ef9a9a"
        if "divergence" in label:                     return "#ab47bc"
        if "gradient" in label:                       return "#ce93d8"
        if "projection" in label:                     return "#e1bee7"
        if "forces" in label:                         return "#ffa726"
        if "BCs" in label:                            return "#78909c"
        if "apply" in label or "FARMS" in label:      return "#5c6bc0"
        if "plotting" in label:                       return "#90a4ae"
        return "#bdbdbd"

    dc = [_colour(l) for l in detail_labels]
    bars3 = ax3.barh(range(len(detail_labels)), detail_pcts, color=dc,
                     edgecolor="white", linewidth=0.4, height=0.7)
    ax3.set_yticks(range(len(detail_labels)))
    ax3.set_yticklabels(detail_labels, fontsize=7)
    ax3.set_xlabel("% of total step time")
    ax3.set_title(f"Detailed breakdown – {args.Nx}×{args.Ny}×{args.Nz}")
    ax3.invert_yaxis()

    for bar, m, p in zip(bars3, detail_means, detail_pcts):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{m:.2f} ms ({p:.1f}%)", va="center", fontsize=6.5)
    ax3.set_xlim(0, max(detail_pcts) * 1.4)
    fig3.tight_layout()
    detail_path = os.path.join(args.out_dir, f"cost_detailed_{tag}.pdf")
    fig3.savefig(detail_path)
    fig3.savefig(detail_path.replace(".pdf", ".png"))
    print(f"  Figure saved → {detail_path}")
    plt.close(fig3)


print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Run multi-grid scaling analysis:                                  ║
║                                                                    ║
║  python run_cost_analysis.py --Nx  64 --Ny  16 --Nz  16           ║
║  python run_cost_analysis.py --Nx 128 --Ny  32 --Nz  32           ║
║  python run_cost_analysis.py --Nx 256 --Ny  64 --Nz  64           ║
║  python run_cost_analysis.py --Nx 512 --Ny 128 --Nz 128           ║
║                                                                    ║
║  Then run:  python plot_scaling.py                                 ║
║  to produce the multi-resolution comparison figure.                ║
╚══════════════════════════════════════════════════════════════════════╝
""")
