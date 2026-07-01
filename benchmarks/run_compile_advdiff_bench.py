#!/usr/bin/env python3
"""
Benchmark: torch.compile speed-up for adv-diff solve + BDIM meta-equation.

Runs the QUICK advection-diffusion stencil and the BDIM2 meta-equation
kernel on realistic 3-D grids with and without torch.compile, measuring
pure GPU time.

Usage
-----
    source /path/to/venv/bin/activate
    python run_compile_advdiff_bench.py
    python run_compile_advdiff_bench.py --Nx 256 --Ny 64 --Nz 64 --n_steps 100
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from lilytorch.src.advection import AdvDiffSolver
from lilytorch.src import operations as ops

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="torch.compile adv-diff + BDIM benchmark")
parser.add_argument("--Nx", type=int, default=128)
parser.add_argument("--Ny", type=int, default=32)
parser.add_argument("--Nz", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=80,
                    help="Measured steps (after warmup)")
parser.add_argument("--warmup", type=int, default=10,
                    help="Warmup steps (not timed)")
parser.add_argument("--compile_warmup", type=int, default=8,
                    help="Extra warmup for compiled path (tracing overhead)")
parser.add_argument("--method", type=str, default="quick",
                    choices=["quick", "abdquickest", "cubista", "vanLeer", "cds"])
args = parser.parse_args()

assert torch.cuda.is_available(), "This benchmark requires CUDA"
device = torch.device("cuda")
dtype = torch.float32

# ── Grid setup ───────────────────────────────────────────────────────
Nx, Ny, Nz = args.Nx + 2, args.Ny + 2, args.Nz + 2  # +2 ghost cells
h = 0.01
dt = 5e-4
nu = 1e-6

x = torch.arange(Nx, device=device, dtype=dtype) * h
y = torch.arange(Ny, device=device, dtype=dtype) * h
z = torch.arange(Nz, device=device, dtype=dtype) * h

grid_shape = (Nx, Ny, Nz)
ncells = args.Nx * args.Ny * args.Nz

print("=" * 72)
print(f"  torch.compile Benchmark: adv-diff + BDIM meta-equation")
print(f"  Grid:    {args.Nx} x {args.Ny} x {args.Nz}  ({ncells:,} cells)")
print(f"  Method:  {args.method}")
print(f"  Steps:   {args.n_steps} (warmup {args.warmup}, "
      f"compile extra warmup {args.compile_warmup})")
print(f"  GPU:     {torch.cuda.get_device_name(0)}")
print("=" * 72)


# ── Create matching fields ───────────────────────────────────────────
def make_fields():
    """Create realistic-ish velocity fields + BDIM coefficients."""
    u = 0.1 * torch.randn(grid_shape, device=device, dtype=dtype)
    v = 0.05 * torch.randn(grid_shape, device=device, dtype=dtype)
    w = 0.05 * torch.randn(grid_shape, device=device, dtype=dtype)

    # SDF-derived BDIM coefficients (smooth transition)
    sdf = torch.randn(grid_shape, device=device, dtype=dtype) * h * 4
    eps = 2 * h
    mu0 = 0.5 * (1.0 + torch.tanh(0.5 * np.pi * sdf / eps))
    mu1 = 0.5 * (1.0 + torch.cos(np.pi * torch.clamp(sdf / eps, -1, 1)))
    m_m0 = 1.0 - mu0

    # Body velocities
    body_u = 0.02 * torch.ones(grid_shape, device=device, dtype=dtype)
    body_v = torch.zeros(grid_shape, device=device, dtype=dtype)
    body_w = torch.zeros(grid_shape, device=device, dtype=dtype)

    # Normals (unit vectors)
    nx = torch.randn(grid_shape, device=device, dtype=dtype)
    ny = torch.randn(grid_shape, device=device, dtype=dtype)
    nz = torch.randn(grid_shape, device=device, dtype=dtype)
    mag = (nx**2 + ny**2 + nz**2).sqrt().clamp(min=1e-10)
    nx, ny, nz = nx / mag, ny / mag, nz / mag

    return u, v, w, mu0, mu1, m_m0, body_u, body_v, body_w, nx, ny, nz


# ── Build solver ─────────────────────────────────────────────────────
adv = AdvDiffSolver(
    device, dt, x, y, nu,
    BC_type_u=("D", "D", "N", "N", "N", "N"),
    BC_values_u=(0.1, 0.1, 0, 0, 0, 0),
    BC_type_v=("N", "N", "D", "D", "N", "N"),
    BC_values_v=(0, 0, 0, 0, 0, 0),
    method=args.method, z=z,
    BC_type_w=("N", "N", "N", "N", "D", "D"),
    BC_values_w=(0, 0, 0, 0, 0, 0),
)


# ── BDIM meta-equation (standalone, same as FluidSolver._bdim_meta) ──
def bdim_meta(phi, mu0, m_m0, body_vel, mu1, nx, ny, nz, h_val, ndim):
    nd = ops.normal_derivative(phi - body_vel, h_val, ndim, nx, ny, nz)
    return mu0 * phi + m_m0 * body_vel + mu1 * nd


# ── Combined kernel: adv-diff + BDIM (what happens per Heun half-step)
def advdiff_bdim_step(u, v, w, mu0_u, m_m0_u, body_u, mu1_u, nx_u, ny_u, nz_u,
                      mu0_v, m_m0_v, body_v, mu1_v, nx_v, ny_v, nz_v,
                      mu0_w, m_m0_w, body_w, mu1_w, nx_w, ny_w, nz_w,
                      adv_solve, bdim_fn, h_val):
    """One Heun half-step: adv-diff + BDIM for all 3 components."""
    up, vp, wp = adv_solve(u, v, w)
    up = bdim_fn(up, mu0_u, m_m0_u, body_u, mu1_u, nx_u, ny_u, nz_u, h_val, 3)
    vp = bdim_fn(vp, mu0_v, m_m0_v, body_v, mu1_v, nx_v, ny_v, nz_v, h_val, 3)
    wp = bdim_fn(wp, mu0_w, m_m0_w, body_w, mu1_w, nx_w, ny_w, nz_w, h_val, 3)
    return up, vp, wp


# ═══════════════════════════════════════════════════════════════════════
# Benchmark helper
# ═══════════════════════════════════════════════════════════════════════

def bench(label, adv_solve_fn, bdim_fn, n_steps, warmup):
    """Time n_steps of advdiff+BDIM, return per-step ms."""
    u, v, w, mu0, mu1, m_m0, body_u, body_v, body_w, nx, ny, nz = make_fields()

    # Use same coefficients for all 3 components (simplification for bench)
    def do_step():
        return advdiff_bdim_step(
            u, v, w,
            mu0, m_m0, body_u, mu1, nx, ny, nz,  # u-component
            mu0, m_m0, body_v, mu1, nx, ny, nz,  # v-component
            mu0, m_m0, body_w, mu1, nx, ny, nz,  # w-component
            adv_solve_fn, bdim_fn, h,
        )

    # Warmup
    for _ in range(warmup):
        up, vp, wp = do_step()
        u, v, w = up.clone(), vp.clone(), wp.clone()
        adv.set_BCs(u, v, w)
    torch.cuda.synchronize()

    # Timed steps
    times_ms = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        up, vp, wp = do_step()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        u, v, w = up.clone(), vp.clone(), wp.clone()
        adv.set_BCs(u, v, w)

    arr = np.array(times_ms)
    print(f"  {label:30s}  mean={arr.mean():.3f} ms  "
          f"std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")
    return arr


# ═══════════════════════════════════════════════════════════════════════
# A. Benchmark adv-diff solve ONLY
# ═══════════════════════════════════════════════════════════════════════

print("\n── A. Advection-diffusion solve only ──────────────────────")

def bench_adv_only(label, solve_fn, n_steps, warmup):
    u, v, w, *_ = make_fields()
    for _ in range(warmup):
        up, vp, wp = solve_fn(u, v, w)
        u, v, w = up.clone(), vp.clone(), wp.clone()
        adv.set_BCs(u, v, w)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        up, vp, wp = solve_fn(u, v, w)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        u, v, w = up.clone(), vp.clone(), wp.clone()
        adv.set_BCs(u, v, w)

    arr = np.array(times_ms)
    print(f"  {label:30s}  mean={arr.mean():.3f} ms  "
          f"std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")
    return arr


# Baseline: eager
adv_eager = adv.solve  # _solve_convective
t_adv_eager = bench_adv_only("eager", adv_eager, args.n_steps, args.warmup)

# Compiled
adv_compiled = torch.compile(adv._solve_convective, mode="reduce-overhead")
t_adv_compiled = bench_adv_only(
    "torch.compile", adv_compiled,
    args.n_steps, args.warmup + args.compile_warmup,
)

speedup_adv = t_adv_eager.mean() / t_adv_compiled.mean()
print(f"  >>> Speedup: {speedup_adv:.2f}x  "
      f"({t_adv_eager.mean():.3f} -> {t_adv_compiled.mean():.3f} ms)")


# ═══════════════════════════════════════════════════════════════════════
# B. Benchmark BDIM meta-equation ONLY
# ═══════════════════════════════════════════════════════════════════════

print("\n── B. BDIM meta-equation only ─────────────────────────────")

def bench_bdim_only(label, bdim_fn, n_steps, warmup):
    u, v, w, mu0, mu1, m_m0, body_u, body_v, body_w, nx, ny, nz = make_fields()

    def do_bdim():
        up = bdim_fn(u, mu0, m_m0, body_u, mu1, nx, ny, nz, h, 3)
        vp = bdim_fn(v, mu0, m_m0, body_v, mu1, nx, ny, nz, h, 3)
        wp = bdim_fn(w, mu0, m_m0, body_w, mu1, nx, ny, nz, h, 3)
        return up, vp, wp

    for _ in range(warmup):
        do_bdim()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        do_bdim()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    print(f"  {label:30s}  mean={arr.mean():.3f} ms  "
          f"std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")
    return arr


t_bdim_eager = bench_bdim_only("eager", bdim_meta, args.n_steps, args.warmup)

bdim_compiled = torch.compile(bdim_meta, mode="reduce-overhead")
t_bdim_compiled = bench_bdim_only(
    "torch.compile", bdim_compiled,
    args.n_steps, args.warmup + args.compile_warmup,
)

speedup_bdim = t_bdim_eager.mean() / t_bdim_compiled.mean()
print(f"  >>> Speedup: {speedup_bdim:.2f}x  "
      f"({t_bdim_eager.mean():.3f} -> {t_bdim_compiled.mean():.3f} ms)")


# ═══════════════════════════════════════════════════════════════════════
# C. Benchmark combined adv-diff + BDIM (one Heun half-step)
# ═══════════════════════════════════════════════════════════════════════

print("\n── C. Combined adv-diff + BDIM (one Heun half-step) ──────")

# Reset adv.solve to eager for baseline
adv.solve = adv._solve_convective

t_combined_eager = bench(
    "eager (adv+BDIM)",
    adv._solve_convective, bdim_meta,
    args.n_steps, args.warmup,
)

# Compiled both
adv_compiled2 = torch.compile(adv._solve_convective, mode="reduce-overhead")
t_combined_compiled = bench(
    "compiled (adv+BDIM)",
    adv_compiled2, bdim_compiled,
    args.n_steps, args.warmup + args.compile_warmup,
)

speedup_combined = t_combined_eager.mean() / t_combined_compiled.mean()
print(f"  >>> Speedup: {speedup_combined:.2f}x  "
      f"({t_combined_eager.mean():.3f} -> {t_combined_compiled.mean():.3f} ms)")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print(f"  SUMMARY  ({args.Nx}x{args.Ny}x{args.Nz}, {args.method}, "
      f"{torch.cuda.get_device_name(0)})")
print("=" * 72)
print(f"  {'Kernel':30s}  {'Eager (ms)':>12s}  {'Compiled (ms)':>14s}  {'Speedup':>8s}")
print("-" * 72)
print(f"  {'Adv-diff solve':30s}  {t_adv_eager.mean():12.3f}  "
      f"{t_adv_compiled.mean():14.3f}  {speedup_adv:7.2f}x")
print(f"  {'BDIM meta-eq (x3)':30s}  {t_bdim_eager.mean():12.3f}  "
      f"{t_bdim_compiled.mean():14.3f}  {speedup_bdim:7.2f}x")
print(f"  {'Combined (adv+BDIM)':30s}  {t_combined_eager.mean():12.3f}  "
      f"{t_combined_compiled.mean():14.3f}  {speedup_combined:7.2f}x")
print("=" * 72)

# Context: what fraction of total step these represent
# From cost analysis: adv-diff ~10.7%, BDIM ~1.8% = ~12.5% of step
total_step_baseline_ms = t_combined_eager.mean()
saved_ms = t_combined_eager.mean() - t_combined_compiled.mean()
print(f"\n  Per Heun half-step saved: {saved_ms:.3f} ms")
print(f"  Per full Heun step (2 half-steps) saved: {2*saved_ms:.3f} ms")
print(f"  (Poisson dominates at ~70% of step time; this targets the ~12% adv+BDIM)")
print()
