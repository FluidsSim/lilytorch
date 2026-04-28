"""Standalone microbenchmark: forces-method comparison.

Benchmarks the cost of the per-step `(SDF update + forces)` pipeline for
the *new* cached-cc-SDF method introduced by the recent refactor of
`bdim_forces_3d_multi`, against three of the most relevant alternatives
that were tested in the parent ``cost_analysis_free_swimming_3d/``
campaign (see ``run_cost_analysis.py``'s ``--streaming_forces_3d``,
``--force_narrow_batch``, ``--force_shared_union`` flags).

Why this script
---------------
The new method **merges** the SDF computation and the forces kernel
through a per-body cell-centred SDF cache (`sparse_cc_flat`).  That
shifts the cost of the per-cell trilinear sample from the forces stage
to the SDF stage, so a like-for-like comparison must time **both**
stages.  The user also raised a specific concern: with the streaming
path's per-body AABB cropping, what happens when the body fills a
substantial fraction of the domain?  The script answers both questions:

* it splits the cost into ``SDF stage`` / ``Forces stage`` / ``Total``;
* it sweeps the per-body AABB fraction so the trade-off
  ``cropping benefit ↔ cropping overhead`` is visible.

The full ``run_cost_analysis.py`` benchmark needs FARMS / MuJoCo /
scikit-fmm / Open3D and a CUDA-synced timer infrastructure that is only
useful in production.  That set of dependencies is too heavy for many
sandboxed environments (and pulls in ~3 GiB of CUDA libraries even on a
CPU box).  This microbenchmark avoids all of that: it builds synthetic
sphere bodies directly, calls the lilytorch kernels in isolation, and
runs cleanly on **CPU only**.

Methods compared
----------------
1. **kernel_cached** — current production path (after the recent PR).
   ``streaming_sdf_min_3d_multi`` writes `sparse_cc_flat`; the new
   ``bdim_forces_3d_multi`` reads it.

2. **kernel_resample** — pre-PR baseline.  Same SDF kernel; forces
   re-samples the body SDF on the fly via the legacy
   ``bdim_forces_3d_multi_legacy_resample`` op (recovered from git
   history and registered side-by-side for benchmarking).

3. **pytorch_narrow_batch** — packs each body's AABB sub-block into a
   ``(B, D, D, D)`` tensor and runs a pure-PyTorch reduction.  Mirrors
   the ``force_narrow_batch`` solver path.

4. **pytorch_full_grid** — runs the same reduction over the full
   ``(Nx, Ny, Nz)`` fluid grid (no AABB cropping, no per-body packing).
   Mirrors the no-narrow-band baseline.  This is the case the user
   flagged: when the body covers most of the domain, the cropping
   overhead in (1)-(3) may not pay off vs. just running once over the
   whole grid.

Usage
-----
::

    python bench_forces_methods.py                       # default sweep
    python bench_forces_methods.py --grid 96 80 72       # custom grid
    python bench_forces_methods.py --reps 20             # more averaging

CSV + PDF/PNG land in ``figures/forces_methods/``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Make the repo importable when this script is run directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import lilytorch.src.kernels  # noqa: F401 — registers the ops
from lilytorch.src.kernels import (
    streaming_sdf_min_3d_multi,
    bdim_forces_3d_multi,
)


# ────────────────────────────────────────────────────────────────────
#  Synthetic input setup
# ────────────────────────────────────────────────────────────────────

def _identity_meta(bx, by, bz):
    """Linear-axes meta tuple expected by the streaming SDF kernel."""
    return [
        float(bx[0]), float(by[0]), float(bz[0]),
        float(bx[-1]), float(by[-1]), float(bz[-1]),
        1.0/float(bx[1]-bx[0]),
        1.0/float(by[1]-by[0]),
        1.0/float(bz[1]-bz[0]),
        1.0/float((bx[1]-bx[0])*(by[1]-by[0])*(bz[1]-bz[0])),
    ]


def _make_sphere_body(seed, *, body_half_extent, M, dtype, device):
    """Build a sphere SDF on an (M,M,M) body grid with random orientation."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    half = float(body_half_extent)
    bx = torch.linspace(-half, half, M, dtype=dtype, device=device)
    by = bx.clone()
    bz = bx.clone()
    Bx, By, Bz = torch.meshgrid(bx, by, bz, indexing="ij")
    radius = 0.5 * half
    F = (torch.sqrt(Bx**2 + By**2 + Bz**2) - radius).contiguous()

    q = torch.randn(4, generator=g)
    q = (q / q.norm()).tolist()
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*(y*y+z*z),     2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z),     2*(y*z-x*w)],
        [2*(x*z-y*w),         2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ], dtype=dtype, device=device)
    R_T = R.T.contiguous().flatten().tolist()
    bp = (torch.rand(3, generator=g) * 0.2 - 0.1).tolist()
    cm = bp[:]
    lv = (torch.rand(3, generator=g) - 0.5).tolist()
    av = (torch.rand(3, generator=g) - 0.5).tolist()
    return dict(F=F, bx=bx, by=by, bz=bz, R_T=R_T, bp=bp, cm=cm,
                lv=lv, av=av, Mx=M, My=M, Mz=M, radius=radius)


def build_inputs(*, Nx, Ny, Nz, B, body_aabb_frac, dtype=torch.float64,
                 device="cpu", seed=0):
    """Build a multi-body benchmark scene with target per-body AABB fraction.

    ``body_aabb_frac`` is the fraction of each axis covered by every
    body's AABB (so total *fluid coverage* is roughly ``B *
    body_aabb_frac**3`` ignoring overlap, but more importantly the
    union AABB then covers a substantial slab of the domain).
    """
    h = 1.0 / max(Nx, Ny, Nz)
    gx = torch.arange(Nx, dtype=dtype, device=device) * h
    gy = torch.arange(Ny, dtype=dtype, device=device) * h
    gz = torch.arange(Nz, dtype=dtype, device=device) * h

    # Per-body AABB extent (cells)
    Ai = max(8, int(round(Nx * body_aabb_frac)))
    Aj = max(8, int(round(Ny * body_aabb_frac)))
    Ak = max(8, int(round(Nz * body_aabb_frac)))
    Ai = min(Ai, Nx - 2)
    Aj = min(Aj, Ny - 2)
    Ak = min(Ak, Nz - 2)

    # Spread bodies across the domain (deterministic stride).
    aabbs = []
    rng = np.random.default_rng(seed)
    for b in range(B):
        i0 = int(rng.integers(1, max(2, Nx - Ai - 1)))
        j0 = int(rng.integers(1, max(2, Ny - Aj - 1)))
        k0 = int(rng.integers(1, max(2, Nz - Ak - 1)))
        aabbs.append((i0, i0 + Ai, j0, j0 + Aj, k0, k0 + Ak))

    # Body grid: pick M ≈ AABB extent so the SDF is sampled at fluid res
    M = max(16, max(Ai, Aj, Ak) // 2)
    body_half = 0.6 * (Ai * h)
    bodies = [_make_sphere_body(seed=seed*1000 + b, body_half_extent=body_half,
                                M=M, dtype=dtype, device=device)
              for b in range(B)]

    # Pack flat tensors
    F_chunks  = [bd["F"].flatten() for bd in bodies]
    bx_chunks = [bd["bx"] for bd in bodies]
    by_chunks = [bd["by"] for bd in bodies]
    bz_chunks = [bd["bz"] for bd in bodies]
    F_off  = [0]; bx_off = [0]; by_off = [0]; bz_off = [0]; cell_off = [0]
    shapes = []; metas = []; kin = []; lo = []; dim_ = []; max_vol = 0
    for bd, ab in zip(bodies, aabbs):
        F_off.append(F_off[-1]   + bd["F"].numel())
        bx_off.append(bx_off[-1] + bd["bx"].numel())
        by_off.append(by_off[-1] + bd["by"].numel())
        bz_off.append(bz_off[-1] + bd["bz"].numel())
        shapes.append([bd["Mx"], bd["My"], bd["Mz"]])
        metas.append(_identity_meta(bd["bx"], bd["by"], bd["bz"]))
        kin.append(bd["R_T"] + bd["bp"] + bd["cm"] + bd["lv"] + bd["av"])
        i0, i1, j0, j1, k0, k1 = ab
        lo.append([i0, j0, k0])
        dim_.append([i1-i0, j1-j0, k1-k0])
        vol = (i1-i0)*(j1-j0)*(k1-k0)
        cell_off.append(cell_off[-1] + vol)
        max_vol = max(max_vol, vol)

    inputs = dict(
        Nx=Nx, Ny=Ny, Nz=Nz, B=B, h=h,
        gx=gx, gy=gy, gz=gz,
        F_flat  = torch.cat(F_chunks).contiguous(),
        bx_flat = torch.cat(bx_chunks).contiguous(),
        by_flat = torch.cat(by_chunks).contiguous(),
        bz_flat = torch.cat(bz_chunks).contiguous(),
        F_off_t  = torch.tensor(F_off,  dtype=torch.int64, device=device),
        bx_off_t = torch.tensor(bx_off, dtype=torch.int64, device=device),
        by_off_t = torch.tensor(by_off, dtype=torch.int64, device=device),
        bz_off_t = torch.tensor(bz_off, dtype=torch.int64, device=device),
        cell_off_t = torch.tensor(cell_off, dtype=torch.int64, device=device),
        shapes_t = torch.tensor(shapes, dtype=torch.int64, device=device),
        metas_t  = torch.tensor(metas,  dtype=dtype,        device=device),
        kin_t    = torch.tensor(kin,    dtype=dtype,        device=device),
        lo_t     = torch.tensor(lo,     dtype=torch.int64, device=device),
        dim_t    = torch.tensor(dim_,   dtype=torch.int64, device=device),
        cell_off_list=cell_off,
        aabb_extent=(Ai, Aj, Ak),
        max_vol=max_vol,
        eps_body=1.5*h, eps_solver=2.0*h, h3=h**3,
    )
    # Persistent stress / pforce buffers (full grid)
    inputs["xs"] = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
    inputs["ys"] = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
    inputs["zs"] = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
    inputs["px"] = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
    inputs["py"] = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
    inputs["pz"] = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
    return inputs


# ────────────────────────────────────────────────────────────────────
#  Stage 1: SDF update (shared between all methods)
# ────────────────────────────────────────────────────────────────────

def run_sdf_stage(inp, *, want_sparse_cc):
    """Run streaming_sdf_min_3d_multi.

    All four methods need the union-min SDF.  Methods 1-2 also need the
    `sparse_cc_flat` cache.  Methods 3-4 don't read the cache, but the
    SDF kernel always writes it (cost is identical).  We allocate the
    cache for all methods so the comparison is fair: this models how
    the production solver runs the SDF kernel exactly once per step
    regardless of which forces variant comes after.
    """
    Nx, Ny, Nz = inp["Nx"], inp["Ny"], inp["Nz"]
    dtype = inp["F_flat"].dtype
    device = inp["F_flat"].device
    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_w  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    bW = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    sparse_cc = torch.zeros(inp["cell_off_list"][-1], dtype=dtype, device=device)

    streaming_sdf_min_3d_multi(
        inp["F_flat"], inp["F_off_t"],
        inp["bx_flat"], inp["bx_off_t"],
        inp["by_flat"], inp["by_off_t"],
        inp["bz_flat"], inp["bz_off_t"],
        inp["shapes_t"], inp["metas_t"], inp["kin_t"],
        inp["lo_t"], inp["dim_t"], inp["cell_off_t"],
        inp["gx"], inp["gy"], inp["gz"], inp["h"], inp["max_vol"],
        sdf_cc, sdf_u, sdf_v, sdf_w,
        bU, bV, bW,
        sparse_cc,
    )
    return sdf_cc, sparse_cc


# ────────────────────────────────────────────────────────────────────
#  Stage 2 variants
# ────────────────────────────────────────────────────────────────────

def forces_kernel_cached(inp, sparse_cc):
    """Method 1 — current: kernel reads cached cc-SDF."""
    B = inp["B"]
    out = torch.zeros((B, 12), dtype=torch.float64, device=inp["gx"].device)
    bdim_forces_3d_multi(
        sparse_cc, inp["cell_off_t"],
        inp["kin_t"],
        inp["lo_t"], inp["dim_t"],
        inp["gx"], inp["gy"], inp["gz"],
        0, 0, 0, inp["Ny"], inp["Nz"],
        inp["xs"], inp["ys"], inp["zs"],
        inp["px"], inp["py"], inp["pz"],
        inp["eps_body"], inp["eps_solver"], inp["h3"],
        inp["max_vol"],
        out,
    )
    return out


def forces_kernel_resample(inp):
    """Method 2 — pre-PR: legacy resample-based kernel."""
    B = inp["B"]
    out = torch.zeros((B, 12), dtype=torch.float64, device=inp["gx"].device)
    torch.ops.lilytorch_kernels.bdim_forces_3d_multi_legacy_resample.default(
        inp["F_flat"], inp["F_off_t"],
        inp["bx_flat"], inp["bx_off_t"],
        inp["by_flat"], inp["by_off_t"],
        inp["bz_flat"], inp["bz_off_t"],
        inp["shapes_t"], inp["metas_t"], inp["kin_t"],
        inp["lo_t"], inp["dim_t"],
        inp["gx"], inp["gy"], inp["gz"],
        0, 0, 0, inp["Ny"], inp["Nz"],
        inp["xs"], inp["ys"], inp["zs"],
        inp["px"], inp["py"], inp["pz"],
        inp["eps_body"], inp["eps_solver"], inp["h3"],
        inp["max_vol"],
        out,
    )
    return out


def _delta_band(d, eps_body):
    """Cosine δ-kernel masked to the body band (matches the C++ kernels)."""
    inv_2eps = 0.5 / eps_body
    band = (d > -eps_body) & (d < eps_body)
    return torch.where(band, (1.0 + torch.cos(math.pi * d / eps_body)) * inv_2eps,
                       torch.zeros_like(d))


def forces_pytorch_narrow_batch(inp, sparse_cc):
    """Method 3 — pure-PyTorch narrow-batch reduction over per-body AABB.

    Mirrors solver._compute_forces_3d's `force_narrow_batch` path: pack
    each body's AABB sub-block into a `(B, Ai, Aj, Ak)` tensor and run
    a single batched reduction.  The per-cell SDF comes from the cache
    `sparse_cc_flat` (the kernel SDF stage already wrote it).
    """
    B = inp["B"]
    Ai, Aj, Ak = inp["aabb_extent"]
    dtype = sparse_cc.dtype
    device = sparse_cc.device

    sdf_b = torch.empty((B, Ai, Aj, Ak), dtype=dtype, device=device)
    xs_b = torch.empty_like(sdf_b)
    ys_b = torch.empty_like(sdf_b)
    zs_b = torch.empty_like(sdf_b)
    px_b = torch.empty_like(sdf_b)
    py_b = torch.empty_like(sdf_b)
    pz_b = torch.empty_like(sdf_b)
    arm_x = torch.empty_like(sdf_b)
    arm_y = torch.empty_like(sdf_b)
    arm_z = torch.empty_like(sdf_b)

    cell_off = inp["cell_off_list"]
    lo = inp["lo_t"].tolist()
    kin = inp["kin_t"]
    gx, gy, gz = inp["gx"], inp["gy"], inp["gz"]
    xs, ys, zs = inp["xs"], inp["ys"], inp["zs"]
    px, py, pz = inp["px"], inp["py"], inp["pz"]
    for b in range(B):
        i0, j0, k0 = lo[b]
        sdf_b[b] = sparse_cc[cell_off[b]: cell_off[b] + Ai*Aj*Ak].view(Ai, Aj, Ak)
        sl = (slice(i0, i0+Ai), slice(j0, j0+Aj), slice(k0, k0+Ak))
        xs_b[b] = xs[sl]; ys_b[b] = ys[sl]; zs_b[b] = zs[sl]
        px_b[b] = px[sl]; py_b[b] = py[sl]; pz_b[b] = pz[sl]
        cm_x = float(kin[b, 12]); cm_y = float(kin[b, 13]); cm_z = float(kin[b, 14])
        arm_x[b] = (gx[i0:i0+Ai].view(-1, 1, 1) - cm_x).expand(Ai, Aj, Ak)
        arm_y[b] = (gy[j0:j0+Aj].view(1, -1, 1) - cm_y).expand(Ai, Aj, Ak)
        arm_z[b] = (gz[k0:k0+Ak].view(1, 1, -1) - cm_z).expand(Ai, Aj, Ak)

    delta_visc = _delta_band(sdf_b - inp["eps_solver"], inp["eps_body"])
    delta_pres = _delta_band(sdf_b,                     inp["eps_body"])

    h3 = inp["h3"]
    fv_x = (xs_b * delta_visc); fv_y = (ys_b * delta_visc); fv_z = (zs_b * delta_visc)
    fp_x = (px_b * delta_pres); fp_y = (py_b * delta_pres); fp_z = (pz_b * delta_pres)
    out = torch.zeros((B, 12), dtype=torch.float64, device=device)
    out[:, 0]  = fv_x.sum(dim=(1,2,3)) * h3
    out[:, 1]  = fv_y.sum(dim=(1,2,3)) * h3
    out[:, 2]  = fv_z.sum(dim=(1,2,3)) * h3
    out[:, 3]  = (arm_y*fv_z - arm_z*fv_y).sum(dim=(1,2,3)) * h3
    out[:, 4]  = (arm_z*fv_x - arm_x*fv_z).sum(dim=(1,2,3)) * h3
    out[:, 5]  = (arm_x*fv_y - arm_y*fv_x).sum(dim=(1,2,3)) * h3
    out[:, 6]  = fp_x.sum(dim=(1,2,3)) * h3
    out[:, 7]  = fp_y.sum(dim=(1,2,3)) * h3
    out[:, 8]  = fp_z.sum(dim=(1,2,3)) * h3
    out[:, 9]  = (arm_y*fp_z - arm_z*fp_y).sum(dim=(1,2,3)) * h3
    out[:, 10] = (arm_z*fp_x - arm_x*fp_z).sum(dim=(1,2,3)) * h3
    out[:, 11] = (arm_x*fp_y - arm_y*fp_x).sum(dim=(1,2,3)) * h3
    return out


def forces_pytorch_full_grid(inp, sdf_cc):
    """Method 4 — pure-PyTorch reduction over the FULL fluid grid.

    No AABB cropping: works directly with the union-SDF on `(Nx, Ny,
    Nz)`.  This is the natural baseline when the body fills most of the
    domain and per-body cropping has nothing to skip.  It loses the
    per-body force decomposition (the union SDF doesn't carry body
    identity), so this method only gives total integrated force/torque
    over all bodies.  We still time it as a reference: it's the path a
    naive implementation would take, and the user explicitly flagged it
    as the case to compare against.
    """
    delta_visc = _delta_band(sdf_cc - inp["eps_solver"], inp["eps_body"])
    delta_pres = _delta_band(sdf_cc,                     inp["eps_body"])
    h3 = inp["h3"]
    fv_x = (inp["xs"] * delta_visc); fv_y = (inp["ys"] * delta_visc); fv_z = (inp["zs"] * delta_visc)
    fp_x = (inp["px"] * delta_pres); fp_y = (inp["py"] * delta_pres); fp_z = (inp["pz"] * delta_pres)
    # Use a single body-centred COM at the domain centre for the torque arm.
    gx, gy, gz = inp["gx"], inp["gy"], inp["gz"]
    cmx = float(gx.mean()); cmy = float(gy.mean()); cmz = float(gz.mean())
    Nx, Ny, Nz = inp["Nx"], inp["Ny"], inp["Nz"]
    arm_x = (gx - cmx).view(-1,1,1).expand(Nx, Ny, Nz)
    arm_y = (gy - cmy).view(1,-1,1).expand(Nx, Ny, Nz)
    arm_z = (gz - cmz).view(1,1,-1).expand(Nx, Ny, Nz)
    out = torch.zeros(12, dtype=torch.float64)
    out[0]  = fv_x.sum() * h3; out[1] = fv_y.sum() * h3; out[2] = fv_z.sum() * h3
    out[3]  = (arm_y*fv_z - arm_z*fv_y).sum() * h3
    out[4]  = (arm_z*fv_x - arm_x*fv_z).sum() * h3
    out[5]  = (arm_x*fv_y - arm_y*fv_x).sum() * h3
    out[6]  = fp_x.sum() * h3; out[7] = fp_y.sum() * h3; out[8] = fp_z.sum() * h3
    out[9]  = (arm_y*fp_z - arm_z*fp_y).sum() * h3
    out[10] = (arm_z*fp_x - arm_x*fp_z).sum() * h3
    out[11] = (arm_x*fp_y - arm_y*fp_x).sum() * h3
    return out


# ────────────────────────────────────────────────────────────────────
#  Timing harness
# ────────────────────────────────────────────────────────────────────

METHODS = ["kernel_cached", "kernel_resample",
           "pytorch_narrow_batch", "pytorch_full_grid"]


def run_step(method, inp):
    """Run a full (SDF + forces) step and return (t_sdf, t_forces, t_total)."""
    t0 = time.perf_counter()
    sdf_cc, sparse_cc = run_sdf_stage(inp, want_sparse_cc=True)
    t1 = time.perf_counter()
    if method == "kernel_cached":
        forces_kernel_cached(inp, sparse_cc)
    elif method == "kernel_resample":
        forces_kernel_resample(inp)
    elif method == "pytorch_narrow_batch":
        forces_pytorch_narrow_batch(inp, sparse_cc)
    elif method == "pytorch_full_grid":
        forces_pytorch_full_grid(inp, sdf_cc)
    else:
        raise ValueError(method)
    t2 = time.perf_counter()
    return (t1 - t0, t2 - t1, t2 - t0)


def benchmark(inp, *, method, reps, warmup):
    times = np.empty((reps, 3))
    for _ in range(warmup):
        run_step(method, inp)
    for r in range(reps):
        times[r] = run_step(method, inp)
    return times


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────

def _fmt_ms(t):
    return f"{1e3*t:7.2f} ms"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, nargs=3, default=[80, 64, 56],
                   metavar=("Nx", "Ny", "Nz"),
                   help="Fluid grid (default 80×64×56 ~287k cells)")
    p.add_argument("--bodies", type=int, default=4,
                   help="Number of synthetic sphere bodies (default 4)")
    p.add_argument("--reps", type=int, default=8,
                   help="Timed repetitions per setting (default 8)")
    p.add_argument("--warmup", type=int, default=2,
                   help="Untimed warmup repetitions (default 2)")
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.10, 0.30, 0.50, 0.70],
                   help="Per-body AABB linear fractions of the domain "
                        "(default 0.10 0.30 0.50 0.70)")
    p.add_argument("--out", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "figures", "forces_methods"),
                   help="Output directory for CSV / figures")
    p.add_argument("--threads", type=int, default=None,
                   help="Cap torch CPU threads (default: torch default)")
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    Nx, Ny, Nz = args.grid
    os.makedirs(args.out, exist_ok=True)

    print("=" * 88)
    print(f"  forces-method microbench  |  grid {Nx}×{Ny}×{Nz} ({Nx*Ny*Nz:,} cells)"
          f"  |  B={args.bodies}  |  threads={torch.get_num_threads()}")
    print("=" * 88)

    rows = []  # CSV rows
    summary = {}    # for plotting
    for frac in args.fractions:
        inp = build_inputs(Nx=Nx, Ny=Ny, Nz=Nz, B=args.bodies,
                           body_aabb_frac=frac, seed=42)
        Ai, Aj, Ak = inp["aabb_extent"]
        per_body_frac = (Ai*Aj*Ak) / (Nx*Ny*Nz)
        union_frac_est = min(1.0, args.bodies * per_body_frac)
        print(f"\n  body_aabb_frac = {frac:.2f}  →  AABB {Ai}×{Aj}×{Ak} "
              f"(per-body {100*per_body_frac:5.1f}% of cells, "
              f"union ≲ {100*union_frac_est:5.1f}%)")
        print(f"  {'method':<22}  {'SDF':>10}  {'Forces':>10}  {'Total':>10}  "
              f"{'Forces/Total':>13}")
        method_results = {}
        for m in METHODS:
            tt = benchmark(inp, method=m, reps=args.reps, warmup=args.warmup)
            med = np.median(tt, axis=0)
            std = np.std(tt, axis=0)
            method_results[m] = (med, std)
            t_sdf, t_for, t_tot = med
            print(f"  {m:<22}  {_fmt_ms(t_sdf)}  {_fmt_ms(t_for)}  "
                  f"{_fmt_ms(t_tot)}  {100*t_for/t_tot:11.1f} %")
            rows.append({
                "body_aabb_frac": frac,
                "method": m,
                "Nx": Nx, "Ny": Ny, "Nz": Nz, "B": args.bodies,
                "per_body_frac": per_body_frac,
                "sdf_ms_med":    1e3*t_sdf,
                "forces_ms_med": 1e3*t_for,
                "total_ms_med":  1e3*t_tot,
                "sdf_ms_std":    1e3*std[0],
                "forces_ms_std": 1e3*std[1],
                "total_ms_std":  1e3*std[2],
            })
        summary[frac] = method_results

    # ── CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.out, f"bench_forces_{Nx}x{Ny}x{Nz}.csv")
    with open(csv_path, "w") as f:
        if rows:
            cols = list(rows[0].keys())
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n  CSV → {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable ({e}); skipping plot")
        return

    fractions = list(summary.keys())
    width = 0.20
    x = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"kernel_cached":        "#1f77b4",
              "kernel_resample":      "#ff7f0e",
              "pytorch_narrow_batch": "#2ca02c",
              "pytorch_full_grid":    "#d62728"}
    for i, m in enumerate(METHODS):
        sdf_t = [summary[f][m][0][0]*1e3 for f in fractions]
        for_t = [summary[f][m][0][1]*1e3 for f in fractions]
        tot_t = [summary[f][m][0][2]*1e3 for f in fractions]
        # Stacked: SDF (bottom) + Forces (top)
        bx = x + (i - 1.5) * width
        ax.bar(bx, sdf_t,  width, color=colors[m], alpha=0.55,
               label=(f"{m} – SDF" if i == 0 else None), edgecolor="white")
        ax.bar(bx, for_t,  width, bottom=sdf_t, color=colors[m], alpha=1.0,
               label=m, edgecolor="white")
        for xi, ti in zip(bx, tot_t):
            ax.text(xi, ti, f"{ti:.1f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.2f}" for f in fractions])
    ax.set_xlabel("Per-body AABB fraction (linear)")
    ax.set_ylabel("Time per step (ms)  —  median over reps")
    ax.set_title(f"Forces methods — grid {Nx}×{Ny}×{Nz}, B={args.bodies}\n"
                 f"(stacked: SDF stage + Forces stage)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plot_path_png = os.path.join(args.out, f"bench_forces_{Nx}x{Ny}x{Nz}.png")
    plot_path_pdf = os.path.join(args.out, f"bench_forces_{Nx}x{Ny}x{Nz}.pdf")
    fig.tight_layout()
    fig.savefig(plot_path_png, dpi=150)
    fig.savefig(plot_path_pdf)
    plt.close(fig)
    print(f"  PNG → {plot_path_png}")
    print(f"  PDF → {plot_path_pdf}")


if __name__ == "__main__":
    main()
