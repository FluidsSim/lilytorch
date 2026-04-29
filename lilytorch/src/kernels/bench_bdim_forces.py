"""Self-contained micro-benchmark for ``bdim_forces_3d_multi`` (CPU).

Mirrors the geometry of the ``nbforces_opt`` low-N regime profiled in
``run_scaling_conditions_pipeline.py`` (the case where this op accounts
for ~22% of the step cost): a handful of bodies whose AABBs cover only
a thin band of the union sub-block, with most cells far from the body
surface (i.e. δ_ε(d) = 0 there).

Reports median wall-clock per call.  Run with:

    python -m lilytorch.src.kernels.bench_bdim_forces
"""
from __future__ import annotations

import os
import time
import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers lilytorch_kernels namespace
from lilytorch.src.kernels import bdim_forces_3d_multi


def _build_inputs(B=4, Ai=48, Aj=40, Ak=32, *, dtype=torch.float64, device="cpu"):
    """Build a (B, Ai, Aj, Ak) AABB stack with a thin signed-distance band.

    Most cells have |sdf| > eps_body so they fall outside the smoothed
    delta band — exactly the situation that makes the early-skip
    optimisation effective.
    """
    torch.manual_seed(0)
    h = 0.025
    Si = max(64, B * 8 + Ai)
    Sj, Sk = max(64, Aj + 16), max(64, Ak + 16)

    gx = torch.arange(Si, dtype=dtype, device=device) * h - 1.0
    gy = torch.arange(Sj, dtype=dtype, device=device) * h - 0.8
    gz = torch.arange(Sk, dtype=dtype, device=device) * h - 0.6

    eps_body = 1.5 * h
    eps_solver = 2.0 * h
    h3 = h * h * h

    # Random stress / pforce, full union sub-block (Si, Sj, Sk).
    xs = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    ys = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    zs = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    px = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    py = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    pz = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)

    # Per-body AABBs (non-overlapping in i-axis) + sparse cc-SDF that
    # represents a sphere just inside each AABB.  Most cells of the AABB
    # have |sdf| > eps_body (= are far inside or far outside the body),
    # mimicking the nbforces_opt regime.
    cell_off = [0]
    lo = []
    dim = []
    sdf_chunks = []
    for b in range(B):
        i0 = 4 + b * 8
        j0 = 6
        k0 = 6
        lo.append([i0, j0, k0])
        dim.append([Ai, Aj, Ak])
        # Build a sphere SDF whose radius takes ~half the AABB extent so
        # the iso-surface lies inside the AABB and the band only
        # intersects ~10% of cells.
        ii = torch.arange(Ai, dtype=dtype, device=device)
        jj = torch.arange(Aj, dtype=dtype, device=device)
        kk = torch.arange(Ak, dtype=dtype, device=device)
        I, J, K = torch.meshgrid(ii, jj, kk, indexing="ij")
        cx, cy, cz = (Ai - 1) * 0.5, (Aj - 1) * 0.5, (Ak - 1) * 0.5
        r = 0.30 * min(Ai, Aj, Ak)
        sdf = (torch.sqrt((I - cx) ** 2 + (J - cy) ** 2 + (K - cz) ** 2) - r) * h
        sdf_chunks.append(sdf.flatten())
        cell_off.append(cell_off[-1] + Ai * Aj * Ak)

    sparse_cc_flat = torch.cat(sdf_chunks).contiguous()
    cell_off_t = torch.tensor(cell_off, dtype=torch.int64, device=device)
    lo_t = torch.tensor(lo, dtype=torch.int64, device=device)
    dim_t = torch.tensor(dim, dtype=torch.int64, device=device)

    # Random kinematics — only kin[:, 12:15] (COM) is consumed by the op.
    kin = torch.randn(B, 21, dtype=dtype, device=device)
    out = torch.zeros((B, 12), dtype=torch.float64, device=device)

    return dict(
        sparse_cc_flat=sparse_cc_flat, cell_offsets=cell_off_t,
        kin=kin, aabb_lo=lo_t, aabb_dim=dim_t,
        gx=gx, gy=gy, gz=gz,
        u_i0=0, u_j0=0, u_k0=0, Sj=Sj, Sk=Sk,
        xs=xs, ys=ys, zs=zs, px=px, py=py, pz=pz,
        eps_body=eps_body, eps_solver=eps_solver, h3=h3,
        max_vol=Ai * Aj * Ak, out=out, B=B,
    )


def _bench_once(args, n_warmup=10, n_iter=200):
    out = args["out"]
    # Warm-up
    for _ in range(n_warmup):
        out.zero_()
        bdim_forces_3d_multi(
            args["sparse_cc_flat"], args["cell_offsets"], args["kin"],
            args["aabb_lo"], args["aabb_dim"],
            args["gx"], args["gy"], args["gz"],
            args["u_i0"], args["u_j0"], args["u_k0"], args["Sj"], args["Sk"],
            args["xs"], args["ys"], args["zs"],
            args["px"], args["py"], args["pz"],
            args["eps_body"], args["eps_solver"], args["h3"],
            args["max_vol"], out,
        )
    samples = []
    for _ in range(n_iter):
        out.zero_()
        t0 = time.perf_counter()
        bdim_forces_3d_multi(
            args["sparse_cc_flat"], args["cell_offsets"], args["kin"],
            args["aabb_lo"], args["aabb_dim"],
            args["gx"], args["gy"], args["gz"],
            args["u_i0"], args["u_j0"], args["u_k0"], args["Sj"], args["Sk"],
            args["xs"], args["ys"], args["zs"],
            args["px"], args["py"], args["pz"],
            args["eps_body"], args["eps_solver"], args["h3"],
            args["max_vol"], out,
        )
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2] * 1e3,
        "min_ms":    samples[0] * 1e3,
        "p90_ms":    samples[int(0.9 * len(samples))] * 1e3,
        "out_norm":  float(out.abs().max().item()),
    }


def main():
    # Some environments default torch to a single thread; honour
    # OMP_NUM_THREADS explicitly so the benchmark exercises the OpenMP
    # parallel region.
    nthreads_env = os.environ.get("OMP_NUM_THREADS")
    if nthreads_env is not None:
        torch.set_num_threads(int(nthreads_env))
    nthreads = torch.get_num_threads()
    print(f"torch.get_num_threads() = {nthreads}"
          f"   (OMP_NUM_THREADS={nthreads_env})")
    print()

    # Three shapes that bracket the low-to-mid grid regime in
    # run_scaling_conditions_pipeline.py.
    cases = [
        ("low-N    (B=1, AABB 32x24x16)",  dict(B=1, Ai=32, Aj=24, Ak=16)),
        ("low-N    (B=4, AABB 48x40x32)",  dict(B=4, Ai=48, Aj=40, Ak=32)),
        ("mid-N    (B=4, AABB 64x56x48)",  dict(B=4, Ai=64, Aj=56, Ak=48)),
    ]
    for label, kw in cases:
        args = _build_inputs(**kw)
        stats = _bench_once(args)
        total_cells = kw["B"] * kw["Ai"] * kw["Aj"] * kw["Ak"]
        rate = total_cells / (stats["median_ms"] * 1e-3) * 1e-9
        print(f"  {label}")
        print(f"      median = {stats['median_ms']:8.4f} ms"
              f"   min = {stats['min_ms']:8.4f} ms"
              f"   p90 = {stats['p90_ms']:8.4f} ms")
        print(f"      throughput ≈ {rate:6.3f} G cell/s"
              f"   (out norm = {stats['out_norm']:.3e})")
        print()

    # Edge case: AABB entirely outside the band (all sdf far from 0).
    # The optimised kernel should hit the early-skip path on every cell
    # and run almost-for-free; the baseline would still spend the full
    # 12-FMA inner loop on every cell.
    args_skip = _build_inputs(B=4, Ai=48, Aj=40, Ak=32)
    args_skip["sparse_cc_flat"] = (
        torch.full_like(args_skip["sparse_cc_flat"], 100.0)
    )
    stats = _bench_once(args_skip)
    print("  edge: all cells far outside band (sdf = 100h)")
    print(f"      median = {stats['median_ms']:8.4f} ms"
          f"   min = {stats['min_ms']:8.4f} ms"
          f"   p90 = {stats['p90_ms']:8.4f} ms"
          f"   (out norm = {stats['out_norm']:.3e})")


if __name__ == "__main__":
    main()
