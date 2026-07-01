"""Peak-GPU-memory experiment for the advection flux step.

The native `advect_flux_add` CUDA kernel's stated rationale (advection_flux.cu
header) is *memory*: it accumulates ``rhs += dt_dh·(F_left − F_right)`` directly
in registers, with **no intermediate F tensor** and none of the B1/B2/flux_in/
flux_lo/flux_hi temporaries the PyTorch ``_flux`` path allocates (~full-grid
each, up to ~9 per step in 3-D).  This bench shows the single-source Warp port
keeps that win.

For one FULL momentum step (every velocity component i × direction d), with the
face-velocity / field views PRECOMPUTED as baseline (so we isolate the flux
kernel itself), it measures peak GPU memory ABOVE baseline for three impls of
the same math:

  * ``python`` — torch ``_flux`` → ``cat`` → ``add_`` (materialises F + temps);
  * ``native`` — the hand-written CUDA ``advect_flux_add`` (writes into rhs);
  * ``warp``   — the single-source ``advect_flux_add_warp`` port.

Two measurements per path (as in bench_bdim_memory.py):
  * **torch Δ** = ``max_memory_allocated`` above baseline;
  * **driver Δ** = ``mem_get_info`` free-memory drop (also sees Warp's allocator).

Expected: ``native ≈ warp ≈ 0`` extra (in-place accumulate, no temporaries),
``python`` scales with the grid — all three numerically equal (printed parity).

Run:  python -m lilytorch.warp_poc.bench_advection_memory --grids 96 128 160
"""
from __future__ import annotations

import argparse
import torch

try:
    import lilytorch.src.kernels  # noqa: F401
    _NATIVE = hasattr(torch.ops.lilytorch_kernels, "advect_flux_add")
except Exception as e:
    print(f"[warn] native unavailable: {e}")
    _NATIVE = False

from lilytorch.warp_poc.warp_advection import advect_flux_add_warp
from lilytorch.src.advection import (
    _face_vel, _field_for_flux, _flux, _inner, _sl, SCHEMES, _CUDA_SCHEME_IDS,
)

MIB = 1024.0 ** 2
DT_DH, C = 0.123, 0.37


# ─── problem ──────────────────────────────────────────────────────────────────

def build(ndim, N, dev, dtype=torch.float64):
    g = torch.Generator(device="cpu").manual_seed(3)
    shape = (N + 2,) * ndim
    vel = [(torch.rand(shape, generator=g, dtype=dtype) - 0.5).to(dev)
           for _ in range(ndim)]
    rhs = [torch.zeros_like(vel[i][_inner(ndim)]) for i in range(ndim)]
    # PRECOMPUTE the (fv, p, rhs, d) flux tasks → part of the baseline, so we
    # measure only the flux kernel's own transient (the F materialisation).
    tasks = []
    for i in range(ndim):
        for d in range(ndim):
            fv = _face_vel(vel, i, d, ndim).contiguous()
            p = _field_for_flux(vel[i], d, ndim)
            tasks.append((fv, p, rhs[i], d))
    return dict(ndim=ndim, vel=vel, rhs=rhs, tasks=tasks)


# ─── the three implementations (identical math) ───────────────────────────────

def run_python(P, scheme_name):
    scheme = SCHEMES[scheme_name]
    ndim = P["ndim"]
    for rhs in P["rhs"]:
        rhs.zero_()
    for fv, p, rhs, d in P["tasks"]:
        F = _flux(scheme, fv, p, d)                       # materialises F (+temps)
        F_diff = (F[_sl(ndim, d, slice(None, -1))]
                  - F[_sl(ndim, d, slice(1, None))])
        rhs.add_(F_diff, alpha=DT_DH)
        del F, F_diff


def run_native(P, scheme_name):
    sid = _CUDA_SCHEME_IDS[scheme_name]
    for rhs in P["rhs"]:
        rhs.zero_()
    for fv, p, rhs, d in P["tasks"]:
        torch.ops.lilytorch_kernels.advect_flux_add(fv, p, rhs, DT_DH, C, sid, d)


def run_warp(P, scheme_name):
    sid = _CUDA_SCHEME_IDS[scheme_name]
    for rhs in P["rhs"]:
        rhs.zero_()
    for fv, p, rhs, d in P["tasks"]:
        advect_flux_add_warp(fv, p, rhs, DT_DH, C, sid, d)


# ─── measurement harness (mirrors bench_bdim_memory.peak_delta) ───────────────

def peak_delta(fn, P, scheme_name):
    fn(P, scheme_name)                       # warmup (compile / prime caches)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated()
    free0, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()

    fn(P, scheme_name)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() - base_alloc
    free1, _ = torch.cuda.mem_get_info()
    driver = free0 - free1
    result = [r.clone() for r in P["rhs"]]
    return peak / MIB, driver / MIB, result


def parity(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[96, 128, 160])
    ap.add_argument("--scheme", default="quick", choices=sorted(_CUDA_SCHEME_IDS))
    ap.add_argument("--ndim", type=int, default=3, choices=(2, 3))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required."); return
    dev = "cuda:0"

    print(f"\nAdvection flux peak-memory experiment ({args.ndim}-D, {args.scheme}, "
          f"float64, {dev})")
    print("'torch Δ' = max_memory_allocated above baseline (flux views precomputed);")
    print("'drv Δ' = driver free-memory drop (sees Warp's allocator).\n")
    hdr = (f"{'grid':>7} {'rhs MiB':>8} | "
           f"{'python torchΔ':>13} {'drvΔ':>7} | "
           f"{'native torchΔ':>13} {'drvΔ':>7} | "
           f"{'warp torchΔ':>12} {'drvΔ':>7} | {'parity':>9}")
    print(hdr); print("-" * len(hdr))

    for N in args.grids:
        P = build(args.ndim, N, dev)
        rhs_mib = sum(r.numel() for r in P["rhs"]) * 8 / MIB

        pp, pd, rp = peak_delta(run_python, P, args.scheme)
        if _NATIVE:
            npk, nd, rn = peak_delta(run_native, P, args.scheme)
        else:
            npk = nd = float("nan"); rn = rp
        wpk, wd, rw = peak_delta(run_warp, P, args.scheme)

        par = max(parity(rn, rw), parity(rn, rp))
        grid = f"{N}{'³' if args.ndim==3 else '²'}"
        print(f"{grid:>7} {rhs_mib:>8.1f} | "
              f"{pp:>13.1f} {pd:>7.1f} | "
              f"{npk:>13.1f} {nd:>7.1f} | "
              f"{wpk:>12.1f} {wd:>7.1f} | {par:>9.1e}")
        del P, rp, rn, rw
        torch.cuda.empty_cache()

    print("\nReading: 'native'/'warp' torchΔ ≈ 0 → flux accumulated in registers, "
          "no intermediate F.\n'python' torchΔ scales with the grid → the F + "
          "B1/B2/flux_* temporaries the fused kernel removes.")


if __name__ == "__main__":
    main()
