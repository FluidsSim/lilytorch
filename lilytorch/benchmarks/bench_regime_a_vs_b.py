#!/usr/bin/env python3
"""cuda_native_port item 2.4 bench — Regime A (direct) vs Regime B (resolve).

The union-AABB ``_multi`` path was deleted in item 2.4, so it can no longer be
an A/B arm; the authoritative ``_multi`` comparison is the in-sim 0.98x recorded
in the 2.4 gate log.  What this bench prices instead is the **Regime-A
retirement** (``facade.USE_REGIME_B_ONLY``): resolve is now the sole streaming
path, so on DISJOINT scenes -- the only scenes where the direct kernel is legal
-- we need to know what taking resolve instead costs.

Result (see the "Phase 2 CLOSED" log): direct is ~12 us/call cheaper where it is
legal, but *selecting* it required an `_aabbs_are_disjoint` `.item()` sync
costing ~75 us/step of pipeline drain -- the guard is ~6x more expensive than
the thing it guards, and for articulated bodies (salamander, eel) adjacent links
always overlap so the fast path was never taken anyway.

Usage::

    python lilytorch/benchmarks/bench_regime_a_vs_b.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)
# The scene builders + op drivers are the ones the 2.4 gate tests use, so the
# bench and the correctness gate cannot drift apart.
sys.path.insert(0, os.path.join(REPO_ROOT, "lilytorch", "tests"))

from test_per_body_buffers import make_scene, run_direct, run_resolve  # noqa: E402


def timeit(fn, reps=300):
    """Min-of-5 samples of `reps` GPU-synchronised launches (min = least noise)."""
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / reps * 1e3)
    return min(ts)


def main():
    if not torch.cuda.is_available():
        print("no CUDA — nothing to bench")
        return
    print(f"{'scene':22s} {'dtype':6s} {'direct':>14s} {'resolve':>10s} {'ratio':>8s}")
    print("-" * 66)
    for dtype, dname in ((torch.float32, "fp32"), (torch.float64, "fp64")):
        for dim in (2, 3):
            for layout in ("single", "separated", "multilink"):
                sc = make_scene(dim, layout, dtype)
                tr = timeit(lambda sc=sc, d=dtype: run_resolve(sc, d))
                name = f"{dim}d_{layout}"
                if layout == "multilink":
                    # Bodies overlap -> the direct kernel would race. Not a legal arm.
                    print(f"{name:22s} {dname:6s} {'n/a (overlap)':>14s} "
                          f"{tr * 1e3:9.1f}us {'--':>8s}")
                    continue
                td = timeit(lambda sc=sc, d=dtype: run_direct(sc, d))
                print(f"{name:22s} {dname:6s} {td * 1e3:13.1f}us "
                      f"{tr * 1e3:9.1f}us {td / tr:7.2f}x")


if __name__ == "__main__":
    main()
