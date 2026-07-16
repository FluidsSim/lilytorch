#!/usr/bin/env python3
"""cuda_native_port item 2.4 bench — streaming SDF resolve timing.

The union-AABB ``_multi`` path was deleted in item 2.4; the Regime-A
``_direct`` path was deleted in CL2 (the guard sync cost ~75 us/step
dominated any kernel savings, and for articulated bodies the fast path was
never legal anyway).  Resolve is now the sole streaming path.  This bench
measures resolve-only wall-clock time per layout/dtype/dimension.

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

from test_per_body_buffers import make_scene, run_resolve  # noqa: E402


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
    print(f"{'scene':22s} {'dtype':6s} {'resolve':>10s}")
    print("-" * 44)
    for dtype, dname in ((torch.float32, "fp32"), (torch.float64, "fp64")):
        for dim in (2, 3):
            for layout in ("single", "separated", "multilink"):
                sc = make_scene(dim, layout, dtype)
                tr = timeit(lambda sc=sc, d=dtype: run_resolve(sc, d))
                name = f"{dim}d_{layout}"
                print(f"{name:22s} {dname:6s} {tr * 1e3:9.1f}us")


if __name__ == "__main__":
    main()
