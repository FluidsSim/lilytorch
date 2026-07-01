"""Benchmark the Warp high-order advection limiter vs native `advect_flux_add`.

Times a FULL momentum advection step (every velocity component i × direction d
accumulation into rhs[i], exactly the production loop in
``AdvDiffSolver._solve_convective``), native CUDA vs the Warp port, eager and
under CUDA-graph capture (the production-relevant timing, per HANDOFF lesson 3).
float64 (the parity dtype).  QUICK by default; pass --scheme to vary.

    python -m lilytorch.warp_poc.bench_advection --grids 64 96 128 --scheme quick
"""
from __future__ import annotations

import argparse, math, time
import torch, warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    _NATIVE = hasattr(torch.ops.lilytorch_kernels, "advect_flux_add")
except Exception as e:
    print(f"[warn] native unavailable: {e}")
    _NATIVE = False

from lilytorch.warp_poc.warp_advection import advect_flux_add_warp
from lilytorch.src.advection import _face_vel, _field_for_flux, _inner, _CUDA_SCHEME_IDS

DT_DH, C = 0.123, 0.37


def _vel(ndim, N, dev):
    g = torch.Generator(device="cpu").manual_seed(3)
    shape = (N + 2,) * ndim
    return [(torch.rand(shape, generator=g, dtype=torch.float64) - 0.5).to(dev)
            for _ in range(ndim)]


def _tasks(vel, rhs, ndim):
    """Precompute the (fv, p, rhs, d) flux tasks ONCE — the face-velocity /
    field slicing is torch work that must stay OUT of the CUDA-graph capture
    (and is an identical cost for native and warp, so excluding it isolates the
    kernel).  Mirrors the production (i, d) loop."""
    tasks = []
    for i in range(ndim):
        for d in range(ndim):
            fv = _face_vel(vel, i, d, ndim)
            p = _field_for_flux(vel[i], d, ndim)
            tasks.append((fv, p, rhs[i], d))
    return tasks


def _native_step(tasks, sid):
    for fv, p, rhs, d in tasks:
        torch.ops.lilytorch_kernels.advect_flux_add(fv, p, rhs, DT_DH, C, sid, d)


def _warp_step(tasks, sid):
    for fv, p, rhs, d in tasks:
        advect_flux_add_warp(fv, p, rhs, DT_DH, C, sid, d)


def time_ms(fn, warmup=5, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def run(grids, scheme, device="cuda:0"):
    sid = _CUDA_SCHEME_IDS[scheme]
    print(f"\n{'─'*74}")
    print(f"  Warp vs native advect_flux_add (full momentum step), {scheme}, float64")
    print(f"{'─'*74}")
    for ndim in (2, 3):
        for N in grids:
            vel = _vel(ndim, N, device)
            rhs = [torch.zeros_like(vel[i][_inner(ndim)]) for i in range(ndim)]
            tasks = _tasks(vel, rhs, ndim)

            t_nat = (time_ms(lambda: _native_step(tasks, sid))
                     if _NATIVE else float("nan"))

            _warp_step(tasks, sid)                          # JIT warmup
            t_eager = time_ms(lambda: _warp_step(tasks, sid))

            with wp.ScopedCapture(device=device) as cap:
                _warp_step(tasks, sid)
            graph = cap.graph
            t_graph = time_ms(lambda: wp.capture_launch(graph))

            ms = lambda v: f"{v:.3f}ms" if not math.isnan(v) else " n/a"
            sp = lambda t: f"{t/t_nat:.2f}×" if not math.isnan(t_nat) else "n/a"
            print(f"  {ndim}-D {N:>4}{'³' if ndim==3 else '²'}  "
                  f"native={ms(t_nat):>9}  "
                  f"warp-eager={ms(t_eager):>9} ({sp(t_eager)})  "
                  f"warp-graph={ms(t_graph):>9} ({sp(t_graph)})")
            del vel, rhs
            torch.cuda.empty_cache()
    print(f"{'─'*74}")
    print("  <1.00× = Warp faster (graph is the production-relevant number).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 96, 128])
    ap.add_argument("--scheme", default="quick", choices=sorted(_CUDA_SCHEME_IDS))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run(a.grids, a.scheme, a.device)
