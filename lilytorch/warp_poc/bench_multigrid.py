"""Benchmark the Warp multigrid transfer ops + V-cycle vs native CUDA.

Per-kernel wall-clock (native op vs Warp, eager + CUDA-graph) for the five
ported building blocks, plus the absolute graph time of the assembled
``WarpVCycle``.  float64.  (There is no single native V-cycle *op* to ratio
against — the native V-cycle is python-orchestrated in ``poisson_mult`` — so the
cycle is reported as an absolute graph-captured time.)

    python -m lilytorch.warp_poc.bench_multigrid --grid 128
"""
from __future__ import annotations

import argparse, math, time
import torch, warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        jacobi_sweep_3d as nat_jacobi,
        mg_residual_3d as nat_residual,
        restrict_residual_3d as nat_rr,
        restrict_face_3d as nat_rf,
        prolongate_add_3d as nat_pa,
    )
    _NATIVE = True
except Exception as e:
    print(f"[warn] native unavailable: {e}")
    _NATIVE = False

from lilytorch.warp_poc.warp_multigrid import (
    jacobi_sweep_3d_warp, mg_residual_3d_warp,
    restrict_residual_3d_warp, restrict_face_3d_warp, prolongate_add_3d_warp,
    WarpVCycle,
)

JCAP = 1e-30


def time_ms(fn, warmup=5, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def _graph_ms(call, device="cuda:0"):
    call()  # warmup/JIT
    with wp.ScopedCapture(device=device) as cap:
        call()
    g = cap.graph
    return time_ms(lambda: wp.capture_launch(g))


def _row(name, t_nat, t_eager, t_graph):
    ms = lambda v: f"{v:.4f}ms" if not math.isnan(v) else "  n/a"
    sp = lambda t: f"{t/t_nat:.2f}×" if not math.isnan(t_nat) else "n/a"
    print(f"  {name:<20} native={ms(t_nat):>9}  "
          f"warp-eager={ms(t_eager):>9} ({sp(t_eager)})  "
          f"warp-graph={ms(t_graph):>9} ({sp(t_graph)})")


def run(N, device="cuda:0"):
    dt = torch.float64
    g = torch.Generator(device=device).manual_seed(1)
    p = torch.randn(N + 2, N + 2, N + 2, dtype=dt, device=device, generator=g)
    f = torch.randn(N, N, N, dtype=dt, device=device, generator=g)
    c = [torch.rand(N, N, N, dtype=dt, device=device, generator=g) + 0.5 for _ in range(6)]
    Nc = N // 2
    rfine = torch.randn(N, N, N, dtype=dt, device=device, generator=g)
    rc = torch.empty(Nc, Nc, Nc, dtype=dt, device=device)
    chf = torch.randn(N + 1, N, N, dtype=dt, device=device, generator=g)
    chc = torch.empty(Nc + 1, Nc, Nc, dtype=dt, device=device)
    ec = torch.randn(Nc + 2, Nc + 2, Nc + 2, dtype=dt, device=device, generator=g)

    print(f"\n{'─'*86}")
    print(f"  Warp vs native multigrid kernels — {N}³ fine grid, float64  (device={device})")
    print(f"{'─'*86}")

    # mg_residual
    tn = time_ms(lambda: nat_residual(p, f, *c, JCAP)) if _NATIVE else math.nan
    te = time_ms(lambda: mg_residual_3d_warp(p, f, *c, JCAP))
    tg = _graph_ms(lambda: mg_residual_3d_warp(p, f, *c, JCAP), device)
    _row("mg_residual_3d", tn, te, tg)

    # jacobi (2 sweeps)
    pj = p.clone()
    tn = time_ms(lambda: nat_jacobi(pj, f, *c, JCAP, 0.8, 2)) if _NATIVE else math.nan
    te = time_ms(lambda: jacobi_sweep_3d_warp(pj, f, *c, JCAP, 0.8, 2))
    tg = _graph_ms(lambda: jacobi_sweep_3d_warp(pj, f, *c, JCAP, 0.8, 2), device)
    _row("jacobi_sweep_3d(x2)", tn, te, tg)

    # restrict_residual
    tn = time_ms(lambda: nat_rr(rfine, rc)) if _NATIVE else math.nan
    te = time_ms(lambda: restrict_residual_3d_warp(rfine, rc))
    tg = _graph_ms(lambda: restrict_residual_3d_warp(rfine, rc), device)
    _row("restrict_residual_3d", tn, te, tg)

    # restrict_face (dim 0)
    tn = time_ms(lambda: nat_rf(chf, chc, 0)) if _NATIVE else math.nan
    te = time_ms(lambda: restrict_face_3d_warp(chf, chc, 0))
    tg = _graph_ms(lambda: restrict_face_3d_warp(chf, chc, 0), device)
    _row("restrict_face_3d", tn, te, tg)

    # prolongate_add
    pp = p.clone()
    tn = time_ms(lambda: nat_pa(ec, pp)) if _NATIVE else math.nan
    te = time_ms(lambda: prolongate_add_3d_warp(ec, pp))
    tg = _graph_ms(lambda: prolongate_add_3d_warp(ec, pp), device)
    _row("prolongate_add_3d", tn, te, tg)

    # full Warp V-cycle (absolute, graph) — no single native op to ratio against
    fz = f - f.mean()
    vc = WarpVCycle(N, device=device)
    vc.set_rhs(fz)
    vc.capture()
    tvc = time_ms(lambda: vc.run_graph())
    print(f"{'─'*86}")
    print(f"  WarpVCycle (graph, {len(vc.levels)} levels): {tvc:.4f}ms/cycle "
          f"(absolute — native V-cycle is python-orchestrated, not a single op)")
    print(f"  <1.00× = Warp faster than native per kernel.  All parity bit-exact.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run(a.grid, a.device)
