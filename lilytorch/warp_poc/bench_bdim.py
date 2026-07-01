"""Benchmark Warp **Kernel B** (`bdim_coeff_3d` + FD normals) vs native CUDA.

Companion to ``bench_bdim_memory.py`` (which covers the peak-MEMORY claim).  This
one covers WALL-CLOCK: native ``bdim_coeff_3d`` vs the Warp port, eager and under
CUDA-graph capture (the production-relevant timing, per HANDOFF).  Full-grid
dirty AABB = maximum work.  float64 (the parity dtype).

    python -m lilytorch.warp_poc.bench_bdim --grids 96 128 160
"""
from __future__ import annotations

import argparse, math, time
import torch, warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import bdim_coeff_3d as native_3d
    _NATIVE = True
except Exception as e:
    print(f"[warn] native unavailable: {e}")
    _NATIVE = False

from lilytorch.warp_poc.warp_bdim import bdim_coeff_3d_warp

RHO, DT = 1000.0, 1e-3


def _problem(N, dev, dtype=torch.float64):
    h = 1.0 / N
    g = torch.Generator(device=dev).manual_seed(0)
    xs = (torch.arange(N, device=dev) - N / 2.0) * h
    X, Y, Z = torch.meshgrid(xs, xs, xs, indexing="ij")
    R = 0.3
    sph = lambda o: (torch.sqrt((X - o) ** 2 + Y ** 2 + Z ** 2) - R).to(dtype)
    rnd = lambda: torch.randn(N, N, N, dtype=dtype, device=dev, generator=g)
    P = dict(h=h, eps=2 * h, N=N,
             su=sph(0.5 * h), sv=sph(0.0), sw=sph(0.0),
             bu=rnd(), bv=rnd(), bw=rnd(), up=rnd(), vp=rnd(), wp=rnd())
    return P


def _outs(P, dev, dtype=torch.float64):
    N = P["N"]
    base = DT / RHO
    return (P["up"].clone(), P["vp"].clone(), P["wp"].clone(),
            torch.full((N - 1, N - 2, N - 2), base, dtype=dtype, device=dev),
            torch.full((N - 2, N - 1, N - 2), base, dtype=dtype, device=dev),
            torch.full((N - 2, N - 2, N - 1), base, dtype=dtype, device=dev))


def _call(fn, P, outs):
    N = P["N"]
    u0, v0, w0, ch, cv, cw = outs
    fn(P["up"], P["vp"], P["wp"], P["su"], P["sv"], P["sw"],
       P["bu"], P["bv"], P["bw"], u0, v0, w0, ch, cv, cw,
       P["eps"], RHO, DT, P["h"], 0, 0, 0, N, N, N, 1)


def time_ms(fn, warmup=5, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def run(grids, device="cuda:0"):
    print(f"\n{'─'*72}")
    print(f"  Warp vs native Kernel B (bdim_coeff_3d), float64  (device={device})")
    print(f"{'─'*72}")
    for N in grids:
        P = _problem(N, device)
        outs = _outs(P, device)

        t_nat = (time_ms(lambda: _call(native_3d, P, outs))
                 if _NATIVE else float("nan"))

        _call(bdim_coeff_3d_warp, P, outs)   # warmup / JIT
        t_eager = time_ms(lambda: _call(bdim_coeff_3d_warp, P, outs))

        # CUDA-graph capture of the single launch (zero-copy views are persistent)
        with wp.ScopedCapture(device=device) as cap:
            _call(bdim_coeff_3d_warp, P, outs)
        graph = cap.graph
        t_graph = time_ms(lambda: wp.capture_launch(graph))

        ms = lambda v: f"{v:.3f}ms" if not math.isnan(v) else " n/a"
        sp = lambda t: f"{t/t_nat:.2f}×" if not math.isnan(t_nat) else "n/a"
        print(f"  {N}³  native={ms(t_nat):>9}  "
              f"warp-eager={ms(t_eager):>9} ({sp(t_eager)})  "
              f"warp-graph={ms(t_graph):>9} ({sp(t_graph)})")
        del P, outs
        torch.cuda.empty_cache()
    print(f"{'─'*72}")
    print("  <1.00× = Warp faster.  See bench_bdim_memory.py for the (load-bearing)")
    print("  peak-memory result: Warp adds 0.0 GiB, identical to native.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[96, 128, 160])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run(a.grids, a.device)
