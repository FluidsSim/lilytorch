#!/usr/bin/env python3
"""Graphed vs eager wall/host time for the streaming Eulerian force readout.

Measures the default-on ``ForcesPostGraph`` CUDA-graph replay against the
production eager launch on the synthetic
flat-table scenes (no FARMS/MuJoCo).  Both paths run the identical per-step
protocol a coupled sim pays: fresh kin/aabb tensors each step (BDIMhandler's
H2D pack), zeroed out-buffer, one n·δ readout.

Two numbers per backend:
  * submit  — host time to enqueue one step's readout (no sync); this is the
              Python/launch overhead the graph replay removes, and the number
              that matters inside a GPU-busy coupled step.
  * wall    — synchronised end-to-end per call (includes GPU execution).

Usage:
    python bench_forces_graph.py --dim 2 --iters 1000
    python bench_forces_graph.py --dim 3 --iters 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lilytorch.src.forces import (                       # noqa: E402
    ForcesPostGraph,
    streaming_sdf_forces_post_2d,
    streaming_sdf_forces_post_3d,
)
from lilytorch.tests.scene_2d import make_synthetic_scene_2d      # noqa: E402
from lilytorch.tests.scene_3d import make_synthetic_scene  # noqa: E402


def build(dim, B, dtype, dev):
    # test helpers run the streaming bridge to build a REAL union SDF, so the
    # readout kernel does representative in-band work (not all-early-return)
    from lilytorch.tests.test_forces import (_fill_union_sdf_2d,
                                             _fill_union_sdf_3d)
    if dim == 2:
        sc = make_synthetic_scene_2d(384, 256, B, device=dev, dtype=dtype)
        keys = ("F_flat", "body_meta", "kin", "gx", "gy")
    else:
        sc = make_synthetic_scene(96, 64, 64, B, device=dev, dtype=dtype)
        keys = ("F_flat", "body_meta", "kin", "gx", "gy", "gz")
    for k in keys:
        sc[k] = sc[k].to(dtype)
    gs = ((sc["Ngx"], sc["Ngy"]) if dim == 2
          else (sc["Ngx"], sc["Ngy"], sc["Ngz"]))
    fill = _fill_union_sdf_2d if dim == 2 else _fill_union_sdf_3d
    sc["sdf_cc_g"] = fill(sc, dtype, dev).contiguous()
    fields = [torch.randn(gs, dtype=dtype, device=dev)
              for _ in range(dim + 1)]
    nrho = torch.tensor([0.13], dtype=dtype, device=dev)
    out = torch.zeros((B, 6 * (dim - 1)), dtype=torch.float64, device=dev)
    return sc, fields, nrho, out


def one_eager(dim, sc, fields, nrho, out, kin):
    h = float(sc["h"])
    out.zero_()
    if dim == 2:
        streaming_sdf_forces_post_2d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            kin, sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
            h, int(sc["max_vol"]), sc["sdf_cc_g"], 0,
            fields[0], fields[1], fields[2], nrho,
            2.0 * h, 0.0, h * h, 1, out, 0, 0.0)
    else:
        streaming_sdf_forces_post_3d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            kin, sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"],
            h, int(sc["max_vol"]), sc["sdf_cc_g"], 0,
            fields[0], fields[1], fields[2], fields[3], nrho,
            2.0 * h, 0.0, h ** 3, 1, out, 0, 0.0)


def one_graph(dim, fg, sc, fields, nrho, out, kin):
    h = float(sc["h"])
    grids = ((sc["gx"], sc["gy"]) if dim == 2
             else (sc["gx"], sc["gy"], sc["gz"]))
    cell = h * h if dim == 2 else h ** 3
    fg.run(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
           kin, sc["aabb_lo"], sc["aabb_dim"], grids,
           h, int(sc["max_vol"]), sc["sdf_cc_g"], 0,
           tuple(fields[:dim]), fields[dim], nrho,
           2.0 * h, 0.0, cell, 1, out)


def bench(dim, iters, B, dtype=torch.float32, dev="cuda"):
    sc, fields, nrho, out = build(dim, B, dtype, dev)
    # per-step fresh kin, cycled through a small pool (allocator-realistic)
    kin_pool = [sc["kin"].clone() for _ in range(4)]

    def timed(fn):
        for i in range(20):                               # warm-up / JIT
            fn(kin_pool[i % 4])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(iters):
            fn(kin_pool[i % 4])
        submit = (time.perf_counter() - t0) / iters * 1e6
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / iters * 1e6
        return submit, wall

    ref = out.clone()
    s_e, w_e = timed(lambda k: one_eager(dim, sc, fields, nrho, out, k))
    torch.cuda.synchronize()
    ref.copy_(out)

    fg = ForcesPostGraph(dim)
    s_g, w_g = timed(lambda k: one_graph(dim, fg, sc, fields, nrho, out, k))
    torch.cuda.synchronize()

    err = (out - ref).abs().max().item()
    print(f"[{dim}-D] B={B} grid={'384x256' if dim == 2 else '96x64x64'} "
          f"max_vol={sc['max_vol']} iters={iters}")
    print(f"  eager : submit {s_e:8.1f} us/call   wall {w_e:8.1f} us/call")
    print(f"  graph : submit {s_g:8.1f} us/call   wall {w_g:8.1f} us/call "
          f"(replays={fg.replays}, captures={fg.captures}, "
          f"eager_fallbacks={fg.eager_calls})")
    print(f"  submit speedup {s_e / s_g:4.1f}x   wall speedup {w_e / w_g:4.1f}x"
          f"   graph-vs-eager |dF|max = {err:.3e}")
    assert fg.replays >= iters - 10, "graph path did not engage"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--bodies", type=int, default=8)
    args = ap.parse_args()
    bench(args.dim, args.iters, args.bodies)
