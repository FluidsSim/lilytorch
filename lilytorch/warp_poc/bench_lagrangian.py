"""Benchmark the Warp Lagrangian force kernel vs the native CUDA kernel.

Per-body surface integration scattered via atomicAdd.  Native is the
hand-written `lagrangian_forces_3d` (one thread/triangle, 2-D grid, double
atomicAdd).  Warp is the single-source `@wp.kernel` (eager + CUDA-graph).

Geometry (triangulation, offsets, elem_body map) is fixed across steps, so it
is built once; fields are GPU-resident torch tensors wrapped zero-copy via
`wp.from_torch`.  We time only the per-step launch + output zero, which is the
production-relevant cost.

    python -m lilytorch.warp_poc.bench_lagrangian --tris 5000 20000 80000
"""
from __future__ import annotations

import argparse, math, time
import torch, warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import lagrangian_forces_3d as native_3d
    _NATIVE = True
except Exception as e:
    print(f"[warn] native unavailable: {e}")
    _NATIVE = False

from lilytorch.warp_poc.warp_lagrangian import (
    lagrangian_forces_3d_kernel, _elem_body_from_offsets,
)


def _sphere_tris(cx, cy, cz, R, lon, lat, device):
    lons = torch.linspace(0, 2 * math.pi, lon + 1, dtype=torch.float64)[:-1]
    lats = torch.linspace(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, lat,
                          dtype=torch.float64)
    verts = []
    for la in lats:
        for lo in lons:
            verts.append((cx + R * math.cos(la) * math.cos(lo),
                          cy + R * math.cos(la) * math.sin(lo),
                          cz + R * math.sin(la)))
    verts = torch.tensor(verts, dtype=torch.float64)
    tris = []
    for j in range(lat - 1):
        for i in range(lon):
            ip = (i + 1) % lon
            v0 = j * lon + i; v1 = j * lon + ip
            v2 = (j + 1) * lon + i; v3 = (j + 1) * lon + ip
            tris.append((v0, v2, v3)); tris.append((v0, v3, v1))
    tris = torch.tensor(tris, dtype=torch.long)
    v = verts[tris]
    cross = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
    area = 0.5 * torch.norm(cross, dim=1)
    n = cross / (2 * area.unsqueeze(1)).clamp_min(1e-30)
    centroid = v.mean(dim=1)
    return (centroid.T.contiguous().to(device),
            n.T.contiguous().to(device),
            area.contiguous().to(device))


def _scene(target_tris, device):
    # Two spheres; pick (lon,lat) so total triangles ≈ target_tris.
    per = max(target_tris // 2, 200)
    # tris per sphere ≈ 2*lon*(lat-1); choose lat≈lon/2.
    lon = max(int(((per / 2) ** 0.5) * (2 ** 0.5)), 8)
    lat = max(lon // 2, 4)
    s1 = _sphere_tris(1.0, 0.9, 0.8, 0.3, lon, lat, device)
    s2 = _sphere_tris(1.5, 1.1, 1.0, 0.25, lon, lat, device)
    centroid = torch.cat([s1[0], s2[0]], dim=1)
    normal = torch.cat([s1[1], s2[1]], dim=1)
    area = torch.cat([s1[2], s2[2]], dim=0)
    offs = torch.tensor([0, s1[0].shape[1], s1[0].shape[1] + s2[0].shape[1]],
                        dtype=torch.int64, device=device)
    com = torch.tensor([[1.0, 0.9, 0.8], [1.5, 1.1, 1.0]],
                       dtype=torch.float64, device=device)
    return centroid, normal, area, offs, com


def time_ms(fn, warmup=5, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def run(targets, device="cuda:0"):
    Mx = My = Mz = 64
    h = 0.05
    torch.manual_seed(0)
    fields = [torch.randn(Mx, My, Mz, dtype=torch.float64, device=device)
              for _ in range(7)]
    nrho = torch.tensor([0.4], dtype=torch.float64, device=device)
    geom = (0.0, 0.0, 0.0, 1.0 / h, 1.0 / h, 1.0 / h, Mx, My, Mz)

    print(f"\n{'─'*78}")
    print(f"  Warp vs native lagrangian_forces_3d  (device={device}, grid={Mx}³)")
    print(f"{'─'*78}")
    for tgt in targets:
        c, n_, a, offs, com = _scene(tgt, device)
        T = int(c.shape[1])

        if _NATIVE:
            exx, eyy, ezz, exy, exz, eyz, p = fields
            t_nat = time_ms(lambda: native_3d(
                exx, eyy, ezz, exy, exz, eyz, p, nrho,
                c, n_, a, offs, com, *geom, method="linear"))
        else:
            t_nat = float("nan")

        # Warp prepared inputs (geometry built once; fields wrapped zero-copy).
        def fw(x): return wp.from_torch(x.reshape(-1).contiguous())
        eb = wp.from_torch(_elem_body_from_offsets(offs, T, c.device).contiguous())
        out = wp.zeros(com.shape[0] * 12, dtype=wp.float64, device=device)
        wp_in = [fw(fields[i]) for i in (0, 1, 2, 3, 4, 5)] + [fw(fields[6]), fw(nrho), 1,
                 fw(c[0]), fw(c[1]), fw(c[2]), fw(n_[0]), fw(n_[1]), fw(n_[2]),
                 fw(a), eb, fw(com),
                 Mx, My, Mz, 0.0, 0.0, 0.0, 1.0/h, 1.0/h, 1.0/h, 0, 0.0, out]

        def eager():
            out.zero_()
            wp.launch(lagrangian_forces_3d_kernel, dim=T, inputs=wp_in, device=device)
        eager()
        t_eager = time_ms(eager)

        with wp.ScopedCapture(device) as cap:
            out.zero_()
            wp.launch(lagrangian_forces_3d_kernel, dim=T, inputs=wp_in, device=device)
        graph = cap.graph
        t_graph = time_ms(lambda: wp.capture_launch(graph))

        def ms(v): return f"{v:.3f}ms" if not math.isnan(v) else " n/a"
        def sp(t): return f"{t/t_nat:.2f}×" if not math.isnan(t_nat) else "n/a"
        print(f"  T={T:>7}  native={ms(t_nat):>9}  "
              f"warp-eager={ms(t_eager):>9} ({sp(t_eager)})  "
              f"warp-graph={ms(t_graph):>9} ({sp(t_graph)})")
    print(f"{'─'*78}")
    print("  <1.00× = Warp faster than native.  Both atomicAdd-scatter into (B,12)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tris", type=int, nargs="+", default=[5000, 20000, 80000])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run(a.tris, a.device)
