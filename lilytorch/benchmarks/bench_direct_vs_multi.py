#!/usr/bin/env python3
"""Benchmark: old multi-kernel vs new direct kernel.

Compares speed (wall time + GPU time), peak GPU memory, and CPU-GPU accuracy
across B ∈ {1, 2, 4, 8} for both float32 and float64 in 2-D and 3-D.
"""

import torch
import time
import gc
import sys
from lilytorch.src import native
from lilytorch.tests.scene_2d import make_synthetic_scene_2d

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False


def _allclose_msg(a, b, rtol, atol):
    diff = (a - b).abs()
    return f"max_diff={diff.max().item():.3e}  |  within rtol={rtol}, atol={atol}: {torch.allclose(a, b, rtol=rtol, atol=atol)}"


def _gpu_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def bench_2d(B, dtype, grid_xy=(128, 96), warmup=5, repeat=20):
    """Benchmark 2-D streaming SDF: direct vs multi kernel."""
    dev = 'cuda:0'
    sc = make_synthetic_scene_2d(grid_xy[0], grid_xy[1], B, device=dev, dtype=dtype)
    for k in ('F_flat', 'body_meta', 'kin', 'gx', 'gy'):
        sc[k] = sc[k].to(dtype)

    Ngx, Ngy = sc['Ngx'], sc['Ngy']
    opt = dict(dtype=dtype, device=dev)
    h = float(sc['h'])

    # Common inputs
    F_flat = sc['F_flat']
    F_offsets = sc['F_offsets']
    body_shapes = sc['body_shapes']
    body_meta = sc['body_meta']
    kin = sc['kin']
    aabb_lo = sc['aabb_lo']
    aabb_dim = sc['aabb_dim']
    gx, gy = sc['gx'], sc['gy']
    max_vol = int(sc['max_vol'])

    sdf_cc = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_u = torch.full((Ngx, Ngy), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy), **opt)
    bv = torch.zeros((Ngx, Ngy), **opt)
    interp = 0

    # Key buffers (needed by multi kernel)
    N = Ngx * Ngy
    key_cc = torch.empty(N, dtype=torch.int64, device=dev)
    key_u = torch.empty(N, dtype=torch.int64, device=dev)
    key_v = torch.empty(N, dtype=torch.int64, device=dev)
    num_u = torch.zeros(1, **opt)
    num_v = torch.zeros(1, **opt)
    den_u = torch.zeros(1, **opt)
    den_v = torch.zeros(1, **opt)

    # Helper: reset output buffers
    def reset():
        sdf_cc.fill_(1e4); sdf_u.fill_(1e4); sdf_v.fill_(1e4)
        bu.zero_(); bv.zero_()

    # ---- CPU reference ----
    cpu_opt = dict(dtype=dtype, device='cpu')
    sc_cpu = make_synthetic_scene_2d(grid_xy[0], grid_xy[1], B, device='cpu', dtype=dtype)
    for k in ('F_flat', 'body_meta', 'kin', 'gx', 'gy'):
        sc_cpu[k] = sc_cpu[k].to(dtype)
    sdf_cpu = torch.full((Ngx, Ngy), 1e4, **cpu_opt)
    native.streaming_sdf_stag_2d_multi(
        sc_cpu['F_flat'], sc_cpu['F_offsets'],
        sc_cpu['body_shapes'].reshape(-1), sc_cpu['body_meta'].reshape(-1),
        sc_cpu['kin'].reshape(-1),
        sc_cpu['aabb_lo'].reshape(-1), sc_cpu['aabb_dim'].reshape(-1),
        sc_cpu['gx'], sc_cpu['gy'], h, max_vol,
        sdf_cpu,
        torch.full((Ngx, Ngy), 1e4, **cpu_opt),
        torch.full((Ngx, Ngy), 1e4, **cpu_opt),
        torch.zeros((Ngx, Ngy), **cpu_opt),
        torch.zeros((Ngx, Ngy), **cpu_opt),
        torch.empty(N, dtype=torch.int64, device='cpu'),
        torch.empty(N, dtype=torch.int64, device='cpu'),
        torch.empty(N, dtype=torch.int64, device='cpu'),
        interp, 0, 0, Ngx, Ngy,
        torch.zeros(1, **cpu_opt), torch.zeros(1, **cpu_opt),
        torch.zeros(1, **cpu_opt), torch.zeros(1, **cpu_opt), 0.0,
    )

    # ---- Direct kernel benchmark ----
    torch.cuda.reset_peak_memory_stats()
    reset()
    # Warmup
    for _ in range(warmup):
        native.streaming_sdf_stag_2d_direct(
            F_flat, F_offsets, body_shapes, body_meta, kin,
            aabb_lo, aabb_dim, gx, gy, h, max_vol,
            sdf_cc, sdf_u, sdf_v, bu, bv,
            interp, 0, 0, Ngx, Ngy,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_direct = []
    for _ in range(repeat):
        reset()
        start.record()
        native.streaming_sdf_stag_2d_direct(
            F_flat, F_offsets, body_shapes, body_meta, kin,
            aabb_lo, aabb_dim, gx, gy, h, max_vol,
            sdf_cc, sdf_u, sdf_v, bu, bv,
            interp, 0, 0, Ngx, Ngy,
        )
        end.record()
        torch.cuda.synchronize()
        times_direct.append(start.elapsed_time(end))
    mem_direct = _gpu_memory_mb()
    sdf_direct = sdf_cc.clone()
    acc_direct = (sdf_cpu - sdf_direct.cpu()).abs().max().item()

    # ---- Multi kernel benchmark ----
    torch.cuda.reset_peak_memory_stats()
    reset()
    for _ in range(warmup):
        native.streaming_sdf_stag_2d_multi(
            F_flat, F_offsets,
            body_shapes, body_meta, kin,
            aabb_lo, aabb_dim,
            gx, gy, h, max_vol,
            sdf_cc, sdf_u, sdf_v, bu, bv,
            key_cc, key_u, key_v,
            interp, 0, 0, Ngx, Ngy,
            num_u, num_v, den_u, den_v, 0.0,
        )
    torch.cuda.synchronize()

    times_multi = []
    for _ in range(repeat):
        reset()
        start.record()
        native.streaming_sdf_stag_2d_multi(
            F_flat, F_offsets,
            body_shapes, body_meta, kin,
            aabb_lo, aabb_dim,
            gx, gy, h, max_vol,
            sdf_cc, sdf_u, sdf_v, bu, bv,
            key_cc, key_u, key_v,
            interp, 0, 0, Ngx, Ngy,
            num_u, num_v, den_u, den_v, 0.0,
        )
        end.record()
        torch.cuda.synchronize()
        times_multi.append(start.elapsed_time(end))
    mem_multi = _gpu_memory_mb()
    sdf_multi = sdf_cc.clone()
    acc_multi = (sdf_cpu - sdf_multi.cpu()).abs().max().item()

    t_direct = sum(times_direct) / len(times_direct)
    t_multi = sum(times_multi) / len(times_multi)

    return {
        'B': B, 'dtype': str(dtype), 'dim': '2D', 'grid': grid_xy,
        'direct_ms': t_direct,
        'multi_ms': t_multi,
        'speedup': t_multi / t_direct if t_direct > 0 else float('inf'),
        'direct_mem_mb': mem_direct,
        'multi_mem_mb': mem_multi,
        'direct_err_cpu': acc_direct,
        'multi_err_cpu': acc_multi,
    }


def bench_3d(B, dtype, grid_xyz=(48, 48, 48), warmup=5, repeat=10):
    """Benchmark 3-D streaming SDF: direct vs multi kernel."""
    dev = 'cuda:0'
    opt = dict(dtype=dtype, device=dev)
    Ngx, Ngy, Ngz = grid_xyz
    h = 0.02

    # Build synthetic 3-D scene manually (no scene_3d helper available)
    torch.manual_seed(42)
    Mx, My, Mz = 12, 12, 12
    Nper = Mx * My * Mz
    F_flat = torch.randn(B * Nper, **opt) * 0.01
    F_offsets = torch.arange(0, (B + 1) * Nper, Nper, dtype=torch.int64, device=dev)
    body_shapes = torch.full((B, 3), Mx, dtype=torch.int64, device=dev)
    body_shapes[:, 1] = My
    body_shapes[:, 2] = Mz

    # body_meta: [bx0, by0, bz0, dx, dy, dz, idx, idy, idz, max_vol] per body (10 fields)
    body_meta = torch.zeros(B, 10, **opt)
    for b in range(B):
        bx, by, bz = -0.05 + 0.15 * (b % 3), -0.05 + 0.15 * ((b // 3) % 3), -0.05 + 0.1 * b
        body_meta[b, 0:3] = torch.tensor([bx, by, bz], **opt)
        body_meta[b, 3:6] = torch.tensor([0.1, 0.1, 0.1], **opt)
        body_meta[b, 6:9] = torch.tensor([100.0, 100.0, 100.0], **opt)
        body_meta[b, 9] = 10000.0

    # kin: 21 fields per body (rotation matrix + position + CoM + velocities)
    kin = torch.zeros(B, 21, **opt)
    for b in range(B):
        kin[b, 0] = 1.0; kin[b, 4] = 1.0; kin[b, 8] = 1.0  # identity rotation
        kin[b, 9] = 0.1 * b; kin[b, 10] = 0.05 * b; kin[b, 11] = 0.02 * b  # body pos
        kin[b, 12] = kin[b, 9]; kin[b, 13] = kin[b, 10]; kin[b, 14] = kin[b, 11]  # CoM
        kin[b, 15] = 0.01; kin[b, 16] = 0.0; kin[b, 17] = 0.0  # linear velocity

    gx = torch.linspace(0.0, (Ngx - 1) * h, Ngx, **opt)
    gy = torch.linspace(0.0, (Ngy - 1) * h, Ngy, **opt)
    gz = torch.linspace(0.0, (Ngz - 1) * h, Ngz, **opt)

    aabb_lo = torch.zeros(B, 3, dtype=torch.int64, device=dev)
    aabb_dim = torch.full((B, 3), 10, dtype=torch.int64, device=dev)
    for b in range(B):
        aabb_lo[b, 0] = b * 3
        aabb_lo[b, 1] = b * 2
        aabb_lo[b, 2] = 1

    max_vol = 10 * 10 * 10

    sdf_cc = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_u = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_v = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    sdf_w = torch.full((Ngx, Ngy, Ngz), 1e4, **opt)
    bu = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bv = torch.zeros((Ngx, Ngy, Ngz), **opt)
    bw = torch.zeros((Ngx, Ngy, Ngz), **opt)
    interp = 0

    N = Ngx * Ngy * Ngz
    key_cc = torch.empty(N, dtype=torch.int64, device=dev)
    key_u = torch.empty(N, dtype=torch.int64, device=dev)
    key_v = torch.empty(N, dtype=torch.int64, device=dev)
    key_w = torch.empty(N, dtype=torch.int64, device=dev)
    num_u = torch.zeros(1, **opt); num_v = torch.zeros(1, **opt); num_w = torch.zeros(1, **opt)
    den_u = torch.zeros(1, **opt); den_v = torch.zeros(1, **opt); den_w = torch.zeros(1, **opt)

    def reset():
        sdf_cc.fill_(1e4); sdf_u.fill_(1e4); sdf_v.fill_(1e4); sdf_w.fill_(1e4)
        bu.zero_(); bv.zero_(); bw.zero_()

    # ---- Direct kernel benchmark ----
    torch.cuda.reset_peak_memory_stats()
    reset()
    for _ in range(warmup):
        native.streaming_sdf_stag_3d_direct(
            F_flat, F_offsets, body_shapes, body_meta, kin,
            aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
            interp, 0, 0, 0, Ngx, Ngy, Ngz,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_direct = []
    for _ in range(repeat):
        reset()
        start.record()
        native.streaming_sdf_stag_3d_direct(
            F_flat, F_offsets, body_shapes, body_meta, kin,
            aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
            interp, 0, 0, 0, Ngx, Ngy, Ngz,
        )
        end.record()
        torch.cuda.synchronize()
        times_direct.append(start.elapsed_time(end))
    mem_direct = _gpu_memory_mb()

    # ---- Multi kernel benchmark ----
    torch.cuda.reset_peak_memory_stats()
    reset()
    for _ in range(warmup):
        native.streaming_sdf_stag_3d_multi(
            F_flat, F_offsets,
            body_shapes, body_meta, kin,
            aabb_lo, aabb_dim,
            gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
            key_cc, key_u, key_v, key_w,
            interp, 0, 0, 0, Ngx, Ngy, Ngz,
            num_u, num_v, num_w, den_u, den_v, den_w, 0.0,
        )
    torch.cuda.synchronize()

    times_multi = []
    for _ in range(repeat):
        reset()
        start.record()
        native.streaming_sdf_stag_3d_multi(
            F_flat, F_offsets,
            body_shapes, body_meta, kin,
            aabb_lo, aabb_dim,
            gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, bu, bv, bw,
            key_cc, key_u, key_v, key_w,
            interp, 0, 0, 0, Ngx, Ngy, Ngz,
            num_u, num_v, num_w, den_u, den_v, den_w, 0.0,
        )
        end.record()
        torch.cuda.synchronize()
        times_multi.append(start.elapsed_time(end))
    mem_multi = _gpu_memory_mb()
    sdf_direct = sdf_cc.clone()
    sdf_multi_cc = sdf_cc.clone()

    t_direct = sum(times_direct) / len(times_direct)
    t_multi = sum(times_multi) / len(times_multi)

    # CPU reference (B=1 only — too slow otherwise)
    acc_direct = acc_multi = float('nan')
    if B == 1:
        cpu_opt = dict(dtype=dtype, device='cpu')
        sdf_cpu = torch.full((Ngx, Ngy, Ngz), 1e4, **cpu_opt)
        native.streaming_sdf_stag_3d_multi(
            F_flat.cpu(), F_offsets.cpu(),
            body_shapes.cpu().reshape(-1), body_meta.cpu().reshape(-1),
            kin.cpu().reshape(-1),
            aabb_lo.cpu().reshape(-1), aabb_dim.cpu().reshape(-1),
            gx.cpu(), gy.cpu(), gz.cpu(), h, max_vol,
            sdf_cpu,
            torch.full((Ngx, Ngy, Ngz), 1e4, **cpu_opt),
            torch.full((Ngx, Ngy, Ngz), 1e4, **cpu_opt),
            torch.full((Ngx, Ngy, Ngz), 1e4, **cpu_opt),
            torch.zeros((Ngx, Ngy, Ngz), **cpu_opt),
            torch.zeros((Ngx, Ngy, Ngz), **cpu_opt),
            torch.zeros((Ngx, Ngy, Ngz), **cpu_opt),
            torch.empty(N, dtype=torch.int64, device='cpu'),
            torch.empty(N, dtype=torch.int64, device='cpu'),
            torch.empty(N, dtype=torch.int64, device='cpu'),
            torch.empty(N, dtype=torch.int64, device='cpu'),
            interp, 0, 0, 0, Ngx, Ngy, Ngz,
            torch.zeros(1, **cpu_opt), torch.zeros(1, **cpu_opt), torch.zeros(1, **cpu_opt),
            torch.zeros(1, **cpu_opt), torch.zeros(1, **cpu_opt), torch.zeros(1, **cpu_opt), 0.0,
        )
        acc_direct = (sdf_cpu - sdf_direct.cpu()).abs().max().item()
        acc_multi = (sdf_cpu - sdf_multi_cc.cpu()).abs().max().item()

    return {
        'B': B, 'dtype': str(dtype), 'dim': '3D', 'grid': grid_xyz,
        'direct_ms': t_direct,
        'multi_ms': t_multi,
        'speedup': t_multi / t_direct if t_direct > 0 else float('inf'),
        'direct_mem_mb': mem_direct,
        'multi_mem_mb': mem_multi,
        'direct_err_cpu': acc_direct,
        'multi_err_cpu': acc_multi,
    }


def main():
    print("=" * 110)
    print(f"{'B':>3} {'dim':>4} {'dtype':>8} {'direct_ms':>10} {'multi_ms':>10} "
          f"{'speedup':>8} {'dir_mem_mb':>10} {'mul_mem_mb':>10} "
          f"{'dir_err':>10} {'mul_err':>10}")
    print("-" * 110)

    results = []

    # 2-D benchmarks
    for dtype in [torch.float32, torch.float64]:
        for B in [1, 2, 4, 8]:
            r = bench_2d(B, dtype)
            results.append(r)
            print(f"{r['B']:>3} {r['dim']:>4} {r['dtype']:>8} "
                  f"{r['direct_ms']:>10.4f} {r['multi_ms']:>10.4f} "
                  f"{r['speedup']:>8.2f}x "
                  f"{r['direct_mem_mb']:>10.1f} {r['multi_mem_mb']:>10.1f} "
                  f"{r['direct_err_cpu']:>10.2e} {r['multi_err_cpu']:>10.2e}")

    print()

    # 3-D benchmarks
    for dtype in [torch.float32, torch.float64]:
        for B in [1, 2, 4, 8]:
            r = bench_3d(B, dtype)
            results.append(r)
            err_d = f"{r['direct_err_cpu']:.2e}" if not (isinstance(r['direct_err_cpu'], float) and r['direct_err_cpu'] != r['direct_err_cpu']) else "N/A"
            err_m = f"{r['multi_err_cpu']:.2e}" if not (isinstance(r['multi_err_cpu'], float) and r['multi_err_cpu'] != r['multi_err_cpu']) else "N/A"
            print(f"{r['B']:>3} {r['dim']:>4} {r['dtype']:>8} "
                  f"{r['direct_ms']:>10.4f} {r['multi_ms']:>10.4f} "
                  f"{r['speedup']:>8.2f}x "
                  f"{r['direct_mem_mb']:>10.1f} {r['multi_mem_mb']:>10.1f} "
                  f"{err_d:>10} {err_m:>10}")

    print("-" * 110)
    # Summary
    for dim in ['2D', '3D']:
        for dt in ['torch.float32', 'torch.float64']:
            subset = [r for r in results if r['dim'] == dim and r['dtype'] == dt]
            if subset:
                avg_speedup = sum(r['speedup'] for r in subset) / len(subset)
                avg_dmem = sum(r['direct_mem_mb'] for r in subset) / len(subset)
                avg_mmem = sum(r['multi_mem_mb'] for r in subset) / len(subset)
                mem_saved = (1.0 - avg_dmem / avg_mmem) * 100 if avg_mmem > 0 else 0
                print(f"  {dim} {dt}: avg speedup={avg_speedup:.2f}x, "
                      f"direct mem={avg_dmem:.0f}MB, multi mem={avg_mmem:.0f}MB "
                      f"({mem_saved:.0f}% less with direct)")


if __name__ == '__main__':
    main()
