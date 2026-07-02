"""Warp streaming-SDF benchmark + synthetic 3-D scene builder.

Measures the streaming-SDF hot path (body_update) across the Warp execution
designs:
  1. Warp eager      — per-body wp.launch calls
  2. Warp + graph    — sequential design, single CUDA-graph replay
  3. Warp fan-graph  — fanned all-body design (const in B), graph replay

``make_synthetic_scene`` / ``setup_warp_runner`` are also imported by the
kernel test files (test_parity.py, test_forces.py).

Metrics reported
────────────────
  * Wall-clock ms/step (GPU-synchronised)

Usage
─────
    python -m lilytorch.src.kernels.bench_viability               # default sizes
    python -m lilytorch.src.kernels.bench_viability --grids 64 128 256 --bodies 3 9

The script is self-contained: it generates synthetic sphere-SDF scenes that
match the production format, so no FARMS/MuJoCo dependency.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import warp as wp

from lilytorch.src.kernels.streaming_sdf import WarpStreamingSDF

# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic scene builder
# ─────────────────────────────────────────────────────────────────────────────

def _random_rotation(rng: np.random.Generator, dtype=np.float32):
    q = rng.standard_normal(4).astype(np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R.astype(dtype)


def make_synthetic_scene(
    Ngx: int, Ngy: int, Ngz: int, B: int,
    body_Mx: int = 20, body_My: int = 16, body_Mz: int = 14,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict:
    """Build all tensors needed by body_update_3d and WarpStreamingSDF.

    Returns a dict with keys:
      F_flat, F_offsets, body_shapes, body_meta,
      kin, aabb_lo, aabb_dim,
      gx, gy, gz, h, max_vol,
      dirty_bounds,
      sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w  (output buffers),
      key_cc, key_u, key_v, key_w,
      num_u, num_v, num_w, den_u, den_v, den_w  (blend stubs)
    """
    rng = np.random.default_rng(seed)
    h = 0.02
    gx = torch.linspace(0.0, (Ngx - 1) * h, Ngx, dtype=dtype, device=device)
    gy = torch.linspace(0.0, (Ngy - 1) * h, Ngy, dtype=dtype, device=device)
    gz = torch.linspace(0.0, (Ngz - 1) * h, Ngz, dtype=dtype, device=device)
    FAR = 1e4

    # Build body SDF tables (sphere of radius ~0.15 on a uniform grid)
    body_vol = body_Mx * body_My * body_Mz
    F_flat_np = np.empty(B * body_vol, dtype=np.float32)
    F_offsets_np = np.arange(B + 1, dtype=np.int64) * body_vol  # uniform size

    half_bx = (body_Mx - 1) * 0.01 * 0.5
    half_by = (body_My - 1) * 0.01 * 0.5
    half_bz = (body_Mz - 1) * 0.01 * 0.5
    bx_np = np.linspace(-half_bx, half_bx, body_Mx, dtype=np.float32)
    by_np = np.linspace(-half_by, half_by, body_My, dtype=np.float32)
    bz_np = np.linspace(-half_bz, half_bz, body_Mz, dtype=np.float32)
    Bx, By, Bz = np.meshgrid(bx_np, by_np, bz_np, indexing="ij")
    sphere_sdf = np.sqrt(Bx**2 + By**2 + Bz**2) - 0.07  # radius 0.07

    for b in range(B):
        F_flat_np[F_offsets_np[b]:F_offsets_np[b] + body_vol] = sphere_sdf.ravel()

    # body_shapes: [B*3]
    body_shapes_np = np.tile([body_Mx, body_My, body_Mz], B).reshape(B, 3).astype(np.int64)

    # body_meta: [B*10] = [bx0, by0, bz0, bx_last, by_last, bz_last, inv_dx, inv_dy, inv_dz, inv_vol]
    inv_dx = 1.0 / (bx_np[1] - bx_np[0])
    inv_dy = 1.0 / (by_np[1] - by_np[0])
    inv_dz = 1.0 / (bz_np[1] - bz_np[0])
    inv_vol = inv_dx * inv_dy * inv_dz
    meta_row = [bx_np[0], by_np[0], bz_np[0],
                bx_np[-1], by_np[-1], bz_np[-1],
                inv_dx, inv_dy, inv_dz, inv_vol]
    body_meta_np = np.tile(meta_row, B).reshape(B, 10).astype(np.float32)

    # kinematics: [B*21] = R_T(9) + bp(3) + cm(3) + lv(3) + av(3)
    kin_rows = []
    aabb_lo_rows = []
    aabb_dim_rows = []
    domain = np.array([Ngx - 1, Ngy - 1, Ngz - 1], dtype=np.float32) * h

    aabb_half = np.array([
        int(Ngx // B * 0.8),
        int(Ngy * 0.6),
        int(Ngz * 0.6),
    ], dtype=np.int64)

    for b in range(B):
        R = _random_rotation(rng)
        bp = rng.uniform(0.1, 0.9, 3).astype(np.float32) * domain
        cm = bp + rng.uniform(-0.01, 0.01, 3).astype(np.float32)
        lv = rng.uniform(-0.2, 0.2, 3).astype(np.float32)
        av = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        kin_rows.append(np.concatenate([R.ravel(), bp, cm, lv, av]))

        # Place AABBs tiled across the grid (no overlap between bodies)
        col = b % max(1, Ngx // aabb_half[0])
        i0 = int(col * (Ngx // B))
        j0 = 0
        k0 = 0
        Ai = min(aabb_half[0], Ngx - i0)
        Aj = min(aabb_half[1], Ngy - j0)
        Ak = min(aabb_half[2], Ngz - k0)
        aabb_lo_rows.append([i0, j0, k0])
        aabb_dim_rows.append([Ai, Aj, Ak])

    kin_np      = np.array(kin_rows, dtype=np.float32)      # [B, 21]
    aabb_lo_np  = np.array(aabb_lo_rows, dtype=np.int64)    # [B, 3]
    aabb_dim_np = np.array(aabb_dim_rows, dtype=np.int64)   # [B, 3]

    max_vol = int(np.prod(aabb_dim_np, axis=1).max())

    # Dirty AABB = union of all body AABBs
    i0_min = int(aabb_lo_np[:, 0].min()); j0_min = int(aabb_lo_np[:, 1].min()); k0_min = int(aabb_lo_np[:, 2].min())
    i1_max = int((aabb_lo_np[:, 0] + aabb_dim_np[:, 0]).max())
    j1_max = int((aabb_lo_np[:, 1] + aabb_dim_np[:, 1]).max())
    k1_max = int((aabb_lo_np[:, 2] + aabb_dim_np[:, 2]).max())
    dirty_bounds = (i0_min, j0_min, k0_min,
                    i1_max - i0_min, j1_max - j0_min, k1_max - k0_min)
    dirty_vol = (i1_max - i0_min) * (j1_max - j0_min) * (k1_max - k0_min)

    # Convert to torch CUDA tensors
    def t(arr, dt=None):
        return torch.from_numpy(arr).to(device=device, dtype=dt)

    F_flat_t      = t(F_flat_np)
    F_offsets_t   = t(F_offsets_np[:B])   # [B] start indices (native uses [B+1] → trim last)
    body_shapes_t = t(body_shapes_np)
    body_meta_t   = t(body_meta_np)
    kin_t         = t(kin_np)
    aabb_lo_t     = t(aabb_lo_np)
    aabb_dim_t    = t(aabb_dim_np)

    # Output buffers
    sdf_cc = torch.full((Ngx * Ngy * Ngz,), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Ngx * Ngy * Ngz,), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Ngx * Ngy * Ngz,), FAR, dtype=dtype, device=device)
    sdf_w  = torch.full((Ngx * Ngy * Ngz,), FAR, dtype=dtype, device=device)
    body_u = torch.zeros(Ngx * Ngy * Ngz, dtype=dtype, device=device)
    body_v = torch.zeros(Ngx * Ngy * Ngz, dtype=dtype, device=device)
    body_w = torch.zeros(Ngx * Ngy * Ngz, dtype=dtype, device=device)

    # Key scratch (for the facade body_update op)
    key_cc_t = torch.empty(dirty_vol, dtype=torch.int64, device=device)
    key_u_t  = torch.empty(dirty_vol, dtype=torch.int64, device=device)
    key_v_t  = torch.empty(dirty_vol, dtype=torch.int64, device=device)
    key_w_t  = torch.empty(dirty_vol, dtype=torch.int64, device=device)

    # Blend stubs (blend_eps=0 → kernel ignores them; distinct tensors required)
    num_u_t = torch.empty(1, dtype=dtype, device=device)
    num_v_t = torch.empty(1, dtype=dtype, device=device)
    num_w_t = torch.empty(1, dtype=dtype, device=device)
    den_u_t = torch.empty(1, dtype=dtype, device=device)
    den_v_t = torch.empty(1, dtype=dtype, device=device)
    den_w_t = torch.empty(1, dtype=dtype, device=device)

    return dict(
        Ngx=Ngx, Ngy=Ngy, Ngz=Ngz, B=B, h=h, max_vol=max_vol,
        dirty_bounds=dirty_bounds, dirty_vol=dirty_vol,
        gx=gx, gy=gy, gz=gz,
        F_flat=F_flat_t, F_offsets=F_offsets_t,
        body_shapes=body_shapes_t, body_meta=body_meta_t,
        kin=kin_t, aabb_lo=aabb_lo_t, aabb_dim=aabb_dim_t,
        sdf_cc=sdf_cc, sdf_u=sdf_u, sdf_v=sdf_v, sdf_w=sdf_w,
        body_u=body_u, body_v=body_v, body_w=body_w,
        key_cc=key_cc_t, key_u=key_u_t, key_v=key_v_t, key_w=key_w_t,
        num_u=num_u_t, num_v=num_v_t, num_w=num_w_t,
        den_u=den_u_t, den_v=den_v_t, den_w=den_w_t,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Warp runner helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_warp_output_arrays(sc: Dict, device: str) -> Dict[str, wp.array]:
    """Allocate Warp output arrays matching the scene's grid shape."""
    N = sc["Ngx"] * sc["Ngy"] * sc["Ngz"]
    out = {}
    for name in ("sdf_cc", "sdf_u", "sdf_v", "sdf_w"):
        out[name] = wp.full(N, 1e4, dtype=wp.float32, device=device)
    for name in ("body_u", "body_v", "body_w"):
        out[name] = wp.zeros(N, dtype=wp.float32, device=device)
    return out


def _reset_warp_outputs(out: Dict[str, wp.array]):
    for name in ("sdf_cc", "sdf_u", "sdf_v", "sdf_w"):
        out[name].fill_(1e4)
    for name in ("body_u", "body_v", "body_w"):
        out[name].zero_()


def setup_warp_runner(sc: Dict, device: str = "cuda:0") -> Tuple[WarpStreamingSDF, Dict]:
    wsdf = WarpStreamingSDF(sc["Ngx"], sc["Ngy"], sc["Ngz"], device=device)
    wsdf.setup(
        sc["F_flat"], sc["F_offsets"],
        sc["body_shapes"], sc["body_meta"],
        sc["gx"], sc["gy"], sc["gz"],
        float(sc["h"]), int(sc["max_vol"]),
    )
    wsdf.update_kinematics(sc["kin"], sc["aabb_lo"], sc["aabb_dim"])
    out = _make_warp_output_arrays(sc, device)
    return wsdf, out


# ─────────────────────────────────────────────────────────────────────────────
#  Timing utility
# ─────────────────────────────────────────────────────────────────────────────

def time_ms(fn, warmup: int = 3, reps: int = 20) -> float:
    """Return mean ms/call with GPU synchronisation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
#  Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(grids: List[int], bodies: List[int], device: str = "cuda:0"):
    print(f"\n{'─'*80}")
    print(f"  Warp streaming-SDF benchmark  (device={device})")
    print(f"  Sequential per-body Warp kernel (B launches) vs fanned all-body design")
    print(f"{'─'*80}")

    for N in grids:
        Ngx, Ngy, Ngz = N, N // 2, N // 2
        for B in bodies:
            sc = make_synthetic_scene(Ngx, Ngy, Ngz, B, device=device)
            grid_str = f"{Ngx}×{Ngy}×{Ngz}"

            # ── Warp eager ─────────────────────────────────────────────────
            wsdf, wp_out = setup_warp_runner(sc, device)

            def _warp_eager_kernel_only():
                wsdf.run_eager(
                    wp_out["sdf_cc"], wp_out["sdf_u"],
                    wp_out["sdf_v"],  wp_out["sdf_w"],
                    wp_out["body_u"], wp_out["body_v"], wp_out["body_w"],
                )

            def _warp_eager_with_reset():
                _reset_warp_outputs(wp_out)
                _warp_eager_kernel_only()

            # Warmup + JIT compile
            _warp_eager_with_reset(); _warp_eager_with_reset()
            t_eager       = time_ms(_warp_eager_kernel_only)
            t_eager_reset = time_ms(_warp_eager_with_reset)

            # ── Warp + CUDA graph ──────────────────────────────────────────
            _reset_warp_outputs(wp_out)
            wsdf.capture_graph(
                wp_out["sdf_cc"], wp_out["sdf_u"],
                wp_out["sdf_v"],  wp_out["sdf_w"],
                wp_out["body_u"], wp_out["body_v"], wp_out["body_w"],
            )

            def _warp_graph_kernel_only():
                wsdf.run_graph()

            def _warp_graph_with_reset():
                _reset_warp_outputs(wp_out)
                wsdf.run_graph()

            t_graph       = time_ms(_warp_graph_kernel_only)
            t_graph_reset = time_ms(_warp_graph_with_reset)

            # ── Warp FANNED (2 launches, constant in B) + CUDA graph ────────
            _reset_warp_outputs(wp_out)
            wsdf.run_fanned_eager(  # warmup/JIT
                wp_out["sdf_cc"], wp_out["sdf_u"],
                wp_out["sdf_v"],  wp_out["sdf_w"],
                wp_out["body_u"], wp_out["body_v"], wp_out["body_w"],
            )
            wsdf.capture_graph_fanned(
                wp_out["sdf_cc"], wp_out["sdf_u"],
                wp_out["sdf_v"],  wp_out["sdf_w"],
                wp_out["body_u"], wp_out["body_v"], wp_out["body_w"],
            )

            def _warp_fan_kernel_only():
                wsdf.run_graph_fanned()

            def _warp_fan_with_reset():
                _reset_warp_outputs(wp_out)
                wsdf.run_graph_fanned()

            t_fan       = time_ms(_warp_fan_kernel_only)
            t_fan_reset = time_ms(_warp_fan_with_reset)

            # ── Print rows ─────────────────────────────────────────────────
            def ms(v): return f"{v:.3f} ms"

            print(f"  {grid_str:>16}  {B:>6}  kernel-only: "
                  f"seq-graph={ms(t_graph)}  fan-graph={ms(t_fan)}  "
                  f"[seq-eager={ms(t_eager)}]")
            print(f"  {grid_str:>16}  {B:>6}  with-reset:  "
                  f"seq-graph={ms(t_graph_reset)}  fan-graph={ms(t_fan_reset)}  "
                  f"[seq-eager={ms(t_eager_reset)}]")
            print()

    print(f"{'─'*80}\n")
    print("Notes:")
    print("  kernel-only : pure kernel time (outputs pre-reset outside the timer)")
    print("  with-reset  : includes the required pre-step output reset (7 fill/zero ops)")
    print("  seq-graph   : sequential per-body Warp (B launches), CUDA-graph replay")
    print("  fan-graph   : fanned all-body Warp (2 launches, const in B), CUDA-graph replay\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp streaming-SDF benchmark")
    parser.add_argument("--grids",  type=int, nargs="+", default=[64, 128])
    parser.add_argument("--bodies", type=int, nargs="+", default=[3, 9])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_benchmark(args.grids, args.bodies, device=args.device)
