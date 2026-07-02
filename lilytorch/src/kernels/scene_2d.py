"""Synthetic 2-D scene builder shared by the 2-D Warp parity tests/benches.

Builds all tensors needed by the native `body_update_2d` /
`bdim_forcing_2d` ops AND by `warp_kernels_2d.WarpStreamingSDF2D`, using disc
(circle) SDF tables on a uniform body grid — no FARMS/MuJoCo dependency.

2-D packed layouts (match `streaming_sdf_2d.cu`):
  body_shapes : [B,2]  = (Mx, My)
  body_meta   : [B,7]  = (bx0, by0, bx_last, by_last, inv_dx, inv_dy, inv_vol)
  kin         : [B,11] = R_T(2x2 row-major,4) + bp(2) + cm(2) + lv(2) + om(1)
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict

FAR = 1e4


def _random_rotation_2d(rng, dtype=np.float32):
    th = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=dtype)


def make_synthetic_scene_2d(
    Ngx: int, Ngy: int, B: int,
    body_Mx: int = 24, body_My: int = 20,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    seed: int = 7,
    blend: bool = False,
    overlap: bool = False,
) -> Dict:
    rng = np.random.default_rng(seed)
    h = 0.02
    gx = torch.linspace(0.0, (Ngx - 1) * h, Ngx, dtype=dtype, device=device)
    gy = torch.linspace(0.0, (Ngy - 1) * h, Ngy, dtype=dtype, device=device)

    body_vol = body_Mx * body_My
    F_flat_np = np.empty(B * body_vol, dtype=np.float32)
    F_offsets_np = np.arange(B + 1, dtype=np.int64) * body_vol

    half_bx = (body_Mx - 1) * 0.01 * 0.5
    half_by = (body_My - 1) * 0.01 * 0.5
    bx_np = np.linspace(-half_bx, half_bx, body_Mx, dtype=np.float32)
    by_np = np.linspace(-half_by, half_by, body_My, dtype=np.float32)
    Bx, By = np.meshgrid(bx_np, by_np, indexing="ij")
    disc_sdf = np.sqrt(Bx**2 + By**2) - 0.07
    for b in range(B):
        F_flat_np[F_offsets_np[b]:F_offsets_np[b] + body_vol] = disc_sdf.ravel()

    body_shapes_np = np.tile([body_Mx, body_My], B).reshape(B, 2).astype(np.int64)

    inv_dx = 1.0 / (bx_np[1] - bx_np[0])
    inv_dy = 1.0 / (by_np[1] - by_np[0])
    inv_vol = inv_dx * inv_dy
    meta_row = [bx_np[0], by_np[0], bx_np[-1], by_np[-1], inv_dx, inv_dy, inv_vol]
    body_meta_np = np.tile(meta_row, B).reshape(B, 7).astype(np.float32)

    domain = np.array([Ngx - 1, Ngy - 1], dtype=np.float32) * h
    kin_rows, aabb_lo_rows, aabb_dim_rows = [], [], []
    aabb_half = np.array([int(Ngx // B * (1.2 if overlap else 0.8)),
                          int(Ngy * 0.7)], dtype=np.int64)
    for b in range(B):
        R = _random_rotation_2d(rng)
        bp = rng.uniform(0.15, 0.85, 2).astype(np.float32) * domain
        cm = bp + rng.uniform(-0.01, 0.01, 2).astype(np.float32)
        lv = rng.uniform(-0.2, 0.2, 2).astype(np.float32)
        om = rng.uniform(-1.0, 1.0, 1).astype(np.float32)
        kin_rows.append(np.concatenate([R.ravel(), bp, cm, lv, om]))

        i0 = int(b * (Ngx // B))
        j0 = 0
        Ai = min(int(aabb_half[0]), Ngx - i0)
        Aj = min(int(aabb_half[1]), Ngy - j0)
        aabb_lo_rows.append([i0, j0])
        aabb_dim_rows.append([Ai, Aj])

    kin_np      = np.array(kin_rows, dtype=np.float32)
    aabb_lo_np  = np.array(aabb_lo_rows, dtype=np.int64)
    aabb_dim_np = np.array(aabb_dim_rows, dtype=np.int64)
    max_vol = int(np.prod(aabb_dim_np, axis=1).max())

    i0_min = int(aabb_lo_np[:, 0].min()); j0_min = int(aabb_lo_np[:, 1].min())
    i1_max = int((aabb_lo_np[:, 0] + aabb_dim_np[:, 0]).max())
    j1_max = int((aabb_lo_np[:, 1] + aabb_dim_np[:, 1]).max())
    dirty_bounds = (i0_min, j0_min, i1_max - i0_min, j1_max - j0_min)

    def t(arr, dt=None):
        return torch.from_numpy(arr).to(device=device, dtype=dt)

    N = Ngx * Ngy
    nblend = N if blend else 1
    return dict(
        Ngx=Ngx, Ngy=Ngy, B=B, h=h, max_vol=max_vol,
        dirty_bounds=dirty_bounds,
        blend_eps=(0.5 * h if blend else 0.0),
        gx=gx, gy=gy,
        F_flat=t(F_flat_np), F_offsets=t(F_offsets_np[:B]),
        body_shapes=t(body_shapes_np), body_meta=t(body_meta_np),
        kin=t(kin_np), aabb_lo=t(aabb_lo_np), aabb_dim=t(aabb_dim_np),
        # native output buffers (flat, viewed as (Ngx,Ngy) at call site)
        sdf_cc=torch.full((N,), FAR, dtype=dtype, device=device),
        sdf_u=torch.full((N,), FAR, dtype=dtype, device=device),
        sdf_v=torch.full((N,), FAR, dtype=dtype, device=device),
        body_u=torch.zeros(N, dtype=dtype, device=device),
        body_v=torch.zeros(N, dtype=dtype, device=device),
        key_cc=torch.empty(N, dtype=torch.int64, device=device),
        key_u=torch.empty(N, dtype=torch.int64, device=device),
        key_v=torch.empty(N, dtype=torch.int64, device=device),
        num_u=torch.zeros(nblend, dtype=dtype, device=device),
        num_v=torch.zeros(nblend, dtype=dtype, device=device),
        den_u=torch.zeros(nblend, dtype=dtype, device=device),
        den_v=torch.zeros(nblend, dtype=dtype, device=device),
    )
