"""Thin Python wrappers around ``torch.ops.lilytorch_kernels.*``.

Mirrors the call signatures of the original ``pytorch_interpolation``
ops so that callers can switch their imports with a one-line change:

    from pytorch_interpolation import streaming_sdf_min_3d_multi
    # becomes:
    from lilytorch.src.kernels import streaming_sdf_min_3d_multi
"""
import torch
from torch import Tensor

__all__ = [
    "streaming_sdf_min_3d",
    "streaming_sdf_min_3d_multi",
    "bdim_forces_3d_multi",
    "apply_bcs_3d",
]


def streaming_sdf_min_3d(
        F: Tensor, bx: Tensor, by: Tensor, bz: Tensor,
        bx0: float, by0: float, bz0: float,
        bx_last: float, by_last: float, bz_last: float,
        inv_dx: float, inv_dy: float, inv_dz: float, inv_vol: float,
        R_T,
        body_pos,
        com_pos,
        lin_vel,
        ang_vel,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float,
        i0: int, i1: int, j0: int, j1: int, k0: int, k1: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        sparse_cc: Tensor) -> None:
    """One-body fused SDF / face-velocity running-min update on a fluid
    grid AABB.  See ``csrc/cuda/streaming_sdf.cu`` for kernel details.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_min_3d.default(
        F, bx, by, bz,
        float(bx0), float(by0), float(bz0),
        float(bx_last), float(by_last), float(bz_last),
        float(inv_dx), float(inv_dy), float(inv_dz), float(inv_vol),
        list(R_T), list(body_pos), list(com_pos),
        list(lin_vel), list(ang_vel),
        gx, gy, gz, float(h_grid),
        int(i0), int(i1), int(j0), int(j1), int(k0), int(k1),
        sdf_cc, sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        sparse_cc,
    )


def streaming_sdf_min_3d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        bx_flat: Tensor, bx_offsets: Tensor,
        by_flat: Tensor, by_offsets: Tensor,
        bz_flat: Tensor, bz_offsets: Tensor,
        body_shapes: Tensor,
        body_meta: Tensor,
        kin: Tensor,
        aabb_lo: Tensor,
        aabb_dim: Tensor,
        cell_offsets: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float,
        max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        sparse_cc_flat: Tensor) -> None:
    """Multi-body fused SDF / face-velocity running-min update."""
    return torch.ops.lilytorch_kernels.streaming_sdf_min_3d_multi.default(
        F_flat, F_offsets,
        bx_flat, bx_offsets,
        by_flat, by_offsets,
        bz_flat, bz_offsets,
        body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, cell_offsets,
        gx, gy, gz, float(h_grid),
        int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        sparse_cc_flat,
    )


def bdim_forces_3d_multi(
        sparse_cc_flat: Tensor, cell_offsets: Tensor,
        kin: Tensor,
        aabb_lo: Tensor,
        aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        u_i0: int, u_j0: int, u_k0: int,
        Sj: int, Sk: int,
        xs: Tensor, ys: Tensor, zs: Tensor,
        px: Tensor, py: Tensor, pz: Tensor,
        eps_body: float, eps_solver: float, h3: float,
        max_vol_per_body: int,
        out: Tensor) -> None:
    """Phase D: per-body force / torque integration.

    Reads the per-body cell-centred SDF cached in ``sparse_cc_flat``
    (populated by :func:`streaming_sdf_min_3d_multi`) instead of
    re-sampling it via trilinear interpolation. ``cell_offsets[b]`` is
    the start index of body ``b``'s AABB-local cc-SDF slab in
    ``sparse_cc_flat``.
    """
    return torch.ops.lilytorch_kernels.bdim_forces_3d_multi.default(
        sparse_cc_flat, cell_offsets,
        kin,
        aabb_lo, aabb_dim,
        gx, gy, gz,
        int(u_i0), int(u_j0), int(u_k0),
        int(Sj), int(Sk),
        xs, ys, zs, px, py, pz,
        float(eps_body), float(eps_solver), float(h3),
        int(max_vol_per_body),
        out,
    )


def apply_bcs_3d(
        u: Tensor, v: Tensor, w: Tensor,
        shapes: Tensor,
        neu_desc: Tensor,
        dir_desc: Tensor,
        dir_val: Tensor,
        max_plane_dim: int) -> None:
    """Phase H: fused 3-D boundary-condition writes (Neumann + Dirichlet).

    Mutates ``u``, ``v``, ``w`` in place.
    """
    return torch.ops.lilytorch_kernels.apply_bcs_3d.default(
        u, v, w,
        shapes, neu_desc, dir_desc, dir_val,
        int(max_plane_dim),
    )
