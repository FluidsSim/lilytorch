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
    "streaming_sdf_min_2d",
    "streaming_sdf_min_2d_multi",
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
        sparse_cc: Tensor,
        interp_method: int = 0) -> None:
    """One-body fused SDF / face-velocity running-min update on a fluid
    grid AABB.  See ``csrc/cuda/streaming_sdf.cu`` for kernel details.

    ``interp_method`` selects the body-SDF sampler:
      * ``0`` -- trilinear (default, matches the historical behaviour);
      * ``1`` -- triquadratic Lagrange (3x3x3 stencil, falls back to
        trilinear in the boundary layer of the body grid).
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
        int(interp_method),
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
        sparse_cc_flat: Tensor,
        interp_method: int = 0) -> None:
    """Multi-body fused SDF / face-velocity running-min update.

    ``interp_method`` selects the body-SDF sampler:
      * ``0`` -- trilinear (default);
      * ``1`` -- triquadratic Lagrange (3x3x3 stencil, falls back to
        trilinear in the boundary layer of the body grid).
    """
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
        int(interp_method),
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


# =====================================================================
#  2-D analogues of the streaming-SDF ops.  Same calling convention as
#  the 3-D versions with the z-axis stripped.  Rotation ``R_T`` is a
#  column-major 2x2 matrix (4 floats), angular velocity is the scalar
#  ``omega`` (out-of-plane), and the kernel writes 3 staggered-grid
#  outputs (cc, u-face, v-face) instead of 4.
# =====================================================================

def streaming_sdf_min_2d(
        F: Tensor, bx: Tensor, by: Tensor,
        bx0: float, by0: float,
        bx_last: float, by_last: float,
        inv_dx: float, inv_dy: float, inv_vol: float,
        R_T,
        body_pos,
        com_pos,
        lin_vel,
        omega: float,
        gx: Tensor, gy: Tensor,
        h_grid: float,
        i0: int, i1: int, j0: int, j1: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        sparse_cc: Tensor,
        interp_method: int = 0) -> None:
    """One-body fused 2-D SDF / face-velocity running-min update on a
    fluid grid AABB.  See ``csrc/cuda/streaming_sdf_2d.cu`` for kernel
    details.

    ``interp_method`` selects the body-SDF sampler:
      * ``0`` -- bilinear (default);
      * ``1`` -- biquadratic Lagrange (3x3 stencil, falls back to
        bilinear in the boundary layer of the body grid).
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_min_2d.default(
        F, bx, by,
        float(bx0), float(by0),
        float(bx_last), float(by_last),
        float(inv_dx), float(inv_dy), float(inv_vol),
        list(R_T), list(body_pos), list(com_pos),
        list(lin_vel), float(omega),
        gx, gy, float(h_grid),
        int(i0), int(i1), int(j0), int(j1),
        sdf_cc, sdf_u, sdf_v,
        body_u, body_v,
        sparse_cc,
        int(interp_method),
    )


def streaming_sdf_min_2d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        bx_flat: Tensor, bx_offsets: Tensor,
        by_flat: Tensor, by_offsets: Tensor,
        body_shapes: Tensor,
        body_meta: Tensor,
        kin: Tensor,
        aabb_lo: Tensor,
        aabb_dim: Tensor,
        cell_offsets: Tensor,
        gx: Tensor, gy: Tensor,
        h_grid: float,
        max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        sparse_cc_flat: Tensor,
        interp_method: int = 0) -> None:
    """Multi-body fused 2-D SDF / face-velocity running-min update.

    Per-body packed layouts:
      * ``body_shapes``  int64 [B,2]   (Mx, My)
      * ``body_meta``    float [B,7]   (bx0, by0, bxL, byL, inv_dx, inv_dy, inv_vol)
      * ``kin``          float [B,11]  (R_T[0..3], bp_x, bp_y, cm_x, cm_y, lv_x, lv_y, omega)
      * ``aabb_lo``      int64 [B,2]
      * ``aabb_dim``     int64 [B,2]
      * ``cell_offsets`` int64 [B+1]   prefix sum of Ai*Aj

    ``interp_method`` selects the body-SDF sampler:
      * ``0`` -- bilinear (default);
      * ``1`` -- biquadratic Lagrange (3x3 stencil, falls back to
        bilinear in the boundary layer of the body grid).
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_min_2d_multi.default(
        F_flat, F_offsets,
        bx_flat, bx_offsets,
        by_flat, by_offsets,
        body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, cell_offsets,
        gx, gy, float(h_grid),
        int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v,
        body_u, body_v,
        sparse_cc_flat,
        int(interp_method),
    )
