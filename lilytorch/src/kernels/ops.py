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
    "streaming_sdf_forces_fused_3d_multi",
    "streaming_sdf_min_rho_3d_multi",
    "streaming_sdf_forces_post_3d",
    "apply_bcs_3d",
    "streaming_sdf_min_2d",
    "streaming_sdf_min_2d_multi",
    "streaming_sdf_min_rho_2d_multi",
    "bdim_forces_2d_multi",
    "streaming_sdf_forces_fused_2d_multi",
    "streaming_sdf_forces_post_2d",
    "apply_bcs_2d",
    "interp_2d",
    "interp_3d",
]

_METHOD_MAP = {"linear": 0, "quadratic": 1}


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
        delta_order: int = 1,
        out: Tensor = None) -> None:
    """Phase D: per-body force / torque integration.

    Reads the per-body cell-centred SDF cached in ``sparse_cc_flat``
    (populated by :func:`streaming_sdf_min_3d_multi`) instead of
    re-sampling it via trilinear interpolation. ``cell_offsets[b]`` is
    the start index of body ``b``'s AABB-local cc-SDF slab in
    ``sparse_cc_flat``.

    ``delta_order`` selects the smoothed-delta order:
      * ``1`` -- standard 1st-order cosine delta (default);
      * ``2`` -- Towers (2008) 2nd-order: divide by |∇φ|, correcting
        for non-unit SDF gradients on numerical grids.
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
        int(delta_order),
        out,
    )


def streaming_sdf_forces_fused_3d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float,
        max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        interp_method: int,
        rho_bodies: Tensor,
        winning_rho_cc: Tensor,
        u_prev: Tensor, v_prev: Tensor, w_prev: Tensor, p_prev: Tensor,
        nx_cc: Tensor, ny_cc: Tensor, nz_cc: Tensor,
        nu_rho_field: Tensor,
        eps_body: float, eps_solver: float, h3: float,
        delta_order: int,
        out: Tensor) -> None:
    """Fused Phase C+D: per-body AABB SDF update + inline lagged force integration.

    Replaces :func:`streaming_sdf_min_3d_multi` + :func:`bdim_forces_3d_multi`
    in a single kernel pass per body.  Memory savings:

    * No ``sparse_cc_flat`` (per-body CC-SDF cache).
    * No union-AABB stress / pressure-force tensors.

    ``nu_rho_field``: ν·ρ tensor, either size=1 (constant viscosity) or
    size equal to the full grid (Smagorinsky / Carreau variable viscosity).

    ``winning_rho_cc`` is updated in-place: when body ``b`` wins the SDF
    minimum at a grid cell, that cell is stamped with ``rho_bodies[b]``.
    Pre-fill with ``rho_fluid`` before calling.

    ``out`` (B, 12) float64 accumulates force/torque; pre-zero before calling.

    Forces are one time-step lagged (computed from beginning-of-step
    u/v/w/p and previous-step normals).

    ``delta_order``: 1 = cosine delta, 2 = Towers (2008) |∇φ| correction.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_fused_3d_multi.default(
        F_flat, F_offsets,
        body_shapes, body_meta, kin,
        aabb_lo, aabb_dim,
        gx, gy, gz,
        float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        int(interp_method),
        rho_bodies, winning_rho_cc,
        u_prev, v_prev, w_prev, p_prev,
        nx_cc, ny_cc, nz_cc,
        nu_rho_field,
        float(eps_body), float(eps_solver), float(h3),
        int(delta_order),
        out,
    )


def streaming_sdf_min_rho_3d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        interp_method: int,
        rho_bodies: Tensor, winning_rho_cc: Tensor,
        u_prev: Tensor, v_prev: Tensor, w_prev: Tensor, p_prev: Tensor,
        nx_cc: Tensor, ny_cc: Tensor, nz_cc: Tensor,
        nu_rho_field: Tensor,
        eps_body: float, eps_solver: float, h3: float,
        delta_order: int,
        out: Tensor) -> None:
    """3-D kernel-mode Phase C update.

    Uses the native streamed implementation to update union geometry and
    winning density without storing Python-visible per-body CC SDF slabs.
    The native implementation shares the streamed sampler with the post
    force op; the force accumulator is caller-owned and ignored by the
    kernel update path.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_fused_3d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        int(interp_method), rho_bodies, winning_rho_cc,
        u_prev, v_prev, w_prev, p_prev, nx_cc, ny_cc, nz_cc, nu_rho_field,
        float(eps_body), float(eps_solver), float(h3), int(delta_order), out,
    )


def streaming_sdf_forces_post_3d(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        interp_method: int,
        rho_bodies: Tensor, winning_rho_cc: Tensor,
        u: Tensor, v: Tensor, w: Tensor, p: Tensor,
        nx_cc: Tensor, ny_cc: Tensor, nz_cc: Tensor,
        nu_rho_field: Tensor,
        eps_body: float, eps_solver: float, h3: float,
        delta_order: int,
        out: Tensor) -> None:
    """3-D kernel-mode Phase D force pass after the fluid step.

    Forces are computed natively from current fluid fields and current union
    normals.  Per-body deltas are obtained by on-demand body-local SDF
    sampling from the packed body metadata; no per-body CC SDF slabs are
    required by the Python kernel path.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_fused_3d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        int(interp_method), rho_bodies, winning_rho_cc,
        u, v, w, p, nx_cc, ny_cc, nz_cc, nu_rho_field,
        float(eps_body), float(eps_solver), float(h3), int(delta_order), out,
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


def streaming_sdf_min_rho_2d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor,
        body_meta: Tensor,
        kin: Tensor,
        aabb_lo: Tensor,
        aabb_dim: Tensor,
        gx: Tensor, gy: Tensor,
        h_grid: float,
        max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        interp_method: int,
        rho_bodies: Tensor,
        winning_rho_cc: Tensor) -> None:
    """Multi-body fused 2-D Phase C update with winning-body density.

    This is the memory-saving update half of the fused 2-D path: it
    updates the union SDF / face velocities and stamps ``winning_rho_cc``
    without materializing ``sparse_cc_flat`` and without computing forces.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_min_rho_2d_multi.default(
        F_flat, F_offsets,
        body_shapes, body_meta, kin,
        aabb_lo, aabb_dim,
        gx, gy, float(h_grid),
        int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v,
        body_u, body_v,
        int(interp_method),
        rho_bodies,
        winning_rho_cc,
    )


def bdim_forces_2d_multi(
        sparse_cc_flat: Tensor, cell_offsets: Tensor,
        kin: Tensor,
        aabb_lo: Tensor,
        aabb_dim: Tensor,
        gx: Tensor, gy: Tensor,
        u_i0: int, u_j0: int,
        Sj: int,
        xs: Tensor, ys: Tensor,
        px: Tensor, py: Tensor,
        eps_body: float, eps_solver: float, h2: float,
        max_vol_per_body: int,
        delta_order: int = 1,
        out: Tensor = None) -> None:
    """2-D analogue of :func:`bdim_forces_3d_multi`.

    Per-body force/torque integration over the AABB.  Reads the
    per-body cell-centred SDF cached in ``sparse_cc_flat`` (populated
    by :func:`streaming_sdf_min_2d_multi`) instead of re-sampling it.

    ``out`` is float64 with 6 channels per body:

        ``[fv_x, fv_y, t_v, fp_x, fp_y, t_p]``

    where ``t_v`` and ``t_p`` are the scalar out-of-plane torques
    ``arm_x*f_y - arm_y*f_x``.

    ``delta_order`` selects the smoothed-delta order (1 or 2); see
    :func:`bdim_forces_3d_multi` for details.
    """
    return torch.ops.lilytorch_kernels.bdim_forces_2d_multi.default(
        sparse_cc_flat, cell_offsets,
        kin,
        aabb_lo, aabb_dim,
        gx, gy,
        int(u_i0), int(u_j0),
        int(Sj),
        xs, ys, px, py,
        float(eps_body), float(eps_solver), float(h2),
        int(max_vol_per_body),
        int(delta_order),
        out,
    )


def streaming_sdf_forces_fused_2d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        interp_method: int,
        rho_bodies: Tensor, winning_rho_cc: Tensor,
        u_prev: Tensor, v_prev: Tensor, p_prev: Tensor,
        nx_cc: Tensor, ny_cc: Tensor,
        nu_rho_field: Tensor,
        eps_body: float, eps_solver: float, h2: float,
        delta_order: int,
        out: Tensor) -> None:
    """Fused 2D Phase C+D: SDF update + current-normal force integration.
    2D analogue of streaming_sdf_forces_fused_3d_multi.
    The ``nx_cc`` / ``ny_cc`` arguments are kept for ABI compatibility;
    the 2-D implementation recomputes force normals from the decoded
    current union SDF inside the op.
    out (B, 6) float64: [fv_x, fv_y, t_v, fp_x, fp_y, t_p]
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_fused_2d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, body_u, body_v, int(interp_method),
        rho_bodies, winning_rho_cc,
        u_prev, v_prev, p_prev, nx_cc, ny_cc, nu_rho_field,
        float(eps_body), float(eps_solver), float(h2), int(delta_order), out,
    )


def streaming_sdf_forces_post_2d(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor,
        interp_method: int,
        u_prev: Tensor, v_prev: Tensor, p_prev: Tensor,
        nu_rho_field: Tensor,
        eps_body: float, eps_solver: float, h2: float,
        delta_order: int,
        out: Tensor) -> None:
    """2-D Phase D only: current-union-normal force integration.

    Reuses the fused-path body metadata and current union SDF produced by
    the update stage, but consumes the current post-fluid-step fields.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_post_2d.default(
        F_flat, F_offsets,
        body_shapes, body_meta, kin,
        aabb_lo, aabb_dim,
        gx, gy, float(h_grid), int(max_vol_per_body),
        sdf_cc,
        int(interp_method),
        u_prev, v_prev, p_prev,
        nu_rho_field,
        float(eps_body), float(eps_solver), float(h2), int(delta_order),
        out,
    )


def interp_2d(
        F: Tensor,
        xq: Tensor, yq: Tensor,
        bx0: float, by0: float,
        inv_dx: float, inv_dy: float,
        Mx: int, My: int,
        method: str = "linear") -> Tensor:
    """Bilinear (method='linear') or biquadratic (method='quadratic')
    scattered-point interpolation on a uniform 2-D grid.

    Parameters
    ----------
    F : Tensor
        Grid values, shape ``(Mx, My)``, contiguous float32/float64.
    xq, yq : Tensor
        Query-point coordinates, each shape ``(N,)``.
    bx0, by0 : float
        Grid origin (first axis coordinate).
    inv_dx, inv_dy : float
        Inverse grid spacing per axis.
    Mx, My : int
        Grid size per axis.
    method : "linear" | "quadratic"
        Interpolation order.  Default "linear".

    Returns
    -------
    Tensor
        Interpolated values, shape ``(N,)``, same dtype/device as ``F``.
    """
    interp_method = _METHOD_MAP.get(method)
    if interp_method is None:
        raise ValueError(f"method must be 'linear' or 'quadratic', got {method!r}")
    G = torch.empty(xq.numel(), dtype=F.dtype, device=F.device)
    torch.ops.lilytorch_kernels.interpolate_2d.default(
        F, xq, yq,
        float(bx0), float(by0),
        float(inv_dx), float(inv_dy),
        int(Mx), int(My),
        int(interp_method),
        G,
    )
    return G


def interp_3d(
        F: Tensor,
        xq: Tensor, yq: Tensor, zq: Tensor,
        bx0: float, by0: float, bz0: float,
        inv_dx: float, inv_dy: float, inv_dz: float,
        Mx: int, My: int, Mz: int,
        method: str = "linear") -> Tensor:
    """Trilinear (method='linear') or triquadratic (method='quadratic')
    scattered-point interpolation on a uniform 3-D grid.

    Parameters
    ----------
    F : Tensor
        Grid values, shape ``(Mx, My, Mz)``, contiguous float32/float64.
    xq, yq, zq : Tensor
        Query-point coordinates, each shape ``(N,)``.
    bx0, by0, bz0 : float
        Grid origin per axis.
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing per axis.
    Mx, My, Mz : int
        Grid size per axis.
    method : "linear" | "quadratic"
        Interpolation order.  Default "linear".

    Returns
    -------
    Tensor
        Interpolated values, shape ``(N,)``, same dtype/device as ``F``.
    """
    interp_method = _METHOD_MAP.get(method)
    if interp_method is None:
        raise ValueError(f"method must be 'linear' or 'quadratic', got {method!r}")
    G = torch.empty(xq.numel(), dtype=F.dtype, device=F.device)
    torch.ops.lilytorch_kernels.interpolate_3d.default(
        F, xq, yq, zq,
        float(bx0), float(by0), float(bz0),
        float(inv_dx), float(inv_dy), float(inv_dz),
        int(Mx), int(My), int(Mz),
        int(interp_method),
        G,
    )
    return G


def apply_bcs_2d(
        u: Tensor, v: Tensor,
        shapes: Tensor,
        neu_desc: Tensor,
        dir_desc: Tensor,
        dir_val: Tensor,
        max_line_dim: int) -> None:
    """Fused 2-D boundary-condition writes (Neumann + Dirichlet).

    Mutates ``u`` and ``v`` in place.  See :func:`apply_bcs_3d` for the
    descriptor layout; here ``shapes`` is int64 ``[2, 2]`` with rows
    (Nx, Ny) for u and v, and ``axis`` is restricted to 0 (x) or 1 (y).
    """
    return torch.ops.lilytorch_kernels.apply_bcs_2d.default(
        u, v,
        shapes, neu_desc, dir_desc, dir_val,
        int(max_line_dim),
    )
