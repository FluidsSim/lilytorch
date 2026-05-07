"""Thin Python wrappers around ``torch.ops.lilytorch_kernels.*``.

Mirrors the call signatures of the original ``pytorch_interpolation``
ops so that callers can switch their imports with a one-line change:

    from lilytorch.src.kernels import streaming_sdf_min_rho_3d_multi
"""
import torch
from torch import Tensor

__all__ = [
    "streaming_sdf_min_rho_3d_multi",
    "streaming_sdf_forces_post_3d",
    "apply_bcs_3d",
    "streaming_sdf_min_rho_2d_multi",
    "streaming_sdf_forces_post_2d",
    "apply_bcs_2d",
    "interp_2d",
    "interp_3d",
]
_METHOD_MAP = {"linear": 0, "quadratic": 1}


def streaming_sdf_min_rho_3d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        interp_method: int,
        rho_bodies: Tensor, winning_rho_cc: Tensor) -> None:
    """3-D memory-saving Phase C update with winning-body density.

    Updates the union SDF / face velocities and stamps ``winning_rho_cc``
    without materializing per-body CC SDF slabs and without computing forces.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_min_rho_3d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        int(interp_method), rho_bodies, winning_rho_cc,
    )


def streaming_sdf_forces_post_3d(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor,
        interp_method: int,
        u: Tensor, v: Tensor, w: Tensor, p: Tensor,
        nu_rho_field: Tensor,
        eps_body: float, eps_solver: float, h3: float,
        delta_order: int,
        out: Tensor) -> None:
    """3-D Phase D only: current-union-normal force integration.

    Reuses the streamed body metadata and current union SDF produced by the
    update stage, but consumes the current post-fluid-step fields.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_post_3d.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, int(interp_method),
        u, v, w, p, nu_rho_field,
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
    """Multi-body memory-saving 2-D Phase C update with winning-body density.

    This is the memory-saving 2-D update path: it
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

    Reuses the streamed body metadata and current union SDF produced by
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
