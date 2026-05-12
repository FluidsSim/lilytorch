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
    "rbgs_sweep_2d",
    "rbgs_sweep_3d",
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
        rho_bodies: Tensor, winning_rho_cc: Tensor,
        dirty_i0: int, dirty_j0: int, dirty_k0: int,
        dirty_Ai: int, dirty_Aj: int, dirty_Ak: int) -> None:
    """3-D memory-saving Phase C update with winning-body density.

    Updates the union SDF / face velocities and stamps ``winning_rho_cc``
    without materializing per-body CC SDF slabs and without computing forces.

    ``dirty_i0/j0/k0`` + ``dirty_Ai/Aj/Ak`` define the dirty sub-block (union
    of previous and current union-AABB).  The init/decode kernels only touch
    this region, reducing them from O(Nx*Ny*Nz) to O(dirty_vol).
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_min_rho_3d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        int(interp_method), rho_bodies, winning_rho_cc,
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
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
        max_dim0: int,
        max_dim1: int) -> None:
    """Phase H: fused 3-D boundary-condition writes (Neumann + Dirichlet).

    Mutates ``u``, ``v``, ``w`` in place.
    Uses a rectangular (max_dim0 × max_dim1) CUDA thread-block grid so that
    non-square faces (e.g. Nx × Nz with Nx >> Nz) do not waste thread blocks.
    """
    return torch.ops.lilytorch_kernels.apply_bcs_3d.default(
        u, v, w,
        shapes, neu_desc, dir_desc, dir_val,
        int(max_dim0), int(max_dim1),
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
        winning_rho_cc: Tensor,
        dirty_i0: int, dirty_j0: int,
        dirty_Ai: int, dirty_Aj: int) -> None:
    """Multi-body memory-saving 2-D Phase C update with winning-body density.

    This is the memory-saving 2-D update path: it
    updates the union SDF / face velocities and stamps ``winning_rho_cc``
    without materializing ``sparse_cc_flat`` and without computing forces.
    The ``dirty_i0, dirty_j0, dirty_Ai, dirty_Aj`` parameters define the
    dirty sub-block (union of previous and current body union-AABBs) so
    that the init / decode kernel passes only touch O(dirty_area) cells
    instead of the full O(Nx*Ny) grid.
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
        int(dirty_i0), int(dirty_j0),
        int(dirty_Ai), int(dirty_Aj),
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


# =====================================================================
# Multigrid RBGS smoother kernels
# =====================================================================

def rbgs_sweep_2d(
        p: Tensor,
        f: Tensor,
        cp0: Tensor, cm0: Tensor,
        cp1: Tensor, cm1: Tensor,
        jcap_tol: float,
        nsmoothing: int) -> None:
    """Tiled 2-D RBGS smoother: ``nsmoothing`` full sweeps in one CUDA call.

    Mutates ``p`` (shape ``(Nx+2, Ny+2)``) in place.  Applies Neumann BCs
    on ghost rows before and after the sweeps.  ``f``, ``cp0``, ``cm0``,
    ``cp1``, ``cm1`` are the interior-cell RHS and face coefficients, all
    of shape ``(Nx, Ny)``.
    """
    # Coefficient arrays may arrive as non-contiguous slices (e.g. strides
    # from a ghost-padded grid).  The CUDA kernel assumes C-contiguous layout,
    # so materialise them here before dispatch.
    torch.ops.lilytorch_kernels.rbgs_sweep_2d.default(
        p, f,
        cp0.contiguous(), cm0.contiguous(),
        cp1.contiguous(), cm1.contiguous(),
        float(jcap_tol), int(nsmoothing),
    )


def rbgs_sweep_3d(
        p: Tensor,
        f: Tensor,
        cp0: Tensor, cm0: Tensor,
        cp1: Tensor, cm1: Tensor,
        cp2: Tensor, cm2: Tensor,
        jcap_tol: float,
        nsmoothing: int) -> None:
    """Thread-per-cell 3-D RBGS smoother: ``nsmoothing`` full sweeps.

    Mutates ``p`` (shape ``(Nx+2, Ny+2, Nz+2)``) in place.  Applies
    Neumann BCs between every half-sweep.  ``f``, ``cp{0,1,2}``,
    ``cm{0,1,2}`` are interior-cell arrays of shape ``(Nx, Ny, Nz)``.
    """
    # Same non-contiguity guard as the 2-D wrapper.
    torch.ops.lilytorch_kernels.rbgs_sweep_3d.default(
        p, f,
        cp0.contiguous(), cm0.contiguous(),
        cp1.contiguous(), cm1.contiguous(),
        cp2.contiguous(), cm2.contiguous(),
        float(jcap_tol), int(nsmoothing),
    )


# ── FakeTensor (abstract) implementations for torch.compile ──────────
# These let torch.compile / TorchInductor trace through the custom ops
# without executing CUDA.  Both ops return () and mutate p in place, so
# the abstract implementation is a no-op.

@torch.library.register_fake("lilytorch_kernels::rbgs_sweep_2d")
def _rbgs_sweep_2d_abstract(p, f, cp0, cm0, cp1, cm1,
                              jcap_tol, nsmoothing):
    pass   # p is mutated in place; no new tensors created


@torch.library.register_fake("lilytorch_kernels::rbgs_sweep_3d")
def _rbgs_sweep_3d_abstract(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                              jcap_tol, nsmoothing):
    pass   # p is mutated in place; no new tensors created


def jacobi_sweep_2d(
        p: Tensor,
        f: Tensor,
        cp0: Tensor, cm0: Tensor,
        cp1: Tensor, cm1: Tensor,
        jcap_tol: float,
        w: float,
        nsmoothing: int) -> None:
    """Tiled 2-D weighted Jacobi smoother: ``nsmoothing`` sweeps in one CUDA call.

    Mutates ``p`` (shape ``(Nx+2, Ny+2)``) in place.  Applies Neumann BCs
    on ghost rows before and after the sweeps.  ``f``, ``cp0``, ``cm0``,
    ``cp1``, ``cm1`` are the interior-cell RHS and face coefficients, all
    of shape ``(Nx, Ny)``.  ``w`` is the relaxation weight (1.0 = plain Jacobi).
    """
    torch.ops.lilytorch_kernels.jacobi_sweep_2d.default(
        p, f,
        cp0.contiguous(), cm0.contiguous(),
        cp1.contiguous(), cm1.contiguous(),
        float(jcap_tol), float(w), int(nsmoothing),
    )


def jacobi_sweep_3d(
        p: Tensor,
        f: Tensor,
        cp0: Tensor, cm0: Tensor,
        cp1: Tensor, cm1: Tensor,
        cp2: Tensor, cm2: Tensor,
        jcap_tol: float,
        w: float,
        nsmoothing: int) -> None:
    """Double-buffer 3-D weighted Jacobi smoother: ``nsmoothing`` sweeps.

    Mutates ``p`` (shape ``(Nx+2, Ny+2, Nz+2)``) in place.  Uses an
    internal ping-pong buffer so every read sees the previous iteration's
    values (true synchronous Jacobi).
    """
    torch.ops.lilytorch_kernels.jacobi_sweep_3d.default(
        p, f,
        cp0.contiguous(), cm0.contiguous(),
        cp1.contiguous(), cm1.contiguous(),
        cp2.contiguous(), cm2.contiguous(),
        float(jcap_tol), float(w), int(nsmoothing),
    )


@torch.library.register_fake("lilytorch_kernels::jacobi_sweep_2d")
def _jacobi_sweep_2d_abstract(p, f, cp0, cm0, cp1, cm1,
                                jcap_tol, w, nsmoothing):
    pass   # p is mutated in place; no new tensors created


@torch.library.register_fake("lilytorch_kernels::jacobi_sweep_3d")
def _jacobi_sweep_3d_abstract(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                                jcap_tol, w, nsmoothing):
    pass   # p is mutated in place; no new tensors created
