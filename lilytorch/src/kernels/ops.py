"""Thin Python wrappers around ``torch.ops.lilytorch_kernels.*``."""
import torch
from torch import Tensor

__all__ = [
    "streaming_sdf_stag_3d_multi",
    "bdim_coeff_3d",
    "bdim_coeff_sigma_3d",
    "streaming_sdf_forces_post_3d",
    "apply_bcs_3d",
    "streaming_sdf_stag_2d_multi",
    "bdim_coeff_2d",
    "bdim_coeff_sigma_2d",
    "streaming_sdf_forces_post_2d",
    "apply_bcs_2d",
    "interp_2d",
    "interp_3d",
    "lagrangian_forces_2d",
    "lagrangian_forces_3d",
    "rbgs_sweep_2d",
    "rbgs_sweep_3d",
    "mg_residual_2d",
    "mg_residual_3d",
    "restrict_residual_2d",
    "restrict_residual_3d",
    "restrict_face_2d",
    "restrict_face_3d",
    "prolongate_add_2d",
    "prolongate_add_3d",
    "poisson_solve_multigrid_2d",
    "poisson_solve_multigrid_3d",
    "poisson_solve_mgcg_2d",
    "poisson_solve_mgcg_3d",
    "poisson_solve_rmgcg_2d",
    "poisson_solve_rmgcg_3d",
]
_METHOD_MAP = {"linear": 0, "quadratic": 1}


def streaming_sdf_stag_3d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        key_cc_t: Tensor, key_u_t: Tensor, key_v_t: Tensor, key_w_t: Tensor,
        interp_method: int,
        dirty_i0: int, dirty_j0: int, dirty_k0: int,
        dirty_Ai: int, dirty_Aj: int, dirty_Ak: int,
        num_u: Tensor, num_v: Tensor, num_w: Tensor,
        den_u: Tensor, den_v: Tensor, den_w: Tensor,
        blend_eps: float) -> None:
    """Phase-I 3-D streaming SDF + face velocity update (no rho).

    Companion to ``bdim_coeff_3d``: fills ``sdf_cc`` (persistent),
    ``sdf_u/v/w`` and ``body_u/v/w`` (per-step temporaries) inside the
    dirty AABB.  Does NOT touch ``winning_rho_cc`` because Kernel B
    computes rho_eff from mu0 in registers.

    ``key_cc_t``, ``key_u_t``, ``key_v_t``, ``key_w_t`` are caller-allocated
    int64 scratch buffers of size ``>= Ngx*Ngy*Ngz`` (one per stagger)
    that the kernel uses to pack/unpack per-cell winning-body keys.  Pass
    persistent buffers (allocated once at solver init) to avoid the 4×
    empty-tensor allocation that the kernel used to do internally.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_stag_3d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        key_cc_t, key_u_t, key_v_t, key_w_t,
        int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        num_u, num_v, num_w, den_u, den_v, den_w, float(blend_eps),
    )


def bdim_coeff_3d(
        u_prime: Tensor, v_prime: Tensor, w_prime: Tensor,
        sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        u0: Tensor, v0: Tensor, w0: Tensor,
        ch: Tensor, cv: Tensor, cw: Tensor,
        eps: float, rho_f: float, dt: float,
        h_grid: float,
        dirty_i0: int, dirty_j0: int, dirty_k0: int,
        dirty_Ai: int, dirty_Aj: int, dirty_Ak: int,
        mu0_projection: int = 1) -> None:
    """Phase-I fused BDIM2 + variable-density Poisson coefficient kernel.

    Reads advdiff outputs (``u_prime/v_prime/w_prime``) plus the Kernel-A
    face SDFs and rigid-body face velocities.  Writes the persistent
    velocity fields ``u0/v0/w0`` and Poisson coefficients ``ch/cv/cw``
    inside the dirty AABB.  ``mu0``, ``mu1`` and the unit normals live
    only in CUDA thread registers.

    Caller responsibilities:
      * ``u_prime/v_prime/w_prime`` must be distinct allocations from
        ``u0/v0/w0`` (otherwise central-difference reads at neighbouring
        cells race with writes).
      * Cells outside the dirty AABB are NOT touched.  Caller must
        ensure ``u0/v0/w0`` already contain the advdiff result there
        (e.g. via ``u0.copy_(u_prime)`` before this call) and that
        ``ch/cv/cw`` already hold the outside-body default
        ``dt / rho_fluid``.
    """
    return torch.ops.lilytorch_kernels.bdim_coeff_3d.default(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0, ch, cv, cw,
        float(eps), float(rho_f), float(dt),
        float(h_grid),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        int(mu0_projection),
    )


def bdim_coeff_sigma_3d(
        u_prime: Tensor, v_prime: Tensor, w_prime: Tensor,
        sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        u0: Tensor, v0: Tensor, w0: Tensor,
        ch: Tensor, cv: Tensor, cw: Tensor,
        key_u: Tensor, key_v: Tensor, key_w: Tensor,
        sigma_shifts: Tensor,
        eps: float, rho_f: float, dt: float,
        h_grid: float,
        dirty_i0: int, dirty_j0: int, dirty_k0: int,
        dirty_Ai: int, dirty_Aj: int, dirty_Ak: int,
        mu0_projection: int = 1) -> None:
    """BDIM-σ variant of :func:`bdim_coeff_3d` (Lauber et al. 2022).

    For each cell the Poisson coefficient is evaluated with ``mu0`` of a
    shifted SDF ``phi - sigma_shifts[body_id]`` where ``body_id`` is
    decoded from the Kernel-A AABB-local key buffer.  The velocity BDIM
    field (``u0/v0/w0``) uses the unmodified ``mu0`` — only the
    ``ch/cv/cw`` Poisson coefficient line uses the shifted ``mu0``.
    """
    return torch.ops.lilytorch_kernels.bdim_coeff_sigma_3d.default(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0, ch, cv, cw,
        key_u, key_v, key_w, sigma_shifts,
        float(eps), float(rho_f), float(dt), float(h_grid),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        int(mu0_projection),
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
        ref_desc: Tensor,
        ref_val: Tensor,
        max_dim0: int,
        max_dim1: int) -> None:
    """Phase H: fused 3-D boundary-condition writes.

    Three op kinds, packed into separate descriptor tensors:
      * ``neu_desc`` (int32 [N_neu, 3]) — Neumann copies
        ``base[ghost] = base[adjacent]``.
      * ``dir_desc`` (int32 [N_dir, 3]) + ``dir_val`` — direct writes
        ``base[offset] = value`` (wall-normal staggered Dirichlet).
      * ``ref_desc`` (int32 [N_ref, 4]) + ``ref_val`` — reflective
        writes ``base[dst] = 2*value - base[src]`` (tangential
        Dirichlet, WaterLily-style).

    Mutates ``u``, ``v``, ``w`` in place.
    """
    return torch.ops.lilytorch_kernels.apply_bcs_3d.default(
        u, v, w,
        shapes, neu_desc, dir_desc, dir_val, ref_desc, ref_val,
        int(max_dim0), int(max_dim1),
    )


# =====================================================================

def streaming_sdf_stag_2d_multi(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        key_cc_t: Tensor, key_u_t: Tensor, key_v_t: Tensor,
        interp_method: int,
        dirty_i0: int, dirty_j0: int,
        dirty_Ai: int, dirty_Aj: int,
        num_u: Tensor, num_v: Tensor,
        den_u: Tensor, den_v: Tensor,
        blend_eps: float) -> None:
    """Phase-I 2-D streaming SDF + face velocity update (no rho).

    Companion to ``bdim_coeff_2d``: fills ``sdf_cc`` (persistent),
    ``sdf_u/v`` and ``body_u/v`` (per-step temporaries) inside the
    dirty AABB.  Does NOT touch ``winning_rho_cc`` -- Kernel B computes
    rho_eff from mu0 in registers.

    ``key_cc_t/key_u_t/key_v_t`` are caller-allocated int64 scratch
    buffers of size ``>= Ngx*Ngy`` (one per stagger) that the kernel
    uses to pack/unpack per-cell winning-body keys; required by the
    BDIM-σ path so Kernel B can decode body_id per cell.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_stag_2d_multi.default(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim,
        gx, gy, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, body_u, body_v,
        key_cc_t, key_u_t, key_v_t,
        int(interp_method),
        int(dirty_i0), int(dirty_j0),
        int(dirty_Ai), int(dirty_Aj),
        num_u, num_v, den_u, den_v, float(blend_eps),
    )


def bdim_coeff_2d(
        u_prime: Tensor, v_prime: Tensor,
        sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        u0: Tensor, v0: Tensor,
        ch: Tensor, cv: Tensor,
        eps: float, rho_f: float, dt: float,
        h_grid: float,
        dirty_i0: int, dirty_j0: int,
        dirty_Ai: int, dirty_Aj: int,
        mu0_projection: int = 1) -> None:
    """Phase-I fused BDIM2 + variable-density Poisson coefficient kernel (2-D).

    2-D analogue of :func:`bdim_coeff_3d`.  See that wrapper for the
    documentation of caller responsibilities.
    """
    return torch.ops.lilytorch_kernels.bdim_coeff_2d.default(
        u_prime, v_prime,
        sdf_u, sdf_v,
        body_u, body_v,
        u0, v0, ch, cv,
        float(eps), float(rho_f), float(dt),
        float(h_grid),
        int(dirty_i0), int(dirty_j0),
        int(dirty_Ai), int(dirty_Aj),
        int(mu0_projection),
    )


def bdim_coeff_sigma_2d(
        u_prime: Tensor, v_prime: Tensor,
        sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        u0: Tensor, v0: Tensor,
        ch: Tensor, cv: Tensor,
        key_u: Tensor, key_v: Tensor,
        sigma_shifts: Tensor,
        eps: float, rho_f: float, dt: float,
        h_grid: float,
        dirty_i0: int, dirty_j0: int,
        dirty_Ai: int, dirty_Aj: int,
        mu0_projection: int = 1) -> None:
    """2-D analogue of :func:`bdim_coeff_sigma_3d`."""
    return torch.ops.lilytorch_kernels.bdim_coeff_sigma_2d.default(
        u_prime, v_prime,
        sdf_u, sdf_v,
        body_u, body_v,
        u0, v0, ch, cv,
        key_u, key_v, sigma_shifts,
        float(eps), float(rho_f), float(dt), float(h_grid),
        int(dirty_i0), int(dirty_j0),
        int(dirty_Ai), int(dirty_Aj),
        int(mu0_projection),
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
    torch.ops.lilytorch_kernels.interp_2d.default(
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
    torch.ops.lilytorch_kernels.interp_3d.default(
        F, xq, yq, zq,
        float(bx0), float(by0), float(bz0),
        float(inv_dx), float(inv_dy), float(inv_dz),
        int(Mx), int(My), int(Mz),
        int(interp_method),
        G,
    )
    return G


# =====================================================================
# Lagrangian (surface-integral) force kernels
# =====================================================================

def lagrangian_forces_2d(
        eps_xx: Tensor, eps_xy: Tensor, eps_yy: Tensor,
        p: Tensor, nu_rho_field: Tensor,
        cnt_flat: Tensor, cnt_offsets: Tensor,
        com_pos: Tensor,
        bx0: float, by0: float,
        inv_dx: float, inv_dy: float,
        Mx: int, My: int,
        method: str = "linear",
        sample_offset: float = 0.0,
        out: Tensor = None) -> Tensor:
    """Fused 2-D Lagrangian surface-integral forces.

    Parameters mirror the schema in ``ops.cpp``.  Returns the per-body
    force/torque tensor of shape ``(B, 6)`` float64 with column layout
    ``[fv_x, fv_y, t_v, fp_x, fp_y, t_p]``.

    ``cnt_flat`` is the concatenation of every body's contour ``(2, M_b)``
    along the marker axis (shape ``(2, sum_b M_b)``).  ``cnt_offsets`` is
    the int64 prefix-offset tensor of shape ``(B+1,)`` such that body
    ``b`` owns markers ``[cnt_offsets[b], cnt_offsets[b+1])``.

    ``nu_rho_field`` may be either a 1-element tensor (constant ν·ρ) or
    a full CC field of shape ``(Mx, My)``.
    """
    interp_method = _METHOD_MAP.get(method)
    if interp_method is None:
        raise ValueError(f"method must be 'linear' or 'quadratic', got {method!r}")
    B = int(com_pos.shape[0])
    if out is None:
        out = torch.zeros(B, 6, dtype=torch.float64, device=p.device)
    torch.ops.lilytorch_kernels.lagrangian_forces_2d.default(
        eps_xx, eps_xy, eps_yy,
        p, nu_rho_field,
        cnt_flat, cnt_offsets,
        com_pos,
        float(bx0), float(by0),
        float(inv_dx), float(inv_dy),
        int(Mx), int(My),
        int(interp_method),
        float(sample_offset),
        out,
    )
    return out


def lagrangian_forces_3d(
        eps_xx: Tensor, eps_yy: Tensor, eps_zz: Tensor,
        eps_xy: Tensor, eps_xz: Tensor, eps_yz: Tensor,
        p: Tensor, nu_rho_field: Tensor,
        tri_centroid: Tensor, tri_normal: Tensor, tri_area: Tensor,
        tri_offsets: Tensor, com_pos: Tensor,
        bx0: float, by0: float, bz0: float,
        inv_dx: float, inv_dy: float, inv_dz: float,
        Mx: int, My: int, Mz: int,
        method: str = "linear",
        sample_offset: float = 0.0,
        out: Tensor = None) -> Tensor:
    """Fused 3-D Lagrangian surface-integral forces.

    Inputs mirror :func:`lagrangian_forces_2d`.  ``tri_centroid``,
    ``tri_normal`` have shape ``(3, T_total)`` (concatenated across all
    bodies along the triangle axis); ``tri_area`` has shape ``(T_total,)``.
    ``tri_offsets`` is the int64 ``(B+1,)`` prefix-offset tensor.

    Returns ``(B, 12)`` float64 with column layout
    ``[fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
       fp_x, fp_y, fp_z, tp_x, tp_y, tp_z]``.
    """
    interp_method = _METHOD_MAP.get(method)
    if interp_method is None:
        raise ValueError(f"method must be 'linear' or 'quadratic', got {method!r}")
    B = int(com_pos.shape[0])
    if out is None:
        out = torch.zeros(B, 12, dtype=torch.float64, device=p.device)
    torch.ops.lilytorch_kernels.lagrangian_forces_3d.default(
        eps_xx, eps_yy, eps_zz,
        eps_xy, eps_xz, eps_yz,
        p, nu_rho_field,
        tri_centroid, tri_normal, tri_area,
        tri_offsets, com_pos,
        float(bx0), float(by0), float(bz0),
        float(inv_dx), float(inv_dy), float(inv_dz),
        int(Mx), int(My), int(Mz),
        int(interp_method),
        float(sample_offset),
        out,
    )
    return out


def apply_bcs_2d(
        u: Tensor, v: Tensor,
        shapes: Tensor,
        neu_desc: Tensor,
        dir_desc: Tensor,
        dir_val: Tensor,
        ref_desc: Tensor,
        ref_val: Tensor,
        max_line_dim: int) -> None:
    """Fused 2-D boundary-condition writes (Neumann + Dirichlet direct +
    reflective).

    Mutates ``u`` and ``v`` in place.  See :func:`apply_bcs_3d` for the
    descriptor layout; here ``shapes`` is int64 ``[2, 2]`` with rows
    (Nx, Ny) for u and v, and ``axis`` is restricted to 0 (x) or 1 (y).
    """
    return torch.ops.lilytorch_kernels.apply_bcs_2d.default(
        u, v,
        shapes, neu_desc, dir_desc, dir_val, ref_desc, ref_val,
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


# =====================================================================
# Multigrid residual kernels (no global J / active allocations)
# =====================================================================

def mg_residual_2d(
        p: Tensor,
        f: Tensor,
        cp0: Tensor, cm0: Tensor,
        cp1: Tensor, cm1: Tensor,
        jcap_tol: float) -> Tensor:
    """Compute the 2-D multigrid residual

        r = (f - A(p)) * (|J| >= jcap_tol)

    where A(p) = sum(c * p_neighbours) - J * p   and
    J = (cp0 + cm0 + cp1 + cm1).  J and the active mask are computed in
    CUDA registers; nothing of that size is ever materialised as a tensor.

    Parameters
    ----------
    p   : ghost-padded pressure, shape ``(Nx+2, Ny+2)``.
    f   : interior RHS, shape ``(Nx, Ny)``.
    cp0/cm0/cp1/cm1 : interior-shape face coefficients, must be contiguous.
    jcap_tol : cells with ``|J| < jcap_tol`` get ``r = 0``.

    Returns
    -------
    Tensor of shape ``(Nx, Ny)`` (same dtype/device as ``f``).
    """
    r = torch.empty_like(f)
    torch.ops.lilytorch_kernels.mg_residual_2d.default(
        p, f,
        cp0.contiguous(), cm0.contiguous(),
        cp1.contiguous(), cm1.contiguous(),
        float(jcap_tol), r,
    )
    return r


def mg_residual_3d(
        p: Tensor,
        f: Tensor,
        cp0: Tensor, cm0: Tensor,
        cp1: Tensor, cm1: Tensor,
        cp2: Tensor, cm2: Tensor,
        jcap_tol: float) -> Tensor:
    """3-D analogue of :func:`mg_residual_2d`.

    ``p`` is ghost-padded ``(Nx+2, Ny+2, Nz+2)``; ``f`` and all coefficient
    tensors are interior-shape ``(Nx, Ny, Nz)``.  Returns ``r`` of shape
    ``(Nx, Ny, Nz)`` without ever allocating ``J`` or ``active`` globally.
    """
    r = torch.empty_like(f)
    torch.ops.lilytorch_kernels.mg_residual_3d.default(
        p, f,
        cp0.contiguous(), cm0.contiguous(),
        cp1.contiguous(), cm1.contiguous(),
        cp2.contiguous(), cm2.contiguous(),
        float(jcap_tol), r,
    )
    return r


@torch.library.register_fake("lilytorch_kernels::mg_residual_2d")
def _mg_residual_2d_abstract(p, f, cp0, cm0, cp1, cm1, jcap_tol, r):
    pass   # r is filled in place; no new tensors created


@torch.library.register_fake("lilytorch_kernels::mg_residual_3d")
def _mg_residual_3d_abstract(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                              jcap_tol, r):
    pass   # r is filled in place; no new tensors created


# =====================================================================
# Multigrid grid-transfer kernels (restriction + prolongation)
# =====================================================================

def restrict_residual_2d(r: Tensor, rc: Tensor) -> None:
    """Sum-of-4-children residual restriction (2-D), bit-exact with the
    PyTorch slicing chain used in poisson_mult._restrict_residual_2d."""
    torch.ops.lilytorch_kernels.restrict_residual_2d.default(r.contiguous(), rc)


def restrict_residual_3d(r: Tensor, rc: Tensor) -> None:
    """Sum-of-8-children residual restriction (3-D)."""
    torch.ops.lilytorch_kernels.restrict_residual_3d.default(r.contiguous(), rc)


def restrict_face_2d(src: Tensor, dst: Tensor, face_dim: int) -> None:
    """WaterLily face restriction (2-D): stride-2 in ``face_dim``,
    sum-of-pairs in transverse, single 0.5 factor."""
    torch.ops.lilytorch_kernels.restrict_face_2d.default(
        src.contiguous(), dst, int(face_dim))


def restrict_face_3d(src: Tensor, dst: Tensor, face_dim: int) -> None:
    """WaterLily face restriction (3-D)."""
    torch.ops.lilytorch_kernels.restrict_face_3d.default(
        src.contiguous(), dst, int(face_dim))


def prolongate_add_2d(ec: Tensor, p: Tensor) -> None:
    """Bilinear (align_corners=False) prolongation of ec[interior]
    added in place into p[interior].  Both tensors are ghost-padded
    ``(N+2, N+2)``."""
    torch.ops.lilytorch_kernels.prolongate_add_2d.default(ec.contiguous(), p)


def prolongate_add_3d(ec: Tensor, p: Tensor) -> None:
    """Trilinear prolongation + in-place add (3-D)."""
    torch.ops.lilytorch_kernels.prolongate_add_3d.default(ec.contiguous(), p)


@torch.library.register_fake("lilytorch_kernels::restrict_residual_2d")
def _restrict_residual_2d_abstract(r, rc): pass

@torch.library.register_fake("lilytorch_kernels::restrict_residual_3d")
def _restrict_residual_3d_abstract(r, rc): pass

@torch.library.register_fake("lilytorch_kernels::restrict_face_2d")
def _restrict_face_2d_abstract(src, dst, face_dim): pass

@torch.library.register_fake("lilytorch_kernels::restrict_face_3d")
def _restrict_face_3d_abstract(src, dst, face_dim): pass

@torch.library.register_fake("lilytorch_kernels::prolongate_add_2d")
def _prolongate_add_2d_abstract(ec, p): pass

@torch.library.register_fake("lilytorch_kernels::prolongate_add_3d")
def _prolongate_add_3d_abstract(ec, p): pass


# =====================================================================
# Monolithic multigrid Poisson solvers
# =====================================================================

_SMOOTHER_MAP = {"rbgs": 0, "jacobi": 1}


def poisson_solve_multigrid_2d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor,
        h2: float, jcap_tol: float, w: float,
        nsmoothing: int, max_vcycles: int,
        tol: float, smoother: str = "rbgs") -> Tensor:
    """Native multigrid Poisson driver (2-D).

    Replaces the Python multi-V-cycle loop in PoissonSolver.solve_multigrid:
    scales ``f`` by ``h2``, runs up to ``max_vcycles`` V-cycles with L∞
    early-exit at ``tol``, then subtracts the float64-computed mean from
    ``p``.  ``p`` (ghost-padded) is mutated in place; the returned tensor
    is the final residual on the interior grid.
    """
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.poisson_solve_multigrid_2d.default(
        p, f, ch, cv,
        float(h2), float(jcap_tol), float(w),
        int(nsmoothing), int(max_vcycles), float(tol), int(sid),
    )


def poisson_solve_multigrid_3d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor, cw: Tensor,
        h2: float, jcap_tol: float, w: float,
        nsmoothing: int, max_vcycles: int,
        tol: float, smoother: str = "rbgs") -> Tensor:
    """Native multigrid Poisson driver (3-D).  See 2-D for semantics."""
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.poisson_solve_multigrid_3d.default(
        p, f, ch, cv, cw,
        float(h2), float(jcap_tol), float(w),
        int(nsmoothing), int(max_vcycles), float(tol), int(sid),
    )


@torch.library.register_fake("lilytorch_kernels::poisson_solve_multigrid_2d")
def _poisson_solve_multigrid_2d_abstract(
        p, f, ch, cv, h2, jcap_tol, w,
        nsmoothing, max_vcycles, tol, smoother_id):
    return torch.empty_like(f)

@torch.library.register_fake("lilytorch_kernels::poisson_solve_multigrid_3d")
def _poisson_solve_multigrid_3d_abstract(
        p, f, ch, cv, cw, h2, jcap_tol, w,
        nsmoothing, max_vcycles, tol, smoother_id):
    return torch.empty_like(f)


def poisson_solve_mgcg_2d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor,
        h2: float, jcap_tol: float, w: float,
        nsmoothing: int, max_cycles: int, precond_vcycles: int,
        tol: float, smoother: str = "rbgs") -> Tensor:
    """Native MGCG Poisson driver (2-D).  ``p`` is mutated in place; returns final residual."""
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.poisson_solve_mgcg_2d.default(
        p, f, ch, cv,
        float(h2), float(jcap_tol), float(w),
        int(nsmoothing), int(max_cycles), int(precond_vcycles),
        float(tol), int(sid),
    )


def poisson_solve_mgcg_3d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor, cw: Tensor,
        h2: float, jcap_tol: float, w: float,
        nsmoothing: int, max_cycles: int, precond_vcycles: int,
        tol: float, smoother: str = "rbgs") -> Tensor:
    """Native MGCG Poisson driver (3-D)."""
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.poisson_solve_mgcg_3d.default(
        p, f, ch, cv, cw,
        float(h2), float(jcap_tol), float(w),
        int(nsmoothing), int(max_cycles), int(precond_vcycles),
        float(tol), int(sid),
    )


@torch.library.register_fake("lilytorch_kernels::poisson_solve_mgcg_2d")
def _poisson_solve_mgcg_2d_abstract(
        p, f, ch, cv, h2, jcap_tol, w,
        nsmoothing, max_cycles, precond_vcycles, tol, smoother_id):
    return torch.empty_like(f)


@torch.library.register_fake("lilytorch_kernels::poisson_solve_mgcg_3d")
def _poisson_solve_mgcg_3d_abstract(
        p, f, ch, cv, cw, h2, jcap_tol, w,
        nsmoothing, max_cycles, precond_vcycles, tol, smoother_id):
    return torch.empty_like(f)


def poisson_solve_rmgcg_2d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor,
        U: Tensor, W: Tensor, harvest_k: int,
        h2: float, jcap_tol: float, w: float,
        nsmoothing: int, max_cycles: int, precond_vcycles: int,
        tol: float, smoother: str = "rbgs"):
    """Native recycled-MGCG driver (2-D).

    ``U`` (kdef, Nx+2, Ny+2) is the B-orthonormal recycle basis and ``W``
    (kdef, Nx, Ny) = B·U; pass empty (kdef=0) tensors for a plain solve.
    ``p`` is mutated in place.  Returns ``(r, D, niter)`` where ``D`` holds the
    last ``harvest_k`` search directions (full grid) for refreshing the space.
    """
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.poisson_solve_rmgcg_2d.default(
        p, f, ch, cv, U, W, int(harvest_k),
        float(h2), float(jcap_tol), float(w),
        int(nsmoothing), int(max_cycles), int(precond_vcycles),
        float(tol), int(sid),
    )


def poisson_solve_rmgcg_3d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor, cw: Tensor,
        U: Tensor, W: Tensor, harvest_k: int,
        h2: float, jcap_tol: float, w: float,
        nsmoothing: int, max_cycles: int, precond_vcycles: int,
        tol: float, smoother: str = "rbgs"):
    """Native recycled-MGCG driver (3-D).  See :func:`poisson_solve_rmgcg_2d`."""
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.poisson_solve_rmgcg_3d.default(
        p, f, ch, cv, cw, U, W, int(harvest_k),
        float(h2), float(jcap_tol), float(w),
        int(nsmoothing), int(max_cycles), int(precond_vcycles),
        float(tol), int(sid),
    )


@torch.library.register_fake("lilytorch_kernels::poisson_solve_rmgcg_2d")
def _poisson_solve_rmgcg_2d_abstract(
        p, f, ch, cv, U, W, harvest_k, h2, jcap_tol, w,
        nsmoothing, max_cycles, precond_vcycles, tol, smoother_id):
    hk = max(int(harvest_k), 1)
    D = f.new_empty((hk, f.shape[0] + 2, f.shape[1] + 2))
    return torch.empty_like(f), D, 0


@torch.library.register_fake("lilytorch_kernels::poisson_solve_rmgcg_3d")
def _poisson_solve_rmgcg_3d_abstract(
        p, f, ch, cv, cw, U, W, harvest_k, h2, jcap_tol, w,
        nsmoothing, max_cycles, precond_vcycles, tol, smoother_id):
    hk = max(int(harvest_k), 1)
    D = f.new_empty((hk, f.shape[0] + 2, f.shape[1] + 2, f.shape[2] + 2))
    return torch.empty_like(f), D, 0
