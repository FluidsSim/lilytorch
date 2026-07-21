"""Thin Python wrappers around ``torch.ops.lilytorch_kernels.*``."""
import torch
from torch import Tensor

# Force the native extension to load (registers TORCH_LIBRARY(lilytorch_kernels))
# before any @torch.library.register_fake decorators are evaluated below.
from lilytorch.src import _C  # noqa: F401

__all__ = [
    "streaming_sdf_stag_2d_resolve",
    "streaming_sdf_stag_3d_resolve",
    "streaming_sdf_forces_post_3d",
    "apply_bcs_3d",
    "bdim_apply_2d",
    "bdim_apply_3d",
    "sl_advect_2d",
    "sl_advect_3d",
    "diffuse_add",
    "streaming_sdf_forces_post_2d",
    "apply_bcs_2d",
    "interp_2d",
    "interp_3d",
    "lagrangian_forces_2d",
    "lagrangian_forces_3d",
    "rbgs_sweep_3d",
    "mg_residual_2d",
    "poisson_solve_multigrid_2d",
    "poisson_solve_multigrid_3d",
    "poisson_solve_mgcg_2d",
    "poisson_solve_mgcg_3d",
    "poisson_solve_rmgcg_2d",
    "poisson_solve_rmgcg_3d",
    "advect_flux_accumulate",
    "cvof_sweep",
    "body_update_2d",
    "body_update_3d",
    "RegularGridInterpolator",
]
_METHOD_MAP = {"linear": 0, "quadratic": 1}

# Persistent dummy for mw-off sdf_cc/div_corr (mirrors bdim._mw_dummy).
_MW_DUMMY_NATIVE: dict = {}


def _mw_dummy_native(u0: torch.Tensor) -> torch.Tensor:
    """Persistent 1-element zero placeholder for ``sdf_cc``/``div_corr`` when
    the Maertens–Weymouth correction is off."""
    key = (u0.device, u0.dtype)
    d = _MW_DUMMY_NATIVE.get(key)
    if d is None:
        d = torch.zeros(1, dtype=u0.dtype, device=u0.device)
        _MW_DUMMY_NATIVE[key] = d
    return d

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
        eps_body: float,
        sample_offset_pressure: float, sample_offset_friction: float,
        h3: float,
        delta_order: int,
        out: Tensor,
        force_submethod: int = 0, ph_tau: float = 0.0) -> None:
    """3-D Phase D only: current-union-normal force integration.

    Reuses the streamed body metadata and current union SDF produced by the
    update stage, but consumes the current post-fluid-step fields.

    ``sample_offset_pressure`` / ``sample_offset_friction``: the iso-surface
    ``φ = offset`` (metres) on which each channel's smoothed delta is centred.
    The Maertens & Weymouth convention — and hence the production default —
    is ``(0, eps_multiplier*h)``: pressure on the body surface, σ shifted out
    of the BDIM band where ε(u_blend) ≈ μ0·ε(u_fluid) contaminates it.  They
    are separate arguments because a like-for-like comparison against the
    lagrangian readout requires pinning both readouts to the same locations.

    ``force_submethod``: 0 = ndelta = union smoothed-Heaviside gradient ∂_iH
    for BOTH channels (the default, and the only gauge-safe readout); 2 = sm2 =
    union coarea magnitude × per-body analytic normal (per-link-accuracy variant,
    gauge-UNSAFE — must never reach the two-phase solver).  Both split the union
    force to bodies by a softmin partition of unity whose blend width is carried
    by ``ph_tau`` (metres; ≤0 → hard nearest-body winner).  (Value 1, the old
    deltaH readout, has been removed; there is an intentional numbering gap.)
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_forces_post_3d.default(
        F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, int(interp_method),
        u, v, w, p, nu_rho_field,
        float(eps_body),
        float(sample_offset_pressure), float(sample_offset_friction),
        float(h3), int(delta_order),
        int(force_submethod), float(ph_tau), out,
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
#  Regime-B streaming SDF: per-body private buffers + resolve
# =====================================================================

def streaming_sdf_stag_2d_resolve(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor,
        body_u: Tensor, body_v: Tensor,
        interp_method: int,
        dirty_i0: int, dirty_j0: int,
        dirty_Ai: int, dirty_Aj: int,
        priv_offsets: Tensor,
        priv_sdf_cc: Tensor, priv_sdf_u: Tensor, priv_sdf_v: Tensor,
        priv_body_u: Tensor, priv_body_v: Tensor,
        blend_eps: float = 0.0) -> None:
    """2-D Regime-B streaming SDF: per-body private buffers + resolve.

    Two-stage pipeline for overlapping-body regimes:
    1. Min-stage: per-body parallel writes raw SDF + staggered velocity to
       per-body private flat buffers (no packed key, no atomics).
    2. Resolve-stage: iterates the union dirty AABB, reads each covering
       body's private buffer, picks min, writes winner to global tensors.
       When ``blend_eps > 0`` the body velocities are the softmin blend
       Σwᵢvᵢ/Σwᵢ, wᵢ = sigmoid(-sᵢ/blend_eps), accumulated in registers
       over the covering bodies (deterministic — no atomics).

    ``priv_offsets``: int64 [B+1] cumulative body_vol offsets.
    ``priv_sdf_cc``, etc.: flat tensors of size ``priv_offsets[-1]``.
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_stag_2d_resolve.default(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim,
        gx, gy, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, body_u, body_v,
        int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
        priv_offsets,
        priv_sdf_cc, priv_sdf_u, priv_sdf_v,
        priv_body_u, priv_body_v,
        float(blend_eps),
    )


def streaming_sdf_stag_3d_resolve(
        F_flat: Tensor, F_offsets: Tensor,
        body_shapes: Tensor, body_meta: Tensor, kin: Tensor,
        aabb_lo: Tensor, aabb_dim: Tensor,
        gx: Tensor, gy: Tensor, gz: Tensor,
        h_grid: float, max_vol_per_body: int,
        sdf_cc: Tensor, sdf_u: Tensor, sdf_v: Tensor, sdf_w: Tensor,
        body_u: Tensor, body_v: Tensor, body_w: Tensor,
        interp_method: int,
        dirty_i0: int, dirty_j0: int, dirty_k0: int,
        dirty_Ai: int, dirty_Aj: int, dirty_Ak: int,
        priv_offsets: Tensor,
        priv_sdf_cc: Tensor, priv_sdf_u: Tensor, priv_sdf_v: Tensor, priv_sdf_w: Tensor,
        priv_body_u: Tensor, priv_body_v: Tensor, priv_body_w: Tensor,
        blend_eps: float = 0.0) -> None:
    """3-D Regime-B streaming SDF: per-body private buffers + resolve.

    See :func:`streaming_sdf_stag_2d_resolve` for the pipeline description
    (including the ``blend_eps > 0`` softmin velocity blend).
    """
    return torch.ops.lilytorch_kernels.streaming_sdf_stag_3d_resolve.default(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim,
        gx, gy, gz, float(h_grid), int(max_vol_per_body),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        priv_offsets,
        priv_sdf_cc, priv_sdf_u, priv_sdf_v, priv_sdf_w,
        priv_body_u, priv_body_v, priv_body_w,
        float(blend_eps),
    )


def bdim_apply_3d(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0, ch, cv, cw,
        eps, rho_f, dt, h_grid,
        dirty_i0, dirty_j0, dirty_k0,
        dirty_Ai, dirty_Aj, dirty_Ak,
        mu0_projection=1,
        sdf_cc=None, div_corr=None,
        eps_mw=1.0, inv_dx=0.0, inv_dy=0.0, inv_dz=0.0,
        rect_dev=None):
    """Native static full-grid BDIM2 forcing (3-D).

    Same signature,
    same semantics.  ``rect_dev`` is an optional pre-allocated int32
    device tensor [i0,j0,k0,Ai,Aj,Ak]; when None, a new tensor is
    allocated per call (eager path).
    """
    mw_on = 1 if div_corr is not None else 0
    if div_corr is None:
        sdf_cc = div_corr = _mw_dummy_native(u0)
    if rect_dev is None:
        rect_dev = torch.tensor(
            [int(dirty_i0), int(dirty_j0), int(dirty_k0),
             int(dirty_Ai), int(dirty_Aj), int(dirty_Ak)],
            dtype=torch.int32, device=u0.device)
    return torch.ops.lilytorch_kernels.bdim_apply_3d.default(
        u_prime, v_prime, w_prime,
        sdf_u, sdf_v, sdf_w,
        body_u, body_v, body_w,
        u0, v0, w0, ch, cv, cw,
        sdf_cc, div_corr, rect_dev,
        float(eps), float(rho_f), float(dt), float(h_grid),
        float(eps_mw), float(inv_dx), float(inv_dy), float(inv_dz),
        int(mu0_projection), int(mw_on),
    )


def bdim_apply_2d(
        u_prime, v_prime,
        sdf_u, sdf_v,
        body_u, body_v,
        u0, v0, ch, cv,
        eps, rho_f, dt, h_grid,
        dirty_i0, dirty_j0,
        dirty_Ai, dirty_Aj,
        mu0_projection=1,
        sdf_cc=None, div_corr=None,
        eps_mw=1.0, inv_dx=0.0, inv_dy=0.0,
        rect_dev=None):
    """Native static full-grid BDIM2 forcing (2-D).

    """
    mw_on = 1 if div_corr is not None else 0
    if div_corr is None:
        sdf_cc = div_corr = _mw_dummy_native(u0)
    if rect_dev is None:
        rect_dev = torch.tensor(
            [int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj)],
            dtype=torch.int32, device=u0.device)
    return torch.ops.lilytorch_kernels.bdim_apply_2d.default(
        u_prime, v_prime,
        sdf_u, sdf_v,
        body_u, body_v,
        u0, v0, ch, cv,
        sdf_cc, div_corr, rect_dev,
        float(eps), float(rho_f), float(dt), float(h_grid),
        float(eps_mw), float(inv_dx), float(inv_dy),
        int(mu0_projection), int(mw_on),
    )


# =====================================================================
#  sl_advect + diffuse_add — item 8.D
# =====================================================================

def sl_advect_2d(u, v, out_u, out_v,
                 gxu, gyu, gxv, gyv,
                 u_bx0, u_by0, u_idx, u_idy,
                 v_bx0, v_by0, v_idx, v_idy, dt):
    """Native fused RK2 semi-Lagrangian advection (2-D).

    """
    return torch.ops.lilytorch_kernels.sl_advect_2d.default(
        u, v, gxu, gyu, gxv, gyv,
        out_u, out_v,
        float(u_bx0), float(u_by0), float(u_idx), float(u_idy),
        float(v_bx0), float(v_by0), float(v_idx), float(v_idy),
        float(dt),
    )


def sl_advect_3d(u, v, w, out_u, out_v, out_w,
                 gxu, gyu, gzu, gxv, gyv, gzv,
                 gxw, gyw, gzw,
                 u_bx0, u_by0, u_bz0, u_idx, u_idy, u_idz,
                 v_bx0, v_by0, v_bz0, v_idx, v_idy, v_idz,
                 w_bx0, w_by0, w_bz0, w_idx, w_idy, w_idz, dt):
    """Native fused RK2 semi-Lagrangian advection (3-D).

    """
    return torch.ops.lilytorch_kernels.sl_advect_3d.default(
        u, v, w,
        gxu, gyu, gzu, gxv, gyv, gzv,
        gxw, gyw, gzw,
        out_u, out_v, out_w,
        float(u_bx0), float(u_by0), float(u_bz0),
        float(u_idx), float(u_idy), float(u_idz),
        float(v_bx0), float(v_by0), float(v_bz0),
        float(v_idx), float(v_idy), float(v_idz),
        float(w_bx0), float(w_by0), float(w_bz0),
        float(w_idx), float(w_idy), float(w_idz),
        float(dt),
    )


def diffuse_add(target, copy_buf, dt, *, dh, nu_eff=None, nu=None):
    """Native in-place explicit-diffusion Laplacian accumulate.

    Mirrors :func:`diffusion.diffuse_add_` exactly.
    Snapshot target → copy_buf (implicit barrier), then fused stencil
    read from copy_buf with accumulate into target.

    Graph-capture-safe: the constant-viscosity path passes a pre-computed
    ``scale_constant`` float instead of creating a GPU tensor or calling
    ``.item()``, both of which are illegal during CUDA graph capture.
    """
    ndim = target.ndim
    is_variable = nu_eff is not None
    scale = float(dt) if is_variable else float(nu) * float(dt)

    # Step 1: snapshot target → copy_buf.
    copy_buf.copy_(target)

    # Step 2: fused laplacian-accumulate.
    if is_variable:
        nu_eff_t = nu_eff
        scale_constant = 0.0  # ignored by kernel (is_variable=1 → scale=dt)
    else:
        # Pass a 1-element dummy tensor so nu_eff.numel()==1 in C++.
        # The kernel does NOT read nu_eff when is_variable=0, so any
        # pre-existing tensor view is safe.  The actual scale is passed
        # via scale_constant, avoiding both torch.tensor() (Python) and
        # .item() (C++) during CUDA graph capture.
        nu_eff_t = copy_buf[0:1, 0:1] if ndim == 2 else copy_buf[0:1, 0:1, 0:1]
        scale_constant = scale
    return torch.ops.lilytorch_kernels.diffuse_add.default(
        target, copy_buf, nu_eff_t,
        float(dt), int(ndim),
        float(dh[0]), float(dh[1]),
        float(dh[2]) if ndim == 3 else 0.0,
        float(scale_constant),
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
        eps_body: float,
        sample_offset_pressure: float, sample_offset_friction: float,
        h2: float,
        delta_order: int,
        out: Tensor,
        force_submethod: int = 0, ph_tau: float = 0.0) -> None:
    """2-D Phase D only: current-union-normal force integration.

    Reuses the streamed body metadata and current union SDF produced by
    the update stage, but consumes the current post-fluid-step fields.

    ``sample_offset_pressure`` / ``sample_offset_friction``: see
    ``streaming_sdf_forces_post_3d``.  Production default ``(0, eps)``.

    ``force_submethod``: 0 = ndelta = union smoothed-Heaviside gradient ∂_iH
    for BOTH channels (the default, and the only gauge-safe readout); 2 = sm2 =
    union coarea magnitude × per-body analytic normal (per-link-accuracy variant,
    gauge-UNSAFE — must never reach the two-phase solver).  Both split the union
    force to bodies by a softmin partition of unity whose blend width is carried
    by ``ph_tau`` (metres; ≤0 → hard nearest-body winner).  (Value 1, the old
    deltaH readout, has been removed; there is an intentional numbering gap.)
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
        float(eps_body),
        float(sample_offset_pressure), float(sample_offset_friction),
        float(h2), int(delta_order),
        int(force_submethod), float(ph_tau),
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
        sample_offset_pressure: float = 0.0,
        sample_offset_friction: float = 0.0,
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

    ``sample_offset_pressure`` / ``sample_offset_friction`` displace the
    query point for ``p`` and for the strain tensor respectively, along the
    outward normal (metres).  They are independent so that this readout can
    be pinned to the same sampling locations as the eulerian one; passing
    the same value for both is the legacy single-knob behaviour.
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
        float(sample_offset_pressure), float(sample_offset_friction),
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
        sample_offset_pressure: float = 0.0,
        sample_offset_friction: float = 0.0,
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
        float(sample_offset_pressure), float(sample_offset_friction),
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
# Multigrid RBGS smoother kernel
# =====================================================================

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

@torch.library.register_fake("lilytorch_kernels::rbgs_sweep_3d")
def _rbgs_sweep_3d_abstract(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                              jcap_tol, nsmoothing):
    pass   # p is mutated in place; no new tensors created


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


@torch.library.register_fake("lilytorch_kernels::mg_residual_2d")
def _mg_residual_2d_abstract(p, f, cp0, cm0, cp1, cm1, jcap_tol, r):
    pass   # r is filled in place; no new tensors created


# =====================================================================
# Monolithic multigrid Poisson solvers
# =====================================================================

_SMOOTHER_MAP = {"rbgs": 0, "jacobi": 1}


def mg_vcycle_2d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor,
        jcap_tol: float, w: float,
        nsmoothing: int, n_vcycles: int, smoother: str = "rbgs") -> Tensor:
    """``n_vcycles`` raw V-cycles (2-D) — the MGCG preconditioner primitive.

    No gauge fix: unlike :func:`poisson_solve_multigrid_2d` this applies no
    ghost-ring Neumann pass and no mean subtraction, and does NOT rescale
    ``f`` (pass the h²-scaled smoother RHS).  ``p`` (ghost-padded) is mutated
    in place; the interior residual is returned.
    """
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.mg_vcycle_2d.default(
        p, f, ch, cv, float(jcap_tol), float(w),
        int(nsmoothing), int(n_vcycles), int(sid),
    )


def mg_vcycle_3d(
        p: Tensor, f: Tensor, ch: Tensor, cv: Tensor, cw: Tensor,
        jcap_tol: float, w: float,
        nsmoothing: int, n_vcycles: int, smoother: str = "rbgs") -> Tensor:
    """``n_vcycles`` raw V-cycles (3-D).  See :func:`mg_vcycle_2d`."""
    sid = _SMOOTHER_MAP[smoother]
    return torch.ops.lilytorch_kernels.mg_vcycle_3d.default(
        p, f, ch, cv, cw, float(jcap_tol), float(w),
        int(nsmoothing), int(n_vcycles), int(sid),
    )


@torch.library.register_fake("lilytorch_kernels::mg_vcycle_2d")
def _mg_vcycle_2d_abstract(p, f, ch, cv, jcap_tol, w,
                           nsmoothing, n_vcycles, smoother_id):
    return torch.empty_like(f)


@torch.library.register_fake("lilytorch_kernels::mg_vcycle_3d")
def _mg_vcycle_3d_abstract(p, f, ch, cv, cw, jcap_tol, w,
                           nsmoothing, n_vcycles, smoother_id):
    return torch.empty_like(f)


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


# =====================================================================
# Convective flux-add (advection) + Weymouth-Yue conservative-VOF sweep
# =====================================================================

def advect_flux_accumulate(
        phi_src: Tensor, dst: Tensor, vel, comp_i: int,
        dt_dh, C_courant: float, scheme_id: int) -> None:
    """Fused per-cell convective flux accumulate for one velocity component.

    Fused flux accumulate: reads the flux stencil from
    ``phi_src`` (full-grid copy of ``vel[comp_i]``), computes face
    velocities from ``vel`` on the fly, and accumulates
    ``dst[cell] += Σ_d dt_dh[d] * (F_L - F_R)`` over the interior of the
    full-grid output ``dst``.  Graph-capture-safe (single kernel on the
    current stream, scalar params passed by value).
    """
    ndim = phi_src.ndim
    w = vel[2] if ndim == 3 else vel[0]   # dummy in 2-D, never read
    return torch.ops.lilytorch_kernels.advect_flux_accumulate.default(
        phi_src, dst, vel[0], vel[1], w,
        int(comp_i),
        float(dt_dh[0]), float(dt_dh[1]),
        float(dt_dh[2]) if ndim == 3 else 0.0,
        float(C_courant), int(scheme_id),
    )


@torch.library.register_fake("lilytorch_kernels::advect_flux_accumulate")
def _advect_flux_accumulate_abstract(
        phi_src, dst, u, v, w, comp_i,
        dt_dh0, dt_dh1, dt_dh2, C, scheme_id):
    pass   # dst is accumulated in place; no new tensors created


def cvof_sweep(
        a: Tensor, u_d: Tensor, cfl: float,
        face_dim: int, out: Tensor) -> None:
    """Weymouth & Yue conservative-VOF donor-acceptor sweep along ``face_dim``.

    Writes the interior of ``out`` in place from the alpha field ``a`` and
    face velocity ``u_d`` at Courant ``cfl``.
    """
    return torch.ops.lilytorch_kernels.cvof_sweep.default(
        a, u_d, float(cfl), int(face_dim), out,
    )


@torch.library.register_fake("lilytorch_kernels::cvof_sweep")
def _cvof_sweep_abstract(a, u_d, cfl, face_dim, out):
    pass   # out interior is written in place; no new tensors created


def strain_rate_magnitude(
        u: Tensor, v: Tensor, w, h: float, out: Tensor) -> None:
    """Strain-rate magnitude ``|S̄| = sqrt(2·S_ij·S_ij)`` on the full grid.

    Writes ``out`` (shaped like ``u``) in place.  ``w`` is required in 3-D and
    ignored in 2-D.  Reproduces the reference ``torch.gradient(edge_order=2)
    + _stag_to_cc`` path exactly, with every gradient kept in registers.
    """
    return torch.ops.lilytorch_kernels.strain_rate_magnitude.default(
        u, v, w, float(h), out,
    )


@torch.library.register_fake("lilytorch_kernels::strain_rate_magnitude")
def _strain_rate_magnitude_abstract(u, v, w, h, out):
    pass   # out is written in place; no new tensors created


# =====================================================================
# RegularGridInterpolator — native CUDA/CPU-backed scattered-point
# interpolation (replaces pytorch_interpolation).
# =====================================================================

from typing import Sequence  # noqa: E402

_VALID_METHODS = ("linear", "quadratic")


class RegularGridInterpolator:
    """Scattered-point interpolator on a uniform regular grid.

    Automatically selects the 2-D or 3-D kernel from the number of axes.

    Parameters
    ----------
    points : tuple of 1-D Tensors
        Axis coordinate arrays — exactly 2 (2-D) or 3 (3-D).
    F : Tensor
        Grid values, shape ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``.
    method : "linear" | "quadratic"
        Interpolation order.  Default "linear" (bilinear / trilinear).
    fill_value : any
        Kept for API compatibility; ignored.  Out-of-bounds queries are
        always clamped to the nearest border value by the kernel.
    """

    def __init__(
        self,
        points: Sequence[Tensor],
        F: Tensor,
        method: str = "linear",
        fill_value=None,          # kept for drop-in compatibility; ignored
    ) -> None:
        ndim = len(points)
        if ndim < 2:
            raise ValueError(
                f"At least 2 grid axes (x, y) are required; got {ndim}. "
                "Pass (x, y) for 2-D or (x, y, z) for 3-D."
            )
        if ndim > 3:
            raise ValueError(
                f"At most 3 grid axes are supported; got {ndim}."
            )
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be 'linear' or 'quadratic', got {method!r}."
            )

        self._ndim   = ndim
        self._method = method

        # Store axis tensors for attribute access compatibility
        axes = [ax.detach() for ax in points]
        self.x  = axes[0]
        self.y  = axes[1]
        self.z  = axes[2] if ndim == 3 else None

        # Pre-compute grid metadata (uniform spacing assumed)
        self.dx = float((axes[0][-1] - axes[0][0]) / max(axes[0].numel() - 1, 1))
        self.dy = float((axes[1][-1] - axes[1][0]) / max(axes[1].numel() - 1, 1))
        self.dz = (
            float((axes[2][-1] - axes[2][0]) / max(axes[2].numel() - 1, 1))
            if ndim == 3 else None
        )

        self._bx0    = float(axes[0][0])
        self._by0    = float(axes[1][0])
        self._bz0    = float(axes[2][0]) if ndim == 3 else 0.0
        self._inv_dx = 1.0 / self.dx if self.dx != 0.0 else 0.0
        self._inv_dy = 1.0 / self.dy if self.dy != 0.0 else 0.0
        self._inv_dz = (1.0 / self.dz if (self.dz is not None and self.dz != 0.0) else 0.0)
        self._Mx     = int(axes[0].numel())
        self._My     = int(axes[1].numel())
        self._Mz     = int(axes[2].numel()) if ndim == 3 else 1

        self._F = F.contiguous()

    # ------------------------------------------------------------------
    # .F property — reassigned every step in the CFD loop
    # ------------------------------------------------------------------
    @property
    def F(self) -> Tensor:
        return self._F

    @F.setter
    def F(self, val: Tensor) -> None:
        self._F = val.contiguous() if not val.is_contiguous() else val

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def __call__(self, *coords: Tensor) -> Tensor:
        """Interpolate at scattered query points.

        Parameters
        ----------
        *coords : Tensor
            One 1-D tensor per dimension — ``(xq, yq)`` for 2-D or
            ``(xq, yq, zq)`` for 3-D.  All must have the same numel.

        Returns
        -------
        Tensor
            Interpolated values, same shape as each coordinate tensor.
        """
        if len(coords) != self._ndim:
            raise ValueError(
                f"Expected {self._ndim} coordinate tensors, got {len(coords)}."
            )
        orig_shape = coords[0].shape
        flat = [c.reshape(-1) for c in coords]

        if self._ndim == 2:
            result = interp_2d(
                self._F, flat[0], flat[1],
                self._bx0, self._by0,
                self._inv_dx, self._inv_dy,
                self._Mx, self._My,
                self._method,
            )
        else:
            result = interp_3d(
                self._F, flat[0], flat[1], flat[2],
                self._bx0, self._by0, self._bz0,
                self._inv_dx, self._inv_dy, self._inv_dz,
                self._Mx, self._My, self._Mz,
                self._method,
            )

        return result.reshape(orig_shape)


# =====================================================================
#  Body-update bridges — streaming SDF dispatch
# =====================================================================

# Grow-only persistent per-body private-buffer cache for the Regime-B resolve.
# Keyed on (device, dtype, ndim); reused across steps so the streaming path does
# NO per-call allocation and NO host sync (the killer was ``total_vol.item()``).
_priv_cache: dict = {}


def _regime_b_priv(B, max_vol, dtype, device, ndim):
    """Sync-free private buffers + offsets for the Regime-B resolve.

    Uses a UNIFORM per-body stride of ``max_vol`` (host-known: ``B`` from
    ``aabb_dim.shape[0]``, ``max_vol`` a python int), so:
      * offsets = arange(B+1)*max_vol — no cumsum, no ``.item()``, no D→H sync;
      * buffers sized to ``B*max_vol`` (≥ Σ body_vol), grow-only + reused.
    Body ``b`` owns the disjoint slice ``[b*max_vol, b*max_vol+body_vol[b])``, which
    is what the min/resolve kernels index via ``priv_offsets[b]+local``.
    """
    need = int(B) * int(max_vol)
    n_bufs = 5 if ndim == 2 else 7
    key = (device, dtype, ndim)
    ent = _priv_cache.get(key)
    if ent is None or ent["cap"] < need:
        cap = max(need, (ent["cap"] if ent else 0))
        ent = {"cap": cap,
               "bufs": [torch.empty(cap, dtype=dtype, device=device)
                        for _ in range(n_bufs)]}
        _priv_cache[key] = ent
    offsets = torch.arange(B + 1, dtype=torch.int64, device=device) * int(max_vol)
    return offsets, ent["bufs"]


def body_update_2d(
    F_flat, F_offsets, body_shapes, body_meta, kin,
    aabb_lo, aabb_dim, gx, gy, h, max_vol,
    sdf_cc, sdf_u, sdf_v, body_u, body_v,
    interp_method,
    dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
    blend_eps=0.0,
):
    """Native 2-D streaming SDF body update (eager path).

    The caller (BDIMhandler) pre-fills ``sdf_*`` to +FAR and ``body_*`` to 0
    before every call — the kernels only write body-covered cells."""
    if int(dirty_Ai) * int(dirty_Aj) <= 0:
        return

    priv_offsets, pb = _regime_b_priv(
        body_shapes.size(0), max_vol, sdf_cc.dtype, sdf_cc.device, 2)
    streaming_sdf_stag_2d_resolve(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, body_u, body_v, int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
        priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4],
        blend_eps=float(blend_eps),
    )


def body_update_3d(
    F_flat, F_offsets, body_shapes, body_meta, kin,
    aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
    sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
    interp_method,
    dirty_i0, dirty_j0, dirty_k0,
    dirty_Ai, dirty_Aj, dirty_Ak,
    blend_eps=0.0,
):
    """Native 3-D streaming SDF body update (eager path).  See 2-D."""
    if int(dirty_Ai) * int(dirty_Aj) * int(dirty_Ak) <= 0:
        return

    priv_offsets, pb = _regime_b_priv(
        body_shapes.size(0), max_vol, sdf_cc.dtype, sdf_cc.device, 3)
    streaming_sdf_stag_3d_resolve(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, gz, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w, int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6],
        blend_eps=float(blend_eps),
    )


