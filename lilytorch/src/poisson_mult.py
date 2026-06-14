"""Dimension-agnostic multigrid Poisson solver with variable coefficients.

Solves   div(c * grad(p)) = f   using geometric multigrid V-cycles
with Jacobi or Red-Black Gauss-Seidel smoothing.  Works in 2-D and
3-D with a single code path.

The variable-coefficient discrete operator on a uniform grid (spacing *h*)
for the *d*-th direction is:

    [c_{d+} p_{i+1} - (c_{d+} + c_{d-}) p_i + c_{d-} p_{i-1}] / h^2

where c_{d+}, c_{d-} are face-averaged coefficients along dimension *d*.

Usage (backward-compatible with old 2-D interface)::

    ps = PoissonSolver(dtype, device, h, tol=1e-2)
    p, r = ps.solve_multigrid(f, p0, ch=ch, cv=cv)        # 2-D
    p, r = ps.solve_multigrid(f, p0, ch=ch, cv=cv, cw=cw)  # 3-D
"""

import torch

# ── Native CUDA smoother kernels ─────────────────────────────────────────────
from lilytorch.src.kernels.ops import (  # noqa: F401 (registers abstract impls)
    rbgs_sweep_2d as _native_rbgs_2d,
    rbgs_sweep_3d as _native_rbgs_3d,
    jacobi_sweep_2d as _native_jac_2d,
    jacobi_sweep_3d as _native_jac_3d,
    mg_residual_2d as _native_mg_residual_2d,
    mg_residual_3d as _native_mg_residual_3d,
)


# =====================================================================
# Smoother kernels (module-level)
# =====================================================================

def _bc_2d(q):
    """Neumann BCs for a 2-D tensor (in-place)."""
    q[0, :]  = q[1, :]
    q[-1, :] = q[-2, :]
    q[:, 0]  = q[:, 1]
    q[:, -1] = q[:, -2]


def _bc_3d(q):
    """Neumann BCs for a 3-D tensor (in-place)."""
    q[0, :, :]  = q[1, :, :]
    q[-1, :, :] = q[-2, :, :]
    q[:, 0, :]  = q[:, 1, :]
    q[:, -1, :] = q[:, -2, :]
    q[:, :, 0]  = q[:, :, 1]
    q[:, :, -1] = q[:, :, -2]


# ── 3-D helper: stencil sum (inlined for compile) ───────────────────
def _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p):
    return (cp0 * p[2:, 1:-1, 1:-1] + cm0 * p[:-2, 1:-1, 1:-1]
          + cp1 * p[1:-1, 2:, 1:-1] + cm1 * p[1:-1, :-2, 1:-1]
          + cp2 * p[1:-1, 1:-1, 2:] + cm2 * p[1:-1, 1:-1, :-2])


def _J3d(cp0, cm0, cp1, cm1, cp2, cm2):
    return cp0 + cm0 + cp1 + cm1 + cp2 + cm2


# ── 2-D helper: stencil sum (inlined for compile) ───────────────────
def _sum2d(cp0, cm0, cp1, cm1, p):
    return (cp0 * p[2:, 1:-1] + cm0 * p[:-2, 1:-1]
          + cp1 * p[1:-1, 2:] + cm1 * p[1:-1, :-2])


def _J2d(cp0, cm0, cp1, cm1):
    return cp0 + cm0 + cp1 + cm1


# ── Jacobi 3-D (compilable) ─────────────────────────────────────────
def _jacobi_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2, w, jcap_tol,
               nsmoothing):
    _bc_3d(p)
    J = _J3d(cp0, cm0, cp1, cm1, cp2, cm2)
    active = torch.abs(J) >= jcap_tol
    Jinv = torch.where(active, J.reciprocal(), torch.zeros_like(J))
    for _ in range(nsmoothing):
        s = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
        p[1:-1, 1:-1, 1:-1] = (
            w * (-f + s) * Jinv + (1 - w) * p[1:-1, 1:-1, 1:-1]
        )
        _bc_3d(p)
    del Jinv
    s = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
    s.addcmul_(J, p[1:-1, 1:-1, 1:-1], value=-1.0)
    del J
    s.neg_().add_(f).mul_(active)
    r = s
    del active
    return p, r


# ── Jacobi 2-D (compilable) ─────────────────────────────────────────
def _jacobi_2d(f, p, cp0, cm0, cp1, cm1, w, jcap_tol, nsmoothing):
    _bc_2d(p)
    J = _J2d(cp0, cm0, cp1, cm1)
    active = torch.abs(J) >= jcap_tol
    Jinv = torch.where(active, J.reciprocal(), torch.zeros_like(J))
    for _ in range(nsmoothing):
        s = _sum2d(cp0, cm0, cp1, cm1, p)
        p[1:-1, 1:-1] = (
            w * (-f + s) * Jinv + (1 - w) * p[1:-1, 1:-1]
        )
        _bc_2d(p)
    del Jinv
    s = _sum2d(cp0, cm0, cp1, cm1, p)
    s.addcmul_(J, p[1:-1, 1:-1], value=-1.0)
    del J
    s.neg_().add_(f).mul_(active)
    r = s
    del active
    return p, r


# ── RBGS 3-D (compilable) ───────────────────────────────────────────
def _rbgs_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol,
             nsmoothing, red, black):
    _bc_3d(p)
    J = _J3d(cp0, cm0, cp1, cm1, cp2, cm2)
    active = torch.abs(J) >= jcap_tol
    Jinv = torch.where(active, J.reciprocal(), torch.zeros_like(J))
    for _ in range(nsmoothing):
        s = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
        p_new = (-f + s) * Jinv
        p[1:-1, 1:-1, 1:-1] = torch.where(red, p_new, p[1:-1, 1:-1, 1:-1])
        _bc_3d(p)
        s = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
        p_new = (-f + s) * Jinv
        p[1:-1, 1:-1, 1:-1] = torch.where(black, p_new, p[1:-1, 1:-1, 1:-1])
        _bc_3d(p)
    del Jinv
    s = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
    s.addcmul_(J, p[1:-1, 1:-1, 1:-1], value=-1.0)
    del J
    s.neg_().add_(f).mul_(active)
    r = s
    del active
    return p, r


# ── RBGS 2-D (compilable) ───────────────────────────────────────────
def _rbgs_2d(f, p, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing,
             red, black):
    _bc_2d(p)
    J = _J2d(cp0, cm0, cp1, cm1)
    active = torch.abs(J) >= jcap_tol
    Jinv = torch.where(active, J.reciprocal(), torch.zeros_like(J))
    for _ in range(nsmoothing):
        s = _sum2d(cp0, cm0, cp1, cm1, p)
        p_new = (-f + s) * Jinv
        p[1:-1, 1:-1] = torch.where(red, p_new, p[1:-1, 1:-1])
        _bc_2d(p)
        s = _sum2d(cp0, cm0, cp1, cm1, p)
        p_new = (-f + s) * Jinv
        p[1:-1, 1:-1] = torch.where(black, p_new, p[1:-1, 1:-1])
        _bc_2d(p)
    del Jinv
    s = _sum2d(cp0, cm0, cp1, cm1, p)
    s.addcmul_(J, p[1:-1, 1:-1], value=-1.0)
    del J
    s.neg_().add_(f).mul_(active)
    r = s
    del active
    return p, r


# =====================================================================
# V-cycle helpers (3-D)
# =====================================================================

def _restrict_face_3d(ch, cv, cw):
    """Restrict face arrays from fine to coarse (3-D, WaterLily convention)."""
    # ch: face along dim 0 — stride-2 in dim 0 first, then SUM in dims 1,2
    ch_c = ch[::2, :, :]
    e1, o1 = ch_c[:, :-1:2, :], ch_c[:, 1::2, :]
    m1 = min(e1.shape[1], o1.shape[1])
    ch_c = e1[:, :m1] + o1[:, :m1]
    e2, o2 = ch_c[:, :, :-1:2], ch_c[:, :, 1::2]
    m2 = min(e2.shape[2], o2.shape[2])
    ch_c = (e2[:, :, :m2] + o2[:, :, :m2]) * 0.5
    # cv: face along dim 1 — stride-2 in dim 1 first, then SUM in dims 0,2
    cv_c = cv[:, ::2, :]
    e0, o0 = cv_c[:-1:2, :, :], cv_c[1::2, :, :]
    m0 = min(e0.shape[0], o0.shape[0])
    cv_c = e0[:m0] + o0[:m0]
    e2, o2 = cv_c[:, :, :-1:2], cv_c[:, :, 1::2]
    m2 = min(e2.shape[2], o2.shape[2])
    cv_c = (e2[:, :, :m2] + o2[:, :, :m2]) * 0.5
    # cw: face along dim 2 — stride-2 in dim 2 first, then SUM in dims 0,1
    cw_c = cw[:, :, ::2]
    e0, o0 = cw_c[:-1:2, :, :], cw_c[1::2, :, :]
    m0 = min(e0.shape[0], o0.shape[0])
    cw_c = e0[:m0] + o0[:m0]
    e1, o1 = cw_c[:, :-1:2, :], cw_c[:, 1::2, :]
    m1 = min(e1.shape[1], o1.shape[1])
    cw_c = (e1[:, :m1] + o1[:, :m1]) * 0.5
    return ch_c, cv_c, cw_c


def _restrict_residual_3d(r):
    """Full-weighting restriction of residual (3-D)."""
    e0, o0 = r[::2, :, :], r[1::2, :, :]
    m0 = min(e0.shape[0], o0.shape[0])
    rc = e0[:m0] + o0[:m0]
    e1, o1 = rc[:, ::2, :], rc[:, 1::2, :]
    m1 = min(e1.shape[1], o1.shape[1])
    rc = e1[:, :m1, :] + o1[:, :m1, :]
    e2, o2 = rc[:, :, ::2], rc[:, :, 1::2]
    m2 = min(e2.shape[2], o2.shape[2])
    rc = e2[:, :, :m2] + o2[:, :, :m2]
    return rc


def _prolongate_3d(err_coarse, target_shape):
    """Trilinear prolongation (3-D) for cell-centred multigrid.

    Uses F.interpolate with align_corners=False, which places cell centres
    at (i+0.5)/N — the correct mapping for cell-centred data.  This gives
    the standard prolongation weights: 3/4 on the parent coarse cell and
    1/4 on the nearest coarse neighbour.
    """
    ec = err_coarse[1:-1, 1:-1, 1:-1]
    out = torch.nn.functional.interpolate(
        ec.unsqueeze(0).unsqueeze(0),
        size=(target_shape[0], target_shape[1], target_shape[2]),
        mode='trilinear',
        align_corners=False,
    )
    return out[0, 0]


def _rb_masks_3d(nx, ny, nz, device):
    """Build red/black masks for interior of shape (nx, ny, nz)."""
    gi = torch.arange(nx, device=device)
    gj = torch.arange(ny, device=device)
    gk = torch.arange(nz, device=device)
    I, J, K = torch.meshgrid(gi, gj, gk, indexing="ij")
    parity = (I + J + K) % 2
    return (parity == 0), (parity == 1)


# ── Full 3-D V-cycle with Jacobi (coarse-level recursive) ──────────────────
def _vcycle_jac_3d(f, p, ch, cv, cw, w, jcap_tol, nsmoothing):
    """Complete 3-D V-cycle with Jacobi smoother (used at coarse levels)."""
    p = p.clone()
    cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
    cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
    cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]

    # pre-smooth
    p, r = _jacobi_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                       w, jcap_tol, nsmoothing)

    nx, ny, nz = f.shape
    if nx > 2 and ny > 2 and nz > 2:
        ch_c, cv_c, cw_c = _restrict_face_3d(ch, cv, cw)
        r_c = _restrict_residual_3d(r)

        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2, r_c.shape[2] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)

        err_c, _ = _vcycle_jac_3d(r_c, p_c, ch_c, cv_c, cw_c,
                                   w, jcap_tol, nsmoothing)

        err = _prolongate_3d(err_c, r.shape)
        p[1:-1, 1:-1, 1:-1] = p[1:-1, 1:-1, 1:-1] + err

        # post-smooth (recompute cfaces from same face arrays)
        cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
        cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
        cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
        p, r = _jacobi_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                           w, jcap_tol, nsmoothing)

    return p, r


# ── Full 3-D V-cycle with RBGS (coarse-level recursive) ────────────────────
def _vcycle_rbgs_3d(f, p, ch, cv, cw, jcap_tol, nsmoothing):
    """Complete 3-D V-cycle with RBGS smoother (used at coarse levels)."""
    p = p.clone()
    cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
    cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
    cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]

    red, black = _rb_masks_3d(f.shape[0], f.shape[1], f.shape[2], p.device)
    p, r = _rbgs_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                     jcap_tol, nsmoothing, red, black)

    nx, ny, nz = f.shape
    if nx > 2 and ny > 2 and nz > 2:
        ch_c, cv_c, cw_c = _restrict_face_3d(ch, cv, cw)
        r_c = _restrict_residual_3d(r)

        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2, r_c.shape[2] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)

        err_c, _ = _vcycle_rbgs_3d(r_c, p_c, ch_c, cv_c, cw_c,
                                    jcap_tol, nsmoothing)

        err = _prolongate_3d(err_c, r.shape)
        p[1:-1, 1:-1, 1:-1] = p[1:-1, 1:-1, 1:-1] + err

        cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
        cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
        cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
        # red/black masks are shape-dependent only; reuse from pre-smooth.
        p, r = _rbgs_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                         jcap_tol, nsmoothing, red, black)

    return p, r


# =====================================================================

def _restrict_face_2d(ch, cv):
    """Restrict face arrays from fine to coarse (2-D, WaterLily convention)."""
    # ch: face along dim 0 — stride-2 in dim 0 first, then SUM in dim 1
    ch_c = ch[::2, :]
    e = ch_c[:, :-1:2]
    o = ch_c[:, 1::2]
    m = min(e.shape[1], o.shape[1])
    ch_c = (e[:, :m] + o[:, :m]) * 0.5
    # cv: face along dim 1 — stride-2 in dim 1 first, then SUM in dim 0
    cv_c = cv[:, ::2]
    e = cv_c[:-1:2, :]
    o = cv_c[1::2, :]
    m = min(e.shape[0], o.shape[0])
    cv_c = (e[:m] + o[:m]) * 0.5
    return ch_c, cv_c


def _restrict_residual_2d(r):
    """Full-weighting restriction of residual (2-D)."""
    e0, o0 = r[::2, :], r[1::2, :]
    m0 = min(e0.shape[0], o0.shape[0])
    rc = e0[:m0] + o0[:m0]
    e1, o1 = rc[:, ::2], rc[:, 1::2]
    m1 = min(e1.shape[1], o1.shape[1])
    rc = e1[:, :m1] + o1[:, :m1]
    return rc


def _prolongate_2d(err_coarse, target_shape):
    """Bilinear prolongation (2-D) for cell-centred multigrid."""
    ec = err_coarse[1:-1, 1:-1]
    out = torch.nn.functional.interpolate(
        ec.unsqueeze(0).unsqueeze(0),
        size=(target_shape[0], target_shape[1]),
        mode='bilinear',
        align_corners=False,
    )
    return out[0, 0]


def _rb_masks_2d(nx, ny, device):
    """Build red/black masks for interior of shape (nx, ny)."""
    gi = torch.arange(nx, device=device)
    gj = torch.arange(ny, device=device)
    I, J = torch.meshgrid(gi, gj, indexing="ij")
    parity = (I + J) % 2
    return (parity == 0), (parity == 1)


# ── Full 2-D V-cycle with native RBGS kernel (hybrid) ──────────────────
# Uses the tiled CUDA kernel ONLY at the finest level for pre- and
# post-smoothing.  All coarse-level corrections fall back to the standard
# PyTorch _vcycle_rbgs_2d path.
#
# Why:  The tiled kernel loads a global-memory snapshot at the start of
# each kernel call and propagates red→black updates only within a tile
# (block-boundary approximation).  This is negligible on large grids but
# degrades the smoothing factor on coarse grids where adjacent blocks cover
# a large fraction of the total cells.  Using PyTorch RBGS for all coarse
# levels preserves the full V-cycle convergence rate while the native kernel
# provides the memory-bandwidth speedup at the dominant fine level.
def _vcycle_rbgs_2d_native(f, p, ch, cv, jcap_tol, nsmoothing):
    """Hybrid 2-D V-cycle: tiled native RBGS at fine level, PyTorch at coarse.

    Both the smoother and the residual computation run as native CUDA ops:
    ``J`` and the active mask are recomputed in CUDA registers inside
    ``mg_residual_2d`` instead of being materialised as ``(Nx, Ny)`` tensors.
    Only ``r`` (the residual itself) is allocated on the fine grid.

    The caller's ``p`` is mutated in place; the returned tensor aliases it.
    """
    # The CUDA kernel indexes coefficients as cp0[gi*Ny + gj], assuming a
    # C-contiguous (Nx, Ny) layout.  ch/cv arrive as non-contiguous slices
    # of the ghost-padded pressure grid (row stride Ny+2, not Ny), so we
    # must materialise them before extracting cp/cm pairs.
    ch = ch.contiguous()
    cv = cv.contiguous()
    cp0, cm0 = ch[1:, :], ch[:-1, :]
    cp1, cm1 = cv[:, 1:], cv[:, :-1]

    # Pre-smooth (fine level): native tiled kernel
    _native_rbgs_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing)
    # Fused residual — J and active live in CUDA registers only.
    r = _native_mg_residual_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol)

    nx, ny = f.shape
    if nx > 2 and ny > 2:
        ch_c, cv_c = _restrict_face_2d(ch, cv)
        r_c = _restrict_residual_2d(r)
        # r is no longer needed after restriction; free it to reduce peak footprint
        # during the coarse correction + prolongation phase.
        del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        # Coarse correction: PyTorch V-cycle (reliable convergence at all sizes)
        err_c, _ = _vcycle_rbgs_2d(r_c, p_c, ch_c, cv_c, jcap_tol, nsmoothing)
        # Use f.shape (same as the former r.shape) since r was freed above.
        err = _prolongate_2d(err_c, f.shape)
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + err

        # Post-smooth (fine level): native tiled kernel
        cp0, cm0 = ch[1:, :], ch[:-1, :]
        cp1, cm1 = cv[:, 1:], cv[:, :-1]
        _native_rbgs_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing)
        r = _native_mg_residual_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol)

    return p, r


# ── Full 3-D V-cycle with native RBGS kernel (hybrid) ──────────────────
# Same hybrid strategy as the 2-D version: native kernel at the finest level
# only; PyTorch _vcycle_rbgs_3d for the coarse correction.
def _vcycle_rbgs_3d_native(f, p, ch, cv, cw, jcap_tol, nsmoothing):
    """Hybrid 3-D V-cycle: thread-per-cell native RBGS at fine level, PyTorch at coarse.

    Both the smoother and the residual run as native CUDA ops.  ``J`` and the
    active mask are computed in CUDA registers by ``mg_residual_3d`` — the
    two 64 MB tensors (``J`` float + ``active`` bool) that previously lived
    for the entire fine-level V-cycle no longer exist.

    The caller's ``p`` is mutated in place; the returned tensor aliases it.
    """
    # ch/cv/cw must be C-contiguous for the native RBGS kernel.
    # .contiguous() is a zero-cost pass-through when the tensor is already
    # C-contiguous (the Phase-I kernel path stores them at face-grid shape),
    # and makes a copy only on the legacy Python-BDIM path (non-contiguous
    # ghost-padded slices).
    ch = ch.contiguous()
    cv = cv.contiguous()
    cw = cw.contiguous()
    cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
    cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
    cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]

    # Pre-smooth (fine level): native kernel
    _native_rbgs_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, nsmoothing)
    # Fused residual — J and active live in CUDA registers only.
    r = _native_mg_residual_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)

    nx, ny, nz = f.shape
    if nx > 2 and ny > 2 and nz > 2:
        ch_c, cv_c, cw_c = _restrict_face_3d(ch, cv, cw)
        r_c = _restrict_residual_3d(r)
        # r is no longer needed after restriction; free it to reduce peak footprint
        # during the coarse correction + prolongation phase.
        del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2, r_c.shape[2] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        # Coarse correction: PyTorch V-cycle (reliable convergence at all sizes)
        err_c, _ = _vcycle_rbgs_3d(r_c, p_c, ch_c, cv_c, cw_c, jcap_tol, nsmoothing)
        # Use f.shape (same as the former r.shape) since r was freed above.
        err = _prolongate_3d(err_c, f.shape)
        p[1:-1, 1:-1, 1:-1] = p[1:-1, 1:-1, 1:-1] + err

        # Post-smooth (fine level): native kernel
        cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
        cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
        cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
        _native_rbgs_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, nsmoothing)
        r = _native_mg_residual_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)

    return p, r


# ── Full 2-D V-cycle with native Jacobi kernel (hybrid) ────────────────
def _vcycle_jac_2d_native(f, p, ch, cv, w, jcap_tol, nsmoothing):
    """Hybrid 2-D V-cycle: tiled native Jacobi at fine level, PyTorch at coarse.

    Residual computed via ``mg_residual_2d``; ``J``/``active`` live only in
    CUDA registers.  The caller's ``p`` is mutated in place; the returned
    tensor aliases it.
    """
    ch = ch.contiguous()
    cv = cv.contiguous()
    cp0, cm0 = ch[1:, :], ch[:-1, :]
    cp1, cm1 = cv[:, 1:], cv[:, :-1]

    # Pre-smooth (fine level): native tiled kernel
    _native_jac_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing)
    r = _native_mg_residual_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol)

    nx, ny = f.shape
    if nx > 2 and ny > 2:
        ch_c, cv_c = _restrict_face_2d(ch, cv)
        r_c = _restrict_residual_2d(r)
        # r is no longer needed after restriction; free it to reduce peak footprint
        # during the coarse correction + prolongation phase.
        del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        # Coarse correction: PyTorch V-cycle
        err_c, _ = _vcycle_jac_2d(r_c, p_c, ch_c, cv_c, w, jcap_tol, nsmoothing)
        # Use f.shape (same as the former r.shape) since r was freed above.
        err = _prolongate_2d(err_c, f.shape)
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + err

        # Post-smooth (fine level): native tiled kernel
        cp0, cm0 = ch[1:, :], ch[:-1, :]
        cp1, cm1 = cv[:, 1:], cv[:, :-1]
        _native_jac_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing)
        r = _native_mg_residual_2d(p, f, cp0, cm0, cp1, cm1, jcap_tol)

    return p, r


# ── Full 3-D V-cycle with native Jacobi kernel (hybrid) ────────────────
def _vcycle_jac_3d_native(f, p, ch, cv, cw, w, jcap_tol, nsmoothing):
    """Hybrid 3-D V-cycle: native Jacobi at fine level, PyTorch at coarse.

    Residual computed via ``mg_residual_3d``; ``J``/``active`` live only in
    CUDA registers.  The caller's ``p`` is mutated in place; the returned
    tensor aliases it.
    """
    ch = ch.contiguous()
    cv = cv.contiguous()
    cw = cw.contiguous()
    cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
    cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
    cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]

    # Pre-smooth (fine level): native double-buffer kernel
    _native_jac_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, w, nsmoothing)
    r = _native_mg_residual_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)

    nx, ny, nz = f.shape
    if nx > 2 and ny > 2 and nz > 2:
        ch_c, cv_c, cw_c = _restrict_face_3d(ch, cv, cw)
        r_c = _restrict_residual_3d(r)
        # r is no longer needed after restriction; free it to reduce peak footprint
        # during the coarse correction + prolongation phase.
        del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2, r_c.shape[2] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        # Coarse correction: PyTorch V-cycle
        err_c, _ = _vcycle_jac_3d(r_c, p_c, ch_c, cv_c, cw_c, w, jcap_tol, nsmoothing)
        # Use f.shape (same as the former r.shape) since r was freed above.
        err = _prolongate_3d(err_c, f.shape)
        p[1:-1, 1:-1, 1:-1] = p[1:-1, 1:-1, 1:-1] + err

        # Post-smooth (fine level): native double-buffer kernel
        cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
        cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
        cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
        _native_jac_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, w, nsmoothing)
        r = _native_mg_residual_3d(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)

    return p, r


# ── Full 2-D V-cycle with Jacobi (coarse-level recursive) ──────────────────
def _vcycle_jac_2d(f, p, ch, cv, w, jcap_tol, nsmoothing):
    """Complete 2-D V-cycle with Jacobi smoother (used at coarse levels)."""
    p = p.clone()
    cp0, cm0 = ch[1:, :], ch[:-1, :]
    cp1, cm1 = cv[:, 1:], cv[:, :-1]

    # pre-smooth
    p, r = _jacobi_2d(f, p, cp0, cm0, cp1, cm1,
                       w, jcap_tol, nsmoothing)

    nx, ny = f.shape
    if nx > 2 and ny > 2:
        ch_c, cv_c = _restrict_face_2d(ch, cv)
        r_c = _restrict_residual_2d(r)

        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)

        err_c, _ = _vcycle_jac_2d(r_c, p_c, ch_c, cv_c,
                                   w, jcap_tol, nsmoothing)

        err = _prolongate_2d(err_c, r.shape)
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + err

        cp0, cm0 = ch[1:, :], ch[:-1, :]
        cp1, cm1 = cv[:, 1:], cv[:, :-1]
        p, r = _jacobi_2d(f, p, cp0, cm0, cp1, cm1,
                           w, jcap_tol, nsmoothing)

    return p, r


# ── Full 2-D V-cycle with RBGS (coarse-level recursive) ────────────────────
def _vcycle_rbgs_2d(f, p, ch, cv, jcap_tol, nsmoothing):
    """Complete 2-D V-cycle with RBGS smoother (used at coarse levels)."""
    p = p.clone()
    cp0, cm0 = ch[1:, :], ch[:-1, :]
    cp1, cm1 = cv[:, 1:], cv[:, :-1]

    red, black = _rb_masks_2d(f.shape[0], f.shape[1], p.device)
    p, r = _rbgs_2d(f, p, cp0, cm0, cp1, cm1,
                     jcap_tol, nsmoothing, red, black)

    nx, ny = f.shape
    if nx > 2 and ny > 2:
        ch_c, cv_c = _restrict_face_2d(ch, cv)
        r_c = _restrict_residual_2d(r)

        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)

        err_c, _ = _vcycle_rbgs_2d(r_c, p_c, ch_c, cv_c,
                                    jcap_tol, nsmoothing)

        err = _prolongate_2d(err_c, r.shape)
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + err

        cp0, cm0 = ch[1:, :], ch[:-1, :]
        cp1, cm1 = cv[:, 1:], cv[:, :-1]
        # red/black masks are shape-dependent only; reuse from pre-smooth.
        p, r = _rbgs_2d(f, p, cp0, cm0, cp1, cm1,
                         jcap_tol, nsmoothing, red, black)

    return p, r


# =====================================================================
# Slicing helpers
# =====================================================================

def _sl(ndim, dim, s):
    """N-D index tuple: slice *s* on dimension *dim*, full elsewhere."""
    idx = [slice(None)] * ndim
    idx[dim] = s
    return tuple(idx)


def _inner(ndim):
    """Index tuple selecting interior cells: [1:-1] on every dimension."""
    return tuple(slice(1, -1) for _ in range(ndim))


# =====================================================================
# Poisson solver
# =====================================================================

class PoissonSolver:
    """Variable-coefficient multigrid Poisson solver (2-D / 3-D).

    Supports two top-level solve strategies:

    * ``solve_multigrid`` — standalone geometric V-cycles (original).
    * ``solve_mgcg``      — Conjugate Gradient with V-cycle preconditioner
      (multigrid-preconditioned CG, a.k.a. MGCG).  Provably optimal for
      problems with large coefficient jumps (e.g. BDIM immersed bodies).
    """

    def __init__(
        self,
        dtype,
        device,
        h,
        tol=1e-2,
        max_cycles=2,
        max_vcycles=1,
        nsmoothing=2,
        w=1,
        verbose=True,
        precond_vcycles=1,
        smoother="jacobi",
        use_kernels=False,
        recycle_k=0,
    ):
        self.dtype       = dtype
        self.h2          = h * h
        self.device      = device
        self.tol         = torch.tensor(tol, dtype=torch.float32, device=device)
        self._tol_float  = tol   # keep raw float for print formatting
        self.max_cycles  = max_cycles
        self.max_vcycles = max_vcycles
        self.nsmoothing  = nsmoothing
        self.verbose     = verbose
        self.jcap_tol    = 1e-12 # lower value helps to reduce degenerate
        self.n_switch    = 2 ** 16
        self.w           = w   # Jacobi relaxation weight
        self.precond_vcycles = precond_vcycles  # V-cycles per CG preconditioner
        assert smoother in ("jacobi", "rbgs"), \
            f"smoother must be 'jacobi' or 'rbgs', got '{smoother}'"
        self.smoother = smoother
        self._rb_mask_cache = {}  # {(shape, device): (red, black)}
        # Optional cell-centred Dirichlet mask (inner shape, bool).
        # When set (e.g. by the free-surface ghost-fluid layer), the
        # smoother forces ``p == 0`` in masked cells after every sweep
        # and at every multigrid level (coarse masks are built by OR
        # downsampling: any-air → coarse-air).  This is the cheap way to
        # enforce a homogeneous Dirichlet boundary on an arbitrary
        # subset of interior cells without touching the per-face
        # coefficient layout.
        self.dirichlet_mask = None
        self.use_kernels = use_kernels
        # ---- Recycled-Krylov (deflation) state -------------------------
        # When recycle_k > 0, solve_rmgcg keeps a small subspace of search
        # directions from previous solves and deflates them out of the next
        # solve.  Because the operator (ch/cv/cw) changes only slightly per
        # timestep, those directions span the slow-converging modes, so the
        # deflated CG converges in far fewer iterations.  Persists across
        # calls on the (long-lived) solver instance; reset on shape change
        # or Cholesky breakdown (stale-space guard).
        self.recycle_k = recycle_k
        self._recycle = None          # {"U": [full-grid dirs]} or None
        self._recycle_cooldown = 0    # steps to stay disengaged after a stall
        self._rmgcg_warned = False
        # Init-only augmentation (project the recycle space out of the initial
        # guess, then run ordinary MGCG) is provably never slower than plain
        # MGCG.  Full in-loop deflation can be faster but is fragile with a
        # non-deflated V-cycle preconditioner (observed to stall in 3-D), so it
        # is OFF by default; flip for experimentation only.
        self._deflate_in_loop = True
        if use_kernels:
            from lilytorch.src.kernels import _C  # noqa: F401  load .so
            from lilytorch.src.kernels import ops as _K
            self._K = _K

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def l2_norm(r):
        return torch.linalg.vector_norm(r)

    @staticmethod
    def _convergence_norm(r):
        """L-infinity norm: returns the exact maximum element — no floating-point
        summation — so it is deterministic on both CPU and CUDA.  Using this for
        the early-exit test guarantees that GPU and CPU perform the same number
        of V-cycles, eliminating pressure-field divergence between backends."""
        return torch.max(torch.abs(r))

    @staticmethod
    def BC(q):
        """Zero-gradient (Neumann) BCs on all faces."""
        ndim = q.ndim
        for d in range(ndim):
            dst = [slice(None)] * ndim; dst[d] = 0
            src = [slice(None)] * ndim; src[d] = 1
            q[tuple(dst)] = q[tuple(src)]
            dst = [slice(None)] * ndim; dst[d] = -1
            src = [slice(None)] * ndim; src[d] = -2
            q[tuple(dst)] = q[tuple(src)]

    # ------------------------------------------------------------------
    # Stencil operations  (dimension-agnostic)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_sum(cfaces, p):
        """Sum of  c_{d+} p_{i+1} + c_{d-} p_{i-1}  over all dims.

        cfaces : list of (c_plus, c_minus) per dimension.
        """
        ndim  = p.ndim
        inner = _inner(ndim)
        s = torch.zeros_like(p[inner])
        for d, (cp, cm) in enumerate(cfaces):
            fwd = list(inner); fwd[d] = slice(2, None)
            bwd = list(inner); bwd[d] = slice(None, -2)
            s = s + cp * p[tuple(fwd)] + cm * p[tuple(bwd)]
        return s

    @staticmethod
    def compute_J(cfaces):
        """Diagonal: J = sum_d (c_{d+} + c_{d-})."""
        J = None
        for cp, cm in cfaces:
            contrib = cp + cm
            J = contrib if J is None else J + contrib
        return J

    # ------------------------------------------------------------------
    # Jacobi smoother
    # ------------------------------------------------------------------
    def Jacobi(self, f, p, cfaces):
        self.BC(p)
        J    = self.compute_J(cfaces)
        active = torch.abs(J) >= self.jcap_tol          # fluid mask
        Jinv = torch.where(active, J.reciprocal(), torch.zeros_like(J))
        inner = _inner(p.ndim)
        dmask = getattr(self, "_active_dirichlet_mask", None)

        for _ in range(self.nsmoothing):
            s = self.compute_sum(cfaces, p)
            p[inner] = self.w * (-f + s) * Jinv + (1 - self.w) * p[inner]
            if dmask is not None:
                p[inner].masked_fill_(dmask, 0.0)
            self.BC(p)

        # residual — zero at degenerate cells (cf. WaterLily residual!)
        s  = self.compute_sum(cfaces, p)
        Au = (s - J * p[inner])
        r  = torch.where(active, f - Au, torch.zeros_like(f))
        return p, r

    # ------------------------------------------------------------------
    # Red-Black Gauss-Seidel smoother
    # ------------------------------------------------------------------
    def _build_rb_masks(self, shape):
        """Build red/black masks for interior cells (cached per shape).

        Red cells: sum of (0-based interior) indices is even.
        Black cells: sum is odd.
        Both masks have the shape of the *interior* grid (no ghost cells).
        """
        key = (shape, self.device)
        if key in self._rb_mask_cache:
            return self._rb_mask_cache[key]
        ndim = len(shape)
        # Build coordinate grids for the interior (each starting at 0)
        ranges = [torch.arange(s, device=self.device) for s in shape]
        grids  = torch.meshgrid(*ranges, indexing="ij")
        parity = sum(grids) % 2            # 0 = red, 1 = black
        red   = (parity == 0)
        black = (parity == 1)
        self._rb_mask_cache[key] = (red, black)
        return red, black

    def RBGS(self, f, p, cfaces):
        """Red-Black Gauss-Seidel smoother.

        Sweeps red cells (sum of interior indices even), then black cells,
        updating p in-place.  Each colour update reads only neighbours of
        the opposite colour, so the ordering is consistent.
        """
        self.BC(p)
        ndim  = p.ndim
        inner = _inner(ndim)
        J     = self.compute_J(cfaces)
        active = torch.abs(J) >= self.jcap_tol
        Jinv  = torch.where(active, 1 / J, torch.zeros_like(J))

        interior_shape = p[inner].shape
        red, black = self._build_rb_masks(interior_shape)
        dmask = getattr(self, "_active_dirichlet_mask", None)

        for _ in range(self.nsmoothing):
            # --- red sweep ---
            s = self.compute_sum(cfaces, p)
            p_new = (-f + s) * Jinv
            p[inner] = torch.where(red, p_new, p[inner])
            if dmask is not None:
                p[inner].masked_fill_(dmask, 0.0)
            self.BC(p)

            # --- black sweep ---
            s = self.compute_sum(cfaces, p)
            p_new = (-f + s) * Jinv
            p[inner] = torch.where(black, p_new, p[inner])
            if dmask is not None:
                p[inner].masked_fill_(dmask, 0.0)
            self.BC(p)

        # residual
        s  = self.compute_sum(cfaces, p)
        Au = (s - J * p[inner])
        r  = torch.where(active, f - Au, torch.zeros_like(f))
        return p, r

    # ------------------------------------------------------------------
    # Smoother dispatch
    # ------------------------------------------------------------------
    def smooth(self, f, p, cfaces):
        """Dispatch to the configured smoother (used by the Python-recursive _vcycle)."""
        if self.smoother == "rbgs":
            p, r = self.RBGS(f, p, cfaces)
        else:
            p, r = self.Jacobi(f, p, cfaces)
        m = getattr(self, "_active_dirichlet_mask", None)
        if m is not None:
            # Force p = 0 in masked (air) cells AND zero the residual
            # there so it does not pollute the coarse-grid restriction.
            inner = _inner(p.ndim)
            p[inner].masked_fill_(m, 0.0)
            r.masked_fill_(m, 0.0)
        return p, r

    @staticmethod
    def _coarsen_mask(mask):
        """Coarsen a bool Dirichlet mask by stride-2 OR-downsampling.

        Any fine cell flagged as Dirichlet causes the enclosing coarse
        cell to also be flagged (conservative: more cells get pinned).
        Returns ``None`` if the result would be smaller than 1 along
        any axis (caller should skip recursion in that case).
        """
        if mask is None:
            return None
        m = mask
        ndim = m.ndim
        for d in range(ndim):
            even = m[_sl(ndim, d, slice(0, None, 2))]
            odd  = m[_sl(ndim, d, slice(1, None, 2))]
            n = min(even.shape[d], odd.shape[d])
            m = (even[_sl(ndim, d, slice(n))] |
                 odd[_sl(ndim, d, slice(n))])
        return m

    # ------------------------------------------------------------------
    # Face array helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_cfaces(face_arrs, ndim):
        """Extract (c_plus, c_minus) pairs from full face arrays.

        For face array cf along dimension d:
          c_plus  = cf[1:]  along dim d
          c_minus = cf[:-1] along dim d
        """
        cfaces = []
        for d, cf in enumerate(face_arrs):
            cp = cf[_sl(ndim, d, slice(1, None))]
            cm = cf[_sl(ndim, d, slice(None, -1))]
            cfaces.append((cp, cm))
        return cfaces

    @staticmethod
    def _default_face_arrs(c, ndim):
        """Build face-averaged coefficient arrays from cell-centred c.

        Returns list of face arrays per dimension, matching the old
        convention:
          ch = 0.5*(c[1:, 1:-1] + c[:-1, 1:-1])   shape (nx-1, ny-2)
          cv = 0.5*(c[1:-1, 1:] + c[1:-1, :-1])   shape (nx-2, ny-1)
        """
        face_arrs = []
        for d in range(ndim):
            idx_fwd = [slice(1, -1)] * ndim
            idx_fwd[d] = slice(1, None)
            idx_bwd = [slice(1, -1)] * ndim
            idx_bwd[d] = slice(None, -1)
            face_arrs.append(0.5 * (c[tuple(idx_fwd)] + c[tuple(idx_bwd)]))
        return face_arrs

    @staticmethod
    def _face_arrs_from_kwargs(kwargs, ndim):
        """Extract face arrays [ch, cv(, cw)] from kwargs.

        Returns (face_arrs, remaining_kwargs) or (None, kwargs).
        """
        labels = ["ch", "cv", "cw"][:ndim]
        if not all(lab in kwargs for lab in labels):
            return None, kwargs
        remaining = dict(kwargs)
        face_arrs = [remaining.pop(lab) for lab in labels]
        return face_arrs, remaining

    # ------------------------------------------------------------------
    # V-cycle dispatch
    # ------------------------------------------------------------------
    def _dispatch_vcycle(self, f, p, face_arrs):
        """Run one V-cycle through the native hybrid path.

        The caller (project) is OK with in-place mutation of its p input:
        the returned tensor is assigned back to fs.p0 immediately, so the
        fine-level p.clone() is skipped to save ~512 MB at 512³ float32.
        """
        # The native hybrid V-cycle uses CUDA-only smoother kernels.  Off CUDA
        # (CPU path), fall back to the pure-PyTorch recursive V-cycle so the
        # CG solvers (solve_mgcg / solve_rmgcg) run on CPU tensors too.
        if not p.is_cuda:
            return self._vcycle(f, p, face_arrs)

        ndim = f.ndim
        if ndim == 3:
            ch, cv, cw = face_arrs
            if self.smoother == "rbgs":
                return _vcycle_rbgs_3d_native(
                    f, p, ch, cv, cw, self.jcap_tol, self.nsmoothing)
            return _vcycle_jac_3d_native(
                f, p, ch, cv, cw, self.w, self.jcap_tol, self.nsmoothing)
        ch, cv = face_arrs
        if self.smoother == "rbgs":
            return _vcycle_rbgs_2d_native(
                f, p, ch, cv, self.jcap_tol, self.nsmoothing)
        return _vcycle_jac_2d_native(
            f, p, ch, cv, self.w, self.jcap_tol, self.nsmoothing)

    # ------------------------------------------------------------------
    # V-cycle  (dimension-agnostic, recursive)
    # ------------------------------------------------------------------
    def _vcycle(self, f, p, face_arrs):
        """Internal V-cycle operating on full face arrays."""
        ndim  = f.ndim
        shape = f.shape

        # ---- Dirichlet mask (free-surface): pick / save / restore ----
        # The top-level caller stores the fine mask on self.dirichlet_mask;
        # within a recursion we read the current-level mask off
        # self._active_dirichlet_mask, restrict it for the coarse call,
        # and restore on the way back up.
        outer_mask = getattr(self, "_active_dirichlet_mask", None)
        if outer_mask is None:
            outer_mask = self.dirichlet_mask
        self._active_dirichlet_mask = (
            outer_mask if (outer_mask is None or outer_mask.shape == shape)
            else None
        )

        # extract (cp, cm) for the smoother
        cfaces = self._extract_cfaces(face_arrs, ndim)

        # pre-smooth
        p, r = self.smooth(f, p, cfaces)

        # coarsen if grid is large enough
        if all(n > 8 for n in shape):

            # CPU offload for very large grids
            on_gpu = (self.device == "cuda"
                      and max(shape) >= self.n_switch)
            if on_gpu:
                f         = f.cpu()
                p         = p.cpu()
                r         = r.cpu()
                face_arrs = [cf.cpu() for cf in face_arrs]

            # ---- restriction of face arrays --------------------------
            # Matches WaterLily.jl's restrictL:
            #   L_coarse[I,i] = 0.5 * sum_{J in up(I,i)} L[J,i]
            # i.e. stride-2 in face direction, SUM in transverse
            # directions, then a single 0.5 factor.
            # In 2D this equals the old  0.5*(even+odd) per transverse dim.
            # In 3D the old code applied 0.5 per transverse dim, giving
            # (0.5)^(ndim-1)*sum instead of the correct 0.5*sum, which
            # made the coarse diagonal too small and caused divergence.
            face_arrs_coarse = []
            for d, cf in enumerate(face_arrs):
                cf_c = cf
                for d2 in range(ndim):
                    if d2 == d:
                        cf_c = cf_c[_sl(ndim, d2, slice(None, None, 2))]
                    else:
                        even = cf_c[_sl(ndim, d2, slice(None, -1, 2))]
                        odd  = cf_c[_sl(ndim, d2, slice(1, None, 2))]
                        cf_c = even + odd          # SUM (not average)
                cf_c = cf_c.mul_(0.5)              # single 0.5 factor (in-place)
                face_arrs_coarse.append(cf_c)

            # ---- restriction of residual (full-weighting) ------------
            # No .clone() needed: each slicing step creates a new tensor
            r_coarse = r
            for d in range(ndim):
                even = r_coarse[_sl(ndim, d, slice(0, None, 2))]
                odd  = r_coarse[_sl(ndim, d, slice(1, None, 2))]
                m = min(even.shape[d], odd.shape[d])
                r_coarse = (even[_sl(ndim, d, slice(m))] +
                            odd[_sl(ndim, d, slice(m))])

            # coarse-grid error
            coarse_shape = tuple(s + 2 for s in r_coarse.shape)
            saved_mask = self._active_dirichlet_mask
            coarse_mask = self._coarsen_mask(saved_mask)
            self._active_dirichlet_mask = coarse_mask
            err_coarse, _ = self._vcycle(
                r_coarse,
                torch.zeros(coarse_shape, device=p.device, dtype=p.dtype),
                face_arrs_coarse,
            )
            self._active_dirichlet_mask = saved_mask

            # ---- prolongation (trilinear / bilinear) -----------------
            inner_c = _inner(ndim)
            ec = err_coarse[inner_c]
            mode = 'trilinear' if ndim == 3 else 'bilinear'
            ec_nd = ec.unsqueeze(0).unsqueeze(0)
            err = torch.nn.functional.interpolate(
                ec_nd, size=r.shape, mode=mode, align_corners=False,
            )[0, 0]

            # correction
            p[_inner(ndim)] += err
            m = self._active_dirichlet_mask
            if m is not None:
                p[_inner(ndim)].masked_fill_(m, 0.0)

            if on_gpu:
                f         = f.cuda()
                p         = p.cuda()
                face_arrs = [cf.cuda() for cf in face_arrs]
                # re-extract after device transfer
                cfaces = self._extract_cfaces(face_arrs, ndim)

            # post-smooth
            p, r = self.smooth(f, p, cfaces)

        return p, r

    # ------------------------------------------------------------------
    # Public V-cycle  (wrapper that builds face_arrs from kwargs)
    # ------------------------------------------------------------------
    def vcycle(self, f, p, **kwargs):
        """V-cycle with face-coefficient arrays ch/cv(/cw)."""
        ndim = f.ndim
        face_arrs, kwargs = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "vcycle: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required."
            )
        return self._vcycle(f, p, face_arrs)

    # ------------------------------------------------------------------
    # Top-level solve
    # ------------------------------------------------------------------
    def _solve_multigrid_native(self, f, p0, face_arrs, ndim):
        """Dispatch solve_multigrid to native CUDA driver."""
        p = p0 if p0.is_contiguous() else p0.contiguous()
        f_c = f.contiguous()
        if ndim == 2:
            ch, cv = face_arrs
            r = self._K.poisson_solve_multigrid_2d(
                p, f_c, ch.contiguous(), cv.contiguous(),
                h2=self.h2, jcap_tol=self.jcap_tol, w=self.w,
                nsmoothing=self.nsmoothing, max_vcycles=self.max_vcycles,
                tol=self._tol_float, smoother=self.smoother,
            )
        else:
            ch, cv, cw = face_arrs
            r = self._K.poisson_solve_multigrid_3d(
                p, f_c, ch.contiguous(), cv.contiguous(), cw.contiguous(),
                h2=self.h2, jcap_tol=self.jcap_tol, w=self.w,
                nsmoothing=self.nsmoothing, max_vcycles=self.max_vcycles,
                tol=self._tol_float, smoother=self.smoother,
            )
        return p, r

    def _solve_mgcg_native(self, f, p0, face_arrs, ndim):
        """Dispatch solve_mgcg to native CUDA driver."""
        p = p0 if p0.is_contiguous() else p0.contiguous()
        f_c = f.contiguous()
        if ndim == 2:
            ch, cv = face_arrs
            r = self._K.poisson_solve_mgcg_2d(
                p, f_c, ch.contiguous(), cv.contiguous(),
                h2=self.h2, jcap_tol=self.jcap_tol, w=self.w,
                nsmoothing=self.nsmoothing, max_cycles=self.max_cycles,
                precond_vcycles=self.precond_vcycles,
                tol=self._tol_float, smoother=self.smoother,
            )
        else:
            ch, cv, cw = face_arrs
            r = self._K.poisson_solve_mgcg_3d(
                p, f_c, ch.contiguous(), cv.contiguous(), cw.contiguous(),
                h2=self.h2, jcap_tol=self.jcap_tol, w=self.w,
                nsmoothing=self.nsmoothing, max_cycles=self.max_cycles,
                precond_vcycles=self.precond_vcycles,
                tol=self._tol_float, smoother=self.smoother,
            )
        return p, r

    def solve_multigrid(self, f, p0, **kwargs):
        """Solve with multigrid V-cycles.

        Parameters
        ----------
        f  : RHS on the interior grid  (no ghost cells)
        p0 : initial guess (with ghost cells)
        ch, cv[, cw] : pre-computed face-averaged coefficients
        pre_scaled : bool, optional
            When True, *f* is already scaled by ``h²``, so the internal
            ``f_scaled = h² * f`` multiplication is skipped (T3a: saves one
            interior-sized allocation on the Python path).  Ignored when
            ``use_kernels=True`` — the native solver applies h² internally.
        """
        pre_scaled = kwargs.pop('pre_scaled', False)
        ndim = f.ndim
        face_arrs, _ = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_multigrid: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required."
            )

        if self.use_kernels:
            return self._solve_multigrid_native(f, p0, face_arrs, ndim)

        # p0 is passed directly; the vcycle clones its input internally,
        # so the redundant clone here is unnecessary and wastes 128-131 MB.
        p = p0
        # T3a: skip h² multiplication when caller has pre-scaled f.
        f_scaled = f if pre_scaled else self.h2 * f
        cycles_run = self.max_vcycles
        for i in range(self.max_vcycles):
            p, r = self._dispatch_vcycle(f_scaled, p, face_arrs)
            r_err = self._convergence_norm(r)
            if r_err < self.tol:
                cycles_run = i + 1
                break
        # float64 mean subtraction: GPU parallel-reduction of float32 gives
        # a different value than CPU sequential sum.
        # Skip when a Dirichlet mask pins p in (a subset of) cells — the
        # null space is removed by the Dirichlet condition, so the
        # absolute level of p is meaningful and must NOT be shifted.
        if self.dirichlet_mask is None:
            p -= p.to(torch.float64).mean().to(p.dtype)
        if self.verbose:
            print(
                f"Multigrid residual = {self.l2_norm(r):.2e}/{self._tol_float:.2e} "
                f"with {cycles_run}/{self.max_vcycles} cycles"
            )
        return p, r

    # ------------------------------------------------------------------
    # SPD operator for CG
    # ------------------------------------------------------------------
    def _apply_op_spd(self, p, cfaces):
        """Apply the SPD operator B(p) = J·p[inner] - compute_sum(cfaces, p).

        This is the discrete *negative* Laplacian with variable coefficients
        scaled by h² (since the V-cycle works with h²-scaled quantities).
        Positive semi-definite: p^T B p ≥ 0  (kernel = constants).

        Degenerate (solid) cells are zeroed out, consistent with the Jacobi
        masking in the V-cycle.
        """
        self.BC(p)
        ndim  = p.ndim
        inner = _inner(ndim)
        J = self.compute_J(cfaces)
        active = J.abs() >= self.jcap_tol
        s = self.compute_sum(cfaces, p)
        s.addcmul_(J, p[inner], value=-1.0)    # s = sum - J*p  (in-place)
        del J
        s.neg_().mul_(active)                   # s = (J*p - sum) * active = result, in-place
        return s

    # ------------------------------------------------------------------
    # MGCG  (multigrid-preconditioned conjugate gradient)
    # ------------------------------------------------------------------
    def solve_mgcg(self, f, p0, **kwargs):
        """Solve with CG using geometric multigrid V-cycles as preconditioner.

        This is the standard MGCG algorithm:  at each CG iteration the
        search direction is preconditioned by approximately inverting the
        operator with ``precond_vcycles`` V-cycles (default 1).

        Advantages over standalone ``solve_multigrid``:

        * CG minimises the error in the A-norm over the Krylov subspace,
          giving *provably optimal* convergence — standalone V-cycles can
          stall on problems with large coefficient contrasts.
        * For smooth flows, MGCG typically converges in 3–6 CG iterations
          (each with 1 V-cycle), comparable to 3–6 standalone V-cycles but
          with a guaranteed residual reduction at every step.

        Parameters  (identical to ``solve_multigrid``)
        ----------------------------------------------
        f  : RHS on the interior grid  (no ghost cells)
        p0 : initial guess (with ghost cells)
        ch, cv[, cw] : pre-computed face-averaged coefficients
        """
        pre_scaled = kwargs.pop('pre_scaled', False)
        ndim = f.ndim
        face_arrs, extra = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_mgcg: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required."
            )

        if self.use_kernels:
            return self._solve_mgcg_native(f, p0, face_arrs, ndim)
        cfaces = self._extract_cfaces(face_arrs, ndim)

        # ------ SPD system:  B(x) = b  where B = Jp - S,  b = -(h²·f) ------
        # T3a: when f is already h²-scaled, skip the multiplication.
        b = -f if pre_scaled else -(self.h2 * f)
        x = p0.clone().detach()
        self.BC(x)

        # Plain MGCG: shared CG core with no deflation / no harvesting.
        x, r, niter, r_norm_final = self._cg_core(
            b, x, cfaces, face_arrs, recycle=None, harvest=None,
        )

        if self.verbose:
            if niter == 0:
                print(f"MGCG converged at initial guess: "
                      f"residual = {r_norm_final:.2e}")
            else:
                print(
                    f"MGCG residual = {r_norm_final:.2e}/{self._tol_float:.2e} "
                    f"with {niter}/{self.max_cycles} CG iterations "
                    f"({self.precond_vcycles} V-cycle"
                    f"{'s' if self.precond_vcycles > 1 else ''}/iter)"
                )
        return x, r

    # ------------------------------------------------------------------
    # Shared (deflated) CG core  — single source of truth for MGCG/RMGCG
    # ------------------------------------------------------------------
    def _cg_core(self, b, x, cfaces, face_arrs, recycle=None, harvest=None):
        """Multigrid-preconditioned CG loop, optionally deflated.

        Parameters
        ----------
        b         : RHS of the SPD system  B(x) = b  (interior-sized).
        x         : initial guess (full grid, ghost cells); modified in place.
        cfaces    : per-dim (c_plus, c_minus) coefficient tuples.
        face_arrs : raw face arrays for the V-cycle preconditioner.
        recycle   : ``None`` for plain MGCG, else a dict ``{U, W, chol}``
                    describing the deflation subspace (see ``_prepare_recycle``).
        harvest   : ``None`` or a list to which each CG search direction is
                    appended (used by RMGCG to refresh the recycle space).

        Returns ``(x, r, niter, r_norm_final)``.  With ``recycle is None`` and
        ``harvest is None`` this reproduces the previous ``solve_mgcg`` loop
        exactly — that is the single source of truth both methods share.
        """
        inner = _inner(b.ndim)

        # Initial residual: r = b - B(x)
        r = b - self._apply_op_spd(x, cfaces)

        # Deflation init: project the recycle subspace out of (x, r) so the
        # CG iteration never has to rediscover those modes.
        if recycle is not None:
            self._deflate_init(x, r, recycle, inner)

        r_norm = self._convergence_norm(r)
        if r_norm < self.tol:
            x -= x.to(torch.float64).mean().to(x.dtype)
            self._last_niter = 0
            return x, r, 0, r_norm

        # Preconditioner: approximately solve B(z) = r via V-cycle(s).
        # The V-cycle solves (S - Jp) = f_arg, i.e. -B(p) = f_arg, so we pass
        # f_arg = -r  →  -B(z) ≈ -r  →  B(z) ≈ r.
        z = torch.zeros_like(x)
        for _ in range(self.precond_vcycles):
            z, _ = self._dispatch_vcycle(-r, z, face_arrs)

        d = z.clone()                              # search direction
        if recycle is not None and self._deflate_in_loop:
            self._deflate_proj(d, z, recycle, inner)   # B-orthogonalise vs U
        self.BC(d)
        rz = (r * z[inner]).to(torch.float64).sum().to(r.dtype)  # r · M⁻¹r

        r_norm_final = r_norm
        k = 0
        for k in range(self.max_cycles):
            q = self._apply_op_spd(d, cfaces)      # q = B·d
            dq = (d[inner] * q).to(torch.float64).sum().to(r.dtype)  # d · B·d
            alpha = rz / dq                        # step length

            x[inner] = x[inner] + alpha * d[inner]
            self.BC(x)
            r = r - alpha * q

            if harvest is not None:
                harvest.append(d.clone())

            r_norm_final = self._convergence_norm(r)
            if r_norm_final < self.tol:
                break

            # --- preconditioner (reuse buffer) ---
            z.zero_()
            for _ in range(self.precond_vcycles):
                z, _ = self._dispatch_vcycle(-r, z, face_arrs)

            rz_new = (r * z[inner]).to(torch.float64).sum().to(r.dtype)
            beta = rz_new / rz
            d[inner] = z[inner] + beta * d[inner]
            if recycle is not None and self._deflate_in_loop:
                self._deflate_proj(d, z, recycle, inner)
            self.BC(d)
            rz = rz_new

        x -= x.to(torch.float64).mean().to(x.dtype)
        niter = min(k + 1, self.max_cycles)
        self._last_niter = niter          # exposed for benchmarking/diagnostics
        return x, r, niter, r_norm_final

    # ------------------------------------------------------------------
    # Recycled-Krylov (deflated MGCG) helpers
    # ------------------------------------------------------------------
    # The recycle space is kept B-ORTHONORMAL (Qᵀ B Q = I) under the current
    # operator, so the Gram matrix is the identity — no matrix solve, and the
    # deflation cannot be poisoned by near-dependent stored directions (they
    # are dropped during re-orthonormalisation).  ``rec["U"]`` holds the
    # B-orthonormal basis q_j and ``rec["W"]`` the matching w_j = B q_j.
    def _deflate_init(self, x, r, rec, inner):
        """Galerkin solve in the recycle space (C = I):
        x += Σ_j (q_jᵀr) q_j ;  r -= Σ_j (q_jᵀr) w_j   ⟹   Qᵀr = 0 afterwards."""
        for q, w in zip(rec["U"], rec["W"]):
            mu = (q[inner] * r).to(torch.float64).sum().to(r.dtype)  # q_jᵀ r
            x[inner] += mu * q[inner]
            r -= mu * w
        self.BC(x)

    def _deflate_proj(self, d, z, rec, inner):
        """B-orthogonalise the search direction against the recycle space:
        d -= Σ_j (w_jᵀz) q_j   (so d_new is B-orthogonal to every q_j)."""
        for q, w in zip(rec["U"], rec["W"]):
            nu = (w * z[inner]).to(torch.float64).sum().to(d.dtype)  # (B q_j)ᵀ z
            d[inner] -= nu * q[inner]

    def _prepare_recycle(self, cfaces, shape, inner):
        """B-orthonormalise the stored directions under the *current* operator.

        The raw directions saved last step were B-orthonormal under last step's
        operator; here we re-orthonormalise them under the current B via
        modified Gram-Schmidt in the B-inner-product, dropping any vector whose
        B-norm collapses (linearly dependent / stale).  The result satisfies
        Qᵀ B Q = I exactly, so no Gram-matrix inversion is needed and the
        deflation is numerically robust even when the operator has drifted.
        Returns ``None`` (→ plain MGCG this step) if nothing survives.
        """
        if self.recycle_k <= 0 or self._recycle is None:
            return None
        raw = self._recycle["U"]
        if not raw or tuple(raw[0].shape) != tuple(shape):
            self._recycle = None
            return None

        # Relative drop tolerance on the Rayleigh quotient uᵀBu/uᵀu.  Vectors
        # whose B-norm collapses relative to the strongest survivor are either
        # linearly dependent or live in the (near-)null space of B — the
        # all-Neumann constant mode and its numerical neighbours.  Deflating
        # those is both useless (the gauge handles the constant) and unstable
        # (catastrophic cancellation in w = B u corrupts orthonormality), so
        # they are dropped rather than amplified by the 1/β normalisation.
        droptol = 1e-4
        Q, W = [], []
        rq_max = 0.0
        for u0 in raw:
            u = u0.clone()
            u[inner] -= u[inner].to(torch.float64).mean().to(u.dtype)  # kill constant
            unorm2 = (u[inner] * u[inner]).to(torch.float64).sum()
            if unorm2 <= 0:
                continue
            w = self._apply_op_spd(u, cfaces)        # w = B u (interior)
            # MGS in the B-inner-product against the accepted basis.
            for q, wq in zip(Q, W):
                proj = (q[inner] * w).to(torch.float64).sum()   # q_jᵀ B u
                u[inner] -= proj.to(u.dtype) * q[inner]
                w -= proj.to(w.dtype) * wq                       # keep w = B u
            bnorm2 = (u[inner] * w).to(torch.float64).sum()      # uᵀ B u
            rq = (bnorm2 / unorm2).item()                        # Rayleigh quotient
            rq_max = max(rq_max, rq)
            if bnorm2 <= 0 or rq <= droptol * rq_max:
                continue                                          # drop near-null/dependent
            beta = bnorm2.sqrt()
            Q.append(u / beta.to(u.dtype))
            W.append(w / beta.to(w.dtype))

        if not Q:
            self._recycle = None
            return None
        return {"U": Q, "W": W}

    def _update_recycle(self, harvest, inner):
        """Refresh the stored directions with this solve's search directions.

        Pools the previous (orthonormal) basis with the newly harvested,
        L2-normalised directions and subsamples evenly across the pool to keep
        ``recycle_k`` vectors.  Even sampling (rather than newest-k) favours
        spectral diversity, which matters because consecutive late CG
        directions are nearly dependent.  Conditioning is guaranteed by the
        B-orthonormalisation in ``_prepare_recycle`` regardless of this choice.
        """
        if not harvest:
            return
        pool = [] if self._recycle is None else list(self._recycle["U"])
        for d in harvest:
            n = torch.linalg.vector_norm(d[inner])
            if n > 0:
                pool.append(d / n)
        if len(pool) > self.recycle_k:
            idx = torch.linspace(0, len(pool) - 1, self.recycle_k)
            keep = sorted(set(int(round(v)) for v in idx.tolist()))
            pool = [pool[i] for i in keep]
        self._recycle = {"U": pool}

    def _finalize_recycle(self, niter, deflated, harvest_list, inner):
        """Apply the recycle-space guards after a solve (shared py/native).

        Stall-safety: a *deflated* solve that hit the iteration cap means the
        space is actively hurting (poisoned CG recurrence) — discard it AND back
        off for a few steps so we don't immediately rebuild from the next plain
        solve and re-stall (the IQN-ILS reuse-poisoning lesson, [[project_iqn_reuse_poisoning]]).
        Harvest guard: only (re)build from genuinely iteration-bound solves
        (niter >= recycle_k); fast solves' directions don't approximate the slow
        modes, so deflating them next step would only misalign CG.
        """
        if self.recycle_k <= 0:
            return
        if deflated and niter >= self.max_cycles:
            self._recycle = None
            self._recycle_cooldown = 5
        elif self._recycle_cooldown > 0:
            self._recycle_cooldown -= 1
            self._recycle = None
        elif niter >= self.recycle_k:
            self._update_recycle(harvest_list, inner)
        else:
            self._recycle = None

    def _solve_rmgcg_native(self, f, p0, face_arrs, cfaces, ndim, inner):
        """Dispatch recycled MGCG to the native CUDA driver.

        The persistent recycle bookkeeping (``_prepare_recycle`` B-ortho­
        normalisation, ``_finalize_recycle`` guards) stays in Python; only the
        deflated CG loop runs natively.  The driver harvests the last
        ``recycle_k`` search directions into ``D`` for the next refresh.
        """
        p = p0.clone().detach()
        if not p.is_contiguous():
            p = p.contiguous()
        self.BC(p)
        f_c = f.contiguous()

        recycle = self._prepare_recycle(cfaces, p.shape, inner)
        if recycle is not None and recycle["U"]:
            U = torch.stack(recycle["U"]).contiguous()
            W = torch.stack(recycle["W"]).contiguous()
        else:
            U = p.new_empty((0, *p.shape))
            W = f_c.new_empty((0, *f_c.shape))
        hk = self.recycle_k

        if ndim == 2:
            ch, cv = face_arrs
            r, D, niter = self._K.poisson_solve_rmgcg_2d(
                p, f_c, ch.contiguous(), cv.contiguous(), U, W, hk,
                h2=self.h2, jcap_tol=self.jcap_tol, w=self.w,
                nsmoothing=self.nsmoothing, max_cycles=self.max_cycles,
                precond_vcycles=self.precond_vcycles,
                tol=self._tol_float, smoother=self.smoother,
            )
        else:
            ch, cv, cw = face_arrs
            r, D, niter = self._K.poisson_solve_rmgcg_3d(
                p, f_c, ch.contiguous(), cv.contiguous(), cw.contiguous(),
                U, W, hk,
                h2=self.h2, jcap_tol=self.jcap_tol, w=self.w,
                nsmoothing=self.nsmoothing, max_cycles=self.max_cycles,
                precond_vcycles=self.precond_vcycles,
                tol=self._tol_float, smoother=self.smoother,
            )

        deflated = U.shape[0] > 0
        harvest_list = list(D.unbind(0)) if hk > 0 else None
        self._last_niter = niter
        self._finalize_recycle(niter, deflated, harvest_list, inner)

        if self.verbose:
            n_rec = 0 if self._recycle is None else len(self._recycle["U"])
            print(
                f"RMGCG[native] residual = {r.abs().max().item():.2e}"
                f"/{self._tol_float:.2e} with {niter}/{self.max_cycles} CG "
                f"iterations (deflated {U.shape[0]} → recycle dim {n_rec})"
            )
        return p, r

    def solve_rmgcg(self, f, p0, **kwargs):
        """Recycled MGCG: ``solve_mgcg`` plus cross-timestep Krylov recycling.

        Identical interface and (with ``recycle_k == 0``) identical behaviour
        to ``solve_mgcg``.  With ``recycle_k > 0`` it deflates the subspace of
        slow-converging modes carried over from previous solves, cutting CG
        iterations for time-stepping problems whose operator changes little
        per step (e.g. slow swimmers).  Runs the native CUDA driver when
        ``use_kernels`` is set, else the PyTorch CG core; both share the recycle
        bookkeeping and guards below.
        """
        pre_scaled = kwargs.pop('pre_scaled', False)
        ndim = f.ndim
        face_arrs, _ = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_rmgcg: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required."
            )

        cfaces = self._extract_cfaces(face_arrs, ndim)
        inner  = _inner(ndim)

        if self.use_kernels:
            return self._solve_rmgcg_native(f, p0, face_arrs, cfaces, ndim, inner)

        b = -f if pre_scaled else -(self.h2 * f)
        x = p0.clone().detach()
        self.BC(x)

        recycle = self._prepare_recycle(cfaces, x.shape, inner)
        harvest = [] if self.recycle_k > 0 else None

        x, r, niter, r_norm_final = self._cg_core(
            b, x, cfaces, face_arrs, recycle=recycle, harvest=harvest,
        )

        self._finalize_recycle(niter, recycle is not None, harvest, inner)

        if self.verbose:
            n_def = 0 if recycle is None else len(recycle["U"])
            n_rec = 0 if self._recycle is None else len(self._recycle["U"])
            if niter == 0:
                print(f"RMGCG converged at initial guess: "
                      f"residual = {r_norm_final:.2e} (deflated {n_def})")
            else:
                print(
                    f"RMGCG residual = {r_norm_final:.2e}/{self._tol_float:.2e} "
                    f"with {niter}/{self.max_cycles} CG iterations "
                    f"(deflated {n_def} → recycle dim {n_rec})"
                )
        return x, r


# ======================================================================
# Stand-alone test
# ======================================================================
if __name__ == "__main__":
    import time, math

    dtype  = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ==================================================================
    # 2-D test  --  Laplacian(phi) = f, constant coefficients
    # ==================================================================
    print("\n=== 2-D Poisson multigrid test ===")
    N = 16
    L = 2 * math.pi
    h = L / N
    nx = ny = N + 2
    x = torch.linspace(-h / 2, L + h / 2, nx, dtype=dtype, device=device)
    y = torch.linspace(-h / 2, L + h / 2, ny, dtype=dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    phi_exact = torch.sin(X) * torch.sin(Y)
    f_inner = -2.0 * phi_exact[1:-1, 1:-1]

    c  = torch.ones(nx, ny, dtype=dtype, device=device)
    ch = 0.5 * (c[1:, 1:-1] + c[:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:] + c[1:-1, :-1])
    p0 = torch.zeros(nx, ny, dtype=dtype, device=device)

    ps = PoissonSolver(dtype, device, h, tol=1e-8, max_vcycles=100,
                       nsmoothing=10, verbose=True)
    t0 = time.time()
    p, r = ps.solve_multigrid(f_inner, p0, ch=ch, cv=cv)
    elapsed = time.time() - t0

    err = torch.abs(p - phi_exact)
    linf = err[1:-1, 1:-1].max().item()
    print(f"  Solve: {elapsed:.3f}s, Linf interior error: {linf:.3e}")

    # ==================================================================
    # 3-D test  --  Laplacian(phi) = f, constant coefficients
    # ==================================================================
    print("\n=== 3-D Poisson multigrid test ===")
    N3 = 16
    h3 = L / N3
    nx3 = ny3 = nz3 = N3 + 2
    x3 = torch.linspace(-h3 / 2, L + h3 / 2, nx3, dtype=dtype, device=device)
    y3 = torch.linspace(-h3 / 2, L + h3 / 2, ny3, dtype=dtype, device=device)
    z3 = torch.linspace(-h3 / 2, L + h3 / 2, nz3, dtype=dtype, device=device)
    X3, Y3, Z3 = torch.meshgrid(x3, y3, z3, indexing="ij")

    phi3 = torch.sin(X3) * torch.sin(Y3) * torch.sin(Z3)
    f3_inner = -3.0 * phi3[1:-1, 1:-1, 1:-1]

    c3  = torch.ones(nx3, ny3, nz3, dtype=dtype, device=device)
    ch3 = 0.5 * (c3[1:, 1:-1, 1:-1] + c3[:-1, 1:-1, 1:-1])
    cv3 = 0.5 * (c3[1:-1, 1:, 1:-1] + c3[1:-1, :-1, 1:-1])
    cw3 = 0.5 * (c3[1:-1, 1:-1, 1:] + c3[1:-1, 1:-1, :-1])
    p3_0 = torch.zeros(nx3, ny3, nz3, dtype=dtype, device=device)

    ps3 = PoissonSolver(dtype, device, h3, tol=1e-8, max_vcycles=100,
                        nsmoothing=10, verbose=True)
    t0 = time.time()
    p3, r3 = ps3.solve_multigrid(f3_inner, p3_0, ch=ch3, cv=cv3, cw=cw3)
    elapsed = time.time() - t0

    err3 = torch.abs(p3 - phi3)
    linf3 = err3[1:-1, 1:-1, 1:-1].max().item()
    print(f"  Solve: {elapsed:.3f}s, Linf interior error: {linf3:.3e}")

    # ==================================================================
    # MGCG tests  (same problems, CG + V-cycle preconditioner)
    # ==================================================================
    print("\n=== 2-D Poisson MGCG test (constant coeff) ===")
    ps_cg = PoissonSolver(dtype, device, h, tol=1e-8,
                          max_cycles=30, max_vcycles=1, nsmoothing=10,
                          precond_vcycles=1, verbose=True)
    t0 = time.time()
    p_cg, r_cg = ps_cg.solve_mgcg(f_inner, p0.clone(), ch=ch, cv=cv)
    elapsed_cg = time.time() - t0
    err_cg = torch.abs(p_cg - phi_exact)
    linf_cg = err_cg[1:-1, 1:-1].max().item()
    print(f"  Solve: {elapsed_cg:.3f}s, Linf interior error: {linf_cg:.3e}")

    print("\n=== 3-D Poisson MGCG test (constant coeff) ===")
    ps3_cg = PoissonSolver(dtype, device, h3, tol=1e-8,
                           max_cycles=30, max_vcycles=1, nsmoothing=10,
                           precond_vcycles=1, verbose=True)
    t0 = time.time()
    p3_cg, r3_cg = ps3_cg.solve_mgcg(f3_inner, p3_0.clone(),
                                       ch=ch3, cv=cv3, cw=cw3)
    elapsed_cg3 = time.time() - t0
    err3_cg = torch.abs(p3_cg - phi3)
    linf3_cg = err3_cg[1:-1, 1:-1, 1:-1].max().item()
    print(f"  Solve: {elapsed_cg3:.3f}s, Linf interior error: {linf3_cg:.3e}")

    # ==================================================================
    # Variable-coefficient test (BDIM-like: c has a sharp jump)
    # ==================================================================
    print("\n=== 2-D Poisson: variable coefficients (jump) ===")
    N_vc = 64
    h_vc = L / N_vc
    nx_vc = ny_vc = N_vc + 2
    x_vc = torch.linspace(-h_vc/2, L + h_vc/2, nx_vc, dtype=dtype, device=device)
    y_vc = torch.linspace(-h_vc/2, L + h_vc/2, ny_vc, dtype=dtype, device=device)
    X_vc, Y_vc = torch.meshgrid(x_vc, y_vc, indexing="ij")

    # Coefficient: c=1 outside a circle, c=1000 inside  (BDIM-like jump)
    radius = L / 4
    centre = L / 2
    dist = torch.sqrt((X_vc - centre)**2 + (Y_vc - centre)**2)
    c_vc = torch.where(dist < radius,
                       1000.0 * torch.ones_like(X_vc),
                       torch.ones_like(X_vc))

    phi_vc = torch.sin(X_vc) * torch.sin(Y_vc)
    # f = div(c * grad(phi)) = c * (-2 sin(x)sin(y))  (for constant c in each region)
    # but c has a jump so f isn't strictly this — use it as a synthetic RHS
    f_vc = -2.0 * c_vc[1:-1, 1:-1] * phi_vc[1:-1, 1:-1]

    ch_vc = 0.5 * (c_vc[1:, 1:-1] + c_vc[:-1, 1:-1])
    cv_vc = 0.5 * (c_vc[1:-1, 1:] + c_vc[1:-1, :-1])
    p0_vc = torch.zeros(nx_vc, ny_vc, dtype=dtype, device=device)

    print("  --- Standalone multigrid ---")
    ps_vc_mg = PoissonSolver(dtype, device, h_vc, tol=1e-6,
                             max_vcycles=50, nsmoothing=10, w=0.8,
                             verbose=True)
    t0 = time.time()
    p_mg, _ = ps_vc_mg.solve_multigrid(f_vc, p0_vc.clone(),
                                        ch=ch_vc, cv=cv_vc)
    t_mg = time.time() - t0
    print(f"  Time: {t_mg:.3f}s")

    print("  --- MGCG ---")
    ps_vc_cg = PoissonSolver(dtype, device, h_vc, tol=1e-6,
                             max_cycles=50, max_vcycles=1, nsmoothing=10,
                             w=0.8, precond_vcycles=1, verbose=True)
    t0 = time.time()
    p_mgcg, _ = ps_vc_cg.solve_mgcg(f_vc, p0_vc.clone(),
                                      ch=ch_vc, cv=cv_vc)
    t_mgcg = time.time() - t0
    print(f"  Time: {t_mgcg:.3f}s")

    diff = torch.abs(p_mg - p_mgcg)
    print(f"  MG vs MGCG max diff: {diff[1:-1, 1:-1].max().item():.3e}")

    print("\nDone.")
