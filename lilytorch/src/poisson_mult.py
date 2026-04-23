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


# =====================================================================
# Compilable smoother kernels  (module-level, for torch.compile)
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
    s  = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
    Au = (s - J * p[1:-1, 1:-1, 1:-1])
    r  = torch.where(active, f - Au, torch.zeros_like(f))
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
    s  = _sum2d(cp0, cm0, cp1, cm1, p)
    Au = (s - J * p[1:-1, 1:-1])
    r  = torch.where(active, f - Au, torch.zeros_like(f))
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
    s  = _sum3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
    Au = (s - J * p[1:-1, 1:-1, 1:-1])
    r  = torch.where(active, f - Au, torch.zeros_like(f))
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
    s  = _sum2d(cp0, cm0, cp1, cm1, p)
    Au = (s - J * p[1:-1, 1:-1])
    r  = torch.where(active, f - Au, torch.zeros_like(f))
    return p, r


# =====================================================================
# Compilable V-cycle helpers (3-D, for torch.compile)
# =====================================================================

def _restrict_face_3d(ch, cv, cw):
    """Restrict face arrays from fine to coarse (3-D, WaterLily convention)."""
    # ch: face along dim 0 — stride-2 in dim 0, SUM in dims 1,2
    ch_c = ch[::2, :, :]
    ch_c = ch_c[:, :-1:2, :] + ch_c[:, 1::2, :]
    ch_c = ch_c[:, :, :-1:2] + ch_c[:, :, 1::2]
    ch_c.mul_(0.5)
    # cv: face along dim 1 — SUM in dim 0, stride-2 in dim 1, SUM in dim 2
    cv_c = cv[:-1:2, :, :] + cv[1::2, :, :]
    cv_c = cv_c[:, ::2, :]
    cv_c = cv_c[:, :, :-1:2] + cv_c[:, :, 1::2]
    cv_c.mul_(0.5)
    # cw: face along dim 2 — SUM in dims 0,1, stride-2 in dim 2
    cw_c = cw[:-1:2, :, :] + cw[1::2, :, :]
    cw_c = cw_c[:, :-1:2, :] + cw_c[:, 1::2, :]
    cw_c = cw_c[:, :, ::2]
    cw_c.mul_(0.5)
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


# ── Full 3-D V-cycle with Jacobi (compilable, recursive) ────────────
def _vcycle_jac_3d(f, p, ch, cv, cw, w, jcap_tol, nsmoothing):
    """Complete 3-D V-cycle with Jacobi smoother.

    torch.compile unrolls the recursion at trace time, producing a single
    flat computation graph that fuses ALL operations across all multigrid
    levels — restriction, smoothing, prolongation — into a handful of
    CUDA kernels.
    """
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

        # recursive call (unrolled by torch.compile tracer)
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


# ── Full 3-D V-cycle with RBGS (compilable, recursive) ──────────────
def _vcycle_rbgs_3d(f, p, ch, cv, cw, jcap_tol, nsmoothing):
    """Complete 3-D V-cycle with Red-Black Gauss-Seidel smoother."""
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
        red, black = _rb_masks_3d(f.shape[0], f.shape[1], f.shape[2], p.device)
        p, r = _rbgs_3d(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                         jcap_tol, nsmoothing, red, black)

    return p, r


# =====================================================================
# Compilable V-cycle helpers (2-D, for torch.compile)
# =====================================================================

def _restrict_face_2d(ch, cv):
    """Restrict face arrays from fine to coarse (2-D, WaterLily convention)."""
    # ch: face along dim 0 — stride-2 in dim 0, SUM in dim 1
    ch_c = ch[::2, :]
    ch_c = ch_c[:, :-1:2] + ch_c[:, 1::2]
    ch_c.mul_(0.5)
    # cv: face along dim 1 — SUM in dim 0, stride-2 in dim 1
    cv_c = cv[:-1:2, :] + cv[1::2, :]
    cv_c = cv_c[:, ::2]
    cv_c.mul_(0.5)
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


# ── Full 2-D V-cycle with Jacobi (compilable, recursive) ────────────
def _vcycle_jac_2d(f, p, ch, cv, w, jcap_tol, nsmoothing):
    """Complete 2-D V-cycle with Jacobi smoother."""
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


# ── Full 2-D V-cycle with RBGS (compilable, recursive) ──────────────
def _vcycle_rbgs_2d(f, p, ch, cv, jcap_tol, nsmoothing):
    """Complete 2-D V-cycle with Red-Black Gauss-Seidel smoother."""
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
        red, black = _rb_masks_2d(f.shape[0], f.shape[1], p.device)
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
        max_vcycles=3,
        nsmoothing=5,
        w=1,
        verbose=True,
        precond_vcycles=1,
        smoother="jacobi",
        compile_smoother=False,
    ):
        self.dtype       = dtype
        self.h2          = h * h
        self.device      = device
        self.tol         = tol
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
        self.compile_smoother = compile_smoother
        self._compiled_fn = {}    # lazily populated {ndim: compiled_fn}
        self._rb_mask_cache = {}  # {(shape, device): (red, black)}

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

        for _ in range(self.nsmoothing):
            s = self.compute_sum(cfaces, p)
            p[inner] = self.w * (-f + s) * Jinv + (1 - self.w) * p[inner]
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

        for _ in range(self.nsmoothing):
            # --- red sweep ---
            s = self.compute_sum(cfaces, p)
            p_new = (-f + s) * Jinv
            p[inner] = torch.where(red, p_new, p[inner])
            self.BC(p)

            # --- black sweep ---
            s = self.compute_sum(cfaces, p)
            p_new = (-f + s) * Jinv
            p[inner] = torch.where(black, p_new, p[inner])
            self.BC(p)

        # residual
        s  = self.compute_sum(cfaces, p)
        Au = (s - J * p[inner])
        r  = torch.where(active, f - Au, torch.zeros_like(f))
        return p, r

    # ------------------------------------------------------------------
    # Smoother dispatch
    # ------------------------------------------------------------------
    def _get_compiled_fn(self, ndim):
        """Lazily compile the smoother for the given dimensionality."""
        if ndim not in self._compiled_fn:
            if self.smoother == "rbgs":
                raw = _rbgs_3d if ndim == 3 else _rbgs_2d
            else:
                raw = _jacobi_3d if ndim == 3 else _jacobi_2d
            self._compiled_fn[ndim] = torch.compile(raw)
        return self._compiled_fn[ndim]

    def _smooth_compiled(self, f, p, cfaces):
        """Compiled smoother path — flattens cfaces for the compiled kernel."""
        ndim = p.ndim
        fn   = self._get_compiled_fn(ndim)

        # Flatten cfaces list into positional args
        flat = []
        for cp, cm in cfaces:
            flat.extend([cp, cm])

        if self.smoother == "rbgs":
            interior_shape = p[_inner(ndim)].shape
            red, black = self._build_rb_masks(interior_shape)
            return fn(f, p, *flat, self.jcap_tol,
                      self.nsmoothing, red, black)
        else:
            return fn(f, p, *flat, self.w, self.jcap_tol,
                      self.nsmoothing)

    def smooth(self, f, p, cfaces):
        """Dispatch to the configured smoother."""
        if self.compile_smoother:
            return self._smooth_compiled(f, p, cfaces)
        if self.smoother == "rbgs":
            return self.RBGS(f, p, cfaces)
        return self.Jacobi(f, p, cfaces)

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
    # Compiled full V-cycle dispatch
    # ------------------------------------------------------------------
    def _get_compiled_vcycle(self, ndim):
        """Lazily compile the full V-cycle for given dimensionality."""
        key = f"vcycle_{ndim}d_{self.smoother}"
        if key not in self._compiled_fn:
            if ndim == 3:
                raw = _vcycle_rbgs_3d if self.smoother == "rbgs" else _vcycle_jac_3d
            else:
                raw = _vcycle_rbgs_2d if self.smoother == "rbgs" else _vcycle_jac_2d
            self._compiled_fn[key] = torch.compile(raw)
        return self._compiled_fn[key]

    def _run_compiled_vcycle(self, f, p, face_arrs):
        """Run the compiled V-cycle (2-D or 3-D)."""
        ndim = f.ndim
        fn = self._get_compiled_vcycle(ndim)
        if ndim == 3:
            ch, cv, cw = face_arrs
            if self.smoother == "rbgs":
                return fn(f, p, ch, cv, cw,
                          self.jcap_tol, self.nsmoothing)
            else:
                return fn(f, p, ch, cv, cw,
                          self.w, self.jcap_tol, self.nsmoothing)
        else:
            ch, cv = face_arrs
            if self.smoother == "rbgs":
                return fn(f, p, ch, cv,
                          self.jcap_tol, self.nsmoothing)
            else:
                return fn(f, p, ch, cv,
                          self.w, self.jcap_tol, self.nsmoothing)

    # ------------------------------------------------------------------
    # V-cycle  (dimension-agnostic, recursive)
    # ------------------------------------------------------------------
    def _vcycle(self, f, p, face_arrs):
        """Internal V-cycle operating on full face arrays."""
        # Use compiled full V-cycle when enabled
        if self.compile_smoother:
            return self._run_compiled_vcycle(f, p, face_arrs)

        ndim  = f.ndim
        shape = f.shape

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
            err_coarse, _ = self._vcycle(
                r_coarse,
                torch.zeros(coarse_shape, device=p.device, dtype=p.dtype),
                face_arrs_coarse,
            )

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
    def solve_multigrid(self, f, p0, **kwargs):
        """Solve with multigrid V-cycles.

        Parameters
        ----------
        f  : RHS on the interior grid  (no ghost cells)
        p0 : initial guess (with ghost cells)
        ch, cv[, cw] : pre-computed face-averaged coefficients
        """
        p = p0.clone().detach()
        for cycle in range(self.max_vcycles):
            p, r = self.vcycle(self.h2 * f, p, **kwargs)
            # L-inf norm: deterministic on both CPU and CUDA (no summation).
            r_err = self._convergence_norm(r)
            if r_err < self.tol:
                break
        # float64 mean subtraction: GPU parallel-reduction of float32 gives
        # a different value than CPU sequential sum.
        p -= p.to(torch.float64).mean().to(p.dtype)
        if self.verbose:
            print(
                f"Multigrid residual = {self.l2_norm(r):.2e}/{self.tol:.2e} "
                f"with {cycle + 1}/{self.max_vcycles} cycles"
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
        s = self.compute_sum(cfaces, p)
        active = torch.abs(J) >= self.jcap_tol
        result = J * p[inner] - s
        return torch.where(active, result, torch.zeros_like(result))

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
        ----------
        f  : RHS on the interior grid  (no ghost cells)
        p0 : initial guess (with ghost cells)
        ch, cv[, cw] : pre-computed face-averaged coefficients
        """
        ndim = f.ndim
        face_arrs, extra = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_mgcg: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required."
            )
        cfaces = self._extract_cfaces(face_arrs, ndim)
        inner  = _inner(ndim)

        # ------ SPD system:  B(x) = b  where B = Jp - S,  b = -(h²·f) ------
        b = -(self.h2 * f)

        x = p0.clone().detach()
        self.BC(x)

        # Initial residual: r = b - B(x)
        r = b - self._apply_op_spd(x, cfaces)
        r_norm = self._convergence_norm(r)

        if r_norm < self.tol:
            x -= x.to(torch.float64).mean().to(x.dtype)
            if self.verbose:
                print(f"MGCG converged at initial guess: "
                      f"residual = {r_norm:.2e}")
            return x, r

        # Preconditioner: approximately solve B(z) = r via V-cycle(s).
        # The V-cycle solves  (S - Jp) = f_arg,  i.e.  -B(p) = f_arg,
        # so we pass f_arg = -r  →  -B(z) ≈ -r  →  B(z) ≈ r.
        z = torch.zeros_like(x)
        for _ in range(self.precond_vcycles):
            z, _ = self._vcycle(-r, z, face_arrs)

        d = z.clone()                              # search direction
        self.BC(d)
        rz = (r * z[inner]).to(torch.float64).sum().to(r.dtype)  # r · M⁻¹r

        r_norm_final = r_norm
        for k in range(self.max_cycles):
            # --- matrix-vector product ---
            q = self._apply_op_spd(d, cfaces)      # q = B·d

            dq = (d[inner] * q).to(torch.float64).sum().to(r.dtype)  # d · B·d
            if dq.abs() < 1e-30:                   # degenerate
                break

            alpha = rz / dq                        # step length

            x[inner] = x[inner] + alpha * d[inner]
            self.BC(x)
            r = r - alpha * q

            r_norm_final = self._convergence_norm(r)
            if r_norm_final < self.tol:
                break

            # --- preconditioner (reuse buffer) ---
            z.zero_()
            for _ in range(self.precond_vcycles):
                z, _ = self._vcycle(-r, z, face_arrs)

            rz_new = (r * z[inner]).to(torch.float64).sum().to(r.dtype)
            if rz.abs() < 1e-30:
                break

            beta = rz_new / rz
            d[inner] = z[inner] + beta * d[inner]
            self.BC(d)
            rz = rz_new

        x -= x.to(torch.float64).mean().to(x.dtype)
        if self.verbose:
            cg_iters = min(k + 1, self.max_cycles)
            print(
                f"MGCG residual = {r_norm_final:.2e}/{self.tol:.2e} "
                f"with {cg_iters}/{self.max_cycles} CG iterations "
                f"({self.precond_vcycles} V-cycle{'s' if self.precond_vcycles > 1 else ''}/iter)"
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
