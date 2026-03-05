"""Dimension-agnostic multigrid Poisson solver with variable coefficients.

Solves   div(c * grad(p)) = f   using geometric multigrid V-cycles
with Jacobi smoothing.  Works in 2-D and 3-D with a single code path.

The variable-coefficient discrete operator on a uniform grid (spacing *h*)
for the *d*-th direction is:

    [c_{d+} p_{i+1} - (c_{d+} + c_{d-}) p_i + c_{d-} p_{i-1}] / h^2

where c_{d+}, c_{d-} are face-averaged coefficients along dimension *d*.

Usage (backward-compatible with old 2-D interface)::

    ps = PoissonSolver(dtype, device, h, tol=1e-2)
    p, r = ps.solve_multigrid(f, p0, c, ch=ch, cv=cv)        # 2-D
    p, r = ps.solve_multigrid(f, p0, c, ch=ch, cv=cv, cw=cw)  # 3-D
"""

import torch


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
    """Variable-coefficient multigrid Poisson solver (2-D / 3-D)."""

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
    ):
        self.dtype       = dtype
        self.h2          = h * h
        self.device      = device
        self.tol         = tol
        self.max_cycles  = max_cycles
        self.max_vcycles = max_vcycles
        self.nsmoothing  = nsmoothing
        self.verbose     = verbose
        self.jcap_tol    = 1e-12
        self.n_switch    = 2 ** 16
        self.w           = w   # Jacobi relaxation weight

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def l2_norm(r):
        return torch.sqrt((r**2).sum())

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
    def Jacobi(self, f, p, cfaces, h2):
        self.BC(p)
        J    = self.compute_J(cfaces)
        active = torch.abs(J) >= self.jcap_tol          # fluid mask
        Jinv = torch.where(active, 1 / J, torch.zeros_like(J))
        inner = _inner(p.ndim)

        for _ in range(self.nsmoothing):
            s = self.compute_sum(cfaces, p)
            p[inner] = self.w * (-f * h2 + s) * Jinv + (1 - self.w) * p[inner]
            self.BC(p)

        # residual — zero at degenerate cells (cf. WaterLily residual!)
        s  = self.compute_sum(cfaces, p)
        J  = self.compute_J(cfaces)
        Au = (s - J * p[inner]) / h2
        r  = torch.where(active, f - Au, torch.zeros_like(f))
        return p, r

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
    # V-cycle  (dimension-agnostic, recursive)
    # ------------------------------------------------------------------
    def _vcycle(self, f, p, face_arrs, h2):
        """Internal V-cycle operating on full face arrays."""
        ndim  = f.ndim
        shape = f.shape

        # extract (cp, cm) for the smoother
        cfaces = self._extract_cfaces(face_arrs, ndim)

        # pre-smooth
        p, r = self.Jacobi(f, p, cfaces, h2)

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
                if not isinstance(h2, float):
                    h2 = h2.cpu()

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
                cf_c = 0.5 * cf_c                  # single 0.5 factor
                face_arrs_coarse.append(cf_c)

            # ---- restriction of residual (full-weighting) ------------
            r_coarse = r.clone()
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
                1,
            )

            # ---- prolongation (piecewise constant injection) ---------
            inner_c = _inner(ndim)
            err = torch.zeros_like(r)
            ec = err_coarse[inner_c]
            for corner in range(2 ** ndim):
                slc = []
                for d in range(ndim):
                    if (corner >> d) & 1:
                        slc.append(slice(1, None, 2))
                    else:
                        slc.append(slice(None, None, 2))
                target = err[tuple(slc)]
                min_shape = tuple(min(t, s) for t, s in
                                  zip(target.shape, ec.shape))
                trim = tuple(slice(0, m) for m in min_shape)
                err[tuple(slc)][trim] = ec[trim]

            # correction
            p[_inner(ndim)] += err

            if on_gpu:
                f         = f.cuda()
                p         = p.cuda()
                face_arrs = [cf.cuda() for cf in face_arrs]
                if not isinstance(h2, float):
                    h2 = h2.cuda()

            # post-smooth (re-extract cfaces after possible GPU round-trip)
            cfaces = self._extract_cfaces(face_arrs, ndim)
            p, r = self.Jacobi(f, p, cfaces, h2)

        return p, r

    # ------------------------------------------------------------------
    # Public V-cycle  (wrapper that builds face_arrs from kwargs)
    # ------------------------------------------------------------------
    def vcycle(self, f, p, c, h2, **kwargs):
        """V-cycle accepting legacy ch/cv/cw kwargs."""
        ndim = f.ndim
        face_arrs, kwargs = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            face_arrs = self._default_face_arrs(c, ndim)
        return self._vcycle(f, p, face_arrs, h2)

    # ------------------------------------------------------------------
    # Top-level solve
    # ------------------------------------------------------------------
    def solve_multigrid(self, f, p0, c, **kwargs):
        """Solve with multigrid V-cycles.

        Parameters
        ----------
        f  : RHS on the interior grid  (no ghost cells)
        p0 : initial guess (with ghost cells)
        c  : cell-centred coefficient field (with ghost cells)
        ch, cv[, cw] : optional pre-computed face-averaged coefficients
        """
        p = p0.clone().detach()
        for cycle in range(self.max_vcycles):
            p, r = self.vcycle(self.h2 * f, p, c, 1, **kwargs)
            r_err = self.l2_norm(r)
            if r_err < self.tol:
                break
        p -= p.mean()
        if self.verbose:
            print(
                f"Multigrid residual = {r_err:.2e}/{self.tol:.2e} "
                f"with {cycle + 1}/{self.max_vcycles} cycles"
            )
        return p, r


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
    p, r = ps.solve_multigrid(f_inner, p0, c, ch=ch, cv=cv)
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
    p3, r3 = ps3.solve_multigrid(f3_inner, p3_0, c3, ch=ch3, cv=cv3, cw=cw3)
    elapsed = time.time() - t0

    err3 = torch.abs(p3 - phi3)
    linf3 = err3[1:-1, 1:-1, 1:-1].max().item()
    print(f"  Solve: {elapsed:.3f}s, Linf interior error: {linf3:.3e}")

    print("\nDone.")
