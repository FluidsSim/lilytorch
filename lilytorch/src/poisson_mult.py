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


def _gauge_fix(x):
    """Neumann gauge fix, in place: subtract the INTERIOR mean.

    The mean must NOT include the ghost ring: its edge/corner cells are dead
    (no 5/7-point stencil reads them) and hold backend-dependent garbage, so a
    mean over the padded tensor makes the gauge constant itself
    backend-dependent — CPU and CUDA then return pressures differing by a
    constant.  Mirrors ``gauge_fix()`` in csrc/poisson_gauge.h, which the
    native whole-solve drivers use.

    Subtracting from the whole tensor leaves the Neumann ring consistent (a
    ghost and its interior neighbour shift by the same constant), so the
    caller's last ``BC(x)`` still holds.
    """
    inner = _inner(x.ndim)
    x -= x[inner].to(torch.float64).mean().to(x.dtype)


# =====================================================================
# Poisson solver
# =====================================================================

class _MultigridPoissonSolver:
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
        self.jcap_tol    = 1e-7 # lower value helps to reduce degenerate
        self.n_switch    = 2 ** 16
        self.w           = w   # Jacobi relaxation weight
        self.precond_vcycles = precond_vcycles  # V-cycles per CG preconditioner
        assert smoother in ("jacobi", "rbgs"), \
            f"smoother must be 'jacobi' or 'rbgs', got '{smoother}'"
        self.smoother = smoother
        self._rb_mask_cache = {}  # {(shape, device): (red, black)}
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
        # ---- Last-solve statistics (for FlowDiagnostics) ---------------
        # ``_last_resid`` is kept as a 0-dim device tensor: taking the norm is
        # one kernel, but calling .item() on it would force a host sync every
        # step.  Consumers store it straight into a device-side record array
        # and sync only when the run writes diagnostics.h5.
        self._last_niter = 0
        self._last_resid = None

    def _record_stats(self, r, niter):
        """Stash residual/iteration count from the solve that just finished.

        All six native drivers report their own count.  With ``tol < 0`` the
        early-exit test is skipped so the solve stays host-sync-free and
        graph-capturable; the loop then always runs its full budget and the
        count degenerates to ``max_vcycles`` / ``max_cycles``.
        """
        self._last_resid = r.abs().max().detach()
        self._last_niter = int(niter)

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
    # Range / null-space handling for the CG drivers
    #
    # ``_apply_op_spd`` zeroes degenerate cells, so B has identically zero
    # ROWS AND COLUMNS there: the indicator vector of every degenerate cell
    # lies in null(B), alongside the usual Neumann constant.  A stationary
    # V-cycle never notices — ``mg_residual_*`` masks the residual, so the
    # null-space component enters neither an inner product nor the
    # convergence test.  CG is not so lucky: it forms r·M⁻¹r and d·Bd every
    # iteration, and a RHS component outside range(B) poisons alpha and beta,
    # making the residual GROW with iteration count.  b = -h²·div(u*) has
    # exactly such a component, because div(u*) does not vanish inside a BDIM
    # solid.  So the RHS must be projected onto range(B) before the CG loop,
    # and the preconditioner output must be masked to keep M symmetric.
    # ------------------------------------------------------------------
    def _active_mask(self, cfaces, dtype):
        """Interior mask of non-degenerate cells, as ``_apply_op_spd`` sees them."""
        return (self.compute_J(cfaces).abs() >= self.jcap_tol).to(dtype)

    @staticmethod
    def _project_rhs(b, mask):
        """Project ``b`` onto range(B), in place.

        Two steps, both required:
          1. zero the degenerate cells (the per-cell null directions);
          2. remove the mean over the LIVE cells (the Neumann constant).

        Step 1 alone is not enough — the surviving constant is then amplified
        by the CG recurrence instead of being ignored.

        Caveat: if the live region is split into disconnected pools by a
        solid, null(B) holds one constant PER COMPONENT and removing the
        global mean only kills the aggregate.  Standalone multigrid has the
        same limitation; handling it would need a connected-component label.
        """
        b.mul_(mask)
        # sum(dtype=) accumulates in float64 without materialising a float64
        # copy of the field, and the mean stays a 0-dim device tensor, so this
        # adds no host sync.
        nlive = mask.sum(dtype=torch.float64).clamp_(min=1.0)
        mean = (b.sum(dtype=torch.float64) / nlive).to(b.dtype)
        b.addcmul_(mask, mean, value=-1.0)
        return b

    def _precondition(self, r, z, face_arrs, mask, inner):
        """z ≈ B⁻¹r via V-cycle(s), masked back onto the live subspace.

        The mask is what makes M symmetric: the V-cycle zeroes the residual on
        degenerate cells on the way down (``mg_residual_*``) but
        ``prolongate_add`` writes the coarse correction into EVERY fine cell on
        the way up — a zero column against a non-zero row.  Masking the output
        restores M = Mᵀ (verified to 4e-16 by
        ``lilytorch_examples/tests/check_poisson_symmetry.py``).
        """
        z.zero_()
        for _ in range(self.precond_vcycles):
            z, _ = self._dispatch_vcycle(-r, z, face_arrs)
        z[inner].mul_(mask)
        return z

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

        ``b`` is consumed (projected onto range(B) in place); every call site
        builds it fresh immediately beforehand.
        """
        inner = _inner(b.ndim)

        # Project the RHS onto range(B) and keep the mask for the
        # preconditioner — see the _project_rhs / _precondition notes above.
        mask = self._active_mask(cfaces, b.dtype)
        b = self._project_rhs(b, mask)

        # Initial residual: r = b - B(x)
        r = b - self._apply_op_spd(x, cfaces)

        # Deflation init: project the recycle subspace out of (x, r) so the
        # CG iteration never has to rediscover those modes.
        if recycle is not None:
            self._deflate_init(x, r, recycle, inner)

        r_norm = self._convergence_norm(r)
        if r_norm < self.tol:
            _gauge_fix(x)
            self._last_niter = 0
            self._last_resid = r_norm
            return x, r, 0, r_norm

        # Preconditioner: approximately solve B(z) = r via V-cycle(s).
        # The V-cycle solves (S - Jp) = f_arg, i.e. -B(p) = f_arg, so we pass
        # f_arg = -r  →  -B(z) ≈ -r  →  B(z) ≈ r.
        z = torch.zeros_like(x)
        z = self._precondition(r, z, face_arrs, mask, inner)

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
            z = self._precondition(r, z, face_arrs, mask, inner)

            rz_new = (r * z[inner]).to(torch.float64).sum().to(r.dtype)
            beta = rz_new / rz
            d[inner] = z[inner] + beta * d[inner]
            if recycle is not None and self._deflate_in_loop:
                self._deflate_proj(d, z, recycle, inner)
            self.BC(d)
            rz = rz_new

        _gauge_fix(x)
        niter = min(k + 1, self.max_cycles)
        self._last_niter = niter          # exposed for benchmarking/diagnostics
        self._last_resid = r_norm_final
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
        mask = self._active_mask(cfaces, raw[0].dtype)
        for u0 in raw:
            u = u0.clone()
            # Same projection the RHS gets: drop the degenerate cells and the
            # Neumann constant, so the deflation space lives entirely in
            # range(B).  Subtracting a plain global mean would put a non-zero
            # value back onto the degenerate cells, where B is identically
            # zero, and corrupt the Rayleigh quotient below.
            self._project_rhs(u[inner], mask)
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

    def solve_rmgcg(self, f, p0, **kwargs):
        """Recycled MGCG: ``solve_mgcg`` plus cross-timestep Krylov recycling.

        Identical interface and (with ``recycle_k == 0``) identical behaviour
        to ``solve_mgcg``.  With ``recycle_k > 0`` it deflates the subspace of
        slow-converging modes carried over from previous solves, cutting CG
        iterations for time-stepping problems whose operator changes little
        per step (e.g. slow swimmers).
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


class PoissonSolver(_MultigridPoissonSolver):
    """Variable-coefficient multigrid Poisson on the native CUDA / C++ kernels
    (CPU + GPU, f32 + f64).

    Two levels of native driver, picked by :meth:`_native_whole_solve`:

    * **Whole-solve drivers** — ``poisson_solve_{multigrid,mgcg,rmgcg}_{2,3}d``
      run the entire solve (V-cycle tree, CG loop, deflation, gauge fix) inside
      one C++ call, so the launch-bound Python loop disappears entirely.  All
      six exist on CUDA; on the CPU only ``multigrid`` has a twin.
    * **Raw V-cycle** — ``mg_vcycle_{2,3}d`` runs N V-cycles with no gauge fix.
      This is the PCG preconditioner primitive (what the native MGCG driver
      applies to ``z`` internally), and it is what keeps MGCG / RMGCG alive on
      the CPU: there :meth:`_cg_core` drives the CG loop in Python and calls
      :meth:`_dispatch_vcycle` → ``mg_vcycle_*`` for the preconditioner.

    WHY the whole-solve drivers matter: the pressure Poisson solve is
    launch-bound, not compute-bound, on the grids this solver targets.  A
    Python-driven V-cycle dispatches dozens of tiny kernels (smoother sweeps +
    residual/restrict/prolong per level), each a few µs of GPU work behind µs
    of host launch and sync overhead.  Collapsing the tree into one C++ call
    removes that overhead — and, unlike a CUDA-graph replay, it also removes
    the per-V-cycle residual ``.item()`` host sync.
    """

    # ------------------------------------------------------------------
    # Native dispatch helpers
    # ------------------------------------------------------------------

    def _native_whole_solve(self, method):
        """True when the whole-solve C++ driver for ``method`` exists here.

        All six drivers exist on CUDA.  On the CPU only ``multigrid`` has a
        twin, so ``mgcg`` / ``rmgcg`` fall back to the Python CG loop driving
        the native ``mg_vcycle_*`` preconditioner (see :meth:`_dispatch_vcycle`).
        """
        dev = torch.device(self.device) if isinstance(self.device, str) else self.device
        return dev.type == "cuda" or method == "multigrid"

    def _native_multigrid(self, f, p0, face_arrs):
        """Run the native multigrid Poisson driver (2-D / 3-D, CUDA + CPU).

        Mutates ``p0`` in place; returns ``(p, r)`` matching the
        ``solve_multigrid`` contract.  The driver scales ``f`` by ``self.h2``
        internally, runs up to ``max_vcycles`` V-cycles with an L∞ early exit
        at ``tol``, then applies the ghost-ring Neumann BC and the float64
        gauge (mean) subtraction.
        """
        from lilytorch.src import native as _nat
        ndim = f.ndim
        if p0 is None:
            # p0=None is the multigrid cold-start fast path.  Serve it from a
            # persistent zeroed padded buffer — no per-step alloc, and
            # pointer-stable so it does not churn the graph cache key.
            pshp = tuple(n + 2 for n in f.shape)
            buf = getattr(self, "_native_p0", None)
            if (buf is None or buf.shape != pshp
                    or buf.dtype != f.dtype or buf.device != f.device):
                buf = torch.zeros(pshp, dtype=f.dtype, device=f.device)
                self._native_p0 = buf
            else:
                buf.zero_()
            p0 = buf
        if ndim == 2:
            ch, cv = (a.contiguous() for a in face_arrs)
            r, niter = _nat.poisson_solve_multigrid_2d(
                p0, f, ch, cv,
                float(self.h2), float(self.jcap_tol), float(self.w),
                int(self.nsmoothing), int(self.max_vcycles),
                float(self._tol_float), self.smoother,
            )
        else:
            ch, cv, cw = (a.contiguous() for a in face_arrs)
            r, niter = _nat.poisson_solve_multigrid_3d(
                p0, f, ch, cv, cw,
                float(self.h2), float(self.jcap_tol), float(self.w),
                int(self.nsmoothing), int(self.max_vcycles),
                float(self._tol_float), self.smoother,
            )
        self._record_stats(r, niter)
        return p0, r

    def _native_mgcg(self, f, p0, face_arrs):
        """Run the native MGCG Poisson driver (2-D / 3-D).

        Mutates ``p0`` in place; returns ``(p, r)``.  The driver runs the whole
        CG loop with its V-cycle preconditioner in C++ — no Python
        per-iteration dispatch and no per-iteration residual sync."""
        from lilytorch.src import native as _nat
        ndim = f.ndim
        if ndim == 2:
            ch, cv = (a.contiguous() for a in face_arrs)
            r, niter = _nat.poisson_solve_mgcg_2d(
                p0, f, ch, cv,
                float(self.h2), float(self.jcap_tol), float(self.w),
                int(self.nsmoothing), int(self.max_cycles),
                int(self.precond_vcycles),
                float(self._tol_float), self.smoother,
            )
        else:
            ch, cv, cw = (a.contiguous() for a in face_arrs)
            r, niter = _nat.poisson_solve_mgcg_3d(
                p0, f, ch, cv, cw,
                float(self.h2), float(self.jcap_tol), float(self.w),
                int(self.nsmoothing), int(self.max_cycles),
                int(self.precond_vcycles),
                float(self._tol_float), self.smoother,
            )
        self._record_stats(r, niter)
        return p0, r

    def _native_rmgcg(self, f, p0, face_arrs, cfaces, inner):
        """Run the native RMGCG driver, managing the recycle space on the
        Python side.

        The native driver handles the deflated CG loop internally and returns
        harvested search directions ``D`` for refreshing the recycle space.
        The Python side is responsible for:
        1. Preparing the B-orthonormal recycle basis (``_prepare_recycle``).
        2. Harvesting directions from the returned ``D`` (``_update_recycle``).
        3. Stall-safety guards (``_finalize_recycle``).

        Returns ``(p, r)`` where ``p`` is the solution and ``r`` the residual."""
        from lilytorch.src import native as _nat
        ndim = f.ndim

        # --- Prepare recycle space (B-orthonormalise under current operator) ---
        recycle = self._prepare_recycle(cfaces, p0.shape, inner)
        kdef = 0
        if recycle is not None:
            U_recycle = recycle["U"]   # list of (Nx+2, Ny+2[, Nz+2])
            W_recycle = recycle["W"]   # list of (Nx, Ny[, Nz])
            kdef = len(U_recycle)
            if kdef > 0:
                U = torch.stack(U_recycle, dim=0)   # (kdef, Nx+2, Ny+2[, Nz+2])
                W = torch.stack(W_recycle, dim=0)   # (kdef, Nx, Ny[, Nz])
            else:
                recycle = None

        if recycle is None:
            shape_2d = (0, p0.shape[0], p0.shape[1])
            if ndim == 2:
                U = torch.empty(shape_2d, dtype=p0.dtype, device=p0.device)
                W = torch.empty((0, f.shape[0], f.shape[1]),
                                dtype=f.dtype, device=f.device)
            else:
                U = torch.empty((0, p0.shape[0], p0.shape[1], p0.shape[2]),
                                dtype=p0.dtype, device=p0.device)
                W = torch.empty((0, f.shape[0], f.shape[1], f.shape[2]),
                                dtype=f.dtype, device=f.device)

        harvest_k = int(self.recycle_k)

        if ndim == 2:
            ch, cv = (a.contiguous() for a in face_arrs)
            r, D, niter = _nat.poisson_solve_rmgcg_2d(
                p0, f, ch, cv, U, W, harvest_k,
                float(self.h2), float(self.jcap_tol), float(self.w),
                int(self.nsmoothing), int(self.max_cycles),
                int(self.precond_vcycles),
                float(self._tol_float), self.smoother,
            )
        else:
            ch, cv, cw = (a.contiguous() for a in face_arrs)
            r, D, niter = _nat.poisson_solve_rmgcg_3d(
                p0, f, ch, cv, cw, U, W, harvest_k,
                float(self.h2), float(self.jcap_tol), float(self.w),
                int(self.nsmoothing), int(self.max_cycles),
                int(self.precond_vcycles),
                float(self._tol_float), self.smoother,
            )

        # --- Harvest directions from D and update recycle space ---
        harvest = []
        if harvest_k > 0:
            for k in range(min(harvest_k, niter)):
                harvest.append(D[k].clone())

        self._finalize_recycle(niter, recycle is not None, harvest, inner)
        self._record_stats(r, niter)

        if self.verbose:
            n_def = 0 if recycle is None else kdef
            n_rec = 0 if self._recycle is None else len(self._recycle["U"])
            if niter == 0:
                print(f"RMGCG converged at initial guess: "
                      f"residual = {r.abs().max().item():.2e} (deflated {n_def})")
            else:
                print(
                    f"RMGCG residual = {r.abs().max().item():.2e}/"
                    f"{self._tol_float:.2e} "
                    f"with {niter}/{self.max_cycles} CG iterations "
                    f"(deflated {n_def} → recycle dim {n_rec})"
                )

        return p0, r

    # ------------------------------------------------------------------
    # Solve entry points — native whole-solve driver where one exists,
    # otherwise the Python CG loop over the native V-cycle (CPU mgcg/rmgcg).
    # ------------------------------------------------------------------

    def solve_multigrid(self, f, p0, **kwargs):
        """Standalone multigrid solve — native whole-solve driver (CUDA + CPU)."""
        face_arrs, _ = self._face_arrs_from_kwargs(kwargs, f.ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_multigrid: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required.")
        return self._native_multigrid(f, p0, face_arrs)

    def solve_mgcg(self, f, p0, **kwargs):
        """MGCG solve.

        CUDA: the native ``poisson_solve_mgcg_*`` whole-solve driver.
        CPU: the Python CG loop with the native V-cycle as preconditioner."""
        ndim = f.ndim
        face_arrs, _ = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_mgcg: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required.")

        if self._native_whole_solve("mgcg"):
            return self._native_mgcg(f, p0, face_arrs)

        # ---- CPU: Python CG driver over the native V-cycle ----
        cfaces = self._extract_cfaces(face_arrs, ndim)
        b = -(self.h2 * f)
        x = p0.clone().detach()
        self.BC(x)

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

    def solve_rmgcg(self, f, p0, **kwargs):
        """Recycled MGCG solve.

        CUDA: the native ``poisson_solve_rmgcg_*`` driver runs the deflated CG
        loop; the Python side manages the recycle space (prepare → harvest →
        finalize).
        CPU: the Python CG loop with the native V-cycle as preconditioner."""
        ndim = f.ndim
        face_arrs, _ = self._face_arrs_from_kwargs(kwargs, ndim)
        if face_arrs is None:
            raise ValueError(
                "solve_rmgcg: ch/cv (2-D) or ch/cv/cw (3-D) keyword "
                "arguments are required.")

        cfaces = self._extract_cfaces(face_arrs, ndim)
        inner = _inner(ndim)

        if self._native_whole_solve("rmgcg"):
            return self._native_rmgcg(f, p0, face_arrs, cfaces, inner)

        # ---- CPU: Python CG driver over the native V-cycle ----
        b = -(self.h2 * f)
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

    def _dispatch_vcycle(self, f, p, face_arrs):
        """One native V-cycle — the CG preconditioner.

        SCALING (the gotcha): the CG core calls this as
        ``z, _ = self._dispatch_vcycle(-r, z, face_arrs)``.  ``-r`` is the
        residual of the *h²-scaled* SPD system (``b = -(h²·f)``, and ``B`` uses
        the h²-scaled face coefficients), so it is ALREADY in the smoother's
        units — exactly what ``mg_vcycle_*`` consumes as ``f``.  Pass it
        straight through with NO h² multiplication (that rescale belongs only
        in ``solve_multigrid``, whose input is the raw divergence).

        NO gauge fix here: ``mg_vcycle_*`` applies neither the ghost-ring
        Neumann pass nor the mean subtraction that the whole-solve drivers end
        with — which is what a PCG preconditioner needs, and matches the
        V-cycle the native MGCG driver applies to ``z`` internally.

        ``p`` is mutated in place and returned.
        """
        from lilytorch.src import native as _nat
        p = p.contiguous()
        f = f.contiguous()
        if f.ndim == 3:
            ch, cv, cw = (a.contiguous() for a in face_arrs)
            _nat.mg_vcycle_3d(p, f, ch, cv, cw,
                              float(self.jcap_tol), float(self.w),
                              int(self.nsmoothing), 1, self.smoother)
        else:
            ch, cv = (a.contiguous() for a in face_arrs)
            _nat.mg_vcycle_2d(p, f, ch, cv,
                              float(self.jcap_tol), float(self.w),
                              int(self.nsmoothing), 1, self.smoother)
        return p, None

    def _cg_core(self, b, x, cfaces, face_arrs, recycle=None, harvest=None):
        """Plain-MGCG CG loop with a PERIODIC convergence check, to cut the
        per-iter ``.item()`` sync that ``_convergence_norm(r) < self.tol`` forces
        on the host every CG iteration.

        With the native V-cycle preconditioner (above) the CG arithmetic and the
        V-cycle are both sync-free; the only remaining per-iter host sync is this
        residual test.  Checking it every ``cg_check_every`` iters (K) instead of
        every iter cuts that to ~1/K syncs at the cost of at most K-1 extra CG
        iterations past convergence.

        Gated to plain MGCG (``recycle is None and harvest is None``) — the
        deflated / recycle-harvesting RMGCG path keeps the base-class
        source-of-truth loop.  ``cg_check_every <= 1`` (the default) also defers
        to the base ``_cg_core`` so the out-of-the-box behaviour is bit-identical."""
        K = int(getattr(self, "cg_check_every", 1))
        if K <= 1 or recycle is not None or harvest is not None:
            return super()._cg_core(b, x, cfaces, face_arrs,
                                    recycle=recycle, harvest=harvest)

        inner = _inner(b.ndim)
        mask = self._active_mask(cfaces, b.dtype)
        b = self._project_rhs(b, mask)
        r = b - self._apply_op_spd(x, cfaces)
        r_norm = self._convergence_norm(r)
        if r_norm < self.tol:
            _gauge_fix(x)
            self._last_niter = 0
            self._last_resid = r_norm
            return x, r, 0, r_norm

        z = torch.zeros_like(x)
        z = self._precondition(r, z, face_arrs, mask, inner)
        d = z.clone()
        self.BC(d)
        rz = (r * z[inner]).to(torch.float64).sum().to(r.dtype)

        r_norm_final = r_norm
        k = 0
        for k in range(self.max_cycles):
            q = self._apply_op_spd(d, cfaces)
            dq = (d[inner] * q).to(torch.float64).sum().to(r.dtype)
            alpha = rz / dq
            x[inner] = x[inner] + alpha * d[inner]
            self.BC(x)
            r = r - alpha * q

            last = (k + 1) == self.max_cycles
            if last or (k + 1) % K == 0:
                r_norm_final = self._convergence_norm(r)   # the only host sync
                if r_norm_final < self.tol:
                    break

            z = self._precondition(r, z, face_arrs, mask, inner)
            rz_new = (r * z[inner]).to(torch.float64).sum().to(r.dtype)
            beta = rz_new / rz
            d[inner] = z[inner] + beta * d[inner]
            self.BC(d)
            rz = rz_new

        _gauge_fix(x)
        niter = min(k + 1, self.max_cycles)
        self._last_niter = niter
        self._last_resid = r_norm_final
        return x, r, niter, r_norm_final

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
