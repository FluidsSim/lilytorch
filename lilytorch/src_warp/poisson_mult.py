"""``src_warp.poisson_mult`` — multigrid/MGCG/RMGCG Poisson on **Warp** kernels.

The native ``poisson_solve_*`` is a monolithic C++ driver with no Python-injectable
smoother seam.  But the solver also has a pure-Python outer driver
(``solve_multigrid``/``solve_mgcg``/``solve_rmgcg`` with ``use_kernels=False``)
whose only *custom-kernel* ops are the fine-level smoother + residual
(``rbgs_sweep_*`` / ``jacobi_sweep_*`` / ``mg_residual_*``); restriction,
prolongation, the coarse recursion, and the CG / Aitken loops are plain torch.

This subclass routes the Poisson solve onto Warp by:

1. overriding :meth:`_dispatch_vcycle` to run the fine-level smoother + residual
   on the single-source Warp kernels (``warp_poc.warp_poisson{,_2d}``), reusing
   the native module's pure-torch restriction / prolongation / coarse V-cycle
   (no ``lilytorch_kernels`` op is touched);
2. forcing the three top-level ``solve_*`` entry points onto that Python+Warp
   path (so the native C++ ``poisson_solve_*`` driver is never called).

The Warp smoothers fold homogeneous Neumann into the stencil by index-clamping
(ghost = self at the boundary), identical math to native's explicit ghost
refresh, and run on Warp ``device="cpu"`` *and* ``"cuda:0"`` from one source.
Parity vs native is residual-level (same converged residual on a manufactured
Neumann Poisson), not bit-exact — the native 2-D fine smoother is a tiled
stale-halo variant, but the V-cycle converges to the same solution.
"""
import torch

from lilytorch.src.poisson_mult import PoissonSolver as _BasePoissonSolver
from lilytorch.src.poisson_mult import _inner
# Pure-torch multigrid helpers (no custom-kernel ops) reused verbatim.
from lilytorch.src.poisson_mult import (
    _restrict_face_2d, _restrict_residual_2d, _prolongate_2d,
    _vcycle_rbgs_2d, _vcycle_jac_2d,
    _restrict_face_3d, _restrict_residual_3d, _prolongate_3d,
    _vcycle_rbgs_3d, _vcycle_jac_3d,
)
from lilytorch.warp_poc.warp_poisson_2d import (
    rbgs_sweep_2d_warp, jacobi_sweep_2d_warp, mg_residual_2d_warp,
)
from lilytorch.warp_poc.warp_poisson import (
    rbgs_sweep_3d_warp, jacobi_sweep_3d_warp, mg_residual_3d_warp,
)


# ── Hybrid V-cycles: Warp fine-level smoother+residual, pure-torch coarse ────
# Structure copied from src.poisson_mult._vcycle_{rbgs,jac}_{2,3}d_native; the
# only change is the two native ops → Warp wrappers (and the contiguous-coeff
# extraction stays identical).

def _vcycle_rbgs_2d_warp(f, p, ch, cv, jcap_tol, nsmoothing):
    ch = ch.contiguous(); cv = cv.contiguous()
    cp0, cm0 = ch[1:, :], ch[:-1, :]
    cp1, cm1 = cv[:, 1:], cv[:, :-1]
    rbgs_sweep_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing)
    r = mg_residual_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol)
    nx, ny = f.shape
    if nx > 2 and ny > 2:
        ch_c, cv_c = _restrict_face_2d(ch, cv)
        r_c = _restrict_residual_2d(r); del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        err_c, _ = _vcycle_rbgs_2d(r_c, p_c, ch_c, cv_c, jcap_tol, nsmoothing)
        err = _prolongate_2d(err_c, f.shape)
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + err
        cp0, cm0 = ch[1:, :], ch[:-1, :]
        cp1, cm1 = cv[:, 1:], cv[:, :-1]
        rbgs_sweep_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing)
        r = mg_residual_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol)
    return p, r


def _vcycle_jac_2d_warp(f, p, ch, cv, w, jcap_tol, nsmoothing):
    ch = ch.contiguous(); cv = cv.contiguous()
    cp0, cm0 = ch[1:, :], ch[:-1, :]
    cp1, cm1 = cv[:, 1:], cv[:, :-1]
    jacobi_sweep_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing)
    r = mg_residual_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol)
    nx, ny = f.shape
    if nx > 2 and ny > 2:
        ch_c, cv_c = _restrict_face_2d(ch, cv)
        r_c = _restrict_residual_2d(r); del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        err_c, _ = _vcycle_jac_2d(r_c, p_c, ch_c, cv_c, w, jcap_tol, nsmoothing)
        err = _prolongate_2d(err_c, f.shape)
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + err
        cp0, cm0 = ch[1:, :], ch[:-1, :]
        cp1, cm1 = cv[:, 1:], cv[:, :-1]
        jacobi_sweep_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing)
        r = mg_residual_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol)
    return p, r


def _vcycle_rbgs_3d_warp(f, p, ch, cv, cw, jcap_tol, nsmoothing):
    ch = ch.contiguous(); cv = cv.contiguous(); cw = cw.contiguous()
    cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
    cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
    cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
    rbgs_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, nsmoothing)
    r = mg_residual_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)
    nx, ny, nz = f.shape
    if nx > 2 and ny > 2 and nz > 2:
        ch_c, cv_c, cw_c = _restrict_face_3d(ch, cv, cw)
        r_c = _restrict_residual_3d(r); del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2, r_c.shape[2] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        err_c, _ = _vcycle_rbgs_3d(r_c, p_c, ch_c, cv_c, cw_c, jcap_tol, nsmoothing)
        err = _prolongate_3d(err_c, f.shape)
        p[1:-1, 1:-1, 1:-1] = p[1:-1, 1:-1, 1:-1] + err
        cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
        cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
        cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
        rbgs_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, nsmoothing)
        r = mg_residual_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)
    return p, r


def _vcycle_jac_3d_warp(f, p, ch, cv, cw, w, jcap_tol, nsmoothing):
    ch = ch.contiguous(); cv = cv.contiguous(); cw = cw.contiguous()
    cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
    cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
    cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
    jacobi_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, w, nsmoothing)
    r = mg_residual_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)
    nx, ny, nz = f.shape
    if nx > 2 and ny > 2 and nz > 2:
        ch_c, cv_c, cw_c = _restrict_face_3d(ch, cv, cw)
        r_c = _restrict_residual_3d(r); del r
        coarse_shape = (r_c.shape[0] + 2, r_c.shape[1] + 2, r_c.shape[2] + 2)
        p_c = torch.zeros(coarse_shape, device=p.device, dtype=p.dtype)
        err_c, _ = _vcycle_jac_3d(r_c, p_c, ch_c, cv_c, cw_c, w, jcap_tol, nsmoothing)
        err = _prolongate_3d(err_c, f.shape)
        p[1:-1, 1:-1, 1:-1] = p[1:-1, 1:-1, 1:-1] + err
        cp0, cm0 = ch[1:, :, :], ch[:-1, :, :]
        cp1, cm1 = cv[:, 1:, :], cv[:, :-1, :]
        cp2, cm2 = cw[:, :, 1:], cw[:, :, :-1]
        jacobi_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, w, nsmoothing)
        r = mg_residual_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol)
    return p, r


class PoissonSolver(_BasePoissonSolver):
    """Variable-coefficient multigrid Poisson with the fine-level smoother +
    residual on Warp kernels (CPU + GPU, f32 + f64).

    The three top-level solvers are forced onto the Python outer driver (so the
    native C++ ``poisson_solve_*`` is bypassed) and :meth:`_dispatch_vcycle` runs
    the Warp hybrid V-cycle."""

    #: op names this backend now serves on Warp (for the WARP_BACKED accounting).
    WARP_POISSON_OPS = frozenset({
        "rbgs_sweep_2d", "rbgs_sweep_3d", "jacobi_sweep_2d", "jacobi_sweep_3d",
        "mg_residual_2d", "mg_residual_3d",
    })

    def _graphed_mg(self, ndim, shape, nvc=None):
        """Lazily build / fetch a CUDA-graph-captured all-Warp multigrid keyed by
        (ndim, shape, dtype, smoother, nsmoothing, n_vcycles).

        ``nvc`` overrides the captured fixed V-cycle count; defaults to
        ``max_vcycles`` (the standalone-multigrid budget).  The MGCG
        preconditioner passes ``nvc=1`` (one captured V-cycle per
        :meth:`_dispatch_vcycle`, repeated by the CG core's ``precond_vcycles``
        loop)."""
        from lilytorch.warp_poc.warp_mg_var import WarpMG3D, WarpMG2D
        cache = getattr(self, "_warp_mg_cache", None)
        if cache is None:
            cache = self._warp_mg_cache = {}
        if nvc is None:
            nvc = self.max_vcycles
        nvc = max(int(nvc), 1)
        key = (ndim, tuple(shape), self.dtype, self.smoother, self.nsmoothing, nvc)
        mg = cache.get(key)
        if mg is None:
            if ndim == 3:
                mg = WarpMG3D(shape[0], shape[1], shape[2], device=self.device,
                              dtype=self.dtype, smoother=self.smoother,
                              nu1=self.nsmoothing, nu2=self.nsmoothing,
                              n_vcycles=nvc, jcap_tol=self.jcap_tol)
            elif ndim == 2 and self.dtype == torch.float64 and self.smoother == "rbgs":
                # 2-D graphed MG is f64 + rbgs (the 2-D MG kernels are f64);
                # other 2-D configs fall through to the Python driver.
                mg = WarpMG2D(shape[0], shape[1], device=self.device,
                              nu1=self.nsmoothing, nu2=self.nsmoothing,
                              n_vcycles=nvc, jcap_tol=self.jcap_tol)
            else:
                return None
            cache[key] = mg
        return mg

    def solve_multigrid(self, f, p0, **kwargs):
        """CUDA-graphed all-Warp multigrid when ``cuda_graph`` is on (3-D),
        else the Item-2 Python-driver path.  The graphed path replays a captured,
        sync-free, fixed-(``max_vcycles``)-cycle V-cycle in one host launch —
        ~4× over the Python loop and competitive with the native C++ driver."""
        if getattr(self, "cuda_graph", False) and f.ndim in (2, 3):
            face_arrs, _ = self._face_arrs_from_kwargs(kwargs, f.ndim)
            mg = self._graphed_mg(f.ndim, tuple(f.shape))
            if mg is not None and face_arrs is not None:
                pre_scaled = kwargs.get("pre_scaled", False)
                f_scaled = f if pre_scaled else self.h2 * f
                if f.ndim == 3:
                    ch, cv, cw = face_arrs
                    p = mg.solve(f_scaled, ch.contiguous(), cv.contiguous(),
                                 cw.contiguous(), p0=p0)
                    if self.dirichlet_mask is None:
                        p -= p.to(torch.float64).mean().to(p.dtype)
                    from lilytorch.warp_poc.warp_poisson import mg_residual_3d_warp
                    cp0, cm0 = ch[1:].contiguous(), ch[:-1].contiguous()
                    cp1, cm1 = cv[:, 1:].contiguous(), cv[:, :-1].contiguous()
                    cp2, cm2 = cw[:, :, 1:].contiguous(), cw[:, :, :-1].contiguous()
                    r = mg_residual_3d_warp(p, f_scaled, cp0, cm0, cp1, cm1,
                                            cp2, cm2, self.jcap_tol)
                    return p, r
                ch, cv = face_arrs
                p = mg.solve(f_scaled, ch.contiguous(), cv.contiguous(), p0=p0)
                if self.dirichlet_mask is None:
                    p -= p.to(torch.float64).mean().to(p.dtype)
                from lilytorch.warp_poc.warp_mg_var import mg_residual_2d_clamped_warp
                cp0, cm0 = ch[1:].contiguous(), ch[:-1].contiguous()
                cp1, cm1 = cv[:, 1:].contiguous(), cv[:, :-1].contiguous()
                r = mg_residual_2d_clamped_warp(p, f_scaled, cp0, cm0, cp1, cm1,
                                                self.jcap_tol)
                return p, r
        # Fallback: Item-2 Python outer driver (bypass the native C++ poisson_solve_*).
        saved = self.use_kernels
        self.use_kernels = False
        try:
            return super().solve_multigrid(f, p0, **kwargs)
        finally:
            self.use_kernels = saved

    def _dispatch_vcycle(self, f, p, face_arrs):
        """One V-cycle with the fine-level smoother + residual on Warp.

        When ``cuda_graph`` is on (and a graphed ``WarpMG`` is available for this
        shape/dtype/smoother), the whole V-cycle is replayed from ONE captured
        CUDA graph in a single host launch — the sync-free MGCG preconditioner
        (C1).  Otherwise it mirrors the native ``_dispatch_vcycle`` hybrid (Warp
        fine level + pure-torch coarse recursion), which also runs on CPU.

        SCALING (the C1 gotcha): the CG core calls this as
        ``z, _ = self._dispatch_vcycle(-r, z, face_arrs)``.  ``-r`` is the
        residual of the *h²-scaled* SPD system (``b = -(h²·f)``, ``B`` uses the
        h²-scaled face coefficients), so it is ALREADY in the smoother's units —
        exactly what the native ``_vcycle_*_warp`` smoothers consume as ``f``
        (no internal h² rescale).  ``WarpMG.solve`` likewise treats its ``f``
        argument as the raw smoother RHS, so we pass ``f`` straight through with
        **no h² multiplication** (the ``h²`` rescale belongs only in
        ``solve_multigrid``, where the input is the raw divergence)."""
        if getattr(self, "cuda_graph", False) and p.is_cuda and f.ndim in (2, 3):
            mg = self._graphed_mg(f.ndim, tuple(f.shape), nvc=1)
            if mg is not None:
                if f.ndim == 3:
                    ch, cv, cw = face_arrs
                    z = mg.solve(f, ch.contiguous(), cv.contiguous(),
                                 cw.contiguous(), p0=p)
                else:
                    ch, cv = face_arrs
                    z = mg.solve(f, ch.contiguous(), cv.contiguous(), p0=p)
                return z, None
        ndim = f.ndim
        if ndim == 3:
            ch, cv, cw = face_arrs
            if self.smoother == "rbgs":
                return _vcycle_rbgs_3d_warp(
                    f, p, ch, cv, cw, self.jcap_tol, self.nsmoothing)
            return _vcycle_jac_3d_warp(
                f, p, ch, cv, cw, self.w, self.jcap_tol, self.nsmoothing)
        ch, cv = face_arrs
        if self.smoother == "rbgs":
            return _vcycle_rbgs_2d_warp(
                f, p, ch, cv, self.jcap_tol, self.nsmoothing)
        return _vcycle_jac_2d_warp(
            f, p, ch, cv, self.w, self.jcap_tol, self.nsmoothing)

    # ── Force the Python outer driver (bypass the native C++ poisson_solve_*) ──
    def solve_mgcg(self, f, p0, **kwargs):
        saved = self.use_kernels
        self.use_kernels = False
        try:
            return super().solve_mgcg(f, p0, **kwargs)
        finally:
            self.use_kernels = saved

    def solve_rmgcg(self, f, p0, **kwargs):
        saved = self.use_kernels
        self.use_kernels = False
        try:
            return super().solve_rmgcg(f, p0, **kwargs)
        finally:
            self.use_kernels = saved

    # ── Sync-free MGCG: periodic convergence check (C1, point 4) ──────────────
    def _cg_core(self, b, x, cfaces, face_arrs, recycle=None, harvest=None):
        """Plain-MGCG CG loop with a PERIODIC convergence check, to cut the
        per-iter ``.item()`` sync that ``_convergence_norm(r) < self.tol`` forces
        on the host every CG iteration.

        With the graphed WarpMG preconditioner (above) the CG arithmetic and the
        V-cycle are both sync-free; the only remaining per-iter host sync is this
        residual test.  Checking it every ``cg_check_every`` iters (K) instead of
        every iter cuts that to ~1/K syncs at the cost of at most K-1 extra CG
        iterations past convergence.

        Gated to plain MGCG (``recycle is None and harvest is None``) — the
        deflated / recycle-harvesting RMGCG path keeps the native source-of-truth
        loop.  ``cg_check_every <= 1`` (the default) also defers to the native
        ``_cg_core`` so the out-of-the-box behaviour is bit-identical."""
        K = int(getattr(self, "cg_check_every", 1))
        if K <= 1 or recycle is not None or harvest is not None:
            return super()._cg_core(b, x, cfaces, face_arrs,
                                    recycle=recycle, harvest=harvest)

        inner = _inner(b.ndim)
        r = b - self._apply_op_spd(x, cfaces)
        r_norm = self._convergence_norm(r)
        if r_norm < self.tol:
            x -= x.to(torch.float64).mean().to(x.dtype)
            self._last_niter = 0
            return x, r, 0, r_norm

        z = torch.zeros_like(x)
        for _ in range(self.precond_vcycles):
            z, _ = self._dispatch_vcycle(-r, z, face_arrs)
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

            z.zero_()
            for _ in range(self.precond_vcycles):
                z, _ = self._dispatch_vcycle(-r, z, face_arrs)
            rz_new = (r * z[inner]).to(torch.float64).sum().to(r.dtype)
            beta = rz_new / rz
            d[inner] = z[inner] + beta * d[inner]
            self.BC(d)
            rz = rz_new

        x -= x.to(torch.float64).mean().to(x.dtype)
        niter = min(k + 1, self.max_cycles)
        self._last_niter = niter
        return x, r, niter, r_norm_final
