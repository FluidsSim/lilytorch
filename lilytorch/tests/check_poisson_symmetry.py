"""Standalone SPD/symmetry check for the mgcg/rmgcg Poisson preconditioner.

Run:  python -m lilytorch.tests.check_poisson_symmetry

On a grid small enough to materialise dense matrices, this builds

    B      -- the SPD operator the CG loop applies   (_apply_op_spd)
    M_inv  -- the V-cycle preconditioner             (_dispatch_vcycle)

column by column, and reports ||X - X^T|| / ||X|| for each, the spectrum of
sym(M_inv), and the spectrum of M_inv*B.  CG is only valid when both operators
are symmetric, M_inv is positive definite, and the RHS lies in the range of B.

It then drives the actual CG core on a solid-body configuration with a
compatible and an incompatible RHS, which is where mgcg/rmgcg break: see the
module docstring of the results section below.

Findings this script is written to expose
-----------------------------------------
1. RANGE (the one that breaks production runs).  B has identically zero rows
   AND columns on degenerate cells (|J| < jcap_tol -- the BDIM solid interior
   and the near-degenerate mu0 band), so indicator vectors of those cells are
   in null(B).  But the CG drivers form b = -h^2*f without that mask, and
   div(u*) does not vanish inside a solid.  The resulting RHS component is
   unreachable: it never leaves the residual, and it contaminates the CG
   inner products r.z and d.Bd, so the iteration DIVERGES with iteration
   count.  Standalone multigrid is immune because mg_residual_* masks the
   residual, so the component never enters any inner product or the
   convergence test.

2. SYMMETRY.  The V-cycle zeroes the residual on degenerate cells on the way
   down (mg_residual_* masks) but prolongate_add writes the coarse correction
   into EVERY fine cell on the way up.  Zero column + nonzero row = M_inv is
   non-symmetric exactly on the degenerate cells (~27% relative, dtype
   independent).  The live-live block is symmetric to machine precision, so
   this is secondary to (1) -- but it is still an invalid preconditioner.

3. RBGS.  With smoother='rbgs' the V-cycle is non-symmetric (~5e-2) even with
   no degenerate cells at all, because the post-smooth repeats the red-black
   order instead of reversing it to black-red.  Latent: production uses
   'jacobi'.  Anyone selecting rbgs together with mgcg/rmgcg hits it.
"""
import numpy as np
import torch

from lilytorch.src.poisson_mult import PoissonSolver, _inner

DEV = "cuda"
SHAPE = (10, 10, 10)
# Padded-index box; leaves a 4^3 core whose every incident face coefficient is
# zero, i.e. genuinely degenerate cells rather than merely zero-c cells.
SOLID = (slice(3, 9), slice(3, 9), slice(3, 9))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build(shape, dtype, contrast=1.0, solid=None):
    ndim = len(shape)
    c = torch.ones([n + 2 for n in shape], dtype=dtype, device=DEV)
    if contrast != 1.0:
        idx = [slice(None)] * ndim
        idx[-1] = slice(shape[-1] // 2, None)
        c[tuple(idx)] = contrast
    if solid is not None:
        c[solid] = 0.0
    return [a.contiguous() for a in PoissonSolver._default_face_arrs(c, ndim)]


def solver(dtype, nsmoothing, smoother, max_cycles=40):
    return PoissonSolver(dtype, DEV, h=1.0, tol=1e-12, max_cycles=max_cycles,
                         nsmoothing=nsmoothing, w=2.0 / 3.0, verbose=False,
                         precond_vcycles=1, smoother=smoother)


def dense(fn, shape, dtype):
    e = torch.zeros(shape, dtype=dtype, device=DEV)
    cols = []
    for i in range(int(np.prod(shape))):
        e.zero_()
        e.view(-1)[i] = 1.0
        cols.append(fn(e).reshape(-1).double().cpu().numpy().copy())
    return np.stack(cols, axis=1)


def asym(X):
    return np.linalg.norm(X - X.T) / max(np.linalg.norm(X), 1e-300)


def operators(ps, cfaces, face_arrs, shape, dtype, mask=None):
    """Dense B and M_inv.

    ``mask=None`` probes the RAW V-cycle.  Passing the active mask probes the
    preconditioner as the CG drivers actually apply it (``_precondition``),
    which is the operator whose symmetry decides whether CG is valid.
    """
    inner = _inner(len(shape))
    pshape = tuple(n + 2 for n in shape)

    def apply_B(e):
        p = torch.zeros(pshape, dtype=dtype, device=DEV)
        p[inner] = e
        return ps._apply_op_spd(p, cfaces)

    def apply_Minv(r):
        src = r * mask if mask is not None else r
        z = torch.zeros(pshape, dtype=dtype, device=DEV)
        z, _ = ps._dispatch_vcycle((-src).contiguous(), z, face_arrs)
        out = z[inner].clone()
        return out * mask if mask is not None else out

    return dense(apply_B, shape, dtype), dense(apply_Minv, shape, dtype)


# ---------------------------------------------------------------------
# Section 1 — dense symmetry
# ---------------------------------------------------------------------
def section_symmetry():
    print("=" * 74)
    print("1. Dense symmetry of B and of the V-cycle preconditioner M_inv")
    print("=" * 74)
    print("   'raw' = bare V-cycle;  'used' = as the CG drivers apply it, i.e.")
    print("   with the active mask (_precondition).  'used' is what CG needs.")
    print(f"\n{'case':<40} {'asym(B)':>10} {'raw M':>10} {'used M':>10} "
          f"{'min eig':>11}")

    cases = [
        ("f64 uniform, nsmooth=1",      SHAPE, torch.float64, 1, "jacobi", 1.0,  None),
        ("f64 uniform, nsmooth=5",      SHAPE, torch.float64, 5, "jacobi", 1.0,  None),
        ("f64 odd coarsening 12x10x6",  (12, 10, 6), torch.float64, 5, "jacobi", 1.0, None),
        ("f64 two-phase 1000:1",        SHAPE, torch.float64, 5, "jacobi", 1e-3, None),
        ("f32 uniform, nsmooth=5",      SHAPE, torch.float32, 5, "jacobi", 1.0,  None),
        ("f64 RBGS smoother",           SHAPE, torch.float64, 2, "rbgs",   1.0,  None),
        ("f64 SOLID BLOCK (degenerate)", SHAPE, torch.float64, 5, "jacobi", 1.0, SOLID),
        ("f32 SOLID BLOCK (degenerate)", SHAPE, torch.float32, 5, "jacobi", 1.0, SOLID),
    ]
    for label, shape, dtype, nsm, sm, contrast, solid in cases:
        fa = build(shape, dtype, contrast, solid)
        ps = solver(dtype, nsm, sm)
        cf = ps._extract_cfaces(fa, len(shape))
        mask = ps._active_mask(cf, dtype)
        B, M_raw = operators(ps, cf, fa, shape, dtype)
        _, M_use = operators(ps, cf, fa, shape, dtype, mask=mask)
        live = np.flatnonzero(mask.reshape(-1).cpu().numpy())
        Ml = M_use[np.ix_(live, live)]
        ev = np.linalg.eigvalsh(0.5 * (Ml + Ml.T))
        flag = "  <-- ASYMMETRIC" if asym(M_use) > 1e-6 else ""
        print(f"{label:<40} {asym(B):10.2e} {asym(M_raw):10.2e} "
              f"{asym(M_use):10.2e} {ev.min():11.3e}{flag}")


# ---------------------------------------------------------------------
# Section 2 — where the asymmetry lives
# ---------------------------------------------------------------------
def section_localise():
    print()
    print("=" * 74)
    print("2. Localising the solid-block asymmetry to the degenerate cells")
    print("=" * 74)
    dtype = torch.float64
    fa = build(SHAPE, dtype, 1.0, SOLID)
    ps = solver(dtype, 5, "jacobi")
    cf = ps._extract_cfaces(fa, 3)
    active = ps.compute_J(cf).abs() >= ps.jcap_tol
    live = np.flatnonzero(active.reshape(-1).cpu().numpy())
    print(f"  degenerate cells: {int((~active).sum())}/{active.numel()}")

    for tag, mask in (("as-is", None), ("with active mask", active.to(dtype))):
        _, M = operators(ps, cf, fa, SHAPE, dtype, mask=mask)
        Ml = M[np.ix_(live, live)]
        ev = np.linalg.eigvalsh(0.5 * (Ml + Ml.T))
        print(f"  {tag:<18} asym(M_inv) = {asym(M):.3e}   "
              f"live-block asym = {asym(Ml):.3e}   "
              f"live-block min eig = {ev.min():.3e}")
    print("  => the live-live block is SPD; all asymmetry is in the "
          "degenerate rows.")


# ---------------------------------------------------------------------
# Section 3 — the range condition, which is what actually diverges
# ---------------------------------------------------------------------
def cg_trace(ps, cf, fa, b, active, mask_z, mask_b, nit, label):
    dtype = b.dtype
    inner = _inner(b.ndim)
    pshape = tuple(n + 2 for n in SHAPE)
    m = active.to(dtype)
    x = torch.zeros(pshape, dtype=dtype, device=DEV)
    b = b * m if mask_b else b.clone()

    def M(r):
        z = torch.zeros(pshape, dtype=dtype, device=DEV)
        z, _ = ps._dispatch_vcycle((-r).contiguous(), z, fa)
        if mask_z:
            z[inner] *= m
        return z

    r = b - ps._apply_op_spd(x, cf)
    z = M(r)
    d = z.clone()
    ps.BC(d)
    rz = (r * z[inner]).sum()
    hist = []
    for _ in range(nit):
        q = ps._apply_op_spd(d, cf)
        alpha = rz / (d[inner] * q).sum()
        x[inner] += alpha * d[inner]
        ps.BC(x)
        r = r - alpha * q
        hist.append(r[active].abs().max().item())
        z = M(r)
        rz_new = (r * z[inner]).sum()
        d[inner] = z[inner] + (rz_new / rz) * d[inner]
        ps.BC(d)
        rz = rz_new
    pts = [i for i in (0, 1, 2, 4, 7, 11, 15, 19, 24, 29) if i < len(hist)]
    print(f"  {label:<36} " + "  ".join(f"{i}:{hist[i]:.1e}" for i in pts))


def section_range():
    print()
    print("=" * 74)
    print("3. RHS range condition -- max|r| on LIVE cells vs CG iteration")
    print("=" * 74)
    print("   This section drives its OWN CG loop so it can switch the fix on")
    print("   and off; 'as-is' reproduces the pre-fix behaviour and is expected")
    print("   to diverge.  Section 5 exercises the shipped solvers.")
    dtype = torch.float64
    fa = build(SHAPE, dtype, 1.0, SOLID)
    ps = solver(dtype, 5, "jacobi")
    cf = ps._extract_cfaces(fa, 3)
    active = ps.compute_J(cf).abs() >= ps.jcap_tol
    m = active.to(dtype)

    print("\n  COMPATIBLE RHS  b = B(x_true)  (exactly in the range of B)")
    g = torch.Generator(device=DEV)
    g.manual_seed(3)
    xt = torch.randn(tuple(n + 2 for n in SHAPE), dtype=dtype, device=DEV,
                     generator=g)
    ps.BC(xt)
    b_ok = ps._apply_op_spd(xt, cf).clone()
    cg_trace(ps, cf, fa, b_ok, active, False, False, 30, "as-is")

    print("\n  INCOMPATIBLE RHS  b = -h^2 f, f nonzero inside the solid")
    print("  (the realistic case: div(u*) does not vanish in BDIM solid cells)")
    g.manual_seed(7)
    f = torch.randn(SHAPE, dtype=dtype, device=DEV, generator=g)
    f -= f.mean()
    b = -(ps.h2 * f)
    cg_trace(ps, cf, fa, b, active, False, False, 30, "as-is")
    cg_trace(ps, cf, fa, b, active, True, True, 30, "masked z and b")
    bm = b * m
    bm = bm - (bm.sum() / m.sum()) * m
    cg_trace(ps, cf, fa, bm, active, True, True, 30,
             "masked + projected onto range(B)")
    print("\n  => only the last one converges.  Masking alone is not enough:"
          "\n     the remaining constant must also be projected out.")


# ---------------------------------------------------------------------
# Section 4 — how big the degenerate set actually is in production
# ---------------------------------------------------------------------
def section_jcap():
    print()
    print("=" * 74)
    print("4. jcap_tol is ABSOLUTE (1e-7) but the BDIM coefficient is "
          "dt*mu0/rho")
    print("=" * 74)
    n = 32
    z = torch.linspace(-1, 1, n + 2, dtype=torch.float64, device=DEV)
    X, Y, Z = torch.meshgrid(z, z, z, indexing="ij")
    rad = (X ** 2 + Y ** 2 + Z ** 2).sqrt()
    eps = 2.0 * (2.0 / n)
    mu0 = 0.5 * (1 + torch.tanh((rad - 0.5) / eps))
    ps = PoissonSolver(torch.float64, DEV, h=1.0, verbose=False)
    print(f"  jcap_tol = {ps.jcap_tol:.1e}\n")
    print(f"  {'coeff scale':>12} {'max J':>11} {'frozen cells':>16} {'%':>8}")
    for scale in (1e0, 1e-3, 1e-5, 1e-6, 1e-7, 1e-8):
        c = (scale * mu0).contiguous()
        arrs = [a.contiguous()
                for a in PoissonSolver._default_face_arrs(c, 3)]
        J = ps.compute_J(ps._extract_cfaces(arrs, 3))
        dead = int((J.abs() < ps.jcap_tol).sum())
        print(f"  {scale:12.0e} {float(J.max()):11.2e} "
              f"{dead:8d}/{J.numel():<7d} {100 * dead / J.numel():7.1f}%")
    print("\n  solver.py puts dt*mu0/rho at '~1e-7 scale' -- the same order as"
          "\n  the ABSOLUTE threshold, so the frozen set is dt- and"
          "\n  resolution-dependent and can cover a wide band, not just the"
          "\n  solid interior.")


# ---------------------------------------------------------------------
# Section 5 — the real solve entry points, end to end
# ---------------------------------------------------------------------
def section_drivers():
    print()
    print("=" * 74)
    print("5. End-to-end solve_* on a solid body with an incompatible RHS")
    print("=" * 74)
    print("   Realistic setup: div(u*) is nonzero inside the solid, which is")
    print("   where the CG drivers used to diverge.  On CUDA these run the")
    print("   native whole-solve drivers, not the Python loop.\n")

    shape = (32, 32, 32)
    dtype = torch.float32          # the production dtype
    ndim = 3
    n = shape[0]
    z = torch.linspace(-1, 1, n + 2, dtype=dtype, device=DEV)
    X, Y, Z = torch.meshgrid(z, z, z, indexing="ij")
    rad = (X ** 2 + Y ** 2 + Z ** 2).sqrt()
    eps = 2.0 * (2.0 / n)
    # BDIM-like mu0: 1 in fluid, 0 inside a sphere, smoothed over ~2 cells.
    c = 1e-6 * (0.5 * (1 + torch.tanh((rad - 0.4) / eps)))
    fa = [a.contiguous() for a in PoissonSolver._default_face_arrs(c, ndim)]

    g = torch.Generator(device=DEV)
    g.manual_seed(11)
    f = torch.randn(shape, dtype=dtype, device=DEV, generator=g)
    f -= f.mean()

    print(f"{'smoother':<10} {'method':<12} {'iters':>6} {'max|r| live':>13} "
          f"{'max|r| all':>13}")
    for sm in ("jacobi", "rbgs"):
        for method, k in (("multigrid", 0), ("mgcg", 0), ("rmgcg", 4)):
            ps = PoissonSolver(dtype, DEV, h=1.0, tol=1e-8, max_cycles=40,
                               max_vcycles=40, nsmoothing=5, w=2.0 / 3.0,
                               verbose=False, smoother=sm, recycle_k=k)
            cf = ps._extract_cfaces(fa, ndim)
            active = ps.compute_J(cf).abs() >= ps.jcap_tol
            p0 = torch.zeros(tuple(s + 2 for s in shape), dtype=dtype,
                             device=DEV)
            fn = {"multigrid": ps.solve_multigrid, "mgcg": ps.solve_mgcg,
                  "rmgcg": ps.solve_rmgcg}[method]
            _, r = fn(f.clone(), p0, ch=fa[0], cv=fa[1], cw=fa[2])
            print(f"{sm:<10} {method:<12} {ps._last_niter:>6} "
                  f"{r[active].abs().max().item():13.3e} "
                  f"{r.abs().max().item():13.3e}")
    print(f"\n  degenerate cells: {int((~active).sum())}/{active.numel()}")


if __name__ == "__main__":
    torch.cuda.init()
    section_symmetry()
    section_localise()
    section_range()
    section_jcap()
    section_drivers()
