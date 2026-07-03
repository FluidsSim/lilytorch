"""A/B harness: recycled MGCG vs plain MGCG on a slow-moving-body sequence.

This emulates the pressure Poisson problem of a slow swimmer without needing
FARMS/MuJoCo: a smoothed body (sphere SDF → mu0 Heaviside) translates a small
step each "timestep", producing the BDIM-style variable coefficient field
``c = dt * mu0 / rho`` on the staggered faces.  The RHS is a fixed divergence
pattern advected with the body.  Both solvers warm-start from the previous
pressure, exactly as the real step loop does; the only difference is that the
RMGCG instance recycles its Krylov subspace across steps.

Run:
    python -m lilytorch.src._verify_rmgcg                # 2-D, k=10
    DIM=3 K=12 STEPS=40 python -m lilytorch.src._verify_rmgcg

It prints per-step CG iteration counts for both methods plus the speed-up, and
checks that RMGCG reaches the same solution as MGCG (max field diff).
"""
import os
import time

import torch

from lilytorch.src.poisson_mult import PoissonSolver


def _smooth_heaviside(sdf, eps):
    """BDIM mu0: 1 in fluid, 0 in body, smoothed over a band of width ~eps."""
    return 0.5 * (1.0 + torch.tanh(sdf / eps))


def _sphere_sdf(coords, center, radius):
    r2 = sum((c - c0) ** 2 for c, c0 in zip(coords, center))
    return torch.sqrt(r2) - radius


def _face_coeffs(c, ndim):
    """Average cell-centred coefficient c onto faces → (ch, cv[, cw])."""
    if ndim == 2:
        ch = 0.5 * (c[1:, 1:-1] + c[:-1, 1:-1])
        cv = 0.5 * (c[1:-1, 1:] + c[1:-1, :-1])
        return {"ch": ch, "cv": cv}
    ch = 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1])
    cw = 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])
    return {"ch": ch, "cv": cv, "cw": cw}


def run(ndim=2, N=128, steps=30, recycle_k=10, contrast=100.0, seed=0,
        tol=1e-6, nsmoothing=2, smoother="rbgs", precond_vcycles=1):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt_t = torch.float64
    torch.manual_seed(seed)

    L = 1.0
    h = L / N
    n = N + 2
    eps = 2.0 * h                              # mu0 smoothing band (~2 cells)
    radius = 0.18 * L

    ax = torch.linspace(-h / 2, L + h / 2, n, dtype=dt_t, device=dev)
    grids = torch.meshgrid(*([ax] * ndim), indexing="ij")

    # Body path: a slow diagonal drift of ~0.4 cell / step (genuinely slow).
    drift = 0.4 * h
    c0 = [0.35 * L] * ndim

    def make_problem(step):
        center = [c0[d] + drift * step for d in range(ndim)]
        sdf = _sphere_sdf(grids, center, radius)
        mu0 = _smooth_heaviside(sdf, eps)
        # Coefficient jump: c=1 in fluid, c=1/contrast inside the body.  A
        # smooth, well-posed variable-coefficient Poisson (the kind MGCG is
        # built for) whose jump location moves slightly each step.
        c = (1.0 / contrast) + (1.0 - 1.0 / contrast) * mu0
        faces = _face_coeffs(c, ndim)
        # RHS: smooth divergence blob riding with the body (zero mean).
        r2 = sum((g - cc) ** 2 for g, cc in zip(grids, center))
        f_full = torch.exp(-r2 / (2 * (0.12 * L) ** 2))
        inner = tuple([slice(1, -1)] * ndim)
        f = f_full[inner].clone()
        f -= f.mean()
        return faces, f

    def make_solver(k):
        return PoissonSolver(
            dt_t, dev, h,
            tol=tol, max_cycles=200, max_vcycles=1, nsmoothing=nsmoothing,
            w=1.0, verbose=False, precond_vcycles=precond_vcycles,
            smoother=smoother, recycle_k=k,
        )

    warm = os.environ.get("WARM", "1") == "1"
    solvers = {"mgcg": make_solver(0), "rmgcg": make_solver(recycle_k)}
    p0 = {m: torch.zeros(*([n] * ndim), dtype=dt_t, device=dev) for m in solvers}
    iters = {m: [] for m in solvers}
    times = {m: 0.0 for m in solvers}
    last_p = {}

    print(f"\n=== {ndim}-D  N={N}  steps={steps}  recycle_k={recycle_k}  "
          f"device={dev} ===")
    print(f"    precond: {smoother} x{nsmoothing}, {precond_vcycles} vcycle/iter, "
          f"tol={tol:g}, contrast={contrast:g}, backend=warp")
    print(f"{'step':>4} | {'mgcg it':>7} | {'rmgcg it':>8} | {'saved':>6}")
    print("-" * 36)
    for step in range(steps):
        faces, f = make_problem(step)
        for m, solver in solvers.items():
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            # Both go through solve_rmgcg; the "mgcg" solver has recycle_k=0,
            # which is the plain path (no deflation/harvest) — a clean A/B on
            # one code path that exposes _last_niter for both.
            p, _ = solver.solve_rmgcg(f, p0[m].clone(), **faces)
            if dev == "cuda":
                torch.cuda.synchronize()
            times[m] += time.time() - t0
            iters[m].append(solver._last_niter)
            if warm:
                p0[m] = p                      # warm start next step
            last_p[m] = p
        it_mg, it_rmg = iters["mgcg"][-1], iters["rmgcg"][-1]
        print(f"{step:>4} | {it_mg:>7} | {it_rmg:>8} | {it_mg - it_rmg:>6}")

    # Summary -----------------------------------------------------------
    def mean(xs):
        return sum(xs) / len(xs)

    # Ignore the first few steps (recycle space still filling) for the mean.
    warm = max(1, recycle_k // 2)
    mg_mean = mean(iters["mgcg"][warm:])
    rm_mean = mean(iters["rmgcg"][warm:])
    diff = (last_p["rmgcg"] - last_p["mgcg"]).abs().max().item()
    pscale = last_p["mgcg"].abs().max().item() + 1e-30

    print("-" * 36)
    print(f"mean CG iters (after warm-up step {warm}):")
    print(f"  mgcg  = {mg_mean:.2f}")
    print(f"  rmgcg = {rm_mean:.2f}   ({mg_mean / max(rm_mean, 1e-9):.2f}x fewer)")
    print(f"total solve time:  mgcg {times['mgcg']:.3f}s   "
          f"rmgcg {times['rmgcg']:.3f}s   "
          f"({times['mgcg'] / max(times['rmgcg'], 1e-9):.2f}x)")
    print(f"max |p_rmgcg - p_mgcg| = {diff:.2e}  "
          f"(rel {diff / pscale:.2e})  [both solved to tol=1e-6]")


if __name__ == "__main__":
    ndim = int(os.environ.get("DIM", 2))
    N = int(os.environ.get("N", 128 if ndim == 2 else 64))
    steps = int(os.environ.get("STEPS", 30))
    k = int(os.environ.get("K", 10))
    run(
        ndim=ndim, N=N, steps=steps, recycle_k=k,
        contrast=float(os.environ.get("CONTRAST", 100.0)),
        tol=float(os.environ.get("TOL", 1e-6)),
        nsmoothing=int(os.environ.get("NSMOOTH", 2)),
        smoother=os.environ.get("SMOOTH", "rbgs"),
        precond_vcycles=int(os.environ.get("PV", 1)),
    )
