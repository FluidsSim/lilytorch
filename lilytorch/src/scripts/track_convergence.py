"""
Track residual over multiple V-cycles for 2D and 3D.
"""
import torch, math, sys
sys.path.insert(0, ".")
from lilytorch.src.poisson_mult import PoissonSolver

dtype = torch.float64
device = "cpu"

def test_convergence(ndim, N, n_cycles=30, nsmoothing=5):
    L = 2 * math.pi
    h = L / N
    ng = N + 2
    coords = [torch.linspace(-h/2, L+h/2, ng, dtype=dtype, device=device)] * ndim
    grids = torch.meshgrid(*coords, indexing="ij")
    exact = torch.ones_like(grids[0])
    for g in grids:
        exact = exact * torch.sin(g)
    inner = tuple(slice(1,-1) for _ in range(ndim))
    f_inner = -float(ndim) * exact[inner]
    c = torch.ones([ng]*ndim, dtype=dtype, device=device)

    labels = ["ch", "cv", "cw"]
    face_arrs = []
    for d in range(ndim):
        idx_fwd = [slice(1,-1)]*ndim; idx_fwd[d] = slice(1, None)
        idx_bwd = [slice(1,-1)]*ndim; idx_bwd[d] = slice(None, -1)
        face_arrs.append(0.5*(c[tuple(idx_fwd)] + c[tuple(idx_bwd)]))

    # Run one V-cycle at a time to track residual
    ps = PoissonSolver(dtype, device, h, tol=1e-12, max_vcycles=1,
                       nsmoothing=nsmoothing, verbose=False, w=0.7)
    p = torch.zeros([ng]*ndim, dtype=dtype, device=device)
    f_scaled = h*h * f_inner
    cfaces = ps._extract_cfaces(face_arrs, ndim)

    print(f"\n{ndim}D N={N} nsmooth={nsmoothing}:")
    print(f"  {'Cycle':>5}  {'Residual L2':>14}  {'Ratio':>8}")

    prev_res = None
    for cyc in range(n_cycles):
        p, residual = ps._vcycle(f_scaled, p, face_arrs, 1)
        res = ps.l2_norm(residual)
        ratio = res / prev_res if prev_res else float('nan')
        if cyc < 10 or cyc % 5 == 0 or cyc == n_cycles-1:
            print(f"  {cyc:5d}  {res:14.6e}  {ratio:8.4f}")
        prev_res = res

# Quick tests
for N in [16, 32]:
    test_convergence(2, N, n_cycles=30, nsmoothing=5)
    test_convergence(3, N, n_cycles=30, nsmoothing=5)

# Also test 3D with more smoothing
test_convergence(3, 16, n_cycles=50, nsmoothing=20)
test_convergence(3, 32, n_cycles=50, nsmoothing=20)

print("\nDone.")
