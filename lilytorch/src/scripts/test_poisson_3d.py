"""
Targeted Poisson multigrid diagnostic.
Tests:
  1. Constant coefficients (sanity)
  2. Variable coefficients with body-like pattern (c=0 inside sphere, 1 outside)
  3. Actual NS-like RHS (divergence from impulsive start)
All in 2D and 3D.
"""
import torch, math, sys
sys.path.insert(0, ".")
from lilytorch.src.poisson_mult import PoissonSolver

dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

def test_poisson(label, ps, f_inner, p0, c, face_kwargs, h, exact=None):
    """Run multigrid solve and report."""
    p, r = ps.solve_multigrid(f_inner, p0, c, **face_kwargs)
    r_norm = torch.sqrt((r**2).sum()).item()
    p_range = (p.min().item(), p.max().item())
    print(f"  [{label}]  residual={r_norm:.3e}  p=[{p_range[0]:+.4e}, {p_range[1]:+.4e}]", end="")
    if exact is not None:
        ndim = p.ndim
        inner = tuple(slice(1, -1) for _ in range(ndim))
        err = torch.abs(p[inner] - exact).max().item()
        print(f"  err_Linf={err:.3e}", end="")
    print()
    return p, r


# =====================================================================
# TEST 1: Constant coefficients — known solution
# =====================================================================
print("=" * 70)
print("TEST 1: Constant coefficients  lap(p) = f")
print("=" * 70)

for ndim, N in [(2, 32), (3, 32)]:
    L = 2 * math.pi
    h = L / N
    ng = N + 2  # with ghost cells
    coords = [torch.linspace(-h/2, L+h/2, ng, dtype=dtype, device=device)] * ndim
    grids = torch.meshgrid(*coords, indexing="ij")

    # exact: product of sines  =>  lap = -ndim * exact
    exact_full = torch.ones_like(grids[0])
    for g in grids:
        exact_full = exact_full * torch.sin(g)
    inner = tuple(slice(1, -1) for _ in range(ndim))
    f_inner = -float(ndim) * exact_full[inner]

    c = torch.ones([ng]*ndim, dtype=dtype, device=device)
    p0 = torch.zeros_like(c)

    # Build face arrays
    face_kw = {}
    labels = ["ch", "cv", "cw"]
    for d in range(ndim):
        idx_fwd = [slice(1, -1)] * ndim; idx_fwd[d] = slice(1, None)
        idx_bwd = [slice(1, -1)] * ndim; idx_bwd[d] = slice(None, -1)
        face_kw[labels[d]] = 0.5 * (c[tuple(idx_fwd)] + c[tuple(idx_bwd)])

    ps = PoissonSolver(dtype, device, h, tol=1e-8, max_vcycles=100,
                       nsmoothing=10, verbose=False)
    test_poisson(f"{ndim}D const-coeff N={N}", ps, f_inner, p0, c, face_kw, h,
                 exact=exact_full[inner])


# =====================================================================
# TEST 2: Variable coefficients — body-like pattern
# =====================================================================
print("\n" + "=" * 70)
print("TEST 2: Variable coefficients (body-like c=0 inside, c=dt/rho outside)")
print("=" * 70)

dt, rho = 0.001, 1.0
coeff = dt / rho

for ndim, N in [(2, 64), (3, 32)]:
    h = 4.0 / N  # domain [-1, 3] in x => length 4
    ng = N + 2
    x = torch.linspace(-1 - h/2, 3 + h/2, ng, dtype=dtype, device=device)
    y = torch.linspace(-1 - h/2, 1 + h/2, ng if ndim == 2 else N//2+2,
                        dtype=dtype, device=device)
    if ndim == 3:
        z = torch.linspace(-1 - h/2, 1 + h/2, N//2+2, dtype=dtype, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        sdf = torch.sqrt(X**2 + Y**2 + Z**2) - 0.2
    else:
        X, Y = torch.meshgrid(x, y, indexing="ij")
        sdf = torch.sqrt(X**2 + Y**2) - 0.2

    eps = 2 * h
    mu0 = torch.where(sdf > eps, 1.0, torch.where(sdf < -eps, 0.0,
          0.5 + sdf/(2*eps) + torch.sin(math.pi*sdf/eps)/(2*math.pi)))

    c_field = coeff * mu0
    c_ones = coeff * torch.ones_like(mu0)

    # Build face arrays from c_field (like the projection does)
    labels_list = ["ch", "cv", "cw"]
    face_kw = {}
    for d in range(ndim):
        idx_fwd = [slice(1, -1)] * ndim; idx_fwd[d] = slice(1, None)
        idx_bwd = [slice(1, -1)] * ndim; idx_bwd[d] = slice(None, -1)
        face_kw[labels_list[d]] = 0.5 * (c_field[tuple(idx_fwd)] + c_field[tuple(idx_bwd)])

    # Make an RHS that looks like the divergence from impulsive start:
    # u=1 outside body shifted to 0 abruptly  => div ~ d(mu0)/dx at body surface
    inner = tuple(slice(1, -1) for _ in range(ndim))
    if ndim == 2:
        u_field = mu0.clone()
        div = torch.zeros_like(mu0)
        div[1:-1, 1:-1] = (u_field[2:, 1:-1] - u_field[1:-1, 1:-1]) / h
        f_inner = div[inner]
    else:
        u_field = mu0.clone()
        div = torch.zeros_like(mu0)
        div[1:-1, 1:-1, 1:-1] = (u_field[2:, 1:-1, 1:-1] - u_field[1:-1, 1:-1, 1:-1]) / h
        f_inner = div[inner]

    p0 = torch.zeros_like(mu0)
    grid_str = "x".join(str(s) for s in mu0.shape)
    print(f"\n  {ndim}D grid {grid_str}, h={h:.4f}, coeff={coeff:.4e}")
    print(f"    mu0 range: [{mu0.min():.4f}, {mu0.max():.4f}]")
    print(f"    div range: [{div.min():.4e}, {div.max():.4e}]")
    print(f"    face coeff ch range: [{face_kw['ch'].min():.4e}, {face_kw['ch'].max():.4e}]")

    for tol, nv, ns in [(1e-4, 30, 5), (1e-4, 100, 10)]:
        ps = PoissonSolver(dtype, device, h, tol=tol, max_vcycles=nv,
                           nsmoothing=ns, verbose=False, w=0.7)
        p, r = test_poisson(
            f"{ndim}D body tol={tol:.0e} vcyc={nv} nsmooth={ns}",
            ps, f_inner, p0, c_ones, face_kw, h)


# =====================================================================
# TEST 3: Constant coeff only — make sure 3D multigrid reduces error
# =====================================================================
print("\n" + "=" * 70)
print("TEST 3: 3D constant-coeff with various grid sizes")
print("=" * 70)

for N in [8, 16, 32, 48]:
    h = 1.0 / N
    ng = N + 2
    c = torch.ones([ng]*3, dtype=dtype, device=device)
    p0 = torch.zeros_like(c)
    # Simple RHS:  f = 1 everywhere
    f_inner = torch.ones([N]*3, dtype=dtype, device=device)
    face_kw = {}
    for d, lab in enumerate(["ch", "cv", "cw"]):
        idx_fwd = [slice(1, -1)]*3; idx_fwd[d] = slice(1, None)
        idx_bwd = [slice(1, -1)]*3; idx_bwd[d] = slice(None, -1)
        face_kw[lab] = 0.5*(c[tuple(idx_fwd)] + c[tuple(idx_bwd)])

    ps = PoissonSolver(dtype, device, h, tol=1e-8, max_vcycles=200,
                       nsmoothing=10, verbose=False, w=0.7)
    test_poisson(f"3D N={N} const-coeff f=1", ps, f_inner, p0, c, face_kw, h)


# =====================================================================
# TEST 4: Only Jacobi (no V-cycle) — see if smoother works in isolation
# =====================================================================
print("\n" + "=" * 70)
print("TEST 4: Pure Jacobi (no coarse correction) — 3D variable coeff")
print("=" * 70)

N = 16; h = 4.0/N; ng = N + 2; ndim = 3
x = torch.linspace(-1-h/2, 3+h/2, ng, dtype=dtype, device=device)
y = torch.linspace(-1-h/2, 1+h/2, N//2+2, dtype=dtype, device=device)
z = torch.linspace(-1-h/2, 1+h/2, N//2+2, dtype=dtype, device=device)
X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
sdf = torch.sqrt(X**2 + Y**2 + Z**2) - 0.2
eps = 2*h
mu0 = torch.where(sdf > eps, 1.0, torch.where(sdf < -eps, 0.0,
      0.5 + sdf/(2*eps) + torch.sin(math.pi*sdf/eps)/(2*math.pi)))

c_field = coeff * mu0
face_kw = {}
for d, lab in enumerate(["ch", "cv", "cw"]):
    idx_fwd = [slice(1,-1)]*3; idx_fwd[d] = slice(1, None)
    idx_bwd = [slice(1,-1)]*3; idx_bwd[d] = slice(None, -1)
    face_kw[lab] = 0.5*(c_field[tuple(idx_fwd)] + c_field[tuple(idx_bwd)])

from lilytorch.src.poisson_mult import _inner
cfaces = PoissonSolver._extract_cfaces(list(face_kw.values()), 3)

# RHS = div of impulsive start
u_field = mu0.clone()
div = torch.zeros_like(mu0)
div[1:-1,1:-1,1:-1] = (u_field[2:,1:-1,1:-1] - u_field[1:-1,1:-1,1:-1]) / h
f_inner = div[tuple(slice(1,-1) for _ in range(3))]

p = torch.zeros_like(mu0)
ps_test = PoissonSolver(dtype, device, h, tol=1e-8, max_vcycles=1,
                        nsmoothing=50, verbose=False, w=0.7)

# Manual Jacobi iterations
f_scaled = h*h * f_inner
for it in range(5):
    p, r = ps_test.Jacobi(f_scaled, p, cfaces, 1)
    r_norm = ps_test.l2_norm(r).item()
    p_range = (p.min().item(), p.max().item())
    print(f"  Jacobi iter {(it+1)*50:4d}: residual={r_norm:.3e}  p=[{p_range[0]:+.4e}, {p_range[1]:+.4e}]")

print("\nDone.")
