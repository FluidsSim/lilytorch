"""
Instrument the V-cycle restriction/prolongation to find the 3D bug.
"""
import torch, sys
sys.path.insert(0, ".")
from lilytorch.src.poisson_mult import PoissonSolver, _inner, _sl

dtype = torch.float64
device = "cpu"  # easier to debug

def test_restrict_prolongate(ndim, N):
    """Create a known field, restrict it, prolongate it, check round-trip."""
    print(f"\n{'='*60}")
    print(f"  Restriction → Prolongation round-trip ({ndim}D, N={N})")
    print(f"{'='*60}")

    # Fine-grid residual: simple linear function so restriction is exact
    shape = (N,) * ndim
    r = torch.zeros(shape, dtype=dtype, device=device)
    inner = _inner(ndim)

    # Fill with 1.0 everywhere
    r.fill_(1.0)
    print(f"  r shape: {r.shape}, sum: {r.sum():.1f}")

    # ---- Restriction (from V-cycle code) ----
    r_coarse = r.clone()
    for d in range(ndim):
        even = r_coarse[_sl(ndim, d, slice(0, None, 2))]
        odd  = r_coarse[_sl(ndim, d, slice(1, None, 2))]
        m = min(even.shape[d], odd.shape[d])
        r_coarse = (even[_sl(ndim, d, slice(m))]
                    + odd[_sl(ndim, d, slice(m))])

    print(f"  r_coarse shape: {r_coarse.shape}, values: [{r_coarse.min():.1f}, {r_coarse.max():.1f}]")
    print(f"    Expected: shape {tuple(N//2 for _ in range(ndim))}, value={2**ndim:.0f} everywhere")

    # The coarse-grid error is solved; assume it's a known field
    coarse_shape = tuple(s + 2 for s in r_coarse.shape)
    err_coarse = torch.ones(coarse_shape, dtype=dtype, device=device) * 3.14
    inner_c = _inner(ndim)
    ec = err_coarse[inner_c]
    print(f"  err_coarse interior shape: {ec.shape}, value: {ec.flatten()[0]:.2f}")

    # ---- Prolongation (from V-cycle code) ----
    err = torch.zeros_like(r)
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

    print(f"  err (prolongated) shape: {err.shape}")
    print(f"    values: [{err.min():.2f}, {err.max():.2f}]")
    print(f"    nonzero count: {(err != 0).sum()} out of {err.numel()}")
    print(f"    zero count:    {(err == 0).sum()} out of {err.numel()}")

    # Show a slice in 3D to see pattern
    if ndim == 3 and N <= 8:
        print(f"  err[:,:,0] =\n{err[:,:,0]}")
        print(f"  err[:,:,1] =\n{err[:,:,1]}")

    # Check: every fine cell should get a value from prolongation
    if (err == 0).any():
        # Find where zeros are
        zero_idx = (err == 0).nonzero()
        print(f"  *** BUG: {len(zero_idx)} zero cells found! First few:")
        for idx in zero_idx[:10]:
            print(f"      index {tuple(idx.tolist())}")

    return err


# Test 2D
test_restrict_prolongate(2, 8)
test_restrict_prolongate(2, 16)

# Test 3D
test_restrict_prolongate(3, 8)
test_restrict_prolongate(3, 16)

# ---- Additional: test that Jacobi+restriction+prolongation+postsmooth
#      produces a meaningful correction ----
print(f"\n{'='*60}")
print(f"  Manual V-cycle trace (3D, N=16, const coeff)")
print(f"{'='*60}")

N = 16; ndim = 3; h = 1.0/N; ng = N+2
c = torch.ones([ng]*3, dtype=dtype, device=device)
p = torch.zeros([ng]*3, dtype=dtype, device=device)
f_inner = torch.ones([N]*3, dtype=dtype, device=device)

face_kw = {}
for d, lab in enumerate(["ch", "cv", "cw"]):
    idx_fwd = [slice(1,-1)]*3; idx_fwd[d] = slice(1, None)
    idx_bwd = [slice(1,-1)]*3; idx_bwd[d] = slice(None, -1)
    face_kw[lab] = 0.5*(c[tuple(idx_fwd)] + c[tuple(idx_bwd)])

ps = PoissonSolver(dtype, device, h, tol=1e-8, max_vcycles=1,
                   nsmoothing=5, verbose=False, w=0.7)

f_scaled = h*h * f_inner
print(f"  f_scaled range: [{f_scaled.min():.6f}, {f_scaled.max():.6f}]")

# One V-cycle
p_before = p.clone()
p, r = ps.vcycle(f_scaled, p, c, 1, **face_kw)
print(f"  After 1 V-cycle: p=[{p.min():.6e}, {p.max():.6e}], residual L2={ps.l2_norm(r):.6e}")
print(f"  p changed from zero? {(p != p_before).any()}")
print(f"  p interior nonzero count: {(p[1:-1,1:-1,1:-1] != 0).sum()}")

# Compare with pure Jacobi
p2 = torch.zeros([ng]*3, dtype=dtype, device=device)
cfaces = ps._extract_cfaces(list(face_kw.values()), 3)
p2, r2 = ps.Jacobi(f_scaled, p2, cfaces, 1)
print(f"  After Jacobi only: p=[{p2.min():.6e}, {p2.max():.6e}], residual L2={ps.l2_norm(r2):.6e}")

print("\nDone.")
