"""
Trace through a single V-cycle step by step to find the 3D bug.
"""
import torch, math, sys
sys.path.insert(0, ".")
from lilytorch.src.poisson_mult import PoissonSolver, _inner, _sl

dtype = torch.float64
device = "cpu"

def trace_vcycle(ndim, N):
    print(f"\n{'='*70}")
    print(f"  TRACING V-CYCLE: {ndim}D, N={N}")
    print(f"{'='*70}")

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
    p = torch.zeros([ng]*ndim, dtype=dtype, device=device)

    labels = ["ch", "cv", "cw"]
    face_arrs = []
    for d in range(ndim):
        idx_fwd = [slice(1,-1)]*ndim; idx_fwd[d] = slice(1, None)
        idx_bwd = [slice(1,-1)]*ndim; idx_bwd[d] = slice(None, -1)
        face_arrs.append(0.5*(c[tuple(idx_fwd)] + c[tuple(idx_bwd)]))

    ps = PoissonSolver(dtype, device, h, tol=1e-8, max_vcycles=1,
                       nsmoothing=5, verbose=False, w=0.7)

    f_scaled = h*h * f_inner
    print(f"  h={h:.4f}, h²={h*h:.6f}")
    print(f"  f_scaled range: [{f_scaled.min():.6e}, {f_scaled.max():.6e}]")
    print(f"  Face arr shapes: {[fa.shape for fa in face_arrs]}")

    # ---- Manual V-cycle ----
    cfaces = ps._extract_cfaces(face_arrs, ndim)
    print(f"  cfaces[0] cp shape: {cfaces[0][0].shape}, cm shape: {cfaces[0][1].shape}")

    # Pre-smooth
    p1, r1 = ps.Jacobi(f_scaled, p, cfaces, 1)
    print(f"\n  PRE-SMOOTH:")
    print(f"    p range: [{p1[inner].min():.6e}, {p1[inner].max():.6e}]")
    print(f"    residual L2: {ps.l2_norm(r1):.6e}")
    print(f"    residual range: [{r1.min():.6e}, {r1.max():.6e}]")

    # Check if recursion happens
    shape = f_scaled.shape
    print(f"\n  Fine grid shape: {shape}, all > 8? {all(n > 8 for n in shape)}")

    if not all(n > 8 for n in shape):
        print("  No recursion — too small. Done.")
        return

    # Restriction of face arrays
    face_arrs_coarse = []
    for d, cf in enumerate(face_arrs):
        cf_c = cf
        for d2 in range(ndim):
            if d2 == d:
                cf_c = cf_c[_sl(ndim, d2, slice(None, None, 2))]
            else:
                even = cf_c[_sl(ndim, d2, slice(None, -1, 2))]
                odd  = cf_c[_sl(ndim, d2, slice(1, None, 2))]
                cf_c = 0.5 * (even + odd)
        face_arrs_coarse.append(cf_c)
    print(f"\n  COARSE face arr shapes: {[fa.shape for fa in face_arrs_coarse]}")
    print(f"    values: [{face_arrs_coarse[0].min():.4f}, {face_arrs_coarse[0].max():.4f}]")

    # Restriction of residual
    r_coarse = r1.clone()
    for d in range(ndim):
        even = r_coarse[_sl(ndim, d, slice(0, None, 2))]
        odd  = r_coarse[_sl(ndim, d, slice(1, None, 2))]
        m = min(even.shape[d], odd.shape[d])
        r_coarse = (even[_sl(ndim, d, slice(m))]
                    + odd[_sl(ndim, d, slice(m))])
    print(f"\n  RESTRICTED RESIDUAL:")
    print(f"    shape: {r_coarse.shape}")
    print(f"    range: [{r_coarse.min():.6e}, {r_coarse.max():.6e}]")
    print(f"    L2: {ps.l2_norm(r_coarse):.6e}")

    # Coarse-grid solve (Jacobi only since coarse grid < 8)
    coarse_shape = tuple(s + 2 for s in r_coarse.shape)
    print(f"    coarse_shape (with ghost): {coarse_shape}")
    err_coarse_p = torch.zeros(coarse_shape, dtype=dtype, device=device)

    cfaces_c = ps._extract_cfaces(face_arrs_coarse, ndim)
    err_coarse_p, r_c = ps.Jacobi(r_coarse, err_coarse_p, cfaces_c, 1)

    inner_c = _inner(ndim)
    ec = err_coarse_p[inner_c]
    print(f"\n  COARSE SOLVE (Jacobi only):")
    print(f"    err_coarse range: [{ec.min():.6e}, {ec.max():.6e}]")
    print(f"    err_coarse L2: {ps.l2_norm(ec):.6e}")
    print(f"    coarse residual L2: {ps.l2_norm(r_c):.6e}")

    # Prolongation
    err = torch.zeros(shape, dtype=dtype, device=device)
    for corner in range(2**ndim):
        slc = []
        for d in range(ndim):
            if (corner >> d) & 1:
                slc.append(slice(1, None, 2))
            else:
                slc.append(slice(None, None, 2))
        target = err[tuple(slc)]
        min_shape = tuple(min(t, s) for t, s in zip(target.shape, ec.shape))
        trim = tuple(slice(0, m) for m in min_shape)
        err[tuple(slc)][trim] = ec[trim]

    print(f"\n  PROLONGATED ERROR:")
    print(f"    err range: [{err.min():.6e}, {err.max():.6e}]")
    print(f"    err L2: {ps.l2_norm(err):.6e}")
    print(f"    err nonzero: {(err != 0).sum()} / {err.numel()}")

    # Correction
    p_corrected = p1.clone()
    p_corrected[inner] += err
    print(f"\n  AFTER CORRECTION:")
    print(f"    p range: [{p_corrected[inner].min():.6e}, {p_corrected[inner].max():.6e}]")

    # Check residual after correction (before post-smooth)
    s_check = ps.compute_sum(cfaces, p_corrected)
    J_check = ps.compute_J(cfaces)
    Au_check = (s_check - J_check * p_corrected[inner])
    r_check = f_scaled - Au_check
    print(f"    residual L2 (before post-smooth): {ps.l2_norm(r_check):.6e}")

    # Post-smooth
    cfaces2 = ps._extract_cfaces(face_arrs, ndim)
    p_final, r_final = ps.Jacobi(f_scaled, p_corrected, cfaces2, 1)
    print(f"\n  POST-SMOOTH:")
    print(f"    p range: [{p_final[inner].min():.6e}, {p_final[inner].max():.6e}]")
    print(f"    residual L2: {ps.l2_norm(r_final):.6e}")

    # Compare with just Jacobi (no coarse correction)
    p_jac = torch.zeros([ng]*ndim, dtype=dtype, device=device)
    p_jac, r_jac = ps.Jacobi(f_scaled, p_jac, cfaces, 1) # pre-smooth
    p_jac, r_jac = ps.Jacobi(f_scaled, p_jac, cfaces, 1) # "post-smooth"
    print(f"\n  COMPARISON — two rounds of Jacobi (no coarse correction):")
    print(f"    p range: [{p_jac[inner].min():.6e}, {p_jac[inner].max():.6e}]")
    print(f"    residual L2: {ps.l2_norm(r_jac):.6e}")

# Test both
trace_vcycle(2, 16)
trace_vcycle(3, 16)

print("\nDone.")
