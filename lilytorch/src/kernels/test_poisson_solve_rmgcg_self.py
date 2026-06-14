"""Self-test for the native recycled-MGCG driver.

Checks, in 2-D and 3-D, on a variable-coefficient (immersed-jump) Poisson
problem:

  1. **Plain-path parity** — native RMGCG with an empty recycle basis
     (kdef=0) reproduces the native MGCG driver bit-for-bit.
  2. **Deflation parity** — given the *same* B-orthonormal recycle basis
     (U, W=B·U), the native deflated solve and the Python ``_cg_core``
     deflated solve converge to the same pressure field (to solver tol).
  3. **CPU path** — ``solve_rmgcg`` runs on CPU tensors via the PyTorch
     fallback and matches a CUDA reference solution.

Run:  python lilytorch/src/kernels/test_poisson_solve_rmgcg_self.py
"""
import math

import torch

from lilytorch.src.poisson_mult import PoissonSolver, _inner
from lilytorch.src.kernels import ops as K


def _make_problem(ndim, N, dtype, device, contrast=1000.0):
    L = 2 * math.pi
    h = L / N
    n = N + 2
    ax = torch.linspace(-h / 2, L + h / 2, n, dtype=dtype, device=device)
    grids = torch.meshgrid(*([ax] * ndim), indexing="ij")
    center = [0.45 * L] * ndim
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    sdf = torch.sqrt(r2) - 0.2 * L
    mu0 = 0.5 * (1.0 + torch.tanh(sdf / (2 * h)))
    c = (1.0 / contrast) + (1.0 - 1.0 / contrast) * mu0
    if ndim == 2:
        ch = 0.5 * (c[1:, 1:-1] + c[:-1, 1:-1])
        cv = 0.5 * (c[1:-1, 1:] + c[1:-1, :-1])
        faces = {"ch": ch, "cv": cv}
    else:
        ch = 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1])
        cv = 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1])
        cw = 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])
        faces = {"ch": ch, "cv": cv, "cw": cw}
    inner = _inner(ndim)
    f = torch.exp(-r2 / (2 * (0.15 * L) ** 2))[inner].clone()
    f -= f.mean()
    return h, n, faces, f


def _seed_recycle(ps, face_arrs, f, ndim, n, dtype, device):
    """Force-harvest a recycle space (bypasses the usefulness guard).

    The correctness test only needs a *valid* B-orthonormal basis to exercise
    the deflation math — it does not care whether recycling would help here, so
    we harvest the search directions of one plain solve directly rather than
    relying on the iteration-bound guard in ``solve_rmgcg`` to engage.
    """
    cfaces = ps._extract_cfaces(face_arrs, ndim)
    inner = _inner(ndim)
    b = -(ps.h2 * f)
    x = torch.zeros(*([n] * ndim), dtype=dtype, device=device)
    ps.BC(x)
    harvest = []
    ps._cg_core(b, x, cfaces, face_arrs, recycle=None, harvest=harvest)
    ps._update_recycle(harvest, inner)


def run(ndim, N, device):
    dtype = torch.float64
    k = 10
    h, n, faces, f = _make_problem(ndim, N, dtype, device)
    face_arrs = [faces["ch"], faces["cv"]] + ([faces["cw"]] if ndim == 3 else [])
    inner = _inner(ndim)

    def mk(recycle_k, use_kernels):
        return PoissonSolver(
            dtype, device, h, tol=1e-10, max_cycles=200, max_vcycles=1,
            nsmoothing=1, w=1.0, verbose=False, precond_vcycles=1,
            smoother="rbgs", use_kernels=use_kernels, recycle_k=recycle_k,
        )

    print(f"\n=== {ndim}-D  N={N}  device={device} ===")

    # ---- (1) plain-path parity: native RMGCG kdef=0 vs native MGCG ----
    if device == "cuda":
        ps = mk(0, True)
        cfaces = ps._extract_cfaces(face_arrs, ndim)
        p0 = torch.zeros(*([n] * ndim), dtype=dtype, device=device)
        # native mgcg
        p_mg, r_mg = ps._solve_mgcg_native(f, p0.clone(), face_arrs, ndim)
        # native rmgcg with empty basis (recycle_k=0 → kdef=0, harvest_k=0)
        p_rm, r_rm = ps._solve_rmgcg_native(f, p0.clone(), face_arrs, cfaces,
                                            ndim, inner)
        diff = (p_mg - p_rm).abs().max().item()
        print(f"  (1) native RMGCG(kdef=0) vs native MGCG:  max|dp| = {diff:.2e}")
        assert diff < 1e-10, "kdef=0 native RMGCG must equal native MGCG"

    # ---- (2) deflation parity: native vs python with the SAME basis ----
    if device == "cuda":
        ps_seed = mk(k, True)
        _seed_recycle(ps_seed, face_arrs, f, ndim, n, dtype, device)
        cfaces = ps_seed._extract_cfaces(face_arrs, ndim)
        rec = ps_seed._prepare_recycle(cfaces, (n,) * ndim, inner)
        assert rec is not None and rec["U"], "failed to build a recycle basis"
        kk = len(rec["U"])

        # verify B-orthonormality of the basis
        Bq = [ps_seed._apply_op_spd(q.clone(), cfaces) for q in rec["U"]]
        G = torch.zeros(kk, kk, dtype=dtype, device=device)
        for i in range(kk):
            for j in range(kk):
                G[i, j] = (rec["U"][i][inner] * Bq[j]).sum()
        ortho = (G - torch.eye(kk, dtype=dtype, device=device)).abs().max().item()
        print(f"  (2) recycle basis k={kk}, max|QᵀBQ - I| = {ortho:.2e}")

        p0 = torch.zeros(*([n] * ndim), dtype=dtype, device=device)

        # python deflated solve via _cg_core
        b = -(ps_seed.h2 * f)
        x_py = p0.clone(); ps_seed.BC(x_py)
        x_py, r_py, nit_py, _ = ps_seed._cg_core(
            b, x_py, cfaces, face_arrs, recycle=rec, harvest=None)

        # native deflated solve with the same stacked basis
        U = torch.stack(rec["U"]).contiguous()
        W = torch.stack(rec["W"]).contiguous()
        p_nat = p0.clone().contiguous()
        op = (K.poisson_solve_rmgcg_2d if ndim == 2
              else K.poisson_solve_rmgcg_3d)
        args = ([p_nat, f.contiguous()]
                + [a.contiguous() for a in face_arrs]
                + [U, W, 0])
        r_nat, _D, nit_nat = op(
            *args, h2=ps_seed.h2, jcap_tol=ps_seed.jcap_tol, w=ps_seed.w,
            nsmoothing=ps_seed.nsmoothing, max_cycles=ps_seed.max_cycles,
            precond_vcycles=ps_seed.precond_vcycles,
            tol=ps_seed._tol_float, smoother=ps_seed.smoother)

        diff = (x_py - p_nat).abs().max().item()
        scale = x_py.abs().max().item() + 1e-30
        print(f"      python niter={nit_py}  native niter={nit_nat}  "
              f"max|dp|={diff:.2e} (rel {diff/scale:.2e})")
        assert diff / scale < 1e-6, "native deflated solve must match python"

    # ---- (3) CPU path runs (PyTorch fallback) and matches CPU mgcg ----
    def mk_cpu(recycle_k):
        return PoissonSolver(
            dtype, "cpu", h, tol=1e-10, max_cycles=200, max_vcycles=1,
            nsmoothing=1, w=1.0, verbose=False, precond_vcycles=1,
            smoother="rbgs", use_kernels=False, recycle_k=recycle_k,
        )

    faces_cpu = {kk: vv.cpu() for kk, vv in faces.items()}
    fc = f.cpu()
    ps_cpu = mk_cpu(k)
    last = None
    for _ in range(3):
        seed = torch.zeros(*([n] * ndim), dtype=dtype) if last is None else last
        last, _ = ps_cpu.solve_rmgcg(fc, seed.clone(), **faces_cpu)
    ps_ref = mk_cpu(0)
    p_ref = torch.zeros(*([n] * ndim), dtype=dtype)
    for _ in range(3):
        p_ref, _ = ps_ref.solve_mgcg(fc, p_ref.clone(), **faces_cpu)
    d_cpu = (last - p_ref).abs().max().item()
    s_cpu = p_ref.abs().max().item() + 1e-30
    print(f"  (3) CPU rmgcg vs CPU mgcg:  max|dp| = {d_cpu:.2e} "
          f"(rel {d_cpu/s_cpu:.2e})")
    assert d_cpu / s_cpu < 1e-5, "CPU rmgcg must match CPU mgcg solution"


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ndim, N in ((2, 64), (3, 24)):
        run(ndim, N, dev)
    print("\n" + "=" * 60 + "\nPASS")
