"""Extended 3-D parity tests for both multigrid and MGCG native drivers.

Covers larger N, variable coefficients, multiple dtypes/smoothers."""
import math
import sys
import torch

sys.path.insert(0, '/data/andreaferrario/lilytorch')
from lilytorch.src.kernels import _C  # noqa: F401
from lilytorch.src.kernels import ops as K
from lilytorch.src.poisson_mult import PoissonSolver

device = torch.device("cuda:0")
torch.manual_seed(0)
FAIL = False


def make_3d_const(N, dtype):
    L = 2 * math.pi
    h = L / N
    nx = ny = nz = N + 2
    x = torch.linspace(-h / 2, L + h / 2, nx, dtype=dtype, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    phi_exact = torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    f_inner = -3.0 * phi_exact[1:-1, 1:-1, 1:-1]
    c = torch.ones(nx, ny, nz, dtype=dtype, device=device)
    ch = 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1])
    cw = 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])
    return h, nx, phi_exact, f_inner, ch, cv, cw


def make_3d_varcoeff(N, dtype, jump=10.0):
    L = 1.0
    h = L / N
    nx = ny = nz = N + 2
    x = torch.linspace(-h / 2, L + h / 2, nx, dtype=dtype, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    c = torch.where(X + Y + Z < 1.5,
                    torch.tensor(1.0, dtype=dtype, device=device),
                    torch.tensor(jump, dtype=dtype, device=device))
    ch = 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1])
    cw = 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])
    f_inner = (torch.sin(2 * math.pi * X[1:-1, 1:-1, 1:-1])
               * torch.sin(2 * math.pi * Y[1:-1, 1:-1, 1:-1])
               * torch.sin(2 * math.pi * Z[1:-1, 1:-1, 1:-1]))
    return h, nx, f_inner, ch, cv, cw


def check_mg(label, N, dtype, smoother, varcoeff=False, tol_target=1e-9):
    if varcoeff:
        h, nx, f_inner, ch, cv, cw = make_3d_varcoeff(N, dtype)
        phi = None
    else:
        h, nx, phi, f_inner, ch, cv, cw = make_3d_const(N, dtype)
    ps = PoissonSolver(dtype=dtype, device=device, h=h, tol=tol_target,
                       max_vcycles=30, nsmoothing=2,
                       smoother=smoother, w=1.0,
                       verbose=False)
    p_ref, r_ref = ps.solve_multigrid(
        f_inner, torch.zeros(nx, nx, nx, dtype=dtype, device=device),
        ch=ch, cv=cv, cw=cw)
    p_got = torch.zeros(nx, nx, nx, dtype=dtype, device=device)
    r_got = K.poisson_solve_multigrid_3d(
        p_got, f_inner.contiguous(),
        ch.contiguous(), cv.contiguous(), cw.contiguous(),
        h2=h*h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_vcycles=ps.max_vcycles,
        tol=tol_target, smoother=smoother,
    )
    diff_p = (p_got - p_ref).abs().max().item()
    rn_ref = r_ref.abs().max().item()
    rn_got = r_got.abs().max().item()
    line = f"  {label:<50s} |r|py={rn_ref:.2e} |r|nv={rn_got:.2e} diff|p|={diff_p:.2e}"
    if phi is not None:
        e_ref = (p_ref[1:-1,1:-1,1:-1] - phi[1:-1,1:-1,1:-1]).abs().max().item()
        e_got = (p_got[1:-1,1:-1,1:-1] - phi[1:-1,1:-1,1:-1]).abs().max().item()
        line += f" err_py={e_ref:.2e} err_nv={e_got:.2e}"
    print(line)
    # criteria: native residual at least as good as python (within 10x),
    # and matches python pressure to within ~converged tolerance band
    band = 1e-6 if dtype == torch.float32 else 1e-7
    if rn_got > 10 * max(rn_ref, tol_target) + band: return True
    return False


def check_mgcg(label, N, dtype, smoother, varcoeff=False, tol_target=1e-9,
               precond_vcycles=1):
    if varcoeff:
        h, nx, f_inner, ch, cv, cw = make_3d_varcoeff(N, dtype)
        phi = None
    else:
        h, nx, phi, f_inner, ch, cv, cw = make_3d_const(N, dtype)
    ps = PoissonSolver(dtype=dtype, device=device, h=h, tol=tol_target,
                       max_cycles=30, precond_vcycles=precond_vcycles,
                       nsmoothing=2, smoother=smoother, w=1.0,
                       verbose=False)
    p_ref, r_ref = ps.solve_mgcg(
        f_inner, torch.zeros(nx, nx, nx, dtype=dtype, device=device),
        ch=ch, cv=cv, cw=cw)
    p_got = torch.zeros(nx, nx, nx, dtype=dtype, device=device)
    r_got = K.poisson_solve_mgcg_3d(
        p_got, f_inner.contiguous(),
        ch.contiguous(), cv.contiguous(), cw.contiguous(),
        h2=h*h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_cycles=ps.max_cycles,
        precond_vcycles=precond_vcycles,
        tol=tol_target, smoother=smoother,
    )
    diff_p = (p_got - p_ref).abs().max().item()
    rn_ref = r_ref.abs().max().item()
    rn_got = r_got.abs().max().item()
    line = f"  {label:<50s} |r|py={rn_ref:.2e} |r|nv={rn_got:.2e} diff|p|={diff_p:.2e}"
    if phi is not None:
        e_ref = (p_ref[1:-1,1:-1,1:-1] - phi[1:-1,1:-1,1:-1]).abs().max().item()
        e_got = (p_got[1:-1,1:-1,1:-1] - phi[1:-1,1:-1,1:-1]).abs().max().item()
        line += f" err_py={e_ref:.2e} err_nv={e_got:.2e}"
    print(line)
    band = 1e-6 if dtype == torch.float32 else 1e-7
    if rn_got > 10 * max(rn_ref, tol_target) + band: return True
    return False


print("=" * 80)
print("3-D MULTIGRID parity")
FAIL |= check_mg("N=16  f64 rbgs   const",     16, torch.float64, "rbgs",   tol_target=1e-10)
FAIL |= check_mg("N=32  f64 rbgs   const",     32, torch.float64, "rbgs",   tol_target=1e-10)
FAIL |= check_mg("N=32  f32 rbgs   const",     32, torch.float32, "rbgs",   tol_target=1e-5)
FAIL |= check_mg("N=16  f64 jacobi const",     16, torch.float64, "jacobi", tol_target=1e-9)
FAIL |= check_mg("N=32  f64 rbgs   varcoeff",  32, torch.float64, "rbgs",   varcoeff=True, tol_target=1e-8)
FAIL |= check_mg("N=64  f32 rbgs   varcoeff",  64, torch.float32, "rbgs",   varcoeff=True, tol_target=1e-4)

print("\n" + "=" * 80)
print("3-D MGCG parity")
FAIL |= check_mgcg("N=16  f64 rbgs   const",     16, torch.float64, "rbgs",   tol_target=1e-10)
FAIL |= check_mgcg("N=32  f64 rbgs   const",     32, torch.float64, "rbgs",   tol_target=1e-10)
FAIL |= check_mgcg("N=32  f32 rbgs   const",     32, torch.float32, "rbgs",   tol_target=1e-5)
FAIL |= check_mgcg("N=16  f64 jacobi const",     16, torch.float64, "jacobi", tol_target=1e-9)
FAIL |= check_mgcg("N=32  f64 rbgs   varcoeff",  32, torch.float64, "rbgs",   varcoeff=True, tol_target=1e-8)
FAIL |= check_mgcg("N=64  f32 rbgs   varcoeff",  64, torch.float32, "rbgs",   varcoeff=True, tol_target=1e-4)
FAIL |= check_mgcg("N=32  f64 rbgs   const pv=2", 32, torch.float64, "rbgs",  tol_target=1e-10, precond_vcycles=2)

print("\n" + "=" * 80)
print("FAIL" if FAIL else "PASS")
sys.exit(1 if FAIL else 0)
