"""Parity test for the native multigrid Poisson driver.

Compares ``lilytorch_kernels::poisson_solve_multigrid_{2,3}d`` against
``PoissonSolver.solve_multigrid`` (Python orchestration) on the same set
of test problems as poisson_mult.py's __main__:

  - 2D N=16 constant coeff
  - 2D N=64 variable-coeff jump
  - 3D N=16 constant coeff

Reports:
  - final residual L∞ for both backends
  - error vs analytic solution for both backends
  - peak |p_native - p_python| (cycle-count + reduction nondeterminism
    means we don't expect bit-exact agreement, but the converged
    pressure fields should match to ~tol).
"""
import math
import sys
import time
import torch

sys.path.insert(0, '/data/andreaferrario/lilytorch')
from lilytorch.src.kernels import _C  # noqa: F401
from lilytorch.src.kernels import ops as K
from lilytorch.src.poisson_mult import PoissonSolver

device = torch.device("cuda:0")
torch.manual_seed(0)


def run_2d_const(N, dtype, smoother):
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

    ps = PoissonSolver(dtype=dtype, device=device, h=h, tol=1e-10, max_vcycles=20, nsmoothing=2,
                       smoother=smoother, w=1.0,
                       verbose=False)
    # ---- Python solve ----
    p0 = torch.zeros(nx, ny, dtype=dtype, device=device)
    p_ref, r_ref = ps.solve_multigrid(f_inner, p0.clone(), ch=ch, cv=cv)

    # ---- Native solve ----
    p_got = torch.zeros(nx, ny, dtype=dtype, device=device)
    r_got = K.poisson_solve_multigrid_2d(
        p_got, f_inner.contiguous(), ch.contiguous(), cv.contiguous(),
        h2=h * h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_vcycles=ps.max_vcycles,
        tol=1e-10, smoother=smoother,
    )

    # Compare interior pressure (subtract analytic to gauge accuracy)
    err_ref = (p_ref[1:-1, 1:-1] - phi_exact[1:-1, 1:-1]).abs().max().item()
    err_got = (p_got[1:-1, 1:-1] - phi_exact[1:-1, 1:-1]).abs().max().item()
    # p_native vs p_python — both go through mean-subtraction so should match closely
    diff_p = (p_got - p_ref).abs().max().item()
    rnorm_ref = r_ref.abs().max().item()
    rnorm_got = r_got.abs().max().item()
    return err_ref, err_got, diff_p, rnorm_ref, rnorm_got


def run_2d_varcoeff(N, dtype, smoother):
    L = 1.0
    h = L / N
    nx = ny = N + 2
    x = torch.linspace(-h / 2, L + h / 2, nx, dtype=dtype, device=device)
    y = torch.linspace(-h / 2, L + h / 2, ny, dtype=dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    # Coefficient field with a jump
    c = torch.where(X + Y < 1.0,
                    torch.tensor(1.0, dtype=dtype, device=device),
                    torch.tensor(10.0, dtype=dtype, device=device))
    ch = 0.5 * (c[1:, 1:-1] + c[:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:] + c[1:-1, :-1])
    f_inner = torch.sin(2 * math.pi * X[1:-1, 1:-1]) \
            * torch.sin(2 * math.pi * Y[1:-1, 1:-1])

    ps = PoissonSolver(dtype=dtype, device=device, h=h, tol=1e-8, max_vcycles=20, nsmoothing=3,
                       smoother=smoother, w=1.0,
                       verbose=False)
    p_ref, r_ref = ps.solve_multigrid(f_inner, torch.zeros(nx, ny, dtype=dtype, device=device),
                                       ch=ch, cv=cv)
    p_got = torch.zeros(nx, ny, dtype=dtype, device=device)
    r_got = K.poisson_solve_multigrid_2d(
        p_got, f_inner.contiguous(), ch.contiguous(), cv.contiguous(),
        h2=h * h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_vcycles=ps.max_vcycles,
        tol=1e-8, smoother=smoother,
    )
    diff_p = (p_got - p_ref).abs().max().item()
    rnorm_ref = r_ref.abs().max().item()
    rnorm_got = r_got.abs().max().item()
    return diff_p, rnorm_ref, rnorm_got


def run_3d_const(N, dtype, smoother):
    L = 2 * math.pi
    h = L / N
    nx = ny = nz = N + 2
    x = torch.linspace(-h / 2, L + h / 2, nx, dtype=dtype, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    phi_exact = torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    f_inner = -3.0 * phi_exact[1:-1, 1:-1, 1:-1]
    c  = torch.ones(nx, ny, nz, dtype=dtype, device=device)
    ch = 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1])
    cw = 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])

    ps = PoissonSolver(dtype=dtype, device=device, h=h, tol=1e-10, max_vcycles=20, nsmoothing=2,
                       smoother=smoother, w=1.0,
                       verbose=False)
    p_ref, r_ref = ps.solve_multigrid(f_inner,
                                       torch.zeros(nx, ny, nz, dtype=dtype, device=device),
                                       ch=ch, cv=cv, cw=cw)
    p_got = torch.zeros(nx, ny, nz, dtype=dtype, device=device)
    r_got = K.poisson_solve_multigrid_3d(
        p_got, f_inner.contiguous(),
        ch.contiguous(), cv.contiguous(), cw.contiguous(),
        h2=h * h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_vcycles=ps.max_vcycles,
        tol=1e-10, smoother=smoother,
    )
    err_ref = (p_ref[1:-1, 1:-1, 1:-1] - phi_exact[1:-1, 1:-1, 1:-1]).abs().max().item()
    err_got = (p_got[1:-1, 1:-1, 1:-1] - phi_exact[1:-1, 1:-1, 1:-1]).abs().max().item()
    diff_p = (p_got - p_ref).abs().max().item()
    rnorm_ref = r_ref.abs().max().item()
    rnorm_got = r_got.abs().max().item()
    return err_ref, err_got, diff_p, rnorm_ref, rnorm_got


FAIL = False

print("=" * 80)
print("2-D constant-coeff Poisson, N=16, dtype=float64, smoother=rbgs")
err_ref, err_got, diff_p, r_ref, r_got = run_2d_const(16, torch.float64, "rbgs")
print(f"  Python:   |p-phi|={err_ref:.3e}   |r|_inf={r_ref:.3e}")
print(f"  Native:   |p-phi|={err_got:.3e}   |r|_inf={r_got:.3e}")
print(f"  |p_native - p_python|_inf = {diff_p:.3e}")
if err_got > 2 * err_ref + 1e-12 or r_got > 1e-8: FAIL = True

print("\n2-D constant-coeff Poisson, N=16, dtype=float32, smoother=rbgs")
err_ref, err_got, diff_p, r_ref, r_got = run_2d_const(16, torch.float32, "rbgs")
print(f"  Python:   |p-phi|={err_ref:.3e}   |r|_inf={r_ref:.3e}")
print(f"  Native:   |p-phi|={err_got:.3e}   |r|_inf={r_got:.3e}")
print(f"  |p_native - p_python|_inf = {diff_p:.3e}")
if err_got > 2 * err_ref + 1e-5 or r_got > 1e-4: FAIL = True

print("\n2-D constant-coeff Poisson, N=16, dtype=float64, smoother=jacobi")
err_ref, err_got, diff_p, r_ref, r_got = run_2d_const(16, torch.float64, "jacobi")
print(f"  Python:   |p-phi|={err_ref:.3e}   |r|_inf={r_ref:.3e}")
print(f"  Native:   |p-phi|={err_got:.3e}   |r|_inf={r_got:.3e}")
print(f"  |p_native - p_python|_inf = {diff_p:.3e}")

print("\n2-D variable-coeff (jump) Poisson, N=64, dtype=float64, smoother=rbgs")
diff_p, r_ref, r_got = run_2d_varcoeff(64, torch.float64, "rbgs")
print(f"  Python residual:   |r|_inf={r_ref:.3e}")
print(f"  Native residual:   |r|_inf={r_got:.3e}")
print(f"  |p_native - p_python|_inf = {diff_p:.3e}")
if r_got > 1e-6: FAIL = True

print("\n3-D constant-coeff Poisson, N=16, dtype=float64, smoother=rbgs")
err_ref, err_got, diff_p, r_ref, r_got = run_3d_const(16, torch.float64, "rbgs")
print(f"  Python:   |p-phi|={err_ref:.3e}   |r|_inf={r_ref:.3e}")
print(f"  Native:   |p-phi|={err_got:.3e}   |r|_inf={r_got:.3e}")
print(f"  |p_native - p_python|_inf = {diff_p:.3e}")
if err_got > 2 * err_ref + 1e-12 or r_got > 1e-8: FAIL = True

print("\n3-D constant-coeff Poisson, N=16, dtype=float32, smoother=jacobi")
err_ref, err_got, diff_p, r_ref, r_got = run_3d_const(16, torch.float32, "jacobi")
print(f"  Python:   |p-phi|={err_ref:.3e}   |r|_inf={r_ref:.3e}")
print(f"  Native:   |p-phi|={err_got:.3e}   |r|_inf={r_got:.3e}")
print(f"  |p_native - p_python|_inf = {diff_p:.3e}")

# ---- Quick perf comparison ------------------------------------------
print("\n" + "=" * 80)
print("Perf benchmark: 2-D N=128, float32, rbgs, 50 calls")
N = 128
L = 1.0
h = L / N
nx = ny = N + 2
dtype = torch.float32
x = torch.linspace(-h / 2, L + h / 2, nx, dtype=dtype, device=device)
X, Y = torch.meshgrid(x, x, indexing="ij")
c  = torch.ones(nx, ny, dtype=dtype, device=device)
ch = (0.5 * (c[1:, 1:-1] + c[:-1, 1:-1])).contiguous()
cv = (0.5 * (c[1:-1, 1:] + c[1:-1, :-1])).contiguous()
f_inner = (torch.sin(2 * math.pi * X[1:-1, 1:-1])
           * torch.sin(2 * math.pi * Y[1:-1, 1:-1])).contiguous()

ps = PoissonSolver(dtype=dtype, device=device, h=h, tol=1e-6, max_vcycles=8, nsmoothing=2,
                   smoother="rbgs", w=1.0,
                   verbose=False)
# warm-up
for _ in range(5):
    p0 = torch.zeros(nx, ny, dtype=dtype, device=device)
    ps.solve_multigrid(f_inner, p0, ch=ch, cv=cv)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    p0 = torch.zeros(nx, ny, dtype=dtype, device=device)
    ps.solve_multigrid(f_inner, p0, ch=ch, cv=cv)
torch.cuda.synchronize()
t_py = (time.time() - t0) / 50 * 1e3

for _ in range(5):
    p0 = torch.zeros(nx, ny, dtype=dtype, device=device)
    K.poisson_solve_multigrid_2d(p0, f_inner, ch, cv,
        h2=h * h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_vcycles=ps.max_vcycles,
        tol=1e-6, smoother="rbgs")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    p0 = torch.zeros(nx, ny, dtype=dtype, device=device)
    K.poisson_solve_multigrid_2d(p0, f_inner, ch, cv,
        h2=h * h, jcap_tol=ps.jcap_tol, w=ps.w,
        nsmoothing=ps.nsmoothing, max_vcycles=ps.max_vcycles,
        tol=1e-6, smoother="rbgs")
torch.cuda.synchronize()
t_nv = (time.time() - t0) / 50 * 1e3
print(f"  Python   solve_multigrid:  {t_py:.3f} ms/solve")
print(f"  Native   solve_multigrid:  {t_nv:.3f} ms/solve")
print(f"  Speedup: {t_py/t_nv:.2f}x")

print("\n" + "=" * 80)
print("FAIL" if FAIL else "PASS")
sys.exit(1 if FAIL else 0)
