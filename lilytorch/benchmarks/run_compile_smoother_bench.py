#!/usr/bin/env python3
"""
Benchmark torch.compile() on the Poisson multigrid smoothers.

Tests Jacobi and RBGS, each in eager vs torch.compile mode, on
realistic 3-D grids with variable coefficients.

Usage
-----
    source /data/andreaferrario/venv_ns_312/bin/activate
    python run_compile_smoother_bench.py
"""

import time
import torch
import numpy as np

torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

# =====================================================================
# Extract smoother kernels as standalone functions for torch.compile
# =====================================================================

def _inner_3d():
    return (slice(1, -1), slice(1, -1), slice(1, -1))

def _bc_3d(q):
    """Neumann BCs on all 6 faces."""
    q[0, :, :]  = q[1, :, :]
    q[-1, :, :] = q[-2, :, :]
    q[:, 0, :]  = q[:, 1, :]
    q[:, -1, :] = q[:, -2, :]
    q[:, :, 0]  = q[:, :, 1]
    q[:, :, -1] = q[:, :, -2]

def _compute_sum_3d(cp0, cm0, cp1, cm1, cp2, cm2, p):
    """Sum of c_plus*p_fwd + c_minus*p_bwd for 3 dims (interior)."""
    inn = _inner_3d()
    s = (cp0 * p[2:, 1:-1, 1:-1] + cm0 * p[:-2, 1:-1, 1:-1]
       + cp1 * p[1:-1, 2:, 1:-1] + cm1 * p[1:-1, :-2, 1:-1]
       + cp2 * p[1:-1, 1:-1, 2:] + cm2 * p[1:-1, 1:-1, :-2])
    return s

def _compute_J_3d(cp0, cm0, cp1, cm1, cp2, cm2):
    return cp0 + cm0 + cp1 + cm1 + cp2 + cm2


# ── Jacobi kernel ───────────────────────────────────────────────────

def jacobi_smooth(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                  h2, w, nsmoothing, jcap_tol):
    """Jacobi smoothing loop for 3-D, returns (p, r)."""
    _bc_3d(p)
    J = _compute_J_3d(cp0, cm0, cp1, cm1, cp2, cm2)
    active = torch.abs(J) >= jcap_tol
    Jinv = torch.where(active, 1.0 / J, torch.zeros_like(J))

    for _ in range(nsmoothing):
        s = _compute_sum_3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
        p[1:-1, 1:-1, 1:-1] = w * (-f * h2 + s) * Jinv + (1 - w) * p[1:-1, 1:-1, 1:-1]
        _bc_3d(p)

    s = _compute_sum_3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
    J = _compute_J_3d(cp0, cm0, cp1, cm1, cp2, cm2)
    Au = (s - J * p[1:-1, 1:-1, 1:-1]) / h2
    r = torch.where(active, f - Au, torch.zeros_like(f))
    return p, r


# ── RBGS kernel ─────────────────────────────────────────────────────

def rbgs_smooth(f, p, cp0, cm0, cp1, cm1, cp2, cm2,
                h2, nsmoothing, jcap_tol, red, black):
    """Red-Black Gauss-Seidel smoothing for 3-D, returns (p, r)."""
    _bc_3d(p)
    J = _compute_J_3d(cp0, cm0, cp1, cm1, cp2, cm2)
    active = torch.abs(J) >= jcap_tol
    Jinv = torch.where(active, 1.0 / J, torch.zeros_like(J))

    for _ in range(nsmoothing):
        # red sweep
        s = _compute_sum_3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
        p_new = (-f * h2 + s) * Jinv
        p[1:-1, 1:-1, 1:-1] = torch.where(red, p_new, p[1:-1, 1:-1, 1:-1])
        _bc_3d(p)
        # black sweep
        s = _compute_sum_3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
        p_new = (-f * h2 + s) * Jinv
        p[1:-1, 1:-1, 1:-1] = torch.where(black, p_new, p[1:-1, 1:-1, 1:-1])
        _bc_3d(p)

    s = _compute_sum_3d(cp0, cm0, cp1, cm1, cp2, cm2, p)
    J = _compute_J_3d(cp0, cm0, cp1, cm1, cp2, cm2)
    Au = (s - J * p[1:-1, 1:-1, 1:-1]) / h2
    r = torch.where(active, f - Au, torch.zeros_like(f))
    return p, r


# =====================================================================
# Benchmark harness
# =====================================================================

def make_problem(nx, ny, nz, device):
    """Create a variable-coefficient 3-D test problem."""
    h = 1.0 / nx
    h2_val = h * h

    # Interior grid for RHS
    f = torch.randn(nx - 2, ny - 2, nz - 2, device=device)
    f -= f.mean()

    # Initial guess with ghost cells
    p = torch.zeros(nx, ny, nz, device=device)

    # Variable coefficient field (smooth, > 0)
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    z = torch.linspace(0, 1, nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    c = 1.0 + 0.5 * torch.sin(2 * 3.14159 * X) * torch.cos(2 * 3.14159 * Y)

    # Face-averaged coefficients
    ch = 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1])
    cv = 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1])
    cw = 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])

    # Extract cfaces: (c_plus, c_minus) per dim
    cp0 = ch[1:, :, :]
    cm0 = ch[:-1, :, :]
    cp1 = cv[:, 1:, :]
    cm1 = cv[:, :-1, :]
    cp2 = cw[:, :, 1:]
    cm2 = cw[:, :, :-1]

    # h2 as a device tensor so torch.compile can use CUDA graphs
    h2 = torch.tensor(h2_val, device=device, dtype=f.dtype)

    # Extract cfaces: (c_plus, c_minus) per dim
    cp0 = ch[1:, :, :]
    cm0 = ch[:-1, :, :]
    cp1 = cv[:, 1:, :]
    cm1 = cv[:, :-1, :]
    cp2 = cw[:, :, 1:]
    cm2 = cw[:, :, :-1]

    # Red/black masks
    interior_shape = (nx - 2, ny - 2, nz - 2)
    ranges = [torch.arange(s, device=device) for s in interior_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    parity = sum(grids) % 2
    red = (parity == 0)
    black = (parity == 1)

    return f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, red, black


def bench_fn(fn, args, n_warmup=10, n_iter=100, label=""):
    """Time a function, returning mean/std in ms."""
    # Warmup (includes compile time for compiled fns)
    for _ in range(n_warmup):
        # Clone p so we don't accumulate state
        args_copy = list(args)
        args_copy[1] = args[1].clone()
        fn(*args_copy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        args_copy = list(args)
        args_copy[1] = args[1].clone()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args_copy)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1e3
    return times_ms.mean(), times_ms.std(), np.median(times_ms)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    grids = [
        (130, 34, 34),    # 128×32×32 + ghost cells
        (258, 66, 66),    # 256×64×64
    ]

    nsmoothing = 5
    w = 0.7
    jcap_tol = 1e-12
    n_warmup = 20
    n_iter = 200

    # Compile the kernels
    print("Compiling smoother kernels with torch.compile()...")
    t0 = time.perf_counter()
    jacobi_compiled = torch.compile(jacobi_smooth)
    rbgs_compiled = torch.compile(rbgs_smooth)
    compile_overhead = time.perf_counter() - t0
    print(f"  torch.compile() call took {compile_overhead:.2f}s (graph capture happens on first call)\n")

    for nx, ny, nz in grids:
        grid_label = f"{nx-2}×{ny-2}×{nz-2}"
        n_cells = (nx - 2) * (ny - 2) * (nz - 2)
        print("=" * 72)
        print(f"  Grid: {grid_label}  ({n_cells:,} interior cells)")
        print(f"  nsmoothing={nsmoothing}, n_warmup={n_warmup}, n_iter={n_iter}")
        print("=" * 72)

        f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, red, black = make_problem(
            nx, ny, nz, DEVICE
        )

        results = {}

        # ── Jacobi eager ─────────────────────────────────────────────
        args_jac = (f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, w, nsmoothing, jcap_tol)
        mean, std, med = bench_fn(jacobi_smooth, args_jac, n_warmup, n_iter,
                                   "Jacobi eager")
        results["Jacobi eager"] = (mean, std, med)
        print(f"  Jacobi  eager  : {mean:.3f} ± {std:.3f} ms  (median {med:.3f})")

        # ── Jacobi compiled ──────────────────────────────────────────
        mean, std, med = bench_fn(jacobi_compiled, args_jac, n_warmup, n_iter,
                                   "Jacobi compiled")
        results["Jacobi compiled"] = (mean, std, med)
        print(f"  Jacobi  compile: {mean:.3f} ± {std:.3f} ms  (median {med:.3f})")

        # ── RBGS eager ───────────────────────────────────────────────
        args_rbgs = (f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, nsmoothing, jcap_tol,
                     red, black)
        mean, std, med = bench_fn(rbgs_smooth, args_rbgs, n_warmup, n_iter,
                                   "RBGS eager")
        results["RBGS eager"] = (mean, std, med)
        print(f"  RBGS    eager  : {mean:.3f} ± {std:.3f} ms  (median {med:.3f})")

        # ── RBGS compiled ────────────────────────────────────────────
        mean, std, med = bench_fn(rbgs_compiled, args_rbgs, n_warmup, n_iter,
                                   "RBGS compiled")
        results["RBGS compiled"] = (mean, std, med)
        print(f"  RBGS    compile: {mean:.3f} ± {std:.3f} ms  (median {med:.3f})")

        # Summary
        print()
        jac_eager = results["Jacobi eager"][0]
        for name, (m, s, med) in results.items():
            speedup_vs_jac = jac_eager / m if m > 0 else float('nan')
            tag = ""
            if "compiled" in name:
                eager_name = name.replace("compiled", "eager")
                eager_m = results[eager_name][0]
                compile_speedup = eager_m / m if m > 0 else float('nan')
                tag = f"  compile speedup: {compile_speedup:.2f}×"
            print(f"    {name:<20s} {m:8.3f} ms  (vs Jacobi eager: {speedup_vs_jac:.2f}×){tag}")
        print()

    # ── Also try compile modes ───────────────────────────────────────
    print("=" * 72)
    print("  Testing compile modes on 128×32×32")
    print("=" * 72)

    nx, ny, nz = 130, 34, 34
    f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, red, black = make_problem(
        nx, ny, nz, DEVICE
    )
    args_jac = (f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, w, nsmoothing, jcap_tol)

    modes = ["default", "reduce-overhead", "max-autotune"]
    for mode in modes:
        try:
            fn_c = torch.compile(jacobi_smooth, mode=mode)
            mean, std, med = bench_fn(fn_c, args_jac, n_warmup, n_iter)
            print(f"  Jacobi compile mode={mode:<20s}: {mean:.3f} ± {std:.3f} ms  (median {med:.3f})")
        except Exception as e:
            print(f"  Jacobi compile mode={mode:<20s}: FAILED — {e}")

    for mode in modes:
        args_rbgs = (f, p, cp0, cm0, cp1, cm1, cp2, cm2, h2, nsmoothing, jcap_tol,
                     red, black)
        try:
            fn_c = torch.compile(rbgs_smooth, mode=mode)
            mean, std, med = bench_fn(fn_c, args_rbgs, n_warmup, n_iter)
            print(f"  RBGS   compile mode={mode:<20s}: {mean:.3f} ± {std:.3f} ms  (median {med:.3f})")
        except Exception as e:
            print(f"  RBGS   compile mode={mode:<20s}: FAILED — {e}")

    print("\nDone.")
