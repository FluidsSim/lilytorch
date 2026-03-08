"""
Test that variable-spacing operators reduce to the uniform-grid operators
when hx = hy = h everywhere.
"""
import torch
import pytest


# ─── helpers ────────────────────────────────────────────────────────────────

def _try_import_adv_diff():
    try:
        from lilytorch.src.adv_diff import AdvDiffSolver
        return AdvDiffSolver
    except ImportError:
        return None


def _try_import_fluid_solver():
    try:
        from lilytorch.src.solver import FluidSolver
        return FluidSolver
    except ImportError:
        return None


def make_fluid_solver(nx=10, ny=10):
    """Build a minimal FluidSolver from a small pars dict."""
    FluidSolver = _try_import_fluid_solver()
    if FluidSolver is None:
        pytest.skip("FluidSolver dependencies not available")
    pars = {
        "solver": {
            "Nx": nx,
            "Ny": ny,
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax": 1.0,
            "dt": 0.01,
            "nt": 10,
            "nu": 0.01,
            "rho": 1.0,
            "use_gpu": False,
            "nthreads": 1,
            "convection_method": "implicit",
            "poisson_tol": 1e-6,
            "poisson_max_mgcg_cycles": 5,
            "poisson_max_cycles": 5,
            "poisson_nsmoothing": 2,
            "jacobi_weight": 0.8,
            "poisson_verbose": False,
            "poisson_folder": "/tmp/poisson_test",
            "starting_iteration": 0,
            "starting_iteration_path": None,
        },
        "boundary_conditions": {
            "BC_type_u": ["D", "D", "D", "D"],
            "BC_values_u": [0, 0, 0, 0],
            "BC_type_v": ["D", "D", "D", "D"],
            "BC_values_v": [0, 0, 0, 0],
        },
        "output": {
            "save_path": "/tmp/test_output",
            "save_every": 10,
            "save_frames": False,
            "save_uv": False,
        },
        "body": {
            "n_bodies": 0,
            "bodies": [],
        },
    }
    return FluidSolver(pars, compute_forces=False)


def make_adv_diff_solver(nx=10, ny=10, method="explicit"):
    """Build a minimal AdvDiffSolver."""
    AdvDiffSolver = _try_import_adv_diff()
    if AdvDiffSolver is None:
        pytest.skip("pytorch_interpolation not available")
    device = torch.device("cpu")
    dtype = torch.float32
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    y = torch.linspace(0.0, 1.0, ny, dtype=dtype)
    dt = torch.tensor(0.001, dtype=dtype)
    nu = torch.tensor(0.01, dtype=dtype)
    return AdvDiffSolver(device, dt, x, y, nu, method=method)


# ─── GridSpacing tests (no external dependencies needed) ─────────────────────

def test_grid_spacing_class_uniform():
    """GridSpacing must create uniform hx/hy/hxy on a uniform grid."""
    from lilytorch.src.grid import GridSpacing
    nx, ny = 8, 10
    dx, dy = 0.1, 0.2
    gs = GridSpacing(nx, ny, dx, dy)

    assert gs.hx.shape == (nx, ny)
    assert gs.hy.shape == (nx, ny)
    assert gs.hxy.shape == (nx, ny)
    assert torch.allclose(gs.hx, dx * torch.ones(nx, ny))
    assert torch.allclose(gs.hy, dy * torch.ones(nx, ny))
    assert torch.allclose(gs.hxy, (dx * dy) * torch.ones(nx, ny))
    assert gs.is_uniform is True
    assert abs(gs.h - dx) < 1e-6


def test_grid_spacing_class_nonuniform():
    """GridSpacing.set_nonuniform must update hx, hy, hxy and clear is_uniform."""
    from lilytorch.src.grid import GridSpacing
    nx, ny = 6, 6
    gs = GridSpacing(nx, ny, 0.1, 0.1)

    hx_new = 0.05 * torch.ones(nx, ny)
    hy_new = 0.05 * torch.ones(nx, ny)
    hx_new[3:, :] = 0.1  # non-uniform: different spacing in upper half
    gs.set_nonuniform(hx_new, hy_new)

    assert torch.allclose(gs.hx, hx_new)
    assert torch.allclose(gs.hy, hy_new)
    assert torch.allclose(gs.hxy, hx_new * hy_new)
    assert gs.is_uniform is False
    with pytest.raises(ValueError):
        _ = gs.h


def test_gradient_uniform_equivalence():
    """Refactored gradient with uniform hx/hy must match original scalar-h gradient."""
    try:
        solver = make_fluid_solver(nx=10, ny=10)
    except Exception:
        pytest.skip("FluidSolver cannot be constructed in this environment")

    h = solver.h
    nx, ny = solver.nx, solver.ny
    torch.manual_seed(42)
    var = torch.rand(nx, ny, dtype=solver.dtype)

    dvar_dx_ref = torch.zeros_like(var)
    dvar_dy_ref = torch.zeros_like(var)
    dvar_dx_ref[1:-1, 1:-1] = (var[1:-1, 1:-1] - var[:-2, 1:-1]) / h
    dvar_dy_ref[1:-1, 1:-1] = (var[1:-1, 1:-1] - var[1:-1, :-2]) / h

    dvar_dx, dvar_dy = solver.gradient(var)

    assert torch.allclose(dvar_dx, dvar_dx_ref, atol=1e-6), \
        "gradient x-component mismatch on uniform grid"
    assert torch.allclose(dvar_dy, dvar_dy_ref, atol=1e-6), \
        "gradient y-component mismatch on uniform grid"


def test_divergence_uniform_equivalence():
    """Refactored divergence with uniform hx/hy must match scalar-h divergence."""
    try:
        solver = make_fluid_solver(nx=10, ny=10)
    except Exception:
        pytest.skip("FluidSolver cannot be constructed in this environment")

    h = solver.h
    nx, ny = solver.nx, solver.ny
    torch.manual_seed(42)
    u = torch.rand(nx, ny, dtype=solver.dtype)
    v = torch.rand(nx, ny, dtype=solver.dtype)

    div_ref = torch.zeros_like(u)
    div_ref[1:-1, 1:-1] = (
        (u[2:, 1:-1] - u[1:-1, 1:-1]) / h +
        (v[1:-1, 2:] - v[1:-1, 1:-1]) / h
    )

    div = solver.divergence(u, v)

    assert torch.allclose(div, div_ref, atol=1e-6), \
        "divergence mismatch on uniform grid"


def test_vorticity_uniform_equivalence():
    """Refactored vorticity with uniform hx/hy must match scalar-h vorticity."""
    try:
        solver = make_fluid_solver(nx=10, ny=10)
    except Exception:
        pytest.skip("FluidSolver cannot be constructed in this environment")

    h = solver.h
    nx, ny = solver.nx, solver.ny
    torch.manual_seed(42)
    u = torch.rand(nx, ny, dtype=solver.dtype)
    v = torch.rand(nx, ny, dtype=solver.dtype)

    dvdx_ref = torch.zeros_like(u)
    dudy_ref = torch.zeros_like(u)
    dvdx_ref[1:-1, 1:-1] = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / h
    dudy_ref[1:-1, 1:-1] = (u[1:-1, 1:-1] - u[1:-1, :-2]) / h
    vort_ref = dvdx_ref - dudy_ref

    vort = solver.vorticity(u, v)

    assert torch.allclose(vort, vort_ref, atol=1e-6), \
        "vorticity mismatch on uniform grid"


def test_gradient_stag_uniform_equivalence():
    """Refactored gradient_xstag/gradient_ystag must match scalar-h version."""
    try:
        solver = make_fluid_solver(nx=10, ny=10)
    except Exception:
        pytest.skip("FluidSolver cannot be constructed in this environment")

    h = solver.h
    nx, ny = solver.nx, solver.ny
    torch.manual_seed(42)
    var = torch.rand(nx, ny, dtype=solver.dtype)

    gx_ref = (var[2:, 1:-1] - var[1:-1, 1:-1]) / h
    gy_ref = (var[1:-1, 2:] - var[1:-1, 1:-1]) / h

    gx = solver.gradient_xstag(var)
    gy = solver.gradient_ystag(var)

    assert torch.allclose(gx, gx_ref, atol=1e-6), \
        "gradient_xstag mismatch on uniform grid"
    assert torch.allclose(gy, gy_ref, atol=1e-6), \
        "gradient_ystag mismatch on uniform grid"


def test_hx_hy_hxy_attributes():
    """FluidSolver must expose hx, hy, hxy of shape (nx, ny)."""
    try:
        solver = make_fluid_solver(nx=8, ny=8)
    except Exception:
        pytest.skip("FluidSolver cannot be constructed in this environment")

    nx, ny = solver.nx, solver.ny
    assert hasattr(solver, "hx"), "FluidSolver missing hx attribute"
    assert hasattr(solver, "hy"), "FluidSolver missing hy attribute"
    assert hasattr(solver, "hxy"), "FluidSolver missing hxy attribute"
    assert solver.hx.shape == (nx, ny), f"hx shape {solver.hx.shape} != ({nx},{ny})"
    assert solver.hy.shape == (nx, ny), f"hy shape {solver.hy.shape} != ({nx},{ny})"
    assert solver.hxy.shape == (nx, ny), f"hxy shape {solver.hxy.shape} != ({nx},{ny})"

    h = solver.h
    assert torch.allclose(solver.hx, h * torch.ones_like(solver.hx)), \
        "hx should be uniform on a square grid"
    assert torch.allclose(solver.hy, h * torch.ones_like(solver.hy)), \
        "hy should be uniform on a square grid"
    assert torch.allclose(solver.hxy, (h**2) * torch.ones_like(solver.hxy)), \
        "hxy should equal h^2 on a square grid"


def test_adv_diff_dx_dy_attributes():
    """AdvDiffSolver must expose DX, DY of shape (nx, ny)."""
    solver = make_adv_diff_solver(nx=8, ny=8, method="explicit")
    nx, ny = solver.nx, solver.ny

    assert hasattr(solver, "DX"), "AdvDiffSolver missing DX attribute"
    assert hasattr(solver, "DY"), "AdvDiffSolver missing DY attribute"
    assert solver.DX.shape == (nx, ny), f"DX shape {solver.DX.shape} != ({nx},{ny})"
    assert solver.DY.shape == (nx, ny), f"DY shape {solver.DY.shape} != ({nx},{ny})"

    assert torch.allclose(solver.DX, solver.dx * torch.ones_like(solver.DX)), \
        "DX should be uniform on a uniform grid"
    assert torch.allclose(solver.DY, solver.dy * torch.ones_like(solver.DY)), \
        "DY should be uniform on a uniform grid"


def test_adv_diff_set_spacing():
    """set_spacing must update DX and DY."""
    solver = make_adv_diff_solver(nx=8, ny=8, method="explicit")
    nx, ny = solver.nx, solver.ny

    new_DX = 0.05 * torch.ones((nx, ny))
    new_DY = 0.05 * torch.ones((nx, ny))
    solver.set_spacing(new_DX, new_DY)

    assert torch.allclose(solver.DX, new_DX), "DX not updated by set_spacing"
    assert torch.allclose(solver.DY, new_DY), "DY not updated by set_spacing"


def test_adv_diff_uniform_equivalence():
    """Refactored AdvDiffSolver with uniform DX/DY must match scalar dtdx2 version."""
    solver = make_adv_diff_solver(nx=12, ny=12, method="explicit")
    nx, ny = solver.nx, solver.ny
    torch.manual_seed(42)

    # Reference: compute diffusion explicitly using scalar spacing
    u = torch.rand(nx, ny, dtype=solver.DX.dtype)
    v = torch.rand(nx, ny, dtype=solver.DX.dtype)

    u_ref = u.clone()
    v_ref = v.clone()
    dx = solver.dx
    dy = solver.dy
    dt = float(solver.dt)
    nu = float(solver.nu)

    u_ref[1:-1, 1:-1] = (
        u[1:-1, 1:-1] -
        (dt / dx) * u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]) -
        (dt / dy) * v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2]) +
        nu * (dt / dx**2) * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
        nu * (dt / dy**2) * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
    )
    v_ref[1:-1, 1:-1] = (
        v[1:-1, 1:-1] -
        (dt / dx) * u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]) -
        (dt / dy) * v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2]) +
        nu * (dt / dx**2) * (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) +
        nu * (dt / dy**2) * (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2])
    )

    u_out, v_out = solver.solve_explicit(u.clone(), v.clone())

    assert torch.allclose(u_out, u_ref, atol=1e-5), \
        "solve_explicit u mismatch on uniform grid"
    assert torch.allclose(v_out, v_ref, atol=1e-5), \
        "solve_explicit v mismatch on uniform grid"
