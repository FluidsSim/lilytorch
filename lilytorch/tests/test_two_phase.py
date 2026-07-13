"""Unit tests for the two-phase VOF helper (:mod:`lilytorch.src.two_phase`).

Run directly (``python -m lilytorch.src.test_two_phase``) or via pytest.
"""

import math
import torch

from lilytorch.src.two_phase import TwoPhase


def _grid(N=40, L=1.0, ndim=2, dtype=torch.float64):
    h = L / N
    coords = [torch.linspace(0.0, L, N, dtype=dtype) for _ in range(ndim)]
    return coords, h


# ---------------------------------------------------------------------------
# Material-field blends (analytic)
# ---------------------------------------------------------------------------
def test_density_and_viscosity_blends():
    (x, y), h = _grid()
    tp = TwoPhase(x, y, h, lambda X, Y: (Y < 0.5).double(),
                  rho_water=1000.0, rho_air=1.0,
                  nu_water=2.0, nu_air=0.5)
    q = tp.recip_density_cc()
    # reciprocal density: 1/ρ_water in water, 1/ρ_air in air
    assert torch.isclose(q[tp.alpha == 1].min(), torch.tensor(1.0 / 1000.0, dtype=q.dtype))
    assert torch.isclose(q[tp.alpha == 0].max(), torch.tensor(1.0 / 1.0, dtype=q.dtype))
    # explicit half-fraction cell: ρ_cc = 500.5 → q = 1/500.5
    tp.alpha.fill_(0.5)
    assert torch.allclose(tp.recip_density_cc(),
                          torch.full_like(tp.alpha, 1.0 / 500.5))
    assert torch.allclose(tp.viscosity_cc(), torch.full_like(tp.alpha, 1.25))


def test_recip_density_face_harmonic():
    """recip_density_face is the arithmetic mean of 1/ρ = reciprocal of the
    harmonic face density."""
    (x, y), h = _grid(N=8)
    # vertical step: left half water, right half air (jump along x = dim 0)
    tp = TwoPhase(x, y, h, lambda X, Y: (X < 0.5).double(),
                  rho_water=1000.0, rho_air=1.0)
    qf = tp.recip_density_face(0)
    # interior all-water / all-air faces reduce to the bulk reciprocal
    assert torch.isclose(qf.min(), torch.tensor(1.0 / 1000.0, dtype=qf.dtype))
    # at the water/air cut face: 0.5*(1/1000 + 1/1) = 0.50050
    cut = qf[(qf > 1.0 / 1000.0 + 1e-9) & (qf < 1.0 / 1.0 - 1e-9)]
    assert cut.numel() > 0
    assert torch.allclose(cut, torch.full_like(cut, 0.5 * (1.0 / 1000.0 + 1.0 / 1.0)))
    # equals the reciprocal of the harmonic density mean 2ρ_iρ_j/(ρ_i+ρ_j)
    harm_density = 2.0 * 1000.0 * 1.0 / (1000.0 + 1.0)
    assert torch.allclose(cut, torch.full_like(cut, 1.0 / harm_density))


def test_arithmetic_face_density_rejected():
    """The legacy arithmetic face-density option is no longer accepted."""
    import pytest
    (x, y), h = _grid(N=8)
    with pytest.raises(TypeError):
        TwoPhase(x, y, h, lambda X, Y: (X < 0.5).double(),
                 face_density="arithmetic")


# ---------------------------------------------------------------------------
# Transport: boundedness + mass conservation
# ---------------------------------------------------------------------------
def _taylor_green(coords, h):
    """Divergence-free velocity from psi = sin(pi x) sin(pi y), zero normal
    velocity on the [0,1]^2 walls (no boundary flux)."""
    X, Y = torch.meshgrid(coords[0], coords[1], indexing="ij")
    u =  math.pi * torch.sin(math.pi * X) * torch.cos(math.pi * Y)   #  d psi/dy
    v = -math.pi * torch.cos(math.pi * X) * torch.sin(math.pi * Y)   # -d psi/dx
    return u, v


def test_boundedness_and_mass_conservation_2d():
    (x, y), h = _grid(N=48)
    cx = cy = 0.5; r = 0.2
    tp = TwoPhase(x, y, h,
                  lambda X, Y: ((X - cx)**2 + (Y - cy)**2 < r**2).double(),
                  rho_water=1000.0, rho_air=1.0)
    u, v = _taylor_green((x, y), h)
    umax = max(u.abs().max().item(), v.abs().max().item())
    dt = 0.2 * h / umax
    V0 = tp.water_volume()
    for _ in range(100):
        tp.advect(u, v, dt=dt)
        assert tp.alpha.min() >= -1e-12 and tp.alpha.max() <= 1.0 + 1e-12
    drift = abs(tp.water_volume() - V0) / V0
    # The Weymouth-Yue scheme conserves volume to round-off for a DISCRETELY
    # divergence-free velocity (as produced by the projection in the real
    # solver — see the dam-break validation's ~round-off vol drift). Here the
    # *analytic* Taylor-Green field sampled at cell centres is not discretely
    # div-free, so the divergence-correction terms leave a small O(h dt Nstep)
    # residual; bound it loosely.
    assert drift < 5e-3, f"water-volume drift too large: {drift:.2e}"


def test_boundedness_3d():
    h = 1.0 / 24
    x = y = z = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    tp = TwoPhase(x, y, h, lambda X, Y, Z: (Z < 0.5).double(), z=z)
    u = torch.full_like(tp.alpha, 0.0)
    w = torch.full_like(tp.alpha, 0.1)
    dt = 0.2 * h / 0.1
    for _ in range(20):
        tp.advect(u, u, w, dt=dt)
        assert tp.alpha.min() >= -1e-12 and tp.alpha.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Body-aware initial interface (carve + volume compensation)
# ---------------------------------------------------------------------------
def _circle_sdf(X, Y, cx, cy, r):
    return ((X - cx) ** 2 + (Y - cy) ** 2).sqrt() - r


def test_body_aware_carve_2d():
    from lilytorch.src.two_phase_solver import body_aware_alpha_init
    (x, y), h = _grid(N=64)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    level = 0.5
    init = lambda X, Y: (Y < level).double()
    sdf = _circle_sdf(X, Y, 0.5, 0.5, 0.15)        # straddles the interface
    eps = 2.0 * h
    inner = (slice(1, -1), slice(1, -1))

    a = body_aware_alpha_init(init, sdf, eps, h, compensate=False,
                              verbose=False)(X, Y)
    # body interior is dry; the far field is untouched
    assert float(a[sdf < -eps].max()) == 0.0
    far_water = (sdf > 0.1) & (Y < level - h)
    assert torch.allclose(a[far_water], torch.ones_like(a[far_water]))
    # carved volume deficit ~ submerged (half-disc) volume
    deficit = float(init(X, Y)[inner].sum() - a[inner].sum()) * h * h
    half_disc = 0.5 * math.pi * 0.15 ** 2
    assert abs(deficit - half_disc) < 0.15 * half_disc


def test_body_aware_volume_compensation_2d():
    from lilytorch.src.two_phase_solver import body_aware_alpha_init
    (x, y), h = _grid(N=64)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    level = 0.5
    init = lambda X, Y: (Y < level).double()
    sdf = _circle_sdf(X, Y, 0.5, 0.5, 0.15)
    eps = 2.0 * h
    inner = (slice(1, -1), slice(1, -1))
    target = float(init(X, Y)[inner].sum())

    a = body_aware_alpha_init(init, sdf, eps, h, compensate=True,
                              verbose=False)(X, Y)
    # exact-volume blend: the total water matches the uncarved init
    assert abs(float(a[inner].sum()) - target) < 1e-6
    # the body interior stays dry (compensation raises the level, never wets
    # the carved interior)
    assert float(a[sdf < -eps].max()) == 0.0
    # the far-field surface rose by ~ displaced volume / free-surface width
    # (half-disc 0.0353 over width 0.7 -> ~0.05): the water column away from
    # the body is now taller than the flat init
    col = a[5, 1:-1]                               # far column (x ~ 0.08)
    height = float(col.sum()) * h
    assert height > level + 0.03
    # bounded
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_body_aware_body_above_water_noop():
    from lilytorch.src.two_phase_solver import body_aware_alpha_init
    (x, y), h = _grid(N=64)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    init = lambda X, Y: (Y < 0.4).double()
    sdf = _circle_sdf(X, Y, 0.5, 0.8, 0.1)         # entirely in the air
    inner = (slice(1, -1), slice(1, -1))
    a = body_aware_alpha_init(init, sdf, 2.0 * h, h, compensate=True,
                              verbose=False)(X, Y)
    # nothing to carve, nothing to compensate
    assert torch.allclose(a, init(X, Y))


def test_body_aware_3d_with_twophase():
    from lilytorch.src.two_phase_solver import body_aware_alpha_init
    N, L = 32, 1.0
    h = L / N
    x = y = z = torch.linspace(0.0, L, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    level = 0.5
    init = lambda X, Y, Z: (Z < level).double()
    sdf = ((X - 0.5) ** 2 + (Y - 0.5) ** 2
           + (Z - 0.5) ** 2).sqrt() - 0.2          # sphere at the waterline
    inner = (slice(1, -1),) * 3
    target = float(init(X, Y, Z)[inner].sum())

    wrapped = body_aware_alpha_init(init, sdf, 2.0 * h, h, verbose=False)
    tp = TwoPhase(x, y, h, wrapped, z=z)
    # dry interior + exact total volume, end to end through TwoPhase
    assert float(tp.alpha[sdf < -2.0 * h].max()) == 0.0
    assert abs(tp.initial_water_volume - target * h ** 3) < 1e-6 * h ** 3


# ---------------------------------------------------------------------------
# Solver-level reduction parity: uniform-density two-phase == single-phase
# ---------------------------------------------------------------------------
# Rationale: with rho_air == rho_water and nu_air == nu_water the variable-
# density VOF projection coefficient ``dt*mu0_eff*(1/rho)`` collapses to the
# single-phase ``dt*mu0/rho`` (recip_density is constant regardless of alpha),
# the air-transparent velocity identity ``a*S+(1-a)*u'`` becomes ``S`` where
# alpha==1, and BDIM/advection/Poisson are inherited unchanged.  So the
# TwoPhaseSolver MUST reproduce the FluidSolver field-by-field.
#
# This is a *fluid* reduction.  It is decoupled from the body-support physics
# (analytical Archimedes in single-phase vs emergent buoyancy in two-phase),
# which legitimately differ and only matter for a FREE body under gravity --
# hence gravity is OFF and the FSI body below is STATIC (pinned), so weight /
# buoyancy never enter and only the fluid+BDIM coupling is compared.

import math as _math


def _parity_pars(ndim, N, dt, nu, rho, body_sdf, two_phase=None):
    """Minimal CPU/float64 ``pars`` dict for a closed no-slip box; identical
    for both solvers except the optional ``solver.two_phase`` block."""
    L = 1.0
    solver = {
        "use_gpu"                : False,
        "nthreads"               : 1,
        "Nx"                     : N,
        "Ny"                     : N,
        "xmin"                   : 0.0, "xmax": L,
        "ymin"                   : 0.0, "ymax": L,
        "nt"                     : 10_000,
        "nu"                     : nu,
        "rho"                    : rho,
        "dt"                     : dt,
        "convection_method"      : "abdquickest",
        "poisson_method"         : "multigrid",
        "poisson_tol"            : 1.0e-12,
        "jacobi_weight"          : 0.8,
        "poisson_max_cycles"     : 40,
        "poisson_max_mgcg_cycles": 40,
        "poisson_nsmoothing"     : 6,
        "poisson_verbose"        : False,
        "bdim_mu0_projection"    : True,   # aligns the coeff with the two-phase form
        "force_method"           : "eulerian",
    }
    n_faces = 2 * ndim
    bcs = {
        "BC_type_u"  : ["D"] * n_faces, "BC_values_u": [0.0] * n_faces,
        "BC_type_v"  : ["D"] * n_faces, "BC_values_v": [0.0] * n_faces,
    }
    if ndim == 3:
        solver["Nz"] = N
        solver["zmin"] = 0.0
        solver["zmax"] = L
        bcs["BC_type_w"]   = ["D"] * n_faces
        bcs["BC_values_w"] = [0.0] * n_faces
    if two_phase is not None:
        solver["two_phase"] = two_phase
    body = {
        "type"       : "composite_analytical",
        "plotting"   : False,
        "sdf"        : body_sdf,
        "update_maps": [
            {"rotation": "lambda t: 0.0*t",
             "translation": ["lambda t: 0.0*t"] * ndim}
            for _ in body_sdf
        ],
    }
    output = {"save_frames": False, "save_every": 10**9, "vmin": -1.0, "vmax": 1.0}
    return {"solver": solver, "boundary_conditions": bcs,
            "body": body, "output": output}


def _taylor_green_ic(solver):
    """Divergence-free Taylor-Green velocity on the solver's own grid (zero
    normal velocity on the [0,1] walls).  Returns (u, v[, w]) all matching
    ``solver.grid_shape``."""
    two_pi = 2.0 * _math.pi
    if solver.ndim == 2:
        X, Y = torch.meshgrid(solver.x, solver.y, indexing="ij")
        u =  torch.sin(two_pi * X) * torch.cos(two_pi * Y)
        v = -torch.cos(two_pi * X) * torch.sin(two_pi * Y)
        return u, v
    X, Y, Z = torch.meshgrid(solver.x, solver.y, solver.z, indexing="ij")
    u =  torch.sin(two_pi * X) * torch.cos(two_pi * Y)
    v = -torch.cos(two_pi * X) * torch.sin(two_pi * Y)
    w =  torch.zeros_like(u)
    return u, v, w


def _set_ic(solver, fields):
    solver.set_initial_conditions()
    solver.u0 = fields[0].clone()
    solver.v0 = fields[1].clone()
    if solver.ndim == 3:
        solver.w0 = fields[2].clone()
    solver.p0 = torch.zeros_like(solver.u0)


def _step_n(solver, nsteps):
    """Drive the explicit fluid core for ``nsteps`` and return the final
    (u, v, p[, w]) — uses the same per-step methods as the production loop."""
    u, v, p = solver.u0, solver.v0, solver.p0
    w = solver.w0 if solver.ndim == 3 else None
    for it in range(nsteps):
        t = it * float(solver.dt)
        out = solver.advance_and_compute_loads(u, v, p, it, t, w_vel=w)
        u, v, p, w = out
        solver.finalize_step(u, v, p, it, w_vel=w)
    return u, v, p, w


def _uniform_two_phase_block(rho, nu, alpha_init):
    return {
        "alpha_init"          : alpha_init,
        "rho_water"           : rho,
        "rho_air"             : rho,       # uniform: must reduce to single-phase
        "nu_water"            : nu,
        "nu_air"              : nu,
        "alpha_exclude_body"  : False,
        "air_transparent_body": False,
    }


def _assert_parity(sp, tp, label, atol=1e-7, rtol=1e-6):
    for name in (("u0", "v0", "w0", "p0") if sp.ndim == 3 else ("u0", "v0", "p0")):
        a = getattr(sp, name)
        b = getattr(tp, name)
        diff = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-30
        assert torch.allclose(a, b, atol=atol, rtol=rtol), (
            f"[{label}] field {name} diverged: max|Δ|={diff:.3e} "
            f"(rel {diff/scale:.3e})")


# ---- interface present + advecting: alpha transport must not leak into the
#      momentum update.  Probe body sits deep in the water half (alpha==1
#      around it -> mu0_eff==mu0), the free water/air interface lives near the
#      top and is transported every step. ----------------------------------
def test_uniform_two_phase_reduces_to_single_phase_2d():
    N, dt, nu, rho, nsteps = 32, 2.0e-3, 1.0e-2, 1000.0, 12
    body = ["lambda x, y: circle(x,y,xt=0.5,yt=0.25,r=0.08)"]
    tp_block = _uniform_two_phase_block(
        rho, nu, "lambda X, Y: (Y < 0.7).double()")   # interface above the body

    # field-only comparison -> skip the body-load readout
    sp = __import__("lilytorch.src.solver", fromlist=["FluidSolver"]).FluidSolver(
        _parity_pars(2, N, dt, nu, rho, body), dtype=torch.float64,
        compute_forces=False)
    tpd = __import__("lilytorch.src.two_phase_solver",
                     fromlist=["TwoPhaseSolver"]).TwoPhaseSolver(
        _parity_pars(2, N, dt, nu, rho, body, two_phase=tp_block),
        dtype=torch.float64, compute_forces=False)

    ic = _taylor_green_ic(sp)
    _set_ic(sp, ic); _set_ic(tpd, ic)
    _step_n(sp, nsteps); _step_n(tpd, nsteps)
    _assert_parity(sp, tpd, "interface-2d")


def test_uniform_two_phase_reduces_to_single_phase_3d():
    N, dt, nu, rho, nsteps = 20, 2.0e-3, 1.0e-2, 1000.0, 8
    body = ["lambda x, y, z: sphere(x,y,z,xt=0.5,yt=0.5,zt=0.25,r=0.1)"]
    tp_block = _uniform_two_phase_block(
        rho, nu, "lambda X, Y, Z: (Z < 0.7).double()")

    # field-only comparison -> skip the body-load readout (the 3-D analytical
    # method2 force path is exercised separately; here we test the fluid fields)
    sp = __import__("lilytorch.src.solver", fromlist=["FluidSolver"]).FluidSolver(
        _parity_pars(3, N, dt, nu, rho, body), dtype=torch.float64,
        compute_forces=False)
    tpd = __import__("lilytorch.src.two_phase_solver",
                     fromlist=["TwoPhaseSolver"]).TwoPhaseSolver(
        _parity_pars(3, N, dt, nu, rho, body, two_phase=tp_block),
        dtype=torch.float64, compute_forces=False)

    ic = _taylor_green_ic(sp)
    _set_ic(sp, ic); _set_ic(tpd, ic)
    _step_n(sp, nsteps); _step_n(tpd, nsteps)
    _assert_parity(sp, tpd, "interface-3d")


# ---- FSI (static body, fully submerged in all-water alpha): the BDIM-coupled
#      fields AND the integrated body loads must match single-phase ----------
def test_uniform_two_phase_fsi_matches_single_phase_2d():
    N, dt, nu, rho, nsteps = 32, 2.0e-3, 1.0e-2, 1000.0, 12
    body = ["lambda x, y: circle(x,y,xt=0.5,yt=0.5,r=0.15)"]
    # all-water alpha (no air anywhere) so the body sits fully in water:
    # mu0_eff = alpha*mu0 + (1-alpha) = mu0, S identity = full BDIM.
    tp_block = _uniform_two_phase_block(rho, nu, "lambda X, Y: torch.ones_like(X)")

    sp = __import__("lilytorch.src.solver", fromlist=["FluidSolver"]).FluidSolver(
        _parity_pars(2, N, dt, nu, rho, body), dtype=torch.float64)
    tpd = __import__("lilytorch.src.two_phase_solver",
                     fromlist=["TwoPhaseSolver"]).TwoPhaseSolver(
        _parity_pars(2, N, dt, nu, rho, body, two_phase=tp_block),
        dtype=torch.float64)

    ic = _taylor_green_ic(sp)
    _set_ic(sp, ic); _set_ic(tpd, ic)
    _step_n(sp, nsteps); _step_n(tpd, nsteps)

    _assert_parity(sp, tpd, "fsi-2d-fields")
    # integrated body loads must agree too (the readout path is shared, but the
    # coefficients/velocity feeding it come through the two-phase override)
    for rec in ("pressure_drag_record", "viscous_drag_record"):
        a = getattr(sp, rec)[:, :, :nsteps]
        b = getattr(tpd, rec)[:, :, :nsteps]
        diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (
            f"[fsi-2d-loads] {rec} diverged: max|Δ|={diff:.3e}")


# ---------------------------------------------------------------------------
# CUDA W&Y sweep kernel (MP10 / T2d) — parity vs the pure-PyTorch oracle
# ---------------------------------------------------------------------------
# Pure-PyTorch Weymouth-Yue reference, kept HERE (test-only) as the independent
# oracle for the single-source Warp ``cvof_sweep`` kernel.  It used to live on
# ``TwoPhase`` as ``_cvof_sweep_python`` / ``_shift``; moved into the test suite
# when the production sweep became Warp-only (no source-side duplicate).
from lilytorch.src.advection import _sl as _oracle_sl


def _oracle_shift(a, s, d, nd):
    """Shift ``a`` by ``s`` cells along dim ``d`` with edge replication."""
    S = lambda sl: _oracle_sl(nd, d, sl)
    if s == 1:
        return torch.cat([a[S(slice(0, 1))], a[S(slice(0, -1))]], dim=d)
    if s == 2:
        return torch.cat([a[S(slice(0, 1))], a[S(slice(0, 1))],
                          a[S(slice(0, -2))]], dim=d)
    if s == -1:
        return torch.cat([a[S(slice(1, None))], a[S(slice(-1, None))]], dim=d)
    raise ValueError(f"_oracle_shift: unsupported offset {s}")


def _oracle_cvof_sweep_python(a, u_d, d, dt, nd, h):
    """Pure-PyTorch reference for one W&Y conservative sweep along ``d``.

    MAC convention: ``u_d[k]`` is the face left of cell ``k``.  Face value is
    the W&Y 2nd-order Courant-corrected, van-Leer-limited donor extrapolation
    plus the divergence correction.  Returns a new tensor with the interior
    updated.  Independent oracle for the Warp kernel."""
    S   = lambda s: _oracle_sl(nd, d, s)
    cfl = dt / h
    C   = u_d * cfl
    a_m1 = _oracle_shift(a,  1, d, nd)
    a_m2 = _oracle_shift(a,  2, d, nd)
    a_p1 = _oracle_shift(a, -1, d, nd)

    def _vleer(db, df):
        denom = torch.where(db + df == 0.0, torch.ones_like(db), db + df)
        s = 2.0 * db * df / denom
        return torch.where(db * df > 0.0, s, torch.zeros_like(s))

    s_pos    = _vleer(a_m1 - a_m2, a - a_m1)
    face_pos = a_m1 + 0.5 * (1.0 - C) * s_pos
    s_neg    = _vleer(a - a_m1, a_p1 - a)
    face_neg = a - 0.5 * (1.0 + C) * s_neg
    F = u_d * torch.where(C >= 0.0, face_pos, face_neg)
    out = a.clone()
    FL = F[S(slice(1, -1))]
    FR = F[S(slice(2, None))]
    uL = u_d[S(slice(1, -1))]
    uR = u_d[S(slice(2, None))]
    ai = a[S(slice(1, -1))]
    out[S(slice(1, -1))] = ai + cfl * (FL - FR + ai * (uR - uL))
    return out


def _cvof_kernel_vs_python(ndim, dtype, noncontig):
    """Compare the single-source Warp ``cvof_sweep`` kernel against the
    pure-PyTorch oracle on identical CUDA inputs (isolates the kernel's
    arithmetic from any CPU/GPU or single/double differences)."""
    dev = torch.device("cuda")
    N = 24 if ndim == 3 else 40
    L = 1.0
    h = L / N
    coords = [torch.linspace(0.0, L, N, dtype=dtype, device=dev) for _ in range(ndim)]
    cx = 0.5
    if ndim == 2:
        X, Y = torch.meshgrid(*coords, indexing="ij")
        alpha = ((X - cx) ** 2 + (Y - cx) ** 2 < 0.2 ** 2).to(dtype)
        # divergence-bearing velocity with BOTH signs (exercises C>=0 and C<0)
        u = math.pi * torch.sin(math.pi * X) * torch.cos(math.pi * Y)
        v = -math.pi * torch.cos(math.pi * X) * torch.sin(math.pi * Y)
        vels = [u, v]
    else:
        X, Y, Z = torch.meshgrid(*coords, indexing="ij")
        alpha = ((X - cx) ** 2 + (Y - cx) ** 2 + (Z - cx) ** 2 < 0.2 ** 2).to(dtype)
        u = math.pi * torch.sin(math.pi * X) * torch.cos(math.pi * Y)
        v = -math.pi * torch.cos(math.pi * X) * torch.sin(math.pi * Y)
        w = 0.5 * torch.cos(math.pi * Z)
        vels = [u, v, w]

    tp = TwoPhase(coords[0], coords[1], h, lambda *a: alpha,
                  z=(coords[2] if ndim == 3 else None), device=dev, dtype=dtype)
    dt = 0.2 * h / max(float(t.abs().max()) for t in vels)

    if noncontig:
        # mimic the stacked-_vel row views: build a genuinely strided view by
        # interleaving into a trailing axis (stride-2 on the last real dim).
        strided = []
        for t in vels:
            buf = torch.empty(t.shape + (2,), dtype=dtype, device=dev)
            buf[..., 0] = t
            view = buf[..., 0]
            assert not view.is_contiguous()
            strided.append(view)
        vels = strided

    a = tp.alpha
    rels = []
    for d in range(ndim):
        from lilytorch.src.two_phase import _neumann_pad
        ad = a.clone(); _neumann_pad(ad)
        out_k = tp._cvof_sweep(ad, vels[d], d, dt)          # kernel (CUDA)
        out_p = _oracle_cvof_sweep_python(ad, vels[d], d, dt, ndim, tp.h)  # oracle
        diff = (out_k - out_p).abs().max().item()
        scale = max(out_p.abs().max().item(), 1.0)
        rels.append(diff / scale)
    return max(rels)


def test_cvof_sweep_kernel_parity_2d_f64():
    r = _cvof_kernel_vs_python(2, torch.float64, noncontig=False)
    if r is None:
        return
    assert r < 1e-12, f"2D f64 cvof_sweep kernel parity: rel={r:.3e}"


def test_cvof_sweep_kernel_parity_3d_f64():
    r = _cvof_kernel_vs_python(3, torch.float64, noncontig=False)
    if r is None:
        return
    assert r < 1e-12, f"3D f64 cvof_sweep kernel parity: rel={r:.3e}"


def test_cvof_sweep_kernel_parity_2d_f32():
    r = _cvof_kernel_vs_python(2, torch.float32, noncontig=False)
    if r is None:
        return
    assert r < 1e-5, f"2D f32 cvof_sweep kernel parity: rel={r:.3e}"


def test_cvof_sweep_kernel_parity_noncontig_f64():
    """Strided velocity views (as produced by the stacked-_vel row views)."""
    r = _cvof_kernel_vs_python(2, torch.float64, noncontig=True)
    if r is None:
        return
    assert r < 1e-12, f"2D f64 non-contig cvof_sweep parity: rel={r:.3e}"


def test_cvof_kernel_advect_bounded_and_conservative_2d():
    """End-to-end: the kernel-backed advect stays bounded + conserves mass to
    the same tolerance as the Python oracle (mirrors the CPU test on CUDA)."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    N = 48; L = 1.0; h = L / N
    x = y = torch.linspace(0.0, L, N, dtype=torch.float64, device=dev)
    cx = cy = 0.5; r = 0.2
    tp = TwoPhase(x, y, h,
                  lambda X, Y: ((X - cx) ** 2 + (Y - cy) ** 2 < r ** 2).double(),
                  device=dev, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u = math.pi * torch.sin(math.pi * X) * torch.cos(math.pi * Y)
    v = -math.pi * torch.cos(math.pi * X) * torch.sin(math.pi * Y)
    dt = 0.2 * h / max(u.abs().max().item(), v.abs().max().item())
    V0 = tp.water_volume()
    for _ in range(100):
        tp.advect(u, v, dt=dt)
        assert tp.alpha.min() >= -1e-12 and tp.alpha.max() <= 1.0 + 1e-12
    drift = abs(tp.water_volume() - V0) / V0
    assert drift < 5e-3, f"kernel-path water-volume drift: {drift:.2e}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS  {name}")
    print("All two-phase unit tests passed.")


# ---------------------------------------------------------------------------
#  cuda_native_port Phase 4.3: graph-captured VOF transport
# ---------------------------------------------------------------------------
#  ``TwoPhase.advect_graph_aware`` replaces ``advect``'s per-sweep clone() with
#  persistent double-buffered intermediates so the directional-split sequence is
#  a static-shape region that ``torch.cuda.CUDAGraph`` can capture.  Two things
#  can silently break and produce WRONG PHYSICS rather than an error:
#
#    * the sweep-parity alternation: parity is a *Python attribute write*, and
#      Python does not execute on graph replay — so it must be toggled OUTSIDE
#      the capture and folded into the graph key (one graph per parity);
#    * ``self.alpha`` must stay pointer-stable, or the key churns and the graph
#      re-captures every step.
#
#  This gate pins both: N steps of the graph path must be BIT-EXACT against the
#  eager ``advect`` reference, and the graph must actually replay.
def _cvof_graph_vs_eager(ndim, dtype):
    if not torch.cuda.is_available():
        return
    from lilytorch.src.graph_capture import NativeWholeStepGraphRunner

    dev = torch.device("cuda")
    N = 32 if ndim == 3 else 64
    L = 1.0
    coords = [torch.linspace(0.0, L, N, dtype=dtype, device=dev)
              for _ in range(ndim)]
    h = float(coords[0][1] - coords[0][0])

    def alpha_init(*G):
        return (G[1] < 0.5).to(dtype)            # water below the waterline

    def build():
        return TwoPhase(coords[0], coords[1], h, alpha_init,
                        z=(coords[2] if ndim == 3 else None),
                        device=dev, dtype=dtype)

    tp_eager, tp_graph = build(), build()

    g = torch.Generator(device=dev).manual_seed(1234)
    vels = [0.3 * torch.rand((N,) * ndim, generator=g, dtype=dtype, device=dev) - 0.15
            for _ in range(ndim)]
    dt = 0.2 * h / 0.15                          # CFL ~ 0.2

    runner = NativeWholeStepGraphRunner()
    alpha_ptrs = set()

    for _ in range(12):
        tp_eager.advect(*vels, dt=dt)            # reference (toggles parity itself)

        # exactly what TwoPhaseSolver._advect_vof does: toggle OUTSIDE the graph,
        # then key on (field pointers, parity, dt).
        tp_graph._sweep_parity = not getattr(tp_graph, "_sweep_parity", False)
        key = tuple(t.data_ptr() for t in (tp_graph.alpha, *vels)) \
            + (int(tp_graph._sweep_parity), dt)
        runner.run(key, "cuda:0",
                   lambda: tp_graph.advect_graph_aware(*vels, dt=dt), stage=None)
        alpha_ptrs.add(tp_graph.alpha.data_ptr())

    # bit-exact: the graph path is a pure re-plumbing of advect(), not an approximation
    assert torch.equal(tp_eager.alpha, tp_graph.alpha), (
        f"{ndim}-D {dtype}: graph VOF diverged from eager, "
        f"max|d|={(tp_eager.alpha - tp_graph.alpha).abs().max().item():.3e}")
    # the graph must actually engage: one capture per parity, then pure replay
    assert runner.captures == 2 and runner.replays > 0 and runner.evictions == 0, (
        f"{ndim}-D {dtype}: VOF graph did not engage "
        f"(captures={runner.captures} replays={runner.replays} "
        f"eager={runner.eager} evictions={runner.evictions})")
    # alpha must be pointer-stable, else the key churns and we re-capture forever
    assert len(alpha_ptrs) == 1, (
        f"{ndim}-D {dtype}: alpha data_ptr churned ({len(alpha_ptrs)} distinct) "
        "-- advect_graph_aware must write alpha in place, never rebind it")


def test_cvof_graph_vs_eager_2d_f32():
    _cvof_graph_vs_eager(2, torch.float32)


def test_cvof_graph_vs_eager_2d_f64():
    _cvof_graph_vs_eager(2, torch.float64)


def test_cvof_graph_vs_eager_3d_f32():
    _cvof_graph_vs_eager(3, torch.float32)


def test_cvof_graph_vs_eager_3d_f64():
    _cvof_graph_vs_eager(3, torch.float64)
