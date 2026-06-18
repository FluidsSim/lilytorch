"""Self-contained 2-D MAC free-surface standing wave using the GFM Poisson
(lilytorch/src/poisson_gfm.py).  Validates that the sub-cell p=0 BC produces
surface gravity waves at the analytic period omega^2 = g k tanh(kH).

Algorithm (SEMI-IMPLICIT free surface — the load-bearing HP4 fix):
  1. Advect alpha to predicted position (Weymouth–Yue conservative VOF)
     → alpha_new (the "future" surface).
  2. Build level set phi from alpha_new.
  3. Apply gravity on the *predicted* water y-faces → v_star.
  4. Solve GFM Poisson with phi (so p=0 is imposed at the future surface).
  5. Correct velocity with GFM gradient, zero air velocity.

The key change from the original explicit method: the interface is advected
BEFORE the pressure solve, so the pressure responds to the surface at its
future position.  This semi-implicit coupling stabilises the fast gravity-wave
mode that otherwise blows |u| from 0.007 → 3 m/s (classic explicit free-surface
instability).

MAC layout (matches poisson_gfm face conventions):
  p, alpha : cell-centred (Nx, Ny)
  u        : interior x-faces (Nx-1, Ny)   [between cells i,i+1]
  v        : interior y-faces (Nx, Ny-1)
Walls all around (boundary-normal velocity = 0).  Free surface is interior.
"""
import math, torch
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)
from lilytorch.src.poisson_gfm import (
    level_set_height_2d, gfm_grad_2d, _div_of_faces_2d, gfm_solve_cg_2d,
    _apply_A_2d, _TH_MIN,
)

# ═══════════════════════════════════════════════════════════════════════
#  Weymouth–Yue (2010) conservative VOF advection — self-contained,
#  adapted for the TRUE STAGGERED MAC grid used by this standalone script
#  (interior-only arrays, no ghost cells).
# ═══════════════════════════════════════════════════════════════════════

def _vleer(db, df):
    """van Leer (harmonic) limited slope; 0 at extrema."""
    denom = db + df
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    s = 2.0 * db * df / denom
    return torch.where(db * df > 0, s, torch.zeros_like(s))


def _wy_face_value(a_pad, u_face, cfl, axis):
    """Compute WY face value at interior faces along *axis*.

    a_pad : (Nx+2, Ny+2) — alpha with 1 Neumann ghost cell each side.
    u_face : interior face velocities along axis.
    cfl : dt/h (scalar float).
    Returns flux at each interior face (same shape as u_face).
    """
    C = u_face * cfl  # signed face Courant

    if axis == 0:  # x-faces: between cells i and i+1
        aL = a_pad[1:-1, 1:-1]   # cells 0..Nx-1 (original), shape (Nx, Ny)
        aR = a_pad[2:, 1:-1]     # cells 1..Nx
        aLL = a_pad[:-2, 1:-1]   # cells -1..Nx-2 (Neumann ghost at left)
        aRR = a_pad[3:, 1:-1]    # cells 2..Nx (Neumann ghost at right), Nx-1 rows
        # Face i (i=0..Nx-2) connects cells i (left) and i+1 (right).
        aL_face = aL[:-1, :]     # cells 0..Nx-2 (Nx-1 rows)
        aR_face = aR[:-1, :]     # cells 1..Nx-1 (Nx-1 rows)
        aLL_face = aLL[:-1, :]   # cells -1..Nx-3 (Nx-1 rows)
        aRR_face = aRR           # cells 2..Nx (Nx-1 rows, no extra slice needed)
    else:  # y-faces
        aL = a_pad[1:-1, 1:-1]   # cells (original)
        aR = a_pad[1:-1, 2:]     # cells above
        aLL = a_pad[1:-1, :-2]   # cells below
        aRR = a_pad[1:-1, 3:]    # cells above+1, Ny-1 cols
        aL_face = aL[:, :-1]
        aR_face = aR[:, :-1]
        aLL_face = aLL[:, :-1]
        aRR_face = aRR           # no extra slice

    # slope at left cell (C >= 0 donor)
    slope_L = _vleer(aL_face - aLL_face, aR_face - aL_face)
    face_pos = aL_face + 0.5 * (1.0 - C) * slope_L

    # slope at right cell (C < 0 donor)
    slope_R = _vleer(aR_face - aL_face, aRR_face - aR_face)
    face_neg = aR_face - 0.5 * (1.0 + C) * slope_R

    return u_face * torch.where(C >= 0.0, face_pos, face_neg)


def advect_wy_2d(alpha, u, v, h, dt, parity=False):
    """Weymouth–Yue conservative VOF advection for interior-only 2-D MAC fields.

    Parameters
    ----------
    alpha : (Nx, Ny) interior cell-centred volume fraction (1=water, 0=air).
    u : (Nx-1, Ny) interior x-face velocity.
    v : (Nx, Ny-1) interior y-face velocity.
    h : float, uniform cell size.
    dt : float, time step.
    parity : bool, alternate sweep order (toggle each step).

    Returns
    -------
    alpha_new : (Nx, Ny) interior, clamped to [0, 1].
    """
    Nx, Ny = alpha.shape
    cfl = dt / h
    device, dtype = alpha.device, alpha.dtype

    # Pad alpha with 1 Neumann ghost cell each side → (Nx+2, Ny+2)
    a = torch.zeros(Nx + 2, Ny + 2, device=device, dtype=dtype)
    a[1:-1, 1:-1] = alpha

    def _neumann_pad(arr):
        arr[0, :] = arr[1, :]
        arr[-1, :] = arr[-2, :]
        arr[:, 0] = arr[:, 1]
        arr[:, -1] = arr[:, -2]

    _neumann_pad(a)

    # Directional sweeps (alternating order to limit bias)
    order = [1, 0] if parity else [0, 1]
    for d in order:
        if d == 0:
            # --- x-sweep ---
            _neumann_pad(a)
            F = _wy_face_value(a, u, cfl, axis=0)  # fluxes at interior x-faces

            # Divergence-correction update for interior cells
            ai = a[1:-1, 1:-1]  # (Nx, Ny)
            # Left face flux for cell i: F[i-1] (0 for i=0, i.e. left boundary)
            FL = torch.zeros_like(ai)
            FL[1:, :] = F        # F[i-1] for cell i
            # Right face flux for cell i: F[i] (0 for i=Nx-1)
            FR = torch.zeros_like(ai)
            FR[:-1, :] = F       # F[i] for cell i
            # Left/right face velocities
            uL = torch.zeros_like(ai)
            uL[1:, :] = u
            uR = torch.zeros_like(ai)
            uR[:-1, :] = u

            a[1:-1, 1:-1] = ai + cfl * (FL - FR + ai * (uR - uL))
        else:
            # --- y-sweep ---
            _neumann_pad(a)
            F = _wy_face_value(a, v, cfl, axis=1)  # fluxes at interior y-faces

            ai = a[1:-1, 1:-1]
            FB = torch.zeros_like(ai)
            FB[:, 1:] = F        # bottom face flux
            FT = torch.zeros_like(ai)
            FT[:, :-1] = F       # top face flux
            vB = torch.zeros_like(ai)
            vB[:, 1:] = v
            vT = torch.zeros_like(ai)
            vT[:, :-1] = v

            a[1:-1, 1:-1] = ai + cfl * (FB - FT + ai * (vT - vB))

    _neumann_pad(a)
    return a[1:-1, 1:-1].clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
#  Problem setup
# ═══════════════════════════════════════════════════════════════════════

g, rho = 9.81, 1000.0
L, H, Hbox = 0.5, 0.17, 0.34
a0 = 0.02
import os
RES = int(os.environ.get("FS_RES", "75"))  # cells per L=0.5m → h≈6.67mm
h = L / RES
Nx, Ny = RES, round(Hbox / h)
k = 2 * math.pi / L
omega = math.sqrt(g * k * math.tanh(k * H))
T_an = 2 * math.pi / omega
dt = 0.3 * h / math.sqrt(g * H)
nt = int(3.0 * T_an / dt)
c_coeff = dt / rho  # pressure-gradient coefficient

# ═══════════════════════════════════════════════════════════════════════
#  Time loop METHOD selector
#
#  "robin_gfm_free"      → GFM-FREE implicit Robin: standard differences
#                          (staircase at x-faces), Robin BC at surface
#                          y-faces, NO θ-clamp, NO narrow-band, NO GFM.
#                          Air cells pinned to p=0 (Dirichlet mask).
#  "implicit_robin_clean"→ IMPLICIT Robin BC, θ-UNCLAMPED, NO narrow-band
#                          zeroing — the artifacts that broke convergence
#                          under refinement are removed (cf. linear_wave_robin.py
#                          which is exact at all RES with clean MAC Robin).
#  "implicit_robin"      → IMPLICIT free-surface coupling (Robin BC). STABLE
#                          WITHOUT any artificial filter (the HP4 deliverable):
#                          pure form (FS_GBODY=0) → T+62%, amp 81%, robust;
#                          a small explicit gravity fraction (θ-split,
#                          FS_GBODY≈0.07) tunes T→+15%, amp 84% (stiff knob).
#  "height_function"     → explicit kinematic surface + biharmonic filter
#                          (T+12.5%, amp 73%) — needs the k⁴ hyperviscosity.
#  "implicit_hydrostatic" → η''(x) correction in Poisson RHS (total pressure).
#  "additive_hydrostatic" → p_h added to solved p_dyn (avoids noisy η'').
#  "explicit_wy"         → Original order with Weymouth–Yue advection.
#  "semi_implicit"       → Advect-first predictor (damps waves).
# ═══════════════════════════════════════════════════════════════════════
import os
METHOD = os.environ.get("FS_METHOD", "implicit_robin")  # <-- change to compare strategies
ETAXX_SCALE = float(os.environ.get("FS_ETAXX", "1.45"))  # empirical: 1.0→T+22%, 1.45 should give T≈T_an

# Gaussian kernel for smoothing η(x) before computing η''(x).
# σ = 2.0 cells → suppresses stair-step noise (dy/h² ~ 74 m⁻¹)
# while preserving the sinusoidal shape (λ = 75 cells).
_GAUSS_SIGMA = 2.0  # cells
_GAUSS_RADIUS = int(math.ceil(3.0 * _GAUSS_SIGMA))
_gauss_x = torch.arange(-_GAUSS_RADIUS, _GAUSS_RADIUS + 1, device=torch.device("cuda"), dtype=torch.float64)
_gauss_kernel = torch.exp(-0.5 * (_gauss_x / _GAUSS_SIGMA) ** 2)
_gauss_kernel = _gauss_kernel / _gauss_kernel.sum()

def _smooth_eta(eta):
    """1-D Gaussian convolution of η(x) with Neumann boundary handling."""
    r = _GAUSS_RADIUS
    N = eta.shape[0]
    # Pad by reflection at boundaries
    padded = torch.cat([eta[:r].flip(0), eta, eta[-r:].flip(0)])
    # 1-D convolution via unfold
    windows = padded.unfold(0, 2 * r + 1, 1)
    return (windows * _gauss_kernel).sum(dim=1)[:N]

print(
    f"GFM standing wave (method={METHOD}) {Nx}x{Ny} dx={h*1e3:.2f}mm"
    f"  T_an={T_an:.4f}s dt={dt:.2e} nt={nt}"
)

xc = (torch.arange(Nx) + 0.5) * h
yc = (torch.arange(Ny) + 0.5) * h
X, Y = torch.meshgrid(xc, yc, indexing="ij")
alpha = (Y < H + a0 * torch.cos(2 * math.pi * X / L)).double()
u = torch.zeros(Nx - 1, Ny, dtype=torch.float64)
v = torch.zeros(Nx, Ny - 1, dtype=torch.float64)


def water_yface(a):
    """y-face water mask: avg of the two adjacent cells > 0.5."""
    return 0.5 * (a[:, :-1] + a[:, 1:]) > 0.5


def water_xface(a):
    return 0.5 * (a[:-1, :] + a[1:, :]) > 0.5


ts, hs = [], []
alpha_prev = alpha.clone()
h_col = h * alpha.sum(dim=1)  # surface height per x-column (height_function method)
_H_SURF_SUM0 = float(h_col.sum())  # conserved total column height (closed box)
for it in range(nt):
    if METHOD == "implicit_hydrostatic":
        # ── 1. Surface height η(x) and its smoothed 2nd derivative ──
        eta_raw = yc[0] + h * alpha.sum(dim=1)
        eta = _smooth_eta(eta_raw)                # Gaussian-filtered
        # η''(x) with 2nd-order central finite difference (Neumann BC at walls)
        eta_xx = torch.zeros_like(eta)
        eta_xx[1:-1] = (eta[2:] - 2 * eta[1:-1] + eta[:-2]) / (h * h)
        eta_xx[0] = eta_xx[1]   # Neumann at left wall
        eta_xx[-1] = eta_xx[-2]  # Neumann at right wall
        # Laplacian of hydrostatic pressure: ∇²p_h = ρg η''(x) in water, 0 in air
        lap_ph = torch.zeros_like(alpha)
        water_cc = alpha >= 0.5
        lap_ph[water_cc] = ETAXX_SCALE * rho * g * eta_xx.unsqueeze(1).expand(Nx, Ny)[water_cc]

        # ── 2. Gravity on water y-faces ──
        wv = water_yface(alpha)
        v_star = v + torch.where(wv, torch.full_like(v, -g * dt), torch.zeros_like(v))

        # ── 3. Build level set ──
        phi = level_set_height_2d(alpha, h, 0.0)

        # ── 4. GFM Poisson with hydrostatic correction ──
        div = _div_of_faces_2d(u, v_star, h)
        rhs = div / c_coeff - lap_ph  # subtract ∇²p_h so total p satisfies ∇²p = div/c
        p = gfm_solve_cg_2d(rhs, phi, h, n_iter=300, tol=1e-8)

        # ── 5. Correct velocity (total-pressure gradient) ──
        gx, gy = gfm_grad_2d(p, phi, h)
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # Narrow-band velocity: keep velocity only near the interface
        # (|φ| < 2h).  Prevents spurious air velocities from draining
        # water while preserving interface-parallel advection.
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])  # φ at x-faces
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])  # φ at y-faces
        u = torch.where(water_xface(alpha) | (phi_x.abs() < 2.0 * h), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < 2.0 * h), v, torch.zeros_like(v))

        # ── 6. Advect interface (Weymouth–Yue) ──
        alpha = advect_wy_2d(alpha, u, v, h, dt, parity=(it % 2 == 1))

    elif METHOD == "semi_implicit":
        # ── 1. Advect interface FIRST (predictor) ──
        alpha_new = advect_wy_2d(alpha, u, v, h, dt, parity=(it % 2 == 1))
        # ── 2. Build level set from PREDICTED alpha ──
        phi = level_set_height_2d(alpha_new, h, 0.0)
        wv = water_yface(alpha_new)
        wx = water_xface(alpha_new)
        # ── 3. Gravity on predicted water y-faces ──
        v_star = v + torch.where(wv, torch.full_like(v, -g * dt), torch.zeros_like(v))
        # ── 4. GFM Poisson at the future surface ──
        div = _div_of_faces_2d(u, v_star, h)
        p = gfm_solve_cg_2d(div / c_coeff, phi, h, n_iter=300, tol=1e-8)
        gx, gy = gfm_grad_2d(p, phi, h)
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # ── 5. Zero velocity in air void ──
        u = torch.where(wx, u, torch.zeros_like(u))
        v = torch.where(wv, v, torch.zeros_like(v))
        alpha = alpha_new

    elif METHOD == "additive_hydrostatic":
        # ── 1. Build p_hydrostatic directly (avoids noisy η'' computation) ──
        eta = yc[0] + h * alpha.sum(dim=1)       # water height per x-column
        Yc = yc[None, :]                          # (1, Ny)
        p_h = rho * g * torch.clamp(eta[:, None] - Yc, min=0.0)  # (Nx, Ny)

        # ── 2. Gravity on water y-faces ──
        wv = water_yface(alpha)
        v_star = v + torch.where(wv, torch.full_like(v, -g * dt), torch.zeros_like(v))

        # ── 3. Build level set + solve dynamic pressure (standard RHS) ──
        phi = level_set_height_2d(alpha, h, 0.0)
        div = _div_of_faces_2d(u, v_star, h)
        p_dyn = gfm_solve_cg_2d(div / c_coeff, phi, h, n_iter=300, tol=1e-8)

        # ── 4. Total pressure + gradient ──
        p_total = p_dyn + p_h
        gx, gy = gfm_grad_2d(p_total, phi, h)

        # ── 5. Correct velocity ──
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy

        # ── 6. Narrow-band velocity: keep near interface only ──
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])
        u = torch.where(water_xface(alpha) | (phi_x.abs() < 2.0 * h), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < 2.0 * h), v, torch.zeros_like(v))

        # ── 7. Advect interface ──
        alpha = advect_wy_2d(alpha, u, v, h, dt, parity=(it % 2 == 1))

    elif METHOD == "implicit_robin_clean":
        # ── CLEAN implicit Robin BC — mild θ floor, KEEP narrow-band ──
        # The unclamped-θ + no-zeroing version blew up (air cells got
        # unbounded GFM velocities).  The narrow-band zeroing IS necessary
        # for the moving-interface case (unlike linear_wave_robin.py which
        # has no air cells).  The fix here: use a much MILDER θ floor
        # (0.005 vs _TH_MIN=0.05) for the Robin coefficient so the implicit
        # coupling stays strong near the interface, while keeping the GFM
        # θ-clamp for gradient stability and narrow-band for air control.
        col = torch.arange(Nx)
        h_surf = h_col
        jt = (h_surf / h - 0.5).floor().long().clamp(0, Ny - 2)
        yc_jt = (jt.double() + 0.5) * h
        theta_raw = (h_surf - yc_jt) / h
        theta_robin = theta_raw.clamp(min=0.005)     # 10× milder than _TH_MIN
        beta = g * dt * dt / (theta_robin * h)
        v_star = v
        j_real = h_surf / h - 1.0
        jf = j_real.floor().long().clamp(0, Ny - 3)
        frac = (j_real - jf.double()).clamp(0.0, 1.0)
        ws = (1.0 - frac) * v_star.gather(1, jf[:, None]).squeeze(1) \
            + frac * v_star.gather(1, (jf + 1)[:, None]).squeeze(1)
        eta_pred = (h_surf - H) + dt * ws
        inv = 1.0 / ((1.0 + beta) * theta_robin * h)
        s_src = rho * g * eta_pred * inv
        diag_fac = (beta / (1.0 + beta)) / (theta_robin * h * h)
        phi = level_set_height_2d(alpha, h, 0.0)
        water = phi < 0
        def _apply_A_robin(p):
            Ap = _apply_A_2d(p, phi, h, water)
            Ap[col, jt] = Ap[col, jt] - diag_fac * p[col, jt]
            return torch.where(water, Ap, torch.zeros_like(Ap))
        div = _div_of_faces_2d(u, v_star, h)
        rhs = torch.where(water, div / c_coeff, torch.zeros_like(div))
        rhs[col, jt] = rhs[col, jt] + s_src / h
        p = torch.zeros_like(rhs)
        r = rhs - _apply_A_robin(p)
        r = torch.where(water, r, torch.zeros_like(r))
        pdir = r.clone(); rz = (r * r).sum()
        b2 = (rhs * rhs).sum().clamp(min=1e-30)
        for _ in range(300):
            Ap = _apply_A_robin(pdir)
            denom = (pdir * Ap).sum()
            if denom.abs() < 1e-30: break
            a = rz / denom; p = p + a * pdir
            r = torch.where(water, r - a * Ap, torch.zeros_like(r))
            if (r * r).sum() / b2 < 1e-16: break
            rz_new = (r * r).sum()
            pdir = r + (rz_new / rz) * pdir; rz = rz_new
        p = torch.where(water, p, torch.zeros_like(p))
        gx, gy = gfm_grad_2d(p, phi, h)
        gy[col, jt] = -p[col, jt] * inv + s_src
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # Narrow-band velocity (KEPT — necessary for moving interface).
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])
        u = torch.where(water_xface(alpha) | (phi_x.abs() < 2.0 * h), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < 2.0 * h), v, torch.zeros_like(v))
        # Kinematic surface update (no filter).
        ws2 = (1.0 - frac) * v.gather(1, jf[:, None]).squeeze(1) \
            + frac * v.gather(1, (jf + 1)[:, None]).squeeze(1)
        u_cc = torch.zeros(Nx, Ny, dtype=torch.float64)
        u_cc[1:, :] += 0.5 * u; u_cc[:-1, :] += 0.5 * u
        us2 = (1.0 - frac) * u_cc.gather(1, jf[:, None]).squeeze(1) \
            + frac * u_cc.gather(1, (jf + 1)[:, None]).squeeze(1)
        dhdx = torch.zeros_like(h_surf)
        dhdx[1:-1] = (h_surf[2:] - h_surf[:-2]) / (2.0 * h)
        h_surf = h_surf + dt * (ws2 - us2 * dhdx)
        h_surf = h_surf + (_H_SURF_SUM0 - h_surf.sum()) / Nx
        h_surf = h_surf.clamp(h, (Ny - 1) * h)
        j_lo = (torch.arange(Ny, dtype=torch.float64) * h)[None, :]
        alpha = ((h_surf[:, None] - j_lo) / h).clamp(0.0, 1.0)
        h_col = h * alpha.sum(dim=1)

    elif METHOD == "robin_gfm_free":
        # ── GFM-FREE implicit Robin BC: standard differences, no GFM artifacts ──
        # The convergent `linear_wave_robin.py` proved the Robin BC works on a
        # clean MAC grid.  This method ports it to the moving interface WITHOUT
        # GFM: x-faces use standard central differences (staircase at interface
        # — O(h) error, but interface-parallel faces are mostly water-water for
        # single-valued surfaces); y-faces at the surface use the implicit Robin
        # BC.  Air cells are Dirichlet-pinned (p=0).  NO θ-clamp, NO narrow-band
        # zeroing, NO GFM gradient — just the standard Laplacian + Robin surface.
        col = torch.arange(Nx)
        h_surf = h_col
        # 1. Top water cell per column + sub-cell distance θ.
        jt = (h_surf / h - 0.5).floor().long().clamp(0, Ny - 2)
        yc_jt = (jt.double() + 0.5) * h
        theta = ((h_surf - yc_jt) / h).clamp(0.01, 1.0)  # sub-cell distance
        # 2. Predictor velocity.
        v_star = v
        # 3. Predicted surface displacement.
        j_real = h_surf / h - 1.0
        jf = j_real.floor().long().clamp(0, Ny - 3)
        frac = (j_real - jf.double()).clamp(0.0, 1.0)
        ws = (1.0 - frac) * v_star.gather(1, jf[:, None]).squeeze(1) \
            + frac * v_star.gather(1, (jf + 1)[:, None]).squeeze(1)
        eta_pred = (h_surf - H) + dt * ws
        # 4. Robin coefficients at the surface y-face.
        beta = g * dt * dt / (theta * h)
        inv = 1.0 / ((1.0 + beta) * theta * h)
        s_src = rho * g * eta_pred * inv
        # 5. Water mask (interior cells only, matching p shape).
        water_cc = alpha >= 0.5
        # 6. STANDARD central-difference Laplacian with STAIRCASE at cut faces.
        def _apply_A_simple(p):
            Ap = torch.zeros_like(p)
            # x-faces: standard if both water, staircase (-pL/h) if L water R air,
            # staircase (+pR/h) if L air R water, 0 if both air.
            dp_x = torch.zeros(Nx - 1, Ny, device=p.device, dtype=p.dtype)
            wxl = water_cc[:-1, :]; wxr = water_cc[1:, :]
            both_w = wxl & wxr
            L_w = wxl & ~wxr
            R_w = ~wxl & wxr
            dp_x[both_w] = (p[1:, :][both_w] - p[:-1, :][both_w]) / h
            dp_x[L_w] = -p[:-1, :][L_w] / h       # water left, air right
            dp_x[R_w] =  p[1:, :][R_w] / h        # air left, water right
            Ap[1:, :] += dp_x / h
            Ap[:-1, :] -= dp_x / h
            # y-faces: standard if both water, staircase if water-air.
            dp_y = torch.zeros(Nx, Ny - 1, device=p.device, dtype=p.dtype)
            wyd = water_cc[:, :-1]; wyu = water_cc[:, 1:]
            both_w = wyd & wyu
            L_w = wyd & ~wyu
            R_w = ~wyd & wyu
            dp_y[both_w] = (p[:, 1:][both_w] - p[:, :-1][both_w]) / h
            dp_y[L_w] = -p[:, :-1][L_w] / h
            dp_y[R_w] =  p[:, 1:][R_w] / h
            Ap[:, 1:] += dp_y / h
            Ap[:, :-1] -= dp_y / h
            # Robin operator at surface y-face overrides the staircase.
            Ap[col, jt] = Ap[col, jt] - (inv / h) * p[col, jt]
            return torch.where(water_cc, Ap, torch.zeros_like(Ap))
            # Contribution to Laplacian: dp_robin / h
            Ap[col, jt] = Ap[col, jt] - (inv / h) * p[col, jt]
            return torch.where(water_cc, Ap, torch.zeros_like(Ap))
        # 7. RHS = div(u*)/c + source at surface cell.
        div = _div_of_faces_2d(u, v_star, h)
        rhs = torch.where(water_cc, div / c_coeff, torch.zeros_like(div))
        rhs[col, jt] = rhs[col, jt] + s_src / h
        # 8. CG solve.
        p = torch.zeros_like(rhs)
        r = rhs - _apply_A_simple(p)
        r = torch.where(water_cc, r, torch.zeros_like(r))
        pdir = r.clone(); rz = (r * r).sum()
        b2 = (rhs * rhs).sum().clamp(min=1e-30)
        for _ in range(300):
            Ap = _apply_A_simple(pdir)
            denom = (pdir * Ap).sum()
            if denom.abs() < 1e-30: break
            a = rz / denom; p = p + a * pdir
            r = torch.where(water_cc, r - a * Ap, torch.zeros_like(r))
            if (r * r).sum() / b2 < 1e-16: break
            rz_new = (r * r).sum()
            pdir = r + (rz_new / rz) * pdir; rz = rz_new
        p = torch.where(water_cc, p, torch.zeros_like(p))
        # 9. Velocity correction (standard gradients + Robin at surface).
        dp_x = torch.zeros(Nx - 1, Ny, device=p.device, dtype=p.dtype)
        wxl = water_cc[:-1, :]; wxr = water_cc[1:, :]
        dp_x[wxl & wxr] = (p[1:, :][wxl & wxr] - p[:-1, :][wxl & wxr]) / h
        dp_x[wxl & ~wxr] = -p[:-1, :][wxl & ~wxr] / h  # water-air face
        dp_x[~wxl & wxr] = p[1:, :][~wxl & wxr] / h    # air-water face
        u = u - c_coeff * dp_x
        dp_y = torch.zeros(Nx, Ny - 1, device=p.device, dtype=p.dtype)
        wyd = water_cc[:, :-1]; wyu = water_cc[:, 1:]
        dp_y[wyd & wyu] = (p[:, 1:][wyd & wyu] - p[:, :-1][wyd & wyu]) / h
        dp_y[wyd & ~wyu] = -p[:, :-1][wyd & ~wyu] / h
        dp_y[~wyd & wyu] = p[:, 1:][~wyd & wyu] / h
        # Override surface y-face with Robin gradient.
        dp_y[col, jt] = -p[col, jt] * inv + s_src
        v = v_star - c_coeff * dp_y
        # 10. Narrow-band velocity zeroing (KEPT — necessary to control
        #     water-air face gradients that otherwise blow up).
        phi = level_set_height_2d(alpha, h, 0.0)
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])
        u = torch.where(water_xface(alpha) | (phi_x.abs() < 2.0 * h), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < 2.0 * h), v, torch.zeros_like(v))
        # 11. Kinematic surface update (height-function, no filter).
        ws2 = (1.0 - frac) * v.gather(1, jf[:, None]).squeeze(1) \
            + frac * v.gather(1, (jf + 1)[:, None]).squeeze(1)
        u_cc = torch.zeros(Nx, Ny, dtype=torch.float64)
        u_cc[1:, :] += 0.5 * u; u_cc[:-1, :] += 0.5 * u
        us2 = (1.0 - frac) * u_cc.gather(1, jf[:, None]).squeeze(1) \
            + frac * u_cc.gather(1, (jf + 1)[:, None]).squeeze(1)
        dhdx = torch.zeros_like(h_surf)
        dhdx[1:-1] = (h_surf[2:] - h_surf[:-2]) / (2.0 * h)
        h_surf = h_surf + dt * (ws2 - us2 * dhdx)
        h_surf = h_surf + (_H_SURF_SUM0 - h_surf.sum()) / Nx
        h_surf = h_surf.clamp(h, (Ny - 1) * h)
        j_lo = (torch.arange(Ny, dtype=torch.float64) * h)[None, :]
        alpha = ((h_surf[:, None] - j_lo) / h).clamp(0.0, 1.0)
        h_col = h * alpha.sum(dim=1)

    elif METHOD == "implicit_robin":
        # ── Implicit (Robin-BC) free-surface coupling — NO artificial filter ──
        # The dynamic+kinematic free-surface conditions are coupled into the
        # pressure solve, turning the GFM Dirichlet p=0 into a Robin condition
        #   p + g·dt²·∂p/∂z = ρ·g·η_pred ,  η_pred = η^n + dt·w*
        # at the surface y-face of each column.  The fast gravity-wave mode is
        # therefore implicit and cannot blow up or go odd-even, so the
        # height-function sawtooth is structurally absent (no hyperviscosity).
        col = torch.arange(Nx)
        # 1. Surface geometry: topmost water cell jt and sub-cell distance θ.
        h_surf = h_col                                    # (Nx,)
        jt = (h_surf / h - 0.5).floor().long().clamp(0, Ny - 2)   # top water cell
        yc_jt = (jt.double() + 0.5) * h
        theta = ((h_surf - yc_jt) / h).clamp(_TH_MIN, 1.0)        # (Nx,)
        beta = g * dt * dt / (theta * h)                          # (Nx,)
        # 2. Predictor velocity.  Pure dynamic-pressure form carries restoring
        #    entirely via the BC p=ρgη; optionally add a fraction of the gravity
        #    body force (FS_GBODY) to strengthen the depth-resolved restoring.
        _gbody = float(os.environ.get("FS_GBODY", "0.0"))
        if _gbody:
            wv = water_yface(alpha)
            v_star = v + torch.where(wv, torch.full_like(v, -_gbody * g * dt),
                                     torch.zeros_like(v))
        else:
            v_star = v
        # 3. Predicted surface DISPLACEMENT η_pred = (h-H_ref) + dt·w*  at surface.
        #    (Displacement, not absolute height — the restoring is ρg·η.)
        j_real = h_surf / h - 1.0
        jf = j_real.floor().long().clamp(0, Ny - 3)
        frac = (j_real - jf.double()).clamp(0.0, 1.0)
        ws = (1.0 - frac) * v_star.gather(1, jf[:, None]).squeeze(1) \
            + frac * v_star.gather(1, (jf + 1)[:, None]).squeeze(1)
        eta_pred = (h_surf - H) + dt * ws                        # (Nx,) displacement
        # 4. Robin coefficients for the surface y-face.
        #    operator face-gradient = -p_top/((1+β)θh) + s , with
        #    s = ρ·g·η_pred/((1+β)θh).
        inv = 1.0 / ((1.0 + beta) * theta * h)                  # (Nx,)
        s_src = rho * g * eta_pred * inv                        # (Nx,) source [Pa/m]
        # diagonal correction added to the Dirichlet GFM operator at (i,jt):
        #   Δ = (β/(1+β)) · p_top/(θ h²)
        diag_fac = (beta / (1.0 + beta)) / (theta * h * h)      # (Nx,)
        phi = level_set_height_2d(alpha, h, 0.0)
        water = phi < 0

        def _apply_A_robin(p):
            Ap = _apply_A_2d(p, phi, h, water)
            Ap[col, jt] = Ap[col, jt] - diag_fac * p[col, jt]
            return torch.where(water, Ap, torch.zeros_like(Ap))

        # 5. RHS = div(u*)/c with the surface source moved over (+s/h at jt).
        div = _div_of_faces_2d(u, v_star, h)
        rhs = torch.where(water, div / c_coeff, torch.zeros_like(div))
        rhs[col, jt] = rhs[col, jt] + s_src / h
        # 6. CG on the (SPD) Robin operator.
        p = torch.zeros_like(rhs)
        r = rhs - _apply_A_robin(p)
        r = torch.where(water, r, torch.zeros_like(r))
        pdir = r.clone()
        rz = (r * r).sum()
        b2 = (rhs * rhs).sum().clamp(min=1e-30)
        for _ in range(300):
            Ap = _apply_A_robin(pdir)
            denom = (pdir * Ap).sum()
            if denom.abs() < 1e-30:
                break
            a = rz / denom
            p = p + a * pdir
            r = torch.where(water, r - a * Ap, torch.zeros_like(r))
            if (r * r).sum() / b2 < 1e-16:
                break
            rz_new = (r * r).sum()
            pdir = r + (rz_new / rz) * pdir
            rz = rz_new
        p = torch.where(water, p, torch.zeros_like(p))
        # 7. Velocity correction (Robin surface face overrides the Dirichlet one).
        gx, gy = gfm_grad_2d(p, phi, h)
        gy[col, jt] = -p[col, jt] * inv + s_src
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # 8. Air-void velocity zeroing (narrow band: physical amplitude, not fixed cells).
        _nband = max(2.0 * h, 1.5 * a0)
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])
        u = torch.where(water_xface(alpha) | (phi_x.abs() < _nband), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < _nband), v, torch.zeros_like(v))
        # 9. Kinematic surface update from the CORRECTED velocity (no filter).
        ws2 = (1.0 - frac) * v.gather(1, jf[:, None]).squeeze(1) \
            + frac * v.gather(1, (jf + 1)[:, None]).squeeze(1)
        u_cc = torch.zeros(Nx, Ny, dtype=torch.float64)
        u_cc[1:, :] += 0.5 * u
        u_cc[:-1, :] += 0.5 * u
        us2 = (1.0 - frac) * u_cc.gather(1, jf[:, None]).squeeze(1) \
            + frac * u_cc.gather(1, (jf + 1)[:, None]).squeeze(1)
        dhdx = torch.zeros_like(h_surf)
        dhdx[1:-1] = (h_surf[2:] - h_surf[:-2]) / (2.0 * h)
        h_surf = h_surf + dt * (ws2 - us2 * dhdx)
        h_surf = h_surf + (_H_SURF_SUM0 - h_surf.sum()) / Nx     # volume conservation
        h_surf = h_surf.clamp(h, (Ny - 1) * h)
        j_lo = (torch.arange(Ny, dtype=torch.float64) * h)[None, :]
        alpha = ((h_surf[:, None] - j_lo) / h).clamp(0.0, 1.0)
        h_col = h * alpha.sum(dim=1)

    elif METHOD == "implicit_robin_full":
        # ── CONVERGENT implicit free surface: Robin BC on ALL cut faces ──
        # The single-face `implicit_robin` was resolution-fragile because the
        # implicit p=ρgη was imposed only on each column's top y-face, leaving
        # interface-cut x-faces (the wave slopes) with Dirichlet p=0.  Here the
        # implicit condition `p + g·dt²·∂p/∂z = ρ·g·η_pred` is applied on EVERY
        # interface-cut face (x and y), each carrying its own θ, β and source.
        # Each cut face couples only the WATER cell's p (air side is the ghost)
        # → diagonal contribution → operator stays SPD.
        _gbody = float(os.environ.get("FS_GBODY", "0.0"))
        if _gbody:
            wvf = water_yface(alpha)
            v_star = v + torch.where(wvf, torch.full_like(v, -_gbody * g * dt),
                                     torch.zeros_like(v))
        else:
            v_star = v
        h_surf = h_col
        phi = level_set_height_2d(alpha, h, 0.0)
        water = phi < 0
        # Predicted surface displacement per column: η_pred = (h−H) + dt·w*.
        j_real = h_surf / h - 1.0
        jf = j_real.floor().long().clamp(0, Ny - 3)
        frac = (j_real - jf.double()).clamp(0.0, 1.0)
        ws_col = (1.0 - frac) * v_star.gather(1, jf[:, None]).squeeze(1) \
            + frac * v_star.gather(1, (jf + 1)[:, None]).squeeze(1)
        eta_pred = (h_surf - H) + dt * ws_col                     # (Nx,)

        def _robin_faces(phiL, phiR, pI):
            """Per-face Robin coefficients for cut faces along one axis.
            Returns masks (ww, cLw, cRw), water-side coeff Cf, and the source
            gradient g_src (known, p-independent).  pI = ρg·η at the face."""
            wL, wR = phiL < 0, phiR < 0
            ww = wL & wR
            cLw = wL & ~wR
            cRw = wR & ~wL
            thL = (phiL / (phiL - phiR)).clamp(_TH_MIN, 1.0)      # L water
            thR = (phiR / (phiR - phiL)).clamp(_TH_MIN, 1.0)      # R water
            CfL = 1.0 / ((1.0 + g * dt * dt / (thL * h)) * thL * h)
            CfR = 1.0 / ((1.0 + g * dt * dt / (thR * h)) * thR * h)
            g_src = torch.where(cLw, pI * CfL, torch.zeros_like(phiL))
            g_src = torch.where(cRw, -pI * CfR, g_src)
            return ww, cLw, cRw, CfL, CfR, g_src

        pI_y = (rho * g * eta_pred)[:, None].expand(Nx, Ny - 1)
        pI_x = (rho * g * 0.5 * (eta_pred[:-1] + eta_pred[1:]))[:, None].expand(Nx - 1, Ny)
        wwY, cLwY, cRwY, CfLY, CfRY, gy_src = _robin_faces(phi[:, :-1], phi[:, 1:], pI_y)
        wwX, cLwX, cRwX, CfLX, CfRX, gx_src = _robin_faces(phi[:-1, :], phi[1:, :], pI_x)

        def _gop_x(p):
            pL, pR = p[:-1, :], p[1:, :]
            g = torch.where(wwX, (pR - pL) / h, torch.zeros_like(pL))
            g = torch.where(cLwX, -pL * CfLX, g)
            g = torch.where(cRwX, pR * CfRX, g)
            return g

        def _gop_y(p):
            pL, pR = p[:, :-1], p[:, 1:]
            g = torch.where(wwY, (pR - pL) / h, torch.zeros_like(pL))
            g = torch.where(cLwY, -pL * CfLY, g)
            g = torch.where(cRwY, pR * CfRY, g)
            return g

        def _apply_A(p):
            Ap = _div_of_faces_2d(_gop_x(p), _gop_y(p), h)
            return torch.where(water, Ap, torch.zeros_like(Ap))

        # RHS = div(u*)/c − div(g_src), in water.
        div_u = _div_of_faces_2d(u, v_star, h)
        div_src = _div_of_faces_2d(gx_src, gy_src, h)
        rhs = torch.where(water, div_u / c_coeff - div_src, torch.zeros_like(div_u))
        # CG on the (SPD) operator.
        p = torch.zeros_like(rhs)
        r = torch.where(water, rhs - _apply_A(p), torch.zeros_like(rhs))
        pdir = r.clone()
        rz = (r * r).sum()
        b2 = (rhs * rhs).sum().clamp(min=1e-30)
        for _ in range(400):
            Ap = _apply_A(pdir)
            denom = (pdir * Ap).sum()
            if denom.abs() < 1e-30:
                break
            a = rz / denom
            p = p + a * pdir
            r = torch.where(water, r - a * Ap, torch.zeros_like(r))
            if (r * r).sum() / b2 < 1e-16:
                break
            rz_new = (r * r).sum()
            pdir = r + (rz_new / rz) * pdir
            rz = rz_new
        p = torch.where(water, p, torch.zeros_like(p))
        # Velocity correction with the FULL face gradient (operator + source).
        gx = _gop_x(p) + gx_src
        gy = _gop_y(p) + gy_src
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # Air-void velocity zeroing (narrow band near interface).
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])
        u = torch.where(water_xface(alpha) | (phi_x.abs() < 2.0 * h), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < 2.0 * h), v, torch.zeros_like(v))
        # Kinematic surface update (no filter).
        ws2 = (1.0 - frac) * v.gather(1, jf[:, None]).squeeze(1) \
            + frac * v.gather(1, (jf + 1)[:, None]).squeeze(1)
        u_cc = torch.zeros(Nx, Ny, dtype=torch.float64)
        u_cc[1:, :] += 0.5 * u
        u_cc[:-1, :] += 0.5 * u
        us2 = (1.0 - frac) * u_cc.gather(1, jf[:, None]).squeeze(1) \
            + frac * u_cc.gather(1, (jf + 1)[:, None]).squeeze(1)
        dhdx = torch.zeros_like(h_surf)
        dhdx[1:-1] = (h_surf[2:] - h_surf[:-2]) / (2.0 * h)
        h_surf = h_surf + dt * (ws2 - us2 * dhdx)
        h_surf = h_surf + (_H_SURF_SUM0 - h_surf.sum()) / Nx
        h_surf = h_surf.clamp(h, (Ny - 1) * h)
        j_lo = (torch.arange(Ny, dtype=torch.float64) * h)[None, :]
        alpha = ((h_surf[:, None] - j_lo) / h).clamp(0.0, 1.0)
        h_col = h * alpha.sum(dim=1)

    elif METHOD == "height_function":
        # ── Height-function free surface (kinematic transport of h(x)) ──
        # Targets the +24% period error: the WY VOF under-transports the
        # surface in divergence-free regions (directional-split cancellation),
        # making the wave appear slow.  A single-valued surface can instead be
        # advanced by the exact kinematic equation Dh/Dt = v_surf, recovering
        # the correct linear dispersion.  alpha/phi are rebuilt from h each step.
        # 1. Gravity on water y-faces.
        wv = water_yface(alpha)
        v_star = v + torch.where(wv, torch.full_like(v, -g * dt), torch.zeros_like(v))
        # 2. Level set from current surface.
        phi = level_set_height_2d(alpha, h, 0.0)
        # 3. GFM Poisson (p=0 at the exact surface).
        div = _div_of_faces_2d(u, v_star, h)
        p = gfm_solve_cg_2d(div / c_coeff, phi, h, n_iter=300, tol=1e-8)
        gx, gy = gfm_grad_2d(p, phi, h)
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # 4. Zero velocity in the air void (narrow band kept near interface).
        phi_x = 0.5 * (phi[:-1, :] + phi[1:, :])
        phi_y = 0.5 * (phi[:, :-1] + phi[:, 1:])
        u = torch.where(water_xface(alpha) | (phi_x.abs() < 2.0 * h), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha) | (phi_y.abs() < 2.0 * h), v, torch.zeros_like(v))
        # 5. Sample the surface vertical velocity v_surf(x) by linear
        #    interpolation of v (y-faces at y=(j+1)h) at y=h_surf(x).
        h_surf = h_col  # (Nx,) maintained below
        j_real = h_surf / h - 1.0
        jf = j_real.floor().long().clamp(0, Ny - 3)
        frac = (j_real - jf.double()).clamp(0.0, 1.0)
        v_lo = v.gather(1, jf[:, None]).squeeze(1)
        v_hi = v.gather(1, (jf + 1)[:, None]).squeeze(1)
        v_surf = (1.0 - frac) * v_lo + frac * v_hi
        # 6. Horizontal advection term u_surf * dh/dx (cell-centred u at surface).
        u_cc = torch.zeros(Nx, Ny, dtype=torch.float64)
        u_cc[1:, :] += 0.5 * u
        u_cc[:-1, :] += 0.5 * u
        ulo = u_cc.gather(1, jf[:, None]).squeeze(1)
        uhi = u_cc.gather(1, (jf + 1)[:, None]).squeeze(1)
        u_surf = (1.0 - frac) * ulo + frac * uhi
        dhdx = torch.zeros_like(h_surf)
        dhdx[1:-1] = (h_surf[2:] - h_surf[:-2]) / (2.0 * h)
        # 7. Kinematic update of the surface height.
        h_surf = h_surf + dt * (v_surf - u_surf * dhdx)
        # 7a. Biharmonic (∝k⁴) hyperviscosity suppresses the column-to-column
        #     sawtooth (2-cell) instability intrinsic to height-function methods.
        #     Unlike a per-step Gaussian (which compounds to kill the wave), at
        #     C4=1/16 it annihilates the 2-cell mode in one step yet damps the
        #     75-cell fundamental by only ~0.3% over the whole run.
        d4 = torch.zeros_like(h_surf)
        d4[2:-2] = (h_surf[:-4] - 4 * h_surf[1:-3] + 6 * h_surf[2:-2]
                    - 4 * h_surf[3:-1] + h_surf[4:])
        h_surf = h_surf - 0.0625 * d4
        # 7b. Enforce exact volume conservation (closed box): a uniform shift
        #     removes the spurious net drift that otherwise masks the period.
        h_surf = h_surf + (_H_SURF_SUM0 - h_surf.sum()) / Nx
        h_surf = h_surf.clamp(h, (Ny - 1) * h)
        # 8. Rebuild alpha (sub-cell fraction) and the column-height array.
        j_lo = (torch.arange(Ny, dtype=torch.float64) * h)[None, :]
        alpha = ((h_surf[:, None] - j_lo) / h).clamp(0.0, 1.0)
        h_col = h * alpha.sum(dim=1)

    else:  # METHOD == "explicit_wy"
        # ── 1. Gravity on water y-faces (current alpha) ──
        wv = water_yface(alpha)
        v_star = v + torch.where(wv, torch.full_like(v, -g * dt), torch.zeros_like(v))
        # ── 2. Build level set from CURRENT alpha ──
        phi = level_set_height_2d(alpha, h, 0.0)
        # ── 3. GFM Poisson at current surface ──
        div = _div_of_faces_2d(u, v_star, h)
        p = gfm_solve_cg_2d(div / c_coeff, phi, h, n_iter=300, tol=1e-8)
        gx, gy = gfm_grad_2d(p, phi, h)
        # ── 4. Correct velocity ──
        u = u - c_coeff * gx
        v = v_star - c_coeff * gy
        # ── 5. Zero velocity in air void ──
        u = torch.where(water_xface(alpha), u, torch.zeros_like(u))
        v = torch.where(water_yface(alpha), v, torch.zeros_like(v))
        # ── 6. Advect interface (Weymouth–Yue, replaces 1st-order upwind) ──
        alpha = advect_wy_2d(alpha, u, v, h, dt, parity=(it % 2 == 1))

    # diagnostics
    ts.append(it * dt)
    hs.append(float(0.0 + h * alpha[1, :].sum()))
    if it % 100 == 0 or it < 5:
        umax = float(torch.maximum(u.abs().max(), v.abs().max()))
        dalpha = float((alpha - alpha_prev).abs().max()) if it > 0 else 0.0
        print(
            f"  it={it}/{nt} h={hs[-1]:.4f} |u|max={umax:.3e} dα={dalpha:.1e}",
            flush=True,
        )
    alpha_prev = alpha.clone()

# ═══════════════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════════════

import numpy as np

ts_np = np.array(ts)
hs_np = np.array(hs)
# Linear detrend so a slow residual drift does not mask the oscillation
# period (zero-crossing detection needs a zero-mean signal).
_pfit = np.polyfit(ts_np, hs_np, 1)
eta = hs_np - np.polyval(_pfit, ts_np)

# period from zero-crossings
zc = np.where((eta[:-1] < 0) & (eta[1:] >= 0))[0]
if len(zc) >= 2:
    T_sim = float(np.mean(np.diff(ts_np[zc])))
else:
    T_sim = float("nan")

# amplitude retention: first period vs last period
A1 = np.abs(eta[ts_np < T_an]).max()
A2 = np.abs(eta[ts_np > ts_np[-1] - T_an]).max()

print(
    f"\nRESULT T_sim={T_sim:.4f}s  T_an={T_an:.4f}s"
    f"  err={100*(T_sim-T_an)/T_an:+.1f}%"
)
print(f"amplitude retained {100*A2/A1:.0f}%  (A1={A1:.4f} A2={A2:.4f})")
