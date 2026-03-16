"""
Differential operators and derived quantities (gradient, divergence,
vorticity, …) used by the fluid solver.

Extracted from solver.py so they can be maintained independently.
All functions are standalone — they take explicit grid parameters
(h, ndim, dx, dy, dz) instead of relying on ``self``.
"""

import torch


# ------------------------------------------------------------------
# First-order partial derivatives
# ------------------------------------------------------------------
def compute_dpdx(p, h):
    """Compute dp/dx via central difference."""
    return torch.gradient(p, spacing=h, dim=0, edge_order=2)[0]


def compute_dpdy(p, h):
    """Compute dp/dy via central difference."""
    return torch.gradient(p, spacing=h, dim=1, edge_order=2)[0]


def compute_dpdz(p, h):
    """Compute dp/dz (3-D only)."""
    return torch.gradient(p, spacing=h, dim=2, edge_order=2)[0]


# ------------------------------------------------------------------
# Gradient
# ------------------------------------------------------------------
def gradient(var, h, ndim):
    """
    Compute gradient(var).
    2-D → (dvar_dx, dvar_dy),  3-D → (dvar_dx, dvar_dy, dvar_dz).
    """
    dvar_dx = torch.zeros_like(var)
    dvar_dy = torch.zeros_like(var)
    if ndim == 2:
        dvar_dx[1:-1, 1:-1] = (var[1:-1, 1:-1] - var[:-2, 1:-1]) / h
        dvar_dy[1:-1, 1:-1] = (var[1:-1, 1:-1] - var[1:-1, :-2]) / h
        return (dvar_dx, dvar_dy)
    else:
        dvar_dz = torch.zeros_like(var)
        dvar_dx[1:-1, 1:-1, 1:-1] = (var[1:-1, 1:-1, 1:-1] - var[:-2, 1:-1, 1:-1]) / h
        dvar_dy[1:-1, 1:-1, 1:-1] = (var[1:-1, 1:-1, 1:-1] - var[1:-1, :-2, 1:-1]) / h
        dvar_dz[1:-1, 1:-1, 1:-1] = (var[1:-1, 1:-1, 1:-1] - var[1:-1, 1:-1, :-2]) / h
        return (dvar_dx, dvar_dy, dvar_dz)


# ------------------------------------------------------------------
# Divergence
# ------------------------------------------------------------------
def divergence(u, v, dx, dy, w=None, dz=None):
    """Compute the divergence — 2-D: div(u,v), 3-D: div(u,v,w)."""
    div = torch.zeros_like(u)
    if w is None:
        div[1:-1, 1:-1] = ((u[2:, 1:-1] - u[1:-1, 1:-1]) / dx
                         + (v[1:-1, 2:] - v[1:-1, 1:-1]) / dy)
    else:
        div[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) / dx
          + (v[1:-1, 2:, 1:-1] - v[1:-1, 1:-1, 1:-1]) / dy
          + (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 1:-1]) / dz
        )
    return div


# ------------------------------------------------------------------
# Normal derivative
# ------------------------------------------------------------------
def normal_derivative(var, h, ndim, normal_x, normal_y, normal_z=None):
    """Compute the normal derivative dvar/dn = n · ∇var."""
    nd = normal_x * compute_dpdx(var, h) + normal_y * compute_dpdy(var, h)
    if ndim == 3 and normal_z is not None:
        nd = nd + normal_z * compute_dpdz(var, h)
    return nd


# ------------------------------------------------------------------
# Vorticity
# ------------------------------------------------------------------
def vorticity(u, v, h, ndim, w=None):
    """
    Compute vorticity.
    2-D: scalar  omega = dv/dx - du/dy
    3-D: magnitude |ω| = sqrt(ωx² + ωy² + ωz²)
    """
    if ndim == 2 or w is None:
        dvdx = torch.zeros_like(u)
        dudy = torch.zeros_like(u)
        dvdx[1:-1, 1:-1] = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / h
        dudy[1:-1, 1:-1] = (u[1:-1, 1:-1] - u[1:-1, :-2]) / h
        return dvdx - dudy
    else:
        # 3-D vorticity magnitude.
        # Start at index 2 so backward differences never reach into
        # ghost cells (index 0), which can have BC-inconsistent values
        # and produce spurious boundary vorticity.
        ox = torch.zeros_like(u)
        ox[2:-2, 2:-2, 2:-2] = (
            (w[2:-2, 2:-2, 2:-2] - w[2:-2, 1:-3, 2:-2]) / h -
            (v[2:-2, 2:-2, 2:-2] - v[2:-2, 2:-2, 1:-3]) / h
        )
        oy = torch.zeros_like(u)
        oy[2:-2, 2:-2, 2:-2] = (
            (u[2:-2, 2:-2, 2:-2] - u[2:-2, 2:-2, 1:-3]) / h -
            (w[2:-2, 2:-2, 2:-2] - w[1:-3, 2:-2, 2:-2]) / h
        )
        oz = torch.zeros_like(u)
        oz[2:-2, 2:-2, 2:-2] = (
            (v[2:-2, 2:-2, 2:-2] - v[1:-3, 2:-2, 2:-2]) / h -
            (u[2:-2, 2:-2, 2:-2] - u[2:-2, 1:-3, 2:-2]) / h
        )
        return torch.sqrt(ox**2 + oy**2 + oz**2)


def vorticity_components(u, v, w, h):
    """
    Return the three signed vorticity components (omega_x, omega_y, omega_z)
    as a dict, plus the magnitude.  Only meaningful in 3-D.
    """
    ox = torch.zeros_like(u)
    ox[2:-2, 2:-2, 2:-2] = (
        (w[2:-2, 2:-2, 2:-2] - w[2:-2, 1:-3, 2:-2]) / h -
        (v[2:-2, 2:-2, 2:-2] - v[2:-2, 2:-2, 1:-3]) / h
    )
    oy = torch.zeros_like(u)
    oy[2:-2, 2:-2, 2:-2] = (
        (u[2:-2, 2:-2, 2:-2] - u[2:-2, 2:-2, 1:-3]) / h -
        (w[2:-2, 2:-2, 2:-2] - w[1:-3, 2:-2, 2:-2]) / h
    )
    oz = torch.zeros_like(u)
    oz[2:-2, 2:-2, 2:-2] = (
        (v[2:-2, 2:-2, 2:-2] - v[1:-3, 2:-2, 2:-2]) / h -
        (u[2:-2, 2:-2, 2:-2] - u[2:-2, 1:-3, 2:-2]) / h
    )
    return {"omega_x": ox, "omega_y": oy, "omega_z": oz,
            "omega_mag": torch.sqrt(ox**2 + oy**2 + oz**2)}


# ------------------------------------------------------------------
# Cross products
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Strain-rate magnitude and Smagorinsky eddy viscosity
# ------------------------------------------------------------------
def strain_rate_magnitude(vel, h, ndim):
    """Compute |S̄| = sqrt(2 * S_ij * S_ij) on the cell-centred grid.

    Parameters
    ----------
    vel  : tuple of tensors (u, v) in 2-D or (u, v, w) in 3-D.
    h    : float — uniform grid spacing.
    ndim : int — 2 or 3.

    Returns
    -------
    |S̄| tensor with same shape as vel[0].  Ghost cells are zero.
    """
    if ndim == 2:
        u, v = vel
        dudx = torch.gradient(u, spacing=h, dim=0, edge_order=2)[0]
        dudy = torch.gradient(u, spacing=h, dim=1, edge_order=2)[0]
        dvdx = torch.gradient(v, spacing=h, dim=0, edge_order=2)[0]
        dvdy = torch.gradient(v, spacing=h, dim=1, edge_order=2)[0]
        # S_ij S_ij = S11² + S22² + 2*S12²
        # S11 = dudx, S22 = dvdy, S12 = 0.5*(dudy + dvdx)
        S2 = dudx**2 + dvdy**2 + 0.5 * (dudy + dvdx)**2
        return torch.sqrt(2.0 * S2)
    else:
        u, v, w = vel
        dudx = torch.gradient(u, spacing=h, dim=0, edge_order=2)[0]
        dudy = torch.gradient(u, spacing=h, dim=1, edge_order=2)[0]
        dudz = torch.gradient(u, spacing=h, dim=2, edge_order=2)[0]
        dvdx = torch.gradient(v, spacing=h, dim=0, edge_order=2)[0]
        dvdy = torch.gradient(v, spacing=h, dim=1, edge_order=2)[0]
        dvdz = torch.gradient(v, spacing=h, dim=2, edge_order=2)[0]
        dwdx = torch.gradient(w, spacing=h, dim=0, edge_order=2)[0]
        dwdy = torch.gradient(w, spacing=h, dim=1, edge_order=2)[0]
        dwdz = torch.gradient(w, spacing=h, dim=2, edge_order=2)[0]
        # S_ij S_ij = S11² + S22² + S33² + 2*(S12² + S13² + S23²)
        S2 = (dudx**2 + dvdy**2 + dwdz**2
              + 0.5 * (dudy + dvdx)**2
              + 0.5 * (dudz + dwdx)**2
              + 0.5 * (dvdz + dwdy)**2)
        return torch.sqrt(2.0 * S2)


def smagorinsky_viscosity(vel, h, ndim, cs=0.1):
    """Compute the Smagorinsky eddy viscosity  ν_t = (Cs·Δ)² |S̄|.

    Parameters
    ----------
    vel  : tuple of tensors — velocity components.
    h    : float — uniform grid spacing (used as filter width Δ).
    ndim : int — 2 or 3.
    cs   : float — Smagorinsky constant (typically 0.1–0.2).

    Returns
    -------
    ν_t tensor with same shape as vel[0].
    """
    S_mag = strain_rate_magnitude(vel, h, ndim)
    return (cs * h) ** 2 * S_mag


def carreau_viscosity(vel, h, ndim, nu_0, nu_inf, lam, n,
                     tau_y=0.0, rho=1000.0, nu_max=None):
    """Compute Carreau (or Herschel-Bulkley–Carreau) viscosity field.

    Without yield stress (tau_y = 0):
        ν(γ̇) = ν_∞ + (ν_0 − ν_∞) · [1 + (λ·γ̇)²]^((n−1)/2)

    With yield stress (tau_y > 0):
        ν(γ̇) = τ_y / (ρ · max(γ̇, γ̇_min)) + ν_∞ + (ν_0 − ν_∞) · [1 + (λ·γ̇)²]^((n−1)/2)

    The total viscosity is clamped to ``nu_max`` to guarantee diffusion CFL
    stability.  When ``nu_max`` is None no clamping is applied (pure Carreau
    without yield stress is bounded by ν_0 anyway).

    Parameters
    ----------
    vel    : tuple of tensors — velocity components (u, v) or (u, v, w).
    h      : float — uniform grid spacing.
    ndim   : int — 2 or 3.
    nu_0   : float — zero-shear-rate kinematic viscosity [m²/s].
    nu_inf : float — infinite-shear-rate kinematic viscosity [m²/s].
    lam    : float — relaxation time [s].
    n      : float — power-law index (< 1 for shear-thinning).
    tau_y  : float — yield stress [Pa]. Default 0 (pure Carreau).
    rho    : float — fluid density [kg/m³]. Only used when tau_y > 0.
    nu_max : float or None — hard upper bound on ν for CFL stability.

    Returns
    -------
    ν tensor (spatially varying kinematic viscosity) with same shape as vel[0].
    """
    S_mag = strain_rate_magnitude(vel, h, ndim)
    nu = nu_inf + (nu_0 - nu_inf) * (1.0 + (lam * S_mag) ** 2) ** ((n - 1.0) / 2.0)
    if tau_y > 0.0:
        gamma_dot_reg = torch.clamp(S_mag, min=1e-6)
        nu = nu + tau_y / (rho * gamma_dot_reg)
    if nu_max is not None:
        nu = torch.clamp(nu, max=nu_max)
    return nu


def cross_product_2d(ax, ay, bx, by):
    """Element-wise 2-D cross product (scalar result)."""
    return ax * by - ay * bx


def cross_product_3d(ax, ay, az, bx, by, bz):
    """Element-wise 3-D cross product  a × b.

    Returns (cx, cy, cz) tensors.
    """
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return cx, cy, cz
