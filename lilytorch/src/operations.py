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
