"""Dimension-agnostic diffusion operators for MAC staggered grids.

Split out of the former monolithic ``adv_diff.py`` (since removed) so the
advection schemes (:mod:`lilytorch.src.advection`) and the diffusion closures
can grow and be ``torch.compile``-d independently.

These are **pure functions** — no class, no boundary conditions (the caller,
e.g. :class:`~lilytorch.src.advection.AdvDiffSolver`, owns the ghost layer via
``set_BCs``).  They return the *interior* diffusion increment, i.e. the value
to add to ``phi[inner]``.

This is the natural place for future diffusion-model expansions
(anisotropic / tensorial viscosity, additional non-Newtonian closures, etc.):
add a new operator here and a branch in :func:`diffuse`.
"""

import torch


def _inner(ndim):
    """Index tuple selecting interior cells: ``[1:-1]`` on every dimension.

    Duplicated (rather than imported from :mod:`advection`) to keep this
    module a dependency-free leaf.
    """
    return tuple(slice(1, -1) for _ in range(ndim))


def laplacian(phi, inv_dh2):
    """Constant-coefficient discrete Laplacian at interior cells (any dim).

    Parameters
    ----------
    phi     : tensor — field (interior + ghost cells).
    inv_dh2 : list of floats — ``1 / dh_d**2`` per dimension.
    """
    ndim  = phi.ndim
    inner = _inner(ndim)
    lap   = torch.zeros_like(phi[inner])
    for d in range(ndim):
        fwd = list(inner); fwd[d] = slice(2, None)
        bwd = list(inner); bwd[d] = slice(None, -2)
        lap += (phi[tuple(fwd)] - 2.0 * phi[inner] + phi[tuple(bwd)]) * inv_dh2[d]
    return lap


def variable_laplacian(phi, nu_eff, dh):
    """Variable-coefficient Laplacian ``div(nu_eff * grad(phi))``.

    Uses face-averaged viscosity::

        sum_d ( nu_{i+1/2}(phi_{i+1} - phi_i)
                - nu_{i-1/2}(phi_i - phi_{i-1}) ) / dh_d**2

    Parameters
    ----------
    phi    : tensor — the field (interior + ghost).
    nu_eff : tensor — effective viscosity at cell centres (same shape as phi).
    dh     : list of floats — grid spacing per dimension.
    """
    ndim  = phi.ndim
    inner = _inner(ndim)
    lap   = torch.zeros_like(phi[inner])
    for d in range(ndim):
        fwd = list(inner); fwd[d] = slice(2, None)
        bwd = list(inner); bwd[d] = slice(None, -2)
        # face-averaged viscosity
        nu_fwd = 0.5 * (nu_eff[inner] + nu_eff[tuple(fwd)])
        nu_bwd = 0.5 * (nu_eff[inner] + nu_eff[tuple(bwd)])
        inv_dh2 = 1.0 / (dh[d] * dh[d])
        lap += (nu_fwd * (phi[tuple(fwd)] - phi[inner])
                - nu_bwd * (phi[inner] - phi[tuple(bwd)])) * inv_dh2
    return lap


def diffuse(phi, dt, *, nu=None, nu_t=None, inv_dh2=None, dh=None):
    """Explicit forward-Euler diffusion increment over interior cells.

    Returns a **fresh** tensor (safe for in-place ``add_``) equal to:

    * ``nu * dt * laplacian(phi)``                    when ``nu_t is None``
      (constant viscosity), or
    * ``dt * variable_laplacian(phi, nu + nu_t)``     otherwise
      (Smagorinsky / variable-coefficient).

    Parameters mirror the fields cached on the solver: ``inv_dh2`` is
    required for the constant path, ``dh`` for the variable path.
    """
    if nu_t is None:
        return nu * dt * laplacian(phi, inv_dh2)
    nu_eff = nu + nu_t
    return dt * variable_laplacian(phi, nu_eff, dh)
