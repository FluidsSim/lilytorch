"""Ghost-Fluid-Method (GFM) free-surface pressure solve (2-D), self-contained.

Imposes the free-surface Dirichlet BC ``p = 0`` at the *exact* (sub-cell)
interface location, so surface gravity waves are captured (unlike a staircase
``p=0``-in-air mask, which only gets statics right).

Method (Fedkiw/Gibou ghost-fluid):
  * a level-set ``phi`` (``phi<0`` water, ``phi>0`` air, ``phi=0`` at the
    surface) locates the interface; for a single-valued free surface we build it
    as a height function ``phi = y - h(x)``;
  * the face pressure gradient used by BOTH the Poisson operator and the
    velocity correction replaces an air cell's pressure by the GHOST value
    ``p_ghost = p_water * (1 - 1/theta)``, where ``theta = phi_water /
    (phi_water - phi_air)`` is the sub-cell distance (in cells) from the water
    cell centre to the interface. This places ``p=0`` exactly on the interface;
  * the discrete operator is ``A p = div(gfm_grad(p))`` so the projected
    velocity ``u - c*gfm_grad(p)`` is discretely divergence-free when ``A p =
    div(u*)/c``. Solved with Jacobi-preconditioned CG (air cells pinned to 0).

Constant (water) density: the coefficient ``c = dt/rho_water`` is a scalar.
"""

import torch

_TH_MIN = 0.05   # clamp theta away from 0 (cell adjacent to interface)


def level_set_height_2d(alpha, dy, y0):
    """Height-function level set for a single-valued free surface.

    ``alpha`` (1 water, 0 air), cell-centred. Water column height per x-column
    ``h(x) = y0 + dy * sum_y alpha``; ``phi[i,j] = y_j - h(x_i)`` (<0 water).
    """
    Nx, Ny = alpha.shape
    h_col = y0 + dy * alpha.sum(dim=1)                       # (Nx,)
    yj = y0 + (torch.arange(Ny, device=alpha.device, dtype=alpha.dtype) + 0.5) * dy
    return yj[None, :] - h_col[:, None]                     # (Nx, Ny)


def _ghost_faces_1d(p, phi, d, h):
    """GFM face gradient along axis ``d`` (interior faces), with air pressures
    replaced by the sub-cell ghost so p=0 sits on the interface."""
    def lo(a):  # cells 0..N-2  (left side of each interior face)
        return a.narrow(d, 0, a.shape[d] - 1)
    def hi(a):  # cells 1..N-1  (right side)
        return a.narrow(d, 1, a.shape[d] - 1)
    pL, pR = lo(p), hi(p)
    fL, fR = lo(phi), hi(phi)
    wL, wR = fL < 0, fR < 0
    denom = fL - fR
    # R air, L water: theta from L; pR -> ghost
    thL = torch.clamp(torch.where(denom != 0, fL / denom, torch.full_like(fL, 0.5)),
                      _TH_MIN, 1.0)
    pR_g = torch.where(wL & ~wR, pL * (1.0 - 1.0 / thL), pR)
    # L air, R water: theta from R; pL -> ghost
    denom2 = fR - fL
    thR = torch.clamp(torch.where(denom2 != 0, fR / denom2, torch.full_like(fR, 0.5)),
                      _TH_MIN, 1.0)
    pL_g = torch.where(wR & ~wL, pR * (1.0 - 1.0 / thR), pL)
    g = (pR_g - pL_g) / h
    return torch.where(wL | wR, g, torch.zeros_like(g))     # both air -> 0


def gfm_grad_2d(p, phi, h):
    """GFM pressure gradient on the MAC interior faces. Returns (gx, gy) where
    gx has shape (Nx-1, Ny) and gy (Nx, Ny-1)."""
    return _ghost_faces_1d(p, phi, 0, h), _ghost_faces_1d(p, phi, 1, h)


def _div_of_faces_2d(gx, gy, h):
    """Cell-centred divergence of interior face fluxes; 0 normal flux at walls."""
    Nx = gx.shape[0] + 1
    Ny = gy.shape[1] + 1
    div = torch.zeros((Nx, Ny), device=gx.device, dtype=gx.dtype)
    div[1:, :]  += gx / h
    div[:-1, :] -= gx / h
    div[:, 1:]  += gy / h
    div[:, :-1] -= gy / h
    return div


def _apply_A_2d(p, phi, h, water):
    gx, gy = gfm_grad_2d(p, phi, h)
    Ap = _div_of_faces_2d(gx, gy, h)
    return torch.where(water, Ap, torch.zeros_like(Ap))


def gfm_solve_cg_2d(rhs, phi, h, n_iter=400, tol=1e-7):
    """Solve ``div(gfm_grad(p)) = rhs`` in water (p=0 in air) by CG.

    ``rhs`` is the projection RHS ``div(u*)/c`` (already cell-centred, defined in
    water). Returns p with p=0 in the air. Symmetric-ish GFM operator -> CG.
    """
    water = phi < 0
    b = torch.where(water, rhs, torch.zeros_like(rhs))
    p = torch.zeros_like(rhs)
    r = b - _apply_A_2d(p, phi, h, water)
    r = torch.where(water, r, torch.zeros_like(r))
    z = r.clone()                       # (no preconditioner; could add Jacobi)
    pdir = z.clone()
    rz = (r * z).sum()
    b2 = (b * b).sum().clamp(min=1e-30)
    for _ in range(n_iter):
        Ap = _apply_A_2d(pdir, phi, h, water)
        denom = (pdir * Ap).sum()
        if denom.abs() < 1e-30:
            break
        a = rz / denom
        p = p + a * pdir
        r = r - a * Ap
        r = torch.where(water, r, torch.zeros_like(r))
        if (r * r).sum() / b2 < tol * tol:
            break
        rz_new = (r * r).sum()
        pdir = r + (rz_new / rz) * pdir
        rz = rz_new
    return torch.where(water, p, torch.zeros_like(p))
