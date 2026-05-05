"""Force-and-torque computation for immersed bodies.

Extracted from ``solver.py`` (item #8 of the HIGH PRIORITY backlog).

Module-level kernels (``_forces_shared_*`` / ``_forces_body_*``) are pure
tensor-in / tensor-out functions kept module-level so they can be wrapped
by :func:`torch.compile`.

The ``forces_method*`` functions take ``self`` (a ``FluidSolver``) as their
first argument and are bound to ``FluidSolver`` as methods at the bottom
of ``solver.py`` -- they are NOT directly callable from this module's
public API.  This keeps the public API of :class:`FluidSolver` unchanged
while moving ~1500 LoC of force code out of ``solver.py``.
"""
import torch

from lilytorch.src import operations as ops


# ======================================================================
# Compilable force-computation kernels  (module-level, for torch.compile)
# ======================================================================

def _forces_shared_3d(u, v, w, p, sdf_val, nx, ny, nz, nu_rho, h):
    """Compute velocity gradients, viscous stress·n, and pressure force density.

    All arguments are plain tensors or scalars — no ``self`` access — so this
    function is safe for ``torch.compile(mode='reduce-overhead')``.


    * **Diagonal** (∂u_i/∂x_i): natural compact stencil exploiting the
      MAC stagger — ``(u[I+δ(i),i] - u[I,i]) / h``, exact at CC.
    * **Cross** (∂u_i/∂x_j, i≠j): 4-point average to CC —
      equivalent to interpolating u_i to CC along its stagger dimension,
      then central-differencing along x_j.

    Stagger convention: u at (x−h/2,y,z), v at (x,y−h/2,z), w at (x,y,z−h/2).

    Returns
    -------
    xstress, ystress, zstress : viscous-stress × normal  (σ·n)_i
    pforce_x, pforce_y, pforce_z : -p * n_i
    """
    # ---- Diagonal derivatives: compact stencil (exact at CC) --------
    dudx = torch.empty_like(u)
    dudx[:-1, :, :] = (u[1:, :, :] - u[:-1, :, :]) / h
    dudx[-1, :, :]  = dudx[-2, :, :]

    dvdy = torch.empty_like(v)
    dvdy[:, :-1, :] = (v[:, 1:, :] - v[:, :-1, :]) / h
    dvdy[:, -1, :]  = dvdy[:, -2, :]

    dwdz = torch.empty_like(w)
    dwdz[:, :, :-1] = (w[:, :, 1:] - w[:, :, :-1]) / h
    dwdz[:, :, -1]  = dwdz[:, :, -2]

    # ---- Cross derivatives: interp to CC along stagger dim, ----------
    # u_cc: u interpolated to CC along dim 0
    u_cc = torch.empty_like(u)
    u_cc[:-1, :, :] = 0.5 * (u[:-1, :, :] + u[1:, :, :])
    u_cc[-1, :, :]  = u[-1, :, :]

    dudy = torch.gradient(u_cc, spacing=h, dim=1, edge_order=2)[0]
    dudz = torch.gradient(u_cc, spacing=h, dim=2, edge_order=2)[0]

    # v_cc: v interpolated to CC along dim 1
    v_cc = torch.empty_like(v)
    v_cc[:, :-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    v_cc[:, -1, :]  = v[:, -1, :]

    dvdx = torch.gradient(v_cc, spacing=h, dim=0, edge_order=2)[0]
    dvdz = torch.gradient(v_cc, spacing=h, dim=2, edge_order=2)[0]

    # w_cc: w interpolated to CC along dim 2
    w_cc = torch.empty_like(w)
    w_cc[:, :, :-1] = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    w_cc[:, :, -1]  = w[:, :, -1]

    dwdx = torch.gradient(w_cc, spacing=h, dim=0, edge_order=2)[0]
    dwdy = torch.gradient(w_cc, spacing=h, dim=1, edge_order=2)[0]

    # viscous stress tensor  σ_{ij} n_j  (summed over j, for each i)
    xstress = nu_rho * (2 * dudx * nx + (dudy + dvdx) * ny + (dudz + dwdx) * nz)
    ystress = nu_rho * ((dvdx + dudy) * nx + 2 * dvdy * ny + (dvdz + dwdy) * nz)
    zstress = nu_rho * ((dwdx + dudz) * nx + (dwdy + dvdz) * ny + 2 * dwdz * nz)

    # Pressure force density: use p directly (NO mu0 masking).
    # See comment in _forces_shared_2d for the mathematical rationale.
    pforce_x = -p * nx
    pforce_y = -p * ny
    pforce_z = -p * nz

    return xstress, ystress, zstress, pforce_x, pforce_y, pforce_z


def _forces_body_integrate_3d(
    xstress, ystress, zstress,
    pforce_x, pforce_y, pforce_z,
    sdf_i, eps_body, eps_solver,
    com_x, com_y, com_z,
    X, Y, Z, h3,
    sdf_grad_mag=None,
):
    """Integrate viscous + pressure force / torque for ONE body.

    Inlines ``body.phi`` (smoothed delta) and ``cross_product_3d`` for
    maximal fusion under ``torch.compile``.

    Parameters
    ----------
    sdf_grad_mag : (Ni, Nj, Nk) or None
        |∇SDF| for this body.  When provided (Towers 2nd-order), deltas
        are divided by this to correct for non-unit SDF gradients.

    Returns 18 scalars: (fv_x,fv_y,fv_z, tv_x,tv_y,tv_z,
                          fp_x,fp_y,fp_z, tp_x,tp_y,tp_z).
    """
    # smoothed delta — viscous (shifted by eps_solver)
    d_visc = sdf_i - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # smoothed delta — pressure
    delta_pres = torch.where(
        torch.abs(sdf_i) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_i / eps_body)) / (2.0 * eps_body),
        0.0,
    )

    # Towers (2008) 2nd-order correction: δ_S = δ_ε(φ) / |∇φ|
    if sdf_grad_mag is not None:
        inv_grad = 1.0 / sdf_grad_mag.clamp(min=1e-3)
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    # viscous forces — cast to float64 before .sum() for deterministic
    # accumulation order (GPU tree-reduction vs CPU sequential).
    fvisc_x = xstress * delta_visc
    fvisc_y = ystress * delta_visc
    fvisc_z = zstress * delta_visc
    _d = torch.float64
    fv_x = fvisc_x.to(_d).sum().to(fvisc_x.dtype) * h3
    fv_y = fvisc_y.to(_d).sum().to(fvisc_y.dtype) * h3
    fv_z = fvisc_z.to(_d).sum().to(fvisc_z.dtype) * h3

    # moment arms from CoM
    rx = X - com_x
    ry = Y - com_y
    rz = Z - com_z

    # torque  r × f_visc
    tv_x = (ry * fvisc_z - rz * fvisc_y).to(_d).sum().to(fvisc_x.dtype) * h3
    tv_y = (rz * fvisc_x - rx * fvisc_z).to(_d).sum().to(fvisc_x.dtype) * h3
    tv_z = (rx * fvisc_y - ry * fvisc_x).to(_d).sum().to(fvisc_x.dtype) * h3

    # pressure forces
    fpres_x = pforce_x * delta_pres
    fpres_y = pforce_y * delta_pres
    fpres_z = pforce_z * delta_pres
    fp_x = fpres_x.to(_d).sum().to(fpres_x.dtype) * h3
    fp_y = fpres_y.to(_d).sum().to(fpres_y.dtype) * h3
    fp_z = fpres_z.to(_d).sum().to(fpres_z.dtype) * h3

    # torque  r × f_pres
    tp_x = (ry * fpres_z - rz * fpres_y).to(_d).sum().to(fpres_x.dtype) * h3
    tp_y = (rz * fpres_x - rx * fpres_z).to(_d).sum().to(fpres_x.dtype) * h3
    tp_z = (rx * fpres_y - ry * fpres_x).to(_d).sum().to(fpres_x.dtype) * h3

    return (fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
            fp_x, fp_y, fp_z, tp_x, tp_y, tp_z)


def _forces_body_batch_3d(
    xstress, ystress, zstress,
    pforce_x, pforce_y, pforce_z,
    sdf_all, eps_body, eps_solver,
    com_x, com_y, com_z,
    X, Y, Z, h3,
    sdf_grad_mag=None,
):
    """Batched force/torque integration for ALL bodies in one fused call.

    Parameters
    ----------
    xstress, ystress, zstress : (Ni, Nj, Nk)  viscous stress·n
    pforce_x, pforce_y, pforce_z : (Ni, Nj, Nk) pressure force density
    sdf_all : (B, Ni, Nj, Nk)  per-body SDF (1e4 outside each AABB)
    eps_body, eps_solver : scalar
    com_x, com_y, com_z : (B,) centre-of-mass positions
    X, Y, Z : (Ni, Nj, Nk)  grid coordinates
    h3 : scalar (h**3)
    sdf_grad_mag : (B, Ni, Nj, Nk) or None
        Per-body |∇SDF|.  When provided (Towers 2nd-order), deltas are
        divided by this to correct for non-unit SDF gradients.
        Pass ``None`` (default) for the standard 1st-order cosine delta.

    Returns
    -------
    12 tensors of shape (B,): fv_xyz, tv_xyz, fp_xyz, tp_xyz

    Torques use the algebraic decomposition:
        τ_x = Σ(Y·f_z - Z·f_y)·h³  -  com_y·F_z  +  com_z·F_y
    which avoids materializing (B,Ni,Nj,Nk) moment arm tensors.
    """
    # smoothed delta — viscous (shifted by eps_solver)   (B, Ni, Nj, Nk)
    d_visc = sdf_all - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # smoothed delta — pressure
    delta_pres = torch.where(
        torch.abs(sdf_all) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_all / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # Towers (2008) 2nd-order correction: δ_S = δ_ε(φ) / |∇φ|
    if sdf_grad_mag is not None:
        inv_grad = 1.0 / sdf_grad_mag.clamp(min=1e-3)
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    # Broadcast stress (1,Ni,Nj,Nk) * delta (B,Ni,Nj,Nk)
    xs = xstress.unsqueeze(0)
    ys = ystress.unsqueeze(0)
    zs = zstress.unsqueeze(0)

    fvisc_x = xs * delta_visc
    fvisc_y = ys * delta_visc
    fvisc_z = zs * delta_visc

    # Cast to float64 before .sum() for deterministic CPU/GPU accumulation.
    _d = torch.float64
    _dt = fvisc_x.dtype
    fv_x = fvisc_x.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fv_y = fvisc_y.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fv_z = fvisc_z.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    # ---- Torques via origin-torque decomposition ------------------
    # τ_x = Σ((Y-com_y)·fz - (Z-com_z)·fy)·h³
    #      = (Σ Y·fz)·h³ - com_y·F_z - (Σ Z·fy)·h³ + com_z·F_y
    # No (B,Ni,Nj,Nk) moment arm allocation needed.
    Xu = X.unsqueeze(0)
    Yu = Y.unsqueeze(0)
    Zu = Z.unsqueeze(0)

    # Raw torques about the origin (fused multiply+reduce)
    raw_v_yz = (Yu * fvisc_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_zy = (Zu * fvisc_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_zx = (Zu * fvisc_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_xz = (Xu * fvisc_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_xy = (Xu * fvisc_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_yx = (Yu * fvisc_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    tv_x = raw_v_yz - com_y * fv_z - raw_v_zy + com_z * fv_y
    tv_y = raw_v_zx - com_z * fv_x - raw_v_xz + com_x * fv_z
    tv_z = raw_v_xy - com_x * fv_y - raw_v_yx + com_y * fv_x

    # Pressure forces
    px = pforce_x.unsqueeze(0)
    py = pforce_y.unsqueeze(0)
    pz = pforce_z.unsqueeze(0)

    fpres_x = px * delta_pres
    fpres_y = py * delta_pres
    fpres_z = pz * delta_pres

    fp_x = fpres_x.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fp_y = fpres_y.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fp_z = fpres_z.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    raw_p_yz = (Yu * fpres_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_zy = (Zu * fpres_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_zx = (Zu * fpres_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_xz = (Xu * fpres_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_xy = (Xu * fpres_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_yx = (Yu * fpres_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    tp_x = raw_p_yz - com_y * fp_z - raw_p_zy + com_z * fp_y
    tp_y = raw_p_zx - com_z * fp_x - raw_p_xz + com_x * fp_z
    tp_z = raw_p_xy - com_x * fp_y - raw_p_yx + com_y * fp_x

    return (fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
            fp_x, fp_y, fp_z, tp_x, tp_y, tp_z)


# ======================================================================
# Compilable 2-D force-computation kernels  (module-level)
# ======================================================================

def _forces_shared_2d(u, v, p, sdf_val, nx, ny, nu_rho, h):
    """Compute stress tensors and pressure force density for 2-D.

    All quantities are evaluated on the cell-centred (CC) grid.

    * **Diagonal** (∂u_i/∂x_i): natural compact stencil exploiting the
      MAC stagger — ``(u[I+δ(i),i] - u[I,i]) / h``, exact at CC.
    * **Cross** (∂u_i/∂x_j, i≠j): 4-point average to CC —
      equivalent to interpolating u_i to CC along its stagger dimension,
      then central-differencing along x_j.

    Stagger convention: u at (x − h/2, y), v at (x, y − h/2).

    Returns
    -------
    xstress, ystress : viscous stress·n components on CC grid
    pforce_x, pforce_y : pressure force density on CC grid
    """
    # ---- Diagonal derivatives: compact stencil (exact at CC) --------
    dudx = torch.empty_like(u)
    dudx[:-1, :] = (u[1:, :] - u[:-1, :]) / h
    dudx[-1, :]  = dudx[-2, :]

    dvdy = torch.empty_like(v)
    dvdy[:, :-1] = (v[:, 1:] - v[:, :-1]) / h
    dvdy[:, -1]  = dvdy[:, -2]

    # ---- Cross derivatives: interp to CC along stagger dim, ----------
    #      then central diff along the other dim
    # u_cc: u interpolated to CC along dim 0
    u_cc = torch.empty_like(u)
    u_cc[:-1, :] = 0.5 * (u[:-1, :] + u[1:, :])
    u_cc[-1, :]  = u[-1, :]

    dudy = torch.gradient(u_cc, spacing=h, dim=1, edge_order=2)[0]

    # v_cc: v interpolated to CC along dim 1
    v_cc = torch.empty_like(v)
    v_cc[:, :-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    v_cc[:, -1]  = v[:, -1]

    dvdx = torch.gradient(v_cc, spacing=h, dim=0, edge_order=2)[0]

    # viscous stress tensor  σ_{ij} n_j  (summed over j, for each i)
    xstress = nu_rho * (2 * dudx * nx + (dudy + dvdx) * ny)
    ystress = nu_rho * ((dvdx + dudy) * nx + 2 * dvdy * ny)

    # Pressure force density: use p directly (NO mu0 masking).
    #
    # The smoothed delta δ_ε(φ) is normalised so that ∫δ_ε dφ = 1 over
    # the full support [-ε, +ε].  Masking p by mu0 (the smooth Heaviside,
    # which equals 0.5 at the surface) halves the integral:
    #     ∫ mu0(φ) δ_ε(φ) dφ = 0.5  (WRONG)
    # vs  ∫        δ_ε(φ) dφ = 1.0  (CORRECT)
    # The Poisson solve produces a continuous pressure field across the
    # interface, so both sides contribute correctly to the surface
    # integral.
    pforce_x = -p * nx
    pforce_y = -p * ny

    return xstress, ystress, pforce_x, pforce_y


def _forces_body_batch_2d(
    xstress, ystress,
    pforce_x, pforce_y,
    sdf_vals_cc,
    eps_body, eps_solver,
    com_x, com_y,
    X, Y, h2,
    sdf_grad_mag=None,
):
    """Batched 2-D force/torque integration for ALL bodies in one call.

    All quantities live on the cell-centred (CC) grid, matching the
    all-CC approach used in ``_forces_shared_2d`` and the 3-D path.

    Parameters
    ----------
    xstress, ystress : (Ni, Nj)  viscous stress·n on CC grid
    pforce_x, pforce_y : (Ni, Nj)  pressure force density on CC grid
    sdf_vals_cc : (B, Ni, Nj)  per-body SDF on CC grid
    eps_body, eps_solver : scalar
    com_x, com_y : (B,)  centre-of-mass positions
    X, Y : (Ni, Nj)  CC grid coordinates
    h2 : scalar (h**2)
    sdf_grad_mag : (B, Ni, Nj) or None
        Per-body |∇SDF| on the CC grid.  When provided (Towers 2nd-order),
        deltas are divided by this to correct for non-unit SDF gradients,
        giving a correct surface measure even when |∇SDF| ≠ 1.
        Pass ``None`` (default) for the standard 1st-order cosine delta.

    Returns
    -------
    6 tensors of shape (B,): fv_x, fv_y, tv_z, fp_x, fp_y, tp_z
    """
    # smoothed delta — viscous (shifted by eps_solver)  (B, Ni, Nj)
    d_visc = sdf_vals_cc - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # smoothed delta — pressure
    delta_pres = torch.where(
        torch.abs(sdf_vals_cc) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_vals_cc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # Towers (2008) 2nd-order correction: δ_S = δ_ε(φ) / |∇φ|.
    # For a perfect SDF |∇φ|=1 so this is a no-op; for numerical SDFs
    # (mesh bodies, near corners) it restores the correct surface measure.
    if sdf_grad_mag is not None:
        inv_grad = 1.0 / sdf_grad_mag.clamp(min=1e-3)
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    # Broadcast stress (1,Ni,Nj) * delta (B,Ni,Nj)
    fvisc_x = xstress.unsqueeze(0) * delta_visc   # (B, Ni, Nj)
    fvisc_y = ystress.unsqueeze(0) * delta_visc   # (B, Ni, Nj)

    # Cast to float64 before .sum() for deterministic CPU/GPU accumulation.
    _d = torch.float64
    _dt = fvisc_x.dtype
    fv_x = fvisc_x.to(_d).sum(dim=(1, 2)).to(_dt) * h2  # (B,)
    fv_y = fvisc_y.to(_d).sum(dim=(1, 2)).to(_dt) * h2  # (B,)

    # Torque via origin-torque decomposition:
    #   τ_z = Σ((X-com_x)·fy - (Y-com_y)·fx)·h²
    #       = (Σ X·fy)·h² - com_x·Fy - (Σ Y·fx)·h² + com_y·Fx
    Xu = X.unsqueeze(0)   # (1, Ni, Nj)
    Yu = Y.unsqueeze(0)   # (1, Ni, Nj)

    raw_x_fy = (Xu * fvisc_y).to(_d).sum(dim=(1, 2)).to(_dt) * h2  # (B,)
    raw_y_fx = (Yu * fvisc_x).to(_d).sum(dim=(1, 2)).to(_dt) * h2  # (B,)

    tv_z = raw_x_fy - com_x * fv_y - raw_y_fx + com_y * fv_x

    # Pressure forces
    fpres_x = pforce_x.unsqueeze(0) * delta_pres  # (B, Ni, Nj)
    fpres_y = pforce_y.unsqueeze(0) * delta_pres  # (B, Ni, Nj)

    fp_x = fpres_x.to(_d).sum(dim=(1, 2)).to(_dt) * h2  # (B,)
    fp_y = fpres_y.to(_d).sum(dim=(1, 2)).to(_dt) * h2  # (B,)

    raw_x_py = (Xu * fpres_y).to(_d).sum(dim=(1, 2)).to(_dt) * h2
    raw_y_px = (Yu * fpres_x).to(_d).sum(dim=(1, 2)).to(_dt) * h2

    tp_z = raw_x_py - com_x * fp_y - raw_y_px + com_y * fp_x

    return fv_x, fv_y, tv_z, fp_x, fp_y, tp_z

# ======================================================================
# Methods bound to FluidSolver (take ``self`` as first argument).
# ======================================================================

def forces_method1(self, u, v, p, iteration):

    # ---- CC normals (computed on-the-fly, not cached on self) ------
    normal_x, normal_y = self.composite_body.compute_normals(
        self.composite_body.sdf_val
    )

    # Build a co-located CC traction field before contour interpolation.
    # The legacy method1 mixed x/y traction components sampled from
    # different staggered grids, which produced noisy force/torque pairs
    # at moving contour points and destabilized MuJoCo coupling.
    nu_rho = self._compute_nu_rho_for_forces(u, v)
    (xstress, ystress,
     pforce_x, pforce_y) = self._forces_shared_2d_compiled(
        u, v, p, self.composite_body.sdf_val,
        normal_x, normal_y,
        nu_rho, self.h,
    )

    if self._compile_forces:
        xstress = xstress.clone()
        ystress = ystress.clone()
        pforce_x = pforce_x.clone()
        pforce_y = pforce_y.clone()

    self.xstress_tensor = xstress
    self.ystress_tensor = ystress
    self.pforce_x = pforce_x
    self.pforce_y = pforce_y


    for i, body in enumerate(self.composite_body.bodies[:]):

              mask       = body.mask
              curv_coord = body.curv_coord[mask].to(torch.float64)

              moment_arm = [
                  body.cnt_update[0]-body.com_pos[0],
                  body.cnt_update[1]-body.com_pos[1],
              ]

              self.interp_utility.F = self.xstress_tensor
              f_x = self.interp_utility(body.cnt_update[0], body.cnt_update[1])
              self.interp_utility.F = self.ystress_tensor
              f_y = self.interp_utility(body.cnt_update[0], body.cnt_update[1])

              visc_torque = ops.cross_product_2d(
                      moment_arm[0][mask],
                      moment_arm[1][mask],
                      f_x[mask],
                      f_y[mask]
                      ).to(torch.float64)

              self.friction_force_lin_x[i] = torch.trapz(
                  f_x[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.friction_force_lin_y[i] = torch.trapz(
                  f_y[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.friction_force_ang_z[i] = torch.trapz(
                  visc_torque, curv_coord
              ).to(self.dtype)

              moment_arm = [
                  body.cnt_update[0]-body.com_pos[0],
                  body.cnt_update[1]-body.com_pos[1],
              ]

              self.interp_utility.F = self.pforce_x
              f_x = self.interp_utility(body.cnt_update[0], body.cnt_update[1])
              self.interp_utility.F = self.pforce_y
              f_y = self.interp_utility(body.cnt_update[0], body.cnt_update[1])

              pres_torque = ops.cross_product_2d(
                      moment_arm[0][mask],
                      moment_arm[1][mask],
                      f_x[mask],
                      f_y[mask]
                      ).to(torch.float64)

              self.pressure_force_x[i] = torch.trapz(
                  f_x[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.pressure_force_y[i] = torch.trapz(
                  f_y[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.pressure_force_ang_z[i] = torch.trapz(
                  pres_torque, curv_coord
              ).to(self.dtype)

              self.viscous_drag_record[i,0,iteration] = self.friction_force_lin_x[i]
              self.viscous_drag_record[i,1,iteration] = self.friction_force_lin_y[i]

              self.pressure_drag_record[i,0,iteration] = self.pressure_force_x[i]
              self.pressure_drag_record[i,1,iteration] = self.pressure_force_y[i]



def forces_method2(self, u, v, p, iteration):

    _fused_out = getattr(self.composite_body, '_fused_forces_out', None)
    if _fused_out is not None:
        comp = self.composite_body
        B = len(comp.bodies)
        out_s = _fused_out if _fused_out.dtype == u.dtype else _fused_out.to(u.dtype)
        self.viscous_drag_record[:B, 0, iteration]  = out_s[:, 0]
        self.viscous_drag_record[:B, 1, iteration]  = out_s[:, 1]
        self.pressure_drag_record[:B, 0, iteration] = out_s[:, 3]
        self.pressure_drag_record[:B, 1, iteration] = out_s[:, 4]
        self.friction_force_lin_x = self.viscous_drag_record[:B, 0, iteration]
        self.friction_force_lin_y = self.viscous_drag_record[:B, 1, iteration]
        self.friction_force_ang_z = out_s[:, 2].clone()
        self.pressure_force_x     = self.pressure_drag_record[:B, 0, iteration]
        self.pressure_force_y     = self.pressure_drag_record[:B, 1, iteration]
        self.pressure_force_ang_z = out_s[:, 5].clone()
        self.xstress_tensor = None
        self.ystress_tensor = None
        comp._fused_forces_out = None
        return

    # ---- CC normals (computed on-the-fly, not cached on self) ------
    normal_x, normal_y = self.composite_body.compute_normals(
        self.composite_body.sdf_val
    )

    comp = self.composite_body
    B = len(comp.bodies)

    # ==============================================================
    # BATCHED path — all bodies in one fused call
    # ==============================================================
    # Shared part: stress + pressure force density (all CC grid)
    nu_rho = self._compute_nu_rho_for_forces(u, v)
    (xstress, ystress,
     pforce_x, pforce_y) = self._forces_shared_2d_compiled(
        u, v, p, comp.sdf_val,
        normal_x, normal_y,
        nu_rho, self.h,
    )

    if self._compile_forces:
        xstress  = xstress.clone()
        ystress  = ystress.clone()
        pforce_x = pforce_x.clone()
        pforce_y = pforce_y.clone()

    # Cache for post-processing
    self.xstress_tensor = xstress
    self.ystress_tensor = ystress
    self.pforce_x = pforce_x
    self.pforce_y = pforce_y

    eps_body = comp.bodies[0].eps

    # ---- 2-D Phase D: fused per-body force integration ----------
    # Reads the per-body cc-SDF cached by ``streaming_sdf_min_2d_multi``
    # (in BDIMhandler._update_2d_streaming_multi) instead of integrating
    # over the full ``comp.sdf_vals`` stack — both saves the (B, Nx, Ny)
    # SDF tensor allocation in BDIMhandler and avoids the dense
    # ``_forces_body_batch_2d`` reduction over the full grid.
    _stream_step = getattr(comp, '_stream_multi_step', None)
    _have_sparse_2d = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )
    _use_streaming_forces_2d = (
        getattr(self, '_streaming_forces_2d', False)
        and _have_sparse_2d
        and _stream_step is not None
    )
    if _use_streaming_forces_2d:
        from lilytorch.src.kernels import bdim_forces_2d_multi

        Ni, Nj = xstress.shape
        u_i0, u_j0 = 0, 0
        Sj = Nj

        # Persistent (B, 6) accumulator.
        out2d = getattr(self, '_phaseD_out_buf_2d', None)
        if out2d is None or out2d.shape[0] != B:
            out2d = torch.zeros((B, 6), dtype=torch.float64, device=self.device)
            self._phaseD_out_buf_2d = out2d
        else:
            out2d.zero_()

        bdim_forces_2d_multi(
            _stream_step['sparse_cc_flat'], _stream_step['cell_offsets'],
            _stream_step['kin'],
            _stream_step['aabb_lo'], _stream_step['aabb_dim'],
            _stream_step['gx'], _stream_step['gy'],
            u_i0, u_j0, Sj,
            xstress, ystress,
            pforce_x, pforce_y,
            eps_body, self.eps, self.h2,
            _stream_step['max_vol'],
            self.force_delta_order,
            out2d,
        )

        out_s = out2d if out2d.dtype == u.dtype else out2d.to(u.dtype)
        # 6-channel layout: [fv_x, fv_y, t_v, fp_x, fp_y, t_p]
        self.viscous_drag_record[:B, 0, iteration]  = out_s[:, 0]
        self.viscous_drag_record[:B, 1, iteration]  = out_s[:, 1]
        self.pressure_drag_record[:B, 0, iteration] = out_s[:, 3]
        self.pressure_drag_record[:B, 1, iteration] = out_s[:, 4]
        # Bind per-body buffers as views into the record arrays.
        self.friction_force_lin_x = self.viscous_drag_record[:B, 0, iteration]
        self.friction_force_lin_y = self.viscous_drag_record[:B, 1, iteration]
        self.friction_force_ang_z = out_s[:, 2].clone()
        self.pressure_force_x     = self.pressure_drag_record[:B, 0, iteration]
        self.pressure_force_y     = self.pressure_drag_record[:B, 1, iteration]
        self.pressure_force_ang_z = out_s[:, 5].clone()
        return

    # Towers (2008) 2nd-order: compute per-body |∇SDF| on CC grid  (B,Ni,Nj)
    # ``comp.sdf_vals`` was the legacy dense per-body SDF stack.  The new
    # 2-D Python update (``_update_2d``) populates ``comp._sdf_sparse``
    # (per-body AABB-cropped slabs) instead — mirroring 3-D.  Reconstruct
    # a dense (B, Ni, Nj) tensor here when the sparse storage is present.
    _have_sparse_2d = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )
    if _have_sparse_2d:
        _FAR = 1e6
        Ni, Nj = comp.sdf_val.shape
        sdf_vals = torch.full(
            (B, Ni, Nj), _FAR,
            device=self.device, dtype=self.dtype,
        )
        for bi in range(B):
            aabb_i, sdf_sub_i = comp._sdf_sparse[bi]
            if aabb_i is not None:
                i0, i1, j0, j1 = aabb_i
                sdf_vals[bi, i0:i1, j0:j1] = sdf_sub_i
            else:
                sdf_vals[bi] = sdf_sub_i
    else:
        sdf_vals = comp.sdf_vals  # legacy dense path

    sdf_grad_mag_2d = None
    if self.force_delta_order == 2:
        gx = torch.gradient(sdf_vals, spacing=self.h, dim=1, edge_order=2)[0]
        gy = torch.gradient(sdf_vals, spacing=self.h, dim=2, edge_order=2)[0]
        sdf_grad_mag_2d = torch.sqrt(gx**2 + gy**2)

    (fv_x, fv_y, tv_z,
     fp_x, fp_y, tp_z) = self._forces_body_batch_2d_compiled(
        xstress, ystress,
        pforce_x, pforce_y,
        sdf_vals,
        eps_body, self.eps,
        comp.com_pos[:, 0], comp.com_pos[:, 1],
        self.grids.X, self.grids.Y, self.h2,
        sdf_grad_mag_2d,
    )

    if self._compile_forces:
        fv_x = fv_x.clone(); fv_y = fv_y.clone(); tv_z = tv_z.clone()
        fp_x = fp_x.clone(); fp_y = fp_y.clone(); tp_z = tp_z.clone()

    for i in range(B):
        self.friction_force_lin_x[i] = fv_x[i]
        self.friction_force_lin_y[i] = fv_y[i]
        self.friction_force_ang_z[i] = tv_z[i]
        self.pressure_force_x[i] = fp_x[i]
        self.pressure_force_y[i] = fp_y[i]
        self.pressure_force_ang_z[i] = tp_z[i]
        self.viscous_drag_record[i, 0, iteration]  = fv_x[i]
        self.viscous_drag_record[i, 1, iteration]  = fv_y[i]
        self.pressure_drag_record[i, 0, iteration] = fp_x[i]
        self.pressure_drag_record[i, 1, iteration] = fp_y[i]


# ==================================================================
# 3-D force computation  (volume-integral with smoothed delta)
# ==================================================================

def forces_method2_3d(self, u, v, w, p, iteration):
    """Compute viscous and pressure forces/torques on each body in 3-D.

    Uses the same smoothed-delta volume-integration approach as the 2-D
    ``forces_method2`` but extended to three dimensions:

    Viscous force:
        F_visc = ∫ (σ · n) δ_ε(d - ε) dV
    where σ_{ij} = ν ρ (∂u_i/∂x_j + ∂u_j/∂x_i) is the viscous stress.

    Pressure force:
        F_pres = -∫ p n δ_ε(d) dV

    Torques are computed about each body's centre of mass via r × f.

    When ``compile_forces=True``, the heavy tensor work is delegated to
    two ``torch.compile``-d kernels (``_forces_shared_3d`` and
    ``_forces_body_integrate_3d``) that fuse ~40 CUDA kernels into one
    or two CUDA-graph launches, giving ~6× wall-clock speedup.
    """
    comp = self.composite_body

    # ============================================================
    # Fused Phase C+D fast path: forces were pre-computed in the
    # SDF update step using lagged velocity/normals.  Just store the
    # cached result into the force records and return early.
    # Checked before _compute_nu_rho_for_forces to skip the
    # potentially expensive variable-viscosity field computation.
    # ============================================================
    _fused_out = getattr(comp, '_fused_forces_out', None)
    if _fused_out is not None:
        B = len(comp.bodies)
        out_s = _fused_out if _fused_out.dtype == u.dtype else _fused_out.to(u.dtype)
        self.viscous_drag_record[:B, :, iteration]    = out_s[:, 0:3]
        self.viscous_torque_record[:B, :, iteration]  = out_s[:, 3:6]
        self.pressure_drag_record[:B, :, iteration]   = out_s[:, 6:9]
        self.pressure_torque_record[:B, :, iteration] = out_s[:, 9:12]
        self.friction_force_lin_x = self.viscous_drag_record[:B, 0, iteration]
        self.friction_force_lin_y = self.viscous_drag_record[:B, 1, iteration]
        self.friction_force_lin_z = self.viscous_drag_record[:B, 2, iteration]
        self.friction_force_ang_x = self.viscous_torque_record[:B, 0, iteration]
        self.friction_force_ang_y = self.viscous_torque_record[:B, 1, iteration]
        self.friction_force_ang_z = self.viscous_torque_record[:B, 2, iteration]
        self.pressure_force_x     = self.pressure_drag_record[:B, 0, iteration]
        self.pressure_force_y     = self.pressure_drag_record[:B, 1, iteration]
        self.pressure_force_z     = self.pressure_drag_record[:B, 2, iteration]
        self.pressure_force_ang_x = self.pressure_torque_record[:B, 0, iteration]
        self.pressure_force_ang_y = self.pressure_torque_record[:B, 1, iteration]
        self.pressure_force_ang_z = self.pressure_torque_record[:B, 2, iteration]
        # Cache zero-out stress tensors (not computed in fused path)
        self.xstress_tensor = None
        self.ystress_tensor = None
        self.zstress_tensor = None
        return

    nu_rho = self._compute_nu_rho_for_forces(u, v, w)
    h      = self.h
    h3     = self.h3

    # ---- CC normals (reuse cached values when available) ---------
    nx = getattr(self, 'normal_x', None)
    if nx is None:
        nx, ny, nz = comp.compute_normals(comp.sdf_val)
    else:
        ny, nz = self.normal_y, self.normal_z

    # ============================================================
    # Phase F (fused per-body forces) was removed; the path below
    # uses the streaming forces / shared-stress kernels (Phase D).
    # ============================================================
    # ---- decide whether to crop shared kernel to union AABB ------
    # Only worthwhile when narrow-band path is on and sparse SDFs are
    # available.  We compute the union AABB across all bodies and run
    # the bandwidth-bound _forces_shared_3d only on that sub-block.
    # Per-body integration then uses indices RELATIVE to the union.
    _have_sparse_for_union = (
        self._forces_shared_union
        and hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )
    # Determine union AABB (only if all bodies have AABBs; if any
    # body uses full-grid evaluation we fall back to the full kernel).
    u_aabb = None
    if _have_sparse_for_union:
        u_i0, u_j0, u_k0 = 1 << 30, 1 << 30, 1 << 30
        u_i1, u_j1, u_k1 = -1, -1, -1
        for aabb_i, _ in comp._sdf_sparse:
            if aabb_i is None:
                u_i0 = -1
                break
            i0, i1, j0, j1, k0, k1 = aabb_i
            if i0 < u_i0: u_i0 = i0
            if j0 < u_j0: u_j0 = j0
            if k0 < u_k0: u_k0 = k0
            if i1 > u_i1: u_i1 = i1
            if j1 > u_j1: u_j1 = j1
            if k1 > u_k1: u_k1 = k1
        if u_i0 != -1:
            # 1-cell halo on each side for gradient stencil safety;
            # AABB indices are already padded by ``_body_aabb_indices``
            # but the cropped boundary cell of the shared kernel
            # uses a one-sided diff, so add an extra halo of 2.
            Ni, Nj, Nk = u.shape
            halo = 2
            u_i0 = max(0, u_i0 - halo); u_i1 = min(Ni, u_i1 + halo)
            u_j0 = max(0, u_j0 - halo); u_j1 = min(Nj, u_j1 + halo)
            u_k0 = max(0, u_k0 - halo); u_k1 = min(Nk, u_k1 + halo)
            u_aabb = (u_i0, u_i1, u_j0, u_j1, u_k0, u_k1)

    if u_aabb is not None:
        # ---- shared kernel on cropped sub-block ------------------
        ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
        usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))
        # Variable-viscosity (Smagorinsky/Carreau) returns a full-grid
        # tensor; crop it to the union sub-block to match the other
        # cropped inputs.  Constant viscosity stays a scalar.
        nu_rho_sub = nu_rho[usl].contiguous() if torch.is_tensor(nu_rho) and nu_rho.ndim == 3 else nu_rho
        (xstress, ystress, zstress,
         pforce_x, pforce_y, pforce_z) = self._forces_shared_dyn_compiled(
            u[usl].contiguous(), v[usl].contiguous(), w[usl].contiguous(),
            p[usl].contiguous(), comp.sdf_val[usl].contiguous(),
            nx[usl].contiguous(), ny[usl].contiguous(), nz[usl].contiguous(),
            nu_rho_sub, h,
        )
    else:
        # ---- full-grid shared kernel (default path) --------------
        (xstress, ystress, zstress,
         pforce_x, pforce_y, pforce_z) = self._forces_shared_compiled(
            u, v, w, p, comp.sdf_val, nx, ny, nz,
            nu_rho, h,
        )

    # When compiled with CUDA graphs (reduce-overhead), the returned
    # tensors live in the graph's replay buffer and will be overwritten
    # by the next compiled call.  Clone them here so the body-integrate
    # kernel can safely read them.
    #
    # SKIPPED when Phase D streaming forces will fire: that kernel
    # consumes stress / pforce in-place during this step and never
    # reads them across step boundaries.  At low-N this saves 6
    # full-grid memcpy launches per step.
    _comp_for_stream = self.composite_body
    _have_sparse_for_stream = (
        hasattr(_comp_for_stream, '_sdf_sparse')
        and len(_comp_for_stream._sdf_sparse) > 0
        and _comp_for_stream._sdf_sparse[0] is not None
    )
    _will_stream_forces = (
        self._streaming_forces_3d
        and _have_sparse_for_stream
        and getattr(_comp_for_stream, '_stream_multi_step', None) is not None
        and getattr(_comp_for_stream, '_stream_multi_static', None) is not None
        and self.force_delta_order == 1
    )
    if self._compile_forces and not _will_stream_forces:
        xstress  = xstress.clone()
        ystress  = ystress.clone()
        zstress  = zstress.clone()
        pforce_x = pforce_x.clone()
        pforce_y = pforce_y.clone()
        pforce_z = pforce_z.clone()

    # Cache stress / pforce for post-processing if needed
    self.xstress_tensor = xstress
    self.ystress_tensor = ystress
    self.zstress_tensor = zstress
    self.pforce_x = pforce_x
    self.pforce_y = pforce_y
    self.pforce_z = pforce_z

    # ---- per-body integration -----------------------------------
    X   = self.composite_body.X
    Y   = self.composite_body.Y
    Z   = self.composite_body.Z_grid
    comp = self.composite_body
    B = len(comp.bodies)

    # Prefer the narrow-band per-body path whenever sparse SDFs are
    # available: it integrates each body only over its AABB sub-block
    # (~50³ cells × B) instead of scattering to (B, Nx, Ny, Nz) and
    # reducing over the full volume.  This is ~50–100× less work for
    # typical thin-body geometries and removes the large Triton kernel
    # shape-dependent cliff seen on 512×128×128 grids.
    _have_sparse = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )

    # ---- Phase D: fused per-body force integration --------------
    _stream_step = getattr(comp, '_stream_multi_step', None)
    _stream_static = getattr(comp, '_stream_multi_static', None)
    _use_streaming_forces = (
        self._streaming_forces_3d
        and _have_sparse
        and _stream_step is not None
        and _stream_static is not None
    )
    if _use_streaming_forces:
        from lilytorch.src.kernels import bdim_forces_3d_multi
        if u_aabb is not None:
            ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
        else:
            # Full-grid fallback: stress / pforce live on the full
            # fluid grid, so the union-relative origin is (0, 0, 0)
            # and the union dims are the full grid dims.  Lets us
            # fire Phase D even when union cropping is disabled
            # (low-N regime where cropping launch overhead dominates).
            Ni, Nj, Nk = u.shape
            ui0, uj0, uk0 = 0, 0, 0
            ui1, uj1, uk1 = Ni, Nj, Nk
        Sj = uj1 - uj0
        Sk = uk1 - uk0
        eps_body = comp.bodies[0].eps
        # Persistent (B, 12) accumulator: avoid per-step torch.zeros
        # allocation (saves a malloc + a memset launch).
        out = getattr(self, '_phaseD_out_buf', None)
        if out is None or out.shape[0] != B:
            out = torch.zeros((B, 12), dtype=torch.float64, device=self.device)
            self._phaseD_out_buf = out
        else:
            out.zero_()
        # Skip redundant .contiguous() — _forces_shared_*_compiled
        # already returns row-major contiguous tensors.
        #
        # Phase D force op signature changed across kernel builds.
        # Detect the loaded schema once and keep a stable fast path.
        _phaseD_sparse_sig = getattr(self, '_phaseD_sparse_sig', None)
        if _phaseD_sparse_sig is None:
            try:
                _schema = str(
                    torch.ops.lilytorch_kernels.bdim_forces_3d_multi.default._schema
                )
                _phaseD_sparse_sig = "Tensor sparse_cc_flat" in _schema
            except Exception:
                _phaseD_sparse_sig = True
            self._phaseD_sparse_sig = _phaseD_sparse_sig

        if _phaseD_sparse_sig:
            bdim_forces_3d_multi(
                _stream_step['sparse_cc_flat'], _stream_step['cell_offsets'],
                _stream_step['kin'],
                _stream_step['aabb_lo'], _stream_step['aabb_dim'],
                _stream_step['gx'], _stream_step['gy'], _stream_step['gz'],
                ui0, uj0, uk0, Sj, Sk,
                xstress, ystress, zstress,
                pforce_x, pforce_y, pforce_z,
                eps_body, self.eps, h3,
                _stream_step['max_vol'],
                self.force_delta_order,
                out,
            )
        else:
            bdim_forces_3d_multi(
                _stream_static['F_flat'], _stream_static['F_offsets'],
                _stream_static['bx_flat'], _stream_static['bx_offsets'],
                _stream_static['by_flat'], _stream_static['by_offsets'],
                _stream_static['bz_flat'], _stream_static['bz_offsets'],
                _stream_static['body_shapes'], _stream_static['body_meta'],
                _stream_step['kin'],
                _stream_step['aabb_lo'], _stream_step['aabb_dim'],
                _stream_step['gx'], _stream_step['gy'], _stream_step['gz'],
                ui0, uj0, uk0, Sj, Sk,
                xstress, ystress, zstress,
                pforce_x, pforce_y, pforce_z,
                eps_body, self.eps, h3,
                _stream_step['max_vol'],
                out,
            )
        # Vectorised dispatch of the (B, 12) result.
        #
        # The previous implementation issued 24 individual slice
        # writes (12 into per-body force/torque scalar buffers and
        # 12 into the (B, 3, nt) record arrays).  Half of those are
        # redundant: the per-body buffers and the corresponding
        # record column at this iteration store the same numbers.
        #
        # New layout:
        #   • 4 bulk slice writes copy the (B, 3) blocks into the
        #     record tensors (1 contiguous DtoD copy each).
        #   • The per-body force/torque buffers are then *rebound*
        #     to views into the freshly-written record columns —
        #     a pure Python attribute assignment, no GPU work.
        #
        # Net launch count: 24 → 4 GPU writes + 12 attribute binds,
        # saving ~50–150 µs of host-side overhead at low N.
        out_s = out if out.dtype == u.dtype else out.to(u.dtype)
        self.viscous_drag_record[:B, :, iteration]    = out_s[:, 0:3]
        self.viscous_torque_record[:B, :, iteration]  = out_s[:, 3:6]
        self.pressure_drag_record[:B, :, iteration]   = out_s[:, 6:9]
        self.pressure_torque_record[:B, :, iteration] = out_s[:, 9:12]
        # Bind per-body buffers as views into the record arrays.
        # Safe across the next time-step because the record tensors
        # are persistent and only the column at ``iteration`` is
        # modified above (future iterations write to different
        # columns).  Views are non-contiguous (B-sized), but every
        # downstream consumer (apply_forces stack/.cpu(), .item(),
        # element-wise sums) handles strided 1-D tensors trivially.
        self.friction_force_lin_x = self.viscous_drag_record[:B, 0, iteration]
        self.friction_force_lin_y = self.viscous_drag_record[:B, 1, iteration]
        self.friction_force_lin_z = self.viscous_drag_record[:B, 2, iteration]
        self.friction_force_ang_x = self.viscous_torque_record[:B, 0, iteration]
        self.friction_force_ang_y = self.viscous_torque_record[:B, 1, iteration]
        self.friction_force_ang_z = self.viscous_torque_record[:B, 2, iteration]
        self.pressure_force_x     = self.pressure_drag_record[:B, 0, iteration]
        self.pressure_force_y     = self.pressure_drag_record[:B, 1, iteration]
        self.pressure_force_z     = self.pressure_drag_record[:B, 2, iteration]
        self.pressure_force_ang_x = self.pressure_torque_record[:B, 0, iteration]
        self.pressure_force_ang_y = self.pressure_torque_record[:B, 1, iteration]
        self.pressure_force_ang_z = self.pressure_torque_record[:B, 2, iteration]
        return


    if self._compile_forces:
        # ---- BATCHED path (compiled CUDA graph) ----------------
        # Process all bodies in one fused kernel launch.
        # Reconstruct dense sdf_vals from sparse storage if needed.
        if hasattr(comp, '_sdf_sparse') and comp._sdf_sparse[0] is not None:
            _FAR = 1e6
            sdf_all = torch.full(
                (B, *self.grid_shape), _FAR,
                device=self.device, dtype=self.dtype,
            )
            for bi in range(B):
                aabb_i, sdf_sub_i = comp._sdf_sparse[bi]
                if aabb_i is not None:
                    i0, i1, j0, j1, k0, k1 = aabb_i
                    sdf_all[bi, i0:i1, j0:j1, k0:k1] = sdf_sub_i
                else:
                    sdf_all[bi] = sdf_sub_i
        else:
            sdf_all = comp.sdf_vals                # legacy dense path
        eps_body = comp.bodies[0].eps

        if not hasattr(self, '_com_buf_x'):
            self._com_buf_x = torch.empty(B, device=self.device, dtype=self.dtype)
            self._com_buf_y = torch.empty(B, device=self.device, dtype=self.dtype)
            self._com_buf_z = torch.empty(B, device=self.device, dtype=self.dtype)
        for i, body in enumerate(comp.bodies):
            self._com_buf_x[i] = body.com_pos[0]
            self._com_buf_y[i] = body.com_pos[1]
            self._com_buf_z[i] = body.com_pos[2]

        # Towers (2008) 2nd-order: per-body |∇SDF|  (B, Ni, Nj, Nk)
        sdf_grad_mag_3d = None
        if self.force_delta_order == 2:
            gx = torch.gradient(sdf_all, spacing=h, dim=1, edge_order=2)[0]
            gy = torch.gradient(sdf_all, spacing=h, dim=2, edge_order=2)[0]
            gz = torch.gradient(sdf_all, spacing=h, dim=3, edge_order=2)[0]
            sdf_grad_mag_3d = torch.sqrt(gx**2 + gy**2 + gz**2)

        (fv_x, fv_y, fv_z,
         tv_x, tv_y, tv_z,
         fp_x, fp_y, fp_z,
         tp_x, tp_y, tp_z) = self._forces_body_batch_compiled(
            xstress, ystress, zstress,
            pforce_x, pforce_y, pforce_z,
            sdf_all, eps_body, self.eps,
            self._com_buf_x, self._com_buf_y, self._com_buf_z,
            X, Y, Z, h3,
            sdf_grad_mag_3d,
        )

        # Clone outputs (CUDA graph buffer reuse)
        fv_x = fv_x.clone(); fv_y = fv_y.clone(); fv_z = fv_z.clone()
        tv_x = tv_x.clone(); tv_y = tv_y.clone(); tv_z = tv_z.clone()
        fp_x = fp_x.clone(); fp_y = fp_y.clone(); fp_z = fp_z.clone()
        tp_x = tp_x.clone(); tp_y = tp_y.clone(); tp_z = tp_z.clone()

        for i in range(B):
            self.friction_force_lin_x[i] = fv_x[i]
            self.friction_force_lin_y[i] = fv_y[i]
            self.friction_force_lin_z[i] = fv_z[i]
            self.friction_force_ang_x[i] = tv_x[i]
            self.friction_force_ang_y[i] = tv_y[i]
            self.friction_force_ang_z[i] = tv_z[i]
            self.pressure_force_x[i] = fp_x[i]
            self.pressure_force_y[i] = fp_y[i]
            self.pressure_force_z[i] = fp_z[i]
            self.pressure_force_ang_x[i] = tp_x[i]
            self.pressure_force_ang_y[i] = tp_y[i]
            self.pressure_force_ang_z[i] = tp_z[i]
            self.viscous_drag_record[i, 0, iteration]  = fv_x[i]
            self.viscous_drag_record[i, 1, iteration]  = fv_y[i]
            self.viscous_drag_record[i, 2, iteration]  = fv_z[i]
            self.pressure_drag_record[i, 0, iteration] = fp_x[i]
            self.pressure_drag_record[i, 1, iteration] = fp_y[i]
            self.pressure_drag_record[i, 2, iteration] = fp_z[i]
            self.viscous_torque_record[i, 0, iteration]  = tv_x[i]
            self.viscous_torque_record[i, 1, iteration]  = tv_y[i]
            self.viscous_torque_record[i, 2, iteration]  = tv_z[i]
            self.pressure_torque_record[i, 0, iteration] = tp_x[i]
            self.pressure_torque_record[i, 1, iteration] = tp_y[i]
            self.pressure_torque_record[i, 2, iteration] = tp_z[i]
    else:
        # ---- PER-BODY path (un-compiled fallback) --------------
        # Use sparse SDF storage when available: integrate forces
        # only over the body's AABB sub-block (much less memory
        # and faster than iterating the full grid).
        _use_sparse = hasattr(comp, '_sdf_sparse') and comp._sdf_sparse[0] is not None
        for i, body in enumerate(comp.bodies):
            eps_body = body.eps

            if _use_sparse:
                aabb_i, sdf_sub_i = comp._sdf_sparse[i]
                if aabb_i is not None:
                    i0, i1, j0, j1, k0, k1 = aabb_i
                    sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
                    grad_mag_i = None
                    if self.force_delta_order == 2:
                        gx = torch.gradient(sdf_sub_i, spacing=h, dim=0, edge_order=2)[0]
                        gy = torch.gradient(sdf_sub_i, spacing=h, dim=1, edge_order=2)[0]
                        gz = torch.gradient(sdf_sub_i, spacing=h, dim=2, edge_order=2)[0]
                        grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)
                    (fv_x, fv_y, fv_z,
                     tv_x, tv_y, tv_z,
                     fp_x, fp_y, fp_z,
                     tp_x, tp_y, tp_z) = _forces_body_integrate_3d(
                        xstress[sl], ystress[sl], zstress[sl],
                        pforce_x[sl], pforce_y[sl], pforce_z[sl],
                        sdf_sub_i, eps_body, self.eps,
                        body.com_pos[0], body.com_pos[1], body.com_pos[2],
                        X[sl], Y[sl], Z[sl], h3,
                        grad_mag_i,
                    )
                else:
                    # Full-grid SDF (rare: body covers most of grid)
                    grad_mag_i = None
                    if self.force_delta_order == 2:
                        gx = torch.gradient(sdf_sub_i, spacing=h, dim=0, edge_order=2)[0]
                        gy = torch.gradient(sdf_sub_i, spacing=h, dim=1, edge_order=2)[0]
                        gz = torch.gradient(sdf_sub_i, spacing=h, dim=2, edge_order=2)[0]
                        grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)
                    (fv_x, fv_y, fv_z,
                     tv_x, tv_y, tv_z,
                     fp_x, fp_y, fp_z,
                     tp_x, tp_y, tp_z) = _forces_body_integrate_3d(
                        xstress, ystress, zstress,
                        pforce_x, pforce_y, pforce_z,
                        sdf_sub_i, eps_body, self.eps,
                        body.com_pos[0], body.com_pos[1], body.com_pos[2],
                        X, Y, Z, h3,
                        grad_mag_i,
                    )
            else:
                # Legacy dense path (comp.sdf_vals exists)
                sdf_i = comp.sdf_vals[i]
                grad_mag_i = None
                if self.force_delta_order == 2:
                    gx = torch.gradient(sdf_i, spacing=h, dim=0, edge_order=2)[0]
                    gy = torch.gradient(sdf_i, spacing=h, dim=1, edge_order=2)[0]
                    gz = torch.gradient(sdf_i, spacing=h, dim=2, edge_order=2)[0]
                    grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)
                (fv_x, fv_y, fv_z,
                 tv_x, tv_y, tv_z,
                 fp_x, fp_y, fp_z,
                 tp_x, tp_y, tp_z) = self._forces_body_compiled(
                    xstress, ystress, zstress,
                    pforce_x, pforce_y, pforce_z,
                    sdf_i, eps_body, self.eps,
                    body.com_pos[0], body.com_pos[1], body.com_pos[2],
                    X, Y, Z, h3,
                    grad_mag_i,
                )

            self.friction_force_lin_x[i] = fv_x
            self.friction_force_lin_y[i] = fv_y
            self.friction_force_lin_z[i] = fv_z
            self.friction_force_ang_x[i] = tv_x
            self.friction_force_ang_y[i] = tv_y
            self.friction_force_ang_z[i] = tv_z
            self.pressure_force_x[i] = fp_x
            self.pressure_force_y[i] = fp_y
            self.pressure_force_z[i] = fp_z
            self.pressure_force_ang_x[i] = tp_x
            self.pressure_force_ang_y[i] = tp_y
            self.pressure_force_ang_z[i] = tp_z
            self.viscous_drag_record[i, 0, iteration]  = fv_x
            self.viscous_drag_record[i, 1, iteration]  = fv_y
            self.viscous_drag_record[i, 2, iteration]  = fv_z
            self.pressure_drag_record[i, 0, iteration] = fp_x
            self.pressure_drag_record[i, 1, iteration] = fp_y
            self.pressure_drag_record[i, 2, iteration] = fp_z
            self.viscous_torque_record[i, 0, iteration]  = tv_x
            self.viscous_torque_record[i, 1, iteration]  = tv_y
            self.viscous_torque_record[i, 2, iteration]  = tv_z
            self.pressure_torque_record[i, 0, iteration] = tp_x
            self.pressure_torque_record[i, 1, iteration] = tp_y
            self.pressure_torque_record[i, 2, iteration] = tp_z
