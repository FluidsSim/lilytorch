
import datetime
import logging
import os
import threading
import warnings

import h5py
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pytorch_interpolation import RegularGridInterpolator
from tqdm import tqdm

from lilytorch.src.adv_diff import AdvDiffSolver
from lilytorch.src.body import (body_from_yaml, _StaggeredGrids,
                                _mu_normals_batched_3d,
                                _mu_normals_batched_3d_compiled)
from lilytorch.src import operations as ops
from lilytorch.src import plotting
from lilytorch.src.poisson_fft import PoissonSolverFFT
from lilytorch.src.poisson_mult import PoissonSolver
from lilytorch.util.yaml_operations import pyobject2yaml

logger = logging.getLogger(__name__)


# ======================================================================
# Flow diagnostics — energy, enstrophy, divergence, CFL monitoring
# ======================================================================

class FlowDiagnostics:
    """Lightweight monitor for kinetic energy, enstrophy, max-divergence,
    and CFL number.  Records scalar time-series and optionally warns when
    energy grows beyond a user-specified factor of its initial value.

    Parameters
    ----------
    nt : int
        Total number of time steps (for pre-allocation).
    ndim : int
        Spatial dimension (2 or 3).
    h : float or Tensor
        Uniform grid spacing.
    device, dtype
        Torch device / dtype for the record arrays.
    check_every : int
        Diagnostics are computed every *check_every* steps.  1 = every step.
    energy_growth_factor : float
        Issue a warning when E_k exceeds *energy_growth_factor* × E_k(0).
        Set to ``None`` or ``inf`` to disable the energy blow-up check.
    """

    def __init__(self, nt, ndim, h, device, dtype,
                 check_every=1, energy_growth_factor=10.0):
        self.nt    = nt
        self.ndim  = ndim
        self.h     = float(h)
        self.hd    = self.h ** ndim          # cell volume  h^d
        self.device = device
        self.dtype  = dtype

        self.check_every = max(1, int(check_every))
        self.energy_growth_factor = energy_growth_factor

        # Pre-allocated record arrays (filled with NaN so uncomputed slots
        # are visually obvious when plotted).
        self.kinetic_energy = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.enstrophy      = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.max_divergence  = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.cfl_number      = torch.full((nt,), float('nan'), device=device, dtype=dtype)

        self._ek0 = None   # E_k at the first computed step (baseline)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, iteration, u, v, p, dt, nu, divergence_fn, vorticity_fn,
               w=None):
        """Compute and record diagnostics for the current step.

        Parameters
        ----------
        iteration : int
            Current time-step index.
        u, v : Tensor
            Velocity components.  *w* is ``None`` in 2-D.
        p : Tensor
            Pressure (unused for now; kept for future pressure-energy).
        dt : float or Tensor
            Time step size (for CFL).
        nu : float or Tensor
            Kinematic viscosity (for CFL).
        divergence_fn : callable(u, v, w=None) -> Tensor
            Solver's divergence method.
        vorticity_fn  : callable(u, v, w=None) -> Tensor
            Solver's vorticity method (returns scalar in 2-D,
            magnitude in 3-D).
        w : Tensor or None
            z-velocity component (3-D only).
        """
        if iteration % self.check_every != 0:
            return

        h  = self.h
        hd = self.hd
        dt_val = float(dt)
        nu_val = float(nu)

        # ---- kinetic energy  E_k = 0.5 * h^d * Σ(u² + v² [+ w²]) ----
        ke = u.square() + v.square()
        if w is not None:
            ke = ke + w.square()
        ek = 0.5 * hd * ke.sum()
        self.kinetic_energy[iteration] = ek

        # ---- enstrophy  Z = 0.5 * h^d * Σ ω² ----
        omega = vorticity_fn(u, v, w)
        enst = 0.5 * hd * omega.square().sum()
        self.enstrophy[iteration] = enst

        # ---- max |div(u)| ----
        div = divergence_fn(u, v, w=w)
        self.max_divergence[iteration] = div.abs().max()

        # ---- CFL = u_max * dt / h ----
        vel_max = u.abs().max()
        vel_max = max(vel_max, v.abs().max())
        if w is not None:
            vel_max = max(vel_max, w.abs().max())
        self.cfl_number[iteration] = float(vel_max) * dt_val / h

        # ---- energy blow-up warning ----
        if self._ek0 is None:
            self._ek0 = float(ek) if float(ek) > 0 else 1.0
        if (self.energy_growth_factor is not None
                and float(ek) > self.energy_growth_factor * self._ek0):
            warnings.warn(
                f"[FlowDiagnostics] E_k = {float(ek):.6e} at iter {iteration} "
                f"exceeds {self.energy_growth_factor}x initial "
                f"({self._ek0:.6e}).  Possible blow-up.",
                RuntimeWarning, stacklevel=2,
            )

        # ---- CFL warning ----
        cfl_val = float(self.cfl_number[iteration])
        if cfl_val > 0.5:
            warnings.warn(
                f"[FlowDiagnostics] CFL = {cfl_val:.3f} > 0.5 at iter {iteration}",
                RuntimeWarning, stacklevel=2,
            )

    # ------------------------------------------------------------------
    def save_h5(self, path, lock):
        """Write diagnostics to ``<path>/diagnostics.h5``."""
        h5_path = os.path.join(path, "diagnostics.h5")
        data = {
            "kinetic_energy": self.kinetic_energy.cpu().numpy().copy(),
            "enstrophy":      self.enstrophy.cpu().numpy().copy(),
            "max_divergence":  self.max_divergence.cpu().numpy().copy(),
            "cfl_number":      self.cfl_number.cpu().numpy().copy(),
        }
        with lock:
            with h5py.File(h5_path, "w") as f:
                for name, arr in data.items():
                    f.create_dataset(name, data=arr)
        logger.info("Saved flow diagnostics to %s", h5_path)

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


def _forces_body_narrow_batch_3d(
    xs_b, ys_b, zs_b,
    px_b, py_b, pz_b,
    sdf_b, eps_body, eps_solver,
    com_x, com_y, com_z,
    X_b, Y_b, Z_b, h3,
    sdf_grad_mag=None,
):
    """Batched narrow-band force/torque integration.

    Same algorithm as ``_forces_body_batch_3d`` but expects per-body
    *padded* AABB sub-blocks of a fixed shape ``(B, Di, Dj, Dk)``.  All
    inputs have a leading body dimension; padded cells must have
    ``sdf_b >> eps_body`` so the smoothed delta vanishes there.  Running
    this on a tiny fixed volume (~(9, 63, 27, 26) ≈ 400 k cells for a
    9-link fish) lets ``torch.compile(mode='reduce-overhead')`` emit a
    single CUDA-graph launch regardless of grid resolution.
    """
    # smoothed deltas (B, Di, Dj, Dk)
    d_visc = sdf_b - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    delta_pres = torch.where(
        torch.abs(sdf_b) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_b / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    if sdf_grad_mag is not None:
        inv_grad = 1.0 / sdf_grad_mag.clamp(min=1e-3)
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    fvisc_x = xs_b * delta_visc
    fvisc_y = ys_b * delta_visc
    fvisc_z = zs_b * delta_visc

    _d = torch.float64
    _dt = fvisc_x.dtype
    fv_x = fvisc_x.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fv_y = fvisc_y.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fv_z = fvisc_z.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    # Torques: per-body absolute-coordinate reductions then com correction.
    raw_v_yz = (Y_b * fvisc_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_zy = (Z_b * fvisc_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_zx = (Z_b * fvisc_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_xz = (X_b * fvisc_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_xy = (X_b * fvisc_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_v_yx = (Y_b * fvisc_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    tv_x = raw_v_yz - com_y * fv_z - raw_v_zy + com_z * fv_y
    tv_y = raw_v_zx - com_z * fv_x - raw_v_xz + com_x * fv_z
    tv_z = raw_v_xy - com_x * fv_y - raw_v_yx + com_y * fv_x

    fpres_x = px_b * delta_pres
    fpres_y = py_b * delta_pres
    fpres_z = pz_b * delta_pres
    fp_x = fpres_x.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fp_y = fpres_y.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    fp_z = fpres_z.to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    raw_p_yz = (Y_b * fpres_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_zy = (Z_b * fpres_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_zx = (Z_b * fpres_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_xz = (X_b * fpres_z).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_xy = (X_b * fpres_y).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3
    raw_p_yx = (Y_b * fpres_x).to(_d).sum(dim=(1, 2, 3)).to(_dt) * h3

    tp_x = raw_p_yz - com_y * fp_z - raw_p_zy + com_z * fp_y
    tp_y = raw_p_zx - com_z * fp_x - raw_p_xz + com_x * fp_z
    tp_z = raw_p_xy - com_x * fp_y - raw_p_yx + com_y * fp_x

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


class FluidSolver:
    """
    Solver class
    """

    # ---- thin wrappers around standalone functions in operations.py ----
    def compute_dpdx(self, p):  return ops.compute_dpdx(p, self.h)
    def compute_dpdy(self, p):  return ops.compute_dpdy(p, self.h)
    def compute_dpdz(self, p):  return ops.compute_dpdz(p, self.h)
    def gradient(self, var):    return ops.gradient(var, self.h, self.ndim)
    def divergence(self, u, v, w=None):
        return ops.divergence(u, v, self.dx, self.dy, w=w, dz=getattr(self, 'dz', None))
    def normal_derivative(self, var, normal_x, normal_y, normal_z=None):
        return ops.normal_derivative(var, self.h, self.ndim, normal_x, normal_y, normal_z)
    def vorticity(self, u, v, w=None):
        return ops.vorticity(u, v, self.h, self.ndim, w=w)
    def vorticity_components(self, u, v, w):
        return ops.vorticity_components(u, v, w, self.h)

    def __init__(self, pars, dtype=torch.float32, custom_update=None, compute_forces=True):
        """
        BDIM2 solver for fluid structure interaction
        """
        solver    = pars["solver"]
        bcs       = pars["boundary_conditions"]
        output    = pars["output"]
        body_pars = pars["body"]

        use_gpu = solver["use_gpu"]
        if torch.cuda.is_available() and use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
            self.device = torch.device("cuda")
        else:
            print("Using the CPU.")
            self.device = torch.device("cpu")
            torch.set_num_threads(solver["nthreads"])

        self.dtype = dtype
        self.nx    = solver["Nx"]+2
        self.ny    = solver["Ny"]+2

        self.xmin  = solver["xmin"]
        self.xmax  = solver["xmax"]
        self.ymin  = solver["ymin"]
        self.ymax  = solver["ymax"]

        self.dx=(self.xmax-self.xmin)/(self.nx-2)
        self.dy=(self.ymax-self.ymin)/(self.ny-2)

        assert abs(float(self.dx-self.dy)) < 1e-10, "Grid spacing in x = {} and y = {} must be equal".format(self.dx, self.dy)
        self.h = torch.tensor(self.dx, device=self.device, dtype=self.dtype)

        self.x = torch.arange(self.xmin-self.h/2, self.xmax+self.h, self.h, device=self.device, dtype=self.dtype)
        self.y = torch.arange(self.ymin-self.h/2, self.ymax+self.h, self.h, device=self.device, dtype=self.dtype)

        # ---- 3D detection ----
        if "Nz" in solver:
            self.ndim = 3
            self.nz   = solver["Nz"] + 2
            self.zmin = solver["zmin"]
            self.zmax = solver["zmax"]
            self.dz   = (self.zmax - self.zmin) / (self.nz - 2)
            assert abs(float(self.dx - self.dz)) < 1e-10, \
                "Grid spacing in x = {} and z = {} must be equal".format(self.dx, self.dz)
            self.z = torch.arange(self.zmin - self.h/2, self.zmax + self.h, self.h,
                                  device=self.device, dtype=self.dtype)
            self.grid_shape = (self.nx, self.ny, self.nz)
        else:
            self.ndim = 2
            self.z = None
            self.grid_shape = (self.nx, self.ny)

        self.h2    = self.h**2
        self.h3    = self.h**3
        self.dt    = torch.tensor(solver["dt"], device=self.device, dtype=self.dtype)
        self.dt_np = self.dt.cpu().numpy()

        self.nt   = solver["nt"]
        self.nu   = torch.tensor(solver["nu"], device=self.device, dtype=self.dtype)   # kinematic viscosity

        self.rho  = torch.tensor(solver["rho"], device=self.device, dtype=self.dtype)  # density
        self.visc = self.nu*self.rho                                                   # dynamic viscosity

        self.eps  = solver.get("eps_multiplier",
                                torch.tensor(2.0, device=self.device, dtype=self.dtype)) * self.h

        self.starting_iteration      = solver.get("starting_iteration", 0)
        self.starting_iteration_path = solver.get("starting_iteration_path", None)
        self.starting_time           = self.starting_iteration * self.dt

        self.perturbation_amplitude  = solver.get("perturbation_amplitude", 0.0)

        # ============= Smagorinsky LES model =============
        self.smagorinsky_cs = solver.get("smagorinsky_cs", 0.0)
        self.use_smagorinsky = self.smagorinsky_cs > 0
        if self.use_smagorinsky:
            print(f"Smagorinsky LES model enabled: Cs = {self.smagorinsky_cs}")

        # ============= Carreau non-Newtonian model =============
        carreau = solver.get("carreau", None)
        if carreau is not None:
            self.use_carreau = True
            self.carreau_nu_0   = carreau["nu_0"]
            self.carreau_nu_inf = carreau["nu_inf"]
            self.carreau_lam    = carreau["lam"]
            self.carreau_n      = carreau["n"]
            self.carreau_tau_y  = carreau.get("tau_y", 0.0)
            # Diffusion CFL stability limit: Δt < h² / (2·ndim·ν_max)
            # → ν_max = safety · h² / (2·ndim·Δt),  safety = 0.4
            cfl_nu_max = 0.4 * float(self.h)**2 / (2.0 * self.ndim * float(self.dt))
            self.carreau_nu_max = carreau.get("nu_max", cfl_nu_max)
            model_name = "Herschel-Bulkley–Carreau" if self.carreau_tau_y > 0 else "Carreau"
            print(f"{model_name} model enabled: "
                  f"nu_0={self.carreau_nu_0}, nu_inf={self.carreau_nu_inf}, "
                  f"lam={self.carreau_lam}, n={self.carreau_n}")
            if self.carreau_tau_y > 0:
                print(f"  yield stress tau_y={self.carreau_tau_y} Pa, "
                      f"nu_max={self.carreau_nu_max:.4e} (CFL limit={cfl_nu_max:.4e})")
            if self.use_smagorinsky:
                raise ValueError("Cannot use both Smagorinsky and Carreau simultaneously.")
            # Override self.nu to nu_inf so that the nu_t pathway
            # (nu_eff = self.nu + nu_t) yields nu_t >= 0 everywhere.
            # The user-supplied "nu" in the config is ignored when Carreau is
            # active — the zero-shear and infinite-shear viscosities are
            # specified entirely by the Carreau parameters.
            if float(self.nu) != self.carreau_nu_inf:
                print(f"  [Carreau] Overriding solver nu={float(self.nu):.6e} "
                      f"→ nu_inf={self.carreau_nu_inf:.6e} for consistency.")
                self.nu   = torch.tensor(self.carreau_nu_inf, device=self.device, dtype=self.dtype)
                self.visc = self.nu * self.rho
        else:
            self.use_carreau = False

        # Whether any variable-viscosity model is active
        self.use_variable_viscosity = self.use_smagorinsky or self.use_carreau
        # Cached spatially-varying ν·ρ field for force computation
        # (set each step when use_variable_viscosity is True, else None)
        self._nu_rho_field = None

        # ============= Yield-stress damping =============
        # Implicit penalty that drives velocity toward zero in unyielded
        # (low-shear) regions, mimicking the solid-like behaviour of a
        # yield-stress fluid.  Enabled only when the user explicitly sets
        # a "yield_damping" dict in the solver config:
        #   yield_damping:
        #     gamma_c  : 0.0625   # critical strain rate [1/s]
        #     strength : 1.1      # max damping coefficient σ_max [1/s]
        yield_damping_cfg = solver.get("yield_damping", None)
        if yield_damping_cfg is not None:
            self.use_yield_damping = True
            self._yield_gamma_c  = yield_damping_cfg["gamma_c"]
            self._yield_strength = yield_damping_cfg["strength"]
            print(f"Yield-stress damping enabled: "
                  f"gamma_c={self._yield_gamma_c:.4f} s^-1, "
                  f"strength={self._yield_strength:.4f} s^-1")
        else:
            self.use_yield_damping = False

        # ============= Sponge / damping layer =============
        sponge = solver.get("sponge", None)
        if sponge is not None:
            self.use_sponge = True
            sponge_width    = sponge.get("width", 0.15)    # [m] thickness of the sponge layer
            sponge_strength = sponge.get("strength", 50.0)  # [1/s] max damping coefficient σ_max
            sponge_axes     = sponge.get("axes", None)      # None = all axes, or list e.g. ["x"]
            # Build quadratic σ(x,y,z) fields on each staggered grid.
            # σ = σ_max · (max(0, Ls - d) / Ls)²
            # where d = distance from the nearest domain boundary.
            self._sponge_sigma_u, self._sponge_sigma_v, self._sponge_sigma_w = \
                self._build_sponge_fields(sponge_width, sponge_strength, axes=sponge_axes)
            axes_str = ",".join(sponge_axes) if sponge_axes else "all"
            print(f"Sponge layer enabled: width={sponge_width} m, strength={sponge_strength} 1/s, axes={axes_str}")
        else:
            self.use_sponge = False
            self._sponge_sigma_u = None
            self._sponge_sigma_v = None
            self._sponge_sigma_w = None

        # ============= time integration =============
        self.time_integration = solver.get("time_integration", "euler")
        assert self.time_integration in ("heun", "euler"), \
            f"Unknown time_integration '{self.time_integration}'. Choose 'heun' or 'euler'."
        # ---- time integration dispatch (set once, used every step) ----
        self._solve = self.solve_heun if self.time_integration == "heun" else self.solve_euler

        print("Setting dt={}s, dx={}".format(self.dt, self.h))
        print(f"Time integration: {self.time_integration}")

        # ============= convection solver =============
        adv_diff_kwargs = dict(
            BC_type_u=bcs["BC_type_u"], BC_values_u=bcs["BC_values_u"],
            BC_type_v=bcs["BC_type_v"], BC_values_v=bcs["BC_values_v"],
            method=solver["convection_method"],
        )
        if self.ndim == 3:
            adv_diff_kwargs.update(
                z=self.z,
                BC_type_w=bcs.get("BC_type_w", ("D", "D")),
                BC_values_w=bcs.get("BC_values_w", (0.0, 0.0)),
            )
        self.adv_diff_solver = AdvDiffSolver(
            self.device, self.dt, self.x, self.y, self.nu,
            **adv_diff_kwargs,
        )

        # ---- optional torch.compile for adv-diff + BDIM kernels -----
        self._compile_adv_diff = solver.get("compile_adv_diff", False)
        if self._compile_adv_diff and self.device.type == "cuda":
            self.adv_diff_solver.solve = torch.compile(
                self.adv_diff_solver.solve, mode="reduce-overhead",
            )
            self._bdim_meta_compiled = torch.compile(
                FluidSolver._bdim_meta, mode="reduce-overhead",
            )
            # Dynamic-shape variant for the union-AABB crop path
            # (sub-block shape varies with body kinematics).
            self._bdim_meta_dyn_compiled = torch.compile(
                FluidSolver._bdim_meta, dynamic=True,
            )
            print("  [compile] adv_diff_solver.solve + BDIM meta-equation compiled (reduce-overhead)")
        else:
            self._bdim_meta_compiled = FluidSolver._bdim_meta
            self._bdim_meta_dyn_compiled = FluidSolver._bdim_meta

        # ---- optional Towers (2008) 2nd-order delta correction -----------
        # When force_delta_order=2, the smoothed delta is divided by |∇SDF|
        # so that the volume integral gives the correct surface measure even
        # when the numerical SDF deviates from unit gradient.
        # For analytical bodies |∇SDF|=1 exactly, so order 2 is a no-op;
        # it matters for mesh bodies or near geometric corners.
        self.force_delta_order = int(solver.get("force_delta_order", 1))
        if self.force_delta_order not in (1, 2):
            raise ValueError(f"force_delta_order must be 1 or 2, got {self.force_delta_order}")

        # ---- optional torch.compile for force computation -----
        self._compile_forces = solver.get("compile_forces", False)
        # Narrow-band forces: when ``comp._sdf_sparse`` is available (set by
        # BDIMhandler 3-D update), integrate each body's surface forces only
        # over its AABB sub-block (~50³ cells) instead of scattering to a
        # full (B, Nx, Ny, Nz) dense tensor and reducing over all cells.
        # Enabled by default; disable via solver.force_narrow_band=False
        # to fall back to the old dense batched path.
        self._forces_narrow_band = bool(solver.get("force_narrow_band", True))
        # Batched narrow-band forces: pack all per-body AABB sub-blocks into
        # a single fixed-shape tensor (B, D_i, D_j, D_k) and dispatch a single
        # compiled CUDA-graph kernel instead of B separate dynamic-shape launches.
        # D is determined once at init from the body-local SDF bbox diagonal
        # (rotation-invariant worst case).  Opt-in until benchmarked.
        self._forces_narrow_batch = bool(solver.get("force_narrow_batch", False))
        if self._forces_narrow_batch and not self._forces_narrow_band:
            # batched path requires the sparse SDF infrastructure
            self._forces_narrow_band = True
        # Shared-stress union-AABB crop: run the bandwidth-bound shared
        # gradient + stress + pressure kernel only over the union AABB
        # of all body sub-blocks (plus halo) instead of the full grid.
        # On large grids the full-grid shared kernel is the dominant cost;
        # cropping to the union AABB (~10× smaller cells for thin swimmers)
        # cuts memory traffic proportionally.  Opt-in.
        self._forces_shared_union = bool(solver.get("force_shared_union", False))
        if self._forces_shared_union and not self._forces_narrow_band:
            self._forces_narrow_band = True
        # mu + normals union-AABB crop: run the batched mu/normal kernel
        # only over the union AABB of all body sub-blocks (with halo for
        # gradient stencils) instead of the full 4×(Nx,Ny,Nz) grid.
        # Outside the union SDF equals _FAR, so mu0=1, mu1=0, normals=0;
        # persistent full-grid buffers are pre-filled with those defaults
        # and only the union sub-block is overwritten each step.  Opt-in.
        self._mu_normals_union = bool(solver.get("mu_normals_union", False))
        self._mu_union_ready = False   # persistent buffers allocated lazily
        # BDIM meta-equation union-AABB crop: outside the union mu0=1,
        # mu1=0, body_vel=0, so the meta-equation is the identity
        # (phi_out = phi).  We can therefore run the elementwise kernel
        # only on the union sub-block (with a 1-cell halo for the normal
        # derivative) and slice-write the result into phi.  Opt-in.
        self._bdim_union = bool(solver.get("bdim_union", False))
        # Phase-B fused-CUDA streaming SDF / face-velocity update.
        # One C++/CUDA kernel per body fuses rotate + 4 trilinear samples
        # + 4 running-min updates + per-body sparse cc store.
        # Internally re-uses the per-body custom-trilinear samplers
        # (pytorch_interpolation / RegularGridInterpolator3D), which are
        # auto-built when this is on (see ``_custom_trilinear_3d`` below).
        # Opt-in.
        self._streaming_sdf_3d = bool(solver.get("streaming_sdf_3d", False))
        # The streaming path requires the per-body C++/CUDA trilinear
        # samplers; this flag is now an internal derivative of
        # ``streaming_sdf_3d`` (no separate user-facing toggle).
        self._custom_trilinear_3d = self._streaming_sdf_3d
        # Phase D: fused per-body force / torque integration.  Re-samples
        # body SDF on the fly inside the same C++/CUDA op that does the
        # delta-function reduction, eliminating the slice-write packing
        # into ``_fnb_*`` buffers and the dense (B, D_i, D_j, D_k)
        # narrow-batch kernel.  Requires ``streaming_sdf_3d`` + the
        # union-AABB shared-stress crop (``force_shared_union``).
        self._streaming_forces_3d = bool(solver.get("streaming_forces_3d", False))
        if self._streaming_forces_3d:
            self._streaming_sdf_3d = True
            self._custom_trilinear_3d = True
            self._forces_shared_union = True
            self._forces_narrow_band = True
        # Lazy-allocated padded buffers (see _init_forces_narrow_batch)
        self._fnb_D = None        # (D_i, D_j, D_k)
        self._fnb_sdf = None      # (B, D_i, D_j, D_k)
        self._fnb_xs = None
        self._fnb_ys = None
        self._fnb_zs = None
        self._fnb_px = None
        self._fnb_py = None
        self._fnb_pz = None
        self._fnb_X = None
        self._fnb_Y = None
        self._fnb_Z = None
        if self._compile_forces and self.device.type == "cuda":
            # 3-D kernels
            self._forces_shared_compiled = torch.compile(
                _forces_shared_3d, mode="reduce-overhead",
            )
            # ``_forces_body_integrate_3d`` runs on per-body AABB sub-blocks
            # whose shapes change slowly as the body rotates.  Use
            # ``dynamic=True`` so we get kernel fusion without a recompile
            # for every orientation (CUDA-graph mode is incompatible with
            # variable shapes).
            self._forces_body_compiled = torch.compile(
                _forces_body_integrate_3d, dynamic=True,
            )
            self._forces_body_batch_compiled = torch.compile(
                _forces_body_batch_3d, mode="reduce-overhead",
            )
            self._forces_body_narrow_batch_compiled = torch.compile(
                _forces_body_narrow_batch_3d, mode="reduce-overhead",
            )
            # Dynamic-shape shared kernel for the union-AABB crop path
            # (sub-block shape varies with body kinematics).
            self._forces_shared_dyn_compiled = torch.compile(
                _forces_shared_3d, dynamic=True,
            )
            # Dynamic-shape batched mu/normals kernel for the union-AABB
            # crop path (same rationale).
            self._mu_normals_batched_3d_dyn_compiled = torch.compile(
                _mu_normals_batched_3d, dynamic=True,
            )
            # 2-D kernels
            self._forces_shared_2d_compiled = torch.compile(
                _forces_shared_2d, mode="reduce-overhead",
            )
            self._forces_body_batch_2d_compiled = torch.compile(
                _forces_body_batch_2d, mode="reduce-overhead",
            )
            print("  [compile] forces_shared + forces_body_batch compiled "
                  "(reduce-overhead, 2D+3D)"
                  + ("  + narrow-band per-body path (dynamic)" if self._forces_narrow_band else "")
                  + ("  + narrow-band BATCHED (reduce-overhead)" if self._forces_narrow_batch else "")
                  + ("  + shared-stress UNION crop (dynamic)" if self._forces_shared_union else "")
                  + ("  + mu/normals UNION crop (dynamic)" if self._mu_normals_union else "")
                  + ("  + BDIM meta UNION crop (dynamic)" if self._bdim_union else "")
                  + ("  + streaming fused-CUDA SDF update (Phase C: multi-body batched)" if self._streaming_sdf_3d else "")
                  + ("  + Phase D: fused forces (re-sample SDF on the fly)" if self._streaming_forces_3d else ""))
        else:
            self._forces_shared_compiled = _forces_shared_3d
            self._forces_body_compiled = _forces_body_integrate_3d
            self._forces_body_batch_compiled = _forces_body_batch_3d
            self._forces_body_narrow_batch_compiled = _forces_body_narrow_batch_3d
            self._forces_shared_dyn_compiled = _forces_shared_3d
            self._mu_normals_batched_3d_dyn_compiled = _mu_normals_batched_3d
            self._forces_shared_2d_compiled = _forces_shared_2d
            self._forces_body_batch_2d_compiled = _forces_body_batch_2d

        # ---- optional torch.compile for SDF / mu / normals ----
        self._compile_sdf = solver.get("compile_sdf", False)
        if self._compile_sdf and self.device.type == "cuda":
            print("  [compile] SDF rotation, staggering, mu+normals compiled (reduce-overhead)")

        # =============  poisson solver =============
        self.poisson_method = solver.get("poisson_method", "multigrid")
        assert self.poisson_method in ("multigrid", "mgcg", "fft"), \
            f"Unknown poisson_method '{self.poisson_method}'. Choose 'multigrid', 'mgcg', or 'fft'."
        print(f"Poisson solver: {self.poisson_method}")

        self.poisson_solver  = PoissonSolver(
            self.dtype,
            self.device,
            self.h,
            tol             = solver["poisson_tol"],
            max_cycles      = solver["poisson_max_mgcg_cycles"],
            max_vcycles     = solver["poisson_max_cycles"],
            nsmoothing      = solver["poisson_nsmoothing"],
            w               = solver["jacobi_weight"],
            verbose         = solver["poisson_verbose"],
            precond_vcycles = solver.get("poisson_precond_vcycles", 1),
            smoother        = solver.get("poisson_smoother", "rbgs"),
            compile_smoother= solver.get("poisson_compile", False),
        )

        # Warm-start: reuse previous pressure as Poisson initial guess
        self.poisson_warm_start = solver.get("poisson_warm_start", False)

        # Only build the FFT solver when it will actually be used.
        # Gfft + U buffer cost ~834 MB on a 512x128x128 grid.
        self.poisson_bc_type = solver.get("poisson_bc_type", "free")
        assert self.poisson_bc_type in ("free", "neumann"), \
            f"Unknown poisson_bc_type '{self.poisson_bc_type}'. Choose 'free' or 'neumann'."

        if self.poisson_method == "fft":
            fft_kwargs = dict(
                bc_type  = self.poisson_bc_type,
                filename = solver["poisson_folder"],
            )
            if self.ndim == 2:
                self.poisson_solverFFT = PoissonSolverFFT(
                    self.x, self.y, **fft_kwargs,
                )
            else:
                self.poisson_solverFFT = PoissonSolverFFT(
                    self.x, self.y, z=self.z, **fft_kwargs,
                )
        else:
            self.poisson_solverFFT = None

        # ---- staggered grids (shared by all bodies) ------------------
        self.grids = _StaggeredGrids(self.x, self.y, self.z)

        self.composite_body = body_from_yaml(
            self.device,
            self.x, self.y,
            body_pars,
            z             = self.z,
            eps           = self.eps,
            custom_update = custom_update,
            starting_time = self.starting_time,
            grids         = self.grids,
        )


        self.X, self.Y = self.grids.X, self.grids.Y
        if self.ndim == 3:
            self.Z_grid = self.grids.Z_grid


        # interpolators (2D only — used by force computation)
        if self.ndim == 2:
            self.force_x_interp = RegularGridInterpolator(
                (self.grids.x_stag, self.y),
                torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
                method=1,
                fill_value=None
            )
            self.force_y_interp = RegularGridInterpolator(
                (self.x, self.grids.y_stag),
                torch.zeros_like(self.Y, device=self.device, dtype=self.dtype),
                method=1,
                fill_value=None
            )

            self.interp_utility = RegularGridInterpolator(
                (self.x,self.y),
                torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
                method=1,
                fill_value=None
            )

        self.n_bodies = len(self.composite_body.bodies)

        # low dimensional utilities
        n_force_comp = self.ndim                # 2 or 3
        n_torque_comp = 1 if self.ndim == 2 else 3  # scalar in 2D, vector in 3D
        self.friction_force_lin_x = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_lin_y = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_ang_z = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.force_x_int          = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.force_y_int          = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_x     = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_y     = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_ang_z = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        if self.ndim == 3:
            self.friction_force_lin_z  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.force_z_int           = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.pressure_force_z      = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            # torque is a 3-vector in 3D
            self.friction_force_ang_x  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.friction_force_ang_y  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.pressure_force_ang_x  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.pressure_force_ang_y  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.viscous_drag_record  = torch.zeros((self.n_bodies,n_force_comp,self.nt),device=self.device,dtype=self.dtype)
        self.pressure_drag_record = torch.zeros((self.n_bodies,n_force_comp,self.nt),device=self.device,dtype=self.dtype)
        # Torques about each body's COM (scalar in 2-D, 3-vector in 3-D).
        # Populated only in 3-D BDIM paths for now.
        self.viscous_torque_record  = torch.zeros((self.n_bodies,n_torque_comp,self.nt),device=self.device,dtype=self.dtype)
        self.pressure_torque_record = torch.zeros((self.n_bodies,n_torque_comp,self.nt),device=self.device,dtype=self.dtype)

        # ===== flow diagnostics (energy, enstrophy, CFL) =====
        _diag_every  = solver.get("diagnostics_every", 1)
        _energy_grow = solver.get("energy_growth_factor", 10.0)
        self.diagnostics = FlowDiagnostics(
            nt      = self.nt,
            ndim    = self.ndim,
            h       = self.h,
            device  = self.device,
            dtype   = self.dtype,
            check_every          = _diag_every,
            energy_growth_factor = _energy_grow,
        )

        # NOTE: xstress_tensor, ystress_tensor, zstress_tensor, and div
        # are created on-the-fly in forces_method* and project() respectively
        # (no pre-allocation needed — they are rebound, not written in-place).

          # ===== set initial conditions =====
        self.set_initial_conditions()

          # ===== plotting parameters =====
        self.extent = (
            self.xmin-self.h/2, self.xmax+self.h/2,
            self.ymin-self.h/2, self.ymax+self.h/2
        )
        self.extent_vstag = (
            self.xmin-self.h/2, self.xmax+self.h/2,
            self.ymin-self.h, self.ymax
        )
        self.extent_ustag = (
            self.xmin-self.h, self.xmax,
            self.ymin-self.h/2, self.ymax+self.h/2
        )
        self.extent_curl = (
            self.xmin-self.h, self.xmax,
            self.ymin-self.h, self.ymax
        )



        self.compute_forces = compute_forces

          # ===== create folder for frames' storage ====
        self.save_frames      = output["save_frames"]
        self.save_every       = output["save_every"]
        # Unified save flag (replaces old save_uv + save_vtk).
        # Backward compat: accept legacy keys from old YAML configs.
        self.save             = output.get("save",
                                    output.get("save_uv", False)
                                    or output.get("save_vtk", False))
        # vmin/vmax: a number → fixed colour limits; "auto" → auto-scale per field
        _vmin = output["vmin"]
        _vmax = output["vmax"]
        self.vmin = None if _vmin == "auto" else _vmin
        self.vmax = None if _vmax == "auto" else _vmax
        self.plot_specs = self._resolve_plot_specs(output.get("plot_specs"))
        self.iso_3d_specs = self._resolve_iso_3d_specs(
            output.get("iso_3d_specs"),
            global_iso_value=output.get("iso_3d_value"),
        )
        self.n_quiver_spacing = 2**3

        # Background thread pool for async I/O (saving + plotting)
        self._io_executor = ThreadPoolExecutor(max_workers=2)
        self._io_futures = []  # track pending I/O tasks
        self._hdf5_lock  = threading.Lock()  # serialise HDF5 writes

        if self.save_frames or self.save:
            path = output["save_path"]
            if "existing_folder" in output:
                results_folder = output["existing_folder"]
            else:
                today          = datetime.datetime.now()
                todaystr       = today.isoformat()
                results_folder = f'{path}{todaystr}'
            os.makedirs(results_folder, exist_ok=True)

            print(f"Frames will be saved in folder: {results_folder}/")

            self.save_path = results_folder+"/"

              # Add save path to the parameters
            pars["output"]["existing_folder"] = results_folder

              # Save body signal (if available)
            if getattr(self.composite_body, 'save_signal', None):
                self.composite_body.save_signal(results_folder)

              # Save the parameters as a yaml file
            pyobject2yaml(
                filename = self.save_path+"parameters.yaml",
                pyobject = pars,
            )

    def inside(self, x):
        """
        Return True if all elements in x are inside the domain
        """
        in_xy = torch.logical_and(
            x[:,0]>self.xmin,
            torch.logical_and(
                x[:,0]<self.xmax,
                torch.logical_and(
                    x[:,1]>self.ymin,
                    x[:,1]<self.ymax
                )
            )
        )
        if self.ndim == 3:
            in_xy = torch.logical_and(
                in_xy,
                torch.logical_and(x[:,2]>self.zmin, x[:,2]<self.zmax)
            )
        return torch.all(in_xy)

    def _load_initial_conditions(self):
        ''' Load initial conditions from a previous simulation '''

        if not self.starting_iteration_path:
            return False

        v0_path = f'{self.starting_iteration_path}/uv_field/v_{self.starting_iteration}.npy'
        u0_path = f'{self.starting_iteration_path}/uv_field/u_{self.starting_iteration}.npy'

          # Verify files
        if not os.path.exists(v0_path) or not os.path.exists(u0_path):
            raise FileNotFoundError(f'Initial conditions not found at {v0_path} or {u0_path}')

        u0 = torch.tensor(np.load(u0_path)).to(device=self.device, dtype=self.dtype)
        v0 = torch.tensor(np.load(v0_path)).to(device=self.device, dtype=self.dtype)
        p0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

          # Verify shape
        assert u0.shape == tuple(self.grid_shape), f"u0 shape: {u0.shape} != {self.grid_shape}"
        assert v0.shape == tuple(self.grid_shape), f"v0 shape: {v0.shape} != {self.grid_shape}"

          # Loaded
        self.u0, self.v0, self.p0 = u0, v0, p0
        if self.ndim == 3:
            w0_path = f'{self.starting_iteration_path}/uv_field/w_{self.starting_iteration}.npy'
            if os.path.exists(w0_path):
                self.w0 = torch.tensor(np.load(w0_path)).to(device=self.device, dtype=self.dtype)
            else:
                self.w0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        return True

    def set_initial_conditions(self):
        """
        initial conditions
        """

          # Load initial conditions
        if self._load_initial_conditions():
            return

        # Set initial conditions
        if self.adv_diff_solver.BC_type_u[0]=="D":
            self.u0 = self.adv_diff_solver.BC_values_u[0]*torch.ones(self.grid_shape, device=self.device, dtype=self.dtype)
        else:
            self.u0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.v0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.p0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        if self.ndim == 3:
            self.w0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        # Add symmetry-breaking perturbation to v (cross-stream) to trigger
        # vortex shedding instead of relying on floating-point roundoff.
        if self.perturbation_amplitude > 0:
            print(f"Adding random perturbation to v0 with amplitude {self.perturbation_amplitude}")
            self.v0 += self.perturbation_amplitude * (2 * torch.rand(self.grid_shape, device=self.device, dtype=self.dtype) - 1)

        # Mask the initial velocity inside the body using the smooth BDIM
        # mask (mu0 = 1 outside, 0 inside).  Without this, the uniform
        # freestream fills the body interior and the first BDIM step creates
        # an impulsive discontinuity that produces a spurious pressure spike
        # and artificial vorticity at t=0.  Each staggered velocity component
        # is masked with its own staggered-grid mu0 for consistency with the
        # rest of the BDIM pipeline.
        if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
            cb = self.composite_body
            mu0_u, _ = cb.mu_funcs(cb.sdf_val_u)
            mu0_v, _ = cb.mu_funcs(cb.sdf_val_v)
            self.u0 = self.u0 * mu0_u
            self.v0 = self.v0 * mu0_v
            if self.ndim == 3:
                mu0_w, _ = cb.mu_funcs(cb.sdf_val_w)
                self.w0 = self.w0 * mu0_w


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

        # Towers (2008) 2nd-order: compute per-body |∇SDF| on CC grid  (B,Ni,Nj)
        sdf_grad_mag_2d = None
        if self.force_delta_order == 2:
            gx = torch.gradient(comp.sdf_vals, spacing=self.h, dim=1, edge_order=2)[0]
            gy = torch.gradient(comp.sdf_vals, spacing=self.h, dim=2, edge_order=2)[0]
            sdf_grad_mag_2d = torch.sqrt(gx**2 + gy**2)

        (fv_x, fv_y, tv_z,
         fp_x, fp_y, tp_z) = self._forces_body_batch_2d_compiled(
            xstress, ystress,
            pforce_x, pforce_y,
            comp.sdf_vals,
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
    def _init_forces_narrow_batch(self, comp, h):
        """Pre-allocate fixed-shape padded buffers for the batched
        narrow-band forces path.

        Mirrors the 2-D ``BDIMhandler._init_batched_sdf_2d`` pattern:
        find the worst-case per-body sub-block size, pad every body to
        the common shape ``(D_i, D_j, D_k)`` with a sentinel, and stack
        into a single ``(B, D_i, D_j, D_k)`` tensor so that a single
        CUDA-graph launch integrates forces for all bodies regardless
        of current orientation.

        Per-axis ``D_a`` is derived once from the body-local SDF bbox
        diagonal (rotation-invariant upper bound on the world-space
        AABB extent), plus a safety margin for the ``pad`` cells added
        by ``_body_aabb_indices``.
        """
        B = len(comp.bodies)
        # Body-local SDF bbox diagonal → rotation-invariant cell bound.
        # In the extreme case a diagonally-oriented body occupies a cube
        # of side = diag_len in every axis, so we use that for all 3 axes.
        max_cells = 0
        for body in comp.bodies:
            sdf_lo = torch.tensor([float(body.sdf.x[0]),
                                   float(body.sdf.y[0]),
                                   float(body.sdf.z[0])])
            sdf_hi = torch.tensor([float(body.sdf.x[-1]),
                                   float(body.sdf.y[-1]),
                                   float(body.sdf.z[-1])])
            diag_len = float(torch.norm(sdf_hi - sdf_lo))
            # + 2*pad(=3) + 2 margin cells against AABB growth between steps
            n_cells = int(diag_len / h) + 2 * 3 + 2 + 1
            if n_cells > max_cells:
                max_cells = n_cells

        D = max_cells
        self._fnb_D = (D, D, D)

        dev, dt = self.device, self.dtype
        shape = (B, D, D, D)
        # SDF padded to _FAR so smoothed delta vanishes at padded cells.
        _FAR = 1e4
        self._fnb_sdf = torch.full(shape, _FAR, device=dev, dtype=dt)
        self._fnb_xs = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_ys = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_zs = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_px = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_py = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_pz = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_X = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_Y = torch.zeros(shape, device=dev, dtype=dt)
        self._fnb_Z = torch.zeros(shape, device=dev, dtype=dt)
        print(f"  [forces-narrow-batch] padded buffers "
              f"(B={B}, D={D}³)  "
              f"{self._fnb_sdf.nelement() * self._fnb_sdf.element_size() / 1e6:.1f} MB each")

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
        nu_rho = self._compute_nu_rho_for_forces(u, v, w)
        h      = self.h
        h3     = self.h3
        comp   = self.composite_body

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
        _use_narrow = self._forces_narrow_band and _have_sparse
        _use_narrow_batch = self._forces_narrow_batch and _have_sparse

        # ---- Phase D: fused per-body force integration --------------
        _stream_step = getattr(comp, '_stream_multi_step', None)
        _stream_static = getattr(comp, '_stream_multi_static', None)
        _use_streaming_forces = (
            self._streaming_forces_3d
            and _have_sparse
            and _stream_step is not None
            and _stream_static is not None
            and self.force_delta_order == 1
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

        if _use_narrow_batch:
            # ---- BATCHED narrow-band path (fixed padded shape) ------
            # Pack every body's AABB sub-block into persistent
            # (B, D_i, D_j, D_k) buffers and dispatch a single CUDA-graph
            # launch.  Uses the same max-over-shapes-then-pad-with-sentinel
            # pattern as the 2-D batched SDF (BDIMhandler._init_batched_sdf_2d).
            if self._fnb_sdf is None:
                self._init_forces_narrow_batch(comp, h)

            _FAR = 1e4
            D_i, D_j, D_k = self._fnb_D

            # --- per-body slice-write packing (contiguous strided copy) -
            # Persistent buffers let the compiled kernel replay from its
            # captured CUDA graph with zero input memcpy.  Slice writes
            # are far cheaper than advanced-indexing gather here because
            # (a) each write is one contiguous kernel and (b) we avoid
            # re-allocating tensors each step.
            self._fnb_sdf.fill_(_FAR)
            fallback_full = False
            # When the shared kernel ran cropped to the union AABB, its
            # outputs are in union-relative coordinates.  Per-body AABBs
            # from ``_sdf_sparse`` are in absolute fluid-grid indices, so
            # we translate them by the union origin before indexing.
            if u_aabb is not None:
                _su_i0, _, _su_j0, _, _su_k0, _ = u_aabb
            else:
                _su_i0 = _su_j0 = _su_k0 = 0
            for bi in range(B):
                aabb_i, sdf_sub_i = comp._sdf_sparse[bi]
                if aabb_i is None:
                    fallback_full = True
                    break
                i0, i1, j0, j1, k0, k1 = aabb_i
                di, dj, dk = i1 - i0, j1 - j0, k1 - k0
                if di > D_i or dj > D_j or dk > D_k:
                    print(f"  [forces-narrow-batch] body {bi} AABB "
                          f"({di},{dj},{dk}) > D={self._fnb_D}; "
                          f"rebuilding buffers (triggers CUDA-graph recompile)")
                    self._fnb_sdf = None
                    self._init_forces_narrow_batch(comp, h)
                    self._fnb_sdf.fill_(_FAR)
                    D_i, D_j, D_k = self._fnb_D
                sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
                # Stress / pforce are in union-relative coords.
                sl_stress = (
                    slice(i0 - _su_i0, i1 - _su_i0),
                    slice(j0 - _su_j0, j1 - _su_j0),
                    slice(k0 - _su_k0, k1 - _su_k0),
                )
                self._fnb_sdf[bi, :di, :dj, :dk] = sdf_sub_i
                self._fnb_xs[bi, :di, :dj, :dk] = xstress[sl_stress]
                self._fnb_ys[bi, :di, :dj, :dk] = ystress[sl_stress]
                self._fnb_zs[bi, :di, :dj, :dk] = zstress[sl_stress]
                self._fnb_px[bi, :di, :dj, :dk] = pforce_x[sl_stress]
                self._fnb_py[bi, :di, :dj, :dk] = pforce_y[sl_stress]
                self._fnb_pz[bi, :di, :dj, :dk] = pforce_z[sl_stress]
                self._fnb_X[bi, :di, :dj, :dk] = X[sl]
                self._fnb_Y[bi, :di, :dj, :dk] = Y[sl]
                self._fnb_Z[bi, :di, :dj, :dk] = Z[sl]

            if fallback_full:
                _use_narrow_batch = False

            if _use_narrow_batch:
                grad_mag_b = None
                if self.force_delta_order == 2:
                    gx = torch.gradient(self._fnb_sdf, spacing=h, dim=1, edge_order=2)[0]
                    gy = torch.gradient(self._fnb_sdf, spacing=h, dim=2, edge_order=2)[0]
                    gz = torch.gradient(self._fnb_sdf, spacing=h, dim=3, edge_order=2)[0]
                    grad_mag_b = torch.sqrt(gx**2 + gy**2 + gz**2)

                eps_body = comp.bodies[0].eps
                (fv_x, fv_y, fv_z,
                 tv_x, tv_y, tv_z,
                 fp_x, fp_y, fp_z,
                 tp_x, tp_y, tp_z) = self._forces_body_narrow_batch_compiled(
                    self._fnb_xs, self._fnb_ys, self._fnb_zs,
                    self._fnb_px, self._fnb_py, self._fnb_pz,
                    self._fnb_sdf, eps_body, self.eps,
                    comp.com_pos[:, 0], comp.com_pos[:, 1], comp.com_pos[:, 2],
                    self._fnb_X, self._fnb_Y, self._fnb_Z, h3,
                    grad_mag_b,
                )
                # Clone CUDA-graph outputs
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
                return

        if _use_narrow:
            # ---- NARROW-BAND per-body path (preferred) --------------
            # Integrate each body on its own AABB sub-block.  Inner kernel
            # is the same as the uncompiled fallback, optionally compiled
            # with dynamic=True so varying AABB shapes don't recompile.
            # When the union-AABB shared-stress crop is active, stress &
            # pforce live on the cropped sub-block: shift per-body indices
            # by the union origin to slice them.  X/Y/Z stay global.
            if u_aabb is not None:
                ui0, _, uj0, _, uk0, _ = u_aabb
            for i, body in enumerate(comp.bodies):
                eps_body = body.eps

                aabb_i, sdf_sub_i = comp._sdf_sparse[i]
                if aabb_i is not None:
                    i0, i1, j0, j1, k0, k1 = aabb_i
                    sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
                    if u_aabb is not None:
                        sl_s = (slice(i0 - ui0, i1 - ui0),
                                slice(j0 - uj0, j1 - uj0),
                                slice(k0 - uk0, k1 - uk0))
                    else:
                        sl_s = sl
                    xs_i, ys_i, zs_i = xstress[sl_s], ystress[sl_s], zstress[sl_s]
                    px_i, py_i, pz_i = pforce_x[sl_s], pforce_y[sl_s], pforce_z[sl_s]
                    X_i, Y_i, Z_i    = X[sl], Y[sl], Z[sl]
                else:
                    # Body covers >90% of grid → full-grid evaluation.
                    xs_i, ys_i, zs_i = xstress, ystress, zstress
                    px_i, py_i, pz_i = pforce_x, pforce_y, pforce_z
                    X_i, Y_i, Z_i    = X, Y, Z

                grad_mag_i = None
                if self.force_delta_order == 2:
                    gx = torch.gradient(sdf_sub_i, spacing=h, dim=0, edge_order=2)[0]
                    gy = torch.gradient(sdf_sub_i, spacing=h, dim=1, edge_order=2)[0]
                    gz = torch.gradient(sdf_sub_i, spacing=h, dim=2, edge_order=2)[0]
                    grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)

                (fv_x, fv_y, fv_z,
                 tv_x, tv_y, tv_z,
                 fp_x, fp_y, fp_z,
                 tp_x, tp_y, tp_z) = self._forces_body_compiled(
                    xs_i, ys_i, zs_i,
                    px_i, py_i, pz_i,
                    sdf_sub_i, eps_body, self.eps,
                    body.com_pos[0], body.com_pos[1], body.com_pos[2],
                    X_i, Y_i, Z_i, h3,
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


    def project(self, u, v, p, w_vel=None, w=1.0, *,
                ch=None, cv=None, cw=None, ch_cc=None):
        """Pressure-Poisson projection.

        Parameters
        ----------
        u, v, p : tensors
            Velocity & pressure fields.
        w_vel : tensor or None
            z-velocity (3-D only).
        w : float
            Heun weight (1.0 = predictor, 0.5 = corrector).
        ch, cv, cw : tensor or None
            Pre-computed Poisson coefficients for each staggered grid
            (``dt / rho_eff`` on the respective face grids).
            When *None* (default), the standard BDIM coefficients
            ``(w*dt/rho) * mu0`` are used.  Pass custom coefficients
            for variable-density formulations (e.g. FARMS coupling where
            ``ch = dt / (rho_body + drho * mu0_u)``).
        ch_cc : tensor or None
            Cell-centred coefficient ``dt / rho_eff_cc`` for the FFT
            Poisson RHS.  When provided the FFT path solves
            ``∇²p = div / ch_cc`` (i.e. ``div * rho_eff_cc / dt``) and
            then corrects using the staggered *ch/cv/cw*.  When *None*
            (default) the FFT path falls back to a single scalar
            coefficient (constant-density behaviour).
        """

        # for general deforming bodies
        if self.ndim == 2:
            self.div  = self.divergence(u, v)
        else:
            self.div  = self.divergence(u, v, w_vel)

        coeff = w * self.dt / self.rho

        if self.poisson_method == "fft":
            # ---- FFT solver ----
            if ch_cc is not None:
                # Variable-density path: RHS uses cell-centred density,
                # correction uses the staggered ch/cv/cw coefficients.
                p = self.poisson_solverFFT.solve(self.div / ch_cc)
                if self.ndim == 2:
                    (p_x, p_y) = self.gradient(p)
                    u = u - ch * p_x
                    v = v - cv * p_y
                else:
                    (p_x, p_y, p_z) = self.gradient(p)
                    u     = u - ch * p_x
                    v     = v - cv * p_y
                    w_vel = w_vel - cw * p_z
            else:
                # Constant-density fallback: single scalar coefficient.
                # If ch was provided, use it as the scalar (backward compat).
                fft_coeff = coeff if ch is None else ch
                p = self.poisson_solverFFT.solve(self.div / fft_coeff)
                if self.ndim == 2:
                    (p_x, p_y) = self.gradient(p)
                    u = u - fft_coeff * p_x
                    v = v - fft_coeff * p_y
                else:
                    (p_x, p_y, p_z) = self.gradient(p)
                    u     = u - fft_coeff * p_x
                    v     = v - fft_coeff * p_y
                    w_vel = w_vel - fft_coeff * p_z
        else:
            # ---- Multigrid / MGCG solver (variable-coefficient Poisson) ----
            has_custom_coeffs = any(arr is not None for arr in (ch, cv, cw))
            if ch is None:
                ch = coeff * self.mu0_all_u
            if cv is None:
                cv = coeff * self.mu0_all_v

            # Select solve method: MGCG or standalone multigrid
            _poisson_solve = (self.poisson_solver.solve_mgcg
                              if self.poisson_method == "mgcg"
                              else self.poisson_solver.solve_multigrid)

            # Variable-density custom coefficients are coupled to a moving
            # immersed geometry; reusing the previous pressure field can carry
            # stale body-interior/interface values and destabilize the solve.
            if self.poisson_warm_start and not has_custom_coeffs:
                p0 = p
            else:
                p0 = torch.zeros_like(p)

            if self.ndim == 2:
                p, _ = _poisson_solve(
                    self.div[1:-1,1:-1],
                    p0,
                    ch = ch[1:,1:-1],
                    cv = cv[1:-1,1:],
                )
                # ====== projection step ======
                (p_x, p_y) = self.gradient(p)
                u          = u - ch * p_x
                v          = v - cv * p_y
            else:
                if cw is None:
                    cw = coeff * self.mu0_all_w
                p, _ = _poisson_solve(
                    self.div[1:-1, 1:-1, 1:-1],
                    p0,
                    ch=ch[1:, 1:-1, 1:-1],
                    cv=cv[1:-1, 1:, 1:-1],
                    cw=cw[1:-1, 1:-1, 1:],
                )
                # ====== projection step ======
                (p_x, p_y, p_z) = self.gradient(p)
                u    = u - ch * p_x
                v    = v - cv * p_y
                w_vel = w_vel - cw * p_z

        if self.ndim == 2:
            return (u, v, p)
        else:
            return (u, v, w_vel, p)



    # ------------------------------------------------------------------
    #  Sponge / damping layer
    # ------------------------------------------------------------------
    def _build_sponge_fields(self, width, strength, axes=None):
        """Build quadratic sponge coefficient σ on each staggered grid.

        σ(x) = σ_max · (max(0, Ls - d) / Ls)²

        where  d  is the distance to the nearest domain boundary and
        Ls = *width*.  Returns (sigma_u, sigma_v, sigma_w) tensors on the
        MAC staggered grids.  For 2-D, sigma_w is ``None``.

        Parameters
        ----------
        axes : list of str or None
            Which axes to sponge.  None = all axes.  E.g. ``["x"]`` damps
            only near the left/right walls (useful for lateral absorbing
            layers without touching inflow/outflow boundaries in y/z).
        """
        Ls = width
        sigma_max = strength
        if axes is None:
            axes = ["x", "y", "z"]

        def _quadratic_ramp_1d(coords, lo, hi):
            """Return σ(x) along one axis for cell centres *coords*."""
            d_lo = coords - lo            # distance from low boundary
            d_hi = hi - coords            # distance from high boundary
            d    = torch.minimum(d_lo, d_hi)  # distance from nearest wall
            ratio = torch.clamp((Ls - d) / Ls, min=0.0)
            return sigma_max * ratio * ratio

        # -- cell-centre coordinates (used for all grids) ----------------
        x = self.x   # (Nx,)
        y = self.y   # (Ny,)

        sx = _quadratic_ramp_1d(x, self.xmin, self.xmax) if "x" in axes else torch.zeros_like(x)
        sy = _quadratic_ramp_1d(y, self.ymin, self.ymax) if "y" in axes else torch.zeros_like(y)

        # Component-selective sponge: each velocity component is damped
        # only near walls where it is the wall-NORMAL component.
        #   u ← damped near x-walls (u is normal to x-walls)
        #   v ← damped near y-walls (v is normal to y-walls)
        #   w ← damped near z-walls (w is normal to z-walls)
        # When ALL axes are active, fall back to isotropic max(sx,sy[,sz])
        # for backward compatibility (each component damped near every wall).

        if self.ndim == 3:
            z  = self.z  # (Nz,)
            sz = _quadratic_ramp_1d(z, self.zmin, self.zmax) if "z" in axes else torch.zeros_like(z)
            all_active = "x" in axes and "y" in axes and "z" in axes
            if all_active:
                sigma_3d = torch.maximum(
                    torch.maximum(sx[:, None, None], sy[None, :, None]),
                    sz[None, None, :],
                )                                      # (Nx, Ny, Nz)
                return sigma_3d, sigma_3d, sigma_3d
            else:
                Nx, Ny, Nz = len(x), len(y), len(z)
                shape = (Nx, Ny, Nz)
                zeros = torch.zeros(shape, device=x.device, dtype=x.dtype)
                sigma_u = sx[:, None, None].expand(shape).contiguous() if "x" in axes else zeros
                sigma_v = sy[None, :, None].expand(shape).contiguous() if "y" in axes else zeros.clone()
                sigma_w = sz[None, None, :].expand(shape).contiguous() if "z" in axes else zeros.clone()
                return sigma_u, sigma_v, sigma_w
        else:
            all_active = "x" in axes and "y" in axes
            if all_active:
                sigma_2d = torch.maximum(sx[:, None], sy[None, :])  # (Nx, Ny)
                return sigma_2d, sigma_2d, None
            else:
                Nx, Ny = len(x), len(y)
                zeros = torch.zeros(Nx, Ny, device=x.device, dtype=x.dtype)
                sigma_u = sx[:, None].expand(Nx, Ny).contiguous() if "x" in axes else zeros
                sigma_v = sy[None, :].expand(Nx, Ny).contiguous() if "y" in axes else zeros.clone()
                return sigma_u, sigma_v, None

    def apply_sponge_damping(self, u, v, w=None):
        """Damp velocity towards zero near domain boundaries.

        u_new = u / (1 + dt·σ)

        Applied in-place via multiplication for efficiency.
        Returns (u, v) in 2-D or (u, v, w) in 3-D.
        """
        if not self.use_sponge:
            return (u, v, w) if w is not None else (u, v)

        dt = float(self.dt)

        # Pre-compute damping factor: 1 / (1 + dt·σ)
        damp_u = 1.0 / (1.0 + dt * self._sponge_sigma_u)
        damp_v = 1.0 / (1.0 + dt * self._sponge_sigma_v)
        u = u * damp_u
        v = v * damp_v

        if w is not None and self._sponge_sigma_w is not None:
            damp_w = 1.0 / (1.0 + dt * self._sponge_sigma_w)
            w = w * damp_w
            return (u, v, w)
        return (u, v)

    # ------------------------------------------------------------------
    #  Yield-stress damping
    # ------------------------------------------------------------------
    def apply_yield_damping(self, u, v, w=None):
        """Damp velocity in unyielded (low-shear-rate) regions.

        Where the local strain rate γ̇ is below γ̇_c, the fluid stress
        is below the yield stress and the material should behave as a
        solid.  We enforce this with an implicit penalty:

            u_new = u / (1 + dt · σ(γ̇))

        where σ(γ̇) = σ_max · max(0, 1 − γ̇/γ̇_c)².

        This is applied to the cell-centred velocity magnitude; the
        same scalar damping factor is used for all components.
        """
        if not self.use_yield_damping:
            return (u, v, w) if w is not None else (u, v)

        vel = (u, v, w) if w is not None else (u, v)
        S_mag = ops.strain_rate_magnitude(vel, float(self.h), self.ndim)

        # Quadratic ramp: full damping at γ̇=0, zero at γ̇≥γ̇_c
        ratio = torch.clamp(1.0 - S_mag / self._yield_gamma_c, min=0.0)
        sigma = self._yield_strength * ratio * ratio
        damp = 1.0 / (1.0 + float(self.dt) * sigma)

        u = u * damp
        v = v * damp
        if w is not None:
            w = w * damp
            return (u, v, w)
        return (u, v)

    # ------------------------------------------------------------------
    #  Smagorinsky LES model
    # ------------------------------------------------------------------
    def _compute_smagorinsky_nu_t(self, *vel):
        """Compute Smagorinsky eddy viscosity ν_t = (Cs·Δ)²|S̄|.

        Only called when ``self.use_smagorinsky`` is True.
        """
        return ops.smagorinsky_viscosity(
            vel, float(self.h), self.ndim, cs=self.smagorinsky_cs,
        )

    # ------------------------------------------------------------------
    #  Carreau non-Newtonian model
    # ------------------------------------------------------------------
    def _compute_carreau_nu_t(self, *vel):
        """Compute Carreau viscosity as a ``nu_t`` offset from ``self.nu``.

        The Carreau model gives a spatially-varying kinematic viscosity:
            ν(γ̇) = ν_∞ + (ν_0 − ν_∞) · [1 + (λ·γ̇)²]^((n−1)/2)

        To plug into the existing ``nu_t`` pathway we return
            nu_t = ν(γ̇) − self.nu
        so that the solver computes  nu_eff = self.nu + nu_t = ν(γ̇).

        Only called when ``self.use_carreau`` is True.
        """
        nu_field = ops.carreau_viscosity(
            vel, float(self.h), self.ndim,
            nu_0=self.carreau_nu_0,
            nu_inf=self.carreau_nu_inf,
            lam=self.carreau_lam,
            n=self.carreau_n,
            tau_y=self.carreau_tau_y,
            rho=float(self.rho),
            nu_max=self.carreau_nu_max,
        )
        return nu_field - float(self.nu)

    # ------------------------------------------------------------------
    #  Unified variable-viscosity dispatcher
    # ------------------------------------------------------------------
    def _compute_nu_t(self, *vel):
        """Return the extra viscosity field for the advection-diffusion step.

        Returns ``None`` when neither Smagorinsky nor Carreau is active,
        keeping the constant-viscosity fast path.
        """
        if self.use_smagorinsky:
            return self._compute_smagorinsky_nu_t(*vel)
        if self.use_carreau:
            return self._compute_carreau_nu_t(*vel)
        return None

    def _compute_nu_rho_for_forces(self, *vel):
        """Return ν·ρ for the force computation (scalar or tensor).

        * Constant viscosity:  returns ``self.nu * self.rho``  (scalar).
        * Smagorinsky/Carreau: returns a spatially-varying tensor field.

        The result is cached in ``self._nu_rho_field`` so that the force
        computation does not recompute strain rates a second time.
        """
        if not self.use_variable_viscosity:
            return self.nu * self.rho
        if self.use_smagorinsky:
            nu_eff = float(self.nu) + self._compute_smagorinsky_nu_t(*vel)
        else:  # Carreau / Herschel-Bulkley–Carreau
            nu_eff = ops.carreau_viscosity(
                vel, float(self.h), self.ndim,
                nu_0=self.carreau_nu_0,
                nu_inf=self.carreau_nu_inf,
                lam=self.carreau_lam,
                n=self.carreau_n,
                tau_y=self.carreau_tau_y,
                rho=float(self.rho),
                nu_max=self.carreau_nu_max,
            )
        self._nu_rho_field = nu_eff * self.rho
        return self._nu_rho_field

    @staticmethod
    def _bdim_meta(
        phi, mu0, body_vel, mu1, *normals_and_extras,
    ):
        """BDIM2 meta-equation (compilable, elementwise).

        phi_out = mu0 * (phi - body_vel) + body_vel
                  + mu1 * normal_derivative(phi - body_vel, ...)
        Parameters
        ----------
        normals_and_extras : variable-length
            2-D: (normal_x, normal_y, h_scalar as 0-d tensor, ndim_int=2)
            3-D: (normal_x, normal_y, normal_z, h_scalar, ndim_int=3)
        Last two positional args are always (h, ndim) for normal_derivative.
        """
        # unpack: last two are (h, ndim), preceding are normals
        h    = normals_and_extras[-2]
        ndim = normals_and_extras[-1]
        normals = normals_and_extras[:-2]
        normal_x = normals[0]
        normal_y = normals[1]
        normal_z = normals[2] if len(normals) == 3 else None
        diff = phi - body_vel
        nd = ops.normal_derivative(diff, h, ndim, normal_x, normal_y, normal_z)
        return mu0 * diff + body_vel + mu1 * nd

    # ------------------------------------------------------------------
    #   3-D BDIM apply with optional union-AABB narrow band
    # ------------------------------------------------------------------
    def _bdim_apply_3d(self, phi, mu0, body_vel, mu1,
                      normal_x, normal_y, normal_z):
        """Apply the BDIM2 meta-equation to a single 3-D staggered grid.

        When ``self._bdim_union`` is on AND a union AABB is available
        (cached on ``self._bdim_union_aabb``), only the union sub-block
        is touched.  Outside the union mu0=1, mu1=0, body_vel=0 makes the
        meta-equation the identity, so phi[outside] is left unchanged
        (we slice-write the cropped result back into phi).

        Otherwise falls back to the full-grid CUDA-graph kernel.
        Returns a tensor that the caller can safely consume; the caller
        is responsible for cloning if it needs to keep a reference past
        the next CUDA-graph replay (full-grid path) — in the union path
        the returned tensor is the input ``phi`` itself (mutated in
        place), which is already an owned tensor.
        """
        _h = self.h
        if self._bdim_union and getattr(self, '_bdim_union_aabb', None) is not None:
            ui0, ui1, uj0, uj1, uk0, uk1 = self._bdim_union_aabb
            usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))
            sub = self._bdim_meta_dyn_compiled(
                phi[usl].contiguous(),
                mu0[usl].contiguous(),
                body_vel[usl].contiguous(),
                mu1[usl].contiguous(),
                normal_x[usl].contiguous(),
                normal_y[usl].contiguous(),
                normal_z[usl].contiguous(),
                _h, 3,
            )
            phi[usl] = sub
            return phi
        return self._bdim_meta_compiled(
            phi, mu0, body_vel, mu1,
            normal_x, normal_y, normal_z, _h, 3,
        ).clone()

    # ------------------------------------------------------------------
    #   Union AABB across all body sub-blocks (3-D)
    # ------------------------------------------------------------------
    def _compute_union_aabb_3d(self, halo=2, bucket=16):
        """Return (i0,i1,j0,j1,k0,k1) union AABB over all body sparse
        SDFs, expanded by ``halo`` cells and clipped to grid extent.
        Returns ``None`` if any body lacks a sparse AABB.

        When ``bucket > 1`` each extent (i1-i0, j1-j0, k1-k0) is rounded
        up to a multiple of ``bucket`` by expanding the high side first
        and, if we hit the grid boundary, the low side.  This stabilizes
        the sub-block shape to a small discrete set so that
        ``dynamic=True`` compiled kernels only pay the recompile cost a
        bounded number of times (once per bucket combination seen during
        warmup) instead of every time the swimmer deforms.
        """
        comp = self.composite_body
        sparse = getattr(comp, '_sdf_sparse', None)
        if not sparse or sparse[0] is None:
            return None
        u_i0 = u_j0 = u_k0 = 1 << 30
        u_i1 = u_j1 = u_k1 = -1
        for entry in sparse:
            if entry is None:
                return None
            aabb_i = entry[0]
            if aabb_i is None:
                return None
            i0, i1, j0, j1, k0, k1 = aabb_i
            if i0 < u_i0: u_i0 = i0
            if j0 < u_j0: u_j0 = j0
            if k0 < u_k0: u_k0 = k0
            if i1 > u_i1: u_i1 = i1
            if j1 > u_j1: u_j1 = j1
            if k1 > u_k1: u_k1 = k1
        Ni, Nj, Nk = comp.sdf_val.shape
        u_i0 = max(0, u_i0 - halo); u_i1 = min(Ni, u_i1 + halo)
        u_j0 = max(0, u_j0 - halo); u_j1 = min(Nj, u_j1 + halo)
        u_k0 = max(0, u_k0 - halo); u_k1 = min(Nk, u_k1 + halo)

        if bucket is not None and bucket > 1:
            def _pad(lo, hi, N, b):
                extent = hi - lo
                target = ((extent + b - 1) // b) * b
                if target > N:
                    target = N
                pad = target - extent
                # Expand high side first, then spill to low side if clipped.
                new_hi = hi + pad
                if new_hi > N:
                    over = new_hi - N
                    new_hi = N
                    lo = max(0, lo - over)
                return lo, new_hi
            u_i0, u_i1 = _pad(u_i0, u_i1, Ni, bucket)
            u_j0, u_j1 = _pad(u_j0, u_j1, Nj, bucket)
            u_k0, u_k1 = _pad(u_k0, u_k1, Nk, bucket)

        # Cropping is only a net win when the union AABB covers a small
        # fraction of the grid: each BDIM apply pays a fixed launch-
        # overhead cost (7 .contiguous() slice copies + 1 slice-assign,
        # ~9 kernel launches) that only beats the full-grid kernel when
        # the saved kernel work exceeds ~9 × launch_overhead. Empirically
        # the break-even point is around 50 % of the full volume; above
        # that, return None so the caller falls back to the full-grid
        # compiled kernel.
        sub_vol  = (u_i1 - u_i0) * (u_j1 - u_j0) * (u_k1 - u_k0)
        full_vol = Ni * Nj * Nk
        if sub_vol > 0.5 * full_vol:
            return None

        return (u_i0, u_i1, u_j0, u_j1, u_k0, u_k1)

    def solver_iteration_heun(self, u, v, p, iteration, w_vel=None):
        """
        Heun (RK2 predictor-corrector) time integration with BDIM2.

        1. **Predictor** (w = 1):
            adv-diff on u^n → BDIM → project → BCs → u_pred (div-free)

        2. **Corrector** (w = 0.5):
            adv-diff on u_pred → rebase from u^n → BDIM →
            average with **projected** predictor → project(w=0.5)

        The corrector average is ``0.5*(u_pred + BDIM(u^n + dt·RHS(u_pred)))``.
        Because ``u_pred`` is divergence-free, ``div(u_avg)`` equals half
        the corrector's BDIM divergence.  ``w=0.5`` compensates this
        halving in the Poisson coefficient so that the stored pressure
        equals the physical dynamic pressure (needed for correct force
        computation).  The velocity correction is ``w``-independent.
        """

        if self.ndim == 2:
            # ====== PREDICTOR ======
            nu_t = self._compute_nu_t(u, v)
            (uprime, vprime) = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()

            # BDIM2 meta-equation (fused when compiled)
            _bdim = self._bdim_meta_compiled
            _h    = self.h
            uprime = _bdim(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vprime = _bdim(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()

            self.adv_diff_solver.set_BCs(uprime, vprime)
            (u1, v1, p1) = self.project(uprime, vprime, p)
            # Re-apply BCs after projection
            # BC!(a.u,...) after every project! call).
            self.adv_diff_solver.set_BCs(u1, v1)

            # ====== CORRECTOR ======
            # Evaluate RHS at the projected predicted velocity
            nu_t = self._compute_nu_t(u1, v1)
            (uprime2, vprime2) = self.adv_diff_solver.solve(u1, v1, nu_t=nu_t)
            # adv_diff.solve returns u1 + dt*RHS(u1).
            # Rebase from u^n: u^n + dt*RHS(u_pred), matching
            # with u⁰ = u^n (saved once at the top of mom_step!).
            uprime2 = u + (uprime2 - u1)
            vprime2 = v + (vprime2 - v1)

            # BDIM2 meta-equation on corrector (fused when compiled)
            uprime2 = _bdim(
                uprime2, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vprime2 = _bdim(
                vprime2, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()

            # Average the PROJECTED predictor with the corrector's BDIM
            # halves  u_pred + BDIM(u^n + dt*RHS(u_pred)).
            u_avg = 0.5 * (u1 + uprime2)
            v_avg = 0.5 * (v1 + vprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg)
            # w=0.5: the corrector average halves the divergence (u_pred
            # is div-free), so w=0.5 doubles the Poisson coefficient to
            # recover the physical pressure.  Velocity correction is
            # w-independent; only the stored pressure changes.
            (u_out, v_out, p_out) = self.project(u_avg, v_avg, p, w=0.5)

            # Sponge damping (2-D)
            if self.use_sponge:
                (u_out, v_out) = self.apply_sponge_damping(u_out, v_out)

            # Yield-stress damping (2-D)
            if self.use_yield_damping:
                (u_out, v_out) = self.apply_yield_damping(u_out, v_out)

            return (u_out, v_out, p_out)

        else:  # 3D
            # ====== PREDICTOR ======
            nu_t = self._compute_nu_t(u, v, w_vel)
            (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()
            wprime = wprime.clone()

            # Cache union AABB for both BDIM passes (predictor + corrector).
            # Cleared at end of step.  Cheap (Python loop over <~10 bodies).
            self._bdim_union_aabb = (
                self._compute_union_aabb_3d(halo=2)
                if self._bdim_union else None
            )

            # BDIM2 meta-equation (fused when compiled)
            _bdim = self._bdim_meta_compiled
            _h    = self.h
            uprime = self._bdim_apply_3d(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u,
            )
            vprime = self._bdim_apply_3d(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v,
            )
            wprime = self._bdim_apply_3d(
                wprime, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w,
            )

            self.adv_diff_solver.set_BCs(uprime, vprime, wprime)
            (u1, v1, w1, p1) = self.project(uprime, vprime, p, w_vel=wprime)
            # Re-apply BCs after projection
            self.adv_diff_solver.set_BCs(u1, v1, w1)

            # ====== CORRECTOR ======
            nu_t = self._compute_nu_t(u1, v1, w1)
            (uprime2, vprime2, wprime2) = self.adv_diff_solver.solve(u1, v1, w1, nu_t=nu_t)
            # Rebase from u^n
            uprime2 = u     + (uprime2 - u1)
            vprime2 = v     + (vprime2 - v1)
            wprime2 = w_vel + (wprime2 - w1)

            # BDIM2 meta-equation on corrector (fused when compiled)
            uprime2 = self._bdim_apply_3d(
                uprime2, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u,
            )
            vprime2 = self._bdim_apply_3d(
                vprime2, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v,
            )
            wprime2 = self._bdim_apply_3d(
                wprime2, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w,
            )

            # Drop the cached union AABB now that both BDIM passes are done.
            self._bdim_union_aabb = None

            # Free mu1 + staggered normals after both BDIM passes
            for _attr in ('mu1_all_u', 'mu1_all_v', 'mu1_all_w',
                          'normal_x_u', 'normal_y_u', 'normal_z_u',
                          'normal_x_v', 'normal_y_v', 'normal_z_v',
                          'normal_x_w', 'normal_y_w', 'normal_z_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            # Average the PROJECTED predictor (div-free) with the
            # corrector's BDIM output
            u_avg = 0.5 * (u1 + uprime2)
            v_avg = 0.5 * (v1 + vprime2)
            w_avg = 0.5 * (w1 + wprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg, w_avg)
            # w=0.5: see 2-D comment.
            (u_out, v_out, w_out, p_out) = self.project(u_avg, v_avg, p, w_vel=w_avg, w=0.5)

            # Free mu0 after project
            for _attr in ('mu0_all_u', 'mu0_all_v', 'mu0_all_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            # Sponge damping: damp velocity near domain boundaries
            if self.use_sponge:
                (u_out, v_out, w_out) = self.apply_sponge_damping(u_out, v_out, w_out)

            # Yield-stress damping (3-D)
            if self.use_yield_damping:
                (u_out, v_out, w_out) = self.apply_yield_damping(u_out, v_out, w_out)

            return (u_out, v_out, p_out, w_out)

    def solve_heun(self, u, v, p, iteration, w_vel=None):
        if self.ndim == 2:
            return self.solver_iteration_heun(u, v, p, iteration)
        else:
            return self.solver_iteration_heun(u, v, p, iteration, w_vel=w_vel)

    def solver_iteration_euler(self, u, v, p, iteration, w_vel=None):
        """Forward Euler time integration with BDIM2.

        Single-stage scheme:
          adv-diff → BDIM → project(w=1)

        Cheaper per step than Heun (one RHS evaluation instead of two),
        but only first-order accurate in time.
        """

        _bdim = self._bdim_meta_compiled
        _h    = self.h

        if self.ndim == 2:
            nu_t = self._compute_nu_t(u, v)
            (uprime, vprime) = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()

            # BDIM2 meta-equation
            uprime = _bdim(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vprime = _bdim(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()

            self.adv_diff_solver.set_BCs(uprime, vprime)
            (u_out, v_out, p_out) = self.project(uprime, vprime, p)

            # Sponge damping (2-D)
            if self.use_sponge:
                (u_out, v_out) = self.apply_sponge_damping(u_out, v_out)

            # Yield-stress damping (2-D Euler)
            if self.use_yield_damping:
                (u_out, v_out) = self.apply_yield_damping(u_out, v_out)

            return (u_out, v_out, p_out)

        else:  # 3D
            nu_t = self._compute_nu_t(u, v, w_vel)
            (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()
            wprime = wprime.clone()

            # Cache union AABB for the BDIM pass.
            self._bdim_union_aabb = (
                self._compute_union_aabb_3d(halo=2)
                if self._bdim_union else None
            )

            # BDIM2 meta-equation
            uprime = self._bdim_apply_3d(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u,
            )
            vprime = self._bdim_apply_3d(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v,
            )
            wprime = self._bdim_apply_3d(
                wprime, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w,
            )
            self._bdim_union_aabb = None

            # Free mu1 and staggered normals — no longer needed after
            # BDIM.  project() only uses mu0_{u,v,w}, and forces
            # recomputes CC normals on-the-fly.  This releases
            # 12 × grid_shape × 4 bytes ≈ 1.5 GB for typical 3-D grids.
            for _attr in ('mu1_all_u', 'mu1_all_v', 'mu1_all_w',
                          'normal_x_u', 'normal_y_u', 'normal_z_u',
                          'normal_x_v', 'normal_y_v', 'normal_z_v',
                          'normal_x_w', 'normal_y_w', 'normal_z_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            self.adv_diff_solver.set_BCs(uprime, vprime, wprime)
            (u_out, v_out, w_out, p_out) = self.project(uprime, vprime, p, w_vel=wprime)

            # Free mu0 — no longer needed after project.  Reduces peak
            # memory during force computation by another ~0.4 GB.
            for _attr in ('mu0_all_u', 'mu0_all_v', 'mu0_all_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            # Sponge damping: damp velocity near domain boundaries
            if self.use_sponge:
                (u_out, v_out, w_out) = self.apply_sponge_damping(u_out, v_out, w_out)

            # Yield-stress damping (3-D Euler)
            if self.use_yield_damping:
                (u_out, v_out, w_out) = self.apply_yield_damping(u_out, v_out, w_out)

            return (u_out, v_out, p_out, w_out)

    def solve_euler(self, u, v, p, iteration, w_vel=None):
        if self.ndim == 2:
            return self.solver_iteration_euler(u, v, p, iteration)
        else:
            return self.solver_iteration_euler(u, v, p, iteration, w_vel=w_vel)

    # ------------------------------------------------------------------
    #   BDIM field cleanup  —  free intermediate tensors between steps
    # ------------------------------------------------------------------
    _BDIM_FIELD_NAMES = (
        # staggered-grid mu / normals (u, v, w)
        'mu0_all_u', 'mu1_all_u', 'mu0_all_v', 'mu1_all_v',
        'normal_x_u', 'normal_y_u', 'normal_x_v', 'normal_y_v',
        # 3-D only (harmlessly absent in 2-D)
        'mu0_all_w', 'mu1_all_w',
        'normal_x_w', 'normal_y_w',
        'normal_z_u', 'normal_z_v', 'normal_z_w',
        # CC-grid mu / normals (recomputed in _recompute_mu_normals_3d)
        'mu0_all', 'mu1_all', 'm_m0_all',
        'normal_x', 'normal_y', 'normal_z',
        # force intermediates (recomputed in forces_method2 / forces_method2_3d)
        'xstress_tensor', 'ystress_tensor', 'zstress_tensor',
        'pforce_x', 'pforce_y', 'pforce_z',
        # divergence from project()
        'div',
    )

    def _release_bdim_fields(self):
        """Set BDIM intermediate fields to *None* so their GPU memory
        can be reclaimed between time-steps (they are recomputed at the
        beginning of every step anyway)."""
        # When mu/normals union crop is active, keep the persistent
        # full-grid mu/normal buffers alive across steps — they hold the
        # outside-body default values that never change, and only the
        # union sub-block is overwritten each step.
        keep = set()
        if getattr(self, '_mu_normals_union', False):
            keep = {
                'mu0_all_u', 'mu1_all_u', 'mu0_all_v', 'mu1_all_v',
                'mu0_all_w', 'mu1_all_w', 'mu0_all', 'mu1_all',
                'm_m0_all',
                'normal_x_u', 'normal_y_u', 'normal_z_u',
                'normal_x_v', 'normal_y_v', 'normal_z_v',
                'normal_x_w', 'normal_y_w', 'normal_z_w',
                'normal_x', 'normal_y', 'normal_z',
            }
        for attr in self._BDIM_FIELD_NAMES:
            if attr in keep:
                continue
            if hasattr(self, attr):
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    #   mu / normal recomputation  (shared by step_() and BDIMhandler)
    # ------------------------------------------------------------------
    def _recompute_mu_normals_2d(self):
        """Recompute mu0/mu1 and normals on u- and v-staggered grids (2-D).

        CC-grid normals are computed on-the-fly inside forces_method1/2.
        """
        comp = self.composite_body

        (self.mu0_all_u, self.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
        (self.normal_x_u, self.normal_y_u) = comp.compute_normals(comp.sdf_val_u)

        (self.mu0_all_v, self.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
        (self.normal_x_v, self.normal_y_v) = comp.compute_normals(comp.sdf_val_v)

        # CC-grid mu0 — used for smooth pressure masking in forces_method2
        (self.mu0_all, self.mu1_all) = comp.mu_funcs(comp.sdf_val)

    def _recompute_mu_normals_3d(self):
        """Recompute mu0/mu1 and normals on all staggered + CC grids (3-D).

        When ``_compile_sdf`` is enabled, uses a batched+compiled kernel that
        processes all four grids (u, v, w, CC) in a single fused CUDA graph.
        When ``_mu_normals_union`` is enabled, the kernel runs only on the
        union AABB of all body sub-blocks (with halo) and results are
        slice-written into persistent full-grid buffers pre-filled with
        the outside-body defaults (mu0=1, mu1=0, normals=0).
        """
        comp = self.composite_body

        # ------------------------------------------------------------------
        # Union-AABB crop path — outside the union SDF is _FAR so mu0=1,
        # mu1=0, normals=0 (these defaults never change between steps).
        # Uses the shared _compute_union_aabb_3d helper which rounds the
        # sub-block extents up to a bucket multiple so the dynamic-shape
        # compiled kernel only recompiles a bounded number of times.
        # ------------------------------------------------------------------
        u_aabb = None
        if self._mu_normals_union:
            u_aabb = self._compute_union_aabb_3d(halo=2, bucket=16)

        if u_aabb is not None:
            # Lazy-allocate a single packed persistent buffer of shape
            # (21, Nx, Ny, Nz) pre-filled with outside-body defaults
            # (mu0=1, mu1=0, normals=0, m_m0=0).  All downstream mu /
            # normal attributes are *views* into this packed tensor, so
            # the union sub-block can be written with a single slice
            # assignment instead of 21 separate slice-writes.
            #
            # Pack layout along dim-0:
            #    0- 3  : mu0 for [u, v, w, cc]
            #    4- 7  : mu1 for [u, v, w, cc]
            #    8-11  : normal_x for [u, v, w, cc]
            #   12-15  : normal_y for [u, v, w, cc]
            #   16-19  : normal_z for [u, v, w, cc]
            #      20  : m_m0_all  (= 1 - mu0_cc)
            if getattr(self, '_mu_pack', None) is None or \
               self._mu_pack.shape[1:] != comp.sdf_val.shape or \
               self._mu_pack.dtype != comp.sdf_val.dtype or \
               self._mu_pack.device != comp.sdf_val.device:
                pack = torch.zeros(
                    (21, *comp.sdf_val.shape),
                    device=comp.sdf_val.device, dtype=comp.sdf_val.dtype,
                )
                pack[0:4].fill_(1.0)  # mu0 defaults to 1 outside body
                self._mu_pack = pack
                self._mu_union_ready = True
            pack = self._mu_pack

            # (Re-)alias every step: cheap Python, and robust to any
            # non-union path overwriting these attributes.
            self.mu0_all_u, self.mu0_all_v, self.mu0_all_w, self.mu0_all = (
                pack[0], pack[1], pack[2], pack[3])
            self.mu1_all_u, self.mu1_all_v, self.mu1_all_w, self.mu1_all = (
                pack[4], pack[5], pack[6], pack[7])
            self.normal_x_u, self.normal_x_v, self.normal_x_w, self.normal_x = (
                pack[8], pack[9], pack[10], pack[11])
            self.normal_y_u, self.normal_y_v, self.normal_y_w, self.normal_y = (
                pack[12], pack[13], pack[14], pack[15])
            self.normal_z_u, self.normal_z_v, self.normal_z_w, self.normal_z = (
                pack[16], pack[17], pack[18], pack[19])
            self.m_m0_all = pack[20]

            ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
            usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))

            mu0_s, mu1_s, nx_s, ny_s, nz_s = self._mu_normals_batched_3d_dyn_compiled(
                comp.sdf_val_u[usl].contiguous(),
                comp.sdf_val_v[usl].contiguous(),
                comp.sdf_val_w[usl].contiguous(),
                comp.sdf_val[usl].contiguous(),
                comp.h, comp.eps,
            )

            # Fused slice-write: stack all 21 sub-block outputs along
            # dim-0 and scatter into the packed buffer with ONE assign.
            # Order must match the pack layout above.
            stacked = torch.cat(
                (mu0_s, mu1_s, nx_s, ny_s, nz_s, 1.0 - mu0_s[3:4]),
                dim=0,
            )  # (21, sub_Nx, sub_Ny, sub_Nz)
            self._mu_pack[:, ui0:ui1, uj0:uj1, uk0:uk1] = stacked
            return

        if self._compile_sdf:
            # ── Batched + compiled path: all 4 grids in one fused pass ──
            _fn = _mu_normals_batched_3d_compiled
            mu0, mu1, nx, ny, nz = _fn(
                comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.sdf_val, comp.h, comp.eps,
            )
            # Clone outputs — CUDA graph buffers are overwritten on
            # subsequent replays, so we must detach before storing.
            mu0, mu1 = mu0.clone(), mu1.clone()
            nx, ny, nz = nx.clone(), ny.clone(), nz.clone()

            # Unstack: order is [u, v, w, cc]
            self.mu0_all_u, self.mu1_all_u = mu0[0], mu1[0]
            self.normal_x_u, self.normal_y_u, self.normal_z_u = nx[0], ny[0], nz[0]

            self.mu0_all_v, self.mu1_all_v = mu0[1], mu1[1]
            self.normal_x_v, self.normal_y_v, self.normal_z_v = nx[1], ny[1], nz[1]

            self.mu0_all_w, self.mu1_all_w = mu0[2], mu1[2]
            self.normal_x_w, self.normal_y_w, self.normal_z_w = nx[2], ny[2], nz[2]

            self.mu0_all, self.mu1_all = mu0[3], mu1[3]
            self.m_m0_all = 1 - self.mu0_all
            self.normal_x, self.normal_y, self.normal_z = nx[3], ny[3], nz[3]
        else:
            # ── Eager path: 4 × individual mu_funcs + compute_normals ──
            # u-grid
            (self.mu0_all_u, self.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
            (self.normal_x_u, self.normal_y_u, self.normal_z_u) = comp.compute_normals(comp.sdf_val_u)

            # v-grid
            (self.mu0_all_v, self.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
            (self.normal_x_v, self.normal_y_v, self.normal_z_v) = comp.compute_normals(comp.sdf_val_v)

            # w-grid
            (self.mu0_all_w, self.mu1_all_w) = comp.mu_funcs(comp.sdf_val_w)
            (self.normal_x_w, self.normal_y_w, self.normal_z_w) = comp.compute_normals(comp.sdf_val_w)

            # CC-grid (p) — cached for forces_method2_3d
            (self.mu0_all, self.mu1_all) = comp.mu_funcs(comp.sdf_val)
            self.m_m0_all = 1 - self.mu0_all
            (self.normal_x, self.normal_y, self.normal_z) = comp.compute_normals(comp.sdf_val)

    def _recompute_mu_normals(self):
        """Dispatch to 2-D or 3-D mu/normal recomputation."""
        if self.ndim == 2:
            self._recompute_mu_normals_2d()
        else:
            self._recompute_mu_normals_3d()

    def _apply_force_feedback(self, iteration, t):
        """Advance any standalone body state that consumes solver loads.

        This is intentionally separate from the FARMS / MuJoCo path: it is
        only used by ``FluidSolver.step_()`` after viscous and pressure loads
        have been accumulated on the current standalone composite body.
        """
        callback = getattr(self.composite_body, "apply_force_feedback", None)
        if callback is None:
            return

        force_x = self.friction_force_lin_x + self.pressure_force_x
        force_y = self.friction_force_lin_y + self.pressure_force_y

        if self.ndim == 2:
            force = torch.stack((force_x, force_y), dim=-1)
            torque = self.friction_force_ang_z + self.pressure_force_ang_z
        else:
            force_z = self.friction_force_lin_z + self.pressure_force_z
            torque_x = self.friction_force_ang_x + self.pressure_force_ang_x
            torque_y = self.friction_force_ang_y + self.pressure_force_ang_y
            torque_z = self.friction_force_ang_z + self.pressure_force_ang_z
            force = torch.stack((force_x, force_y, force_z), dim=-1)
            torque = torch.stack((torque_x, torque_y, torque_z), dim=-1)

        callback(
            force=force,
            torque=torque,
            iteration=iteration,
            time=t,
            dt=self.dt,
            solver=self,
        )

    def step_(self, u, v, p, iteration, t, w_vel=None):

        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        # --- recompute mu / normals on all staggered grids ---
        self._recompute_mu_normals()

        ##### just for plotting
        self.sdf_properties = [[self.composite_body.sdf_val_u]]

        if self.ndim == 2:
            (u, v, p) = self._solve(u, v, p, iteration)

            if self.compute_forces:
                self.forces_method2(u, v, p, iteration)
                self._apply_force_feedback(iteration, t)

        else:
            (u, v, p, w_vel) = self._solve(u, v, p, iteration, w_vel=w_vel)

            if self.compute_forces:
                self.forces_method2_3d(u, v, w_vel, p, iteration)
                self._apply_force_feedback(iteration, t)

        # ---- flow diagnostics (energy, enstrophy, CFL, divergence) ----
        self.diagnostics.update(
            iteration, u, v, p, self.dt, self.nu,
            divergence_fn=self.divergence,
            vorticity_fn=self.vorticity,
            w=w_vel,
        )

        # ---- free BDIM fields to reclaim GPU memory between steps ----
        self._release_bdim_fields()

        # ---- flush CUDA allocator cache to reduce nvidia-smi usage ----
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # ---- plotting / saving  (works for 2D and 3D) ----
        terminate = self.plotting_and_saving(u, v, p, iteration, w_vel=w_vel)

        if self.ndim == 2:
            return (u, v, p, terminate)
        else:
            return (u, v, p, w_vel, terminate)

    # ------------------------------------------------------------------
    #   Unified plotting / saving   (replaces old plotting_debug)
    # ------------------------------------------------------------------

    # Default plot specifications.
    # Each entry: (name, field_lambda, vmin, vmax, body_contours)
    # field_lambda receives (solver, u, v, p, w_vel) and returns a CPU tensor/array.
    # vmin/vmax per-spec:  None or "yaml" → use the global output.vmin/vmax
    #                      float          → fixed limit for this field
    # The YAML output.vmin/vmax can be a number (fixed) or "auto" (auto-scale).
    # ``output.plot_specs`` can override these defaults with a list of
    # field names or dicts such as {"name": "pressure", "vmin": "auto"}.

    DEFAULT_PLOT_SPECS = [
        ("curl",       lambda s, u, v, p, w: (
            s.vorticity(u, v, w).cpu() if s.ndim == 2
            else FluidSolver._curl_slice_fields(s, u, v, w)
        ), None, None, True),
        # ("v",          lambda s, u, v, p, w: v.cpu(), "auto", "auto", True),
        ("pressure",   lambda s, u, v, p, w: p.cpu(), "auto", "auto", True),
        # ("divergence", lambda s, u, v, p, w: s.divergence(u, v, w).cpu(),None, None, False),
    ]

    # Additional specs used only for 3-D isosurface rendering.
    # Signed vorticity components give much better 3-D visualisation
    # than the magnitude (which loses rotational direction).
    # We cache the vorticity dict on the solver so it's computed once per frame.
    @staticmethod
    def _cached_vort(s, u, v, w):
        """Compute vorticity_components once per frame and cache on solver."""
        if not hasattr(s, "_vort_cache") or s._vort_cache_id != id(u):
            s._vort_cache = s.vorticity_components(u, v, w)
            s._vort_cache_id = id(u)
        return s._vort_cache

    @staticmethod
    def _curl_slice_fields(s, u, v, w):
        """Return the out-of-plane curl component for each orthogonal slice.

        XY uses ``omega_z`` as usual. For XZ, the 2-D-like scalar curl is
        ``dw/dx - du/dz = -omega_y``. For YZ, it is ``dw/dy - dv/dz = omega_x``.
        """
        vort = FluidSolver._cached_vort(s, u, v, w)
        return {
            "xy": vort["omega_z"].cpu(),
            "xz": (-vort["omega_y"]).cpu(),
            "yz": vort["omega_x"].cpu(),
        }

    @staticmethod
    def _vel_mag(s, u, v, w):
        """Velocity magnitude.  u, v, w all share the same (Nx+2, Ny+2, Nz+2)
        shape in this solver, so we can compute |V| element-wise.  The
        O(dx/2) stagger offset is negligible for visualisation."""
        return (u**2 + v**2 + w**2).sqrt()

    # ``output.iso_3d_specs`` can override these defaults with field names
    # or dicts such as {"name": "omega_mag", "iso_value": 5.0}. When
    # ``output.iso_3d_value`` is set, it becomes the default threshold for
    # all configured 3-D isosurface fields; otherwise the automatic peak-
    # fraction thresholding in plotting.plot_field_3d is used.
    DEFAULT_3D_ISO_SPECS = [
        # ("omega_x",    lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_x"].cpu(), None, True),
        # ("omega_y",    lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_y"].cpu(), None, True),
        # ("omega_z",    lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_z"].cpu(), None, True),
        ("omega_mag",  lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_mag"].cpu(), None, True),
        ("vel_mag",    lambda s, u, v, p, w: FluidSolver._vel_mag(s, u, v, w).cpu(),                 None,   True),
        # ("pressure",   lambda s, u, v, p, w: p.cpu(),                                                None,   True),
    ]

    @classmethod
    def _plot_spec_registry(cls):
        registry = {spec[0]: spec for spec in cls.DEFAULT_PLOT_SPECS}
        registry.update({
            "v": (
                "v",
                lambda s, u, v, p, w: v.cpu(),
                "auto",
                "auto",
                True,
            ),
            "divergence": (
                "divergence",
                lambda s, u, v, p, w: s.divergence(u, v, w).cpu(),
                None,
                None,
                False,
            ),
        })
        return registry

    @classmethod
    def _iso_3d_spec_registry(cls):
        registry = {spec[0]: spec for spec in cls.DEFAULT_3D_ISO_SPECS}
        registry.update({
            "omega_x": (
                "omega_x",
                lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_x"].cpu(),
                None,
                True,
            ),
            "omega_y": (
                "omega_y",
                lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_y"].cpu(),
                None,
                True,
            ),
            "omega_z": (
                "omega_z",
                lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_z"].cpu(),
                None,
                True,
            ),
            "pressure": (
                "pressure",
                lambda s, u, v, p, w: p.cpu(),
                None,
                True,
            ),
        })
        return registry

    @staticmethod
    def _normalize_iso_threshold(value):
        return None if value == "auto" else value

    def _resolve_plot_specs(self, plot_specs_cfg):
        if plot_specs_cfg is None:
            return list(self.DEFAULT_PLOT_SPECS)
        if isinstance(plot_specs_cfg, str):
            plot_specs_cfg = [plot_specs_cfg]

        registry = self._plot_spec_registry()
        resolved = []
        for entry in plot_specs_cfg:
            if isinstance(entry, str):
                name = entry
                overrides = {}
            elif isinstance(entry, dict):
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Each output.plot_specs entry must define a non-empty 'name'.")
                overrides = entry
            else:
                raise TypeError(
                    "output.plot_specs entries must be strings or dicts with a 'name' key."
                )

            if name not in registry:
                raise ValueError(
                    f"Unknown plot field '{name}'. Available fields: {sorted(registry)}"
                )

            base = registry[name]
            show_body = base[4] if "show_body" not in overrides else bool(overrides["show_body"])
            resolved.append((
                name,
                base[1],
                overrides.get("vmin", base[2]),
                overrides.get("vmax", base[3]),
                show_body,
            ))
        return resolved

    def _resolve_iso_3d_specs(self, iso_specs_cfg, *, global_iso_value=None):
        if self.ndim != 3:
            return []

        global_iso_value = self._normalize_iso_threshold(global_iso_value)
        if iso_specs_cfg is None:
            if global_iso_value is None:
                return list(self.DEFAULT_3D_ISO_SPECS)
            iso_specs_cfg = [spec[0] for spec in self.DEFAULT_3D_ISO_SPECS]
        if isinstance(iso_specs_cfg, str):
            iso_specs_cfg = [iso_specs_cfg]

        registry = self._iso_3d_spec_registry()
        resolved = []
        for entry in iso_specs_cfg:
            if isinstance(entry, str):
                name = entry
                overrides = {}
            elif isinstance(entry, dict):
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Each output.iso_3d_specs entry must define a non-empty 'name'.")
                overrides = entry
            else:
                raise TypeError(
                    "output.iso_3d_specs entries must be strings or dicts with a 'name' key."
                )

            if name not in registry:
                raise ValueError(
                    f"Unknown 3D isosurface field '{name}'. Available fields: {sorted(registry)}"
                )

            base = registry[name]
            iso_value = overrides.get("iso_value")
            if "iso_value" not in overrides:
                iso_value = global_iso_value if global_iso_value is not None else base[2]
            iso_value = self._normalize_iso_threshold(iso_value)

            if len(base) > 3:
                show_body = base[3] if "show_body" not in overrides else bool(overrides["show_body"])
                resolved.append((name, base[1], iso_value, show_body))
            else:
                resolved.append((name, base[1], iso_value))
        return resolved

    # Maximum number of pending I/O futures before _submit_io blocks.
    # Each future can capture hundreds of MB of numpy arrays (3-D field
    # snapshots, per-body SDFs, …).  Without a cap the queue grows faster
    # than the 2-worker pool can drain it, causing OOM on large grids.
    _MAX_PENDING_IO = 10

    def _submit_io(self, fn, *args, **kwargs):
        """Submit *fn* to the background I/O pool and track the future."""
        # Reap already-finished futures to avoid unbounded growth
        self._io_futures = [f for f in self._io_futures if not f.done()]
        # Throttle: block until the queue drains below the cap so that
        # captured numpy arrays from old frames can be garbage-collected.
        while len(self._io_futures) >= self._MAX_PENDING_IO:
            # Wait for the oldest pending future to finish
            self._io_futures[0].result()
            self._io_futures = [f for f in self._io_futures if not f.done()]
        fut = self._io_executor.submit(fn, *args, **kwargs)
        self._io_futures.append(fut)
        return fut

    def flush_io(self):
        """Block until all pending background I/O tasks have completed."""
        for fut in self._io_futures:
            fut.result()          # re-raises any exception from the worker
        self._io_futures.clear()

    def _snapshot_body_sdf_vals_for_plots(self, crop_slices=None):
        """Snapshot per-body SDFs on CPU for plot colouring.

        2-D uses the dense ``composite_body.sdf_vals`` stack directly.
        3-D reconstructs a dense stack from the sparse per-body SDF blocks
        cached in ``composite_body._sdf_sparse``.
        """
        comp = getattr(self, "composite_body", None)
        bodies = getattr(comp, "bodies", None) if comp is not None else None
        if comp is None or bodies is None or len(bodies) == 0:
            return None

        ndim = len(self.grid_shape)
        if crop_slices is None:
            crop_slices = (slice(None),) * ndim
        elif not isinstance(crop_slices, tuple):
            crop_slices = (crop_slices,) * ndim

        if self.ndim == 2:
            sdf_vals = getattr(comp, "sdf_vals", None)
            if sdf_vals is None:
                return None
            sdf_vals_np = np.asarray(
                sdf_vals.detach().cpu().numpy() if hasattr(sdf_vals, "detach") else sdf_vals,
                dtype=np.float32,
            )
            return np.array(sdf_vals_np[(slice(None), *crop_slices)], dtype=np.float32, copy=True)

        sdf_sparse = getattr(comp, "_sdf_sparse", None)
        if sdf_sparse is None:
            return None

        full_shape = tuple(int(n) for n in self.grid_shape)
        crop_bounds = []
        cropped_shape = []
        for axis, sl in enumerate(crop_slices):
            start = 0 if sl.start is None else (full_shape[axis] + sl.start if sl.start < 0 else sl.start)
            stop = full_shape[axis] if sl.stop is None else (full_shape[axis] + sl.stop if sl.stop < 0 else sl.stop)
            start = max(0, min(full_shape[axis], start))
            stop = max(start, min(full_shape[axis], stop))
            crop_bounds.append((start, stop))
            cropped_shape.append(stop - start)

        sdf_vals_np = np.full((len(bodies), *cropped_shape), 1e4, dtype=np.float32)

        for body_i, sparse_entry in enumerate(sdf_sparse):
            if sparse_entry is None:
                continue
            aabb, sdf_body = sparse_entry
            sdf_body_np = np.asarray(
                sdf_body.detach().cpu().numpy() if hasattr(sdf_body, "detach") else sdf_body,
                dtype=np.float32,
            )

            if aabb is None:
                sdf_vals_np[body_i] = np.array(sdf_body_np[crop_slices], dtype=np.float32, copy=True)
                continue

            axis_ranges = [
                (aabb[0], aabb[1]),
                (aabb[2], aabb[3]),
                (aabb[4], aabb[5]),
            ]
            dst_slices = []
            src_slices = []
            intersects = True
            for axis, (body_lo, body_hi) in enumerate(axis_ranges):
                crop_lo, crop_hi = crop_bounds[axis]
                src_lo = max(body_lo, crop_lo)
                src_hi = min(body_hi, crop_hi)
                if src_lo >= src_hi:
                    intersects = False
                    break
                dst_slices.append(slice(src_lo - crop_lo, src_hi - crop_lo))
                src_slices.append(slice(src_lo - body_lo, src_hi - body_lo))

            if not intersects:
                continue

            sdf_vals_np[(body_i, *dst_slices)] = sdf_body_np[tuple(src_slices)]

        return sdf_vals_np

    def plotting_and_saving(self, u, v, p, iteration, *, w_vel=None, check_termination=True):
        """
        Unified plotting + data saving for 2-D and 3-D.
        Replaces the old ``plotting_debug`` and ``plotting_saving`` methods.

        Plotting and saving are offloaded to a background thread so the
        solver loop is not blocked by synchronous disk I/O.
        """
        if iteration % self.save_every != 0:
            if check_termination:
                return self.check_termination(iteration, u, v, p)
            return False

        # ---- snapshot tensors to CPU numpy *once* (still on main thread
        #      so the GPU transfer is overlapped with the previous kernel)
        # We clone/detach to decouple from the live computation graph.

        # ---- frame plots ----
        if self.save_frames:
            specs = getattr(self, "plot_specs", self.DEFAULT_PLOT_SPECS)
            bodies = getattr(self.composite_body, "bodies", None) if hasattr(self, "composite_body") else None

            # Snapshot per-body COM positions to CPU numpy *now* so the
            # background IO thread reads a frozen copy rather than the
            # live tensor (which the next solver step will overwrite).
            _com_positions = None
            if bodies is not None:
                _com_positions = []
                for _b in bodies:
                    _cp = getattr(_b, "com_pos", None)
                    if _cp is not None:
                        _com_positions.append(_cp.detach().cpu().numpy().copy())
                    else:
                        _com_positions.append(None)

            if self.ndim == 2:
                _phys_extent = (self.xmin, self.xmax, self.ymin, self.ymax)
                _crop_2d = (slice(1, -1), slice(1, -1))
                _body_sdf_vals_np = self._snapshot_body_sdf_vals_for_plots(_crop_2d)
                _sdf_2d = None
                if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                    _sdf_2d = self.composite_body.sdf_val.cpu().numpy().copy()[1:-1, 1:-1]
                _body_mu0_rgba = None
                if _body_sdf_vals_np is not None:
                    _body_mu0_rgba = plotting.build_body_mu0_rgba(
                        bodies,
                        _body_sdf_vals_np.shape[1:],
                        float(self.eps),
                        sdf_vals=_body_sdf_vals_np,
                    )

                for (name, field_fn, vmin, vmax, show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    field_np = field_np[1:-1, 1:-1]  # strip ghost cells
                    eff_vmin = self.vmin if vmin is None else (None if vmin == "auto" else vmin)
                    eff_vmax = self.vmax if vmax is None else (None if vmax == "auto" else vmax)
                    save_path = self.save_path
                    _bodies  = bodies if show_body else None
                    self._submit_io(
                        plotting.plot_field_2d,
                        field_np, _phys_extent,
                        name, iteration, save_path,
                        vmin=eff_vmin, vmax=eff_vmax, bodies=_bodies,
                        sdf_2d=_sdf_2d if show_body else None,
                        body_mu0_rgba=_body_mu0_rgba if show_body else None,
                        body_com_positions=_com_positions if show_body else None,
                    )
            else:
                # Crop ghost cells (index 0 and -1 on each axis) so plots
                # show only the physical domain, not BC-padded boundaries.
                _s = slice(1, -1)  # reusable ghost-cell crop slice
                coords = {
                    "x": self.x.cpu().numpy().copy()[_s],
                    "y": self.y.cpu().numpy().copy()[_s],
                    "z": self.z.cpu().numpy().copy()[_s],
                }
                _body_sdf_vals_np = self._snapshot_body_sdf_vals_for_plots((_s, _s, _s))
                # snapshot SDF for body-shape overlay on 2-D slice plots
                sdf_np = None
                if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                    sdf_np = self.composite_body.sdf_val.cpu().numpy().copy()[_s, _s, _s]
                for (name, field_fn, vmin, vmax, _show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    if isinstance(field, dict):
                        field_np = {
                            key: (value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.array(value))[_s, _s, _s]
                            for key, value in field.items()
                        }
                    else:
                        field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                        field_np = field_np[_s, _s, _s]  # strip ghost cells
                    eff_vmin = self.vmin if vmin is None else (None if vmin == "auto" else vmin)
                    eff_vmax = self.vmax if vmax is None else (None if vmax == "auto" else vmax)
                    save_path = self.save_path
                    _pass_bodies = bodies if _show_body else None
                    self._submit_io(
                        plotting.plot_field_3d_slices,
                        field_np, coords,
                        name, iteration, save_path,
                        vmin=eff_vmin, vmax=eff_vmax, bodies=_pass_bodies,
                        sdf_3d=sdf_np,
                        body_mu0_coloring=_show_body and _body_sdf_vals_np is not None,
                        body_eps=float(self.eps) if _body_sdf_vals_np is not None else None,
                        sdf_vals=_body_sdf_vals_np if _show_body else None,
                        body_com_positions=_com_positions if _show_body else None,
                    )

                # ---- HDF5 field export (3-D only, replaces old VTK) ----
                # Saves u, v, w, p, sdf — all derived fields (vorticity,
                # vel_mag, divergence) can be recomputed from these.
                # HDF5 stores the *full* arrays (including ghost cells)
                # so post-processing retains boundary information.
                if self.save:
                    h5_fields = {
                        "u": u.detach().cpu().numpy().copy(),
                        "v": v.detach().cpu().numpy().copy(),
                        "p": p.detach().cpu().numpy().copy(),
                    }
                    if w_vel is not None:
                        h5_fields["w"] = w_vel.detach().cpu().numpy().copy()
                    _sdf_full = None
                    if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                        _sdf_full = self.composite_body.sdf_val.cpu().numpy().copy()
                    if _sdf_full is not None:
                        h5_fields["sdf"] = _sdf_full

                    # Grids only on first snapshot (full coords incl. ghost cells)
                    grids = None
                    if iteration == 0 or not hasattr(self, '_grids_saved'):
                        grids = {
                            "x": self.x.cpu().numpy().copy(),
                            "y": self.y.cpu().numpy().copy(),
                            "z": self.z.cpu().numpy().copy(),
                        }
                        self._grids_saved = True

                    self._submit_io(
                        self._save_fields_h5,
                        h5_fields, grids, iteration,
                    )

                # ---- 3-D isosurface renders ----
                # sdf_np already snapshotted above for slice plots
                iso_specs = getattr(
                    self,
                    "iso_3d_specs",
                    self.DEFAULT_3D_ISO_SPECS if self.ndim == 3 else [],
                )
                for iso_entry in iso_specs:
                    name, field_fn = iso_entry[0], iso_entry[1]
                    iso_thresh = iso_entry[2] if len(iso_entry) > 2 else None
                    if iso_thresh == "vmax":
                        iso_thresh = getattr(self, "vmax", None)
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    field_np = field_np[_s, _s, _s]  # strip ghost cells
                    self._submit_io(
                        plotting.plot_field_3d,
                        field_np, coords,
                        name, iteration, self.save_path,
                        sdf_3d=sdf_np,
                        iso_value=iso_thresh,
                        bodies=bodies,
                        sdf_vals=_body_sdf_vals_np,
                    )

        # ---- raw data save (2-D) ----
        if self.save and self.ndim == 2:
            h5_fields = {
                "u": u.detach().cpu().numpy().copy(),
                "v": v.detach().cpu().numpy().copy(),
                "p": p.detach().cpu().numpy().copy(),
            }
            # SDF snapshot
            sdf_np = None
            if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                sdf_np = self.composite_body.sdf_val.cpu().numpy().copy()
                h5_fields["sdf"] = sdf_np

            # Grids only on first snapshot
            grids = None
            if iteration == 0 or not hasattr(self, '_grids_saved'):
                grids = {
                    "x": self.x.cpu().numpy().copy(),
                    "y": self.y.cpu().numpy().copy(),
                }
                self._grids_saved = True

            self._submit_io(
                self._save_fields_h5,
                h5_fields, grids, iteration,
            )

            # 2-D body contours
            if hasattr(self, "composite_body") and hasattr(self.composite_body, "bodies"):
                cnt_arrays = [
                    body.cnt_update.cpu().numpy().copy()
                    for body in self.composite_body.bodies
                    if hasattr(body, "cnt_update")
                ]
                if cnt_arrays:
                    self._submit_io(
                        self._save_contours_h5, cnt_arrays, iteration,
                    )

        if check_termination:
            return self.check_termination(iteration, u, v, p)
        return False

    # keep old name as an alias so call-sites in alternative solver variants still work
    def plotting_debug(self, u, v, p, iteration, check_termination=True):
        return self.plotting_and_saving(u, v, p, iteration, check_termination=check_termination)

    def check_termination(self, iteration, u, v, p):
        # NaN in any velocity or pressure field
        has_nan = (torch.isnan(u).any() or torch.isnan(v).any()
                   or torch.isnan(p).any())
        if iteration == self.nt - 1 or has_nan:
                if has_nan:
                    logger.warning("Termination: NaN detected in velocity/pressure fields")
                else:
                    logger.info("Termination: reached max iterations (%d)", self.nt)
                terminate = True
        else:
            if hasattr(self.composite_body, "com_pos"):
                    terminate = not self.inside(self.composite_body.com_pos)
                    if terminate:
                        logger.warning("Termination condition met: body exited domain")
            else:
                terminate = False
        return terminate

    def plotting_saving(self, u, v, p, iteration):
        """Legacy alias — delegates to the unified method."""
        self.plotting_and_saving(u, v, p, iteration, check_termination=False)

    def save_results(self, u, v, p, iteration, *, w_vel=None):
        """Legacy alias — delegates to the unified HDF5 pipeline."""
        if not self.save:
            return
        h5_fields = {
            "u": u.detach().cpu().numpy().copy(),
            "v": v.detach().cpu().numpy().copy(),
            "p": p.detach().cpu().numpy().copy(),
        }
        if w_vel is not None:
            h5_fields["w"] = w_vel.detach().cpu().numpy().copy()
        if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
            h5_fields["sdf"] = self.composite_body.sdf_val.cpu().numpy().copy()
        grids = None
        if iteration == 0 or not hasattr(self, '_grids_saved'):
            grids = {"x": self.x.cpu().numpy().copy(),
                     "y": self.y.cpu().numpy().copy()}
            if self.z is not None:
                grids["z"] = self.z.cpu().numpy().copy()
            self._grids_saved = True
        self._submit_io(self._save_fields_h5, h5_fields, grids, iteration)

    # ------------------------------------------------------------------
    #   Unified HDF5 saving helpers (used by both 2-D and 3-D paths)
    # ------------------------------------------------------------------

    def _save_fields_h5(self, fields, grids, iteration):
        """Write velocity / pressure / SDF fields to a single HDF5 file.

        File layout::

            grids/x, grids/y [, grids/z]   – written once
            fields/000000/u
            fields/000000/v
            fields/000000/p
            fields/000000/w   (3-D only)
            fields/000000/sdf (when a body is present)
            ...

        All derived quantities (vorticity, velocity magnitude, divergence)
        can be recomputed from u, v, [w] and the grid coordinates.
        """
        h5_path = f'{self.save_path}/fields.h5'
        grp = f'fields/{iteration:06d}'
        lock = self._hdf5_lock
        with lock:
            with h5py.File(h5_path, 'a') as f:
                for name, arr in fields.items():
                    f.create_dataset(f'{grp}/{name}', data=arr)
                if grids is not None and 'grids' not in f:
                    for name, arr in grids.items():
                        f.create_dataset(f'grids/{name}', data=arr)

    def _save_contours_h5(self, cnt_arrays, iteration):
        """Write 2-D body contour data to a dedicated HDF5 file."""
        cnt_h5 = f'{self.save_path}/contours.h5'
        lock = self._hdf5_lock
        with lock:
            with h5py.File(cnt_h5, 'a') as f:
                for i, arr in enumerate(cnt_arrays):
                    f.create_dataset(f'{iteration:06d}/body_{i}', data=arr)

    def save_drags_h5(self, path=None):
        """Persist per-body force/torque histories to HDF5.

        Writes ``<save_path>/drags.h5`` with datasets:
            viscous_drags    (n_bodies, 3, nt)  viscous (skin) force  [N]
            pressure_drags   (n_bodies, 3, nt)  pressure (form) force [N]
            viscous_torques  (n_bodies, 3, nt)  viscous torque about COM  [N m]
            pressure_torques (n_bodies, 3, nt)  pressure torque about COM [N m]

        Metadata (dt, rho, nt, link names, geometry) is intentionally
        *not* written here — post-processing scripts can recover it from
        ``parameters.yaml``, ``simulation.hdf5`` and the cached SDFs.
        """
        if path is None:
            path = f'{self.save_path}/drags.h5'
        vd = self.viscous_drag_record.detach().cpu().numpy().copy()
        pd = self.pressure_drag_record.detach().cpu().numpy().copy()
        vt = self.viscous_torque_record.detach().cpu().numpy().copy()
        pt = self.pressure_torque_record.detach().cpu().numpy().copy()
        lock = self._hdf5_lock

        def _save(path, vd, pd, vt, pt):
            with lock:
                with h5py.File(path, 'w') as f:
                    f.create_dataset('viscous_drags',    data=vd)
                    f.create_dataset('pressure_drags',   data=pd)
                    f.create_dataset('viscous_torques',  data=vt)
                    f.create_dataset('pressure_torques', data=pt)

        self._submit_io(_save, path, vd, pd, vt, pt)

    def run_from_initial(self, u0, v0, w0=None):
        u = u0
        v = v0
        p = torch.zeros_like(u)
        if self.ndim == 2:
            for iteration in tqdm(range(self.nt)):
                t                = iteration*self.dt
                (u,v,p,stop_sim) = self.step_(u, v, p, iteration, t)
        else:
            w = w0 if w0 is not None else torch.zeros_like(u)
            for iteration in tqdm(range(self.nt)):
                t                      = iteration*self.dt
                (u,v,p,w,stop_sim) = self.step_(u, v, p, iteration, t, w_vel=w)

    def run_sim(self):
        u = self.u0
        v = self.v0
        p = self.p0
        if self.ndim == 2:
            for iteration in tqdm(range(self.starting_iteration, self.nt)):
                t                = iteration*self.dt
                (u,v,p,stop_sim) = self.step_(u, v, p, iteration, t)
        else:
            w = self.w0
            for iteration in tqdm(range(self.starting_iteration, self.nt)):
                t                      = iteration*self.dt
                (u,v,p,w,stop_sim) = self.step_(u, v, p, iteration, t, w_vel=w)

        if self.compute_forces and self.save:
            self.save_drags_h5()

        # ---- save flow diagnostics ----
        if self.save:
            self._submit_io(self.diagnostics.save_h5,
                            self.save_path, self._hdf5_lock)

        # Block until all background I/O is complete before returning
        self.flush_io()


if __name__ == "__main__":

    solver = FluidSolver()
    solver.run_sim()


