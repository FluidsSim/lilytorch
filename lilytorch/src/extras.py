"""Non-standard / optional solver features extracted from ``solver.py``.

Item #4 of the HIGH PRIORITY backlog: move sponge layer, yield damping,
Smagorinsky LES, Carreau non-Newtonian, and the variable-viscosity
dispatchers into a single auxiliary file so that the core
:class:`FluidSolver` is easier to read.

All public symbols here take ``self`` (a ``FluidSolver``) as their first
argument and are bound to ``FluidSolver`` as methods at the bottom of
``solver.py``.  The public API of :class:`FluidSolver` is unchanged.
"""
import torch

from lilytorch.src import operations as ops


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

def _compute_nu_rho_for_forces(self, *vel, out=None):
    """Return ν·ρ for the force computation (scalar or tensor).

    * Constant viscosity:  returns ``self.nu * self.rho``  (scalar).
    * Smagorinsky/Carreau: returns a spatially-varying tensor field.

    The result is cached in ``self._nu_rho_field`` so that the force
    computation does not recompute strain rates a second time.

    Parameters
    ----------
    out : torch.Tensor, optional
        Pre-allocated full-grid buffer.  If given, the variable-viscosity
        result is written in place into ``out`` (avoiding a per-step
        full-grid allocation).  Ignored for constant viscosity.
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
    if out is not None:
        torch.mul(nu_eff, self.rho, out=out)
        self._nu_rho_field = out
    else:
        self._nu_rho_field = nu_eff * self.rho
    return self._nu_rho_field
