"""
Level-set free-surface (fluid-air) module for lilytorch.

This module implements a *single-fluid* free surface alongside the existing
BDIM solid-body machinery.  The second phase (air) is **not** resolved as a
real Navier–Stokes fluid: it is represented purely as a kinematic boundary
on which the pressure is fixed to ``p_atm == 0`` (gauge).

The implementation follows the standard cheap recipe used in many
research codes (Foster–Fedkiw, Enright et al., Gibou–Fedkiw):

  1. an advected level-set scalar ``phi_fs`` (negative in fluid,
     positive in air) carried on the cell-centred grid;
  2. a few PDE-iteration reinitialisation sweeps to keep ``|∇phi| ≈ 1``;
  3. a ghost-fluid (GFM) pressure boundary condition baked in at the
     Poisson assembly level by rescaling per-face coefficients on faces
     that cross the interface; air-air faces get zero coefficient so
     the existing RBGS smoother automatically drives ``p_air → 0``
     (its ``Jinv = where(|J|≥tol, 1/J, 0)`` masking gives a Dirichlet
     ``p == 0`` for free, see ``poisson_mult._rbgs_*``);
  4. a constant-along-normal velocity extension into a narrow air band
     so the next advection step has a meaningful velocity in air cells
     near the interface.

The class is **fully decoupled** from BDIM: it stores no SDF for any
solid body, never touches ``composite_body``, and exposes only the
plumbing the solver needs (a pressure mask, face-coefficient scales,
and per-step ``advect / reinitialize / extend_velocity`` hooks).

All operations are vectorised pure-PyTorch and run on either CPU or CUDA
with the dtype of the host solver.  They are deliberately simple — no
fused kernels — because the per-step cost is a handful of cell-centred
scalar operations and a few narrow-band sweeps, dwarfed by the
projection / BDIM cost.

Sign convention
---------------
``phi_fs < 0``   → fluid (water)
``phi_fs > 0``   → air
``phi_fs == 0`` → interface
"""

from __future__ import annotations

import torch

from lilytorch.src.advection import advect_scalar, SCHEMES


# =====================================================================
# Small helpers
# =====================================================================

def _cc_velocity_2d(u_mac, v_mac):
    """Interpolate MAC-staggered velocity to cell centres.

    Lilytorch convention (see ``ops.divergence`` and the comments in
    ``solver._fluid_step_kernel_2d``): ``u`` lives on x-faces with shape
    ``(Nx, Ny)``; face ``i`` separates cells ``i-1`` and ``i``.  The
    cell-centred value at cell ``i`` is the average of faces ``i`` and
    ``i+1``.  Same for ``v`` along the y-axis.
    """
    u_cc = 0.5 * (u_mac[:-1, :] + u_mac[1:, :])      # (Nx-1, Ny)
    v_cc = 0.5 * (v_mac[:, :-1] + v_mac[:, 1:])      # (Nx, Ny-1)
    # Pad back to (Nx, Ny) by copying the closest interior column / row.
    u_cc_full = torch.zeros_like(u_mac)
    u_cc_full[:-1, :] = u_cc
    u_cc_full[-1, :]  = u_cc[-1, :]
    v_cc_full = torch.zeros_like(v_mac)
    v_cc_full[:, :-1] = v_cc
    v_cc_full[:, -1]  = v_cc[:, -1]
    return u_cc_full, v_cc_full


def _cc_velocity_3d(u_mac, v_mac, w_mac):
    u_cc_full = torch.zeros_like(u_mac)
    u_cc_full[:-1, :, :] = 0.5 * (u_mac[:-1, :, :] + u_mac[1:, :, :])
    u_cc_full[-1, :, :]  = u_cc_full[-2, :, :]
    v_cc_full = torch.zeros_like(v_mac)
    v_cc_full[:, :-1, :] = 0.5 * (v_mac[:, :-1, :] + v_mac[:, 1:, :])
    v_cc_full[:, -1, :]  = v_cc_full[:, -2, :]
    w_cc_full = torch.zeros_like(w_mac)
    w_cc_full[:, :, :-1] = 0.5 * (w_mac[:, :, :-1] + w_mac[:, :, 1:])
    w_cc_full[:, :, -1]  = w_cc_full[:, :, -2]
    return u_cc_full, v_cc_full, w_cc_full


def _neumann_pad(q):
    """Zero-gradient (Neumann) padding of ghost cells, in-place."""
    if q.ndim == 2:
        q[0, :]  = q[1, :]
        q[-1, :] = q[-2, :]
        q[:, 0]  = q[:, 1]
        q[:, -1] = q[:, -2]
    else:
        q[0, :, :]  = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        q[:, 0, :]  = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[:, :, 0]  = q[:, :, 1]
        q[:, :, -1] = q[:, :, -2]


def _upwind_grad(phi, vel, h, dim):
    """First-order upwind gradient ``vel · ∂phi/∂x_dim`` (a scalar field
    of the same shape as ``phi``).

    Uses forward difference where ``vel < 0`` and backward difference where
    ``vel > 0``.  Boundary cells are filled with the closest interior cell
    value (zero-gradient extension).  This is the canonical level-set
    advection stencil; combined with forward-Euler in time it is
    monotone-stable under the standard CFL ``dt * |v_max| / h < 1``.
    """
    fwd = torch.zeros_like(phi)
    bwd = torch.zeros_like(phi)
    if dim == 0:
        fwd[:-1, ...] = (phi[1:, ...]  - phi[:-1, ...]) / h
        bwd[1:, ...]  = (phi[1:, ...]  - phi[:-1, ...]) / h
    elif dim == 1:
        fwd[:, :-1, ...] = (phi[:, 1:, ...] - phi[:, :-1, ...]) / h
        bwd[:, 1:, ...]  = (phi[:, 1:, ...] - phi[:, :-1, ...]) / h
    else:  # dim == 2
        fwd[:, :, :-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / h
        bwd[:, :, 1:]  = (phi[:, :, 1:] - phi[:, :, :-1]) / h
    return torch.where(vel >= 0, vel * bwd, vel * fwd)


# =====================================================================
# FreeSurface
# =====================================================================

class FreeSurface:
    """Single-fluid free-surface tracker via a cell-centred level set.

    Parameters
    ----------
    x, y : 1-D tensors
        Cell-centre coordinates of the host solver (including ghost
        cells); their length matches ``solver.nx`` / ``solver.ny``.
    z : 1-D tensor or None
        Cell-centre z-coordinates for 3-D mode; ``None`` in 2-D.
    h : float or 0-D tensor
        Uniform grid spacing.
    phi_init : callable
        ``phi_init(X, Y[, Z]) -> Tensor`` returning the initial
        level-set field on the cell-centred grid.  Convention:
        ``phi < 0`` in fluid, ``phi > 0`` in air.
    theta_min : float
        Lower clamp on the fluid-fraction ``θ`` used in the ghost-fluid
        face-coefficient scaling (1/θ).  Prevents singular stencils at
        cut faces where the interface grazes a cell face.
    band_cells : int
        Half-width (in cells) of the narrow band over which
        reinitialisation, velocity extension and GFM stencil bookkeeping
        are non-trivial.  All ops are still applied globally — the band
        is informational and used only by reinit/extend convergence
        sizing.
    reinit_iters : int
        Number of explicit reinitialisation sub-steps per
        ``reinitialize()`` call.  Each sub-step is one forward-Euler
        update of ``∂φ/∂τ = sign(φ)(1 − |∇φ|)`` with the Godunov
        upwind discretisation of ``|∇φ|``.
    extend_iters : int
        Number of explicit velocity-extension sub-steps per
        ``extend_velocity()`` call.  Each sub-step solves one
        forward-Euler update of ``∂q/∂τ + sign(φ) n̂ · ∇q = 0``
        in the air half-space (``φ > 0``).
    device, dtype : torch
        Inherited from the host solver.
    """

    # ------------------------------------------------------------------
    def __init__(self, x, y, h, phi_init, *,
                 z=None, theta_min=0.01, band_cells=4,
                 reinit_iters=4, extend_iters=4,
                 convection_method="quick",
                 device=None, dtype=None):
        self.device = device if device is not None else x.device
        self.dtype  = dtype  if dtype  is not None else x.dtype
        self.h      = float(h)
        self.ndim   = 2 if z is None else 3
        self.theta_min   = float(theta_min)
        self.band_cells  = int(band_cells)
        self.reinit_iters = int(reinit_iters)
        self.extend_iters = int(extend_iters)

        # Convective scheme for the level-set advection — reuses the same
        # registry as the Navier–Stokes advection (advection.SCHEMES) so the
        # interface is transported at the same order as the flow rather than
        # with a bespoke first-order upwind.  Time integration is forward
        # Euler (one substep per solver step on the projected velocity).
        if convection_method not in SCHEMES:
            raise ValueError(
                f"Unknown free-surface convection_method "
                f"'{convection_method}'. Choose from {sorted(SCHEMES)}."
            )
        self._scheme = SCHEMES[convection_method]
        self._dh     = [self.h] * self.ndim

        # Mesh tensors
        if self.ndim == 2:
            X, Y = torch.meshgrid(x, y, indexing="ij")
            self.phi_fs = phi_init(X, Y).to(device=self.device, dtype=self.dtype).contiguous()
        else:
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            self.phi_fs = phi_init(X, Y, Z).to(device=self.device, dtype=self.dtype).contiguous()
        _neumann_pad(self.phi_fs)

        # Cached views
        self._x = x
        self._y = y
        self._z = z

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def air_mask_cc(self):
        """Boolean cell-centred mask: ``True`` where ``phi_fs > 0`` (air)."""
        return self.phi_fs > 0

    @property
    def fluid_mask_cc(self):
        return self.phi_fs <= 0

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def advect(self, *vels, dt):
        """Forward-Euler advection of ``phi_fs`` by ``vels``.

        Uses the shared convective schemes (QUICK / ADBQUICKEST / … via
        :func:`lilytorch.src.advection.advect_scalar`) on the MAC velocity,
        in conservative flux form.  For the divergence-free projected
        velocity this is equivalent to ``∂φ/∂t + u·∇φ = 0`` but transported
        at the scheme's order rather than first-order upwind.

        Parameters
        ----------
        *vels : MAC-staggered velocity tensors ``(u, v)`` in 2-D
                or ``(u, v, w)`` in 3-D (same grid shape as ``phi_fs``).
        dt : float
            Time step.
        """
        self.phi_fs = advect_scalar(
            self.phi_fs, *vels,
            scheme=self._scheme, dt=float(dt), dh=self._dh,
        )
        _neumann_pad(self.phi_fs)

    # ------------------------------------------------------------------
    # Reinitialisation:  ∂φ/∂τ = sign(φ)(1 − |∇φ|)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reinitialize(self, n_iter=None):
        """Drive ``|∇phi| → 1`` while preserving the zero-level set.

        Uses Sussman et al.'s standard pseudo-time PDE with a smoothed
        sign function and a forward-Euler update with sub-step
        ``Δτ = 0.5 h``.  Boundary cells are zero-gradient.
        """
        if n_iter is None:
            n_iter = self.reinit_iters
        if n_iter <= 0:
            return

        h = self.h
        dtau = 0.5 * h                           # CFL ≤ 1 for unit speed
        # Smoothed sign uses the *frozen* initial phi so the zero-level set
        # is not allowed to drift during the inner sub-steps.
        phi0 = self.phi_fs.clone()
        eps  = h
        sign = phi0 / torch.sqrt(phi0 * phi0 + eps * eps)

        for _ in range(n_iter):
            grad_mag = self._godunov_grad_magnitude(self.phi_fs, sign)
            self.phi_fs.add_(sign * (1.0 - grad_mag), alpha=dtau)
            _neumann_pad(self.phi_fs)

    # ------------------------------------------------------------------
    def _godunov_grad_magnitude(self, phi, sign):
        """Godunov-upwind ``|∇phi|`` consistent with the sign of phi.

        Standard formula: for ``sign > 0`` use
        ``sqrt( max(D-,0)² + min(D+,0)² )`` along each axis;
        for ``sign < 0`` swap.
        """
        h = self.h
        s_pos = sign > 0

        def _axis(dim):
            fwd = torch.zeros_like(phi)
            bwd = torch.zeros_like(phi)
            if dim == 0:
                fwd[:-1, ...] = (phi[1:, ...] - phi[:-1, ...]) / h
                bwd[1:, ...]  = (phi[1:, ...] - phi[:-1, ...]) / h
            elif dim == 1:
                fwd[:, :-1, ...] = (phi[:, 1:, ...] - phi[:, :-1, ...]) / h
                bwd[:, 1:, ...]  = (phi[:, 1:, ...] - phi[:, :-1, ...]) / h
            else:
                fwd[:, :, :-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / h
                bwd[:, :, 1:]  = (phi[:, :, 1:] - phi[:, :, :-1]) / h
            a = torch.where(s_pos,
                            torch.clamp(bwd,  min=0.0),
                            torch.clamp(bwd,  max=0.0))
            b = torch.where(s_pos,
                            torch.clamp(fwd,  max=0.0),
                            torch.clamp(fwd,  min=0.0))
            return a * a + b * b

        g2 = _axis(0) + _axis(1)
        if self.ndim == 3:
            g2 = g2 + _axis(2)
        return torch.sqrt(g2 + 1e-30)

    # ------------------------------------------------------------------
    # Velocity extension (constant along normal in the air band)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extend_velocity(self, *vels_cc, n_iter=None, full=False):
        """Extend ``vels_cc`` from the fluid half-space (``phi<=0``)
        into the air half-space (``phi>0``) along ``∇phi``.

        Parameters
        ----------
        *vels_cc : cell-centred velocity components (modified in-place).
        n_iter   : optional override of ``extend_iters``.
        full     : if True, ignore ``n_iter`` and iterate enough sub-steps
                   to sweep the **entire** air region (not just the
                   near-interface band).  The extension front advances
                   ~½ cell per sub-step (``Δτ = ½h``), so ``2·max(grid)``
                   sub-steps guarantee full coverage; once a cell's value
                   is constant along the normal it is a fixed point, so
                   over-iterating is harmless.  This makes the air a clean
                   normal-extension of the water velocity everywhere,
                   preventing the predictor from accumulating spurious,
                   undamped vorticity in the (decoupled) bulk air.

        Notes
        -----
        Solves ``∂q/∂τ + s n̂·∇q = 0`` with ``s = sign(phi)`` and ``n̂ =
        ∇phi/|∇phi|`` using an upwind discretisation.  Each sub-step is
        forward-Euler with ``Δτ = 0.5 h``.  Only cells with ``phi > 0``
        are updated; fluid cells are frozen.
        """
        if full:
            n_iter = 2 * max(self.phi_fs.shape)
        elif n_iter is None:
            n_iter = self.extend_iters
        if n_iter <= 0:
            return

        h = self.h
        dtau = 0.5 * h

        # Normal direction = sign(phi)·∇phi/|∇phi|, taken on cell centres.
        nx, ny, nz = self._cc_unit_normal()
        air = self.air_mask_cc.to(self.dtype)

        for q in vels_cc:
            for _ in range(n_iter):
                dqdx = _upwind_grad(q, nx, h, 0)
                dqdy = _upwind_grad(q, ny, h, 1)
                rhs  = dqdx + dqdy
                if self.ndim == 3:
                    rhs = rhs + _upwind_grad(q, nz, h, 2)
                q.sub_(air * rhs, alpha=dtau)
                _neumann_pad(q)

    # ------------------------------------------------------------------
    def _cc_unit_normal(self):
        """Cell-centred unit normal ``n̂ = sign(phi)·∇phi/|∇phi|``.

        ``sign(phi)`` factor folds the propagation direction so that the
        upwind scheme transports info *outward* from the interface into
        the air band (same convention as the extension PDE
        ``∂q/∂τ + s n̂·∇q = 0``).
        """
        h = self.h
        phi = self.phi_fs

        def _cc(dim):
            g = torch.zeros_like(phi)
            if dim == 0:
                g[1:-1, ...] = (phi[2:, ...] - phi[:-2, ...]) / (2 * h)
                g[0, ...]    = g[1, ...]
                g[-1, ...]   = g[-2, ...]
            elif dim == 1:
                g[:, 1:-1, ...] = (phi[:, 2:, ...] - phi[:, :-2, ...]) / (2 * h)
                g[:, 0, ...]    = g[:, 1, ...]
                g[:, -1, ...]   = g[:, -2, ...]
            else:
                g[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * h)
                g[:, :, 0]    = g[:, :, 1]
                g[:, :, -1]   = g[:, :, -2]
            return g

        gx = _cc(0)
        gy = _cc(1)
        if self.ndim == 3:
            gz = _cc(2)
        else:
            gz = None
        mag2 = gx * gx + gy * gy + (gz * gz if gz is not None else 0.0)
        inv  = torch.rsqrt(mag2 + 1e-30)
        s    = torch.sign(phi)
        return (s * gx * inv,
                s * gy * inv,
                (s * gz * inv) if gz is not None else None)

    # ------------------------------------------------------------------
    # Ghost-fluid pressure-Poisson coefficient scaling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ghost_fluid_face_scales(self):
        """Return per-face GFM rescaling factors ``s_u, s_v[, s_w]``.

        The host solver multiplies its existing staggered Poisson
        coefficients (``ch, cv, cw`` — i.e. ``dt/ρ_eff`` on each face
        grid) **elementwise** by these scales before passing them into
        :class:`PoissonSolver`.  The rule on each face is:

        * fluid-fluid face → scale ``= 1`` (no change);
        * air-air     face → scale ``= 0``  (decouples air cells from
          the smoother — combined with RBGS / weight-1 Jacobi this
          drives the air-side ``p_air → 0``);
        * cut         face (fluid ↔ air) → scale ``= 1 / max(θ, θ_min)``
          where ``θ = |φ_fluid| / (|φ_fluid| + |φ_air|)`` is the fluid
          fraction of the cell-pair gap.  The lower clamp ``θ_min``
          stabilises the stencil when the interface grazes a face.

        Returned tensors have **face-grid** shapes:

        * 2-D: ``s_u`` shape ``(Nx-1, Ny)``, ``s_v`` shape ``(Nx, Ny-1)``;
        * 3-D: also ``s_w`` shape ``(Nx, Ny, Nz-1)``.

        These match the slice conventions used in ``solver.project``
        when ``_face_grid`` is True (see ``solver.project`` for slice
        details).
        """
        phi = self.phi_fs
        theta_min = self.theta_min

        def _scale_along(dim):
            if dim == 0:
                phi_a = phi[:-1, ...]
                phi_b = phi[1:, ...]
            elif dim == 1:
                phi_a = phi[:, :-1, ...]
                phi_b = phi[:, 1:, ...]
            else:
                phi_a = phi[:, :, :-1]
                phi_b = phi[:, :, 1:]
            fluid_a = phi_a <= 0
            fluid_b = phi_b <= 0
            both_fluid = fluid_a & fluid_b
            both_air   = (~fluid_a) & (~fluid_b)
            cut        = ~(both_fluid | both_air)
            # Fluid-side magnitude / total magnitude → fluid fraction.
            abs_a = phi_a.abs()
            abs_b = phi_b.abs()
            theta = torch.where(
                fluid_a,
                abs_a / (abs_a + abs_b + 1e-30),
                abs_b / (abs_a + abs_b + 1e-30),
            )
            theta_clamped = torch.clamp(theta, min=theta_min)
            inv_theta = torch.where(cut, 1.0 / theta_clamped,
                                    torch.ones_like(theta))
            scale = torch.where(both_air, torch.zeros_like(theta), inv_theta)
            return scale.contiguous()

        s_u = _scale_along(0)
        s_v = _scale_along(1)
        if self.ndim == 3:
            s_w = _scale_along(2)
            return s_u, s_v, s_w
        return s_u, s_v

    # ------------------------------------------------------------------
    # Gauge anchor (pressure datum at the free surface)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def interface_gauge_offset(self, p):
        """Constant pressure offset ``C`` that anchors the free-surface gauge.

        The ghost-fluid solve fixes the pressure *gradient* but leaves the
        additive constant (the DC null-space mode) only weakly constrained:
        air cells are decoupled (zero-coefficient air–air faces) and couple
        to the water only through thin cut faces, so the (effectively
        singular) coarse multigrid problems do **not** pin the water datum —
        in practice it even drifts further with more V-cycles.  We instead
        recover the datum from the physical condition ``p = p_atm = 0`` at
        the interface: for every water cell adjacent to air, linearly
        extrapolate the *water* pressure to the φ=0 crossing using the cut
        face's water fraction ``θ`` and the next water cell inward,

            C ≈ p_w + θ (p_w − p_w2),

        which (for a locally linear pressure) equals the gauge constant
        exactly.  Averaging over all cut faces gives a robust single
        offset; subtract it from ``p`` to restore ``p ≈ 0`` at the free
        surface.  This is ρ/g-agnostic and works for a deforming interface.

        Returns a 0-d tensor (``0`` when there are no cut faces).
        """
        phi   = self.phi_fs
        nd    = self.ndim
        water = phi <= 0
        aphi  = phi.abs()
        tot = p.new_zeros(())
        cnt = p.new_zeros(())
        for d in range(nd):
            for sgn in (1, -1):
                # neighbour towards the candidate air side, and the opposite
                # (deeper-water) cell, via roll; the single wrapped boundary
                # row is masked out below.
                phi_nb = torch.roll(phi, shifts=-sgn, dims=d)
                p_opp  = torch.roll(p,   shifts=sgn,  dims=d)
                cut = water & (phi_nb > 0)
                idx = [slice(None)] * nd
                idx[d] = (-1 if sgn == 1 else 0)
                cut[tuple(idx)] = False
                theta = aphi / (aphi + phi_nb.abs() + 1e-30)
                C = p + theta * (p - p_opp)
                tot = tot + torch.where(cut, C, torch.zeros_like(C)).sum()
                cnt = cnt + cut.sum()
        return tot / cnt.clamp(min=1)

    # ------------------------------------------------------------------
    # Air-cell pressure mask
    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply_pressure_mask(self, p):
        """Zero ``p`` in air cells (defensive: the smoother should have
        done this already, but the FFT path and warm-start initial
        guesses can leak non-zero air pressures)."""
        p.masked_fill_(self.air_mask_cc, 0.0)
        return p
