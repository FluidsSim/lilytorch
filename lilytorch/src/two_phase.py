"""Two-phase (water + real air) interface tracking via Volume-of-Fluid.

The two-phase model represents the air as a *real* (light) fluid: the interface
is carried by a **volume fraction** ``alpha`` (``1`` water, ``0`` air), and one
set of Navier–Stokes equations is solved with a spatially varying density /
viscosity. This class owns ``alpha``, transports it with the Weymouth & Yue
(2010) conservative VOF scheme (bounded, mass-conservative, no clamping or
reconstruction — see :meth:`advect`), and builds the cell-centred / face
density and viscosity fields that the variable-density pressure projection
consumes.

It is intentionally **decoupled** from the solver: it stores no body SDF, never
touches ``composite_body``, and exposes only plain field accessors. The
:class:`~lilytorch.src.two_phase_solver.TwoPhaseSolver` subclass wires it into
the projection without modifying the base :class:`~lilytorch.src.solver.FluidSolver`.

Sign / value convention::

    alpha = 1   water
    alpha = 0   air
    0 < alpha < 1   interface band
"""

import torch

from lilytorch.src.advection import _sl


def _neumann_pad(q):
    """Copy the first interior cell into the ghost layer on every face
    (zero-gradient), in place."""
    nd = q.ndim
    for d in range(nd):
        q[_sl(nd, d, slice(0, 1))]   = q[_sl(nd, d, slice(1, 2))]
        q[_sl(nd, d, slice(-1, None))] = q[_sl(nd, d, slice(-2, -1))]


class TwoPhase:
    """Volume-of-Fluid water/air interface + variable density/viscosity.

    Parameters
    ----------
    x, y : 1-D coordinate tensors (cell centres incl. ghost cells).
    h : float -- uniform cell size.
    alpha_init : callable -- ``alpha_init(X, Y[, Z])`` returning the initial
        volume fraction on the cell-centred grid (1 water, 0 air).
    z : 1-D tensor or None -- enables 3-D when given.
    rho_water, rho_air : float -- phase densities (default 1000 / 1).
    nu_water, nu_air : float -- phase kinematic viscosities.

    The face material coefficient is the **harmonic** density mean (Weymouth &
    Yue 2011, Eq. 33) — the standard, well-conditioned choice for the
    variable-density pressure Poisson. It is carried as the reciprocal density
    ``1/rho`` (:meth:`recip_density_cc` / :meth:`recip_density_face`), because
    the harmonic density mean is the *arithmetic* mean of ``1/rho`` and the
    projection coefficient ``dt*mu0/rho`` only ever needs the reciprocal.

    Note: the interface scheme is the hardwired Weymouth & Yue conservative
    VOF (:meth:`advect`); there is no scheme/compression choice to make.
    """

    def __init__(self, x, y, h, alpha_init, *,
                 z=None,
                 rho_water=1000.0, rho_air=1.0,
                 nu_water=1.0e-6, nu_air=1.5e-5,
                 device=None, dtype=None):
        self.device = device if device is not None else x.device
        self.dtype  = dtype  if dtype  is not None else x.dtype
        self.h      = float(h)
        self.ndim   = 2 if z is None else 3
        self.rho_water = float(rho_water)
        self.rho_air   = float(rho_air)
        self.nu_water  = float(nu_water)
        self.nu_air    = float(nu_air)

        if self.ndim == 2:
            X, Y = torch.meshgrid(x, y, indexing="ij")
            a = alpha_init(X, Y)
        else:
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            a = alpha_init(X, Y, Z)
        self.alpha = a.to(device=self.device, dtype=self.dtype).contiguous()
        self.alpha.clamp_(0.0, 1.0)
        _neumann_pad(self.alpha)

        self._x, self._y, self._z = x, y, z
        # reference water volume for the mass-conservation diagnostic
        self.initial_water_volume = self.water_volume()

    def reinit_alpha(self, alpha_init):
        """Re-evaluate the interface field from ``alpha_init`` on the stored
        grid (same path as ``__init__``) and reset the mass-diagnostic
        reference.  Used by the deferred body-aware carve, which must wait for
        the FARMS-coupled body poses (unknown at construction time)."""
        if self.ndim == 2:
            X, Y = torch.meshgrid(self._x, self._y, indexing="ij")
            a = alpha_init(X, Y)
        else:
            X, Y, Z = torch.meshgrid(self._x, self._y, self._z, indexing="ij")
            a = alpha_init(X, Y, Z)
        self.alpha = a.to(device=self.device, dtype=self.dtype).contiguous()
        self.alpha.clamp_(0.0, 1.0)
        _neumann_pad(self.alpha)
        self.initial_water_volume = self.water_volume()

    # ------------------------------------------------------------------
    # Masks (for plotting / diagnostics; the solve uses the smooth fields)
    # ------------------------------------------------------------------
    @property
    def water_mask_cc(self):
        return self.alpha >= 0.5

    @property
    def air_mask_cc(self):
        return self.alpha < 0.5

    # ------------------------------------------------------------------
    # Variable material fields
    # ------------------------------------------------------------------
    def recip_density_cc(self):
        """Cell-centred **reciprocal** fluid density ``1 / rho_cc`` with
        ``rho_cc = alpha*rho_water + (1-alpha)*rho_air``.

        The variable-density projection only ever needs ``1/rho`` (the Poisson
        coefficient is ``dt*mu0/rho`` and the harmonic face density mean is the
        arithmetic mean of ``1/rho``). Carrying the reciprocal directly avoids
        materialising the dimensional ``rho`` field and a separate harmonic
        blend.
        """
        return 1.0 / (self.alpha * self.rho_water
                      + (1.0 - self.alpha) * self.rho_air)

    def viscosity_cc(self):
        """Cell-centred kinematic viscosity (volume-weighted)."""
        return self.alpha * self.nu_water + (1.0 - self.alpha) * self.nu_air

    def recip_density_face(self, d):
        """Reciprocal fluid density ``1/rho`` on the staggered *d*-face grid
        (full-grid tensor).

        MAC convention: the *d*-face at index ``i`` lies between cells ``i-1``
        and ``i``; the face reciprocal is the arithmetic mean
        ``0.5*(1/rho_{i-1} + 1/rho_i)``. This is exactly the reciprocal of the
        **harmonic** face density (Weymouth & Yue 2011, Eq. 33). The boundary
        face (``i=0``) copies the adjacent cell.
        """
        q   = self.recip_density_cc()
        nd  = self.ndim
        lo  = q[_sl(nd, d, slice(None, -1))]   # cells i-1
        hi  = q[_sl(nd, d, slice(1, None))]    # cells i
        out = q.clone()
        out[_sl(nd, d, slice(1, None))] = 0.5 * (lo + hi)
        return out

    def water_volume(self):
        """Total water volume ``h^ndim * sum(alpha_interior)`` (mass proxy)."""
        inner = tuple(slice(1, -1) for _ in range(self.ndim))
        return float(self.alpha[inner].sum().item()) * (self.h ** self.ndim)

    # ------------------------------------------------------------------
    # VOF transport
    # ------------------------------------------------------------------
    @torch.no_grad()
    def advect(self, *vels, dt):
        """Weymouth & Yue (2010) conservative VOF transport.

        Dimensional operator split (Lie); per direction ``d`` the interior
        update is

            a_i += (dt/h) [ F_{i-1/2} - F_{i+1/2} + a_i (u_{i+1/2} - u_{i-1/2}) ]

        with an upwind face flux ``F = u_face * a_upwind``.  The
        **divergence-correction** term ``a_i (u_R - u_L)`` is the key
        ingredient: it makes each 1-D sweep bounded in ``[0,1]`` (for
        CFL <= 1) and, summed over the ``D`` sweeps with a discretely
        divergence-free velocity, conserves total volume to round-off — with
        **no clamping and no interface reconstruction**
        (Weymouth & Yue, *Conservative VOF method...*, JCP **229** (2010) 2853;
        the scheme used by lily-pad / WaterLily and the BDIM+VOF coupling).
        The sweep order alternates each step to limit directional bias.  The
        face value is the W&Y 2nd-order Courant-corrected, van-Leer-limited
        donor extrapolation (see :meth:`_cvof_sweep`), which keeps the
        interface sharp while staying bounded.
        """
        dt = float(dt)
        order = list(range(self.ndim))
        self._sweep_parity = not getattr(self, "_sweep_parity", False)
        if self._sweep_parity:
            order = order[::-1]
        a = self.alpha
        for d in order:
            _neumann_pad(a)
            a = self._cvof_sweep(a, vels[d], d, dt)
        _neumann_pad(a)
        self.alpha = a

    def _shift(self, a, s, d):
        """Shift ``a`` by ``s`` cells along dim ``d`` with **edge replication**
        (Neumann-consistent), unlike ``torch.roll`` which wraps the boundary.
        Supports ``s in (1, 2, -1)`` (the cVOF stencil offsets)."""
        nd = self.ndim
        S  = lambda sl: _sl(nd, d, sl)
        if s == 1:                       # a[k-1], boundary -> a[0]
            return torch.cat([a[S(slice(0, 1))], a[S(slice(0, -1))]], dim=d)
        if s == 2:                       # a[k-2], boundary -> a[0]
            return torch.cat([a[S(slice(0, 1))], a[S(slice(0, 1))],
                              a[S(slice(0, -2))]], dim=d)
        if s == -1:                      # a[k+1], boundary -> a[-1]
            return torch.cat([a[S(slice(1, None))], a[S(slice(-1, None))]], dim=d)
        raise ValueError(f"_shift: unsupported offset {s}")

    def _cvof_sweep(self, a, u_d, d, dt):
        """One Weymouth-Yue conservative directional sweep along dim ``d``.

        MAC convention: ``u_d[k]`` is the face *left* of cell ``k`` (between
        cells ``k-1`` and ``k``).  The face value is the W&Y 2nd-order
        Courant-corrected, van-Leer-limited extrapolation of the **donor**
        cell to the face (first-order upwind when the slope limiter kills the
        gradient), plus the divergence correction.  Returns a new tensor with
        the interior updated.
        """
        nd  = self.ndim
        S   = lambda s: _sl(nd, d, s)
        cfl = dt / self.h
        C   = u_d * cfl                                   # face Courant number
        # neighbour shifts along d with EDGE-CLAMP (Neumann-consistent), NOT
        # torch.roll: roll wraps top<->bottom / left<->right, which corrupts the
        # face values at the domain corners in an order-dependent way and shows
        # up as a one-corner asymmetry in an otherwise symmetric problem.
        a_m1 = self._shift(a,  1, d)                      # a[k-1]
        a_m2 = self._shift(a,  2, d)                      # a[k-2]
        a_p1 = self._shift(a, -1, d)                      # a[k+1]

        def _vleer(db, df):
            # van Leer (harmonic) limited slope; 0 at extrema / sign changes.
            denom = torch.where(db + df == 0.0,
                                torch.ones_like(db), db + df)
            s = 2.0 * db * df / denom
            return torch.where(db * df > 0.0, s, torch.zeros_like(s))

        # C >= 0: donor = cell k-1, extrapolate forward to the face
        s_pos    = _vleer(a_m1 - a_m2, a - a_m1)
        face_pos = a_m1 + 0.5 * (1.0 - C) * s_pos
        # C < 0: donor = cell k, extrapolate backward to the face
        s_neg    = _vleer(a - a_m1, a_p1 - a)
        face_neg = a - 0.5 * (1.0 + C) * s_neg
        F = u_d * torch.where(C >= 0.0, face_pos, face_neg)   # flux at face k
        out = a.clone()
        FL = F[S(slice(1, -1))]        # left face of interior cell i  (index i)
        FR = F[S(slice(2, None))]      # right face of interior cell i (index i+1)
        uL = u_d[S(slice(1, -1))]
        uR = u_d[S(slice(2, None))]
        ai = a[S(slice(1, -1))]
        out[S(slice(1, -1))] = ai + cfl * (FL - FR + ai * (uR - uL))
        return out
