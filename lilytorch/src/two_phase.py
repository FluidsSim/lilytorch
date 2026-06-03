"""Two-phase (water + real air) interface tracking via Volume-of-Fluid.

The two-phase model represents the air as a *real* (light) fluid: the interface
is carried by a **volume fraction** ``alpha`` (``1`` water, ``0`` air), and one
set of Navier–Stokes equations is solved with a spatially varying density /
viscosity. This class owns ``alpha``, transports it (bounded, mass-conservative,
optionally interface-compressed), and builds the cell-centred / face density and
viscosity fields that the variable-density pressure projection consumes.

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

from lilytorch.src.advection import SCHEMES, advect_scalar, _sl


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
    advection : str -- bounded convective scheme from
        :data:`lilytorch.src.advection.SCHEMES` (default ``"cubista"``; use a
        TVD scheme — ``cubista`` / ``vanLeer`` — so ``alpha`` stays bounded).
    compression : float -- interface-compression strength ``C_alpha``
        (0 disables; ~1 sharpens the interface against numerical smearing).
    face_density : ``"arithmetic"`` | ``"harmonic"`` -- face-density average
        used to build the projection coefficients.
    """

    def __init__(self, x, y, h, alpha_init, *,
                 z=None,
                 rho_water=1000.0, rho_air=1.0,
                 nu_water=1.0e-6, nu_air=1.5e-5,
                 advection="cubista", compression=1.0,
                 face_density="harmonic",
                 device=None, dtype=None):
        self.device = device if device is not None else x.device
        self.dtype  = dtype  if dtype  is not None else x.dtype
        self.h      = float(h)
        self.ndim   = 2 if z is None else 3
        self.rho_water = float(rho_water)
        self.rho_air   = float(rho_air)
        self.nu_water  = float(nu_water)
        self.nu_air    = float(nu_air)
        self.compression = float(compression)
        if face_density not in ("arithmetic", "harmonic"):
            raise ValueError("face_density must be 'arithmetic' or 'harmonic'")
        self.face_density = face_density

        if advection not in SCHEMES:
            raise ValueError(
                f"Unknown two-phase advection scheme '{advection}'. "
                f"Choose a bounded scheme from {sorted(SCHEMES)} "
                f"(cubista / vanLeer recommended)."
            )
        self._scheme = SCHEMES[advection]
        self._dh     = [self.h] * self.ndim

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
    def density_cc(self):
        """Cell-centred fluid density ``alpha*rho_water + (1-alpha)*rho_air``."""
        return self.alpha * self.rho_water + (1.0 - self.alpha) * self.rho_air

    def viscosity_cc(self):
        """Cell-centred kinematic viscosity (volume-weighted)."""
        return self.alpha * self.nu_water + (1.0 - self.alpha) * self.nu_air

    def density_face(self, d):
        """Fluid density on the staggered *d*-face grid (full-grid tensor).

        With the MAC convention used by the projection coefficients, the
        *d*-face at index ``i`` lies between cells ``i-1`` and ``i``; the
        boundary face (``i=0``) copies the adjacent cell. Arithmetic or
        harmonic average per ``face_density``.
        """
        rho = self.density_cc()
        nd  = self.ndim
        lo  = rho[_sl(nd, d, slice(None, -1))]   # cells i-1
        hi  = rho[_sl(nd, d, slice(1, None))]    # cells i
        if self.face_density == "harmonic":
            face_in = 2.0 * lo * hi / (lo + hi)
        else:
            face_in = 0.5 * (lo + hi)
        out = rho.clone()
        out[_sl(nd, d, slice(1, None))] = face_in
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
        """Advance ``alpha`` one forward-Euler step under the MAC velocity.

        Conservative flux form with the configured bounded scheme, plus an
        optional MULES-style interface-compression flux
        ``-div(C_alpha |u| n_hat alpha(1-alpha))`` that counteracts numerical
        smearing of the interface. ``alpha`` is clamped to ``[0,1]`` and
        zero-gradient padded afterwards.
        """
        a = advect_scalar(self.alpha, *vels,
                          scheme=self._scheme, dt=float(dt), dh=self._dh)
        if self.compression > 0.0:
            a = a + self._compression_increment(vels, float(dt))
        # Conservative clamp: a bounded scheme can still over/undershoot
        # [0,1] slightly; naive clamping would DISCARD that mass (a floating
        # body would then spuriously sink/rise). Instead redistribute the
        # clamped defect back into the interface band (weight alpha(1-alpha)),
        # which conserves total volume to round-off while staying bounded.
        inner = tuple(slice(1, -1) for _ in range(self.ndim))
        ai     = a[inner]
        ai_cl  = ai.clamp(0.0, 1.0)
        defect = (ai - ai_cl).sum()
        w      = ai_cl * (1.0 - ai_cl)
        wsum   = w.sum()
        if float(wsum) > 1e-12:
            ai_cl = ai_cl + defect * (w / wsum)
        a[inner] = ai_cl.clamp(0.0, 1.0)
        self.alpha = a
        _neumann_pad(self.alpha)

    def _compression_increment(self, vels, dt):
        """Conservative interface-compression increment on the interior.

        Per axis ``d`` the compressive face flux is
        ``F_d = C_alpha * |u_d|_face * n_hat_d|_face * (alpha(1-alpha))_face``;
        the increment is ``-dt/h * (F[i+1] - F[i])``. Sharpens the interface
        without moving its 0.5 contour (the ``alpha(1-alpha)`` factor and the
        normal direction make it a self-limiting anti-diffusion).
        """
        nd  = self.ndim
        a   = self.alpha
        # cell-centred normal n_hat = grad(alpha)/|grad(alpha)|
        eps = 1.0e-12
        grads = []
        for d in range(nd):
            gp = a[_sl(nd, d, slice(2, None))] - a[_sl(nd, d, slice(None, -2))]
            g  = torch.zeros_like(a)
            g[_sl(nd, d, slice(1, -1))] = gp / (2.0 * self.h)
            grads.append(g)
        gmag = torch.sqrt(sum(g * g for g in grads) + eps)
        nhat = [g / gmag for g in grads]

        ab = (a * (1.0 - a))                       # interface indicator, cc
        rhs = torch.zeros_like(a[tuple(slice(1, -1) for _ in range(nd))])
        for d in range(nd):
            # face-centred quantities on the d-faces (index i between i-1,i)
            u_d   = vels[d]
            uf    = u_d[_sl(nd, d, slice(1, None))]            # face vel (i>=1)
            nf    = 0.5 * (nhat[d][_sl(nd, d, slice(1, None))]
                           + nhat[d][_sl(nd, d, slice(None, -1))])
            abf   = 0.5 * (ab[_sl(nd, d, slice(1, None))]
                           + ab[_sl(nd, d, slice(None, -1))])
            Fface = self.compression * uf.abs() * nf * abf    # on faces i>=1
            # restrict transverse dims to interior to match rhs shape
            inner_t = [slice(1, -1)] * nd
            inner_t[d] = slice(None)
            F = Fface[tuple(inner_t)]                          # (..., n_d-1, ...)
            F_diff = (F[_sl(nd, d, slice(None, -1))]
                      - F[_sl(nd, d, slice(1, None))])
            rhs.add_(F_diff, alpha=dt / self.h)
        out = torch.zeros_like(a)
        out[tuple(slice(1, -1) for _ in range(nd))] = rhs
        return out
