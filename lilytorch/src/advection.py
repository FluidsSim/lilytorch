"""Dimension-agnostic advection for MAC staggered grids + the AdvDiffSolver.

Split out of the former monolithic ``adv_diff.py`` (since removed).  This
module owns:

* the pluggable convective **scheme functions** (QUICK, ADBQUICKEST, CUBISTA,
  van Leer, CDS) — pure ``f(upstream, center, downstream)`` kernels;
* the location-agnostic **flux assembler** :func:`_flux`;
* **momentum** advection (:func:`advect_momentum`, the MAC self/cross-advection
  used by the Navier–Stokes predictor);
* **scalar** advection (:func:`advect_scalar`) for a cell-centred passive
  scalar (e.g. the free-surface level set) — reuses the same schemes/flux;
* the :class:`AdvDiffSolver` orchestrator (BCs + scheme dispatch +
  semi-Lagrangian), which composes :mod:`lilytorch.src.diffusion` for the
  diffusion term so advection and diffusion stay independently testable and
  ``torch.compile``-able.

Dependency rule: this module imports :mod:`diffusion` (a leaf) but **never**
imports ``solver`` or ``two_phase``.  The two-phase model depends on this
module (it reuses ``advect_scalar`` for VOF transport), not the other way round.

Works identically in 2-D ``(x, y)`` and 3-D ``(x, y, z)`` by looping over
spatial dimensions rather than duplicating code per axis -- inspired by
WaterLily.jl.
"""

import torch
from lilytorch.src.kernels import RegularGridInterpolatorAutomatic
from lilytorch.src.kernels import _C as _lilytorch_kernels_C  # noqa: F401  -- registers torch.ops.lilytorch_kernels.*

from lilytorch.src import diffusion



# =====================================================================
# Convective scheme functions:  f(upstream, center, downstream)
# =====================================================================
#
# Convention -- face between cell L (left) and cell R (right):
#   positive flow (L->R):
#       upstream   = f[L-1]   (far upstream)
#       center     = f[L]     (upwind cell)
#       downstream = f[R]     (downwind cell)
#   negative flow (R->L):
#       upstream   = f[R+1]
#       center     = f[R]
#       downstream = f[L]

def median(a, b, c):
    """Element-wise median of three tensors."""
    return torch.maximum(
        torch.minimum(a, b),
        torch.minimum(torch.maximum(a, b), c),
    )


def quick(u, c, d):
    """QUICK scheme -- 3rd-order.

    Uses in-place operations to keep simultaneous intermediate tensors at 3
    (vs. the 5-6 created by the original nested ``median`` calls), cutting
    the peak transient allocation roughly in half at large grid sizes.
    """
    # inner_median = median(10*c - 9*u, c, d)
    t  = 10.0 * c - 9.0 * u       # owned temp #1
    lo = torch.minimum(t, c)       # owned temp #2  (= min(t, c))
    torch.maximum(t, c, out=t)     # t = max(t, c), in-place
    torch.minimum(t, d, out=t)     # t = min(max(t, c), d), in-place
    torch.maximum(lo, t, out=lo)   # lo = median(10c-9u, c, d)
    del t                          # free; lo holds inner_median

    # outer = (5*c + 2*d - u) / 6
    outer = 5.0 * c
    outer.add_(d, alpha=2.0)       # in-place: 5c + 2d
    outer.add_(u, alpha=-1.0)      # in-place: 5c + 2d - u
    outer.div_(6.0)                # in-place: / 6

    # result = median(outer, c, lo)  -- lo is inner_median
    # Peak: lo + outer + lo2 = 3 tensors
    lo2 = torch.minimum(outer, c)
    torch.maximum(outer, c, out=outer)
    torch.minimum(outer, lo, out=outer)
    torch.maximum(lo2, outer, out=lo2)
    del outer, lo
    return lo2


def van_leer(u, c, d):
    """Van Leer flux limiter -- 2nd-order TVD."""
    denom = d - c
    rf = (c - u) / (denom + 1e-30)
    psi = (rf + rf.abs()) / (1.0 + rf.abs())
    # T1a: inline the TVD face value ``c + 0.5*(d-c)*psi`` in-place on the
    # owned ``psi``, reusing the already-live ``denom == d-c`` instead of
    # re-materialising it.  Ordering (×0.5 first) is bit-exact: ×0.5 is an
    # exact power-of-two scale, so this is a single rounding of the same real
    # product, and the final add is commutative in IEEE-754.
    psi.mul_(0.5).mul_(denom).add_(c)
    return torch.where(denom.abs() < 1e-30, c, psi)


def cds(u, c, d):
    """Central difference scheme -- 2nd-order, not TVD."""
    return 0.5 * (c + d)


def abdquickest(u, c, d, C=0.1):
    """ADBQUICKEST scheme -- 3rd-order TVD, Courant-number dependent.

    Refactored to keep peak simultaneous tensor count at 4 (was 5) by
    replacing ``torch.full_like(rf, C_upper)`` with a scalar clamp and
    sequencing A / B so only one extra tensor beyond {denom, rf, psi} is
    live at the peak moment.  Saves ~1 full-grid tensor (~0.5 GiB at 512³).
    """
    C2      = C * C
    C_upper = 2.0 * (1.0 - C)                             # scalar

    denom = d - c                                          # owned temp #1
    rf    = (c - u) / (denom + 1e-30)                     # owned temp #2

    # Compute psi = clamp(min(A, B, C_upper), 0)
    #   A = C_upper * rf
    #   B = [(2+C2-3C) + (1-C2)*rf] / (3-3C)
    # Order: B → scalar-clamp at C_upper → min with A → scalar-clamp at 0
    # Peak: denom + rf + psi + A_temp = 4 tensors (vs 5 in the original).
    _scale  = (1.0 - C2) / (3.0 - 3.0 * C)
    _offset = (2.0 + C2 - 3.0 * C) / (3.0 - 3.0 * C)
    psi  = rf * _scale                                     # owned temp #3
    psi += _offset                                         # in-place: psi = B
    psi.clamp_(max=C_upper)                                # in-place: min(B, C_upper)
    torch.minimum(psi, rf * C_upper, out=psi)              # rf*C_upper = temp #4 (brief)
    del rf
    psi.clamp_(min=0.0)

    # T1a: inline ``c + 0.5*(d-c)*psi`` in-place on the owned ``psi``,
    # reusing the live ``denom`` (bit-exact — see van_leer).
    psi.mul_(0.5).mul_(denom).add_(c)
    return torch.where(denom.abs() < 1e-30, c, psi)


def cubista(u, c, d):
    """CUBISTA scheme -- 2nd-order TVD (Alves, Oliveira & Pinho, 2003).

    Same tensor-sequencing improvement as ``abdquickest``: scalar clamp
    replaces ``torch.full_like(rf, 1.5)`` to avoid a 5th simultaneous
    tensor at the peak moment.
    """
    denom = d - c
    rf    = (c - u) / (denom + 1e-30)

    psi  = 0.75 * rf                                       # owned temp
    psi += 0.25                                            # in-place: 0.75*rf + 0.25
    psi.clamp_(max=1.5)                                    # in-place: min(0.75*rf+0.25, 1.5)
    torch.minimum(psi, rf * 1.5, out=psi)                  # min with 1.5*rf (brief temp)
    del rf
    psi.clamp_(min=0.0)

    # T1a: inline ``c + 0.5*(d-c)*psi`` in-place on the owned ``psi``,
    # reusing the live ``denom`` (bit-exact — see van_leer).
    psi.mul_(0.5).mul_(denom).add_(c)
    return torch.where(denom.abs() < 1e-30, c, psi)


# Scheme registry — shared by AdvDiffSolver and the free-surface level set.
SCHEMES = {
    "quick": quick, "abdquickest": abdquickest,
    "vanLeer": van_leer, "van_leer": van_leer,
    "cds": cds, "cubista": cubista,
}

# Scheme IDs for the fused CUDA ``advect_flux_add`` kernel (T2a).
# Must match the compile-time enum in advection_flux.cu.
_CUDA_SCHEME_IDS: dict[str, int] = {
    "quick": 0,
    "abdquickest": 1,
    "vanLeer": 2,
    "van_leer": 2,
    "cds": 3,
    "cubista": 4,
}


# =====================================================================
# Slicing helpers -- dimension-agnostic index construction
# =====================================================================

def _sl(ndim, dim, s):
    """N-D index tuple: slice *s* on dimension *dim*, full elsewhere."""
    idx = [slice(None)] * ndim
    idx[dim] = s
    return tuple(idx)


def _inner(ndim):
    """Index tuple selecting interior cells: [1:-1] on every dimension."""
    return tuple(slice(1, -1) for _ in range(ndim))


# =====================================================================
# Flux computation  (dimension-agnostic, scheme-parameterised)
# =====================================================================

def _flux(scheme, fv, p, dim):
    """Scheme-weighted flux along dimension *dim*.

    Location-agnostic: works for staggered momentum components and for
    cell-centred scalars alike — it only needs a face-velocity array and a
    field array of compatible shapes.

    Parameters
    ----------
    scheme : callable ``f(upstream, center, downstream)`` — one of the
             scheme functions above.
    fv     : face velocities  (``n[dim]-1`` on *dim*, interior on others).
    p      : field values      (``n[dim]``   on *dim*, interior on others).
    """
    lam = scheme
    D   = dim
    S   = lambda s: _sl(p.ndim, D, s)

    # interior faces (full 3-point stencil available)
    # Compute B1 and B2 as separate named tensors so cond, B1, B2 can
    # be freed immediately after torch.where, reducing the live-tensor
    # count at the torch.cat step.
    fv_in = fv[S(slice(1, -1))]
    cond  = fv_in > 0
    B1    = fv_in * lam(p[S(slice(None, -3))], p[S(slice(1, -2))], p[S(slice(2, -1))])
    B2    = fv_in * lam(p[S(slice(3, None))],  p[S(slice(2, -1))], p[S(slice(1, -2))])
    flux_in = torch.where(cond, B1, B2)
    del cond, B1, B2

    # lo boundary face — CDS fallback for positive flow
    fv_lo = fv[S(slice(0, 1))]
    flux_lo = torch.where(
        fv_lo > 0,
        fv_lo * 0.5 * (p[S(slice(0, 1))] + p[S(slice(1, 2))]),
        fv_lo * lam(p[S(slice(2, 3))], p[S(slice(1, 2))], p[S(slice(0, 1))]),
    )

    # hi boundary face — CDS fallback for negative flow
    fv_hi = fv[S(slice(-1, None))]
    flux_hi = torch.where(
        fv_hi > 0,
        fv_hi * lam(p[S(slice(-3, -2))], p[S(slice(-2, -1))], p[S(slice(-1, None))]),
        fv_hi * 0.5 * (p[S(slice(-2, -1))] + p[S(slice(-1, None))]),
    )

    return torch.cat([flux_lo, flux_in, flux_hi], dim=D)


# =====================================================================
# Face-velocity construction  (dimension-agnostic)
# =====================================================================

def _face_vel(vel, i, d, ndim):
    """Face velocity for the *i*-th momentum eq. along direction *d*.

    d == i  : self-advection — average vel[i] at consecutive d-faces
    d != i  : cross-advection — average vel[d] along dim i to reach
              vel[i]'s stagger location
    """
    if d == i:
        # average vel[i] along its own stagger direction
        lo = [slice(1, -1)] * ndim
        hi = [slice(1, -1)] * ndim
        lo[d] = slice(None, -1)
        hi[d] = slice(1, None)
        return 0.5 * (vel[i][tuple(lo)] + vel[i][tuple(hi)])
    else:
        # average vel[d] along dim i (two adjacent in i-direction)
        # and select full face range in dim d
        lo = [slice(1, -1)] * ndim
        hi = [slice(1, -1)] * ndim
        lo[i] = slice(None, -2)
        hi[i] = slice(1, -1)
        lo[d] = slice(1, None)
        hi[d] = slice(1, None)
        return 0.5 * (vel[d][tuple(lo)] + vel[d][tuple(hi)])


def _field_for_flux(phi, d, ndim):
    """Extract *phi* with interior on all dims except *d* (full)."""
    idx = [slice(1, -1)] * ndim
    idx[d] = slice(None)
    return phi[tuple(idx)]


def _scalar_face_vel(vd, d, ndim):
    """Face velocity for a **cell-centred scalar** along direction *d*.

    A cc scalar's *d*-face (between cells ``j`` and ``j+1``) sits at the
    MAC location of velocity component *d* — face-centred in *d*,
    cell-centred in the transverse dims — so **no transverse averaging is
    needed**: the face velocity is just ``vel[d]`` sampled at the interior
    *d*-faces.

    With the MAC convention ``vel[d][k]`` = the face *left* of cell ``k``,
    the face between cells ``j`` and ``j+1`` is ``vel[d][j+1]``; selecting
    ``[1:]`` on *d* gives the ``n_d - 1`` interior faces, interior on the
    transverse dims (matching :func:`_field_for_flux`'s interior).
    """
    idx = [slice(1, -1)] * ndim
    idx[d] = slice(1, None)
    return vd[tuple(idx)]


# =====================================================================
# Advection update kernels  (forward-Euler, conservative flux form)
# =====================================================================

def advect_momentum(scheme, vel, dt_dh, ndim, accum=None):
    """Convective increment for each MAC momentum component (no diffusion).

    Returns a tuple of per-component interior increments ``rhs_i`` such that
    ``vel_new[i][inner] = vel[i][inner] + rhs_i``.  If *accum* is provided it
    must be a list of per-component starting tensors (e.g. the diffusion
    increment) that the convective fluxes are added into **in place**; this
    preserves the original fused advection+diffusion memory behaviour.
    """
    inner = _inner(ndim)
    out = []
    for i in range(ndim):
        rhs = accum[i] if accum is not None else torch.zeros_like(vel[i][inner])
        for d in range(ndim):
            fv = _face_vel(vel, i, d, ndim)
            p  = _field_for_flux(vel[i], d, ndim)
            F  = _flux(scheme, fv, p, d)
            # In-place accumulation: rhs += dt_dh * (F[:-1] - F[1:]).
            F_diff = (F[_sl(ndim, d, slice(None, -1))]
                      - F[_sl(ndim, d, slice(1, None))])
            rhs.add_(F_diff, alpha=float(dt_dh[d]))
            del fv, F, F_diff  # free before next d-iteration to avoid stacking
        out.append(rhs)
    return tuple(out)


def advect_scalar(phi, *vel, scheme, dt, dh):
    """Forward-Euler advection of a cell-centred scalar by a MAC velocity.

        phi^{n+1} = phi^n - dt * div(vel ⊗ phi)

    Conservative flux form; for a divergence-free ``vel`` this equals the
    advective form ``vel·∇phi``.  Reuses the same scheme/flux machinery as
    the momentum path via :func:`_flux` and :func:`_scalar_face_vel`.

    Parameters
    ----------
    phi    : cell-centred scalar (interior + ghost cells).
    *vel   : MAC velocity components ``(u, v[, w])``, same shape as ``phi``.
    scheme : callable convective scheme (e.g. :func:`quick`).
    dt     : float time step.
    dh     : list of floats — grid spacing per dimension.

    Returns a new tensor (ghost cells copied from ``phi``; the caller is
    expected to re-apply its scalar BC / ghost padding afterwards).
    """
    ndim  = phi.ndim
    inner = _inner(ndim)
    rhs   = torch.zeros_like(phi[inner])
    for d in range(ndim):
        fv = _scalar_face_vel(vel[d], d, ndim)
        p  = _field_for_flux(phi, d, ndim)
        F  = _flux(scheme, fv, p, d)
        F_diff = (F[_sl(ndim, d, slice(None, -1))]
                  - F[_sl(ndim, d, slice(1, None))])
        rhs.add_(F_diff, alpha=float(dt) / float(dh[d]))
        del fv, F, F_diff
    out = phi.clone()
    out[inner] += rhs
    return out


# =====================================================================
# Advection-diffusion solver
# =====================================================================
class AdvDiffSolver:
    """
    Dimension-agnostic advection-diffusion solver on a MAC staggered grid.

    Works identically in 2-D ``(x, y)`` and 3-D ``(x, y, z)`` by looping
    over spatial dimensions rather than duplicating code per axis.

    Supported methods
    -----------------
    * ``'quick'``                              -- QUICK (default)
    * ``'abdquickest'``                        -- ADBQUICKEST TVD
    * ``'cubista'``                            -- CUBISTA TVD
    * ``'vanLeer'``                            -- van Leer TVD
    * ``'cds'``                                -- central difference
    * ``'semi-lagrangian'`` / ``'implicit'``   -- Stam 1999

    For explicit schemes the time integration is forward-Euler:

        u^{n+1} = u^n + dt * [-div(vel (x) u) + nu * laplacian(u)]

    The convective fluxes come from :func:`advect_momentum` and the
    diffusion term from :mod:`lilytorch.src.diffusion`, composed here.
    """

    # -- construction ---------------------------------------------------
    def __init__(
        self,
        device,
        dt,
        x,
        y,
        nu,
        BC_type_u=("D", "D", "D", "D"),
        BC_values_u=(0, 0, 0, 0),
        BC_type_v=("D", "D", "D", "D"),
        BC_values_v=(0, 0, 0, 0),
        method="quick",
        z=None,
        BC_type_w=None,
        BC_values_w=None,
    ):
        """
        Parameters
        ----------
        device : torch.device
        dt     : float -- time step
        x, y   : 1-D tensors -- cell-centre coordinates (incl. ghost cells)
        nu     : float -- kinematic viscosity
        method : str -- convection scheme name
        z      : 1-D tensor or None -- cell-centre z-coordinates (3-D mode)
        BC_type_w, BC_values_w : boundary conditions for w (3-D only)
        """
        self.device = device
        self.dtype  = x.dtype
        self.dt     = float(dt)   # ensure Python float so _dt_dh never holds tensors
        self.nu     = nu

        # ---- dimension-agnostic grid setup ----------------------------
        self.coords = [x, y] if z is None else [x, y, z]
        self.ndim   = len(self.coords)
        self.n      = [len(c) for c in self.coords]
        self.dh     = [float(c[1] - c[0]) for c in self.coords]

        self._dt_dh  = [self.dt / h for h in self.dh]   # self.dt is already float
        self._inv_dh2 = [1.0 / (h * h) for h in self.dh]

        # ---- legacy accessors (backward-compat with solver.py) -------
        self.x, self.y = x, y
        self.nx, self.ny = self.n[0], self.n[1]
        self.dx, self.dy = self.dh[0], self.dh[1]
        self.dtdx, self.dtdy = self._dt_dh[0], self._dt_dh[1]
        self.dtdx2 = self.dtdx / self.dh[0]
        self.dtdy2 = self.dtdy / self.dh[1]
        if self.ndim == 3:
            self.z  = z
            self.nz = self.n[2]
            self.dz = self.dh[2]
            self.dtdz  = self._dt_dh[2]
            self.dtdz2 = self.dtdz / self.dh[2]

        # ---- boundary conditions (2*ndim faces per component) --------
        n_faces = 2 * self.ndim

        def _pad(seq, length, default):
            out = list(seq)
            return out + [default] * max(0, length - len(out))

        self._bc_types = [
            _pad(BC_type_u, n_faces, "N"),
            _pad(BC_type_v, n_faces, "N"),
        ]
        self._bc_values = [
            _pad(BC_values_u, n_faces, 0),
            _pad(BC_values_v, n_faces, 0),
        ]
        if self.ndim == 3:
            self._bc_types.append(_pad(BC_type_w or (), n_faces, "N"))
            self._bc_values.append(_pad(BC_values_w or (), n_faces, 0))

        # legacy BC accessors (backward-compat with solver.py)
        self.BC_type_u   = self._bc_types[0]
        self.BC_values_u = self._bc_values[0]
        self.BC_type_v   = self._bc_types[1]
        self.BC_values_v = self._bc_values[1]
        if self.ndim == 3:
            self.BC_type_w   = self._bc_types[2]
            self.BC_values_w = self._bc_values[2]

        # ---- precompute BC operations (avoid per-call allocations) ---
        (self._bc_neumann_ops,
         self._bc_dirichlet_ops,
         self._bc_reflect_ops) = self._build_bc_ops()

        # ---- packed descriptors for the fused set_BCs CUDA op --------
        # Built once here (shape-independent); ``shapes`` and ``dir_val``
        # are cached lazily on first set_BCs() call (they need vel
        # tensor info we don't yet have).  Same descriptor format for
        # 2-D and 3-D — only ``axis`` is restricted to {0, 1} in 2-D.
        self._bc_fused_3d_packed = None
        self._bc_fused_3d_cache  = None
        self._bc_fused_2d_packed = None
        self._bc_fused_2d_cache  = None
        if self.ndim == 3:
            self._bc_fused_3d_packed = self._pack_bc_descriptors_3d()
        elif self.ndim == 2:
            self._bc_fused_2d_packed = self._pack_bc_descriptors_3d()

        # ---- method dispatch -----------------------------------------
        if method in SCHEMES:
            self._scheme      = SCHEMES[method]
            self._scheme_name = method        # used by _get_step_scheme
            self.solve        = self._solve_convective
        elif method in ("semi-lagrangian", "implicit"):
            self._scheme_name = method
            self._init_semi_lagrangian()
            self.solve = self._solve_semi_lagrangian
        else:
            raise ValueError(
                f"Unknown convection method '{method}'. Choose from: "
                f"{sorted(set(list(SCHEMES.keys()) + ['semi-lagrangian', 'implicit']))}"
            )

        # Multi-stream dispatch: set True externally (e.g. from FluidSolver)
        # when device is CUDA.  Lazy-initialised per-component streams stored
        # in _adv_streams; None until first use.
        _dev = device if isinstance(device, torch.device) else torch.device(device)
        self._is_cuda     = _dev.type == "cuda"
        self._use_streams = False
        self._adv_streams = None

        print(f"Using the {method} method for the adv-diff equation ({self.ndim}D)")

    # -----------------------------------------------------------------
    # Semi-Lagrangian initialisation  (Stam 1999, N-D)
    # -----------------------------------------------------------------
    def _init_semi_lagrangian(self):
        ndim = self.ndim
        stag = [c - h / 2 for c, h in zip(self.coords, self.dh)]

        self._interps     = []
        self._flat_coords = []

        for i in range(ndim):
            # component-i lives on a grid staggered in dim i only
            grid = tuple(stag[d] if d == i else self.coords[d]
                         for d in range(ndim))
            interp = RegularGridInterpolatorAutomatic(
                grid,
                torch.zeros(tuple(self.n), device=self.device, dtype=self.dtype),
                fill_value=None, method="quadratic",
            )
            self._interps.append(interp)

            grids = torch.meshgrid(*grid, indexing="ij")
            self._flat_coords.append(
                [g.flatten().clone().detach() for g in grids]
            )

    # =================================================================
    # Convective-scheme solve  (advection + diffusion, dimension-agnostic)
    # =================================================================

    def _get_step_scheme(self, vel):
        """Return the advection scheme callable for this step.

        For ABDQUICKEST the TVD limiter parameter C must equal the actual
        advective Courant number |u|·dt/h.  Computing max-|u| requires one
        GPU→CPU sync (.amax().item()), so it is done HERE — once per step,
        outside any CUDA stream — and captured in a closure passed to _flux.
        All other schemes return self._scheme unchanged (no sync).
        """
        if self._scheme_name == 'abdquickest':
            h_min  = min(self.dh)
            umax   = float(max(v.abs().amax() for v in vel))
            C_step = min(max(umax * self.dt / h_min, 0.1), 0.99)
            return lambda u, c, d, _C=C_step: self._scheme(u, c, d, C=_C)
        return self._scheme

    def _solve_convective(self, *vel, nu_t=None, iteration=0):
        """Forward-Euler advection-diffusion step.

            phi^{n+1} = phi^n + dt * [-div(vel (x) phi) + diff(phi)]

        When *nu_t* is ``None`` (constant viscosity):
            diff = nu * lap(phi)
        When *nu_t* is a tensor (Smagorinsky LES):
            diff = div((nu + nu_t) * grad(phi))   [variable-coeff Laplacian]

        Accepts (u, v) in 2-D or (u, v, w) in 3-D.

        Lazy clone: start with aliases of the input velocity components.
        We replace ``vel_new[i]`` with a real clone only at the END of
        iteration ``i`` (just before mutating it via ``+= rhs``).  During
        iteration i's _flux peak, only the i already-cloned earlier
        components are alive; the not-yet-mutated components are still
        aliases of ``vel`` (the persistent u0/v0/w0) and cost zero extra
        memory.

        Multi-stream: when ``self._use_streams`` is True and device is CUDA,
        each velocity component is processed on a separate CUDA stream.
        The components are mutually independent (all read from the original
        ``vel`` tuple), so concurrent execution is safe.  Trade-off: all
        ``ndim`` rhs tensors are live simultaneously (vs one at a time in the
        sequential path), so peak intermediate memory is ~ndim× higher for
        the adv-diff phase.
        """
        ndim    = self.ndim
        vel_new = list(vel)
        inner   = _inner(ndim)

        # ---- CUDA fused-flux path (T2a) ----
        # One kernel call per (velocity component i, spatial direction d)
        # directly accumulates dt_dh*(F_left - F_right) into rhs without
        # materialising the intermediate F tensor.  Not combined with the
        # multi-stream path: both are forms of CUDA parallelism that are
        # alternatives at this level.
        use_cuda_kernel = (
            self._is_cuda
            and self._scheme_name in _CUDA_SCHEME_IDS
            and not (self._use_streams and ndim > 1)
        )
        if use_cuda_kernel:
            scheme_id = _CUDA_SCHEME_IDS[self._scheme_name]
            if self._scheme_name == 'abdquickest':
                h_min     = min(self.dh)
                umax      = float(max(v.abs().amax() for v in vel))
                C_courant = float(min(max(umax * self.dt / h_min, 0.1), 0.99))
            else:
                C_courant = 0.0
            for i in range(ndim):
                rhs = diffusion.diffuse(
                    vel[i], self.dt, nu=self.nu, nu_t=nu_t,
                    inv_dh2=self._inv_dh2, dh=self.dh,
                )
                for d in range(ndim):
                    fv = _face_vel(vel, i, d, ndim)
                    p  = _field_for_flux(vel[i], d, ndim)
                    torch.ops.lilytorch_kernels.advect_flux_add(
                        fv, p, rhs,
                        float(self._dt_dh[d]), C_courant,
                        scheme_id, d,
                    )
                    del fv, p
                vel_new[i] = vel[i].clone()
                vel_new[i][inner] += rhs
                del rhs
            return tuple(vel_new)

        # ABDQUICKEST has a .amax().item() sync — must happen BEFORE any
        # stream dispatch so all streams see the same C value.
        scheme = self._get_step_scheme(vel)

        # ---- multi-stream path (CUDA only, ndim > 1) ----
        if self._use_streams and self._is_cuda and ndim > 1:
            if self._adv_streams is None:
                self._adv_streams = [
                    torch.cuda.Stream(device=self.device) for _ in range(ndim)
                ]
            cur = torch.cuda.current_stream()
            for i, s in enumerate(self._adv_streams):
                s.wait_stream(cur)          # inherit prior work from main stream
                with torch.cuda.stream(s):
                    rhs = diffusion.diffuse(
                        vel[i], self.dt, nu=self.nu, nu_t=nu_t,
                        inv_dh2=self._inv_dh2, dh=self.dh,
                    )
                    for d in range(ndim):
                        fv = _face_vel(vel, i, d, ndim)
                        p  = _field_for_flux(vel[i], d, ndim)
                        F  = _flux(scheme, fv, p, d)
                        F_diff = (F[_sl(ndim, d, slice(None, -1))]
                                  - F[_sl(ndim, d, slice(1, None))])
                        rhs.add_(F_diff, alpha=float(self._dt_dh[d]))
                        del fv, F, F_diff
                    vel_new[i] = vel[i].clone()
                    vel_new[i][inner] += rhs
            for s in self._adv_streams:
                cur.wait_stream(s)          # main stream waits for all components
            return tuple(vel_new)

        # ---- sequential path (CPU, single-stream, or 1-D) ----
        for i in range(ndim):
            # diffusion increment (fresh, writable tensor)
            rhs = diffusion.diffuse(
                vel[i], self.dt, nu=self.nu, nu_t=nu_t,
                inv_dh2=self._inv_dh2, dh=self.dh,
            )
            # convective fluxes in each direction, accumulated into rhs.
            # (inlined single-component form of advect_momentum so the
            #  clone happens AFTER the d-loop peak — see lazy-clone note.)
            for d in range(ndim):
                fv = _face_vel(vel, i, d, ndim)
                p  = _field_for_flux(vel[i], d, ndim)
                F  = _flux(scheme, fv, p, d)
                F_diff = (F[_sl(ndim, d, slice(None, -1))]
                          - F[_sl(ndim, d, slice(1, None))])
                rhs.add_(F_diff, alpha=float(self._dt_dh[d]))
                del fv, F, F_diff
            # NOW materialise the clone of vel[i] — we mutate it immediately.
            vel_new[i] = vel[i].clone()
            vel_new[i][inner] += rhs
            del rhs  # free this component's rhs before the next i-iteration

        return tuple(vel_new)

    # =================================================================
    # Semi-Lagrangian solve  (Stam 1999, dimension-agnostic)
    # =================================================================
    def _solve_semi_lagrangian(self, *vel, nu_t=None, iteration=0):
        """Unconditionally-stable advection via RK2 back-tracing (midpoint method).

        Uses a two-stage departure: first trace to x - 0.5*dt*u(x) (midpoint),
        then evaluate u at the midpoint to get the full-step departure
        x - dt*u(x_mid).  This is 2nd-order accurate in the Lagrangian path
        (vs. 1st-order for the original Euler back-trace) with the same number
        of field evaluations per component as one full Euler step needs
        (ndim interpolations at current position + ndim at midpoint).
        """
        ndim  = self.ndim
        shape = tuple(self.n)

        # update interpolator data
        for i in range(ndim):
            self._interps[i].F = vel[i]

        vel_new = list(vel)
        half_dt = 0.5 * self.dt
        for i in range(ndim):
            # Stage 1: velocity at current grid position → midpoint departure
            vel_at_i = [
                self._interps[d](*self._flat_coords[i]).clone()
                for d in range(ndim)
            ]
            midpoint = [
                self._flat_coords[i][d] - half_dt * vel_at_i[d]
                for d in range(ndim)
            ]
            # Stage 2: velocity at midpoint → full-step departure
            vel_at_mid = [
                self._interps[d](*midpoint).clone()
                for d in range(ndim)
            ]
            departure = [
                self._flat_coords[i][d] - self.dt * vel_at_mid[d]
                for d in range(ndim)
            ]
            vel_new[i] = self._interps[i](*departure).reshape(shape).clone()

        # explicit diffusion
        inner = _inner(ndim)
        for i in range(ndim):
            vel_new[i][inner] += diffusion.diffuse(
                vel_new[i], self.dt, nu=self.nu, nu_t=nu_t,
                inv_dh2=self._inv_dh2, dh=self.dh,
            )

        return tuple(vel_new)

    # =================================================================
    # Boundary conditions  (dimension-agnostic)
    # =================================================================
    def _build_bc_ops(self):
        """Precompute BC operations as three flat lists (run once at init).

        WaterLily-style MAC-grid BCs.  Component i is staggered in
        direction i (u in x, v in y, w in z).  For each face d, side s:

          * ``bc[d,s] == "N"`` (free-slip / zero-gradient): copy the
            adjacent interior into the ghost, ``base[ghost] = base[adj]``.
          * ``bc[d,s] == "D"`` (Dirichlet wall value ``g``):
              - if ``d == i`` (component normal to the wall — the
                staggered face *is* the wall): direct-write the wall
                value into the wall slot (index 1 on LO, -1 on HI).
                On LO additionally constant-extrapolate the ghost
                (index 0 ← ``g``) so interior stencils that reach
                across see a sensible value.  On HI there is no slot
                beyond the wall — one write suffices.
              - else (component tangential to the wall): reflective
                ghost ``base[ghost] = 2*g - base[adj]`` so the average
                of ghost and first interior equals ``g``.  No interior
                clamp.

        Returns ``(neumann_ops, dirichlet_ops, reflect_ops)``:
          * ``neumann_ops``:   ``(component, dst_idx, src_idx)``
          * ``dirichlet_ops``: ``(component, dst_idx, value)``
          * ``reflect_ops``:   ``(component, dst_idx, src_idx, value)``
            (semantics: ``base[dst] = 2*value - base[src]``)
        """
        ndim = self.ndim
        n_components = len(self._bc_types)
        neumann_ops   = []
        dirichlet_ops = []
        reflect_ops   = []

        def _idx(d, off):
            return tuple(off if k == d else slice(None) for k in range(ndim))

        for i in range(n_components):
            bc_t = self._bc_types[i]
            bc_v = self._bc_values[i]

            for face in range(2 * ndim):
                d    = face // 2
                side = face % 2          # 0 = lo, 1 = hi

                if bc_t[face] == "N":
                    if side == 0:
                        neumann_ops.append((i, _idx(d, 0), _idx(d, 1)))
                    else:
                        neumann_ops.append((i, _idx(d, -1), _idx(d, -2)))
                elif bc_t[face] == "D":
                    value = bc_v[face]
                    if d == i:
                        # Wall-normal staggered: wall sits on the face.
                        wall_off = 1 if side == 0 else -1
                        dirichlet_ops.append((i, _idx(d, wall_off), value))
                        if side == 0:
                            # Constant-extrapolate the ghost beyond the wall.
                            dirichlet_ops.append((i, _idx(d, 0), value))
                    else:
                        # Tangential: reflective ghost write so the wall
                        # value g is enforced at the half-cell midpoint:
                        #   (base[ghost] + base[adj])/2 = g
                        if side == 0:
                            dst_off, src_off = 0, 1
                        else:
                            dst_off, src_off = -1, -2
                        reflect_ops.append(
                            (i, _idx(d, dst_off), _idx(d, src_off), value)
                        )

        return neumann_ops, dirichlet_ops, reflect_ops

    def _pack_bc_descriptors_3d(self):
        """Pack ``_bc_neumann_ops`` / ``_bc_dirichlet_ops`` / ``_bc_reflect_ops``
        into compact int32 / float descriptor tensors for the fused
        ``apply_bcs_2d`` / ``apply_bcs_3d`` CUDA / CPU ops.

        Descriptors:
          * ``neu_desc`` int32 [N_neu, 3] — ``(comp, axis, side)``.
            side 0 → dst=0, src=1.  side 1 → dst=-1, src=-2.
          * ``dir_desc`` int32 [N_dir, 3] — ``(comp, axis, offset)``.
            offset is the *signed* index along ``axis`` (``{0, 1, -1, -2}``);
            kernel runs ``base[offset] = dir_val``.
          * ``ref_desc`` int32 [N_ref, 4] — ``(comp, axis, dst_off, src_off)``.
            kernel runs ``base[dst_off] = 2 * ref_val - base[src_off]``
            (reflective ghost for tangential Dirichlet walls).

        The descriptor format is ndim-agnostic — the only difference
        between 2-D and 3-D is that ``axis`` is restricted to ``{0, 1}``
        in 2-D — so this helper handles both cases.  The legacy
        ``_pack_bc_descriptors_3d`` name is kept for compatibility with
        external callers.
        """
        ndim = self.ndim
        assert ndim in (2, 3), "fused BC packing only supports 2-D / 3-D"

        def _axis_and_index(idx_tuple):
            # idx_tuple is e.g. (0, slice(None), slice(None))
            for axis, x in enumerate(idx_tuple):
                if isinstance(x, int):
                    return axis, x
            raise ValueError(f"unexpected BC index tuple: {idx_tuple}")

        neu_rows = []
        for comp, dst, src in self._bc_neumann_ops:
            axis_d, dst_v = _axis_and_index(dst)
            axis_s, src_v = _axis_and_index(src)
            assert axis_d == axis_s, "Neumann dst/src axes must match"
            # side 0: dst=0, src=1 ; side 1: dst=-1, src=-2
            if dst_v == 0 and src_v == 1:
                side = 0
            elif dst_v == -1 and src_v == -2:
                side = 1
            else:
                raise ValueError(
                    f"unexpected Neumann (dst,src)=({dst_v},{src_v})")
            neu_rows.append((int(comp), int(axis_d), int(side)))

        dir_rows = []
        dir_vals = []
        for comp, dst, val in self._bc_dirichlet_ops:
            axis_d, dst_v = _axis_and_index(dst)
            dir_rows.append((int(comp), int(axis_d), int(dst_v)))
            dir_vals.append(float(val))

        ref_rows = []
        ref_vals = []
        for comp, dst, src, val in self._bc_reflect_ops:
            axis_d, dst_v = _axis_and_index(dst)
            axis_s, src_v = _axis_and_index(src)
            assert axis_d == axis_s, "Reflective dst/src axes must match"
            ref_rows.append((int(comp), int(axis_d), int(dst_v), int(src_v)))
            ref_vals.append(float(val))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        neu_desc = torch.tensor(
            neu_rows if neu_rows else [[0, 0, 0]],
            dtype=torch.int32, device=device,
        )
        if not neu_rows:
            neu_desc = neu_desc[:0].contiguous()

        dir_desc = torch.tensor(
            dir_rows if dir_rows else [[0, 0, 0]],
            dtype=torch.int32, device=device,
        )
        if not dir_rows:
            dir_desc = dir_desc[:0].contiguous()

        ref_desc = torch.tensor(
            ref_rows if ref_rows else [[0, 0, 0, 0]],
            dtype=torch.int32, device=device,
        )
        if not ref_rows:
            ref_desc = ref_desc[:0].contiguous()

        return {
            "neu_desc": neu_desc,
            "dir_desc": dir_desc,
            "dir_vals_py": dir_vals,  # cast to vel dtype on first use
            "ref_desc": ref_desc,
            "ref_vals_py": ref_vals,
        }

    def _build_fused_bc_cache(self, vel):
        """Lazily build the per-call cache for ``apply_bcs_3d``.

        Captures per-component shapes, the max plane dim, and a
        dtype-correct ``dir_val`` tensor.  Stored on
        ``self._bc_fused_3d_cache``; rebuilt automatically if the
        signature (shapes / dtype / device) of ``vel`` changes.
        """
        u, v, w = vel
        sig = (
            tuple(u.shape), tuple(v.shape), tuple(w.shape),
            u.dtype, u.device,
        )
        cache = self._bc_fused_3d_cache
        if cache is not None and cache["sig"] == sig:
            return cache

        device = u.device
        shapes = torch.tensor(
            [list(u.shape), list(v.shape), list(w.shape)],
            dtype=torch.int64, device=device,
        )
        # Separate max extents for the two thread-block grid dimensions.
        # Per-face dim0/dim1 in apply_bcs_3d_kernel:
        #   axis 0 (x-face): dim0 = Ny, dim1 = Nz
        #   axis 1 (y-face): dim0 = Nx, dim1 = Nz
        #   axis 2 (z-face): dim0 = Nx, dim1 = Ny
        # max_dim0 = max over all faces of dim0 = max(Ny, Nx, Nx) = Nx
        # max_dim1 = max over all faces of dim1 = max(Nz, Nz, Ny) = max(Ny, Nz)
        Nx, Ny, Nz = u.shape
        max_dim0 = int(max(Ny, Nx))   # = Nx (always Nx ≥ Ny for normal grids)
        max_dim1 = int(max(Nz, Ny))

        packed = self._bc_fused_3d_packed
        # Move int descriptors to vel's device if they're not already.
        neu_desc = packed["neu_desc"]
        dir_desc = packed["dir_desc"]
        ref_desc = packed["ref_desc"]
        if neu_desc.device != device:
            neu_desc = neu_desc.to(device)
        if dir_desc.device != device:
            dir_desc = dir_desc.to(device)
        if ref_desc.device != device:
            ref_desc = ref_desc.to(device)

        dir_val = torch.tensor(
            packed["dir_vals_py"] if packed["dir_vals_py"] else [0.0],
            dtype=u.dtype, device=device,
        )
        if not packed["dir_vals_py"]:
            dir_val = dir_val[:0].contiguous()

        ref_val = torch.tensor(
            packed["ref_vals_py"] if packed["ref_vals_py"] else [0.0],
            dtype=u.dtype, device=device,
        )
        if not packed["ref_vals_py"]:
            ref_val = ref_val[:0].contiguous()

        cache = {
            "sig": sig,
            "shapes": shapes,
            "neu_desc": neu_desc,
            "dir_desc": dir_desc,
            "dir_val": dir_val,
            "ref_desc": ref_desc,
            "ref_val": ref_val,
            "max_dim0": max_dim0,
            "max_dim1": max_dim1,
        }
        # Persist updated descriptor device too, so future calls skip the move.
        self._bc_fused_3d_packed["neu_desc"] = neu_desc
        self._bc_fused_3d_packed["dir_desc"] = dir_desc
        self._bc_fused_3d_packed["ref_desc"] = ref_desc
        self._bc_fused_3d_cache = cache
        return cache

    def _build_fused_bc_cache_2d(self, vel):
        """Lazily build the per-call cache for ``apply_bcs_2d``.

        2-D analogue of :meth:`_build_fused_bc_cache`: ``shapes`` is
        ``int64 [2, 2]`` (rows for u, v) and ``max_line_dim`` is the
        longest 1-D ghost line across both components.
        """
        u, v = vel
        sig = (
            tuple(u.shape), tuple(v.shape),
            u.dtype, u.device,
        )
        cache = self._bc_fused_2d_cache
        if cache is not None and cache["sig"] == sig:
            return cache

        device = u.device
        shapes = torch.tensor(
            [list(u.shape), list(v.shape)],
            dtype=torch.int64, device=device,
        )
        max_line_dim = int(max(max(u.shape), max(v.shape)))

        packed = self._bc_fused_2d_packed
        neu_desc = packed["neu_desc"]
        dir_desc = packed["dir_desc"]
        ref_desc = packed["ref_desc"]
        if neu_desc.device != device:
            neu_desc = neu_desc.to(device)
        if dir_desc.device != device:
            dir_desc = dir_desc.to(device)
        if ref_desc.device != device:
            ref_desc = ref_desc.to(device)

        dir_val = torch.tensor(
            packed["dir_vals_py"] if packed["dir_vals_py"] else [0.0],
            dtype=u.dtype, device=device,
        )
        if not packed["dir_vals_py"]:
            dir_val = dir_val[:0].contiguous()

        ref_val = torch.tensor(
            packed["ref_vals_py"] if packed["ref_vals_py"] else [0.0],
            dtype=u.dtype, device=device,
        )
        if not packed["ref_vals_py"]:
            ref_val = ref_val[:0].contiguous()

        cache = {
            "sig": sig,
            "shapes": shapes,
            "neu_desc": neu_desc,
            "dir_desc": dir_desc,
            "dir_val": dir_val,
            "ref_desc": ref_desc,
            "ref_val": ref_val,
            "max_line_dim": max_line_dim,
        }
        self._bc_fused_2d_packed["neu_desc"] = neu_desc
        self._bc_fused_2d_packed["dir_desc"] = dir_desc
        self._bc_fused_2d_packed["ref_desc"] = ref_desc
        self._bc_fused_2d_cache = cache
        return cache

    def set_BCs(self, *vel):
        """Apply Dirichlet / Neumann BCs on the ghost layer.

        Face ordering per component::

            (dim0_lo, dim0_hi, dim1_lo, dim1_hi, [dim2_lo, dim2_hi])

        i.e. (west, east, south, north, [bottom, top]) in 3-D.

        With all-CUDA, contiguous, same-floating-dtype velocity tensors
        and a registered fused descriptor pack, dispatches to the
        ``apply_bcs_2d`` / ``apply_bcs_3d`` CUDA op (one kernel launch
        per BC op instead of many small slice copies).  Otherwise falls
        back to the precomputed Python loop.
        """
        if (self.ndim == 3
                and self._bc_fused_3d_packed is not None
                and len(vel) == 3
                and all(t.is_cuda for t in vel)
                and vel[0].dtype == vel[1].dtype == vel[2].dtype
                and vel[0].dtype in (torch.float32, torch.float64)
                and vel[0].is_contiguous()
                and vel[1].is_contiguous()
                and vel[2].is_contiguous()):
            cache = self._build_fused_bc_cache(vel)
            torch.ops.lilytorch_kernels.apply_bcs_3d(
                vel[0], vel[1], vel[2],
                cache["shapes"],
                cache["neu_desc"],
                cache["dir_desc"],
                cache["dir_val"],
                cache["ref_desc"],
                cache["ref_val"],
                cache["max_dim0"],
                cache["max_dim1"],
            )
            return

        if (self.ndim == 2
                and self._bc_fused_2d_packed is not None
                and len(vel) == 2
                and all(t.is_cuda for t in vel)
                and vel[0].dtype == vel[1].dtype
                and vel[0].dtype in (torch.float32, torch.float64)
                and vel[0].is_contiguous()
                and vel[1].is_contiguous()):
            cache = self._build_fused_bc_cache_2d(vel)
            torch.ops.lilytorch_kernels.apply_bcs_2d(
                vel[0], vel[1],
                cache["shapes"],
                cache["neu_desc"],
                cache["dir_desc"],
                cache["dir_val"],
                cache["ref_desc"],
                cache["ref_val"],
                cache["max_line_dim"],
            )
            return

        # Eager fallback: apply ops in stable order
        # Neumann → direct Dirichlet → reflective Dirichlet.
        # Reflective is last because it reads the *current* interior cell
        # and writes the ghost; running it after the Dirichlet pass means
        # any wall-face direct write (staggered-normal case) is already
        # committed and the reflective form uses up-to-date values.
        for comp, dst, src in self._bc_neumann_ops:
            vel[comp][dst] = vel[comp][src]
        for comp, dst, val in self._bc_dirichlet_ops:
            vel[comp][dst] = val
        for comp, dst, src, val in self._bc_reflect_ops:
            vel[comp][dst] = 2 * val - vel[comp][src]
