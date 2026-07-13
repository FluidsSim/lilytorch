"""Dimension-agnostic advection for MAC staggered grids + the AdvDiffSolver.

Split out of the former monolithic ``adv_diff.py`` (since removed).  This
module owns:

* the fused single-source Warp flux kernel :func:`advect_flux_add_warp` — all
  five high-order convective schemes (QUICK, ADBQUICKEST, CUBISTA, van Leer,
  CDS), 2-D and 3-D, f32+f64 — the single production convective path on CPU
  (C++/OpenMP) and CUDA;
* the :class:`AdvDiffSolver` orchestrator (BCs + scheme dispatch +
  semi-Lagrangian), which composes :mod:`lilytorch.src.diffusion` for the
  diffusion term so advection and diffusion stay independently testable and
  ``torch.compile``-able.

Dependency rule: this module imports the leaf kernel modules (``diffusion``,
``interpolation``) but **never** imports ``solver``, ``two_phase`` or
``facade``.  ``two_phase`` reuses the ``_sl`` slicing helper.

Works identically in 2-D ``(x, y)`` and 3-D ``(x, y, z)`` by looping over
spatial dimensions rather than duplicating code per axis -- inspired by
WaterLily.jl.
"""
from __future__ import annotations

import torch
# Warp primitives are imported from the leaf kernel modules (``interpolation``,
# ``diffusion``); this module must never import ``solver``, ``two_phase`` or
# ``facade`` (it sits upstream of them in the import graph).
from lilytorch.src.interpolation import RegularGridInterpolator

from lilytorch.src import diffusion
from lilytorch.src import native

# apply_bcs_{2,3}d + the ApplyBcs{2,3}DGraphRunner CUDA-graph caches (used by
# AdvDiffSolver.set_BCs) are defined in the Warp section at the bottom of this
# module — merged from the former misc_2d.py / misc_3d.py.



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

    The convective fluxes come from the fused :func:`advect_flux_add_warp`
    kernel and the diffusion term from :mod:`lilytorch.src.diffusion`,
    composed here.
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
        # Pre-convert nu to Python float (avoids GPU→CPU .item() sync in
        # diffuse_add_ hot path, which would trigger CUDA error
        # 900 during whole-step wp.ScopedCapture).
        self._nu_float = float(nu)

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
        # Flux schemes all run through the single fused Warp ``advect_flux_add``
        # kernel (CPU + CUDA); the semi-Lagrangian / implicit solves are a
        # separate ``solve`` method.
        # ``graph_capturable``: may the caller record ``solve`` into a
        # ``torch.cuda.CUDAGraph``?  ONLY if every kernel it launches goes to
        # torch's current CUDA stream, i.e. the native extension ops.  The flux
        # schemes still run on Warp (``advect_flux_accumulate_warp``,
        # ``_accumulate_interior_warp`` and the ``diffusion`` Warp helpers),
        # and raw ``wp.launch`` goes to Warp's OWN stream: torch stream capture
        # does not record it, so those kernels EXECUTE during the capture pass
        # and are silently DROPPED from every replay — wrong physics, no error.
        # The semi-Lagrangian path is native end-to-end (``native.sl_advect_*``
        # + ``native.diffuse_add``) and is safe to capture.
        if method in _CUDA_SCHEME_IDS:
            self._scheme_name = method
            self.solve        = self._solve_convective
            self.graph_capturable = False
        elif method in ("semi-lagrangian", "implicit"):
            self._scheme_name = method
            self._init_semi_lagrangian()
            self.solve = self._solve_semi_lagrangian
            self.graph_capturable = True
        else:
            raise ValueError(
                f"Unknown convection method '{method}'. Choose from: "
                f"{sorted(list(_CUDA_SCHEME_IDS.keys()) + ['semi-lagrangian', 'implicit'])}"
            )

        # ---- persistent buffers for convective solve (Phase 2) --------
        # _conv_copy[i]: full-grid copy buffer for vel[i] (flux stencil source)
        # _conv_out[i]:  full-grid output buffer (dst = vel[i] + diff + flux)
        # _diff_copy[i]: full-grid double-buffer for diffuse_add_
        # _rhs_flux[i]:  interior buffer for flux-only accumulation
        self._conv_copy = None
        self._conv_out = None
        self._diff_copy = None
        self._rhs_flux = None

        print(f"Using the {method} method for the adv-diff equation ({self.ndim}D)")

    # -----------------------------------------------------------------
    # Semi-Lagrangian initialisation  (Stam 1999, N-D)
    # -----------------------------------------------------------------
    def _init_semi_lagrangian(self):
        ndim = self.ndim
        stag = [c - h / 2 for c, h in zip(self.coords, self.dh)]

        self._interps     = []
        self._flat_coords = []
        self._sl_axes_1d  = []   # per-comp tuple of 1-D axis tensors
        self._sl_out      = None # persistent fused-kernel output buffers
        self._diff_out    = None # persistent full-grid copy buffers (double-buffer diffusion)

        for i in range(ndim):
            # component-i lives on a grid staggered in dim i only
            grid = tuple(stag[d] if d == i else self.coords[d]
                         for d in range(ndim))
            interp = RegularGridInterpolator(
                grid,
                torch.zeros(tuple(self.n), device=self.device, dtype=self.dtype),
                fill_value=None, method="quadratic",
            )
            self._interps.append(interp)
            self._sl_axes_1d.append(tuple(g.contiguous() for g in grid))

            grids = torch.meshgrid(*grid, indexing="ij")
            self._flat_coords.append(
                [g.flatten().clone().detach() for g in grids]
            )

    # -----------------------------------------------------------------
    # Convective-scheme persistent buffers  (Phase 2 double-buffer pattern)
    # -----------------------------------------------------------------
    def _init_convective_buffers(self, *vel):
        """Ensure persistent buffers exist for each velocity component
        (full-grid, same shape/dtype/device as *vel*).

        Called once on first solve; reallocated transparently if the
        signature changes (grid growth / dtype / device switch).
        """
        ndim = len(vel)
        dev = vel[0].device
        dtype = vel[0].dtype

        def _bufs_ok(bufs):
            return (bufs is not None
                    and len(bufs) == ndim
                    and all(b.shape == v.shape and b.dtype == dtype and b.device == dev
                           for b, v in zip(bufs, vel)))

        if not _bufs_ok(self._conv_copy):
            self._conv_copy = tuple(torch.empty_like(v) for v in vel)
        if not _bufs_ok(self._conv_out):
            self._conv_out = tuple(torch.empty_like(v) for v in vel)
        if not _bufs_ok(self._diff_copy):
            self._diff_copy = tuple(torch.empty_like(v) for v in vel)
        # _rhs_flux is interior-only (Nix x Niy [x Niz]).
        if self._rhs_flux is None or len(self._rhs_flux) != ndim:
            self._rhs_flux = tuple(
                torch.empty([n - 2 for n in v.shape], dtype=dtype, device=dev)
                for v in vel
            )

    # =================================================================
    # Convective-scheme solve  (advection + diffusion, dimension-agnostic)
    # =================================================================

    @property
    def uses_cuda_flux_kernel(self):
        """Whether ``solve`` takes the fused Warp ``advect_flux_add`` path.

        True for every flux scheme (the single production convective path, on
        CPU and CUDA alike); False only for the semi-Lagrangian solve, which is
        a distinct ``solve`` method.  The flux path is a custom op plus
        host-side syncs, so ``torch.compile`` gives no benefit and trips
        dynamo's speculation log — callers use this to skip compiling ``solve``.
        """
        return self._scheme_name in _CUDA_SCHEME_IDS

    def _solve_convective(self, *vel, nu_t=None, nu_eff=None, iteration=0):
        """Forward-Euler advection-diffusion step — pure Warp, zero torch ops.

            phi^{n+1} = phi^n + dt * [-div(vel (x) phi) + diff(phi)]

        When *nu_t* is ``None`` (constant viscosity):
            diff = nu * lap(phi)
        When *nu_t* is a tensor (Smagorinsky LES):
            diff = div((nu + nu_t) * grad(phi))   [variable-coeff Laplacian]

        Accepts (u, v) in 2-D or (u, v, w) in 3-D.

        Uses ``diffuse_add_`` for diffusion (in-place on output) and a
        separate flux-only accumulation, all pure-Warp graph-capturable:
          1. ``conv_out[i] = copy(vel[i])``
          2. ``diffuse_add_(conv_out[i], diff_copy[i], ...)``  (in-place)
          3. ``conv_copy[i] = copy(vel[i])``  (stencil for flux)
          4. ``zero(rhs_flux[i])``
          5. ``advect_flux_accumulate(conv_copy[i], rhs_flux[i], vel, ...)``
          6. ``conv_out[i][inner] += rhs_flux[i]``
        """
        ndim    = self.ndim
        vel_new = list(vel)
        inner   = _inner(ndim)

        scheme_id = _CUDA_SCHEME_IDS[self._scheme_name]
        # ABDQUICKEST uses a fixed Courant number C=0.1 — safe default that
        # avoids the GPU→CPU sync of a live |u|·dt/h, making the flux kernel
        # graph-capturable (no per-step varying scalar parameter).
        C_courant = 0.1 if self._scheme_name == 'abdquickest' else 0.0

        # Normalise effective viscosity: if the caller already passed a
        # pre-computed nu_eff (graph-safe path), use it as-is; otherwise
        # build it from nu + nu_t (eager path) or leave it None (constant).
        if nu_eff is None and nu_t is not None:
            nu_eff = self._nu_float + nu_t

        # Ensure persistent buffers exist (lazy init, realloc on shape change).
        self._init_convective_buffers(*vel)

        for i in range(ndim):
            # Step 1: copy vel[i] → conv_out[i] (final output buffer).
            diffusion._copy_full_grid_eager(vel[i], self._conv_out[i])

            # Step 2: in-place diffusion on output (double-buffer: copy + fused accumulate).
            diffusion.diffuse_add_(
                self._conv_out[i], self._diff_copy[i], self.dt,
                dh=self.dh, nu_eff=nu_eff, nu=self._nu_float,
            )

            # Step 3: copy vel[i] → conv_copy[i] (stencil source for flux).
            diffusion._copy_full_grid_eager(vel[i], self._conv_copy[i])

            # Step 4: zero the flux-only interior buffer.
            diffusion._zero_interior_eager(self._rhs_flux[i])

            # Step 5: advection flux into rhs_flux (interior buffer).
            advect_flux_accumulate_warp(
                self._conv_copy[i], self._rhs_flux[i], vel, i,
                self._dt_dh, C_courant, scheme_id, ndim,
            )

            # Step 6: accumulate flux into output.
            _accumulate_interior_warp(self._conv_out[i], self._rhs_flux[i], ndim)

            vel_new[i] = self._conv_out[i]

        return tuple(vel_new)

    # =================================================================
    # Semi-Lagrangian solve  (Stam 1999, dimension-agnostic)
    # =================================================================
    def _solve_semi_lagrangian(self, *vel, nu_t=None, nu_eff=None, iteration=0):
        """Unconditionally-stable advection via RK2 back-tracing (midpoint method).

        Uses a two-stage departure: first trace to x - 0.5*dt*u(x) (midpoint),
        then evaluate u at the midpoint to get the full-step departure
        x - dt*u(x_mid).  This is 2nd-order accurate in the Lagrangian path
        (vs. 1st-order for the original Euler back-trace) with the same number
        of field evaluations per component as one full Euler step needs
        (ndim interpolations at current position + ndim at midpoint).

        The fused :func:`sl_advect_2d_kernel` / :func:`sl_advect_3d_kernel`
        Warp kernels are the sole production path (CPU + CUDA), one launch
        per solve.  The retired pure-Python interpolator reference is kept
        as a standalone oracle in ``tests/test_advection.py``.

        When *nu_eff* is given (pre-computed ``nu + nu_t``), it is
        forwarded — no torch add inside the call, safe for
        ``wp.ScopedCapture``.
        """
        ndim = self.ndim
        if ndim not in (2, 3):
            raise ValueError(f"SL solve supports 2-D or 3-D, got {ndim}-D")
        if len(vel) != ndim:
            raise ValueError(
                f"SL solve expects {ndim} velocity components, got {len(vel)}")
        return self._solve_semi_lagrangian_warp(*vel, nu_t=nu_t, nu_eff=nu_eff)

    def _solve_semi_lagrangian_warp(self, *vel, nu_t=None, nu_eff=None):
        """Fused semi-Lagrangian solve (2-D or 3-D) — one Warp launch
        (or CUDA-graph replay) does the full RK2 back-trace for all
        staggered components, writing into persistent output buffers
        (pointer-stable ⇒ graph-capturable).  The explicit-diffusion
        pass stays separate (cheap).

        When *nu_eff* is given (pre-computed ``nu + nu_t``), it is
        forwarded directly to :func:`diffuse_add_` — no torch add
        inside the call, safe for ``wp.ScopedCapture``."""
        ndim = self.ndim
        out = self._sl_out
        dev = vel[0].device
        dtype = vel[0].dtype

        # Build / rebuild output buffers if the signature changed.
        if (out is None
                or len(out) != ndim
                or out[0].shape != vel[0].shape
                or out[0].dtype != dtype
                or out[0].device != dev):
            self._sl_out = out = tuple(torch.empty_like(v) for v in vel)
            # Persistent full-grid copy buffers for the double-buffer
            # diffusion accumulate (diffusion.diffuse_add_): each buffer
            # is the same shape as the velocity field.
            self._diff_out = tuple(
                torch.empty_like(v) for v in vel)
            # Move grid axes to the right device/dtype once.
            flat_axes = []
            for comp_axes in self._sl_axes_1d:
                for g in comp_axes:
                    flat_axes.append(g.to(device=dev, dtype=dtype).contiguous())
            self._sl_axes_dev = tuple(flat_axes)

        if ndim == 2:
            u, v = vel
            out_u, out_v = out
            gxu, gyu, gxv, gyv = self._sl_axes_dev
            iu, iv = self._interps
            native.sl_advect_2d(
                u, v, out_u, out_v, gxu, gyu, gxv, gyv,
                iu._bx0, iu._by0, iu._inv_dx, iu._inv_dy,
                iv._bx0, iv._by0, iv._inv_dx, iv._inv_dy,
                self.dt,
            )
            vel_new = [out_u, out_v]
        else:  # ndim == 3
            u, v, w = vel
            out_u, out_v, out_w = out
            (gxu, gyu, gzu, gxv, gyv, gzv, gxw, gyw, gzw) = self._sl_axes_dev
            iu, iv, iw = self._interps
            native.sl_advect_3d(
                u, v, w, out_u, out_v, out_w,
                gxu, gyu, gzu, gxv, gyv, gzv, gxw, gyw, gzw,
                iu._bx0, iu._by0, iu._bz0, iu._inv_dx, iu._inv_dy, iu._inv_dz,
                iv._bx0, iv._by0, iv._bz0, iv._inv_dx, iv._inv_dy, iv._inv_dz,
                iw._bx0, iw._by0, iw._bz0, iw._inv_dx, iw._inv_dy, iw._inv_dz,
                self.dt,
            )
            vel_new = [out_u, out_v, out_w]

        # Normalise effective viscosity (same pattern as _solve_convective).
        if nu_eff is None and nu_t is not None:
            nu_eff = self._nu_float + nu_t

        # Explicit diffusion — native in-place accumulate (no torch ops,
        # graph-capturable).
        for i in range(ndim):
            native.diffuse_add(
                vel_new[i], self._diff_out[i], self.dt,
                dh=self.dh, nu_eff=nu_eff, nu=self._nu_float,
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
            native.apply_bcs_3d(
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
            native.apply_bcs_2d(
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


# =====================================================================
# Warp advection kernels  (merged from the former src/kernels/advection.py)
# ---------------------------------------------------------------------
# The fused high-order flux-add kernel (all 5 schemes, 2-D+3-D, f32+f64) is the
# single production advection path (GPU *and* CPU).
# =====================================================================

from typing import Any

import warp as wp

wp.init()



# ═════════════════════════════════════════════════════════════════════════════
#  HIGH-ORDER LIMITER PORT — faithful single-source replica of the native fused
#  `advect_flux_add` CUDA kernel (src/kernels/csrc/cuda/advection_flux.cu).
#
#  The native op is called once per (velocity component i, spatial direction d):
#      advect_flux_add(fv, p, rhs, dt_dh, C_courant, scheme_id, face_dim)
#  and accumulates IN PLACE:   rhs[i_fd] += dt_dh * (F_left - F_right)
#  where, for the interior cell i_fd in [0, Nfd-3]:
#      F_left  = flux at global face f_L = i_fd
#      F_right = flux at global face f_R = i_fd + 1
#
#  Layout (from _field_for_flux / _face_vel):
#      p  : (Nfd,   Nt1[, Nt2]) — FULL on face_dim, interior on the rest
#      fv : (Nfd-1, Nt1[, Nt2])
#      rhs: (Ni_0, Ni_1[, Ni_2]) in ORIGINAL grid-dim order, C-contiguous
#           → rhs.stride(face_dim) is NOT Nt1*Nt2 unless face_dim is outermost,
#             so the rhs strides are passed SEPARATELY (HANDOFF caveat).
#
#  Dtype-generic (single source): the kernel/scheme funcs are written over a
#  Warp generic float (``Any``); float literals are materialised in the bound
#  type via ``type(x)(literal)`` and ``wp.overload`` registers the float32 AND
#  float64 specialisations.  float64 lands on bit-parity with the native op
#  (AT_DISPATCH_FLOATING_TYPES → scalar_t = double); the codegen for f64 is
#  unchanged from the original concrete kernel, so the existing parity tests stay
#  bit-identical.  float32 is for an f32 solver (the native op then runs
#  scalar_t = float, so f32 parity is at single precision).
#  Flat 1-D array addressing + explicit element strides mirror the native
#  pointer arithmetic (lesson 2/13); the same kernel serves 2-D (Nt2=1,
#  s_t2=0) and 3-D, exactly like the `.cu`.
#
#  Scheme IDs (match _CUDA_SCHEME_IDS in advection.py and the .cu enum):
#      0 = QUICK   1 = ABDQUICKEST   2 = vanLeer   3 = CDS   4 = CUBISTA
# ═════════════════════════════════════════════════════════════════════════════

_AD_TINY_F64 = 1e-30


@wp.func
def _median3(a: Any, b: Any, c: Any):
    # max(min(a,b), min(max(a,b), c)) — native median3
    return wp.max(wp.min(a, b), wp.min(wp.max(a, b), c))


# All five scheme funcs share the uniform signature (u, c, d, C) — C is the
# Courant number, used only by ABDQUICKEST and ignored by the rest.  The uniform
# signature lets a single kernel template (`_make_flux_kernel`) close over any
# one of them, giving COMPILE-TIME scheme specialization (one kernel per scheme,
# the scheme call fully inlined) — the Warp analogue of the native
# `template <int scheme_id>` dispatch.  A runtime `if scheme_id==…` branch keeps
# all five code paths live in one kernel → ~1.4× slower than native in 3-D;
# specialization closes that to ~1.0× (HANDOFF lesson 17).




@wp.func
def _scheme_quick(u: Any, c: Any, d: Any, C: Any):
    inner = _median3(type(c)(10.0) * c - type(c)(9.0) * u, c, d)
    outer = (type(c)(5.0) * c + type(c)(2.0) * d - u) / type(c)(6.0)
    return _median3(outer, c, inner)


@wp.func
def _scheme_abdquickest(u: Any, c: Any, d: Any, C: Any):
    zero = type(c)(0.0); half = type(c)(0.5)
    one = type(c)(1.0); two = type(c)(2.0); three = type(c)(3.0)
    denom = d - c
    res = c
    if wp.abs(denom) >= type(c)(_AD_TINY_F64):
        rf = (c - u) / denom
        C2 = C * C
        C_upper = two * (one - C)
        scale = (one - C2) / (three - three * C)
        offset = (two + C2 - three * C) / (three - three * C)
        psi = wp.min(rf * scale + offset, C_upper)
        psi = wp.min(psi, rf * C_upper)
        psi = wp.max(psi, zero)
        res = c + half * denom * psi
    return res


@wp.func
def _scheme_van_leer(u: Any, c: Any, d: Any, C: Any):
    denom = d - c
    res = c
    if wp.abs(denom) >= type(c)(_AD_TINY_F64):
        rf = (c - u) / denom
        abs_rf = wp.abs(rf)
        psi = (rf + abs_rf) / (type(c)(1.0) + abs_rf)
        res = c + type(c)(0.5) * denom * psi
    return res


@wp.func
def _scheme_cds(u: Any, c: Any, d: Any, C: Any):
    return type(c)(0.5) * (c + d)


@wp.func
def _scheme_cubista(u: Any, c: Any, d: Any, C: Any):
    zero = type(c)(0.0); half = type(c)(0.5)
    denom = d - c
    res = c
    if wp.abs(denom) >= type(c)(_AD_TINY_F64):
        rf = (c - u) / denom
        psi = wp.min(type(c)(0.75) * rf + type(c)(0.25), type(c)(1.5))
        psi = wp.min(psi, rf * type(c)(1.5))
        psi = wp.max(psi, zero)
        res = c + half * denom * psi
    return res


def _make_flux_kernel(scheme):
    """Build a scheme-SPECIALIZED ``advect_flux_add`` kernel.

    The kernel closes over the ``scheme`` ``@wp.func`` (Warp resolves the closure
    at kernel-creation time and inlines it), so there is no runtime scheme branch
    and the four other scheme code paths are not even present — one compiled
    kernel per scheme, exactly like native's ``template <int scheme_id>``.
    """
    @wp.kernel
    def advect_flux_add_kernel(
        p:   wp.array(dtype=Any),   # flat storage view (full on face_dim)
        fv:  wp.array(dtype=Any),   # flat storage view
        rhs: wp.array(dtype=Any),   # flat storage view, accumulated in place
        Nfd: wp.int32, Nt1: wp.int32, Nt2: wp.int32,
        p_s_fd: wp.int32,   p_s_t1: wp.int32,   p_s_t2: wp.int32,
        fv_s_fd: wp.int32,  fv_s_t1: wp.int32,  fv_s_t2: wp.int32,
        rhs_s_fd: wp.int32, rhs_s_t1: wp.int32, rhs_s_t2: wp.int32,
        dt_dh: Any, C: Any,
    ):
        gid = wp.tid()
        Ni_fd = Nfd - 2
        NT = Nt1 * Nt2
        if gid >= Ni_fd * NT:
            return

        # flat decode: t2 fastest (matches native threadIdx.x → t2 coalescing)
        i_fd = gid / NT
        rem = gid - i_fd * NT
        i_t1 = rem / Nt2
        i_t2 = rem - i_t1 * Nt2

        tp = i_t1 * p_s_t1 + i_t2 * p_s_t2
        tfv = i_t1 * fv_s_t1 + i_t2 * fv_s_t2

        f_L = i_fd
        f_R = i_fd + 1
        fv_L = fv[tfv + f_L * fv_s_fd]
        fv_R = fv[tfv + f_R * fv_s_fd]
        zero = type(fv_L)(0.0)

        # ---- left face flux ----
        pc = p[tp + f_L * p_s_fd]
        pd = p[tp + (f_L + 1) * p_s_fd]
        F_L = type(pc)(0.0)
        if fv_L > zero:
            if f_L == 0:
                # lo boundary: no upstream → CDS fallback
                F_L = fv_L * type(pc)(0.5) * (pc + pd)
            else:
                pu = p[tp + (f_L - 1) * p_s_fd]
                F_L = fv_L * scheme(pu, pc, pd, C)
        else:
            # negative flow: f_L is never the hi boundary (max = Nfd-3 < Nfd-2)
            pdd = p[tp + (f_L + 2) * p_s_fd]
            F_L = fv_L * scheme(pdd, pd, pc, C)

        # ---- right face flux ----
        pc2 = p[tp + f_R * p_s_fd]
        pd2 = p[tp + (f_R + 1) * p_s_fd]
        F_R = type(pc2)(0.0)
        if fv_R > zero:
            # positive flow: f_R is never the lo boundary (min = 1 > 0)
            pu2 = p[tp + (f_R - 1) * p_s_fd]
            F_R = fv_R * scheme(pu2, pc2, pd2, C)
        else:
            if f_R == Nfd - 2:
                # hi boundary: no downstream → CDS fallback
                F_R = fv_R * type(pc2)(0.5) * (pc2 + pd2)
            else:
                pdd2 = p[tp + (f_R + 2) * p_s_fd]
                F_R = fv_R * scheme(pdd2, pd2, pc2, C)

        ridx = i_fd * rhs_s_fd + i_t1 * rhs_s_t1 + i_t2 * rhs_s_t2
        rhs[ridx] = rhs[ridx] + dt_dh * (F_L - F_R)

    return advect_flux_add_kernel


# One compiled kernel per scheme_id (keys match _CUDA_SCHEME_IDS / the .cu enum).
_FLUX_KERNELS = {
    0: _make_flux_kernel(_scheme_quick),        # QUICK
    1: _make_flux_kernel(_scheme_abdquickest),  # ABDQUICKEST
    2: _make_flux_kernel(_scheme_van_leer),     # vanLeer
    3: _make_flux_kernel(_scheme_cds),          # CDS
    4: _make_flux_kernel(_scheme_cubista),      # CUBISTA
}

# Register float32 + float64 specialisations for every scheme kernel.
for _k in _FLUX_KERNELS.values():
    for _dt in (wp.float32, wp.float64):
        _A = wp.array(dtype=_dt)
        wp.overload(_k, {"p": _A, "fv": _A, "rhs": _A, "dt_dh": _dt, "C": _dt})


# ─────────────────────────────────────────────────────────────────────────────
#  Host wrapper: mirror the native op signature, mutate rhs in place.
# ─────────────────────────────────────────────────────────────────────────────

def _wp_device(t: torch.Tensor) -> str:
    return f"cuda:{t.device.index}" if t.is_cuda else "cpu"


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _flat(t: torch.Tensor):
    """Zero-copy flat Warp view (f32/f64) over t's storage, honouring the
    storage offset so element 0 == t's logical [0,0,..] (native pointer base)."""
    assert t.dtype in (torch.float64, torch.float32), "warp advection: f32/f64 only"
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    return wp.array(ptr=t.data_ptr(), dtype=_wp_dtype(t),
                    shape=(int(remaining),), device=_wp_device(t))


# Persistent solver buffers keep a stable (pointer, length, dtype, device) across
# steps, so the zero-copy Warp view built by ``_flat`` can be created once and
# reused instead of rebuilt every iteration.  Keyed on full storage identity: a
# cache hit is only returned when the layout matches exactly, so a pool-reused
# pointer either describes the caller's tensor precisely (the view is correct by
# construction) or misses and rebuilds.  Entries are host-only wrappers over
# externally-owned memory — they never keep a torch block alive.
_FLAT_VIEW_CACHE: dict = {}


def _flat_cached(t: torch.Tensor):
    """Cached :func:`_flat` — reuse the Warp view for a given buffer/layout."""
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    key = (t.data_ptr(), int(remaining), t.dtype, _wp_device(t))
    view = _FLAT_VIEW_CACHE.get(key)
    if view is None:
        view = _flat(t)
        _FLAT_VIEW_CACHE[key] = view
    return view


def advect_flux_add_warp(fv_t, p_t, rhs_t, dt_dh, C_courant, scheme_id, face_dim):
    """Warp port of the retired native ``advect_flux_add`` op.

    Accumulates ``rhs_t[i_fd] += dt_dh * (F_left - F_right)`` in place for one
    (velocity component, spatial direction) pair.  Faithful to the original C++
    wrapper's stride extraction (transverse dims gathered in order, skipping
    face_dim; rhs strides taken in ORIGINAL grid-dim order).
    """
    ndim = p_t.dim()
    assert ndim in (2, 3)
    Nfd = p_t.size(face_dim)
    Ni_fd = Nfd - 2
    if Ni_fd <= 0:
        return

    t_dims = [d for d in range(ndim) if d != face_dim]
    Nt1 = p_t.size(t_dims[0]) if len(t_dims) > 0 else 1
    Nt2 = p_t.size(t_dims[1]) if len(t_dims) > 1 else 1

    p_s_fd = p_t.stride(face_dim)
    p_s_t1 = p_t.stride(t_dims[0]) if len(t_dims) > 0 else 0
    p_s_t2 = p_t.stride(t_dims[1]) if len(t_dims) > 1 else 0
    fv_s_fd = fv_t.stride(face_dim)
    fv_s_t1 = fv_t.stride(t_dims[0]) if len(t_dims) > 0 else 0
    fv_s_t2 = fv_t.stride(t_dims[1]) if len(t_dims) > 1 else 0
    rhs_s_fd = rhs_t.stride(face_dim)
    rhs_s_t1 = rhs_t.stride(t_dims[0]) if len(t_dims) > 0 else 0
    rhs_s_t2 = rhs_t.stride(t_dims[1]) if len(t_dims) > 1 else 0

    kernel = _FLUX_KERNELS[int(scheme_id)]   # compile-time scheme specialization
    dev = _wp_device(p_t)
    wpf = _wp_dtype(p_t)
    n_threads = Ni_fd * Nt1 * Nt2
    wp.launch(
        kernel,
        dim=n_threads,
        inputs=[
            _flat_cached(p_t), _flat_cached(fv_t), _flat_cached(rhs_t),
            int(Nfd), int(Nt1), int(Nt2),
            int(p_s_fd), int(p_s_t1), int(p_s_t2),
            int(fv_s_fd), int(fv_s_t1), int(fv_s_t2),
            int(rhs_s_fd), int(rhs_s_t1), int(rhs_s_t2),
            wpf(float(dt_dh)), wpf(float(C_courant)),
        ],
        device=dev,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Fused per-cell flux-accumulate kernel — one launch per velocity component.
#
#  Replaces the ``_face_vel`` + ``_field_for_flux`` + per-direction
#  ``advect_flux_add_warp`` triple with a SINGLE kernel that reads the
#  stencil directly from a read-only copy buffer (double-buffer pattern,
#  same as ``diffuse_add_``), computes face velocities from the original
#  velocity fields on the fly, and accumulates Σ_d dt_dh_d * (F_L - F_R)
#  directly into the interior output buffer.
#
#  Per-cell (one thread per interior cell), loops over spatial directions
#  internally.  2-D + 3-D unified (Nz=1 for 2-D, z-terms vanish).
#  Scheme-specialized via ``_make_flux_accumulate_kernel`` (compile-time
#  inlining, zero branch overhead) — one compiled kernel per scheme_id.
# ═════════════════════════════════════════════════════════════════════════════

def _make_flux_accumulate_kernel(scheme):
    """Build a scheme-SPECIALIZED per-cell flux-accumulate kernel.

    Each thread handles one interior cell.  It reads the stencil from
    *phi_src* (full-grid copy of ``vel[comp_i]``), computes face velocities
    from the original staggered velocity fields, evaluates the left- and
    right-face fluxes using the inlined *scheme* ``@wp.func``, and
    accumulates ``Σ_d dt_dh_d * (F_L - F_R)`` into ``phi_dst[tid]``.

    Boundary handling mirrors the native ``advect_flux_add`` kernel:
      - Leftmost interior face → CDS fallback for positive flow.
      - Rightmost interior face → CDS fallback for negative flow.
    """
    @wp.kernel
    def kernel(
        phi_src: wp.array(dtype=Any),   # flat full-grid copy of vel[comp_i]
        phi_dst: wp.array(dtype=Any),   # flat interior rhs buffer (accum. in place)
        u_flat: wp.array(dtype=Any),    # vel[0] full grid, flat (for face velocities)
        v_flat: wp.array(dtype=Any),    # vel[1] full grid, flat
        w_flat: wp.array(dtype=Any),    # vel[2] full grid, flat (dummy in 2-D)
        comp_i: int,                    # which component we are solving for
        Nx: int, Ny: int, Nz: int,     # full grid dims (Nz=1 for 2-D)
        Nix: int, Niy: int, Niz: int,  # interior dims
        s_x: int, s_y: int, s_z: int,  # full-grid strides (same for all components)
        dt_dh_0: Any, dt_dh_1: Any, dt_dh_2: Any,
        C: Any,                        # Courant number (ABDQUICKEST only)
    ):
        tid = wp.tid()
        total = Nix * Niy * Niz
        if tid >= total:
            return

        # Decode flat tid → interior (ix, iy, iz); z is fastest-varying.
        iz = tid % Niz
        ixy = tid // Niz
        iy = ixy % Niy
        ix = ixy // Niy

        # Full-grid indices (add 1 for ghost offset).
        gx = ix + 1
        gy = iy + 1
        gz = iz + 1

        # Linear index into full-grid flat arrays at (gx, gy, gz).
        c = gx * s_x + gy * s_y + gz * s_z

        zero = type(phi_src[0])(0.0)
        half = type(phi_src[0])(0.5)
        acc = zero

        # ---- direction 0 (x) ----
        # Stencil for LEFT face (face gx-1, between cells gx-1 and gx).
        #   pu = phi_src[gx-2, gy, gz], pc = phi_src[gx-1, gy, gz], pd = phi_src[gx, gy, gz]
        # Stencil for RIGHT face (face gx, between cells gx and gx+1).
        #   pu2= phi_src[gx-1, gy, gz], pc2= phi_src[gx, gy, gz], pd2= phi_src[gx+1, gy, gz]

        # Face velocities at direction-0 faces.
        #   fv from u_flat.  For self-advection: average u along x (lo=0:-1, hi=1:).
        #   For cross-advection: average u along comp_i at x-face positions
        #   (lo[d]=1:, hi[d]=1: → offset of +1 between fv and p face indices).
        if comp_i == 0:
            # self-advection: fv at midpoint between consecutive x-faces
            fv_L = half * (u_flat[c - s_x] + u_flat[c])
            fv_R = half * (u_flat[c] + u_flat[c + s_x])
        elif comp_i == 1:
            # cross (v advected by u): fv at x-face, averaged along y
            #   fv face index = p face index + 1  (the +1 offset)
            #   left face at x=gx-1 → fv at x=gx
            fv_L = half * (u_flat[c - s_y] + u_flat[c])
            fv_R = half * (u_flat[c + s_x - s_y] + u_flat[c + s_x])
        else:  # comp_i == 2
            # cross (w advected by u): fv at x-face, averaged along z
            fv_L = half * (u_flat[c - s_z] + u_flat[c])
            fv_R = half * (u_flat[c + s_x - s_z] + u_flat[c + s_x])

        # ---- left face (face gx-1) ----
        pc  = phi_src[c - s_x]      # face value at gx-1
        pd  = phi_src[c]            # face value at gx
        F_L = zero
        if fv_L > zero:
            if gx == 1:
                # lo boundary: no upstream → CDS fallback
                F_L = fv_L * half * (pc + pd)
            else:
                pu = phi_src[c - 2 * s_x]
                F_L = fv_L * scheme(pu, pc, pd, C)
        else:
            # negative flow; gx is never the hi boundary for left face
            pdd = phi_src[c + s_x]
            F_L = fv_L * scheme(pdd, pd, pc, C)

        # ---- right face (face gx) ----
        pc2 = phi_src[c]            # face value at gx
        pd2 = phi_src[c + s_x]      # face value at gx+1
        F_R = zero
        if fv_R > zero:
            # positive flow; gx is never the lo boundary for right face
            pu2 = phi_src[c - s_x]
            F_R = fv_R * scheme(pu2, pc2, pd2, C)
        else:
            if gx == Nx - 2:
                # hi boundary: no downstream → CDS fallback
                F_R = fv_R * half * (pc2 + pd2)
            else:
                pdd2 = phi_src[c + 2 * s_x]
                F_R = fv_R * scheme(pdd2, pd2, pc2, C)

        acc = acc + dt_dh_0 * (F_L - F_R)

        # ---- direction 1 (y) ----
        if comp_i == 1:
            # self-advection: fv at midpoint between consecutive y-faces
            fv_L = half * (v_flat[c - s_y] + v_flat[c])
            fv_R = half * (v_flat[c] + v_flat[c + s_y])
        elif comp_i == 0:
            # cross (u advected by v): fv at y-face, averaged along x
            #   fv face index = p face index + 1
            fv_L = half * (v_flat[c - s_x] + v_flat[c])
            fv_R = half * (v_flat[c + s_y - s_x] + v_flat[c + s_y])
        else:  # comp_i == 2
            # cross (w advected by v): fv at y-face, averaged along z
            fv_L = half * (v_flat[c - s_z] + v_flat[c])
            fv_R = half * (v_flat[c + s_y - s_z] + v_flat[c + s_y])

        # ---- left face (face gy-1) ----
        pc  = phi_src[c - s_y]
        pd  = phi_src[c]
        F_L = zero
        if fv_L > zero:
            if gy == 1:
                F_L = fv_L * half * (pc + pd)
            else:
                pu = phi_src[c - 2 * s_y]
                F_L = fv_L * scheme(pu, pc, pd, C)
        else:
            pdd = phi_src[c + s_y]
            F_L = fv_L * scheme(pdd, pd, pc, C)

        # ---- right face (face gy) ----
        pc2 = phi_src[c]
        pd2 = phi_src[c + s_y]
        F_R = zero
        if fv_R > zero:
            pu2 = phi_src[c - s_y]
            F_R = fv_R * scheme(pu2, pc2, pd2, C)
        else:
            if gy == Ny - 2:
                F_R = fv_R * half * (pc2 + pd2)
            else:
                pdd2 = phi_src[c + 2 * s_y]
                F_R = fv_R * scheme(pdd2, pd2, pc2, C)

        acc = acc + dt_dh_1 * (F_L - F_R)

        # ---- direction 2 (z) — only if 3-D ----
        if Nz > 1:
            if comp_i == 2:
                # self-advection: fv at midpoint between consecutive z-faces
                fv_L = half * (w_flat[c - s_z] + w_flat[c])
                fv_R = half * (w_flat[c] + w_flat[c + s_z])
            elif comp_i == 0:
                # cross (u advected by w): fv at z-face, averaged along x
                #   fv face index = p face index + 1
                fv_L = half * (w_flat[c - s_x] + w_flat[c])
                fv_R = half * (w_flat[c + s_z - s_x] + w_flat[c + s_z])
            else:  # comp_i == 1
                # cross (v advected by w): fv at z-face, averaged along y
                fv_L = half * (w_flat[c - s_y] + w_flat[c])
                fv_R = half * (w_flat[c + s_z - s_y] + w_flat[c + s_z])

            # ---- left face (face gz-1) ----
            pc  = phi_src[c - s_z]
            pd  = phi_src[c]
            F_L = zero
            if fv_L > zero:
                if gz == 1:
                    F_L = fv_L * half * (pc + pd)
                else:
                    pu = phi_src[c - 2 * s_z]
                    F_L = fv_L * scheme(pu, pc, pd, C)
            else:
                pdd = phi_src[c + s_z]
                F_L = fv_L * scheme(pdd, pd, pc, C)

            # ---- right face (face gz) ----
            pc2 = phi_src[c]
            pd2 = phi_src[c + s_z]
            F_R = zero
            if fv_R > zero:
                pu2 = phi_src[c - s_z]
                F_R = fv_R * scheme(pu2, pc2, pd2, C)
            else:
                if gz == Nz - 2:
                    F_R = fv_R * half * (pc2 + pd2)
                else:
                    pdd2 = phi_src[c + 2 * s_z]
                    F_R = fv_R * scheme(pdd2, pd2, pc2, C)

            acc = acc + dt_dh_2 * (F_L - F_R)

        # Accumulate total flux contribution into output.
        phi_dst[tid] = phi_dst[tid] + acc

    return kernel


# One compiled kernel per scheme_id.
_FLUX_ACCUMULATE_KERNELS = {
    0: _make_flux_accumulate_kernel(_scheme_quick),        # QUICK
    1: _make_flux_accumulate_kernel(_scheme_abdquickest),  # ABDQUICKEST
    2: _make_flux_accumulate_kernel(_scheme_van_leer),     # vanLeer
    3: _make_flux_accumulate_kernel(_scheme_cds),          # CDS
    4: _make_flux_accumulate_kernel(_scheme_cubista),      # CUBISTA
}

# Register float32 + float64 specialisations for every scheme kernel.
for _k in _FLUX_ACCUMULATE_KERNELS.values():
    for _dt in (wp.float32, wp.float64):
        _A = wp.array(dtype=_dt)
        wp.overload(_k, {
            "phi_src": _A, "phi_dst": _A,
            "u_flat": _A, "v_flat": _A, "w_flat": _A,
            "dt_dh_0": _dt, "dt_dh_1": _dt, "dt_dh_2": _dt,
            "C": _dt,
        })


def advect_flux_accumulate_warp(phi_src_t, phi_dst_t, vel, comp_i,
                                 dt_dh, C_courant, scheme_id, ndim):
    """Eager launch of the fused per-cell flux-accumulate kernel.

    Reads stencil from *phi_src_t* (full-grid copy of ``vel[comp_i]``),
    computes face velocities from *vel* (all components, original time
    step), and accumulates ``Σ_d dt_dh[d] * (F_L - F_R)`` into
    *phi_dst_t* (interior buffer, already holding the diffusion increment).

    One launch replaces the old ``_face_vel`` + ``_field_for_flux`` +
    ``advect_flux_add_warp`` triple.
    """
    Nx, Ny = phi_src_t.shape[0], phi_src_t.shape[1]
    Nz = phi_src_t.shape[2] if ndim == 3 else 1
    Nix, Niy = Nx - 2, Ny - 2
    Niz = Nz - 2 if ndim == 3 else 1
    n_interior = Nix * Niy * Niz
    if n_interior <= 0:
        return

    s_x = int(phi_src_t.stride(0))
    s_y = int(phi_src_t.stride(1))
    s_z = int(phi_src_t.stride(2)) if ndim == 3 else 0

    wpf = _wp_dtype(phi_src_t)
    dev = _wp_device(phi_src_t)

    kernel = _FLUX_ACCUMULATE_KERNELS[int(scheme_id)]

    # Build flat views.  w_flat is a dummy in 2-D.
    u_flat = _flat_cached(vel[0])
    v_flat = _flat_cached(vel[1])
    if ndim == 3:
        w_flat = _flat_cached(vel[2])
    else:
        w_flat = u_flat  # dummy, never read

    wp.launch(
        kernel,
        dim=n_interior,
        inputs=[
            _flat_cached(phi_src_t),
            _flat_cached(phi_dst_t),
            u_flat, v_flat, w_flat,
            int(comp_i),
            Nx, Ny, Nz, Nix, Niy, Niz,
            s_x, s_y, s_z,
            wpf(float(dt_dh[0])), wpf(float(dt_dh[1])),
            wpf(float(dt_dh[2])) if ndim == 3 else wpf(0.0),
            wpf(float(C_courant)),
        ],
        device=dev,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Interior accumulate kernel — pure-Warp ``dst[interior] += src``.
#
#  Replaces ``vel_new[i][inner] += rhs`` (torch slice assignment, illegal
#  inside ``wp.ScopedCapture``).  One launch per component, reads from the
#  compacted interior buffer and accumulates into the full-grid output.
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def accumulate_interior_kernel(
    dst: wp.array(dtype=Any),          # flat full-grid output buffer
    src: wp.array(dtype=Any),          # flat compacted interior buffer
    Nx: int, Ny: int, Nz: int,
    Nix: int, Niy: int, Niz: int,
    s_x: int, s_y: int, s_z: int,
):
    """``dst[interior_cell] += src[tid]`` for each interior cell."""
    tid = wp.tid()
    total = Nix * Niy * Niz
    if tid >= total:
        return
    iz = tid % Niz
    ixy = tid // Niz
    iy = ixy % Niy
    ix = ixy // Niy
    gx = ix + 1
    gy = iy + 1
    gz = iz + 1
    c = gx * s_x + gy * s_y + gz * s_z
    dst[c] = dst[c] + src[tid]


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(accumulate_interior_kernel, {"dst": _A, "src": _A})


def _accumulate_interior_warp(dst_t, src_t, ndim):
    """Eager launch: ``dst_t[interior] += src_t`` (pure Warp, no torch ops)."""
    Nx, Ny = dst_t.shape[0], dst_t.shape[1]
    Nz = dst_t.shape[2] if ndim == 3 else 1
    Nix, Niy = Nx - 2, Ny - 2
    Niz = Nz - 2 if ndim == 3 else 1
    n_interior = Nix * Niy * Niz
    if n_interior <= 0:
        return
    s_x = int(dst_t.stride(0))
    s_y = int(dst_t.stride(1))
    s_z = int(dst_t.stride(2)) if ndim == 3 else 0
    wp.launch(
        accumulate_interior_kernel,
        dim=n_interior,
        inputs=[
            _flat_cached(dst_t), _flat_cached(src_t),
            Nx, Ny, Nz, Nix, Niy, Niz,
            s_x, s_y, s_z,
        ],
        device=_wp_device(dst_t),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Fused semi-Lagrangian advection (2-D) — RK2 back-trace in registers
#
#  Replaces the 10 eager RegularGridInterpolator calls of the python SL path
#  (5 biquadratic samples per velocity component: u,v at the node, u,v at the
#  midpoint, the advected component at the departure point) with ONE Warp
#  launch over both staggered components.  Per-sample math is the shared
#  ``biquadratic_sample_off_2d`` @wp.func — the exact routine the python
#  interpolator kernel calls — so clamping/border-fallback semantics are
#  identical by construction; only the midpoint/departure arithmetic moves
#  from eager torch into registers.
#
#  Node positions are READ from the component's 1-D axis tensors (not
#  recomputed as bx0 + i*dx) so the query points match the python path's
#  ``_flat_coords`` meshgrid bit-for-bit.  Grid scalars (origin, 1/dh) are
#  passed per component because the staggered axes yield subtly different
#  float values for the same nominal spacing.
# ═════════════════════════════════════════════════════════════════════════════

from lilytorch.src.interpolation import (
    biquadratic_sample_off_2d,
    triquadratic_sample_off,
)


@wp.kernel
def sl_advect_2d_kernel(
    u: wp.array(dtype=Any),      # flat (Mx*My) staggered u field, C-contiguous
    v: wp.array(dtype=Any),      # flat (Mx*My) staggered v field
    gxu: wp.array(dtype=Any), gyu: wp.array(dtype=Any),   # u-grid axes
    gxv: wp.array(dtype=Any), gyv: wp.array(dtype=Any),   # v-grid axes
    Mx: int, My: int,
    u_bx0: Any, u_by0: Any, u_idx: Any, u_idy: Any,       # u-grid origin, 1/dh
    v_bx0: Any, v_by0: Any, v_idx: Any, v_idy: Any,       # v-grid origin, 1/dh
    dt: Any,
    out_u: wp.array(dtype=Any),
    out_v: wp.array(dtype=Any),
):
    tid = wp.tid()
    total = Mx * My
    if tid >= 2 * total:
        return
    comp = tid / total           # 0 → u node, 1 → v node
    lin = tid - comp * total
    ixq = lin / My
    iyq = lin - ixq * My

    X = gxu[ixq]
    Y = gyu[iyq]
    if comp == 1:
        X = gxv[ixq]
        Y = gyv[iyq]

    half = type(dt)(0.5)

    # Stage 1: velocity at the node → midpoint x - 0.5*dt*u(x).
    u1 = biquadratic_sample_off_2d(u, 0, Mx, My, u_bx0, u_by0,
                                   u_idx, u_idy, X, Y)
    v1 = biquadratic_sample_off_2d(v, 0, Mx, My, v_bx0, v_by0,
                                   v_idx, v_idy, X, Y)
    xm = X - half * dt * u1
    ym = Y - half * dt * v1

    # Stage 2: velocity at the midpoint → departure x - dt*u(x_mid).
    u2 = biquadratic_sample_off_2d(u, 0, Mx, My, u_bx0, u_by0,
                                   u_idx, u_idy, xm, ym)
    v2 = biquadratic_sample_off_2d(v, 0, Mx, My, v_bx0, v_by0,
                                   v_idx, v_idy, xm, ym)
    xd = X - dt * u2
    yd = Y - dt * v2

    # Sample the advected component at the departure point.
    if comp == 0:
        out_u[lin] = biquadratic_sample_off_2d(u, 0, Mx, My, u_bx0, u_by0,
                                               u_idx, u_idy, xd, yd)
    else:
        out_v[lin] = biquadratic_sample_off_2d(v, 0, Mx, My, v_bx0, v_by0,
                                               v_idx, v_idy, xd, yd)


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(sl_advect_2d_kernel, {
        "u": _A, "v": _A,
        "gxu": _A, "gyu": _A, "gxv": _A, "gyv": _A,
        "u_bx0": _dt, "u_by0": _dt, "u_idx": _dt, "u_idy": _dt,
        "v_bx0": _dt, "v_by0": _dt, "v_idx": _dt, "v_idy": _dt,
        "dt": _dt,
        "out_u": _A, "out_v": _A,
    })


def sl_advect_2d_warp(u_t, v_t, out_u_t, out_v_t, gxu, gyu, gxv, gyv,
                      u_bx0, u_by0, u_idx, u_idy,
                      v_bx0, v_by0, v_idx, v_idy, dt):
    """Eager launch of :func:`sl_advect_2d_kernel` (CPU + CUDA).

    Writes the advected u into ``out_u_t`` and v into ``out_v_t`` (pure
    gather — inputs are never mutated, so the launch is idempotent)."""
    Mx, My = u_t.shape
    wpf = _wp_dtype(u_t)
    wp.launch(
        sl_advect_2d_kernel,
        dim=2 * int(Mx) * int(My),
        inputs=[
            _flat_cached(u_t), _flat_cached(v_t),
            _flat_cached(gxu), _flat_cached(gyu),
            _flat_cached(gxv), _flat_cached(gyv),
            int(Mx), int(My),
            wpf(u_bx0), wpf(u_by0), wpf(u_idx), wpf(u_idy),
            wpf(v_bx0), wpf(v_by0), wpf(v_idx), wpf(v_idy),
            wpf(float(dt)),
            _flat_cached(out_u_t), _flat_cached(out_v_t),
        ],
        device=_wp_device(u_t),
    )



# ═════════════════════════════════════════════════════════════════════════════
#  Fused semi-Lagrangian advection (3-D) — RK2 back-trace in registers
#
#  Same fused RK2 pattern as the 2-D kernel, but over three staggered
#  components (u,v,w) using ``triquadratic_sample_off`` — the exact
#  @wp.func the python interpolator kernel calls — so clamping and
#  border-fallback semantics are identical by construction.
#
#  One Warp launch replaces the 21 eager RegularGridInterpolator calls
#  of the python SL path (7 triquadratic samples per component).
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def sl_advect_3d_kernel(
    u: wp.array(dtype=Any), v: wp.array(dtype=Any), w: wp.array(dtype=Any),
    gxu: wp.array(dtype=Any), gyu: wp.array(dtype=Any), gzu: wp.array(dtype=Any),
    gxv: wp.array(dtype=Any), gyv: wp.array(dtype=Any), gzv: wp.array(dtype=Any),
    gxw: wp.array(dtype=Any), gyw: wp.array(dtype=Any), gzw: wp.array(dtype=Any),
    Mx: int, My: int, Mz: int,
    u_bx0: Any, u_by0: Any, u_bz0: Any, u_idx: Any, u_idy: Any, u_idz: Any,
    v_bx0: Any, v_by0: Any, v_bz0: Any, v_idx: Any, v_idy: Any, v_idz: Any,
    w_bx0: Any, w_by0: Any, w_bz0: Any, w_idx: Any, w_idy: Any, w_idz: Any,
    dt: Any,
    out_u: wp.array(dtype=Any), out_v: wp.array(dtype=Any), out_w: wp.array(dtype=Any),
):
    tid = wp.tid()
    total = Mx * My * Mz
    if tid >= 3 * total:
        return
    comp = tid / total           # 0 → u node, 1 → v node, 2 → w node
    lin = tid - comp * total
    ixy = lin / Mz
    izq = lin - ixy * Mz
    ixq = ixy / My
    iyq = ixy - ixq * My

    X = gxu[ixq]
    Y = gyu[iyq]
    Z = gzu[izq]
    if comp == 1:
        X = gxv[ixq]
        Y = gyv[iyq]
        Z = gzv[izq]
    elif comp == 2:
        X = gxw[ixq]
        Y = gyw[iyq]
        Z = gzw[izq]

    half = type(dt)(0.5)

    # Stage 1: velocity at the node → midpoint x - 0.5*dt*u(x).
    u1 = triquadratic_sample_off(u, 0, Mx, My, Mz, u_bx0, u_by0, u_bz0,
                                 u_idx, u_idy, u_idz, X, Y, Z)
    v1 = triquadratic_sample_off(v, 0, Mx, My, Mz, v_bx0, v_by0, v_bz0,
                                 v_idx, v_idy, v_idz, X, Y, Z)
    w1 = triquadratic_sample_off(w, 0, Mx, My, Mz, w_bx0, w_by0, w_bz0,
                                 w_idx, w_idy, w_idz, X, Y, Z)
    xm = X - half * dt * u1
    ym = Y - half * dt * v1
    zm = Z - half * dt * w1

    # Stage 2: velocity at the midpoint → departure x - dt*u(x_mid).
    u2 = triquadratic_sample_off(u, 0, Mx, My, Mz, u_bx0, u_by0, u_bz0,
                                 u_idx, u_idy, u_idz, xm, ym, zm)
    v2 = triquadratic_sample_off(v, 0, Mx, My, Mz, v_bx0, v_by0, v_bz0,
                                 v_idx, v_idy, v_idz, xm, ym, zm)
    w2 = triquadratic_sample_off(w, 0, Mx, My, Mz, w_bx0, w_by0, w_bz0,
                                 w_idx, w_idy, w_idz, xm, ym, zm)
    xd = X - dt * u2
    yd = Y - dt * v2
    zd = Z - dt * w2

    # Sample the advected component at the departure point.
    if comp == 0:
        out_u[lin] = triquadratic_sample_off(u, 0, Mx, My, Mz, u_bx0, u_by0, u_bz0,
                                             u_idx, u_idy, u_idz, xd, yd, zd)
    elif comp == 1:
        out_v[lin] = triquadratic_sample_off(v, 0, Mx, My, Mz, v_bx0, v_by0, v_bz0,
                                             v_idx, v_idy, v_idz, xd, yd, zd)
    else:
        out_w[lin] = triquadratic_sample_off(w, 0, Mx, My, Mz, w_bx0, w_by0, w_bz0,
                                             w_idx, w_idy, w_idz, xd, yd, zd)


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(sl_advect_3d_kernel, {
        "u": _A, "v": _A, "w": _A,
        "gxu": _A, "gyu": _A, "gzu": _A,
        "gxv": _A, "gyv": _A, "gzv": _A,
        "gxw": _A, "gyw": _A, "gzw": _A,
        "u_bx0": _dt, "u_by0": _dt, "u_bz0": _dt,
        "u_idx": _dt, "u_idy": _dt, "u_idz": _dt,
        "v_bx0": _dt, "v_by0": _dt, "v_bz0": _dt,
        "v_idx": _dt, "v_idy": _dt, "v_idz": _dt,
        "w_bx0": _dt, "w_by0": _dt, "w_bz0": _dt,
        "w_idx": _dt, "w_idy": _dt, "w_idz": _dt,
        "dt": _dt,
        "out_u": _A, "out_v": _A, "out_w": _A,
    })


def sl_advect_3d_warp(u_t, v_t, w_t, out_u_t, out_v_t, out_w_t,
                      gxu, gyu, gzu, gxv, gyv, gzv, gxw, gyw, gzw,
                      u_bx0, u_by0, u_bz0, u_idx, u_idy, u_idz,
                      v_bx0, v_by0, v_bz0, v_idx, v_idy, v_idz,
                      w_bx0, w_by0, w_bz0, w_idx, w_idy, w_idz, dt):
    """Eager launch of :func:`sl_advect_3d_kernel` (CPU + CUDA).

    Writes the advected u,v,w into ``out_u_t``, ``out_v_t``, ``out_w_t``
    (pure gather — inputs are never mutated, so the launch is idempotent)."""
    Mx, My, Mz = u_t.shape
    wpf = _wp_dtype(u_t)
    wp.launch(
        sl_advect_3d_kernel,
        dim=3 * int(Mx) * int(My) * int(Mz),
        inputs=[
            _flat_cached(u_t), _flat_cached(v_t), _flat_cached(w_t),
            _flat_cached(gxu), _flat_cached(gyu), _flat_cached(gzu),
            _flat_cached(gxv), _flat_cached(gyv), _flat_cached(gzv),
            _flat_cached(gxw), _flat_cached(gyw), _flat_cached(gzw),
            int(Mx), int(My), int(Mz),
            wpf(u_bx0), wpf(u_by0), wpf(u_bz0), wpf(u_idx), wpf(u_idy), wpf(u_idz),
            wpf(v_bx0), wpf(v_by0), wpf(v_bz0), wpf(v_idx), wpf(v_idy), wpf(v_idz),
            wpf(w_bx0), wpf(w_by0), wpf(w_bz0), wpf(w_idx), wpf(w_idy), wpf(w_idz),
            wpf(float(dt)),
            _flat_cached(out_u_t), _flat_cached(out_v_t), _flat_cached(out_w_t),
        ],
        device=_wp_device(u_t),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Fused boundary-condition ghost writes — apply_bcs_2d / apply_bcs_3d
#  Merged from the former misc_2d.py / misc_3d.py.  Consumed by
#  AdvDiffSolver.set_BCs above (the ApplyBcs{2,3}DGraphRunner lazies).
#  Dtype-generic f32+f64 via wp.overload.
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def apply_bcs_2d_kernel(
    u: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
    shapes: wp.array(dtype=wp.int64),     # [4] = (uNx,uNy, vNx,vNy)
    neu_desc: wp.array(dtype=wp.int32),   # [N_neu*3]
    N_neu: int,
    dir_desc: wp.array(dtype=wp.int32),   # [N_dir*3]
    dir_val:  wp.array(dtype=Any),
    N_dir: int,
    ref_desc: wp.array(dtype=wp.int32),   # [N_ref*4]
    ref_val:  wp.array(dtype=Any),
    N_ref: int,
):
    op, line = wp.tid()
    total = N_neu + N_dir + N_ref
    if op >= total:
        return

    kind = int(0)
    comp = int(0)
    axis = int(0)
    dst_along = int(0)
    src_along = int(0)
    value = type(u[0])(0.0)

    if op < N_neu:
        kind = 0
        comp = neu_desc[op * 3 + 0]
        axis = neu_desc[op * 3 + 1]
        side = neu_desc[op * 3 + 2]
        sz = int(shapes[comp * 2 + axis])
        if side == 0:
            dst_along = 0; src_along = 1
        else:
            dst_along = sz - 1; src_along = sz - 2
    elif op < N_neu + N_dir:
        d = op - N_neu
        kind = 1
        comp = dir_desc[d * 3 + 0]
        axis = dir_desc[d * 3 + 1]
        offset = dir_desc[d * 3 + 2]
        sz = int(shapes[comp * 2 + axis])
        if offset >= 0:
            dst_along = offset
        else:
            dst_along = sz + offset
        value = dir_val[d]
    else:
        r = op - N_neu - N_dir
        kind = 2
        comp = ref_desc[r * 4 + 0]
        axis = ref_desc[r * 4 + 1]
        dst_off = ref_desc[r * 4 + 2]
        src_off = ref_desc[r * 4 + 3]
        sz = int(shapes[comp * 2 + axis])
        if dst_off >= 0:
            dst_along = dst_off
        else:
            dst_along = sz + dst_off
        if src_off >= 0:
            src_along = src_off
        else:
            src_along = sz + src_off
        value = ref_val[r]

    Nx = int(shapes[comp * 2 + 0])
    Ny = int(shapes[comp * 2 + 1])
    dim0_max = Ny
    if axis != 0:
        dim0_max = Nx
    if line >= dim0_max:
        return

    if axis == 0:
        dst_lin = dst_along * Ny + line
        src_lin = src_along * Ny + line
    else:
        dst_lin = line * Ny + dst_along
        src_lin = line * Ny + src_along

    two = type(u[0])(2.0)
    if comp == 0:
        if kind == 0:
            u[dst_lin] = u[src_lin]
        elif kind == 1:
            u[dst_lin] = value
        else:
            u[dst_lin] = two * value - u[src_lin]
    else:
        if kind == 0:
            v[dst_lin] = v[src_lin]
        elif kind == 1:
            v[dst_lin] = value
        else:
            v[dst_lin] = two * value - v[src_lin]


# Register float32 + float64 specialisations (only the value arrays are generic).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(apply_bcs_2d_kernel,
                {"u": _A, "v": _A, "dir_val": _A, "ref_val": _A})


def _i32(t, wdev):
    if t is None or t.numel() == 0:
        return wp.zeros(1, dtype=wp.int32, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(torch.int32))


def _valf(t, wdev, tdtype):
    """Flat Warp view of a value array cast to the field dtype (f32/f64)."""
    if t is None or t.numel() == 0:
        wpf = wp.float64 if tdtype == torch.float64 else wp.float32
        return wp.zeros(1, dtype=wpf, device=wdev)
    return wp.from_torch(t.reshape(-1).contiguous().to(tdtype))


def apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc, dir_val,
                      ref_desc, ref_val, max_line_dim):
    """Warp port of native ``apply_bcs_2d``; mutates u/v in place.

    Dtype-generic: u/v and the BC value arrays run in the field dtype
    (float32 or float64).
    """
    wdev = "cuda:0" if u.device.type == "cuda" else "cpu"
    N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
    N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
    N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
    if N_neu + N_dir + N_ref == 0 or max_line_dim <= 0:
        return
    uw = _flat_cached(u); vw = _flat_cached(v)
    shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
    neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev); refw = _i32(ref_desc, wdev)
    dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)

    # Stage 1: Neumann + Dirichlet (N_ref=0).
    if N_neu + N_dir > 0:
        wp.launch(apply_bcs_2d_kernel, dim=(N_neu + N_dir, int(max_line_dim)),
                  inputs=[uw, vw, shw, neu, N_neu, dirw, dvw, N_dir,
                          refw, rvw, 0],
                  device=wdev)
    # Stage 2: reflective (N_neu=N_dir=0 so op index maps into ref range).
    if N_ref > 0:
        wp.launch(apply_bcs_2d_kernel, dim=(N_ref, int(max_line_dim)),
                  inputs=[uw, vw, shw, neu, 0, dirw, dvw, 0,
                          refw, rvw, N_ref],
                  device=wdev)


@wp.kernel
def apply_bcs_3d_kernel(
    u: wp.array(dtype=Any), v: wp.array(dtype=Any),
    w: wp.array(dtype=Any),
    shapes: wp.array(dtype=wp.int64),     # [3*3] = (Nx,Ny,Nz) per comp
    neu_desc: wp.array(dtype=wp.int32), N_neu: int,
    dir_desc: wp.array(dtype=wp.int32), dir_val: wp.array(dtype=Any), N_dir: int,
    ref_desc: wp.array(dtype=wp.int32), ref_val: wp.array(dtype=Any), N_ref: int,
):
    op, i, j = wp.tid()
    total = N_neu + N_dir + N_ref
    if op >= total:
        return
    kind = int(0); comp = int(0); axis = int(0)
    dst_along = int(0); src_along = int(0); value = type(u[0])(0.0)
    if op < N_neu:
        kind = 0
        comp = neu_desc[op*3+0]; axis = neu_desc[op*3+1]; side = neu_desc[op*3+2]
        sz = int(shapes[comp*3+axis])
        if side == 0:
            dst_along = 0; src_along = 1
        else:
            dst_along = sz-1; src_along = sz-2
    elif op < N_neu + N_dir:
        d = op - N_neu; kind = 1
        comp = dir_desc[d*3+0]; axis = dir_desc[d*3+1]; offset = dir_desc[d*3+2]
        sz = int(shapes[comp*3+axis])
        if offset >= 0:
            dst_along = offset
        else:
            dst_along = sz + offset
        value = dir_val[d]
    else:
        r = op - N_neu - N_dir; kind = 2
        comp = ref_desc[r*4+0]; axis = ref_desc[r*4+1]
        dst_off = ref_desc[r*4+2]; src_off = ref_desc[r*4+3]
        sz = int(shapes[comp*3+axis])
        if dst_off >= 0:
            dst_along = dst_off
        else:
            dst_along = sz + dst_off
        if src_off >= 0:
            src_along = src_off
        else:
            src_along = sz + src_off
        value = ref_val[r]

    Nx = int(shapes[comp*3+0]); Ny = int(shapes[comp*3+1]); Nz = int(shapes[comp*3+2])
    if axis == 0:
        d0 = Ny; d1 = Nz
    elif axis == 1:
        d0 = Nx; d1 = Nz
    else:
        d0 = Nx; d1 = Ny
    if i >= d0 or j >= d1:
        return
    s1 = Ny * Nz
    s2 = Nz
    if axis == 0:
        dst = dst_along*s1 + i*s2 + j
        src = src_along*s1 + i*s2 + j
    elif axis == 1:
        dst = i*s1 + dst_along*s2 + j
        src = i*s1 + src_along*s2 + j
    else:
        dst = i*s1 + j*s2 + dst_along
        src = i*s1 + j*s2 + src_along

    two = type(u[0])(2.0)
    if comp == 0:
        if kind == 0: u[dst] = u[src]
        elif kind == 1: u[dst] = value
        else: u[dst] = two*value - u[src]
    elif comp == 1:
        if kind == 0: v[dst] = v[src]
        elif kind == 1: v[dst] = value
        else: v[dst] = two*value - v[src]
    else:
        if kind == 0: w[dst] = w[src]
        elif kind == 1: w[dst] = value
        else: w[dst] = two*value - w[src]


# Register float32 + float64 specialisations (only the value arrays are generic).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(apply_bcs_3d_kernel,
                {"u": _A, "v": _A, "w": _A, "dir_val": _A, "ref_val": _A})


def apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc, dir_val,
                      ref_desc, ref_val, max_dim0, max_dim1=None):
    """Warp port of native ``apply_bcs_3d``; mutates u/v/w in place.

    Dtype-generic (f32/f64).  ``max_dim0``/``max_dim1`` are the two face-grid
    extents (native passes both, see ``_build_fused_bc_cache``); a single
    positional arg keeps backward-compat for cubic faces (``max_dim1=max_dim0``).
    Threads outside a face's own ``(d0, d1)`` early-return inside the kernel, so
    launching the per-face max is correct for non-cubic grids.
    """
    wdev = "cuda:0" if u.device.type == "cuda" else "cpu"
    N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
    N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
    N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
    M0 = int(max_dim0)
    M1 = int(max_dim1) if max_dim1 is not None else M0
    if N_neu + N_dir + N_ref == 0 or M0 <= 0 or M1 <= 0:
        return
    uw = _flat_cached(u); vw = _flat_cached(v); ww = _flat_cached(w)
    shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
    neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev); refw = _i32(ref_desc, wdev)
    dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)
    if N_neu + N_dir > 0:
        wp.launch(apply_bcs_3d_kernel, dim=(N_neu + N_dir, M0, M1),
                  inputs=[uw, vw, ww, shw, neu, N_neu, dirw, dvw, N_dir, refw, rvw, 0],
                  device=wdev)
    if N_ref > 0:
        wp.launch(apply_bcs_3d_kernel, dim=(N_ref, M0, M1),
                  inputs=[uw, vw, ww, shw, neu, 0, dirw, dvw, 0, refw, rvw, N_ref],
                  device=wdev)



