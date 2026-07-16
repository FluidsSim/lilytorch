"""Dimension-agnostic advection for MAC staggered grids + the AdvDiffSolver.

Split out of the former monolithic ``adv_diff.py`` (since removed).  This
module owns the :class:`AdvDiffSolver` orchestrator: BCs, scheme dispatch and
semi-Lagrangian transport.

Every kernel it drives is a native CUDA / C++ op from :mod:`lilytorch.src.native`
— ``advect_flux_accumulate`` (all five high-order convective schemes: QUICK,
ADBQUICKEST, CUBISTA, van Leer, CDS; 2-D and 3-D; f32+f64), ``sl_advect_{2,3}d``,
``diffuse_add`` and ``apply_bcs_{2,3}d`` — each with a CPU twin, so the same path
runs on CPU and CUDA.

Dependency rule: this module imports the leaf kernel modules (``native``,
``interpolation``) but **never** imports ``solver``, ``two_phase`` or
``facade``.  ``two_phase`` reuses the ``_sl`` slicing helper.

Works identically in 2-D ``(x, y)`` and 3-D ``(x, y, z)`` by looping over
spatial dimensions rather than duplicating code per axis -- inspired by
WaterLily.jl.
"""
from __future__ import annotations

import torch
# This module must never import ``solver``, ``two_phase`` or ``facade`` (it sits
# upstream of them in the import graph).
from lilytorch.src.interpolation import RegularGridInterpolator

from lilytorch.src import native

# apply_bcs_{2,3}d (used by AdvDiffSolver.set_BCs) is a native CUDA / C++ op
# with a CPU twin; see the descriptor caches built in this
# module — merged from the former misc_2d.py / misc_3d.py.



# Scheme IDs for the fused ``advect_flux_accumulate`` kernel (13c).
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

    The convective fluxes come from the fused ``native.advect_flux_accumulate``
    kernel and the diffusion term from ``native.diffuse_add``,
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
        # 900 during whole-step CUDA-graph capture).
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
        # Flux schemes all run through the single fused native
        # ``advect_flux_accumulate`` kernel (CPU + CUDA); the
        # semi-Lagrangian / implicit solves are a separate ``solve`` method.
        # ``graph_capturable``: may the caller record ``solve`` into a
        # ``torch.cuda.CUDAGraph``?  ONLY if every kernel it launches goes to
        # torch's current CUDA stream, i.e. the native extension ops.  A kernel
        # launched on any other stream is silently DROPPED from every replay.
        # Both paths are native end-to-end (flux: torch ``copy_`` +
        # ``native.diffuse_add`` + ``native.advect_flux_accumulate``;
        # semi-Lagrangian: ``native.sl_advect_*`` + ``native.diffuse_add``),
        # so both are safe to capture.
        if method in _CUDA_SCHEME_IDS:
            self._scheme_name = method
            self.solve        = self._solve_convective
            self.graph_capturable = True
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
        # _diff_copy[i]: full-grid double-buffer for native.diffuse_add
        self._conv_copy = None
        self._conv_out = None
        self._diff_copy = None

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

    # =================================================================
    # Convective-scheme solve  (advection + diffusion, dimension-agnostic)
    # =================================================================

    def _solve_convective(self, *vel, nu_t=None, nu_eff=None, iteration=0):
        """Forward-Euler advection-diffusion step — native end-to-end.

            phi^{n+1} = phi^n + dt * [-div(vel (x) phi) + diff(phi)]

        When *nu_t* is ``None`` (constant viscosity):
            diff = nu * lap(phi)
        When *nu_t* is a tensor (Smagorinsky LES):
            diff = div((nu + nu_t) * grad(phi))   [variable-coeff Laplacian]

        Accepts (u, v) in 2-D or (u, v, w) in 3-D.

        Every launch goes to torch's current CUDA stream, so the whole
        solve is CUDA-graph-capturable (13c):
          1. ``conv_out[i].copy_(vel[i])``
          2. ``native.diffuse_add(conv_out[i], diff_copy[i], ...)`` (in-place)
          3. ``conv_copy[i].copy_(vel[i])``  (stencil for flux)
          4. ``native.advect_flux_accumulate(conv_copy[i], conv_out[i],
             vel, ...)`` — fused flux + interior accumulate
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
            self._conv_out[i].copy_(vel[i])

            # Step 2: in-place diffusion on output (double-buffer: copy + fused accumulate).
            native.diffuse_add(
                self._conv_out[i], self._diff_copy[i], self.dt,
                dh=self.dh, nu_eff=nu_eff, nu=self._nu_float,
            )

            # Step 3: copy vel[i] → conv_copy[i] (stencil source for flux).
            self._conv_copy[i].copy_(vel[i])

            # Step 4: fused advection flux, accumulated straight into the
            # interior of conv_out (no separate rhs buffer / zero pass).
            native.advect_flux_accumulate(
                self._conv_copy[i], self._conv_out[i], vel, i,
                self._dt_dh, C_courant, scheme_id,
            )

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
        Native kernels are the sole production path (CPU + CUDA), one launch
        per solve.  The retired pure-Python interpolator reference is kept
        as a standalone oracle in ``tests/test_advection.py``.

        When *nu_eff* is given (pre-computed ``nu + nu_t``), it is
        forwarded — no torch add inside the call, safe for
        a CUDA graph.
        """
        ndim = self.ndim
        if ndim not in (2, 3):
            raise ValueError(f"SL solve supports 2-D or 3-D, got {ndim}-D")
        if len(vel) != ndim:
            raise ValueError(
                f"SL solve expects {ndim} velocity components, got {len(vel)}")
        return self._solve_semi_lagrangian(*vel, nu_t=nu_t, nu_eff=nu_eff)

    def _solve_semi_lagrangian(self, *vel, nu_t=None, nu_eff=None):
        """Fused semi-Lagrangian solve (2-D or 3-D) — one native launch
        (or CUDA-graph replay) does the full RK2 back-trace for all
        staggered components, writing into persistent output buffers
        (pointer-stable ⇒ graph-capturable).  The explicit-diffusion
        pass stays separate (cheap).

        When *nu_eff* is given (pre-computed ``nu + nu_t``), it is
        forwarded directly to :func:`diffuse_add_` — no torch add
        inside the call, safe for CUDA-graph capture."""
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
            # diffusion accumulate (native.diffuse_add): each buffer
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
