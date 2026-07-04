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
from lilytorch.src.interpolation import RegularGridInterpolatorAutomatic

from lilytorch.src import diffusion

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
        if method in _CUDA_SCHEME_IDS:
            self._scheme_name = method
            self.solve        = self._solve_convective
        elif method in ("semi-lagrangian", "implicit"):
            self._scheme_name = method
            self._init_semi_lagrangian()
            self.solve = self._solve_semi_lagrangian
        else:
            raise ValueError(
                f"Unknown convection method '{method}'. Choose from: "
                f"{sorted(list(_CUDA_SCHEME_IDS.keys()) + ['semi-lagrangian', 'implicit'])}"
            )

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

    def _solve_convective(self, *vel, nu_t=None, iteration=0):
        """Forward-Euler advection-diffusion step via the fused Warp flux kernel.

            phi^{n+1} = phi^n + dt * [-div(vel (x) phi) + diff(phi)]

        When *nu_t* is ``None`` (constant viscosity):
            diff = nu * lap(phi)
        When *nu_t* is a tensor (Smagorinsky LES):
            diff = div((nu + nu_t) * grad(phi))   [variable-coeff Laplacian]

        Accepts (u, v) in 2-D or (u, v, w) in 3-D.

        One ``advect_flux_add`` launch per (velocity component i, spatial
        direction d) accumulates ``dt_dh*(F_left - F_right)`` into ``rhs`` in
        place without materialising the intermediate flux tensor.  The same
        single-source Warp kernel runs on CPU (C++/OpenMP) and CUDA.

        Lazy clone: ``vel_new[i]`` becomes a real clone only at the END of
        iteration ``i`` (just before the ``+= rhs`` mutation); the
        not-yet-mutated components stay aliases of the persistent ``vel``
        (u0/v0/w0) and cost zero extra memory.
        """
        ndim    = self.ndim
        vel_new = list(vel)
        inner   = _inner(ndim)

        scheme_id = _CUDA_SCHEME_IDS[self._scheme_name]
        if self._scheme_name == 'abdquickest':
            # ABDQUICKEST's TVD limiter C must equal the advective Courant
            # number |u|·dt/h — one host sync (.amax()), once per step.
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
                advect_flux_add(
                    fv, p, rhs,
                    float(self._dt_dh[d]), C_courant,
                    scheme_id, d,
                )
                del fv, p
            vel_new[i] = vel[i].clone()
            vel_new[i][inner] += rhs
            del rhs
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

    @property
    def _bcs_runner_2d(self):
        """Lazy Warp ``apply_bcs_2d`` graph runner (fused ghost-line BC writes)."""
        r = getattr(self, "_bcs_graph_2d", None)
        if r is None:
            r = ApplyBcs2DGraphRunner()
            self._bcs_graph_2d = r
        return r

    @property
    def _bcs_runner_3d(self):
        """Lazy Warp ``apply_bcs_3d`` graph runner (fused ghost-line BC writes)."""
        r = getattr(self, "_bcs_graph_3d", None)
        if r is None:
            r = ApplyBcs3DGraphRunner()
            self._bcs_graph_3d = r
        return r

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
            self._bcs_runner_3d(
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
            self._bcs_runner_2d(
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
            _flat(p_t), _flat(fv_t), _flat(rhs_t),
            int(Nfd), int(Nt1), int(Nt2),
            int(p_s_fd), int(p_s_t1), int(p_s_t2),
            int(fv_s_fd), int(fv_s_t1), int(fv_s_t2),
            int(rhs_s_fd), int(rhs_s_t1), int(rhs_s_t2),
            wpf(float(dt_dh)), wpf(float(C_courant)),
        ],
        device=dev,
    )


# In-module alias used by the AdvDiffSolver call site above.
advect_flux_add = advect_flux_add_warp


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
    uw = wp.from_torch(u.reshape(-1))
    vw = wp.from_torch(v.reshape(-1))
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


class ApplyBcs2DGraphRunner:
    """CUDA-graph-cached ``apply_bcs_2d``: the eager wrapper's per-call host floor
    (~90 µs: ~17 µs Warp-array wrapping + 2× ~36 µs ``wp.launch`` submission) is
    replaced by a single ``wp.capture_launch`` (~3 µs) once the (u, v, descriptor,
    max_line) pointer signature is stable.  Ghost writes are in-place into the
    persistent velocity fields, so there is **no extra memory** vs native.

    Churn guard: a pointer signature is only captured on its **second** sighting,
    so one-shot tensors (e.g. fresh projection outputs) stay eager and never pay
    the (expensive) capture cost.  CPU and the unstable first-sighting path
    delegate to :func:`apply_bcs_2d_warp` (bit-identical)."""

    def __init__(self):
        self._graphs = {}   # key -> (graph, cached wp arrays)
        self._seen = {}     # key -> sighting count

    def __call__(self, u, v, shapes, neu_desc, dir_desc, dir_val,
                 ref_desc, ref_val, max_line_dim):
        N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
        N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
        N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
        if N_neu + N_dir + N_ref == 0 or max_line_dim <= 0:
            return
        if u.device.type != "cuda":     # CPU: eager (no graph capture)
            return apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc, dir_val,
                                     ref_desc, ref_val, max_line_dim)
        key = (u.data_ptr(), v.data_ptr(), shapes.data_ptr(),
               neu_desc.data_ptr(), dir_desc.data_ptr(), ref_desc.data_ptr(),
               int(max_line_dim), str(u.dtype))
        ent = self._graphs.get(key)
        if ent is None:
            n = self._seen.get(key, 0) + 1
            self._seen[key] = n
            if n < 2:        # first sighting → eager (might be a one-shot ptr)
                return apply_bcs_2d_warp(u, v, shapes, neu_desc, dir_desc,
                                         dir_val, ref_desc, ref_val, max_line_dim)
            wdev = "cuda:0"
            uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1))
            shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
            neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev)
            refw = _i32(ref_desc, wdev)
            dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)
            ml = int(max_line_dim)

            def _launch():
                if N_neu + N_dir > 0:
                    wp.launch(apply_bcs_2d_kernel, dim=(N_neu + N_dir, ml),
                              inputs=[uw, vw, shw, neu, N_neu, dirw, dvw, N_dir,
                                      refw, rvw, 0], device=wdev)
                if N_ref > 0:
                    wp.launch(apply_bcs_2d_kernel, dim=(N_ref, ml),
                              inputs=[uw, vw, shw, neu, 0, dirw, dvw, 0,
                                      refw, rvw, N_ref], device=wdev)

            _launch()  # warm-up / JIT (idempotent ghost writes)
            with wp.ScopedCapture(device=wdev) as cap:
                _launch()
            ent = (cap.graph, (uw, vw, shw, neu, dirw, refw, dvw, rvw))
            self._graphs[key] = ent
        wp.capture_launch(ent[0])


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
    uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1)); ww = wp.from_torch(w.reshape(-1))
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


class ApplyBcs3DGraphRunner:
    """CUDA-graph-cached ``apply_bcs_3d`` — 3-D analogue of
    :class:`ApplyBcs2DGraphRunner`.  In-place ghost
    writes into the persistent u/v/w fields (no extra memory); captured on the
    second sighting of a stable (u, v, w, descriptor, face-dims) signature, eager
    otherwise.  CPU delegates to :func:`apply_bcs_3d_warp`."""

    def __init__(self):
        self._graphs = {}
        self._seen = {}

    def __call__(self, u, v, w, shapes, neu_desc, dir_desc, dir_val,
                 ref_desc, ref_val, max_dim0, max_dim1=None):
        N_neu = int(neu_desc.size(0)) if neu_desc is not None and neu_desc.numel() else 0
        N_dir = int(dir_desc.size(0)) if dir_desc is not None and dir_desc.numel() else 0
        N_ref = int(ref_desc.size(0)) if ref_desc is not None and ref_desc.numel() else 0
        M0 = int(max_dim0)
        M1 = int(max_dim1) if max_dim1 is not None else M0
        if N_neu + N_dir + N_ref == 0 or M0 <= 0 or M1 <= 0:
            return
        if u.device.type != "cuda":
            return apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc, dir_val,
                                     ref_desc, ref_val, M0, M1)
        key = (u.data_ptr(), v.data_ptr(), w.data_ptr(), shapes.data_ptr(),
               neu_desc.data_ptr(), dir_desc.data_ptr(), ref_desc.data_ptr(),
               M0, M1, str(u.dtype))
        ent = self._graphs.get(key)
        if ent is None:
            n = self._seen.get(key, 0) + 1
            self._seen[key] = n
            if n < 2:
                return apply_bcs_3d_warp(u, v, w, shapes, neu_desc, dir_desc,
                                         dir_val, ref_desc, ref_val, M0, M1)
            wdev = "cuda:0"
            uw = wp.from_torch(u.reshape(-1)); vw = wp.from_torch(v.reshape(-1))
            ww = wp.from_torch(w.reshape(-1))
            shw = wp.from_torch(shapes.reshape(-1).contiguous().to(torch.int64))
            neu = _i32(neu_desc, wdev); dirw = _i32(dir_desc, wdev)
            refw = _i32(ref_desc, wdev)
            dvw = _valf(dir_val, wdev, u.dtype); rvw = _valf(ref_val, wdev, u.dtype)

            def _launch():
                if N_neu + N_dir > 0:
                    wp.launch(apply_bcs_3d_kernel, dim=(N_neu + N_dir, M0, M1),
                              inputs=[uw, vw, ww, shw, neu, N_neu, dirw, dvw,
                                      N_dir, refw, rvw, 0], device=wdev)
                if N_ref > 0:
                    wp.launch(apply_bcs_3d_kernel, dim=(N_ref, M0, M1),
                              inputs=[uw, vw, ww, shw, neu, 0, dirw, dvw, 0,
                                      refw, rvw, N_ref], device=wdev)

            _launch()
            with wp.ScopedCapture(device=wdev) as cap:
                _launch()
            ent = (cap.graph, (uw, vw, ww, shw, neu, dirw, refw, dvw, rvw))
            self._graphs[key] = ent
        wp.capture_launch(ent[0])
