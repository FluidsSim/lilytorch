"""Dimension-agnostic advection-diffusion solver for MAC staggered grids.

Supports pluggable convective schemes (QUICK, ADBQUICKEST, CUBISTA, van Leer,
CDS) and a semi-Lagrangian (Stam 1999) method.  Works in 2-D and 3-D with a
single code path -- inspired by WaterLily.jl.
"""

import torch
from pytorch_interpolation import RegularGridInterpolatorAutomatic
from lilytorch.src.kernels import _C as _lilytorch_kernels_C  # noqa: F401  -- registers torch.ops.lilytorch_kernels.*


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
    """QUICK scheme -- 3rd-order, median-based (WaterLily default)."""
    return median((5 * c + 2 * d - u) / 6, c, median(10 * c - 9 * u, c, d))


def van_leer(u, c, d):
    """Van Leer flux limiter -- 2nd-order TVD."""
    denom = d - c
    rf = (c - u) / (denom + 1e-30)
    psi = (rf + rf.abs()) / (1.0 + rf.abs())
    return torch.where(denom.abs() < 1e-30, c, _tvd_face(c, d, psi))


def cds(u, c, d):
    """Central difference scheme -- 2nd-order, not TVD."""
    return 0.5 * (c + d)


def _tvd_face(c, d, psi):
    """Generic TVD face value: c + 0.5*(d - c)*psi(rf)."""
    return c + 0.5 * (d - c) * psi


def abdquickest(u, c, d, C=0.1):
    """ADBQUICKEST scheme -- 3rd-order TVD, Courant-number dependent."""
    C2 = C * C
    denom = d - c
    rf = (c - u) / (denom + 1e-30)
    psi = torch.clamp(
        torch.minimum(
            2.0 * rf * (1.0 - C),
            torch.minimum(
                (2.0 + C2 - 3.0 * C + (1.0 - C2) * rf) / (3.0 - 3.0 * C),
                torch.full_like(rf, 2.0 * (1.0 - C)),
            ),
        ),
        min=0.0,
    )
    return torch.where(denom.abs() < 1e-30, c, _tvd_face(c, d, psi))


def cubista(u, c, d):
    """CUBISTA scheme -- 2nd-order TVD (Alves, Oliveira & Pinho, 2003)."""
    denom = d - c
    rf = (c - u) / (denom + 1e-30)
    psi = torch.clamp(
        torch.minimum(
            1.5 * rf,
            torch.minimum(
                0.75 * rf + 0.25,
                torch.full_like(rf, 1.5),
            ),
        ),
        min=0.0,
    )
    return torch.where(denom.abs() < 1e-30, c, _tvd_face(c, d, psi))


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
        self.dt     = dt
        self.nu     = nu

        # ---- dimension-agnostic grid setup ----------------------------
        self.coords = [x, y] if z is None else [x, y, z]
        self.ndim   = len(self.coords)
        self.n      = [len(c) for c in self.coords]
        self.dh     = [float(c[1] - c[0]) for c in self.coords]

        self._dt_dh  = [dt / h for h in self.dh]
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
        self._bc_neumann_ops, self._bc_dirichlet_ops = self._build_bc_ops()

        # ---- packed descriptors for the fused 3-D set_BCs CUDA op ----
        # Built once here (shape-independent); ``shapes`` and ``dir_val``
        # are cached lazily on first set_BCs() call (they need vel
        # tensor info we don't yet have).
        self._bc_fused_3d_packed = None
        self._bc_fused_3d_cache  = None
        if self.ndim == 3:
            self._bc_fused_3d_packed = self._pack_bc_descriptors_3d()

        # ---- method dispatch -----------------------------------------
        _schemes = {
            "quick": quick, "abdquickest": abdquickest,
            "vanLeer": van_leer, "van_leer": van_leer,
            "cds": cds, "cubista": cubista,
        }
        if method in _schemes:
            self._scheme = _schemes[method]
            self.solve   = self._solve_convective
        elif method in ("semi-lagrangian", "implicit"):
            self._init_semi_lagrangian()
            self.solve = self._solve_semi_lagrangian
        else:
            raise ValueError(
                f"Unknown convection method '{method}'. Choose from: "
                f"{sorted(set(list(_schemes.keys()) + ['semi-lagrangian', 'implicit']))}"
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
                fill_value=None, method=1,
            )
            self._interps.append(interp)

            grids = torch.meshgrid(*grid, indexing="ij")
            self._flat_coords.append(
                [g.flatten().clone().detach() for g in grids]
            )

    # =================================================================
    # Flux computation  (dimension-agnostic)
    # =================================================================
    def _flux(self, fv, p, dim):
        """Scheme-weighted flux along dimension *dim*.

        Parameters
        ----------
        fv  : face velocities  (n[dim]-1 on *dim*, interior on others)
        p   : field values      (n[dim]   on *dim*, interior on others)
        """
        lam = self._scheme
        D   = dim
        S   = lambda s: _sl(p.ndim, D, s)

        # interior faces (full 3-point stencil available)
        fv_in = fv[S(slice(1, -1))]
        flux_in = torch.where(
            fv_in > 0,
            fv_in * lam(p[S(slice(None, -3))], p[S(slice(1, -2))], p[S(slice(2, -1))]),
            fv_in * lam(p[S(slice(3, None))],  p[S(slice(2, -1))], p[S(slice(1, -2))]),
        )

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

    # =================================================================
    # Face-velocity construction  (dimension-agnostic)
    # =================================================================
    def _face_vel(self, vel, i, d):
        """Face velocity for the *i*-th momentum eq. along direction *d*.

        d == i  : self-advection — average vel[i] at consecutive d-faces
        d != i  : cross-advection — average vel[d] along dim i to reach
                  vel[i]'s stagger location
        """
        ndim = self.ndim
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

    @staticmethod
    def _field_for_flux(phi, d, ndim):
        """Extract *phi* with interior on all dims except *d* (full)."""
        idx = [slice(1, -1)] * ndim
        idx[d] = slice(None)
        return phi[tuple(idx)]

    # =================================================================
    # Diffusion  (dimension-agnostic Laplacian)
    # =================================================================
    @staticmethod
    def _laplacian(phi, inv_dh2):
        """Discrete Laplacian at interior cells (arbitrary dimension)."""
        ndim  = phi.ndim
        inner = _inner(ndim)
        lap   = torch.zeros_like(phi[inner])
        for d in range(ndim):
            fwd = list(inner); fwd[d] = slice(2, None)
            bwd = list(inner); bwd[d] = slice(None, -2)
            lap += (phi[tuple(fwd)] - 2.0 * phi[inner] + phi[tuple(bwd)]) * inv_dh2[d]
        return lap

    @staticmethod
    def _variable_laplacian(phi, nu_eff, dh):
        """Variable-coefficient Laplacian: div(nu_eff * grad(phi)).

        Uses face-averaged viscosity:
            sum_d  (nu_{i+1/2}(phi_{i+1} - phi_i) - nu_{i-1/2}(phi_i - phi_{i-1})) / dh_d^2

        Parameters
        ----------
        phi     : tensor — the field (interior + ghost).
        nu_eff  : tensor — effective viscosity at cell centres (same shape as phi).
        dh      : list of floats — grid spacing per dimension.
        """
        ndim  = phi.ndim
        inner = _inner(ndim)
        lap   = torch.zeros_like(phi[inner])
        for d in range(ndim):
            fwd = list(inner); fwd[d] = slice(2, None)
            bwd = list(inner); bwd[d] = slice(None, -2)
            # face-averaged viscosity
            nu_fwd = 0.5 * (nu_eff[inner] + nu_eff[tuple(fwd)])
            nu_bwd = 0.5 * (nu_eff[inner] + nu_eff[tuple(bwd)])
            inv_dh2 = 1.0 / (dh[d] * dh[d])
            lap += (nu_fwd * (phi[tuple(fwd)] - phi[inner])
                    - nu_bwd * (phi[inner] - phi[tuple(bwd)])) * inv_dh2
        return lap

    # =================================================================
    # Convective-scheme solve  (dimension-agnostic)
    # =================================================================
    def _solve_convective(self, *vel, nu_t=None, iteration=0):
        """Forward-Euler advection-diffusion step.

            phi^{n+1} = phi^n + dt * [-div(vel (x) phi) + diff(phi)]

        When *nu_t* is ``None`` (constant viscosity):
            diff = nu * lap(phi)
        When *nu_t* is a tensor (Smagorinsky LES):
            diff = div((nu + nu_t) * grad(phi))   [variable-coeff Laplacian]

        Accepts (u, v) in 2-D or (u, v, w) in 3-D.
        """
        ndim    = self.ndim
        vel_new = [v.clone() for v in vel]
        inner   = _inner(ndim)

        for i in range(ndim):
            # diffusion
            if nu_t is None:
                rhs = self.nu * self.dt * self._laplacian(vel[i], self._inv_dh2)
            else:
                nu_eff = self.nu + nu_t
                rhs = self.dt * self._variable_laplacian(vel[i], nu_eff, self.dh)
            # convective fluxes in each direction
            for d in range(ndim):
                fv = self._face_vel(vel, i, d)
                p  = self._field_for_flux(vel[i], d, ndim)
                F  = self._flux(fv, p, d)
                rhs = rhs + self._dt_dh[d] * (
                    F[_sl(ndim, d, slice(None, -1))]
                    - F[_sl(ndim, d, slice(1, None))]
                )
            vel_new[i][inner] += rhs

        return tuple(vel_new)

    # =================================================================
    # Semi-Lagrangian solve  (Stam 1999, dimension-agnostic)
    # =================================================================
    def _solve_semi_lagrangian(self, *vel, nu_t=None, iteration=0):
        """Unconditionally-stable advection via back-tracing."""
        ndim  = self.ndim
        shape = tuple(self.n)

        # update interpolator data
        for i in range(ndim):
            self._interps[i].F = vel[i]

        vel_new = list(vel)
        for i in range(ndim):
            # all velocity components interpolated to component-i's grid
            vel_at_i = [
                self._interps[d](*self._flat_coords[i]).clone().detach()
                for d in range(ndim)
            ]
            # departure points
            departure = [
                self._flat_coords[i][d] - vel_at_i[d] * self.dt
                for d in range(ndim)
            ]
            vel_new[i] = (
                self._interps[i](*departure)
                .reshape(shape).clone().detach()
            )

        # explicit diffusion
        inner = _inner(ndim)
        for i in range(ndim):
            if nu_t is None:
                vel_new[i][inner] += (
                    self.nu * self.dt * self._laplacian(vel_new[i], self._inv_dh2)
                )
            else:
                nu_eff = self.nu + nu_t
                vel_new[i][inner] += (
                    self.dt * self._variable_laplacian(vel_new[i], nu_eff, self.dh)
                )

        return tuple(vel_new)

    # =================================================================
    # Boundary conditions  (dimension-agnostic)
    # =================================================================
    def _build_bc_ops(self):
        """Precompute BC operations as two flat lists (run once at init).

        Reproduces the original two-pass semantics exactly:
          1. Neumann zero-gradient copy on every ghost face
          2. Dirichlet overwrite on specified faces

        Returns ``(neumann_ops, dirichlet_ops)`` where each entry is a
        tuple of precomputed index tuples — no allocations or string
        comparisons needed at runtime.
        """
        ndim = self.ndim
        n_components = len(self._bc_types)
        neumann_ops   = []   # (component, dst_idx, src_idx)
        dirichlet_ops = []   # (component, dst_idx, value)

        for i in range(n_components):
            bc_t = self._bc_types[i]
            bc_v = self._bc_values[i]

            # --- pass 1: Neumann on every face ---
            for d in range(ndim):
                dst_lo = tuple(0  if k == d else slice(None) for k in range(ndim))
                src_lo = tuple(1  if k == d else slice(None) for k in range(ndim))
                neumann_ops.append((i, dst_lo, src_lo))

                dst_hi = tuple(-1 if k == d else slice(None) for k in range(ndim))
                src_hi = tuple(-2 if k == d else slice(None) for k in range(ndim))
                neumann_ops.append((i, dst_hi, src_hi))

            # --- pass 2: Dirichlet overwrite where specified ---
            # Each Dirichlet face zeroes both boundary layers:
            #   lo: interior cell (1) AND ghost cell (0)
            #   hi: ghost cell (-1) AND last interior cell (-2)
            # This is required for non-staggered components at hi walls:
            # e.g. v at east wall — v[-1,:] is outside the domain (ghost),
            # so v[-2,:] (last interior) must also be constrained.
            # For staggered components the extra cell is harmless.
            for face in range(2 * ndim):
                if bc_t[face] == "D":
                    d    = face // 2
                    side = face % 2          # 0 = lo, 1 = hi
                    for offset in ((1, 0) if side == 0 else (-1, -2)):
                        idx = tuple(
                            offset if k == d else slice(None)
                            for k in range(ndim)
                        )
                        dirichlet_ops.append((i, idx, bc_v[face]))

        return neumann_ops, dirichlet_ops

    def _pack_bc_descriptors_3d(self):
        """Pack ``_bc_neumann_ops`` / ``_bc_dirichlet_ops`` into compact
        int32 / float descriptor tensors for the fused
        ``apply_bcs_3d`` CUDA op.

        Each Neumann tuple ``(comp, dst_idx, src_idx)`` is reduced to
        ``(comp, axis, side)`` -- where *axis* is the dimension along
        which the index is a scalar and *side* is 0 (lo: dst=0, src=1)
        or 1 (hi: dst=N-1, src=N-2).

        Each Dirichlet tuple ``(comp, dst_idx, value)`` is reduced to
        ``(comp, axis, offset)`` plus a separate value array, where
        ``offset`` is one of ``{0, 1, -1, -2}``.
        """
        ndim = self.ndim
        assert ndim == 3, "fused BC packing only supports 3-D"

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        neu_desc = torch.tensor(
            neu_rows if neu_rows else [[0, 0, 0]],
            dtype=torch.int32, device=device,
        )
        # Empty rows: we still need a 0-row tensor with 3 columns.
        if not neu_rows:
            neu_desc = neu_desc[:0].contiguous()

        dir_desc = torch.tensor(
            dir_rows if dir_rows else [[0, 0, 0]],
            dtype=torch.int32, device=device,
        )
        if not dir_rows:
            dir_desc = dir_desc[:0].contiguous()

        return {
            "neu_desc": neu_desc,
            "dir_desc": dir_desc,
            "dir_vals_py": dir_vals,  # cast to vel dtype on first use
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
        max_plane_dim = int(max(max(u.shape), max(v.shape), max(w.shape)))

        packed = self._bc_fused_3d_packed
        # Move int descriptors to vel's device if they're not already.
        neu_desc = packed["neu_desc"]
        dir_desc = packed["dir_desc"]
        if neu_desc.device != device:
            neu_desc = neu_desc.to(device)
        if dir_desc.device != device:
            dir_desc = dir_desc.to(device)

        dir_val = torch.tensor(
            packed["dir_vals_py"] if packed["dir_vals_py"] else [0.0],
            dtype=u.dtype, device=device,
        )
        if not packed["dir_vals_py"]:
            dir_val = dir_val[:0].contiguous()

        cache = {
            "sig": sig,
            "shapes": shapes,
            "neu_desc": neu_desc,
            "dir_desc": dir_desc,
            "dir_val": dir_val,
            "max_plane_dim": max_plane_dim,
        }
        # Persist updated descriptor device too, so future calls skip the move.
        self._bc_fused_3d_packed["neu_desc"] = neu_desc
        self._bc_fused_3d_packed["dir_desc"] = dir_desc
        self._bc_fused_3d_cache = cache
        return cache

    def set_BCs(self, *vel):
        """Apply Dirichlet / Neumann BCs on the ghost layer.

        Face ordering per component:
            (dim0_lo, dim0_hi, dim1_lo, dim1_hi, [dim2_lo, dim2_hi])
        i.e. (west, east, south, north, [bottom, top]) in 3-D.

        In 3-D with all-CUDA, contiguous, same-dtype velocity tensors
        and floating dtype, dispatches to the fused ``apply_bcs_3d``
        CUDA op (one kernel launch instead of 18+ small slice copies).
        Otherwise falls back to the precomputed Python loop.
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
                cache["max_plane_dim"],
            )
            return

        for comp, dst, src in self._bc_neumann_ops:
            vel[comp][dst] = vel[comp][src]
        for comp, dst, val in self._bc_dirichlet_ops:
            vel[comp][dst] = val

    # =================================================================
    # CFL helper
    # =================================================================
    def clf(self, *vel, nu_t_max=0.0):
        """Adjust dt to satisfy CFL.

        Parameters
        ----------
        *vel    : velocity component tensors.
        nu_t_max : float — maximum eddy viscosity (0 when Smagorinsky is off).
        """
        vel_max = max(torch.max(torch.abs(v)).item() for v in vel)
        nu_total = float(self.nu) + float(nu_t_max)
        self.dt = min(self.dh) / (vel_max + 3.0 * nu_total)
        self._dt_dh = [self.dt / h for h in self.dh]
        # legacy
        self.dtdx = self._dt_dh[0]
        self.dtdy = self._dt_dh[1]
        self.dtdx2 = self.dtdx / self.dh[0]
        self.dtdy2 = self.dtdy / self.dh[1]
        if self.ndim == 3:
            self.dtdz  = self._dt_dh[2]
            self.dtdz2 = self.dtdz / self.dh[2]


# =====================================================================
#  Stand-alone demo  --  2-D square-pulse advection + 3-D sanity test
# =====================================================================
if __name__ == "__main__":
    import time

    use_gpu = False
    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Using CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    # ---- 2-D demo ----------------------------------------------------
    N  = 2 ** 8
    nt = 130
    nu = 0
    dt = 0.01
    x  = torch.linspace(-60, 180, N, device=device)
    y  = torch.linspace(-60, 180, N, device=device)

    def make_solver(method):
        return AdvDiffSolver(
            device, dt, x, y, nu,
            BC_type_u=["D", "D", "D", "D"],
            BC_values_u=[1, 0, 0, 0],
            BC_type_v=["D", "D", "D", "D"],
            BC_values_v=[0, 0, 0, 0],
            method=method,
        )

    def initial_conditions():
        u0 = torch.zeros((N, N), device=device)
        v0 = torch.zeros((N, N), device=device)
        dy_val = float(y[1] - y[0])
        nm  = int(N / 2) - int(30 / dy_val)
        np_ = int(N / 2) + int(30 / dy_val)
        u0[nm:np_, nm:np_] = 1
        v0[nm:np_, nm:np_] = 1
        return u0, v0

    u0, v0 = initial_conditions()

    from matplotlib import pyplot
    methods = ["quick", "vanLeer", "cds", "semi-lagrangian"]
    fig, axes = pyplot.subplots(1, len(methods) + 1,
                                figsize=(5 * (len(methods) + 1), 5))
    solver_demo = make_solver("quick")
    stag_x = x - float(x[1] - x[0]) / 2
    stag_y = y - float(y[1] - y[0]) / 2
    X, Y = torch.meshgrid(stag_x, stag_y, indexing="ij")
    axes[0].contourf(X.cpu(), Y.cpu(), u0.cpu(), 20)
    axes[0].set_title("Initial")

    for k, meth in enumerate(methods):
        solver = make_solver(meth)
        u, v = u0.clone(), v0.clone()
        t0 = time.time()
        for n in range(nt + 1):
            u, v = solver.solve(u, v)
            solver.set_BCs(u, v)
        elapsed = time.time() - t0
        print(f"{meth:20s} took {elapsed:.3f}s")
        cs = axes[k + 1].contourf(X.cpu(), Y.cpu(), u.cpu(), 20)
        axes[k + 1].set_title(meth)
        fig.colorbar(cs, ax=axes[k + 1])

    pyplot.tight_layout()
    pyplot.savefig("adv_diff_2d_demo.png", dpi=100)
    print("Saved adv_diff_2d_demo.png")

    # ---- 3-D sanity test ---------------------------------------------
    print("\n--- 3-D sanity test ---")
    N3 = 32
    x3 = torch.linspace(-1, 1, N3, device=device)
    y3 = torch.linspace(-1, 1, N3, device=device)
    z3 = torch.linspace(-1, 1, N3, device=device)

    for meth in ["quick", "abdquickest", "cubista", "vanLeer", "cds"]:
        s3 = AdvDiffSolver(
            device, 0.001, x3, y3, 0.01,
            BC_type_u=("D", "D", "N", "N", "N", "N"),
            BC_values_u=(1, 0),
            BC_type_v=("N", "N", "D", "D", "N", "N"),
            BC_values_v=(0, 0, 0, 0),
            method=meth, z=z3,
            BC_type_w=("N", "N", "N", "N", "D", "D"),
            BC_values_w=(0, 0, 0, 0, 0, 0),
        )
        u3 = torch.zeros(N3, N3, N3, device=device)
        v3 = torch.zeros(N3, N3, N3, device=device)
        w3 = torch.zeros(N3, N3, N3, device=device)
        q = N3 // 4
        u3[q:3*q, q:3*q, q:3*q] = 1.0
        t0 = time.time()
        for _ in range(5):
            u3, v3, w3 = s3.solve(u3, v3, w3)
            s3.set_BCs(u3, v3, w3)
        elapsed = time.time() - t0
        ok = not torch.isnan(u3).any()
        print(f"  {meth:15s}: max(u)={u3.max():.4f}  stable={ok}  {elapsed:.3f}s")

    print("Done.")
