"""
Advection-diffusion solver for MAC staggered grids.

Supports pluggable convective schemes (QUICK, ADBQUICKEST, CUBISTA, van Leer, CDS)
and a semi-Lagrangian (Stam 1999) method.

Inspired by WaterLily.jl's dimension-agnostic design.
"""

import torch
from pytorch_interpolation import RegularGridInterpolator


# =====================================================================
# Convective scheme functions:  lambda(upstream, center, downstream)
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
    return torch.where(
        (c <= torch.minimum(u, d)) | (c >= torch.maximum(u, d)),
        c,
        c + (d - c) * (c - u) / (d - u + 1e-10),
    )


def cds(u, c, d):
    """Central difference scheme -- 2nd-order, not TVD."""
    return 0.5 * (c + d)


def _tvd_face(c, d, psi):
    """Generic TVD face value: c + 0.5*(d - c)*psi(rf)."""
    return c + 0.5 * (d - c) * psi


def abdquickest(u, c, d, C=0.1):
    """ADBQUICKEST scheme -- 3rd-order TVD, Courant-number dependent.

    Uses a flux limiter that depends on the local Courant number *C*.
    The default C=0.1 matches the original lilytorch implementation.
    """
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
# Advection-diffusion solver
# =====================================================================
class AdvDiffSolver:
    """
    Advection-diffusion solver on a MAC staggered grid.

    Supported methods
    -----------------
    * 'quick'                            -- QUICK scheme (default)
    * 'abdquickest'                      -- ADBQUICKEST TVD (Courant-dependent)
    * 'cubista'                          -- CUBISTA TVD limiter
    * 'vanLeer'                          -- van Leer TVD limiter
    * 'cds'                              -- central difference
    * 'semi-lagrangian' / 'implicit'     -- Stam 1999

    For quick / vanLeer / cds the time integration is forward-Euler:

        u^{n+1} = u^n + dt * [-div(u (x) u) + nu * laplacian(u)]
    """

    # -- construction ----------------------------------------------------
    def __init__(
        self,
        device,
        dt,
        x,
        y,
        nu,
        BC_type_u=("D", "D", "D", "D"),   # west, east, south, north
        BC_values_u=(0, 0, 0, 0),
        BC_type_v=("D", "D", "D", "D"),
        BC_values_v=(0, 0, 0, 0),
        method="quick",
    ):
        """
        Parameters
        ----------
        device : torch.device
        dt     : float -- time step
        x, y   : 1-D tensors -- cell-centre coordinates (incl. ghost cells)
        nu     : float -- kinematic viscosity
        method : str -- convection scheme name
        """
        self.device = device
        self.dtype  = x.dtype

        self.dt  = dt
        dx, dy   = x[1] - x[0], y[1] - y[0]
        self.dx  = float(dx)
        self.dy  = float(dy)
        self.dtdx  = dt / dx
        self.dtdy  = dt / dy
        self.dtdx2 = self.dtdx / dx
        self.dtdy2 = self.dtdy / dy
        self.nu  = nu

        self.x, self.y = x, y
        self.nx, self.ny = len(x), len(y)

        # boundary conditions
        self.BC_type_u   = list(BC_type_u)
        self.BC_values_u = list(BC_values_u)
        self.BC_type_v   = list(BC_type_v)
        self.BC_values_v = list(BC_values_v)

        # ---- method dispatch -------------------------------------------
        _schemes = {
            "quick":        quick,
            "abdquickest":  abdquickest,
            "vanLeer":      van_leer,
            "van_leer":     van_leer,
            "cds":          cds,
            "cubista":      cubista,
        }

        if method in _schemes:
            self._scheme = _schemes[method]
            self.solve   = self._solve_convective
        elif method in ("semi-lagrangian", "implicit"):
            self._init_semi_lagrangian()
            self.solve = self._solve_semi_lagrangian
        else:
            raise ValueError(
                f"Unknown convection method '{method}'. "
                f"Choose from: {sorted(set(list(_schemes.keys()) + ['semi-lagrangian', 'implicit']))}"
            )

        # meshgrid utility (used by the __main__ demo & solver.py)
        x_stag = x - dx / 2
        y_stag = y - dy / 2
        self.X_u,  self.Y_u  = torch.meshgrid(x_stag, y, indexing="ij")
        self.X_v,  self.Y_v  = torch.meshgrid(x, y_stag, indexing="ij")
        self.X_cc, self.Y_cc = torch.meshgrid(x_stag, y_stag, indexing="ij")

        print(f"Using the {method} method for the adv-diff equation")

    # -----------------------------------------------------------------
    # Semi-Lagrangian initialisation (Stam 1999)
    # -----------------------------------------------------------------
    def _init_semi_lagrangian(self):
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        x_stag = self.x - dx / 2
        y_stag = self.y - dy / 2

        self._gu = RegularGridInterpolator(
            (x_stag, self.y),
            torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype),
            fill_value=None, method=1,
        )
        self._gv = RegularGridInterpolator(
            (self.x, y_stag),
            torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype),
            fill_value=None, method=1,
        )

        X_u, Y_u = torch.meshgrid(x_stag, self.y, indexing="ij")
        X_v, Y_v = torch.meshgrid(self.x, y_stag, indexing="ij")

        self._xflat_xgrid = X_u.flatten().clone().detach()
        self._yflat_xgrid = Y_u.flatten().clone().detach()
        self._xflat_ygrid = X_v.flatten().clone().detach()
        self._yflat_ygrid = Y_v.flatten().clone().detach()

    # =================================================================
    # Core flux computation
    # =================================================================
    def _flux_x(self, fv, p):
        """Scheme-weighted flux at every x-direction face.

        Parameters
        ----------
        fv : (N-1, M) face velocities between adjacent x-cells
        p  : (N, M)   field values (ghost cells along x, inner along y)

        Returns
        -------
        flux : (N-1, M) -- ready to be differenced across cells
        """
        lam = self._scheme

        # interior faces (full 3-point stencil available)
        fv_in = fv[1:-1]
        flux_in = torch.where(
            fv_in > 0,
            fv_in * lam(p[:-3],  p[1:-2], p[2:-1]),
            fv_in * lam(p[3:],   p[2:-1], p[1:-2]),
        )

        # left boundary face -- CDS for positive flow (stencil missing)
        fv_lo = fv[:1]
        flux_lo = torch.where(
            fv_lo > 0,
            fv_lo * 0.5 * (p[:1] + p[1:2]),
            fv_lo * lam(p[2:3], p[1:2], p[:1]),
        )

        # right boundary face -- CDS for negative flow
        fv_hi = fv[-1:]
        flux_hi = torch.where(
            fv_hi > 0,
            fv_hi * lam(p[-3:-2], p[-2:-1], p[-1:]),
            fv_hi * 0.5 * (p[-2:-1] + p[-1:]),
        )

        return torch.cat([flux_lo, flux_in, flux_hi], dim=0)

    def _flux_y(self, fv, p):
        """Same as _flux_x but along the y-axis (dim 1)."""
        lam = self._scheme

        fv_in = fv[:, 1:-1]
        flux_in = torch.where(
            fv_in > 0,
            fv_in * lam(p[:, :-3],  p[:, 1:-2], p[:, 2:-1]),
            fv_in * lam(p[:, 3:],   p[:, 2:-1], p[:, 1:-2]),
        )

        fv_lo = fv[:, :1]
        flux_lo = torch.where(
            fv_lo > 0,
            fv_lo * 0.5 * (p[:, :1] + p[:, 1:2]),
            fv_lo * lam(p[:, 2:3], p[:, 1:2], p[:, :1]),
        )

        fv_hi = fv[:, -1:]
        flux_hi = torch.where(
            fv_hi > 0,
            fv_hi * lam(p[:, -3:-2], p[:, -2:-1], p[:, -1:]),
            fv_hi * 0.5 * (p[:, -2:-1] + p[:, -1:]),
        )

        return torch.cat([flux_lo, flux_in, flux_hi], dim=1)

    # =================================================================
    # Diffusion (Laplacian)
    # =================================================================
    @staticmethod
    def _laplacian(phi, inv_dx2, inv_dy2):
        """Discrete Laplacian at interior cells."""
        return (
            (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) * inv_dx2
            + (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) * inv_dy2
        )

    # =================================================================
    # Convective-scheme solve  (QUICK / van Leer / CDS)
    # =================================================================
    def _solve_convective(self, u, v, iteration=0):
        """
        Forward-Euler step:

            phi^{n+1} = phi^n + dt * [-div(vel (x) phi) + nu * laplacian(phi)]

        where vel = (u, v) and phi is each velocity component.
        """
        u_new = u.clone()
        v_new = v.clone()

        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)

        # ============================================================
        # u-momentum
        # ============================================================
        # x-face velocities for u  (average u at consecutive x-faces)
        fv_x_u = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])          # (nx-1, ny-2)
        # y-face velocities for u  (interpolate v to u-grid in x)
        fv_y_u = 0.5 * (v[:-2, 1:] + v[1:-1, 1:])             # (nx-2, ny-1)

        # convective fluxes
        Fx_u = self._flux_x(fv_x_u, u[:, 1:-1])               # (nx-1, ny-2)
        Fy_u = self._flux_y(fv_y_u, u[1:-1, :])                # (nx-2, ny-1)

        # update interior  (flux_in - flux_out)
        u_new[1:-1, 1:-1] += (
            self.dtdx * (Fx_u[:-1] - Fx_u[1:])
            + self.dtdy * (Fy_u[:, :-1] - Fy_u[:, 1:])
            + self.nu * self.dt * self._laplacian(u, inv_dx2, inv_dy2)
        )

        # ============================================================
        # v-momentum
        # ============================================================
        # x-face velocities for v  (interpolate u to v-grid in y)
        fv_x_v = 0.5 * (u[1:, :-2] + u[1:, 1:-1])             # (nx-1, ny-2)
        # y-face velocities for v  (average v at consecutive y-faces)
        fv_y_v = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])            # (nx-2, ny-1)

        # convective fluxes
        Fx_v = self._flux_x(fv_x_v, v[:, 1:-1])               # (nx-1, ny-2)
        Fy_v = self._flux_y(fv_y_v, v[1:-1, :])                # (nx-2, ny-1)

        v_new[1:-1, 1:-1] += (
            self.dtdx * (Fx_v[:-1] - Fx_v[1:])
            + self.dtdy * (Fy_v[:, :-1] - Fy_v[:, 1:])
            + self.nu * self.dt * self._laplacian(v, inv_dx2, inv_dy2)
        )

        return (u_new, v_new)

    # =================================================================
    # Semi-Lagrangian solve  (Stam 1999)
    # =================================================================
    def _solve_semi_lagrangian(self, u, v, iteration=0):
        """
        Unconditionally-stable advection via back-tracing:

            u^{n+1}(x) = u^n(x - dt * u(x))
        """
        self._gu.F = u
        self._gv.F = v

        # cross-interpolate to get v on x-stagger and u on y-stagger
        v_xstag = self._gv(self._xflat_xgrid, self._yflat_xgrid).clone().detach()
        u_ystag = self._gu(self._xflat_ygrid, self._yflat_ygrid).clone().detach()

        # back-trace and interpolate
        u = self._gu(
            self._xflat_xgrid - u.flatten() * self.dt,
            self._yflat_xgrid - v_xstag * self.dt,
        ).reshape(self.nx, self.ny).clone().detach()

        v = self._gv(
            self._xflat_ygrid - u_ystag * self.dt,
            self._yflat_ygrid - v.flatten() * self.dt,
        ).reshape(self.nx, self.ny).clone().detach()

        # explicit diffusion
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        u[1:-1, 1:-1] += self.nu * self.dt * self._laplacian(u, inv_dx2, inv_dy2)
        v[1:-1, 1:-1] += self.nu * self.dt * self._laplacian(v, inv_dx2, inv_dy2)

        return (u, v)

    # =================================================================
    # Boundary conditions
    # =================================================================
    def set_BCs(self, u, v):
        """Apply Dirichlet / Neumann BCs on the ghost layer."""
        # Neumann default (zero-gradient)
        u[:, 0]  = u[:, 1]
        u[-1, :] = u[-2, :]
        u[0, :]  = u[1, :]
        u[:, -1] = u[:, -2]

        v[:, 0]  = v[:, 1]
        v[-1, :] = v[-2, :]
        v[0, :]  = v[1, :]
        v[:, -1] = v[:, -2]

        # overwrite with Dirichlet where specified
        if self.BC_type_u[0] == "D":
            u[1, :]  = self.BC_values_u[0]
        if self.BC_type_u[1] == "D":
            u[-1, :] = self.BC_values_u[1]
        if self.BC_type_u[2] == "D":
            u[:, 1]  = self.BC_values_u[2]
        if self.BC_type_u[3] == "D":
            u[:, -1] = self.BC_values_u[3]

        if self.BC_type_v[0] == "D":
            v[1, :]  = self.BC_values_v[0]
        if self.BC_type_v[1] == "D":
            v[-1, :] = self.BC_values_v[1]
        if self.BC_type_v[2] == "D":
            v[:, 1]  = self.BC_values_v[2]
        if self.BC_type_v[3] == "D":
            v[:, -1] = self.BC_values_v[3]

    # =================================================================
    # CFL helper
    # =================================================================
    def clf(self, u, v):
        """Adjust dt to satisfy CFL."""
        vel_max = torch.max(torch.max(torch.abs(u)), torch.max(torch.abs(v)))
        self.dt = self.dx / (vel_max + 3 * self.nu)


# =====================================================================
#  Stand-alone demo  --  advecting a square pulse
# =====================================================================
if __name__ == "__main__":

    use_gpu = False

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Using CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

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
        nm = int(N / 2) - int(30 / dy_val)
        np_ = int(N / 2) + int(30 / dy_val)
        u0[nm:np_, nm:np_] = 1
        v0[nm:np_, nm:np_] = 1
        return u0, v0

    u0, v0 = initial_conditions()

    from matplotlib import pyplot
    import time

    methods = ["quick", "vanLeer", "cds", "semi-lagrangian"]
    fig, axes = pyplot.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 5))

    solver_demo = make_solver("quick")
    X, Y = solver_demo.X_cc, solver_demo.Y_cc
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
    pyplot.show()