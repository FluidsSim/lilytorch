"""Dimension-agnostic FFT-based Poisson solver for unbounded domains.

Solves  nabla^2 phi = f  in free space via Green function convolution
using zero-padded FFT.  Works in 2-D and 3-D with a single code path.

Green function regularisations
------------------------------
* **2-D** -- Hejlesen et al., *Appl. Math. Lett.* 2013,
  8th-order algebraic smoothing.
* **3-D** -- Gaussian (erf) regularisation (2nd order in sigma).

Usage::

    # 2-D  (backward-compatible with old interface)
    ps = PoissonSolverFFT(x, y, bc_type="free")
    phi = ps.solve(rhs)

    # 3-D
    ps = PoissonSolverFFT(x, y, z=z)
    phi = ps.solve(rhs)
"""

import os
import numpy
import torch
from scipy import special


class PoissonSolverFFT:
    """FFT-based Poisson solver for unbounded (free-space) domains."""

    def __init__(
        self,
        x,
        y,
        bc_type="free",
        overwrite=True,
        filename="lilytorch/data/",
        z=None,
    ):
        """
        Parameters
        ----------
        x, y : 1-D tensors -- cell-centre coordinates
        z    : 1-D tensor or None -- cell-centre z-coordinates (3-D mode)
        bc_type  : str -- only ``"free"`` is supported
        overwrite : bool -- recompute even if a cached file exists
        filename  : str -- directory for caching the Green function FFT
        """
        if bc_type != "free":
            raise ValueError(
                f"Only bc_type='free' is supported, got '{bc_type}'"
            )

        self.dtype = x.dtype
        self.dtype_np = {
            torch.float32: numpy.float32,
            torch.float64: numpy.float64,
        }[self.dtype]
        self.device = x.device

        # -- dimension-agnostic grid ------------------------------------
        self.coords = [x, y] if z is None else [x, y, z]
        self.ndim   = len(self.coords)
        self.n      = [len(c) for c in self.coords]
        self.dh     = [float(c[1] - c[0]) for c in self.coords]
        self._coords_np = [c.cpu().numpy() for c in self.coords]

        # -- legacy 2-D accessors (backward-compat with solver.py) -----
        self.x  = self._coords_np[0]
        self.y  = self._coords_np[1]
        self.nx = self.n[0]
        self.ny = self.n[1]
        self.dx = self.dh[0]
        self.dy = self.dh[1]
        if self.ndim == 3:
            self.z  = self._coords_np[2]
            self.nz = self.n[2]
            self.dz = self.dh[2]

        # -- zero-padded work buffer (doubled in each dim) -------------
        shape_2x = tuple(2 * ni for ni in self.n)
        self.U = torch.zeros(shape_2x, dtype=self.dtype, device=self.device)

        # -- cache filename --------------------------------------------
        coords_tag = "_".join(
            f"{float(c[0])}_{float(c[-1])}" for c in self.coords
        )
        dims_tag = "_".join(str(ni) for ni in self.n)
        self.name = f"Gfft_free_{coords_tag}_{dims_tag}"
        if not os.path.exists(filename):
            os.makedirs(filename)
        self.save_filename = os.path.join(filename, self.name + ".pt")

        # -- compute or load Green function FFT ------------------------
        if os.path.exists(self.save_filename) and not overwrite:
            self.Gfft = torch.load(self.save_filename,
                                   map_location=self.device)
        else:
            self.Gfft = self._compute_green_fft()

        print(
            f"PoissonSolverFFT ready  ({self.ndim}D, "
            f"{'x'.join(str(ni) for ni in self.n)})"
        )

    # ==================================================================
    # Green function computation
    # ==================================================================
    def _compute_green_fft(self):
        """Return FFT( -dV * G ) on the doubled grid."""
        # shifted coordinates: origin at domain minimum
        shifts = [c - c.min() for c in self._coords_np]

        # mirrored extension for zero-padded linear convolution
        exts = []
        for s, h in zip(shifts, self.dh):
            exts.append(numpy.concatenate([
                s,
                [-s[-1] - h],
                -s[::-1][:-1],
            ]))

        # N-D distance field
        grids = numpy.meshgrid(*exts, indexing="ij")
        R = numpy.sqrt(sum(g**2 for g in grids))

        # regularised Green function
        eps = 2.0 * max(self.dh)
        if self.ndim == 2:
            G = self._green_2d(R, eps)
        else:
            G = self._green_3d(R, eps)

        # Gfft = FFT( -dV * G )
        dV = float(numpy.prod(self.dh))
        G_t = torch.from_numpy(G.astype(self.dtype_np)).to(self.device)
        Gfft = torch.fft.fftn(-dV * G_t)

        torch.save(Gfft, self.save_filename)
        return Gfft

    # ------------------------------------------------------------------
    def _green_2d(self, R, eps):
        r"""Hejlesen et al. 2013 -- 8th-order algebraic regularisation.

        .. math::
            G_\varepsilon(r) \approx
            -\frac{1}{2\pi}\!\left[\ln r
                + \tfrac12 E_1\!\bigl(\xi^2/2\bigr)
                - W(\xi)\,e^{-\xi^2/2}\right],
            \quad \xi = r/\varepsilon
        """
        xi = R / eps
        old = numpy.seterr(divide="ignore", invalid="ignore")
        logR = numpy.log(R)

        G = -(
            logR
            + 0.5 * special.exp1(xi**2 / 2)
            - (
                  137 / 120
                - 163 / 240 * xi**2
                + 137 / 960 * xi**4
                -   7 / 640 * xi**6
                +   1 / 3840 * xi**8
            ) * numpy.exp(-xi**2 / 2)
        ) / (2.0 * numpy.pi)

        # regularised singularity at the origin
        G[0, 0] = (
            numpy.euler_gamma / 2
            - numpy.log(numpy.sqrt(2) * eps)
            + 137 / 120
        ) / (2.0 * numpy.pi)

        numpy.seterr(**old)
        return G

    # ------------------------------------------------------------------
    def _green_3d(self, R, eps):
        r"""Gaussian-regularised 3-D free-space Green function.

        Returns :math:`-G_{\text{fund}}` (positive) so that the calling
        convention ``FFT(-dV \cdot G_{\text{func}})`` produces the correct
        sign for the convolution.

        .. math::
            G_{\text{func}}(r)
              = \frac{1}{4\pi r}\,\mathrm{erf}\!\!\left(
                  \frac{r}{\sqrt{2}\,\varepsilon}\right)

        Finite singularity:

        .. math::
            G_{\text{func}}(0) = \frac{1}{(2\pi)^{3/2}\,\varepsilon}
        """
        xi = R / (eps * numpy.sqrt(2))
        old = numpy.seterr(divide="ignore", invalid="ignore")

        G = numpy.where(
            R > 0,
            special.erf(xi) / (4.0 * numpy.pi * R),
            0.0,
        )
        G[0, 0, 0] = 1.0 / ((2.0 * numpy.pi) ** 1.5 * eps)

        numpy.seterr(**old)
        return G

    # ==================================================================
    # Solve
    # ==================================================================
    def solve(self, u):
        r"""Solve  :math:`\nabla^2\varphi = u`  in free space.

        Parameters
        ----------
        u : tensor, shape ``(n0, n1)`` or ``(n0, n1, n2)``

        Returns
        -------
        phi : tensor, same shape as *u*
        """
        slc = tuple(slice(ni) for ni in self.n)
        self.U[slc] = u
        return torch.real(
            torch.fft.ifftn(self.Gfft * torch.fft.fftn(self.U))
        )[slc]

    def solve_free_space(self, u):
        """Alias for :meth:`solve` (backward-compat)."""
        return self.solve(u)


# ======================================================================
# Stand-alone demo / test
# ======================================================================
if __name__ == "__main__":
    import time

    dtype  = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ==================================================================
    # 2-D test  -- compact bump function with analytical Laplacian
    # ==================================================================
    print("\n=== 2-D test ===")
    nx, ny = 512, 128
    x = torch.linspace(0, 5, nx, dtype=dtype, device=device)
    y = torch.linspace(-5, 5, ny, dtype=dtype, device=device)

    ps2 = PoissonSolverFFT(x, y, overwrite=True)

    X, Y = torch.meshgrid(x, y, indexing="ij")
    X0, Y0 = 2.0, 2.0
    XX, YY = X - X0, Y - Y0
    RR2 = XX**2 + YY**2
    c = 6.0

    # source = laplacian of exp(-c/(1-r^2))
    f2 = (
        4 * c * torch.exp(-c / (1 - RR2))
        * (c * RR2 + XX**4 + YY**4 + 2 * XX**2 * YY**2 - 1)
        * (-1 + RR2) ** (-4)
    )
    f2[RR2 >= 1] = 0

    phi_exact_2d = torch.exp(-c / (1 - RR2))
    phi_exact_2d[RR2 >= 1] = 0

    t0 = time.time()
    phi_approx_2d = ps2.solve(f2)
    elapsed = time.time() - t0

    diff2 = torch.abs(phi_approx_2d - phi_exact_2d)
    linf2 = diff2.max().item() / phi_exact_2d.abs().max().item()
    l2_2  = torch.sqrt((diff2**2).mean()).item() / phi_exact_2d.abs().max().item()

    print(f"  Solve time: {elapsed:.4f}s")
    print(f"  Relative Linf error: {linf2:.2e}")
    print(f"  Relative L2   error: {l2_2:.2e}")

    # ==================================================================
    # 3-D test  -- Gaussian with analytical Laplacian
    #   phi(r) = exp(-alpha * r^2)
    #   nabla^2 phi = (4*alpha^2*r^2 - 6*alpha) * exp(-alpha*r^2)   [3-D]
    # ==================================================================
    print("\n=== 3-D test ===")
    N3 = 64
    x3 = torch.linspace(-5, 5, N3, dtype=dtype, device=device)
    y3 = torch.linspace(-5, 5, N3, dtype=dtype, device=device)
    z3 = torch.linspace(-5, 5, N3, dtype=dtype, device=device)

    t0 = time.time()
    ps3 = PoissonSolverFFT(x3, y3, z=z3, overwrite=True)
    build_time = time.time() - t0
    print(f"  Green function build: {build_time:.2f}s")

    X3, Y3, Z3 = torch.meshgrid(x3, y3, z3, indexing="ij")
    alpha = 1.0
    R2_3d = X3**2 + Y3**2 + Z3**2

    phi_exact_3d = torch.exp(-alpha * R2_3d)
    f3 = (4.0 * alpha**2 * R2_3d - 6.0 * alpha) * torch.exp(-alpha * R2_3d)

    t0 = time.time()
    phi_approx_3d = ps3.solve(f3)
    elapsed = time.time() - t0

    diff3 = torch.abs(phi_approx_3d - phi_exact_3d)
    linf3 = diff3.max().item() / phi_exact_3d.abs().max().item()
    l2_3  = torch.sqrt((diff3**2).mean()).item() / phi_exact_3d.abs().max().item()

    print(f"  Solve time: {elapsed:.4f}s")
    print(f"  Relative Linf error: {linf3:.2e}")
    print(f"  Relative L2   error: {l2_3:.2e}")

    print("\nDone.")
