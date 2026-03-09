r"""Dimension-agnostic FFT-based Poisson solver.

Two boundary condition types are supported:

* **free-space** (``bc_type="free"``) -- unbounded domain via Green
  function convolution with zero-padded FFT.
* **Neumann** (``bc_type="neumann"``) -- all-Neumann
  (:math:`\partial p / \partial n = 0`) via the Discrete Cosine
  Transform (DCT-II / IDCT-II).  The DCT naturally diagonalises the
  Neumann discrete Laplacian with eigenvalues
  :math:`\lambda_k = (2/h^2)(\cos(\pi k/N) - 1)`.  The solve is
  simply  DCT → divide by eigenvalues → IDCT  on the original
  N-sized grid (no data mirroring required).

Green function regularisations (free-space only)
-------------------------------------------------
* **2-D** -- Hejlesen et al., *Appl. Math. Lett.* 2013,
  8th-order algebraic smoothing.
* **3-D** -- Gaussian (erf) regularisation (2nd order in sigma).

Usage::

    # Free-space (2-D or 3-D)
    ps = PoissonSolverFFT(x, y, bc_type="free")
    phi = ps.solve(rhs)

    # Neumann BCs (2-D or 3-D)
    ps = PoissonSolverFFT(x, y, z=z, bc_type="neumann")
    phi = ps.solve(rhs)
"""

import os
import numpy
import torch
from scipy import special


class PoissonSolverFFT:
    """FFT-based Poisson solver (free-space or Neumann BCs).

    Supported boundary conditions
    -----------------------------
    * ``"free"``    -- unbounded (free-space) via Green function convolution.
    * ``"neumann"`` -- all-Neumann (dp/dn = 0) via DCT with Neumann
      eigenvalues.
    """

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
        bc_type  : ``"free"`` | ``"neumann"``
        overwrite : bool -- recompute even if a cached file exists
        filename  : str -- directory for caching the Green function FFT
        """
        if bc_type not in ("free", "neumann"):
            raise ValueError(
                f"bc_type must be 'free' or 'neumann', got '{bc_type}'"
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

        self.bc_type = bc_type

        # -- mode-specific initialisation ------------------------------
        if self.bc_type == "free":
            # Doubled work buffer for zero-padded free-space convolution
            shape_2x = tuple(2 * ni for ni in self.n)
            self.U = torch.zeros(shape_2x, dtype=self.dtype, device=self.device)
            self._init_free_space(filename, overwrite)
        else:  # neumann
            self._init_neumann()

    # ------------------------------------------------------------------
    def _init_free_space(self, filename, overwrite):
        """Precompute (or load) the Green function FFT for free-space."""
        coords_tag = "_".join(
            f"{float(c[0])}_{float(c[-1])}" for c in self.coords
        )
        dims_tag = "_".join(str(ni) for ni in self.n)
        self.name = f"Gfft_free_{coords_tag}_{dims_tag}"
        if not os.path.exists(filename):
            os.makedirs(filename)
        self.save_filename = os.path.join(filename, self.name + ".pt")

        if os.path.exists(self.save_filename) and not overwrite:
            self.Gfft = torch.load(self.save_filename,
                                   map_location=self.device,
                                   weights_only=True)
        else:
            self.Gfft = self._compute_green_fft()

        print(
            f"PoissonSolverFFT [free-space] ready  ({self.ndim}D, "
            f"{'x'.join(str(ni) for ni in self.n)})"
        )

    # ------------------------------------------------------------------
    def _init_neumann(self):
        r"""Precompute inverse eigenvalues for the Neumann (DCT) solver.

        The DCT-II diagonalises the discrete Laplacian with Neumann BCs
        (cell-centred grid, ghost cell = replicate).  The eigenvalues are

        .. math::
            \lambda_k^{(d)} = \frac{2}{h_d^2}
            \left(\cos\frac{\pi k}{N_d} - 1\right),
            \quad k = 0, 1, \ldots, N_d-1

        These are N-sized (not 2N); no data mirroring is needed.
        """
        shape_n = tuple(self.n)
        eig = torch.zeros(shape_n, dtype=self.dtype, device=self.device)

        for d in range(self.ndim):
            Nd = self.n[d]
            k = torch.arange(Nd, dtype=self.dtype, device=self.device)
            lam_d = (2.0 / self.dh[d] ** 2) * (
                torch.cos(torch.pi * k / Nd) - 1.0
            )
            shape = [1] * self.ndim
            shape[d] = Nd
            eig = eig + lam_d.reshape(shape)

        # Fix the zero mode (pressure determined up to a constant).
        zero_idx = tuple(0 for _ in range(self.ndim))
        eig[zero_idx] = 1.0
        self.inv_eig = 1.0 / eig
        self.inv_eig[zero_idx] = 0.0        # zero mean pressure

        # Pre-compute the DCT/IDCT twiddle factors and permutation
        # indices for each dimension (avoids recomputing every solve).
        self._dct_idx     = []   # forward permutation per dim
        self._idct_idx    = []   # inverse permutation per dim
        self._dct_twiddle = []   # exp(-j*pi*k/(2N))  per dim
        cdtype = (torch.complex128 if self.dtype == torch.float64
                  else torch.complex64)
        for d in range(self.ndim):
            Nd = self.n[d]
            # Forward permutation: even indices, then odd indices reversed
            idx = torch.cat([
                torch.arange(0, Nd, 2, device=self.device),
                torch.arange(1, Nd, 2, device=self.device).flip(0),
            ])
            self._dct_idx.append(idx)
            self._idct_idx.append(torch.argsort(idx))

            k = torch.arange(Nd, dtype=self.dtype, device=self.device)
            shape = [1] * self.ndim
            shape[d] = Nd
            W = torch.exp(
                -1j * torch.pi * k.reshape(shape) / (2 * Nd)
            ).to(cdtype)
            self._dct_twiddle.append(W)

        print(
            f"PoissonSolverFFT [Neumann/DCT] ready  ({self.ndim}D, "
            f"{'x'.join(str(ni) for ni in self.n)})"
        )

    # ==================================================================
    # DCT-II / IDCT-II  (Makhoul algorithm, FFT of size N)
    # ==================================================================
    def _dct_1d(self, x, d):
        r"""Type-II DCT along dimension *d* using size-N FFT.

        .. math::
            X_k = 2\sum_{n=0}^{N-1} x_n\,
                  \cos\!\frac{\pi k(2n+1)}{2N}

        Algorithm (Makhoul 1980):
        1. Reorder *x*: even-indexed elements, then odd-indexed reversed.
        2. FFT of the reordered vector (size N).
        3. Multiply by twiddle :math:`2 e^{-j\pi k/(2N)}`, take real part.
        """
        v = x.index_select(d, self._dct_idx[d])
        V = torch.fft.fft(v, dim=d)
        return 2.0 * (V * self._dct_twiddle[d]).real

    def _idct_1d(self, X, d):
        r"""Inverse Type-II DCT along dimension *d*.

        Recovers *x* from its DCT-II coefficients *X*:

        .. math::
            x_n = \frac{1}{N}\!\left[\frac{X_0}{2}
                  + \sum_{k=1}^{N-1} X_k\,
                  \cos\!\frac{\pi k(2n+1)}{2N}\right]

        Algorithm:
        1. Set :math:`C'_0 = X_0/2,\; C'_k = X_k` for :math:`k \geq 1`.
        2. Multiply by conjugate twiddle
           :math:`e^{-j\pi k/(2N)}`.
        3. Forward FFT (size N), take real part.
        4. Undo the Makhoul permutation, divide by N.
        """
        Nd = X.shape[d]
        cdtype = self._dct_twiddle[d].dtype

        # Halve the k=0 coefficient
        Cp = X.clone()
        sl = [slice(None)] * X.ndim
        sl[d] = slice(0, 1)
        Cp[tuple(sl)] = Cp[tuple(sl)] * 0.5

        # Twiddle → FFT → real part → un-permute → scale
        Z = Cp.to(cdtype) * self._dct_twiddle[d]   # same twiddle as forward
        V = torch.fft.fft(Z, dim=d)
        v = V.real
        return v.index_select(d, self._idct_idx[d]) / Nd

    def _dctn(self, x):
        """N-dimensional DCT-II (separable, applied dim-by-dim)."""
        for d in range(self.ndim):
            x = self._dct_1d(x, d)
        return x

    def _idctn(self, X):
        """N-dimensional inverse DCT-II (separable, applied dim-by-dim)."""
        for d in range(self.ndim):
            X = self._idct_1d(X, d)
        return X

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
        r"""Solve  :math:`\nabla^2\varphi = u`.

        Dispatches to the free-space or Neumann implementation depending
        on :pyattr:`bc_type`.

        Parameters
        ----------
        u : tensor, shape ``(n0, n1)`` or ``(n0, n1, n2)``

        Returns
        -------
        phi : tensor, same shape as *u*
        """
        if self.bc_type == "free":
            return self._solve_free(u)
        return self._solve_neumann(u)

    # ------------------------------------------------------------------
    def _solve_free(self, u):
        """Free-space solve via Green function convolution."""
        slc = tuple(slice(ni) for ni in self.n)
        self.U[slc] = u
        return torch.real(
            torch.fft.ifftn(self.Gfft * torch.fft.fftn(self.U))
        )[slc]

    # ------------------------------------------------------------------
    def _solve_neumann(self, u):
        r"""Neumann solve via DCT.

        1. :math:`\hat f = \text{DCT-II}(f)`
        2. :math:`\hat\varphi_k = \hat f_k / \lambda_k`
           (zero mode set to 0)
        3. :math:`\varphi = \text{IDCT-II}(\hat\varphi)`
        """
        f_hat = self._dctn(u)
        return self._idctn(f_hat * self.inv_eig)

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

    # ==================================================================
    # ██ Neumann BC tests
    # ==================================================================
    # The FFT Neumann solver exactly inverts the *discrete* Laplacian
    # (with replicate / Neumann ghost cells), NOT the continuous one.
    # We therefore test with the discrete Laplacian of the exact solution.

    def discrete_laplacian_neumann(phi, dh):
        """Discrete Laplacian with Neumann (replicate) ghost cells."""
        ndim = phi.ndim
        Lh = torch.zeros_like(phi)
        for d in range(ndim):
            N = phi.shape[d]
            # Build ghost-padded view along dimension d
            # ghost[-1] = phi[0], ghost[N] = phi[N-1]  (Neumann)
            sl_lo = [slice(None)] * ndim
            sl_hi = [slice(None)] * ndim
            sl_lo[d] = slice(0, 1)  # first element
            sl_hi[d] = slice(-1, None)  # last element
            padded = torch.cat([phi[tuple(sl_lo)], phi, phi[tuple(sl_hi)]], dim=d)
            # Second difference
            sl_m = [slice(None)] * ndim
            sl_0 = [slice(None)] * ndim
            sl_p = [slice(None)] * ndim
            sl_m[d] = slice(0, -2)
            sl_0[d] = slice(1, -1)
            sl_p[d] = slice(2, None)
            Lh = Lh + (padded[tuple(sl_m)] - 2 * padded[tuple(sl_0)]
                       + padded[tuple(sl_p)]) / dh[d] ** 2
        return Lh

    # ------------------------------------------------------------------
    # 2-D Neumann test
    #   phi(x,y) = cos(pi*x/Lx)*cos(pi*y/Ly)
    #   dp/dn = 0 on all boundaries (cosine modes satisfy this)
    # ------------------------------------------------------------------
    print("\n=== 2-D Neumann test ===")
    Lx, Ly = 2.0, 3.0
    nx_n, ny_n = 128, 128
    hx, hy = Lx / nx_n, Ly / ny_n
    x_n = torch.linspace(hx / 2, Lx - hx / 2, nx_n, dtype=dtype, device=device)
    y_n = torch.linspace(hy / 2, Ly - hy / 2, ny_n, dtype=dtype, device=device)

    ps_n2 = PoissonSolverFFT(x_n, y_n, bc_type="neumann")

    Xn, Yn = torch.meshgrid(x_n, y_n, indexing="ij")
    phi_exact_n2 = torch.cos(torch.pi * Xn / Lx) * torch.cos(torch.pi * Yn / Ly)

    # Compute RHS as the *discrete* Laplacian of the exact solution
    f_n2 = discrete_laplacian_neumann(phi_exact_n2, [hx, hy])

    t0 = time.time()
    phi_approx_n2 = ps_n2.solve(f_n2)
    elapsed = time.time() - t0

    # Remove mean (Neumann solution is unique up to a constant)
    phi_approx_n2 = phi_approx_n2 - phi_approx_n2.mean()
    phi_exact_n2  = phi_exact_n2  - phi_exact_n2.mean()

    diff_n2 = torch.abs(phi_approx_n2 - phi_exact_n2)
    scale   = phi_exact_n2.abs().max().item()
    linf_n2 = diff_n2.max().item() / scale
    l2_n2   = torch.sqrt((diff_n2 ** 2).mean()).item() / scale

    print(f"  Solve time: {elapsed:.4f}s")
    print(f"  Relative Linf error: {linf_n2:.2e}")
    print(f"  Relative L2   error: {l2_n2:.2e}")
    assert linf_n2 < 1e-10, f"2-D Neumann Linf error too large: {linf_n2:.2e}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # 3-D Neumann test
    #   phi = cos(pi*x/Lx)*cos(2*pi*y/Ly)*cos(pi*z/Lz)
    # ------------------------------------------------------------------
    print("\n=== 3-D Neumann test ===")
    Lx3, Ly3, Lz3 = 2.0, 3.0, 1.5
    nn3 = 64
    hx3 = Lx3 / nn3
    hy3 = Ly3 / nn3
    hz3 = Lz3 / nn3
    x_n3 = torch.linspace(hx3 / 2, Lx3 - hx3 / 2, nn3, dtype=dtype, device=device)
    y_n3 = torch.linspace(hy3 / 2, Ly3 - hy3 / 2, nn3, dtype=dtype, device=device)
    z_n3 = torch.linspace(hz3 / 2, Lz3 - hz3 / 2, nn3, dtype=dtype, device=device)

    ps_n3 = PoissonSolverFFT(x_n3, y_n3, z=z_n3, bc_type="neumann")

    Xn3, Yn3, Zn3 = torch.meshgrid(x_n3, y_n3, z_n3, indexing="ij")
    kx = torch.pi / Lx3
    ky = 2.0 * torch.pi / Ly3
    kz = torch.pi / Lz3
    phi_exact_n3 = torch.cos(kx * Xn3) * torch.cos(ky * Yn3) * torch.cos(kz * Zn3)

    # Discrete Laplacian with Neumann ghost cells
    f_n3 = discrete_laplacian_neumann(phi_exact_n3, [hx3, hy3, hz3])

    t0 = time.time()
    phi_approx_n3 = ps_n3.solve(f_n3)
    elapsed = time.time() - t0

    phi_approx_n3 = phi_approx_n3 - phi_approx_n3.mean()
    phi_exact_n3  = phi_exact_n3  - phi_exact_n3.mean()

    diff_n3 = torch.abs(phi_approx_n3 - phi_exact_n3)
    scale3  = phi_exact_n3.abs().max().item()
    linf_n3 = diff_n3.max().item() / scale3
    l2_n3   = torch.sqrt((diff_n3 ** 2).mean()).item() / scale3

    print(f"  Solve time: {elapsed:.4f}s")
    print(f"  Relative Linf error: {linf_n3:.2e}")
    print(f"  Relative L2   error: {l2_n3:.2e}")
    assert linf_n3 < 1e-10, f"3-D Neumann Linf error too large: {linf_n3:.2e}"
    print("  PASSED")

    print("\nDone.")
