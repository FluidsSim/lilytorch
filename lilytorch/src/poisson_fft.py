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

        # Halve the k=0 coefficient — build the twiddle in-place
        # to avoid a full .clone() of the 3-D array (~264 MB saved).
        W = self._dct_twiddle[d]
        sl = [slice(None)] * X.ndim
        sl[d] = slice(0, 1)
        W_mod = W.clone()          # clone only the small 1-D twiddle
        W_mod[tuple(sl)] = W_mod[tuple(sl)] * 0.5

        # Twiddle (with halved k=0) → cast → FFT → real → un-permute → scale
        Z = X.to(cdtype) * W_mod
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
        on :attr:`bc_type`.

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
