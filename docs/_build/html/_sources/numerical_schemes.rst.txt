.. _numerical_schemes:

Numerical Schemes
=================

This page describes the spatial and temporal discretisation used by
LilyTorch.  For the continuous equations see :doc:`mathematical_formulation`.

.. contents:: On this page
   :local:
   :depth: 2


Grid layout — staggered MAC grid
---------------------------------

LilyTorch uses a **Marker-and-Cell (MAC) staggered grid**:

* **Pressure** :math:`p` — cell-centred.
* **Velocity** :math:`u` — :math:`x`-face-centred (staggered :math:`+\tfrac{h}{2}` in :math:`x`).
* **Velocity** :math:`v` — :math:`y`-face-centred (staggered :math:`+\tfrac{h}{2}` in :math:`y`).
* **Velocity** :math:`w` (3-D) — :math:`z`-face-centred (staggered :math:`+\tfrac{h}{2}` in :math:`z`).

One **ghost cell** is added on every side of the domain, giving array shapes
of ``(Nx+2, Ny+2)`` (2-D) or ``(Nx+2, Ny+2, Nz+2)`` (3-D).  The interior
is indexed as ``[1:-1, 1:-1, ...]``.

.. code-block:: text

   Ghost │  Interior cells (Nx)  │ Ghost
    [0]  │ [1] [2] … [Nx] │[Nx+1]

Discrete differential operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All spatial operators are second-order central differences on the
staggered grid:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Operator
     - Function
     - Stencil
   * - Pressure gradient
     - ``operations.gradient(p)``
     - Backward difference :math:`(p_i - p_{i-1})/h`
   * - Velocity divergence
     - ``operations.divergence(u, v [, w])``
     - Forward difference :math:`(u_{i+1} - u_i)/h`
   * - Normal derivative
     - ``operations.normal_derivative(var, nx, ny [, nz])``
     - :math:`\hat{n}\cdot\nabla\phi`
   * - Vorticity (2-D)
     - ``operations.vorticity(u, v)``
     - Scalar :math:`\omega_z = \partial v/\partial x - \partial u/\partial y`
   * - Vorticity (3-D)
     - ``operations.vorticity_components(u, v, w)``
     - :math:`(\omega_x, \omega_y, \omega_z, |\boldsymbol\omega|)`, clipped
       ``[2:-2]`` to avoid ghost artefacts


.. _heun-scheme:

Time integration — Heun's method (RK2)
---------------------------------------

The full Navier–Stokes time-step follows a **Heun predictor–corrector**
(explicit Runge–Kutta 2) scheme mirroring WaterLily.jl's ``mom_step!``:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Stage
     - Operations
   * - **Predictor**
     - 1. Advection–diffusion on :math:`\mathbf{u}^n` → :math:`\mathbf{u}^*`
       2. BDIM meta-equation: :math:`\mathbf{u}^* \leftarrow \text{BDIM}(\mathbf{u}^*, \mathbf{u}^n_{\text{body}})`
       3. Pressure projection with weight :math:`w = 1` → :math:`\tilde{\mathbf{u}}`
   * - **Corrector**
     - 1. Advection–diffusion on :math:`\tilde{\mathbf{u}}` → :math:`\mathbf{u}^{**}`
       2. Rebase: :math:`\mathbf{u}^{**} \leftarrow \tfrac12(\mathbf{u}^{**} + \mathbf{u}^n)`
       3. BDIM meta-equation
       4. Pressure projection with weight :math:`w = \tfrac12` → :math:`\mathbf{u}^{n+1}`

The advection–diffusion sub-step itself uses **Forward Euler**.


Advection (convection) schemes
------------------------------

The advection fluxes :math:`\nabla\cdot(\mathbf{u}\otimes\phi)` are
evaluated with one of six configurable schemes, selected by the
``convection_method`` YAML key:

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Scheme
     - Order
     - Description
   * - ``quick``
     - 3rd
     - Quadratic Upstream Interpolation for Convective Kinematics
       (Leonard 1979) with median limiter.
   * - ``adbquickest``
     - 3rd TVD
     - Adaptive QUICK with Total Variation Diminishing (TVD) limiter.
   * - ``cubista``
     - 2nd TVD
     - Convergent & Universally Bounded Interpolation Scheme for the
       Treatment of Advection (Alves et al. 2003).
   * - ``van_leer``
     - 2nd TVD
     - Van Leer flux limiter (van Leer 1979).
   * - ``cds``
     - 2nd
     - Central Difference Scheme — unbounded, for smooth problems.
   * - ``semi_lagrangian``
     - —
     - Stam (1999) unconditionally-stable backtracing.  Useful for
       large time-steps but introduces numerical diffusion.


.. _multigrid-solver:

Poisson solvers
---------------

FFT solvers
^^^^^^^^^^^

Two FFT-based solvers are available, selected via ``poisson_bc_type``:

**Neumann** (``poisson_bc_type: neumann``)
   Spectral solve using the Discrete Cosine Transform (DCT-II /
   IDCT-II).  The eigenvalues of the discrete Neumann Laplacian are

   .. math::

      \lambda_k = \frac{2}{h^2}\left(\cos\frac{\pi k}{N} - 1\right)

   giving spectral accuracy.  Only applicable when all boundaries are
   zero-normal-gradient (Neumann).

**Free-space** (``poisson_bc_type: free``)
   Zero-padded convolution with a regularised Green's function.  The
   Green's function is pre-computed (or loaded from ``poisson_folder``
   cache) and stored in Fourier space.  The RHS is zero-padded to twice
   the domain size, Fourier-transformed, multiplied by the Green's
   function, and inverse-transformed.

   See :doc:`mathematical_formulation` for the Green's function
   expressions.


Multigrid / MGCG solver
^^^^^^^^^^^^^^^^^^^^^^^^

For variable-coefficient problems a **geometric multigrid** solver is
used (class :class:`~lilytorch.src.poisson_mult.PoissonSolver`).

**V-cycle** structure:

1. **Pre-smooth** — ``nsmoothing`` sweeps of the chosen smoother.
2. **Restrict** — full-weighting restriction of the residual and face
   coefficients to the next-coarser level.
3. **Recurse** — repeat until the 2×2(×2) coarsest grid, where a
   direct solve is trivial.
4. **Prolongate** — bilinear (2-D) / trilinear (3-D) interpolation to
   the finer level.
5. **Post-smooth** — ``nsmoothing`` sweeps.

**MGCG** (``poisson_method: mgcg``) wraps the V-cycle as a
**preconditioner** inside a Conjugate Gradient iteration.

Smoothers
"""""""""

.. list-table::
   :header-rows: 1

   * - Smoother
     - Key
     - Description
   * - Weighted Jacobi
     - ``jacobi``
     - :math:`p^{k+1} = (1-\omega)\,p^k + \omega\,\tilde{p}`, default
       :math:`\omega = 0.7`.
   * - Red-Black Gauss-Seidel
     - ``rbgs``
     - Lexicographic colouring; two half-sweeps per iteration.  Better
       smoothing per sweep than Jacobi but less parallelisable.

Tuning parameters
"""""""""""""""""

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``poisson_tol``
     - ``1e-7``
     - Relative residual tolerance :math:`\|r\|/\|f\|`.
   * - ``poisson_max_cycles``
     - 5
     - Max outer CG iterations (MGCG mode).
   * - ``poisson_max_mgcg_cycles``
     - 3
     - Max standalone V-cycles (multigrid mode).
   * - ``poisson_nsmoothing``
     - 10
     - Pre-/post-smoothing sweeps per level.
   * - ``jacobi_weight``
     - 0.7
     - Weighted Jacobi relaxation parameter :math:`\omega`.
   * - ``poisson_precond_vcycles``
     - 1
     - V-cycles per CG preconditioner application.
   * - ``poisson_compile``
     - ``null``
     - Enable ``torch.compile`` for smoother/V-cycle kernels.
   * - ``poisson_smoother``
     - ``jacobi``
     - Smoother type: ``"jacobi"`` or ``"rbgs"``.


``torch.compile`` integration
-----------------------------

LilyTorch can JIT-compile hot kernels via ``torch.compile`` with
``mode="reduce-overhead"`` (CUDA-graph fusion).  Compilation targets:

.. list-table::
   :header-rows: 1

   * - Target
     - Config key
   * - Advection–diffusion
     - ``compile_adv_diff``
   * - Force computation
     - ``compile_forces``
   * - SDF / :math:`\mu` / normals
     - ``compile_sdf``
   * - Multigrid smoother
     - ``poisson_compile``
   * - Full V-cycle
     - ``poisson_compile``

All compiled functions are **module-level pure functions** (not class
methods), because ``torch.compile`` performs best on stateless functions.
Variants are generated per ``(ndim, smoother)`` combination.

.. note::

   Compilation incurs a one-time warm-up cost.  For short simulations it
   may be faster to leave compilation disabled.
