.. _parameters:

Configuration Reference
=======================

Every LilyTorch simulation is driven by a YAML configuration file with
four top-level sections: ``solver``, ``boundary_conditions``, ``body``,
and ``output``.  A fifth section ``physics`` is used only in FARMS-coupled
mode.

.. contents:: On this page
   :local:
   :depth: 2


``solver`` — Domain and numerics
---------------------------------

Grid and domain
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``Nx``
     - int
     - 512
     - Number of grid cells in :math:`x`.
   * - ``Ny``
     - int
     - 128
     - Number of grid cells in :math:`y`.
   * - ``Nz``
     - int
     - *null*
     - Number of grid cells in :math:`z`.  ``null`` → 2-D simulation.
   * - ``xmin``, ``xmax``
     - float
     - -0.9, 1.5
     - :math:`x` domain bounds.
   * - ``ymin``, ``ymax``
     - float
     - -0.3, 0.3
     - :math:`y` domain bounds.
   * - ``zmin``, ``zmax``
     - float
     - -0.3, 0.3
     - :math:`z` domain bounds (3-D only).

Time stepping
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``dt``
     - float
     - 0.001
     - Time-step size :math:`\Delta t`.
   * - ``nt``
     - int
     - 10001
     - Total number of iterations.
   * - ``starting_iteration``
     - int
     - 0
     - Iteration index to resume from (for restarts).
   * - ``starting_iteration_path``
     - str
     - *null*
     - Path to a saved HDF5 state to load when resuming.

Physics
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``nu``
     - float
     - 1e-6
     - Kinematic viscosity :math:`\nu`.
   * - ``rho``
     - float
     - 1000.0
     - Fluid density :math:`\rho`.
   * - ``rho_body``
     - float
     - *null*
     - Body density (enables variable-density Poisson when set).

Numerics — advection
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``convection_method``
     - str
     - ``"quick"``
     - Advection scheme.  One of: ``"quick"``, ``"adbquickest"``,
       ``"cubista"``, ``"van_leer"``, ``"cds"``, ``"semi_lagrangian"``.
       See :doc:`numerical_schemes`.

.. _smagorinsky-params:

Numerics — Smagorinsky LES
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``smagorinsky_cs``
     - float
     - 0.0
     - Smagorinsky constant :math:`C_s`.  Set to a value in the range
       0.1–0.2 to enable the subgrid-scale eddy-viscosity model.
       When 0 (default), the model is **disabled** and the solver uses
       constant viscosity with zero overhead.
       See :doc:`mathematical_formulation` and :doc:`numerical_schemes`.

Numerics — Poisson solver
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 10 12 54

   * - Key
     - Type
     - Default
     - Description
   * - ``poisson_method``
     - str
     - ``"multigrid"``
     - Solver type: ``"multigrid"`` (V-cycles), ``"mgcg"``
       (CG + V-cycle precond.), or ``"fft"`` (spectral).
   * - ``poisson_bc_type``
     - str
     - ``"neumann"``
     - Pressure BC type: ``"neumann"`` (DCT) or ``"free"``
       (free-space Green's function).
   * - ``poisson_tol``
     - float
     - 1e-7
     - Relative residual tolerance :math:`\|r\|/\|f\|`.
   * - ``poisson_max_cycles``
     - int
     - 5
     - Max outer CG iterations (MGCG).
   * - ``poisson_max_mgcg_cycles``
     - int
     - 3
     - Max standalone V-cycles (multigrid).
   * - ``poisson_nsmoothing``
     - int
     - 10
     - Pre-/post-smoother sweeps per multigrid level.
   * - ``jacobi_weight``
     - float
     - 0.7
     - Weighted Jacobi relaxation parameter :math:`\omega`.
   * - ``poisson_precond_vcycles``
     - int
     - 1
     - V-cycles per CG preconditioner step (MGCG).
   * - ``poisson_smoother``
     - str
     - ``"jacobi"``
     - Smoother type: ``"jacobi"`` or ``"rbgs"``
       (Red-Black Gauss-Seidel).
   * - ``poisson_verbose``
     - bool
     - false
     - Print residual convergence per solve.
   * - ``poisson_folder``
     - str
     - *null*
     - Directory to cache pre-computed Green's functions
       (free-space mode).
   * - ``poisson_warm_start``
     - bool
     - false
     - Reuse previous :math:`p` as Poisson initial guess.

Compilation
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``poisson_compile``
     - bool
     - *null*
     - ``torch.compile`` for multigrid smoother / V-cycle.
   * - ``compile_adv_diff``
     - bool
     - *null*
     - ``torch.compile`` for advection–diffusion kernel.
   * - ``compile_forces``
     - bool
     - *null*
     - ``torch.compile`` for force-integration kernels.
   * - ``compile_sdf``
     - bool
     - *null*
     - ``torch.compile`` for SDF / :math:`\mu` / normals evaluation.

Solver mode
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``use_kernels``
     - bool
     - true
     - Single user-facing switch between the two solver variants.
       ``true`` selects the streaming C++/CUDA kernel path
       (per-body fused SDF + force kernels, union-AABB crops for
       shared stress, :math:`\mu` / normals, and the BDIM
       meta-equation).  ``false`` selects the suboptimal pure-PyTorch
       reference path (no batching, no per-body cropping) which is
       useful for validation.  Independent of ``use_gpu``: kernel
       mode is supported on both CPU and CUDA.
   * - ``sdf_interp_method``
     - str
     - ``"trilinear"``
     - Body-SDF sampling stencil used inside the streaming kernels.
       ``"trilinear"`` is the 2×2×2 stencil that matches the
       historical behaviour; ``"triquadratic"`` uses a 3×3×3 Lagrange
       stencil for higher-order SDF accuracy near the body surface
       and falls back to trilinear in the boundary layer of the body
       grid.

Hardware
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``use_gpu``
     - bool
     - true
     - Use CUDA if available.
   * - ``nthreads``
     - int
     - 4
     - CPU threads (when ``use_gpu: false``).
   * - ``dtype``
     - str
     - ``"float32"``
     - ``"float32"`` or ``"float64"``.  Honoured by both the
       pure-PyTorch path and the C++/CUDA streaming kernels (which
       dispatch via ``AT_DISPATCH_FLOATING_TYPES``).  Set on the
       :class:`~lilytorch.examples.base_sim_config.BaseSimConfig`
       — the value flows through to :class:`FluidSolver` and
       :class:`BDIMhandler` automatically; passing
       ``dtype=torch.float64`` to ``FluidSolver(...)`` directly is no
       longer required.  The aliases ``"single"`` / ``"double"`` are
       also accepted.

Miscellaneous
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``perturbation_amplitude``
     - float
     - *null*
     - Amplitude of random perturbation added to initial velocity.
   * - ``zero_pressure_inside``
     - bool
     - *null*
     - Explicitly set :math:`p = 0` inside bodies.
   * - ``force_method``
     - str
     - ``"eulerian"``
     - Force-integration method. ``"eulerian"`` — volumetric smoothed-delta band
       integral (:math:`\int \sigma\cdot\delta_\varepsilon\,dV` and
       :math:`-\int p\,\mathbf{n}\,\delta_\varepsilon\,dV`); works on both the
       python and kernel paths. ``"lagrangian"`` — surface integral
       :math:`\oint \sigma\cdot\mathbf{n}\,dS` on per-body Lagrangian markers
       (2-D arc-length contour / 3-D triangulation), most accurate for
       fully-resolved bodies. The legacy aliases ``"method2"`` → ``"eulerian"``
       and ``"method1"`` → ``"lagrangian"`` are still accepted (with a
       ``DeprecationWarning``).
   * - ``force_submethod``
     - str
     - ``"ndelta"``
     - Pressure-force readout for ``force_method="eulerian"``. ``"ndelta"`` —
       per-body smoothed-delta band integral :math:`-\sum p\,\mathbf{n}\,
       \delta_\varepsilon(\phi_b)` (historical default). ``"deltaH"`` —
       *partial-Heaviside* readout :math:`-\sum p\,\partial_i H_\varepsilon`,
       where the pressure force density is taken from the **union** SDF (one
       closed surface, no internal inter-link seams) so it obeys
       summation-by-parts and does not leak the hydrostatic baseline; the union
       force is split back to the individual bodies by a softmin partition of
       unity :math:`w_b = \mathrm{softmax}(-\phi_b/\tau)` so
       :math:`\sum_b \mathbf{F}_b` equals the union force exactly. Viscous
       force/torque are unchanged. Implemented in the native CUDA/CPU force
       kernels (2-D and 3-D), bit-matched to the python reference
       (:py:meth:`~lilytorch.src.two_phase_solver.TwoPhaseSolver._apply_partition_heaviside`).
       Useful for surface-straddling / multi-link swimmers where the per-body
       ``ndelta`` quadrature admits a depth-linear spurious force.
   * - ``force_ph_blend_cells``
     - float
     - ``1.5``
     - Softmin partition temperature for ``force_submethod="deltaH"``, in grid
       cells (:math:`\tau = \texttt{force\_ph\_blend\_cells}\cdot h`). Sets how
       sharply the union force is handed between abutting links at a seam.
   * - ``body_velocity_blend_eps_cells``
     - float
     - *null*
     - Width (in grid cells) of the smooth SDF-weighted velocity blend in
       the overlap band between articulated links. ``null`` / ``0`` keeps the
       legacy running-min winner-take-all (hard velocity switch at seams,
       which can destabilise the coupling for overlapping links, e.g.
       ``convexify=True``). ``2.0`` matches the BDIM kernel half-width and is
       recommended. See :ref:`velocity-blend`.


``boundary_conditions`` — Velocity BCs
----------------------------------------

See :doc:`boundary_conditions` for a detailed explanation of face ordering
and enforcement.

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Key
     - Type
     - Description
   * - ``BC_type_u``
     - list[str]
     - BC type per face for :math:`u`.  ``"D"`` = Dirichlet,
       ``"N"`` = Neumann.
   * - ``BC_values_u``
     - list[float]
     - Value per face for :math:`u`.
   * - ``BC_type_v``
     - list[str]
     - Same for :math:`v`.
   * - ``BC_values_v``
     - list[float]
     - Same for :math:`v`.
   * - ``BC_type_w``
     - list[str]
     - Same for :math:`w` (3-D only).
   * - ``BC_values_w``
     - list[float]
     - Same for :math:`w` (3-D only).


``body`` — Immersed-body geometry
----------------------------------

General keys
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Key
     - Type
     - Description
   * - ``type``
     - str
     - Body type.  See table below.
   * - ``sdf_folder``
     - str
     - Folder with pre-computed SDF arrays.
   * - ``save_folder``
     - str
     - Folder to save computed SDF / interpolation data.
   * - ``plotting``
     - bool
     - Whether to compute SDF for visualisation.
   * - ``compute_interp``
     - bool
     - Whether to compute interpolation data.
   * - ``plotting_meshes``
     - bool
     - Whether to render mesh surfaces in plots.
   * - ``suit``
     - float
     - SDF offset (thickens/thins the body envelope).
   * - ``scale``
     - float
     - Mesh scale factor.
   * - ``convexify``
     - bool
     - Convexify mesh before SDF computation.
   * - ``n_samples``
     - int
     - Sample count for SDF computation.
   * - ``force_scaling``
     - float
     - Multiply computed forces by this factor.
   * - ``contour_mask``
     - float
     - Contour-mask threshold for force integration.

Supported body types
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - ``type`` value
     - Class
     - Description
   * - ``composite_analytical``
     - :class:`~lilytorch.src.body.CompositeBodyAnalytical`
     - Union of analytical shapes (circle, box, capsule, segment, NACA
       fish spine, ...). The only user-facing analytical body type.
   * - ``composite_mesh``
     - :class:`~lilytorch.src.body.CompositeBodyMesh`
     - Union of 3-D meshes (STL / OBJ). The only user-facing mesh body
       type.

.. note::

   Direct (non-composite) body types — ``analytical``, ``mesh``,
   ``fish_analytical``, ``fish_experimental``, ``multi_animat``,
   ``composite_segment_body`` — were removed from
   :func:`~lilytorch.src.body.body_from_yaml`. The underlying classes
   still exist as internal building blocks used by the two composite
   wrappers above and by the FARMS coupling layer.

Analytical-specific keys
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Type
     - Description
   * - ``sdf_type``
     - str
     - ``"circle"``, ``"box"``, ``"capsule"``, ``"segment"``.
   * - ``position``
     - list
     - Centre position :math:`[x, y]` or :math:`[x, y, z]`.
   * - ``radius``
     - float
     - Shape radius.
   * - ``velocity``
     - list
     - Prescribed linear velocity.
   * - ``angular_velocity``
     - float
     - Prescribed angular velocity.
   * - ``update_maps``
     - dict
     - Kinematics: ``{"rotation": ..., "translation": [...]}``.


``output`` — Saving and visualisation
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``save_path``
     - str
     - ``""``
     - Base directory for all output.
   * - ``existing_folder``
     - str
     - *null*
     - Re-use an existing output folder (skip timestamp subfolder).
   * - ``save_frames``
     - bool
     - true
     - Save PNG visualisation frames.
   * - ``save_every``
     - int
     - 100
     - Save every *N*-th iteration.
   * - ``save_uv``
     - bool
     - false
     - Save raw velocity / pressure fields (HDF5).
   * - ``save_vtk``
     - bool
     - false
     - Save VTK rectilinear grids for ParaView.
   * - ``vmin``
     - float
     - *null*
     - Colour-bar minimum (``"auto"`` for automatic).
   * - ``vmax``
     - float
     - *null*
     - Colour-bar maximum (``"auto"`` for automatic).
   * - ``plot_specs``
     - list
     - ``["curl", "pressure"]``
     - Fields used for saved 2-D plots and 3-D orthogonal slice plots. Each entry can be a field name or a dict such as ``{"name": "pressure", "vmin": "auto", "vmax": "auto", "show_body": true}``.
   * - ``iso_3d_specs``
     - list
     - ``["omega_mag", "vel_mag"]``
     - Fields rendered as 3-D isosurfaces. Each entry can be a field name or a dict such as ``{"name": "omega_mag", "iso_value": 5.0}``.
   * - ``iso_3d_value``
     - float
     - *null*
     - Fixed threshold used for all configured 3-D isosurface fields when set. Use ``null`` or ``"auto"`` to keep the existing automatic peak-fraction thresholding.


``physics`` — FARMS coupling (optional)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Key
     - Type
     - Description
   * - ``solref``
     - list
     - MuJoCo solver reference parameters ``[timeconst, dampratio]``.
