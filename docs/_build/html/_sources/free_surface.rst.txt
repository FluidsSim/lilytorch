.. _free_surface:

Free Surface — Level-Set Fluid–Air
===================================

LilyTorch can track a **water–air free surface** with a level-set method
(Osher & Sethian 1988; Sussman, Smereka & Osher 1994; the ghost-fluid
treatment follows Fedkiw et al. 1999 / Gibou et al. 2002). The
implementation lives in :class:`~lilytorch.src.free_surface.FreeSurface`
and is fully **decoupled** from the BDIM solid-body machinery: it stores
no body SDF, never touches ``composite_body``, and exposes only the
plumbing the solver needs.

.. contents:: On this page
   :local:
   :depth: 2


Single-fluid model
-------------------

The air phase is **not** resolved as a real Navier–Stokes fluid. It is
treated as a passive, massless region on which the pressure is pinned to
the atmospheric gauge value :math:`p_{\text{atm}} = 0`. Only the water is
advanced by the solver. This is the standard *single-fluid* (or
*free-surface*) approximation: valid when the air's inertia and viscous
stress are negligible compared with the water's (a :math:`\sim\!10^3`
density ratio), e.g. sloshing, dam break, sedimentation with a top
surface, hydrostatic columns.

The interface is carried by a cell-centred **level-set scalar**
:math:`\phi(\mathbf{x},t)` with the sign convention

.. math::

   \phi < 0 \ \text{(water)}, \qquad
   \phi = 0 \ \text{(interface)}, \qquad
   \phi > 0 \ \text{(air)} .

In its well-formed state :math:`\phi` is a **signed distance function**,
:math:`|\nabla\phi| = 1`. Note this is the *opposite* sign convention to
the BDIM body SDF :math:`d` (positive in fluid); the two fields are
independent and never mixed.


The per-step cycle
------------------

Each solver time step does, in order:

1. **predictor / projection** of the water velocity (unchanged BDIM
   pipeline) with the ghost-fluid pressure boundary condition baked into
   the Poisson coefficients (see :ref:`fs-ghost-fluid`);
2. **advect** :math:`\phi` with the projected, divergence-free velocity
   (:ref:`fs-advection`);
3. periodically **reinitialise** :math:`\phi` back to a signed distance
   function (:ref:`fs-reinit`);
4. periodically **extend** the water velocity into a narrow air band so
   the next advection has a meaningful velocity there (:ref:`fs-extend`).

Steps 2–4 are bundled in
:meth:`FluidSolver._fs_post_step <lilytorch.src.solver.FluidSolver>` and
called once per step from
:meth:`~lilytorch.src.solver.FluidSolver.finalize_step`. Step 1 is woven
into :meth:`~lilytorch.src.solver.FluidSolver.project`.


.. _fs-advection:

Interface advection
-------------------

The interface moves with the flow,

.. math::
   :label: fs-advect

   \frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi = 0 .

The interface is advected with the **same convective schemes as the
Navier–Stokes solver** (QUICK / ADBQUICKEST / CUBISTA / van Leer / CDS),
shared from :mod:`lilytorch.src.advection`.
:meth:`FreeSurface.advect <lilytorch.src.free_surface.FreeSurface.advect>`
calls :func:`~lilytorch.src.advection.advect_scalar`, which applies the
chosen scheme in **conservative flux form** on the MAC velocity and steps
**forward Euler** in time. The MAC velocity component normal to each
cell-centred face already sits at that face (no transverse averaging), so
the level set is transported at the scheme's order rather than with a
bespoke first-order upwind. Because the advecting velocity is the
projected, divergence-free field, the conservative form
:math:`\nabla\cdot(\mathbf{u}\phi)` equals the advective form
:math:`\mathbf{u}\cdot\nabla\phi`. Stable under the usual CFL condition

.. math::

   \Delta t \, \frac{|\mathbf{u}|_{\max}}{h} < 1 .

The scheme is selected with the ``convection_method`` key in the
``free_surface`` config block (default ``"quick"``; falls back to
``"quick"`` if the solver's own ``convection_method`` is a non-finite-volume
method such as ``semi-lagrangian``). Boundary ghost cells are zero-gradient
(Neumann) padded after every update.

.. note::

   Forward Euler is used deliberately (not Heun/RK2): the level set is
   advected **once per solver step** in :meth:`finalize_step
   <lilytorch.src.solver.FluidSolver.finalize_step>` on the already-converged
   divergence-free velocity, so it is not wrapped in the predictor–corrector
   loop that integrates the momentum equations.


.. _fs-reinit:

Reinitialisation
----------------

Advecting :math:`\phi` with a non-uniform velocity field distorts it away
from a signed distance function (gradients steepen or flatten near the
interface), which degrades the accuracy of the normal, the curvature, and
the ghost-fluid face fractions. Reinitialisation restores
:math:`|\nabla\phi| = 1` **without moving the zero level set** by
iterating Sussman's pseudo-time PDE to steady state:

.. math::
   :label: fs-reinit-pde

   \frac{\partial \phi}{\partial \tau}
     = \operatorname{sign}(\phi)\bigl(1 - |\nabla\phi|\bigr) .

Key numerical choices in :meth:`FreeSurface.reinitialize
<lilytorch.src.free_surface.FreeSurface.reinitialize>`:

* The sign function is **smoothed** and computed from the *frozen* field
  :math:`\phi_0` at the start of the call,
  :math:`\operatorname{sign}(\phi_0) = \phi_0/\sqrt{\phi_0^2+\varepsilon^2}`
  with :math:`\varepsilon = h`, so the zero crossing is pinned and not
  allowed to drift across sub-steps.
* :math:`|\nabla\phi|` uses the **Godunov upwind** selection
  (:meth:`_godunov_grad_magnitude`): per axis, for
  :math:`\operatorname{sign} > 0` take
  :math:`\sqrt{\max(D^-,0)^2 + \min(D^+,0)^2}` (and the reflected choice
  for :math:`\operatorname{sign} < 0`), which propagates information
  outward from the interface.
* Forward Euler with sub-step :math:`\Delta\tau = 0.5\,h`
  (CFL :math:`\le 1` for unit propagation speed).

Run every ``reinit_every`` solver steps for ``reinit_iters`` sub-steps
each. Reinitialising too aggressively can erode thin features and shift
the interface (the classic level-set mass-loss failure mode); too rarely
lets :math:`\phi` drift and the face fractions become inaccurate.


.. _fs-extend:

Velocity extension
------------------

Cells on the air side have no physically meaningful velocity (air is not
simulated), yet the next advection step :eq:`fs-advect` reads a velocity
in the band just outside the interface. The water velocity is
extrapolated **constant along the normal** into the air half-space by
iterating

.. math::
   :label: fs-extend-pde

   \frac{\partial q}{\partial \tau}
     + \operatorname{sign}(\phi)\,\hat{\mathbf{n}}\cdot\nabla q = 0 ,
   \qquad
   \hat{\mathbf{n}} = \frac{\nabla\phi}{|\nabla\phi|} ,

separately for each velocity component :math:`q`. Only cells with
:math:`\phi > 0` are updated; water cells are frozen. The
:math:`\operatorname{sign}(\phi)` factor (folded into the unit normal in
:meth:`_cc_unit_normal`) makes the upwind scheme transport values
*outward* from the interface. Forward Euler with
:math:`\Delta\tau = 0.5\,h`, run every ``extend_every`` solver steps. See
:meth:`FreeSurface.extend_velocity
<lilytorch.src.free_surface.FreeSurface.extend_velocity>`.

.. important::

   **Extend over the whole air region, not just a band** (``extend_full:
   true``, the default). The air is decoupled from the pressure solve
   (zero-coefficient air–air faces) and gravity is gated to the water, so
   the bulk air has *no* mechanism to damp velocity. If the extension only
   fills a narrow band, the predictor (advection–diffusion still runs on
   the full grid) slowly leaks the small water-surface residual into the
   undamped bulk air, where it accumulates as growing spurious vorticity
   (observed as faint, slowly-growing curl above the interface). Sweeping
   the extension across the entire air half-space each step makes the air
   a clean normal-extension of the water velocity — bounded and slaved to
   the water rather than free-running. The front advances :math:`\sim\!½`
   cell per sub-step, so the cost is :math:`O(\text{air depth})` sub-steps
   of cheap cell-centred ops per step; set ``extend_full: false`` (and use
   ``extend_iters``) to restrict to a band if that cost matters.

   Note this only constrains the *air*; it does not remove the water's own
   (tiny) projection residual, which a quiescent test's auto-ranged
   vorticity plot will still amplify (see :ref:`fs-air-viz`).

.. note::

   The extension is applied directly to the **MAC** velocity components
   even though the normal is computed at cell centres. Within a
   few-cell-wide band this is a deliberate first-order approximation —
   the upwind *direction* matters more than the half-cell staggering
   offset there — and is documented as such in
   :meth:`FluidSolver._fs_post_step`.

.. _fs-air-viz:

Visualising the air
^^^^^^^^^^^^^^^^^^^^

The air is **not a real fluid** here — its pressure is pinned to 0 and
its velocity is only a kinematic extension of the water — so plotting its
fields can mislead. The 2-D field plots therefore **blank the air
half-space** (cells with :math:`\phi>0` are set to ``NaN`` and rendered in
neutral grey), and the symmetric auto colour-range is computed over the
**water only** (NaN-aware), so it is not skewed by the air. This happens
automatically whenever a free surface is active.

This removes spurious air structure from the plots, but a *quiescent*
case (e.g. the hydrostatic column, where the whole flow is at rest to
:math:`\sim\!10^{-4}`) will still show faint grid-scale noise **in the
water**, because symmetric auto-ranging stretches the velocity/vorticity
noise floor to full contrast. That is a property of auto-ranging a
near-zero field, not a solver defect — set fixed ``vmin``/``vmax`` (or
plot a field with real dynamic range) to suppress it.


.. _fs-ghost-fluid:

Ghost-fluid pressure boundary condition
---------------------------------------

The defining feature of the free surface is the Dirichlet pressure
condition :math:`p = p_{\text{atm}} = 0` **on the interface**, not at the
nearest cell centre. Imposing it at cell centres would smear the surface
by up to one cell and lose the hydrostatic balance. Instead LilyTorch
uses the **Ghost-Fluid Method (GFM)**: the boundary condition is baked
into the pressure-Poisson assembly by rescaling the per-face
coefficients, so no separate interface mesh or cut-cell bookkeeping is
needed.

Recall the staggered projection (see :ref:`immersed_boundary`) solves
:math:`\nabla\cdot(c\,\nabla p) = \nabla\cdot\tilde{\mathbf u}` with a
face coefficient :math:`c`. For each MAC face separating a cell pair
:math:`(a,b)`, :meth:`FreeSurface.ghost_fluid_face_scales
<lilytorch.src.free_surface.FreeSurface.ghost_fluid_face_scales>`
multiplies :math:`c` by

.. math::

   s =
   \begin{cases}
     1 & \text{water–water face} \\[4pt]
     0 & \text{air–air face} \\[4pt]
     \dfrac{1}{\max(\theta,\theta_{\min})}
       & \text{cut face (water}\leftrightarrow\text{air)}
   \end{cases}
   \qquad
   \theta = \frac{|\phi_{\text{water}}|}{|\phi_a| + |\phi_b|} ,

where :math:`\theta\in(0,1]` is the **water fraction** of the cell-to-cell
gap, i.e. the fractional distance from the water cell centre to the
interpolated zero crossing of :math:`\phi`. The :math:`1/\theta` factor
places the Dirichlet :math:`p=0` *exactly at the interface* between the
two cell centres (first-order accurate interface location). The lower
clamp :math:`\theta_{\min}` (config ``theta_min``, default ``0.01``)
keeps the stencil non-singular when the interface grazes a face.

The :math:`s=0` air–air faces **decouple** air cells from the linear
system. Combined with the smoother's masking (``Jinv = where(|J| ≥ tol,
1/J, 0)``) this drives :math:`p_{\text{air}} \to 0` for free in the
interior of the air region.

Solver integration (in :meth:`FluidSolver.project
<lilytorch.src.solver.FluidSolver.project>`):

* the GFM scales multiply the staggered Poisson coefficients in-place
  (:meth:`_fs_apply_gfm_to_coeffs`, handling both the face-grid kernel
  path and the full-grid legacy path);
* the divergence RHS is **zeroed in air cells** so the smoother sees no
  spurious driving term where :math:`p\equiv0`;
* the inner air mask is handed to the multigrid solver as an explicit
  ``dirichlet_mask`` that re-pins :math:`p_{\text{air}} = 0` after every
  sweep at every level — this is needed because the cut-face
  :math:`1/\theta` scaling makes the *first* layer of air cells
  non-degenerate (:math:`J\neq0`), so the smoother's :math:`J=0`
  mechanism alone would not pin them;
* after the solve, the **pressure datum is anchored** (see below) and
  :meth:`apply_pressure_mask` defensively re-zeros air pressures.

Pressure datum (gauge anchor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ghost-fluid solve fixes the pressure *gradient* but leaves the
additive constant — the **DC null-space mode** — only weakly constrained.
Because air cells are decoupled (zero-coefficient air–air faces) and
couple to the water only through the thin layer of cut faces, the
(effectively singular) coarse multigrid problems do not pin the water
datum; in practice the constant even **drifts further with more
V-cycles** (e.g. a hydrostatic column at 30 vs 200 V-cycles drifts to a
larger uniform offset, with the gradient unchanged). Anchoring on the
air-cell mean does nothing — the air is already pinned to 0.

The datum is instead recovered from the physical condition
:math:`p = p_{\text{atm}} = 0` **at the interface**. For every water cell
adjacent to air, the water pressure is linearly extrapolated to the
:math:`\phi=0` crossing using the cut-face water fraction :math:`\theta`
and the next water cell inward,

.. math::

   C \approx p_w + \theta\,(p_w - p_{w2}),

which equals the gauge constant exactly for a locally linear pressure.
Averaging :math:`C` over all cut faces gives a single robust offset that
is subtracted from the whole field (:meth:`FreeSurface.interface_gauge_offset
<lilytorch.src.free_surface.FreeSurface.interface_gauge_offset>`). It is
:math:`\rho`/:math:`g`-agnostic and valid for a deforming interface. On
the hydrostatic-column validation this brings the water pressure error
from :math:`\sim\!120\%` (pure-offset) down to the O(:math:`h`)
discretisation floor (:math:`\sim\!0.25\%`).

.. warning::

   The ``dirichlet_mask`` re-pinning is wired through the **multigrid /
   MGCG** path. The FFT Poisson backend solves only a constant-coefficient
   Laplacian and does not consume the per-face GFM scales the same way;
   use ``poisson_method: multigrid`` (or ``mgcg``) for free-surface runs.


Gravity coupling
----------------

A free surface is only interesting under a body force. Gravity is an
independent opt-in block (``solver.gravity``) added as a predictor-side
forward-Euler body force :math:`\Delta t\,\mathbf{g}` right before the
projection, so the Poisson solve balances it into the hydrostatic
pressure field. When the free surface is active the body force is
**gated by the cell-centred water mask**: gravity accelerates only water
cells. Without the gate the massless air band cells free-fall and
dominate :math:`|\mathbf{u}|_{\max}` even though their pressure is pinned
to zero. See :meth:`FluidSolver._apply_gravity_body_force
<lilytorch.src.solver.FluidSolver>`.


Configuration
-------------

Enable the free surface with the ``solver.free_surface`` YAML block (and
usually ``solver.gravity``):

.. code-block:: yaml

   solver:
     # ... grid, nu, rho, dt ...
     poisson_method: multigrid       # GFM Dirichlet pin needs the MG path
     gravity: [0.0, -9.81]           # body force (length = ndim)
     free_surface:
       phi_init: "lambda X, Y: Y - 0.5"   # < 0 water, > 0 air; interface at y = 0.5
       convection_method: quick      # level-set advection scheme (advection.SCHEMES)
       theta_min:    0.01            # cut-face water-fraction clamp (1/theta)
       band_cells:   4               # narrow-band half-width (informational)
       reinit_iters: 4               # reinit sub-steps per call
       reinit_every: 10              # reinitialise every N solver steps
       extend_iters: 2               # band extension sub-steps (used when extend_full: false)
       extend_every: 1               # extend every N solver steps
       extend_full:  true            # extend over the WHOLE air region (recommended)

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Meaning
   * - ``phi_init``
     - *required*
     - Callable (or lambda string evaluated in a ``torch``-only namespace)
       returning the initial level set on the cell-centred grid:
       ``phi_init(X, Y)`` in 2-D, ``phi_init(X, Y, Z)`` in 3-D. Negative
       in water, positive in air.
   * - ``convection_method``
     - solver's, else ``"quick"``
     - Convective scheme for the level-set advection, from
       :data:`lilytorch.src.advection.SCHEMES` (``quick``, ``abdquickest``,
       ``cubista``, ``vanLeer``, ``cds``). Defaults to the solver's
       ``convection_method`` when it is a finite-volume scheme, otherwise
       ``"quick"``.
   * - ``theta_min``
     - ``0.01``
     - Lower clamp on the cut-face water fraction :math:`\theta`, bounding
       the :math:`1/\theta` coefficient when the interface grazes a face.
   * - ``band_cells``
     - ``4``
     - Narrow-band half-width in cells. Informational only — the reinit /
       extend sweeps are applied globally.
   * - ``reinit_iters``
     - ``4``
     - Reinitialisation sub-steps per :meth:`reinitialize` call.
   * - ``reinit_every``
     - ``5``
     - Reinitialise every N solver steps (``0`` disables).
   * - ``extend_iters``
     - ``4``
     - Band velocity-extension sub-steps per :meth:`extend_velocity` call
       (used only when ``extend_full`` is ``False``).
   * - ``extend_every``
     - ``1``
     - Extend every N solver steps (``0`` disables).
   * - ``extend_full``
     - ``True``
     - Extend the water velocity across the **whole** air region each step
       (recommended — keeps the decoupled bulk air from accumulating
       spurious vorticity). ``False`` restricts to an ``extend_iters``
       band (cheaper, but the bulk air free-runs).

The ``phi_init`` lambda uses the same evaluation convention as
``body.sdf`` (a callable or a string ``eval``\ ed against ``{"torch":
torch}``). The same block works in 3-D — pass a three-argument lambda;
2-D vs 3-D is auto-detected from the presence of ``Nz`` in the grid.


Validation: hydrostatic column
------------------------------

``lilytorch/validation/free_surface_2d/run_hydrostatic.py`` exercises the
full path on a closed box of water with the interface at :math:`y = 0.5`
under downward gravity. At steady state the expected solution
(gauge :math:`p_{\text{atm}} = 0`) is

.. math::

   \mathbf{u} \approx \mathbf{0},
   \qquad
   p(y) =
   \begin{cases}
     \rho\,g\,(H - y) & y < H \ \text{(water, linear)} \\[4pt]
     0                & y > H \ \text{(air)}
   \end{cases}

The script reports :math:`|\mathbf{u}|_{\max}`, the max deviation from
the hydrostatic reference inside the water, and the max pressure in the
air, and passes when the relative pressure error is below 5 % and the
residual velocity is below 5 % of the gravity-wave speed
:math:`\sqrt{gH}`. It runs CPU-only in ``float64`` and is a good
first check when debugging the free-surface plumbing.

The unit-level self-tests in
``lilytorch/src/test_free_surface.py`` cover each operation
(advection, reinitialisation, velocity extension, GFM face scales,
pressure mask) against analytic references::

   python -m lilytorch.src.test_free_surface


References
----------

* S. Osher and J. A. Sethian, *Fronts propagating with
  curvature-dependent speed*, J. Comput. Phys. **79** (1988).
* M. Sussman, P. Smereka and S. Osher, *A level set approach for
  computing solutions to incompressible two-phase flow*, J. Comput. Phys.
  **114** (1994).
* R. P. Fedkiw, T. Aslam, B. Merriman and S. Osher, *A non-oscillatory
  Eulerian approach to interfaces in multimaterial flows (the ghost fluid
  method)*, J. Comput. Phys. **152** (1999).
* F. Gibou, R. Fedkiw, L.-T. Cheng and M. Kang, *A second-order-accurate
  symmetric discretization of the Poisson equation on irregular domains*,
  J. Comput. Phys. **176** (2002).
