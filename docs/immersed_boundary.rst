.. _immersed_boundary:

Immersed Boundary Method — BDIM2
=================================

LilyTorch implements the **Boundary Data Immersion Method** (BDIM,
Weymouth & Yue 2011, Maertens & Weymouth 2015) to handle arbitrarily
shaped bodies on a fixed Cartesian grid without body-conforming meshing.

.. contents:: On this page
   :local:
   :depth: 2


Signed-distance functions (SDF)
-------------------------------

Every body is represented by a **signed-distance function** :math:`d(\mathbf{x})`:

* :math:`d > 0` — fluid region.
* :math:`d = 0` — body surface.
* :math:`d < 0` — body interior.

Analytical SDFs
^^^^^^^^^^^^^^^^

The following analytical shapes are built in:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Shape
     - SDF expression
   * - **circle** / sphere
     - :math:`d = \|\mathbf{x} - \mathbf{x}_c\| - R`
   * - **box**
     - :math:`d = \max(|x_i - c_i| - b_i)` for each half-extent :math:`b_i`
   * - **capsule / segment**
     - :math:`d = \|\mathbf{x} - \text{closest-point-on-segment}\| - R`

Mesh-based SDFs
^^^^^^^^^^^^^^^^

For complex 3-D geometries (STL/OBJ), the SDF is computed in a
pre-processing step:

1. Closest-point queries via **Open3D** ray-casting.
2. Sign determination from winding numbers.
3. Fast-marching (**scikit-fmm**) to extend the distance field.
4. Staggered-grid projection onto :math:`u`, :math:`v`, :math:`w` face grids.


Kernel functions
-----------------

The BDIM2 meta-equation uses two smooth kernel functions constructed from
the SDF:

**Smoothed Heaviside** :math:`\mu_0`:

.. math::

   \mu_0(d) =
   \begin{cases}
     0 & d \le -\varepsilon \\[4pt]
     \displaystyle
     \frac{1}{2}\!\left(
       1 + \frac{d}{\varepsilon}
       + \frac{\sin(\pi d/\varepsilon)}{\pi}
     \right)
     & |d| < \varepsilon \\[4pt]
     1 & d \ge \varepsilon
   \end{cases}

**First-moment kernel** :math:`\mu_1`:

.. math::

   \mu_1(d) =
   \varepsilon\!\left(
     \frac{1}{4}
     - \left(\frac{d}{2\varepsilon}\right)^{\!2}
     - \frac{\sin(\pi d/\varepsilon) + (1+\cos(\pi d/\varepsilon))/\pi}
            {2\pi}
   \right)

The **kernel half-width** is :math:`\varepsilon = 1.5\,h` where :math:`h`
is the grid spacing.  The ``suit`` body parameter can shift the effective
SDF origin.


BDIM2 meta-equation
---------------------

At each Runge–Kutta sub-step, the provisional velocity :math:`\phi` is
blended with the body velocity :math:`\mathbf{v}_b` via:

.. math::
   :label: bdim-meta

   \phi_{\text{out}}
     = \mu_0\,(\phi - \mathbf{v}_b)
     + \mathbf{v}_b
     + \mu_1\,\hat{\mathbf{n}}\cdot\nabla(\phi - \mathbf{v}_b)

where :math:`\hat{\mathbf{n}} = \nabla d / |\nabla d|` is the outward
unit normal.

This achieves:

* **Inside the body** (:math:`d \ll -\varepsilon`):
  :math:`\mu_0 = 0,\; \mu_1 = 0` → :math:`\phi_{\text{out}} = \mathbf{v}_b`.
* **In the fluid** (:math:`d \gg \varepsilon`):
  :math:`\mu_0 = 1,\; \mu_1 = 0` → :math:`\phi_{\text{out}} = \phi`.
* **In the transition layer** (:math:`|d| < \varepsilon`):
  smooth blending with a normal-derivative correction that enforces the
  correct boundary-layer profile to second order.


Pressure projection with BDIM
-------------------------------

The pressure Poisson equation is modified to account for the body mask:

.. math::

   \nabla\cdot\!\left(\frac{w\,\Delta t}{\rho}\,\mu_0\,\nabla p\right)
   = \nabla\cdot\tilde{\mathbf{u}}

The face coefficients for the multigrid solver are
:math:`c = (w\,\Delta t / \rho)\,\mu_0`, ensuring zero pressure gradient
inside the body.

When ``zero_pressure_inside`` is enabled, :math:`p = 0` is explicitly
enforced in cells where :math:`\mu_0 < \text{threshold}`.


Body hierarchy
--------------

The class hierarchy in :mod:`lilytorch.src.body` is:

.. code-block:: text

   Body (base)                     # mu_funcs, compute_normals
   ├── BodyAnalytical              # Analytical SDFs
   │   ├── CompositeBodyAnalytical # Union of analytical bodies
   │   ├── BodyFishAnalytical      # NACA-foil fish spine
   │   └── BodyFishExperimental    # Experimental kinematics replay
   ├── BodyMesh                    # Mesh-based SDF
   │   ├── CompositeBodyMesh       # Union of mesh bodies
   │   └── CompositeSegmentBody    # Articulated segments
   └── MultiAnimatBodies           # Multiple swimming animats

Each body exposes:

* ``compute_sdf(x, y [, z])`` — evaluate the SDF on the grid.
* ``compute_velocity(x, y [, z])`` — body-velocity field for BDIM.
* ``update(t, iteration)`` — advance kinematics.

Composite bodies take the **union** of their children via
:math:`d_{\text{union}} = \min(d_1, d_2, \dots)`.


FARMS coupling (3-D)
---------------------

In coupled mode (:class:`~lilytorch.integration.BDIMhandler.BDIMhandler`),
each MuJoCo link is an independent body.  At every time-step:

1. **Read** link poses from MuJoCo.
2. **Recompute** per-link SDFs, take the union.
3. **Run** the fluid step (adv-diff → BDIM → project).
4. **Integrate** hydrodynamic forces on each link.
5. **Write** forces back to ``physics.data.xfrc_applied``.

An **AABB narrow-band optimisation** avoids evaluating the SDF over the
full grid for links that occupy less than 90 % of the domain.


Performance: union narrow-band
-------------------------------

Articulated swimmers (e.g. the 9-link 1guilla, salamander or pleurodeles)
evaluate *many* per-link SDFs, normals and BDIM kernels on every
fractional step.  For realistic grid sizes (:math:`\sim 10^6`–:math:`10^7`
cells) the bodies only touch a tiny fraction of cells, yet the naive
implementation touches the **full grid** for every link and for every
operator.  This makes the whole pipeline **memory-bandwidth bound** and
dominated by **kernel-launch overhead** — exactly the regime where GPUs
perform worst.

LilyTorch therefore implements a *union narrow-band* strategy that is
activated by a single user flag and covers four different operators.

Motivation
^^^^^^^^^^

For :math:`N_b` bodies on a grid of :math:`N` cells the naive per-body
pipeline has cost

.. math::

   C_{\text{naive}}
     = \mathcal{O}\!\bigl(N_b\,N\bigr)
       \;+\;
       \mathcal{O}\!\bigl(N_b\,N_{\text{ops}}\bigr)\;\text{kernel launches.}

In practice every body lies in a small axis-aligned bounding box (AABB),
and the *union* of all per-body AABBs, padded by a halo of a few cells,
still covers only a small sub-block :math:`N_\cup \ll N` of the domain.
Restricting the bandwidth-bound operators to the union sub-block and
**batching** all bodies into a single launch converts the cost to

.. math::

   C_{\text{opt}}
     = \mathcal{O}\!\bigl(N_\cup\bigr)
       \;+\;
       \mathcal{O}\!\bigl(N_{\text{ops}}\bigr)\;\text{kernel launches.}

The speed-up therefore grows with both :math:`N_b` (launch-overhead
amortisation) and :math:`N/N_\cup` (memory-traffic amortisation), so the
approach is *most* useful for **many moving bodies** and **large grids**
— precisely the scenarios that LilyTorch targets.

Components
^^^^^^^^^^

The optimisation is split into four composable pieces, each of which can
be toggled independently:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - What it restricts to the union AABB
   * - ``solver.forces_shared_union``
     - Shared viscous-stress tensor used by the hydrodynamic-force
       integrator (one compiled kernel over the union, instead of one
       full-grid kernel per body).
   * - ``solver.mu_normals_union``
     - :math:`\mu_0`, :math:`\mu_1` and the outward normals needed by the
       BDIM meta-equation and by the variable-density Poisson
       coefficients.
   * - ``solver.bdim_union``
     - The BDIM2 meta-equation itself (Eq. :eq:`bdim-meta`), applied to
       each face-velocity component only on union cells.

The three union-crop flags above are independent toggles.  The
production cost-analysis pipeline (``run_scaling_conditions_pipeline.py``)
combines them with the streaming fused-CUDA SDF / forces flags:

.. code-block:: yaml

   solver:
     force_shared_union: true
     mu_normals_union: true
     bdim_union: true
     force_narrow_batch: true
     streaming_sdf_3d: true
     streaming_forces_3d: true

or, in the cost-analysis harness,

.. code-block:: bash

   python run_multigrid_cost_analysis.py --preset full \
       --force_shared_union --mu_normals_union --bdim_union \
       --force_narrow_batch --streaming_sdf_3d --streaming_forces_3d

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

* **Union AABB construction** — per-link AABBs in world coordinates are
  expanded by a halo of 2 cells (so stencil operators remain in bounds)
  and their outer envelope is rounded up to a **bucket of 16** cells per
  axis.  The bucket rounding caps the number of distinct sub-block
  shapes that ``torch.compile`` has to specialise for, keeping recompile
  counts bounded while the swimmer moves.

* **Dynamic-shape compilation** — the union-crop kernels are compiled
  with ``dynamic=True`` instead of ``mode="reduce-overhead"``, because
  the sub-block shape changes over time.  The static, full-grid kernels
  (adv-diff, Poisson) keep the reduce-overhead mode.

* **Streaming fused-CUDA SDF update** — when ``streaming_sdf_3d`` is on,
  one C++/CUDA kernel per body fuses rotate + 4 trilinear samples + 4
  running-min updates + per-body sparse cc store into a single op call,
  replacing the Python per-body loop in ``BDIMhandler._update_3d``.

Measured impact
^^^^^^^^^^^^^^^

On an RTX 4080 SUPER, for the 9-link 1guilla free-swimming benchmark:

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 18

   * - Grid
     - Body update
     - Forces
     - BDIM meta
   * - 256 × 64 × 64
     - 8.24 → 2.93 ms  (**2.8×**)
     - 2.3× faster
     - 6–11× faster
   * - 512 × 128 × 128
     - 8.4 → 3.39 ms  (**2.5×**)
     - 2.3× faster
     - 6–11× faster

Because the three operator flags share the *same* union AABB, their
savings are additive: enabling them together eliminates most of the
kernel-launch overhead that previously scaled linearly in :math:`N_b`.

When to use it
^^^^^^^^^^^^^^

The meta-flag is enabled by default on the supplied multi-grid cost
benchmarks and on all articulated FARMS cases.  You may want to
**disable** it only when:

* A single body occupies a large fraction of the domain
  (:math:`N_\cup \approx N`), in which case the overhead of AABB
  bookkeeping is not recovered.
* You are debugging a new analytical SDF and prefer full-grid kernels
  for numerical inspection.

For every other scenario — and in particular for long multi-body
swimming simulations on fine grids — the production flag set above
should be on.

