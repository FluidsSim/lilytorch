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
