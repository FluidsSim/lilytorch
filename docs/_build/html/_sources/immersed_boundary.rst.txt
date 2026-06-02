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

The pressure Poisson equation is modified to account for the body mask.
For a single-density flow the BDIM2 projection solves

.. math::
   :label: bdim-proj

   \nabla\cdot\!\left(c\,\nabla p\right) = \nabla\cdot\tilde{\mathbf{u}},
   \qquad
   c = \frac{w\,\Delta t}{\rho}\,\mu_0 ,
   \qquad
   \mathbf{u}^{n+1} = \tilde{\mathbf{u}} - c\,\nabla p .

The :math:`\mu_0` factor in the coefficient :math:`c` makes the velocity
correction :math:`c\,\nabla p` **vanish inside the body** (:math:`\mu_0=0`),
so the no-slip body velocity imposed by the BDIM meta-equation is
preserved exactly. This :math:`\mu_0`-weighting is controlled by the
``bdim_mu0_projection`` flag (default ``True``).

Variable-density (FSI) projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the body density :math:`\rho_b` differs from the fluid density
:math:`\rho_f` (rigid-body sedimentation, FARMS coupling), the projection
uses the **effective density** blended across the interface band,

.. math::

   \rho_{\text{eff}}(\mathbf{x}) = \rho_b + (\rho_f - \rho_b)\,\mu_0(\mathbf{x}),
   \qquad
   c = \frac{w\,\Delta t}{\rho_{\text{eff}}}\,\mu_0 ,

so that the coefficient is :math:`w\Delta t/\rho_f` in the fluid
(:math:`\mu_0=1`) and tends to :math:`0` inside the body
(:math:`\mu_0\to0`). Set ``rho_body`` to the body density; it should match
the body's morphology density (the value that sets its MuJoCo mass), or
the body will be over/under-buoyant.

.. _fft-vs-multigrid-projection:

Multigrid vs FFT: why the RHS divisor must stay bounded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two Poisson backends treat the variable coefficient :math:`c`
differently:

* **Multigrid / MGCG** (``poisson_method: multigrid`` | ``mgcg``) solves
  the *variable-coefficient* operator :eq:`bdim-proj` directly — the
  smoother uses the face coefficients :math:`c` on both sides, so cells
  with :math:`c\to0` simply decouple. This is robust for any
  :math:`\mu_0`-weighting.

* **FFT** (``poisson_method: fft``) can only solve a *constant-coefficient*
  Laplacian :math:`\nabla^2 p = f`. The variable coefficient is therefore
  moved to the right-hand side as a **division**,

  .. math::

     \nabla^2 p = \frac{\nabla\cdot\tilde{\mathbf{u}}}{c_{\text{rhs}}} ,
     \qquad
     \mathbf{u}^{n+1} = \tilde{\mathbf{u}} - c\,\nabla p .

  If the :math:`\mu_0`-weighted coefficient :math:`c=(w\Delta t/\rho_{\text{eff}})\,\mu_0`
  is used as the RHS divisor :math:`c_{\text{rhs}}`, then :math:`c_{\text{rhs}}\to0`
  in the immersed-boundary band and :math:`\nabla\cdot\tilde{\mathbf{u}}/c_{\text{rhs}}`
  becomes **singular** there — the constant-coefficient solver cannot
  represent it, and the pressure (and velocity) blow up exponentially
  (observed as a :math:`\sim\!10^5\times` per-step growth that aborts
  coupled runs after a few iterations).

  **Fix.** For the FFT path the RHS divisor uses the **bounded** scalar
  fluid coefficient :math:`c_{\text{rhs}} = w\,\Delta t/\rho`, while the
  :math:`\mu_0`-weighted :math:`c` is applied **only in the velocity
  correction** :math:`\mathbf{u}^{n+1}=\tilde{\mathbf{u}}-c\,\nabla p`.
  The correction therefore still vanishes inside the body (preserving the
  no-slip velocity), but the Poisson RHS stays well posed.
  ``flow_past_circle`` and other single-phase cases are unaffected
  (:math:`c` is a scalar there). This fix lives in
  :meth:`FluidSolver.project` and applies to both 2-D and 3-D.

When ``zero_pressure_inside`` is enabled, :math:`p = 0` is explicitly
enforced in cells where :math:`\mu_0 < \text{threshold}`.


Body hierarchy
--------------

Only **composite bodies** are user-instantiable. The
:func:`~lilytorch.src.body.body_from_yaml` factory accepts the
``composite_analytical`` and ``composite_mesh`` body types; the other
classes are internal building blocks that exist to be wrapped by a
composite. The class hierarchy in :mod:`lilytorch.src.body` is:

.. code-block:: text

   Body (base)                     # mu_funcs, compute_normals
   ├── BodyAnalytical              # internal: analytical SDFs
   │   ├── CompositeBodyAnalytical # USER: union of analytical bodies
   │   ├── BodyFishAnalytical      # internal: NACA-foil fish spine
   │   └── BodyFishExperimental    # internal: experimental kinematics
   ├── BodyMesh                    # internal: mesh-based SDF
   │   ├── CompositeBodyMesh       # USER: union of mesh bodies
   │   └── CompositeSegmentBody    # internal: articulated segments
   └── MultiAnimatBodies           # internal: multiple swimming animats

Each body exposes:

* ``compute_sdf(x, y [, z])`` — evaluate the SDF on the grid.
* ``compute_velocity(x, y [, z])`` — body-velocity field for BDIM.
* ``update(t, iteration)`` — advance kinematics.

Composite bodies take the **union** of their children via
:math:`d_{\text{union}} = \min(d_1, d_2, \dots)`.


.. _velocity-blend:

Overlapping links: smooth velocity blend
-----------------------------------------

Articulated swimmers built from per-link meshes often have **overlapping
links** — most commonly because ``convexify=True`` replaces each link
mesh with its convex hull, which inflates the link and makes adjacent
hulls intersect (for the 9-link 1guilla this is ≈16 % of each link's
volume even in the straight pose, and more on the concave side of a
bend).

The geometry union :math:`d_{\text{union}} = \min_i d_i` handles the
*shape* correctly, but the **imposed solid velocity** is ambiguous in the
overlap band: two rigid links generally have different velocities
:math:`\mathbf{v}_i = \mathbf{v}^{\text{lin}}_i + \boldsymbol{\omega}_i
\times (\mathbf{x}-\mathbf{x}^{\text{com}}_i)` there. The legacy
*winner-take-all* rule assigns each face the velocity of whichever link
has the most-negative SDF. Where ownership switches, the imposed velocity
**jumps**, which injects a grid-scale divergence the projection cannot
remove — producing a pressure spike pinned at the seam. In the
explicit FARMS coupling this corrupts the per-link forces and, for a
fast undulating body, drives a per-step feedback blow-up (it is *not* a
CFL/time-step instability — halving ``dt`` does not help; only damping
the force feedback, e.g. ``force_relaxation`` < 1, masks it).

The fix replaces the hard switch with a **smooth SDF-weighted blend** of
the imposed velocity (the geometry SDF still uses the running-min):

.. math::

   \mathbf{v}_{\text{body}}(\mathbf{x}) =
   \frac{\sum_i w_i\,\mathbf{v}_i}{\sum_i w_i},
   \qquad
   w_i = \sigma\!\left(-\,d_i/\varepsilon_w\right),

where :math:`\sigma` is the logistic function. The weight :math:`w_i` is
≈1 deep inside link :math:`i`, :math:`0.5` on its surface and decays to 0
a band :math:`\varepsilon_w` outside. This is **continuous** across the
seam (it equals :math:`\tfrac12(\mathbf{v}_A+\mathbf{v}_B)` exactly where
:math:`d_A=d_B`) and is **exact for a single, non-overlapping body**
(:math:`\mathbf{v}_{\text{body}} = \mathbf{v}_i`), so it has no effect
away from overlaps.

Enable it with the config parameter
``body_velocity_blend_eps_cells`` (band width :math:`\varepsilon_w` in
grid cells; ``None``/``0`` keeps the legacy winner-take-all). A value of
``2.0`` matches the BDIM kernel half-width and is recommended:

.. code-block:: python

   self.convexify                     = True
   self.body_velocity_blend_eps_cells = 2.0   # smooth seam blend
   # force_relaxation no longer needed

The blend is implemented identically in the pure-PyTorch union
(:meth:`BDIMhandler._update_2d` / ``_update_3d``) and in the fused
CUDA/C++ streaming-SDF kernels (``streaming_sdf*.cu`` /
``streaming_sdf_cpu*.cpp``): the per-body reduction accumulates
:math:`\sum_i w_i \mathbf{v}_i` and :math:`\sum_i w_i` alongside the
existing running-min, and the decode step divides.

.. warning::

   The blend **reduces** the overlap-induced divergence (it is a smoothed
   average of two rigid fields, so :math:`\nabla\!\cdot\mathbf{v}_{\text{body}}
   = \nabla w \cdot (\mathbf{v}_A-\mathbf{v}_B) \neq 0`, but with a much
   smaller magnitude than the hard jump) — it does **not** make the imposed
   velocity divergence-free.  For **heavy** overlap (e.g. ``convexify=True``
   convex hulls, ~16 %+ per link) on a **fine production grid** it is *not*
   sufficient on its own: the residual divergence still drives the velocity
   past the (tighter) CFL limit. In practice it only *delays* the blow-up,
   and combining it with ``force_relaxation`` < 1 delays it further but does
   not guarantee stability. The blend is enough on coarser grids and for
   *mild* overlap.

   **The robust fix for overlapping links is to remove the overlap at the
   source: ``convexify=False``** (use the raw watertight collision meshes,
   which do not overlap). This is stable with no blend and no force
   relaxation. Use the velocity blend only when overlap is unavoidable and
   mild. (Verified at production resolution on the zebrafish swimmer:
   ``convexify=False`` is stable indefinitely; ``convexify=True`` blows up
   with the blend alone, with ``force_relaxation`` alone, and — later — with
   both.)


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

The kernel-mode optimisation bundles five composable pieces, all of
which are activated together by a single user-facing switch:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Effect when ``solver.use_kernels = true``
   * - shared-stress union crop
     - Shared viscous-stress tensor used by the hydrodynamic-force
       integrator runs as one compiled kernel over the union AABB,
       instead of one full-grid kernel per body.
   * - mu/normals union crop
     - :math:`\mu_0`, :math:`\mu_1` and the outward normals needed by
       the BDIM meta-equation and by the variable-density Poisson
       coefficients are computed only on union cells.
   * - BDIM-meta union crop
     - The BDIM2 meta-equation itself (Eq. :eq:`bdim-meta`), applied to
       each face-velocity component only on union cells.
   * - streaming SDF kernels (2-D and 3-D)
     - Per-body fused C++/CUDA kernels (rotate + sample + running-min
       + sparse-cc store) replace the Python per-body loop in
       ``BDIMhandler._update_{2,3}d``.
   * - streaming fused-force kernels (Phase D, 2-D and 3-D)
     - Per-body force/torque integration is folded into the same
       streaming op, removing the dense ``(B, Nx, Ny, Nz)`` SDF
       reduction.

These five pieces are no longer individually configurable — they are
all enabled together by ``solver.use_kernels = true`` (the default)
and disabled together by ``solver.use_kernels = false`` (which selects
the suboptimal pure-PyTorch reference path).  ``use_kernels`` is
**independent** of ``solver.use_gpu``: kernel mode is supported on
both CPU and CUDA, and pure-Python mode also runs on either device.

.. code-block:: yaml

   solver:
     use_kernels: true
     # ... rest of the solver config

or, in the cost-analysis harness,

.. code-block:: bash

   python run_multigrid_cost_analysis.py --preset full --use_kernels

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

* **Streaming fused-CUDA SDF update** — when ``solver.use_kernels`` is
  on, one C++/CUDA kernel per body fuses rotate + 4 trilinear samples +
  4 running-min updates + per-body sparse cc store into a single op
  call, replacing the Python per-body loop in ``BDIMhandler._update_3d``.

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

