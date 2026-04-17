.. _boundary_conditions:

Boundary Conditions
===================

LilyTorch supports **Dirichlet** and **Neumann** boundary conditions on
each face of the rectangular domain.  This page explains how they are
specified, ordered, and enforced.

.. contents:: On this page
   :local:
   :depth: 2


Overview
--------

Boundary conditions are set *independently* for each velocity component
(:math:`u`, :math:`v`, and :math:`w` in 3-D) on each face of the
domain.  They are specified in the ``boundary_conditions`` section of the
YAML config.


Face ordering convention
------------------------

Faces are enumerated in **axis-minor** order — low face first, then high
face, for each spatial dimension:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Index
     - Face
     - Location
   * - 0
     - :math:`x_{\min}`
     - Left boundary (``dim 0, low``)
   * - 1
     - :math:`x_{\max}`
     - Right boundary (``dim 0, high``)
   * - 2
     - :math:`y_{\min}`
     - Bottom boundary (``dim 1, low``)
   * - 3
     - :math:`y_{\max}`
     - Top boundary (``dim 1, high``)
   * - 4
     - :math:`z_{\min}`
     - Back boundary (``dim 2, low``) — 3-D only
   * - 5
     - :math:`z_{\max}`
     - Front boundary (``dim 2, high``) — 3-D only

2-D simulations use indices 0–3; 3-D simulations use indices 0–5.


BC types
--------

Dirichlet (``"D"``)
^^^^^^^^^^^^^^^^^^^

The field is fixed to the prescribed value at the boundary.
Implementation uses the **image/mirror method** on the ghost cell:

.. math::

   \phi_{\text{ghost}} = 2\,\phi_{\text{bc}} - \phi_{\text{interior}}

This achieves second-order accuracy: the value at the cell face (midway
between ghost and interior) is exactly :math:`\phi_{\text{bc}}`.


Neumann (``"N"``)
^^^^^^^^^^^^^^^^^

The normal gradient of the field at the boundary is zero:

.. math::

   \frac{\partial\phi}{\partial n}\bigg|_{\text{boundary}} = 0

This is implemented by copying the interior cell into the ghost cell:

.. math::

   \phi_{\text{ghost}} = \phi_{\text{interior}}


YAML specification
------------------

Each velocity component has a ``BC_type_*`` list (strings) and a
``BC_values_*`` list (floats), both following the face ordering above.

.. code-block:: yaml
   :caption: 2-D example — uniform inflow, zero-gradient outflow

   boundary_conditions:
     BC_type_u:   ["D", "N", "D", "D"]
     BC_values_u: [1.0, 0.0, 1.0, 1.0]
     BC_type_v:   ["D", "N", "D", "D"]
     BC_values_v: [0.0, 0.0, 0.0, 0.0]

.. code-block:: yaml
   :caption: 3-D example — inflow on x_min, free-slip on y/z

   boundary_conditions:
     BC_type_u:   ["D", "N", "N", "N", "N", "N"]
     BC_values_u: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     BC_type_v:   ["D", "N", "D", "D", "D", "D"]
     BC_values_v: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     BC_type_w:   ["D", "N", "D", "D", "D", "D"]
     BC_values_w: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


Common configurations
---------------------

Uniform inflow / convective outflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :math:`x_{\min}`: Dirichlet :math:`u = U_\infty`,  :math:`v = 0`.
* :math:`x_{\max}`: Neumann (zero-gradient) — approximates a convective
  outflow for moderately-long domains.
* :math:`y` walls: Dirichlet :math:`u = U_\infty`, :math:`v = 0`
  (free-stream) or :math:`u = 0`, :math:`v = 0` (no-slip walls).

Channel flow (no-slip walls)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :math:`x`: periodic (not yet exposed via YAML; use Neumann as a proxy
  for long domains) or Dirichlet inflow / Neumann outflow.
* :math:`y`: Dirichlet :math:`u = 0`, :math:`v = 0` on both walls.

Free-swimming (free-space)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For truly unbounded problems, use the **free-space Poisson solver**
(``poisson_bc_type: free``) with Neumann velocity BCs on all faces and a
sufficiently large domain.

.. tip::

   When using the free-space Poisson solver, set all velocity BCs to
   Neumann (``"N"``) with value 0 so that no artificial momentum is
   injected at the boundaries.


Poisson boundary conditions
----------------------------

The Poisson equation for pressure always uses **Neumann** conditions
(:math:`\partial p / \partial n = 0`) on all domain faces.  This is
handled internally:

* **FFT Neumann** — enforced by the DCT basis functions.
* **FFT free-space** — naturally satisfied by the Green's function
  convolution (zero-padded domain).
* **Multigrid** — ghost-cell mirroring identical to the velocity Neumann
  implementation.

You cannot set Dirichlet pressure conditions from the YAML config.  For
specialised pressure BCs, modify the solver source directly.


Immersed-body boundary conditions
-----------------------------------

Bodies embedded in the domain impose **no-slip** or **prescribed-velocity**
conditions through the BDIM2 meta-equation (see :doc:`immersed_boundary`),
*not* through the ghost-cell mechanism described above.  The BDIM blending
smoothly forces :math:`\mathbf{u} \to \mathbf{v}_{\text{body}}` inside
the body envelope without requiring the flow grid to conform to the body
surface.
