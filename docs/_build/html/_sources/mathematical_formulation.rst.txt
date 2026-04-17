.. _mathematical_formulation:

Mathematical Formulation
========================

This page describes the continuous equations solved by LilyTorch.  For
discrete stencils and time-stepping details see :doc:`numerical_schemes`.

.. contents:: On this page
   :local:
   :depth: 2


Governing equations
-------------------

LilyTorch solves the **incompressible Navier–Stokes equations** in the
velocity–pressure formulation:

.. math::
   :label: momentum

   \frac{\partial \mathbf{u}}{\partial t}
     + (\mathbf{u}\cdot\nabla)\mathbf{u}
   = -\frac{1}{\rho}\,\nabla p
     + \nu\,\nabla^{2}\mathbf{u}

.. math::
   :label: continuity

   \nabla\cdot\mathbf{u} = 0

where

* :math:`\mathbf{u}=(u,v)` (2-D) or :math:`(u,v,w)` (3-D) is the velocity field,
* :math:`p` is the pressure,
* :math:`\rho` is the (constant) fluid density,
* :math:`\nu` is the kinematic viscosity.


Fractional-step pressure projection
------------------------------------

The divergence-free constraint :eq:`continuity` is enforced by a
**pressure-projection** (Chorin–Témam) step.  After computing a
provisional velocity :math:`\tilde{\mathbf{u}}` that satisfies the
momentum equation without the pressure gradient, a Poisson equation is
solved for the pressure correction:

.. math::
   :label: poisson

   \nabla\cdot\!\left(\frac{w\,\Delta t}{\rho}\,\mu_0\,\nabla p\right)
     = \nabla\cdot\tilde{\mathbf{u}}

and the velocity is corrected:

.. math::
   :label: projection

   \mathbf{u}^{n+1}
     = \tilde{\mathbf{u}}
     - \frac{w\,\Delta t}{\rho}\,\mu_0\,\nabla p

The weight :math:`w` depends on the Runge–Kutta stage
(:math:`w=1` in the predictor, :math:`w=\tfrac12` in the corrector; see
:ref:`heun-scheme`).

The mask function :math:`\mu_0` equals 1 in the fluid and smoothly
transitions to 0 inside immersed bodies (see :doc:`immersed_boundary`).
In the absence of bodies, :math:`\mu_0\equiv 1` and equation :eq:`poisson`
reduces to the standard constant-coefficient Poisson equation.


Poisson equation — FFT approach (Neumann)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When all domain boundaries carry **Neumann** conditions
(:math:`\partial p/\partial n = 0`), a spectral solve is used via the
**Discrete Cosine Transform** (DCT-II, Makhoul 1980).

The eigenvalues of the discrete Laplacian under Neumann BCs are

.. math::

   \lambda_k = \frac{2}{h^{2}}
     \!\left(\cos\!\left(\frac{\pi k}{N}\right) - 1\right),
   \qquad k = 0, 1, \dots, N-1

The solution is: DCT-II :math:`\to` divide by eigenvalues :math:`\to` inverse
DCT-II, yielding spectral accuracy for a smooth right-hand side.


Poisson equation — FFT approach (free-space)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For **unbounded** (free-space) domains the solver computes the discrete
convolution of the right-hand side with a regularised Green's function via
zero-padded FFTs.

**2-D Green's function** (Hejlesen et al. 2013, 8th-order algebraic
smoothing):

.. math::

   G(r) =
   \frac{1}{2\pi}
   \begin{cases}
     \displaystyle
     -\tfrac{1}{2}\ln(r^{2}+\sigma^{2})
     + \sum_{k=1}^{4} \frac{1}{2k}
       \frac{\sigma^{2k}}{(r^{2}+\sigma^{2})^{k}}
     & r > 0 \\[6pt]
     \displaystyle
     -\tfrac{1}{2}\ln(\sigma^{2})
     + \sum_{k=1}^{4} \frac{1}{2k}
     & r = 0
   \end{cases}

**3-D Green's function** (Gaussian/erf regularisation):

.. math::

   G(r) = -\frac{1}{4\pi r}\,
     \operatorname{erf}\!\left(\frac{r}{\sqrt{2}\,\sigma}\right)

where :math:`\sigma = h` is the grid spacing.


Poisson equation — variable-coefficient (multigrid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When immersed bodies are present (:math:`\mu_0 \neq 1`), the Poisson
equation :eq:`poisson` becomes **variable-coefficient**:

.. math::

   \nabla\cdot(c\,\nabla p) = f

Discrete stencil along dimension :math:`d`:

.. math::

   \frac{c_{d+}\,p_{i+1}
       - (c_{d+}+c_{d-})\,p_{i}
       + c_{d-}\,p_{i-1}}{h^{2}}

where face coefficients :math:`c_{d\pm}` are the harmonic averages of
:math:`\mu_0` evaluated at cell faces.  This is solved by geometric
multigrid V-cycles or a multigrid-preconditioned conjugate-gradient (MGCG)
iteration (see :ref:`multigrid-solver`).


Variable-density pressure projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In coupled FARMS simulations the body may have a different density
:math:`\rho_b` from the fluid :math:`\rho_f`.  The effective density is

.. math::

   \rho(\mathbf{x})
     = \rho_b + (\rho_f - \rho_b)\,\mu_0(\mathbf{x})

and the face-centred Poisson coefficients become
:math:`c_u = \Delta t / \rho_u`, etc.


Advection–diffusion
--------------------

The advection–diffusion sub-step advances each velocity component by one
Forward-Euler step:

.. math::

   u^{*} = u^{n} + \Delta t
     \left[
       -\nabla\cdot(\mathbf{u}\otimes u)
       + \nu\,\nabla^{2} u
     \right]

The CFL stability condition is

.. math::

   \Delta t
     \le \frac{\min(h)}{v_{\max} + 3\nu}


Force computation
-----------------

Hydrodynamic forces on immersed bodies are computed by integrating
the stress tensor over the body surface using smoothed delta functions.

**Viscous forces** (contour integral):

.. math::

   \mathbf{F}_{\text{visc}}
     = \int (\boldsymbol{\sigma}\cdot\hat{\mathbf{n}})\;
       \delta_{\varepsilon}(d - \varepsilon)\,\mathrm{d}V

**Pressure forces**:

.. math::

   \mathbf{F}_{\text{pres}}
     = -\int p\,\hat{\mathbf{n}}\;
       \delta_{\varepsilon}(d)\,\mathrm{d}V

where the **smoothed delta function** is

.. math::

   \delta_{\varepsilon}(d)
     = \frac{1+\cos(\pi d/\varepsilon)}{2\varepsilon}

Torque decomposition (avoids constructing the full moment-arm tensor):

.. math::

   \tau_x = \sum(Y\,f_z)\,h^{3}
          - \mathrm{com}_y\,F_z
          - \sum(Z\,f_y)\,h^{3}
          + \mathrm{com}_z\,F_y


Buoyancy (FARMS coupling)
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the coupled MuJoCo mode, buoyancy on each link is computed as

.. math::

   F_{z,\text{buoy}}
     = -\rho_{\text{water}}\,
       \frac{m_{\text{link}}}{\rho_{\text{body}}}\,g
       \cdot
       \min\!\left(
         \frac{z_{\text{surface}} + h_{\text{link}} - z_{\text{link}}}
              {2\,h_{\text{link}}},\;1
       \right)
