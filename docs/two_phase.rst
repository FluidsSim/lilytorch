.. _two_phase:

Two-Phase Free Surface (Water + Air)
====================================

LilyTorch models a water–air free surface as a **two-phase** flow: a *single*
set of incompressible Navier–Stokes equations with a **spatially varying
density and viscosity**. The air is a real (light) fluid, so a body sitting at
the waterline gets a genuine pressure reaction — the regime needed for
**partially-submerged / floating bodies** (boats, buoyant swimming robots,
paddling animals). The implementation lives in
:class:`~lilytorch.src.two_phase.TwoPhase` (the interface + material fields) and
:class:`~lilytorch.src.two_phase_solver.TwoPhaseSolver` (a thin subclass of
:class:`~lilytorch.src.solver.FluidSolver`).

.. contents:: On this page
   :local:
   :depth: 2


Model
-----

The interface is captured by a **Volume-of-Fluid (VOF)** fraction
:math:`\alpha(\mathbf{x},t)\in[0,1]`:

.. math::

   \alpha = 1 \ \text{(water)}, \qquad
   \alpha = 0 \ \text{(air)}, \qquad
   0 < \alpha < 1 \ \text{(interface band)} .

The density and viscosity are blended from the two phases (and, where an
immersed body is present, the body via the BDIM Heaviside :math:`\mu_0`):

.. math::

   \rho_{\text{fluid}} &= \alpha\,\rho_w + (1-\alpha)\,\rho_a, \\
   \rho_{\text{eff}}   &= \mu_0\,\rho_{\text{fluid}} + (1-\mu_0)\,\rho_{\text{body}} .

VOF is chosen over a level set for its **mass conservation** — essential for a
floating body, which would otherwise slowly sink or rise as volume drifts.


Per-step cycle
--------------

Each step is the standard fractional-step scheme with the density field folded
into the projection:

1. **predictor** — advection–diffusion + the uniform gravity body force
   :math:`\Delta t\,\mathbf{g}` (applied to *all* cells; the air has mass);
2. **projection** — solve the variable-coefficient Poisson
   :math:`\nabla\cdot\!\big(\tfrac{\Delta t}{\rho_{\text{eff}}}\nabla p\big)
   = \nabla\cdot\tilde{\mathbf u}` and correct
   :math:`\mathbf u \mathrel{-}= \tfrac{\Delta t}{\rho_{\text{eff}}}\nabla p`
   (the existing MGCG path; FFT cannot do variable density);
3. **VOF transport** — advect :math:`\alpha` once per step (:ref:`tp-advection`).

The hydrostatic interface jump and the buoyancy on a body therefore emerge
automatically from the density-weighted pressure
(:math:`\nabla p = \rho_{\text{eff}}\,\mathbf g` at rest).
:class:`TwoPhaseSolver` overrides only three things on the base solver — the
density-based projection coefficients, the VOF transport in ``finalize_step``,
and the body force (below); everything else (advection, diffusion, BDIM, the
Poisson solver) is inherited unchanged.


.. _tp-advection:

VOF transport
-------------

``alpha`` is advected in **conservative flux form** with a bounded TVD scheme
(``cubista`` / ``vanLeer``) shared from :mod:`lilytorch.src.advection`
(:func:`~lilytorch.src.advection.advect_scalar`), plus an optional MULES-style
**interface-compression** flux that counters numerical smearing. Because a
bounded scheme can still over/undershoot :math:`[0,1]` slightly, the clamp is
**conservative**: the clamped defect is redistributed back into the interface
band, so total volume is conserved to round-off rather than discarded.
Forward Euler, once per step, on the projected (divergence-free) velocity.


Buoyancy & body loads
----------------------

The naive surface-pressure integral :math:`\oint -p\,\mathbf n\,dS` cannot
extract the small buoyancy from the *large* hydrostatic pressure
(:math:`\rho_w g H`): the smoothed-delta discretisation suffers catastrophic
cancellation. Instead the buoyancy is taken **analytically from the displaced
fluid** (gauge-robust),

.. math::

   \mathbf F_{\text{buoy}} = -\,\mathbf g \int \rho_{\text{fluid}}\,(1-\mu_0)\,dV ,

and the viscous stress from the inherited routine. See
:meth:`~lilytorch.src.two_phase_solver.TwoPhaseSolver._displaced_buoyancy`.

.. note::

   **Current limitations** (follow-ups, not yet implemented):

   * the pressure-based *dynamic* load (form drag, added mass, wave radiation)
     on a **moving** body is omitted — recovering it cleanly needs a
     **well-balanced reduced-pressure** (``p_rgh``) solve so the dynamic
     pressure is a solve variable (a naive post-hoc subtraction is not
     well-balanced and drives parasitic currents);
   * there are stable :math:`\sim\!0.5\,\text{m/s}` parasitic currents at the
     body waterline (same well-balancing issue).

   Buoyancy and quasi-static floating are correct; violent splashes / fully
   dynamic body loads await the ``p_rgh`` work.


Configuration
-------------

Enable with the ``solver.two_phase`` YAML block (with ``solver.gravity`` and
``poisson_method: mgcg``):

.. code-block:: yaml

   solver:
     poisson_method: mgcg            # required (FFT cannot do variable density)
     gravity: [0.0, -9.81]           # body force (length = ndim)
     two_phase:
       alpha_init: "lambda X, Y: (Y < 0.5).double()"  # 1 water, 0 air
       rho_water:  1000.0
       rho_air:    1.0               # realistic ~1000:1
       nu_water:   1.0e-6
       nu_air:     1.5e-5
       advection:  cubista           # bounded VOF scheme (advection.SCHEMES)
       compression: 1.0              # interface-compression strength (0 = off)
       face_density: harmonic        # arithmetic | harmonic

Build a :class:`TwoPhaseSolver` with
:func:`~lilytorch.src.two_phase_solver.build_two_phase_solver` (or instantiate
it directly). The ``two_phase`` and (legacy) per-run free-surface paths are
mutually exclusive; ``alpha_init`` uses the same lambda-string convention as
``body.sdf``.


Validation
----------

``lilytorch/validation/two_phase_2d/`` exercises the full path (CPU, float64):

* ``run_hydrostatic.py`` — water+air column at rest: variable-density
  hydrostatic pressure (~1.5 % across the 1000:1 jump), :math:`|\mathbf u|\to0`,
  volume conserved.
* ``run_dam_break.py`` — collapsing column; the surge front matches the
  Martin & Moyce experiment and mass is conserved to round-off.
* ``run_floating_cylinder.py`` — a fixed half-submerged cylinder feels the
  correct Archimedes buoyancy (within ~10 %).

Unit tests for the VOF operators (boundedness, mass conservation, density
blends)::

   python -m lilytorch.src.test_two_phase


References
----------

* C. W. Hirt and B. D. Nichols, *Volume of fluid (VOF) method for the dynamics
  of free boundaries*, J. Comput. Phys. **39** (1981).
* O. Ubbink and R. I. Issa, *A method for capturing sharp fluid interfaces on
  arbitrary meshes* (CICSAM), J. Comput. Phys. **153** (1999).
* S. M. Damián, *An extended mixture model for the simultaneous treatment of
  short and long scale interfaces* (interFoam / MULES), 2013.
