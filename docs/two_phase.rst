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

``alpha`` is advected with the **Weymouth & Yue (2010) conservative VOF**
method (the scheme used by lily-pad / WaterLily and the BDIM+VOF coupling):
**dimensional operator split**, one direction per sub-sweep, with the interior
update

.. math::

   a_i \mathrel{+}= \frac{\Delta t}{h}\,
     \big[\,F_{i-1/2} - F_{i+1/2} + a_i\,(u_{i+1/2} - u_{i-1/2})\,\big] .

The face flux ``F = u_f a_f`` uses a **Courant-corrected, van-Leer-limited**
extrapolation of the *donor* cell to the face
(:math:`a_f = a_{\text{don}} + \tfrac12(1-|C|)\,\psi`, second order, reducing to
first-order upwind at extrema).  The **divergence-correction** term
:math:`a_i(u_{i+1/2}-u_{i-1/2})` is the key ingredient: it makes each 1-D sweep
**bounded** in :math:`[0,1]` (CFL :math:`\le 1`) and, summed over the ``D``
sweeps of a discretely divergence-free velocity, conserves total volume **to
round-off — with no clamping and no interface reconstruction**.  The sweep
order alternates each step to limit directional bias.  See
:meth:`~lilytorch.src.two_phase.TwoPhase.advect` / ``_cvof_sweep``.

In practice the conservation error is set by the *divergence residual* of the
projected velocity (``poisson_tol``): quiescent / rigid-body fields drift at
round-off (:math:`\sim\!10^{-13}`), a violent dam-break at :math:`\sim\!10^{-3}`
over hundreds of steps.  (The ``advection`` / ``compression`` config knobs are
legacy and **unused** by this conservative scheme.)


Buoyancy & body loads
----------------------

Buoyancy is **emergent**: rather than adding an analytic displaced-volume term,
:meth:`~lilytorch.src.two_phase_solver.TwoPhaseSolver._two_phase_forces`
integrates the *real* variable-density pressure over the BDIM band, so the
hydrostatic gradient across the body yields the buoyant load (and the dynamic
load) directly, alongside the viscous stress from the inherited routine. The
challenge is that this band integral must stay gauge-robust — the small buoyancy
sits on a *large* hydrostatic baseline (:math:`\rho_w g H`) — which is exactly
what the partial-Heaviside readout below addresses.

Partial-Heaviside (:math:`\partial H`) pressure readout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the body loads *are* read from the pressure field (the eulerian path,
``force_method = "eulerian"``), the per-body smoothed-delta quadrature
:math:`-\sum p\,\mathbf n\,\delta_\varepsilon(\phi_b)` is only gauge-invariant
if the discrete :math:`\sum \mathbf n\,\delta_\varepsilon = 0`; on a coarsely
resolved or articulated body it is not, so the large hydrostatic baseline leaks
into a spurious force that is **linear in depth** (and, on a multi-link body,
also surfaces at the internal inter-link seams).

Setting ``force_submethod = "deltaH"`` switches the pressure readout to the
**partial-Heaviside** form :math:`-\sum p\,\partial_i H_\varepsilon`, where the
force density is the discrete gradient of the smooth Heaviside of the
**union** SDF — one closed surface with *no* internal seams. This obeys
summation-by-parts (:math:`\sum \partial_i H_\varepsilon = 0` discretely), so
the hydrostatic baseline cancels exactly and a static submerged body recovers
:math:`F_z/F_{\text{Arch}} \approx 1` with a seam-free, depth-independent
horizontal force. The union force is redistributed to the individual links by a
softmin partition of unity :math:`w_b = \mathrm{softmax}(-\phi_b/\tau)`
(:math:`\tau = \texttt{force\_ph\_blend\_cells}\cdot h`), so
:math:`\sum_b \mathbf F_b` equals the union force exactly while each link still
receives its own force and torque. Viscous loads are unchanged. The readout is
implemented directly in the native CUDA/CPU force kernels (2-D and 3-D) and is
bit-matched to the python reference
:meth:`~lilytorch.src.two_phase_solver.TwoPhaseSolver._apply_partition_heaviside`;
see ``lilytorch/validation/two_phase_3d/band_treatment_check.py``. It is the
recommended readout for surface-straddling and multi-link swimmers.

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
       face_density: harmonic        # arithmetic | harmonic (projection face rho)
       # advection / compression: legacy keys, unused by the W&Y conservative VOF

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
* ``run_drop_sphere.py`` — a circular body driven down through the surface
  (2-D analogue of the BDIM+VOF water-entry / drop-sphere): the immersed body
  **crosses the interface stably** (the regime single-fluid could not survive),
  mass is conserved through entry, and the vertical force rises from ~0 (in
  air) to the full-submergence Archimedes buoyancy.

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
