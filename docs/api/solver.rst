.. _api_solver:

``solver`` — Navier–Stokes Solver
===================================

.. automodule:: lilytorch.src.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


``forces`` — Hydrodynamic Force Integrators
=============================================

Hydrodynamic force and torque integration on immersed bodies. These
routines were factored out of ``solver.py`` so the inner per-body loops
can be ``torch.compile``-d independently.

* ``forces_method1`` / ``forces_method2`` — 2-D viscous + pressure
  integration via Heaviside / smoothed-delta weighting.
* ``forces_method2_3d`` — 3-D batched variant used by the unified
  ``BDIMhandler``.
* ``_forces_shared`` — shared per-cell stress / pressure kernel.
* ``_forces_body_batch`` / ``_forces_body_integrate_3d`` — per-body
  reductions, designed for ``torch.compile``.

.. automodule:: lilytorch.src.forces
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


``extras`` — Sponge, LES & Non-Newtonian Viscosity
====================================================

Optional add-on physics dispatched from the solver:

* ``_build_sponge_fields`` / ``apply_sponge_damping`` — quadratic
  damping ramp near domain walls.
* ``apply_yield_damping`` — Herschel–Bulkley yield-stress damping.
* ``_compute_smagorinsky_nu_t`` — Smagorinsky LES eddy viscosity.
* ``_compute_carreau_nu_t`` — Carreau / Herschel–Bulkley shear-thinning
  viscosity.
* ``_compute_nu_t`` / ``_compute_nu_rho_for_forces`` — unified
  dispatchers selecting the active viscosity model.

.. automodule:: lilytorch.src.extras
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
