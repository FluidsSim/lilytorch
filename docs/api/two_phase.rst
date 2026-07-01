.. _api_two_phase:

``two_phase`` — Two-Phase (Water + Air) Solver
===============================================

Variable-density two-phase model for partially-submerged / floating bodies.
The air is a *real* (light) fluid; the interface is carried by a Volume-of-Fluid
fraction. See :ref:`two_phase` for the theory and configuration.

:class:`~lilytorch.src.two_phase.TwoPhase` owns the VOF field ``alpha``,
transports it (bounded, mass-conservative), and builds the density / viscosity
fields. :class:`~lilytorch.src.two_phase_solver.TwoPhaseSolver` is a thin
subclass of :class:`~lilytorch.src.solver.FluidSolver` that feeds those fields
into the variable-density pressure projection.

.. automodule:: lilytorch.src.two_phase
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: lilytorch.src.two_phase_solver
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
