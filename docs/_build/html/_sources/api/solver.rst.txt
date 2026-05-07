.. _api_solver:

``solver`` — Navier–Stokes Solver
===================================

.. automodule:: lilytorch.src.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: Module-level compiled kernels

   The following module-level functions are designed for use with
   ``torch.compile`` and are called internally by
   :class:`FluidSolver`:

   .. autofunction:: _forces_shared_2d
   .. autofunction:: _forces_body_batch_2d
   .. autofunction:: _forces_shared_3d
   .. autofunction:: _forces_body_integrate_3d
   .. autofunction:: _forces_body_batch_3d
