.. _api_advection:

``advection`` — Convective Schemes & AdvDiffSolver
===================================================

Dimension-agnostic advection on the MAC staggered grid: the pluggable
convective schemes (QUICK, ADBQUICKEST, CUBISTA, van Leer, CDS), the
location-agnostic flux assembler, momentum and scalar advection kernels,
and the :class:`~lilytorch.src.advection.AdvDiffSolver` orchestrator (which
composes :mod:`lilytorch.src.diffusion`).

:func:`~lilytorch.src.advection.advect_scalar` is reused by the two-phase VOF
transport (see :ref:`two_phase`).

.. automodule:: lilytorch.src.advection
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
