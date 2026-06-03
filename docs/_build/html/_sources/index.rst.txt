.. LilyTorch documentation master file

LilyTorch Documentation
=======================

**LilyTorch** is a GPU-accelerated 2-D / 3-D incompressible Navier–Stokes
solver with immersed-boundary support, built entirely on `PyTorch
<https://pytorch.org>`_.  It targets fluid–structure interaction problems —
from freely-swimming organisms to sedimentation — and can optionally couple
to the `FARMS <https://github.com/farmsim>`_ MuJoCo-based robotics framework
for full neuromechanical–fluid simulations.

The package supports **two modes of use**:

* **Standalone mode** — the :class:`~lilytorch.src.solver.FluidSolver` runs
  on its own. Bodies are described by signed-distance functions (analytical
  shapes, meshes, or fully prescribed kinematics). MuJoCo is *not* required.
  Good for validation problems, prescribed-kinematics swimmers, sphere
  sedimentation, the 3-D jellyfish, etc.
* **Coupled mode** — the solver is driven as a FARMS/MuJoCo task extension
  (:class:`~lilytorch.integration.extensions.FluidExtension`). MuJoCo
  integrates the multibody dynamics, lilytorch computes the hydrodynamic
  forces, and the two are advanced together at each time step.

Key features
------------

* **Staggered MAC grid** with ghost-cell boundary conditions (Dirichlet / Neumann).
* **BDIM2 immersed-boundary method** (Weymouth & Yue 2011, Maertens & Weymouth 2015).
* **Heun (RK2) predictor–corrector** time stepping.
* Multiple advection schemes: QUICK, ADBQUICKEST, CUBISTA, van Leer, CDS,
  semi-Lagrangian.
* Optional **Smagorinsky LES** subgrid-scale model for under-resolved
  turbulent flows.
* Optional **non-Newtonian viscosity** via Carreau / Herschel–Bulkley models.
* Optional **sponge / damping layer** for quasi-infinite domains.
* FFT Poisson solvers (Neumann DCT & free-space Green's function) and
  variable-coefficient multigrid / MGCG.
* Full ``torch.compile`` support for all hot kernels.
* **HDF5** checkpoint output and automatic PNG/MP4/GIF post-processing.
* Optional MuJoCo coupling via the FARMS extension API.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   examples
   parameters
   boundary_conditions
   visualization

.. toctree::
   :maxdepth: 2
   :caption: Theory & Numerics

   mathematical_formulation
   numerical_schemes
   immersed_boundary
   free_surface
   strong_coupling

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/solver
   api/advection
   api/diffusion
   api/adv_diff
   api/poisson
   api/free_surface
   api/body
   api/operations
   api/plotting
   api/integration
   api/util

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
