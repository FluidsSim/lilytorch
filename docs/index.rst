.. LilyTorch documentation master file

LilyTorch Documentation
=======================

**LilyTorch** is a GPU-accelerated 2-D / 3-D incompressible Navier–Stokes
solver with immersed-boundary support, built entirely on `PyTorch
<https://pytorch.org>`_.  It targets fluid–structure interaction problems —
from freely-swimming organisms to sedimentation — and can couple to the
`FARMS <https://github.com/farmsim>`_ MuJoCo-based robotics framework for
full neuromechanical-fluid simulations.

Key features
------------

* **Staggered MAC grid** with ghost-cell boundary conditions (Dirichlet / Neumann).
* **BDIM2 immersed-boundary method** (Weymouth & Yue 2011, Maertens & Weymouth 2015).
* **Heun (RK2) predictor–corrector** time stepping.
* Multiple advection schemes: QUICK, ADBQUICKEST, CUBISTA, van Leer, CDS,
  semi-Lagrangian.
* Optional **Smagorinsky LES** subgrid-scale model for under-resolved
  turbulent flows.
* FFT Poisson solvers (Neumann DCT & free-space Green's function) and
  variable-coefficient multigrid / MGCG.
* Full ``torch.compile`` support for all hot kernels.
* **HDF5** checkpoint output.
* Optional MuJoCo coupling via the FARMS extension API.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   parameters
   boundary_conditions

.. toctree::
   :maxdepth: 2
   :caption: Theory & Numerics

   mathematical_formulation
   numerical_schemes
   immersed_boundary

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/solver
   api/adv_diff
   api/poisson
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
