.. _getting_started:

Getting Started
===============

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ferrarioa5/lilytorch.git
   cd lilytorch

   # Create a virtual environment (Python ≥ 3.9)
   python -m venv venv
   source venv/bin/activate

   # Install in editable mode
   pip install -e .

   # Install PyTorch with CUDA support (adjust CUDA version as needed)
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # Install pytorch_interpolation
   pip install git+https://github.com/ferrarioa5/pytorch_interpolation.git

.. tip::

   For GPU-accelerated simulations you need a CUDA-capable GPU.  LilyTorch
   also runs on CPU — set ``use_gpu: false`` in the YAML config.


Quick start — running a simulation
-----------------------------------

Every simulation is driven by a **YAML configuration file** that specifies
the domain, physics, body geometry, boundary conditions, and output options.

.. code-block:: python

   from lilytorch.src.solver import FluidSolver
   from lilytorch.util.yaml_operations import pyobject2yaml
   import yaml

   # Load a config
   with open("config.yaml") as f:
       pars = yaml.safe_load(f)

   # Create and run
   solver = FluidSolver(pars)
   solver.run_sim()

Minimal YAML config (2-D flow past a cylinder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   solver:
     Nx: 512
     Ny: 128
     xmin: -0.9
     xmax: 1.5
     ymin: -0.3
     ymax: 0.3
     dt: 0.001
     nt: 10000
     nu: 1.0e-4
     rho: 1000.0
     use_gpu: true
     convection_method: quick
     poisson_method: multigrid

   boundary_conditions:
     BC_type_u:   ["D", "N", "D", "D"]
     BC_values_u: [1.0, 0.0, 1.0, 1.0]
     BC_type_v:   ["D", "N", "D", "D"]
     BC_values_v: [0.0, 0.0, 0.0, 0.0]

   body:
     type: analytical
     sdf_type: circle
     position: [0.0, 0.0]
     radius: 0.05
     suit: 0.0

   output:
     save_path: results
     save_frames: true
     save_every: 100
     save_uv: true

See :doc:`parameters` for a complete reference of every configuration key.


Project layout
--------------

.. code-block:: text

   lilytorch/
   ├── src/               Core solver modules
   │   ├── solver.py      Main FluidSolver class
   │   ├── adv_diff.py    Advection–diffusion solver
   │   ├── poisson_fft.py FFT-based Poisson solver
   │   ├── poisson_mult.py Multigrid / MGCG Poisson solver
   │   ├── body.py        Body geometry, SDF, kinematics
   │   ├── operations.py  Discrete differential operators
   │   └── plotting.py    Visualisation helpers
   ├── integration/        FARMS coupling layer
   │   ├── BDIMhandler.py  BDIM ↔ MuJoCo bridge
   │   └── kinematics.py   Kinematics replay controller
   ├── FARMS_V2/           FARMS submodules setup
   ├── farms_examples/     Ready-to-run coupled examples
   ├── util/               I/O, YAML, path helpers
   └── validation/         Validation & benchmarking scripts


Building the documentation
--------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make html

The output is generated in ``docs/_build/html/``.
