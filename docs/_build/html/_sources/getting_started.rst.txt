.. _getting_started:

Getting Started
===============

.. contents:: On this page
   :local:
   :depth: 2


Prerequisites
-------------

* **Python ≥ 3.9**
* A C compiler (required for the Cython extensions shipped with the FARMS
  submodules — only needed if you want MuJoCo coupling).
* Optional: a CUDA-capable GPU. LilyTorch runs on CPU as well, but a GPU
  is strongly recommended for any 3-D problem.


Installation
------------

The recommended workflow uses a dedicated virtual environment. Two install
paths are supported, depending on whether you need MuJoCo coupling.

Path A — Standalone only (no MuJoCo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this path if you only need the pure fluid solver with analytical,
mesh-based, or prescribed-kinematics bodies (validation cases, the 3-D
jellyfish, sphere drops, cylinders, …).

.. code-block:: bash

   # 1. Clone
   git clone https://github.com/ferrarioa5/lilytorch.git
   cd lilytorch

   # 2. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate

   # 3. Install PyTorch (CPU or CUDA build) from the official selector:
   #    https://pytorch.org/get-started/locally/
   # Example — CUDA 12.1:
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # 4. Install the remaining Python dependencies
   pip install -r requirements.txt

   # 5. Install lilytorch in editable mode
   pip install -e .

You can now run any example or script that does **not** import
``farms_*`` modules — see :doc:`examples`.

Path B — Standalone + MuJoCo coupling (FARMS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this path if you want two-way fluid–multibody coupling (anguilliform
eel, salamander, zebrafish, submarine, …).

.. code-block:: bash

   # 1. Clone with submodules
   git clone --recurse-submodules https://github.com/ferrarioa5/lilytorch.git
   cd lilytorch
   # (if already cloned without --recurse-submodules)
   git submodule update --init --recursive

   # 2. Virtual environment + PyTorch + requirements (same as Path A)
   python -m venv venv
   source venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt

   # 3. Build and install the four FARMS submodules
   cd lilytorch/FARMS_V2
   python setup_farms.py
   cd -

   # 4. Install lilytorch itself in editable mode
   pip install -e .

This installs ``farms_core``, ``farms_mujoco``, ``farms_sim`` and
``farms_amphibious`` in editable mode with their Cython extensions
compiled.

.. tip::

   All coupled examples run MuJoCo headlessly by default. To watch the
   simulation live in the MuJoCo viewer, set ``headless = False`` in the
   example's ``gen_configs_*.py`` file.

Key dependencies
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Purpose
   * - `PyTorch <https://pytorch.org>`_
     - GPU-accelerated tensor computation (the whole solver lives on
       ``torch.Tensor``).
   * - NumPy / SciPy
     - Array utilities, ODE integration, splines.
   * - `Open3D <https://www.open3d.org/>`_
     - Mesh loading and closest-point queries for mesh-based SDFs.
   * - ``scikit-fmm``
     - Fast-marching for SDF extension.
   * - Matplotlib / OpenCV / ffmpeg
     - Frame plotting and video assembly.
   * - `MuJoCo <https://mujoco.org>`_ / ``dm_control``
     - Multibody physics (only for coupled mode).
   * - `FARMS <https://github.com/farmsim>`_
     - Neuromechanical simulation framework (only for coupled mode).
   * - PETSc / ``petsc4py``
     - Optional sparse parallel Poisson solver.


Two ways to drive a simulation
------------------------------

All lilytorch simulations are controlled by a **YAML configuration
file**. The file is the same shape in both modes; what changes is who
calls the solver:

* **Standalone mode** — a small Python driver instantiates
  :class:`~lilytorch.src.solver.FluidSolver` and calls ``run_sim()``.
* **Coupled mode** — a FARMS ``SimConfig`` builds the MuJoCo world and
  attaches :class:`~lilytorch.integration.extensions.FluidExtension`,
  which loads the YAML, instantiates the solver, and hooks it into the
  MuJoCo stepping loop.

See :doc:`parameters` for every configuration key, and :doc:`examples`
for ready-to-run cases in both modes.


Quick start — standalone (no MuJoCo)
------------------------------------

Any YAML config can be run from a few lines of Python:

.. code-block:: python

   import yaml
   from lilytorch.src.solver import FluidSolver

   with open("config.yaml") as f:
       pars = yaml.safe_load(f)

   solver = FluidSolver(pars)
   solver.run_sim()

A **minimal 2-D YAML** for impulsively started flow past a cylinder
(Koumoutsakos & Leonard, Re = 550) looks like this:

.. code-block:: yaml

   solver:
     Nx: 512
     Ny: 512
     xmin: -0.5
     xmax:  1.5
     ymin: -1.0
     ymax:  1.0
     dt: 1.0e-4
     nt: 20000
     nu: 1.0e-6
     rho: 1000.0
     use_gpu: true
     convection_method: quick
     poisson_method: multigrid

   boundary_conditions:
     BC_type_u:   ["D", "N", "D", "D"]
     BC_values_u: [0.00275, 0.0, 0.00275, 0.00275]
     BC_type_v:   ["D", "N", "D", "D"]
     BC_values_v: [0.0, 0.0, 0.0, 0.0]

   body:
     type: analytical
     sdf_type: circle
     position: [0.0, 0.0]
     radius: 0.1
     suit: 0.0

   output:
     save_path: results
     save_frames: true
     save_every: 100
     save_uv: true

Run it with:

.. code-block:: bash

   python -c "from lilytorch.src.solver import FluidSolver; \
              import yaml; \
              FluidSolver(yaml.safe_load(open('config.yaml'))).run_sim()"

See :doc:`examples` for larger, fully worked runs including the 3-D
jellyfish, sphere sedimentation, and the cylinder-drag validation script.


Quick start — coupled (with MuJoCo / FARMS)
-------------------------------------------

Coupled examples are driven by a FARMS launcher. Each shipped example
provides a ``gen_configs_*.py`` file that builds the YAML on the fly:

.. code-block:: bash

   # 2-D anguilliform eel pinned in a channel
   python -m lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_2d

   # 3-D anguilliform eel swimming freely
   python -m lilytorch.farms_examples._1guillasim.gen_configs_one_free_3d

   # 3-D free-swimming submarine with a propeller
   python -m lilytorch.farms_examples.submarine.gen_configs_drag

Some validation cases are shipped as FARMS experiment bundles and are
launched with ``farmsim``:

.. code-block:: bash

   cd lilytorch/farms_examples/single_sphere_drop_gazzola
   ./run.sh   # invokes farmsim with experiment_config.yaml

Output is written under the folder specified by ``output.save_path`` in
the generated YAML (usually a timestamped subfolder). See
:doc:`visualization` for turning saved frames into videos.


Project layout
--------------

.. code-block:: text

   lilytorch/
   ├── src/                 Core solver modules
   │   ├── solver.py        Main FluidSolver class
   │   ├── adv_diff.py      Advection–diffusion solver
   │   ├── poisson_fft.py   FFT / DCT Poisson solver
   │   ├── poisson_mult.py  Multigrid / MGCG Poisson solver
   │   ├── body.py          Body geometry, SDF, kinematics
   │   ├── operations.py    Discrete differential operators
   │   ├── plotting.py      Visualisation helpers
   │   ├── video_postprocess.py         Frame → MP4 / GIF
   │   └── projected_field_postprocess.py  Top-view PIV-style curl
   ├── integration/         FARMS / MuJoCo coupling layer
   │   ├── extensions.py    FluidExtension, DataLogger
   │   ├── flow_viewer.py   In-viewer flow visualisation
   │   └── BDIMhandler.py   Body ↔ MuJoCo bridge
   ├── FARMS_V2/            FARMS submodules setup
   ├── farms_examples/      Ready-to-run examples (2-D & 3-D)
   ├── util/                I/O, YAML, path helpers
   └── validation/          Validation & benchmarking scripts


Building this documentation
---------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make html

The output is generated in ``docs/_build/html/``.
