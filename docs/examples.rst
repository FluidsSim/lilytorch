.. _examples:

Examples
========

This page walks through the simulations that ship with lilytorch, split by
whether they need MuJoCo or not. All examples assume you have completed
the install steps in :doc:`getting_started`.

.. contents:: On this page
   :local:
   :depth: 2


Standalone examples (no MuJoCo required)
----------------------------------------

These examples call :class:`~lilytorch.src.solver.FluidSolver` directly.
They only need Path A of :doc:`getting_started` (lilytorch + PyTorch, no
FARMS).

2-D cylinder drag — validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impulsively started flow past a 2-D cylinder at Re = 550, reproducing
Koumoutsakos & Leonard (1995).

.. code-block:: bash

   # Run the simulation (default is 512 × 512, saves drag history)
   python lilytorch/validation/cylinder_drag_2d/run_cylinder_drag.py

   # Plot the drag coefficient vs. the reference
   python lilytorch/validation/cylinder_drag_2d/plot_cylinder_drag.py

What it exercises: QUICK advection, multigrid Poisson, Dirichlet inflow /
Neumann outflow, analytical circle SDF, and pressure+viscous force
integration.

2-D cylinder spatial error analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Grid-refinement study on the same cylinder benchmark using the
Maertens–Weymouth scaling:

.. code-block:: bash

   python lilytorch/validation/error_analysis_cylinder_2d/run_error_analysis_cylinder_2d_MW.py
   python lilytorch/validation/error_analysis_cylinder_2d/plot_error_analysis_MW.py

3-D jellyfish — free-swimming with analytical bell kinematics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A pulsing jellyfish whose bell is actuated analytically inside the body
class; the global 6-DoF pose is integrated by the standalone BDIM loop
from the hydrodynamic loads. **No MuJoCo involvement.**

.. code-block:: bash

   # Uses config_fluid.yaml next to the script
   python -m lilytorch.farms_examples.jellyfish.run_jellyfish_fluid

   # Or point to a custom YAML
   python -m lilytorch.farms_examples.jellyfish.run_jellyfish_fluid /path/to/my_config.yaml

Key settings in ``config_fluid.yaml``: 256³ cube, FFT Neumann Poisson,
``eps_multiplier: 2.0`` for a slightly wider BDIM interface to stabilise
the 6-DoF feedback.

3-D free-swimming cost analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sweeps the free-swimming 3-D solver configuration to measure the per-step
cost of the Poisson solve, advection–diffusion, and BDIM stages:

.. code-block:: bash

   python lilytorch/validation/cost_analysis_free_swimming_3d/run_multigrid_cost_analysis.py

Memory / runtime benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A few top-level scripts profile the solver in isolation:

.. code-block:: bash

   python run_compile_advdiff_bench.py     # advection–diffusion micro-bench
   python run_compile_smoother_bench.py    # Poisson smoother micro-bench
   python run_cost_analysis_3d.py          # end-to-end cost sweep
   python run_memory_profile_free_3d.py    # peak GPU memory vs. grid size
   python run_poisson_comparison_3d.py     # FFT vs. multigrid vs. MGCG

All of them import ``FluidSolver`` only; no MuJoCo / FARMS is needed.


Coupled examples (MuJoCo + FARMS)
---------------------------------

These examples require Path B of :doc:`getting_started`. At each step the
FARMS ``FluidExtension`` reads link poses from MuJoCo, runs the BDIM
fluid sub-step, and writes hydrodynamic forces back into
``physics.data.xfrc_applied``.

2-D anguilliform eel (pinned)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single eel held in place by a revolute joint, driven by a PD muscle
controller. The surrounding flow is a channel with uniform inlet and
zero-gradient outlet.

.. code-block:: bash

   python -m lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_2d

Variants in the same folder:

* ``gen_configs_one_pinned_3d.py`` — same problem in 3-D.
* ``gen_configs_one_free_3d.py``   — one free-swimming eel, 3-D.
* ``gen_configs_pair_swimmers.py`` — two swimmers, study of mutual induction.
* ``gen_configs_two_pinned.py``    — two pinned eels side by side.

Zebrafish larva
^^^^^^^^^^^^^^^

A simplified zebrafish swimmer with analytical muscle actuation:

.. code-block:: bash

   python -m lilytorch.farms_examples.zebrafishsim.gen_configs

Salamander
^^^^^^^^^^

Swimming and paddling salamander gaits (SalamandraRobotica-style
multibody). Both 2-D and 3-D variants are shipped:

.. code-block:: bash

   python -m lilytorch.farms_examples.salamander.gen_configs_swim_2d
   python -m lilytorch.farms_examples.salamander.gen_configs_paddle_2d
   python -m lilytorch.farms_examples.salamander.gen_configs_swim_3d
   python -m lilytorch.farms_examples.salamander.gen_configs_underwater_walking_3d

Pleurodeles
^^^^^^^^^^^

A more realistic salamander model (*Pleurodeles waltl*) driven by joint
kinematics replay from recorded animal data:

.. code-block:: bash

   python -m lilytorch.farms_examples.pleurodeles.gen_configs_swim

Submarine
^^^^^^^^^

A free-swimming submarine in a 3-D pool with a propeller controller and
ballast-stabilised roll — two-way coupled to BDIM:

.. code-block:: bash

   python -m lilytorch.farms_examples.submarine.gen_configs_drag

Single-sphere drop (FARMS experiment bundle)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validation cases for sphere sedimentation (Coquerelle & Cottet, Gazzola
et al.). These are packaged as self-contained FARMS experiments and
launched with ``farmsim``:

.. code-block:: bash

   # Gazzola low-Re sphere drop
   cd lilytorch/farms_examples/single_sphere_drop_gazzola
   ./run.sh

   # 3-D Coquerelle study across densities and viscosities
   cd lilytorch/farms_examples/single_sphere_drop_coquerelle_3d
   ./run_all_cases.sh

Each folder contains ``experiment_config.yaml`` (FARMS), an
``arena_config.yaml`` (MuJoCo world), a ``simulation_config.yaml``
(lilytorch fluid solver) and a ``plot_data.py`` analysis script.


Writing your own example
------------------------

The cleanest template for a *standalone* case is the jellyfish driver
(:mod:`lilytorch.farms_examples.jellyfish.run_jellyfish_fluid`): it
loads a YAML, builds the solver, optionally replaces
``solver.composite_body`` with a custom :class:`~lilytorch.src.body.Body`
subclass, and runs the loop.

The cleanest template for a *coupled* case is any
``gen_configs_*.py`` in :mod:`lilytorch.farms_examples._1guillasim`:
a :class:`BaseSimConfig` subclass defines the grid, physics, animat(s),
and extensions list; the
:class:`~lilytorch.integration.extensions.FluidExtension` entry turns
the fluid solver on.

A typical ``extensions`` list for a coupled 3-D run looks like

.. code-block:: python

   extensions = [
       {
           "loader": "lilytorch.integration.extensions.FluidExtension",
           "config": {
               "solver_config": "path/to/fluid_config.yaml",
               "save_forces":   True,
           },
       },
       # Optional — render the flow inside the MuJoCo viewer / recording
       {
           "loader": "lilytorch.integration.flow_viewer.FlowViewer",
           "config": {
               "field":        "omega_z",
               "max_spheres":  4000,
               "iso_fraction": 0.15,
               "sphere_size":  0.004,
           },
       },
   ]

See :doc:`visualization` for a full description of ``FlowViewer``.
