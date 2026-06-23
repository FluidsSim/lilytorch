.. _api_integration:

``integration`` — FARMS / MuJoCo Coupling Layer
=================================================

BDIMhandler
-----------

Unified 2-D / 3-D bridge between FARMS body kinematics and the fluid
solver. Drives the staggered SDFs each step and writes hydrodynamic
forces back to ``data.xfrc_applied``.

.. automodule:: lilytorch.integration.BDIMhandler
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Extensions
----------

FARMS ``TaskExtension`` hooks that own the solver lifecycle and HDF5
logging.

.. automodule:: lilytorch.integration.extensions
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Kinematics
----------

Helpers converting FARMS sensor frames into the rotations / translations
consumed by ``BDIMhandler.update``.

.. automodule:: lilytorch.integration.kinematics
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Flow viewers
------------

In-viewer flow visualisation. ``flow_viewer.py`` draws coloured spheres
for 3-D fields; ``flow_viewer_2d.py`` draws a 2-D tile overlay;
``flow_viewer_2d_gpu.py`` is a CUDA→OpenGL fast path that avoids the CPU
round-trip; ``flow_viewer_gl_hook.py`` is an ``LD_PRELOAD`` shim used
when the standard ``user_scn`` injection path is not available.

.. automodule:: lilytorch.integration.flow_viewer
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: lilytorch.integration.flow_viewer_2d
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: lilytorch.integration.flow_viewer_2d_gpu
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: lilytorch.integration.flow_viewer_gl_hook
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Particle viewer
---------------

Lagrangian dye-particle overlay seeded from the body surface and
advected with RK2.

.. automodule:: lilytorch.integration.particle_viewer
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Per-body colours
----------------

.. automodule:: lilytorch.integration.body_color_override
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Camera
------

.. automodule:: lilytorch.integration.camera
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Gamepad
-------

Optional interactive gamepad controller (incl. paddling mode) for
steering coupled simulations live.

.. automodule:: lilytorch.integration.gamepad
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Pool SDF generator
------------------

Generates SDF XML files defining rectangular pool arenas with collision
walls for MuJoCo.

.. automodule:: lilytorch.integration.gen_pool_sdf
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
