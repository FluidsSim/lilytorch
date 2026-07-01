"""Gazzola sphere sedimentation – fluid controller.

This file is superseded by the unified
``lilytorch.integration.BDIMhandler.BDIMhandler``.

The simulation is now driven entirely through ``simulation_config.yaml``:

    handler_path: lilytorch.integration.BDIMhandler.BDIMhandler
    bdim_yaml:
      solver:
        heun: true          # Heun (RK2) for BDIM-projection cycle
        ...
      body:
        2d_plane: "xz"      # MuJoCo (x,z) → fluid (x,y)
        force_scaling: 1.0
        ...

Key config flags that replicate the original custom behaviour:
  - ``solver.heun: true``          — Heun (RK2) outer time integration
  - ``body.2d_plane: "xz"``        — sphere falls in MuJoCo z-direction
  - ``body.force_scaling: 1.0``    — no z-extent scaling for 2-D body
  - ``physics.solref: [0.001, 0.5]`` — contact solver parameters
"""
