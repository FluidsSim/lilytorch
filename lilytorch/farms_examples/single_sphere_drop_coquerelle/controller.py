"""Coquerelle sphere sedimentation – fluid controller.

This file is superseded by the unified
``lilytorch.integration.BDIMhandler.BDIMhandler``.

The simulation is now driven entirely through ``simulation_config.yaml``:

    handler_path: lilytorch.integration.BDIMhandler.BDIMhandler
    bdim_yaml:
      solver:
        heun: true          # Heun (RK2) for BDIM-projection cycle
        dtype: "float64"    # match original float64 accuracy
        ...
      body:
        2d_plane: "xz"      # MuJoCo (x,z) → fluid (x,y)
        force_scaling: 1.0  # no z-extent scaling for 2-D body
        ...

Key config flags that replicate the original custom behaviour:
  - ``solver.heun: true``              — Heun (RK2) outer time integration
  - ``solver.dtype: "float64"``        — double precision (original default)
  - ``body.2d_plane: "xz"``           — sphere falls in MuJoCo z-direction
  - ``body.force_scaling: 1.0``       — no z-extent scaling for 2-D body
  - ``physics.solref: [0.00001, 10]`` — contact solver parameters
"""
