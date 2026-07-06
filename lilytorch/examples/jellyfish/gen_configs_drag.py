"""Jellyfish simulation in MuJoCo using the FARMS drag model.

Follows the analytical jellyfish SDF from WaterLily-jl
(``examples/ThreeD_Jelly.jl``): a thin spherical shell intersected with
the half-space ``z > h`` forms a bell-shaped upper cap.  The mesh and the
inertial properties of the bell (mass, centre of mass, inertia tensor)
are produced by
``lilytorch.examples.sdfs.jellyfish.generate_jellyfish_mesh``
from the same analytical SDF and referenced by
``examples/sdfs/jellyfish/jellyfish.sdf``.

The jellyfish is spawned as a single free rigid body
(``SpawnMode.FREE``).  MuJoCo then integrates the full 6-DOF Newton-Euler
equations of motion for the body

    m * dv/dt   = sum(F_ext)          (linear)
    I * domega/dt + omega × (I omega) = sum(tau_ext)   (angular)

using the mass/inertia supplied by the SDF and the forces/torques from
gravity, buoyancy and the FARMS quadratic drag model
(``BaseSimConfig.constant_drags``).

This example mirrors ``examples/submarine/gen_configs_drag.py``:
the FARMS drag model is active (``use_bdim = False``) and a full BDIM
fluid-coupling block is kept but disabled, so switching to a coupled
fluid simulation later only requires ``self.use_bdim = True``.

Run with::

    python -m lilytorch.examples.jellyfish.gen_configs_drag
"""

import os

from farms_core.model.options import SpawnMode

from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', 'jellyfish',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu  = True
        self.headless = False

        # ── Simulation flags ──────────────────────────────────────────
        # FARMS drag model only (Newton's EoM integrated by MuJoCo).
        # Flip to True to couple with the BDIM fluid solver.
        self.use_bdim = False

        # ── Animats ───────────────────────────────────────────────────
        # Single jellyfish rigid body (no joints).  The mesh and inertia
        # come from the analytical SDF (generate_jellyfish_mesh.py).
        # Constant drag coefficients [linear, quadratic] applied per
        # link are inherited from BaseSimConfig.
        self.animats_pars = [
            {
                "model_name"  : "jellyfish",
                "sdf_name"    : "jellyfish.sdf",
                "control_type": "position",
                "gains"       : [0.0, 0.0, 0.0],
                "spawn_mode"  : SpawnMode.FREE,
                # Spawn the bell near the top of the pool with its
                # opening facing downwards so gravity/drag/buoyancy act
                # realistically on the free body.
                "pose"        : [0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
            },
        ]

        # ── Body physics ──────────────────────────────────────────────
        # Density of the jellyfish flesh (must match
        # generate_jellyfish_mesh.py so buoyancy/inertia are consistent
        # with the SDF).
        self.rho_body          = 1025.0
        self.rho               = 1000.0

        # ── 3-D grid (used by BDIM and to size the pool arena) ───────
        self.Nx   = 128
        self.Ny   = 128
        self.Nz   = 128
        self.xmin = -0.2
        self.xmax =  0.2
        self.ymin = -0.2
        self.ymax =  0.2
        self.zmin = -0.3
        self.zmax =  0.2

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.005
        self.n_iterations      = 4001
        self.save_every        = 50
        self.num_sub_steps     = 1

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 1.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        # Auto-generated pool sized from the grid extents; drag and
        # buoyancy are driven by the FARMS drag model when
        # use_bdim=False.
        self.wall_thickness = 0.01
        self.arena_pose     = [0, 0, 0, 0, 0, 0]

        # ─────────────────────────────────────────────────────────────
        #  BDIM template (kept disabled; activate by setting
        #  self.use_bdim = True above and tuning the values below).
        # ─────────────────────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations
        self.convection_method       = "quick"
        self.zero_pressure_inside    = False
        self.poisson_method          = "multigrid"
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"

        # Boundary conditions for a 3-D fluid box (Dirichlet on the
        # lateral / top-bottom walls; ready for the BDIM solver).
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Body sampling for BDIM (used only when use_bdim=True).
        self.contour_mask          = True
        self.n_samples             = (2000, 2000)
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # Top-down camera auto-fitted to the pool, also used to record
        # an .mp4 of the run.
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
        )
        extensions.append({
            "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
            "config": {
                "path"            : os.path.join(output_folder, "output", "video.mp4"),
                "animat_id"       : None,
                "fps"             : 30,
                "speed"           : 1.0,
                "angular_velocity": 0,
                **cam,
            },
        })

        return extensions


if __name__ == "__main__":
    SimConfig().run()
