
import os
import numpy as np
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.farms_examples.base_sim_config import BaseSimConfig


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'salamander',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_bdim     = True
        self.use_gpu      = True
        self.compute_sdf  = True
        self.wall_height  = 0.02
        self.water_height = 0.015
        self.stack_folder = "salamander"

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"  : "salamander_v5",
                "sdf_name"    : "sdf/salamander.sdf",
                "control_type": "position",
                "gains"       : [0.001, .0002, 0],
                "controller_config": {
                    'path'      : "lilytorch.farms_examples.salamander.pd_controller_paddle.PositionController",
                    'freq'      : 1,
                    'twl'       : 10,
                    'amp'       : 200,
                    'limb_pose1': -0.35 * 3.141592653589793,
                    'limb_pose2': -0.2 * 3.141592653589793,
                },
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose"      : [0, 0, 0.015, 0, 0, 3.141592653589793],
            },
        ]

        self.solver_method    = "kernel"
        self.poisson_compile  = False
        self.compile_adv_diff = False
        self.compile_forces   = True


        # ── 2-D grid ─────────────────────────────────────────────────
        self.Nx   = 1024
        self.Ny   = 256
        self.xmin = -0.13
        self.xmax =  0.27
        self.ymin = -0.05
        self.ymax =  0.05

        # ── Physics ───────────────────────────────────────────────────
        self.timestep          = 0.001
        self.convection_method = "abdquickest"
        self.n_iterations      = 50001
        self.save_every        = 50
        self.num_sub_steps     = 5

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 10.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.03
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = False
        self.water_buoyancy = False

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                  = self.timestep
        self.bdim_nt                  = self.n_iterations
        self.zero_pressure_inside     = True
        self.rho_body                 = 900.0

        # ── Boundary conditions ──────────────────────────────────────
        self.bc_type_u   = ["D", "D", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.contour_mask = False
        # self.n_samples    = (2000, 2000)

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_joint_initials(self, joints_list):
        for joint in joints_list:
            if joint['name'] in ("joint_leg_0_L_0", "joint_leg_0_R_0"):
                joint['initial'] = [-np.pi / 3, 0.0]
            if joint['name'] in ("joint_leg_0_L_3", "joint_leg_0_R_3"):
                joint['initial'] = [-np.pi / 4, 0.0]
            if joint['name'] in ("joint_leg_1_L_0", "joint_leg_1_R_0"):
                joint['initial'] = [-np.pi / 3, 0.0]
            if joint['name'] in ("joint_leg_1_L_3", "joint_leg_1_R_3"):
                joint['initial'] = [-np.pi / 4, 0.0]


if __name__ == "__main__":
    SimConfig().run()
