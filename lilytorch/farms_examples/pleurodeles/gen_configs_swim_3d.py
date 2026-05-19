import os
import numpy as np

from farms_core.model.options import SpawnMode

from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        # Reuse the existing pleurodeles mesh SDF and generate a separate
        # 3-D interpolation cache so it does not clash with the 2-D data.
        self.compute_sdf    = False
        self.save           = True

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'pleurodeles',
        )

        # Hardware / runtime
        self.use_bdim       = True
        self.use_gpu        = True
        self.headless       = False
        self.poisson_method = "multigrid"

        # Animat
        self.animats_pars = [
            {
                "model_name": "pleurodeles",
                "sdf_name": "salamander_animal_fmsv0.21_2D.sdf",
                "control_type": "position",
                "gains": [0.2, 0.005, 0.0],
                "controller_config": {
                    "path": "lilytorch.farms_examples.pleurodeles.pd_controller_swim.PositionController",
                },
                "spawn_mode": SpawnMode.TRANSVERSE,
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, np.pi],
            },
        ]

        # 3-D grid
        self.Nx = 1024
        self.Ny = 192
        self.Nz = 96
        self.xmin = -0.13 * 2
        self.xmax = 0.27 * 2

        # Keep the long swimming domain in x and derive the transverse
        # extents from the same cell size so dx = dy = dz. With the FFT
        # Neumann pressure solve we can choose smaller transverse counts for
        # a tighter tank without being constrained by multigrid powers of 2.
        dx = (self.xmax - self.xmin) / self.Nx
        self.ymin = -0.5 * self.Ny * dx
        self.ymax = 0.5 * self.Ny * dx
        self.zmin = -0.5 * self.Nz * dx
        self.zmax = 0.5 * self.Nz * dx

        # Physics
        self.timestep = 0.0005
        self.convection_method = "quick"
        self.n_iterations = 8001
        self.save_every = 50
        self.num_sub_steps = 1

        # MuJoCo
        self.visual_scale = 10.0
        self.extent = 10.0

        # Arena
        self.wall_thickness = 0.003
        self.arena_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.water_drag = False
        self.water_buoyancy = False

        # BDIM solver
        self.bdim_dt = self.timestep
        self.bdim_nt = self.n_iterations
        self.zero_pressure_inside = False
        self.rho_body = 1000.0
        self.poisson_tol = 1.0e-4
        self.poisson_max_cycles = 3
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start = True
        self.poisson_smoother = "jacobi"
        self.poisson_nsmoothing = 5
        self.poisson_bc_type = "neumann"
        self.compile_adv_diff = True

        # Boundary conditions
        self.bc_type_u = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_v = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bc_type_w = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Body
        self.convexify = False
        self.contour_mask = True
        # self.n_samples = (2000, 2000)
        self.force_scaling = 1.0
        self.interp_data_subfolder = "interp_data_3d"

        # BDIM physics
        self.bdim_physics = {"solref": [0.001, 0.5]}

    def customize_joint_initials(self, joints_list):
        for joint in joints_list:
            if joint["name"] in ("joint_leg_0_L_0", "joint_leg_0_R_0"):
                joint["initial"] = [-0.3 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_0_L_1", "joint_leg_0_R_1"):
                joint["initial"] = [-0.2 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_0", "joint_leg_1_R_0"):
                joint["initial"] = [-0.35 * np.pi, -0.0]
            if joint["name"] in ("joint_leg_1_L_1", "joint_leg_1_R_1"):
                joint["initial"] = [-0.2 * np.pi, -0.0]

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1
        )
        extensions.append({
            "loader": "farms_mujoco.sensors.camera.CameraRecording",
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