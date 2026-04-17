
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):

        self.freqs = [3,8,12]

        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'zebrafishsim',
        )

        self.constant_drags = [
            [-0.0, -0.0102864, -0.0080005],
            [0, 0, 0]
        ]

        # ── Hardware ──────────────────────────────────────────────────
        self.compute_sdf    = True
        self.use_gpu        = True
        self.use_bdim       = True
        self.headless       = True
        self.smagorinsky_cs = 0.

        # ── Animats ───────────────────────────────────────────────────
        self.filter_fixed_joints = False

        self.animats_pars = [
            {
                "sdf_file"       : os.path.join(sdfs_path, "zebrafish", "zebrafish_v1_triangulated", "sdf", "zebrafish_old.sdf"),
                "control_type"   : "position",
                "gains"          : [0.001, .00002, 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [0, 0, 0.0, 0, 0, 3.141592653589793],
                "controller_path": "lilytorch.farms_examples.zebrafishsim.pd_controller.PositionController",
                "control_pars"   : {
                    'freq': 5.0, 'twl': 20, 'amp': 120,
                    'bout_duration': None, 'glide_duration': 1,
                    'bout_ramp': 0.2,
                },
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        # Body length ~4 mm; domain ~10× body length in x
        self.Nx   = 1024
        self.Ny   = 256
        self.Nz   = 128
        self.xmin = -0.02
        self.xmax =  0.08
        self.ymin = -0.0125
        self.ymax =  0.0125
        self.zmin = -0.00625
        self.zmax =  0.00625

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.timestep          = 0.0005
        self.convection_method = "quick"
        self.n_iterations      = 10001
        self.save_every        = 50
        self.cb_sub_steps      = 2
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 100.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        self.wall_thickness = 0.003
        self.wall_height    = 0.01

        # ── BDIM solver ──────────────────────────────────────────────
        self.dtype                    = "float32"
        self.zero_pressure_inside     = True
        self.bdim_dt                  = self.timestep
        self.bdim_nt                  = self.n_iterations + 1
        self.poisson_tol              = 1.0e-4
        self.poisson_max_cycles       = 30
        self.poisson_max_mgcg_cycles  = 10
        self.poisson_precond_vcycles  = 1
        self.poisson_warm_start       = False
        # self.poisson_method           = "fft"
        self.poisson_smoother         = "jacobi"
        self.poisson_nsmoothing       = 5
        self.poisson_bc_type          = "neumann"

        self.poisson_compile          = False
        self.compile_adv_diff         = False
        self.compile_forces           = False
        self.compile_sdf              = False

        # ── Boundary conditions (3-D, all Dirichlet no-slip) ────────
        self.bc_type_u   = ["D", "D", "N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "D", "D", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["N", "N", "N", "N", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.convexify             = False
        # self.n_samples             = (2000, 2000)
        self.interp_data_subfolder = "interp_data"

    # ── Extensions ────────────────────────────────────────────────────

    # def extra_simulation_extensions(self, output_folder):
    #     extensions = []

    #     # # Top-down camera auto-fitted to the domain
    #     # cam = top_down_camera_config(
    #     #     self.xmin, self.xmax,
    #     #     self.ymin, self.ymax,
    #     #     self.zmin, self.zmax,
    #     #     overshoot=1,
    #     # )
    #     extensions.append({
    #         "loader": "farms_mujoco.sensors.camera.CameraRecording",
    #         "config": {
    #             "path"            : os.path.join(output_folder, "output", "video.mp4"),
    #             "animat_id"       : None,
    #             "fps"             : 30,
    #             "speed"           : 1.0,
    #             "angular_velocity": 0,
    #             # **cam,
    #         },
    #     })

    #     return extensions


    def customize_animat(self, animat_i, animat_pars, n_joints, index):
        animat_pars["control_pars"]["freq"] = float(self.freqs[index])

    def run(self):
        for i in range(len(self.freqs)):
            self.single_run(i)


if __name__ == "__main__":
    SimConfig().run()



