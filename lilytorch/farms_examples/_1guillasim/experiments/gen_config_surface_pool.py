
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config

# z = 0 is the centre of the z domain; water fills z < 0, air above.
# The fish spawns at z=0.0 and straddles the air-water interface.
WATERLINE = 0.0


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu                       = True
        self.use_bdim                      = True
        self.compute_sdf                   = True
        self.convexify                     = True
        self.force_method                  = "lagrangian"
        self.zero_pressure_inside          = True
        self.body_velocity_blend_eps_cells = None
        self.bdim_mu0_projection           = False
        self.bdim_body_div_correction      = True
        self.poisson_method                = "multigrid"

        self.headless             = False
        self.smagorinsky_cs       = 0.

        self.compile_adv_diff = True

        # ── Two-phase free-surface flags ──────────────────────────────
        # BDIM handles all hydrodynamics; disable FARMS buoyancy/drag
        self.water_drag     = False
        self.water_buoyancy = False
        self.water_height   = WATERLINE

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                "sdf_name"       : "1guilla.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.TRANSVERSE,
                "pose"           : [4.75, 0.1, 0.0, 0, 0, 0.05],
                "controller_path": "lilytorch.farms_examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms004mpt003log.csv"
                    ),
                },
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        self.Nx   = 900
        self.Ny   = 300
        self.Nz   = 52
        self.xmin = 0
        self.xmax = 6
        self.ymin = -1
        self.ymax = 1
        self.zmin = -(2/300*52/2)
        self.zmax = (2/300*52/2)

        # ── Physics ───────────────────────────────────────────────────
        self.rho_body          = 1000.0
        self.rho               = 1000.0
        self.nu                = 1.0e-6
        self.timestep          = 0.001
        self.convection_method = "quick"
        self.n_iterations      = 20001
        self.save_every        = 200
        self.vmin              = -10.0
        self.vmax              = 10.0
        self.save              = False

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale  = 10.0
        self.extent        = 100.0

        # ── BDIM solver ──────────────────────────────────────────────
        self.bdim_dt                 = self.timestep
        self.bdim_nt                 = self.n_iterations + 1
        self.poisson_tol             = 1.0e-4
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"

        # ── Boundary conditions (3-D, all Dirichlet / no-slip) ───────
        self.bc_type_u   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_u = [0, 0, 0, 0, 0, 0]
        self.bc_type_v   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_v = [0, 0, 0, 0, 0, 0]
        self.bc_type_w   = ["D", "D", "D", "D", "D", "D"]
        self.bc_values_w = [0, 0, 0, 0, 0, 0]

        # ── Body ─────────────────────────────────────────────────────
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

        # ── Visualization ─────────────────────────────────────────────
        self.sky_color         = [0.02, 0.05, 0.15]
        self.floor_color       = "#E0D4D4"
        self.viewer_body_color = "#D09F23"
        self.wall_alpha        = 1.
        self.water_alpha       = 0.05
        self.grid_spacing      = 0.5*(self.ymax - self.ymin)

    # ── Two-phase BDIM extension ──────────────────────────────────────

    def _bdim_extension(self, output_folder):
        bdim_ext = super()._bdim_extension(output_folder)
        solver = bdim_ext["config"]["bdim_yaml"]["solver"]

        solver["gravity"] = [0, 0, -9.81]
        solver["two_phase"] = {
            "alpha_init"            : f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
            "rho_water"             : 1000.0,
            "rho_air"               : 1.2,
            "nu_water"              : self.nu,
            "nu_air"                : 1.5e-5,
            "rho_solid"             : 1000.0,
            "alpha_exclude_body"    : True,
            "alpha_volume_compensate": True,
        }

        return bdim_ext

    # ── Arena: cosmetic water stops at WATERLINE (not tank top) ──────

    def gen_arena_config(self, output_folder, index=0):
        from lilytorch.integration.gen_pool_sdf import create_pool_sdf, create_water_sdf
        from lilytorch.util.paths import sdfs_path
        from farms_core.io.yaml import pyobject2yaml
        from farms_core.model.options import SpawnMode

        wt = self.wall_thickness
        if wt is None:
            pool_dims = [self.xmax - self.xmin, self.ymax - self.ymin,
                         self.zmax - self.zmin]
            wt = max(round(0.08 * min(pool_dims), 4), 0.01)

        create_pool_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=self.zmax,
            wall_thickness=wt, plotting=False,
            wall_alpha=self.wall_alpha,
            grid_spacing=self.grid_spacing,
            floor_color=self.floor_color,
        )
        water_sdf = create_water_sdf(
            self.xmin, self.xmax, self.ymin, self.ymax,
            zmin=self.zmin, zmax=WATERLINE,
            water_height=self.water_height,
            water_alpha=self.water_alpha,
        )
        arena_sdf = os.path.join(sdfs_path, "pool", "sdf", "pool.sdf")

        arena_dict = {
            "sdf": arena_sdf,
            "spawn": {
                "loader"  : 0,
                "mode"    : SpawnMode.FREE,
                "pose"    : list(self.arena_pose),
                "velocity": [0, 0, 0, 0, 0, 0],
                "extras"  : {},
            },
            "water": {
                "sdf"      : water_sdf,
                "drag"     : self._water_drag,
                "buoyancy" : self._water_buoyancy,
                "height"   : WATERLINE,
                "velocity" : [0, 0, 0],
                "viscosity": 1.0,
                "density"  : self.rho,
                "maps"     : ["", ""],
            },
            "ground_height": self.ground_height,
        }
        pyobject2yaml(
            os.path.join(output_folder, 'arena_config.yaml'), arena_dict,
        )

    # ── Extensions ────────────────────────────────────────────────────

    def extra_simulation_extensions(self, output_folder):
        extensions = []

        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                "diffuse": [1, 1, 1],
                "ambient": [0.3, 0.3, 0.3],
            },
        })

        # Air/water interface + vorticity wake visualisation
        extensions.append({
            "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
            "config": {
                "update_every"      : 1,
                "max_vertices"      : 20 * self.Nx * self.Ny,
                "crop_boundary"     : 0,
                "debug_force_visible": False,
                "fields": [
                    {
                        "field"     : "interface",
                        "iso_value" : 0.5,
                        "alpha"     : 0.45,
                        "color"     : "#3399FF",
                        "smooth_sigma": 0,
                        "exclude_body": False,
                    },
                    {
                        "field"     : "omega_mag",
                        "iso_value" : 10.0,
                        "alpha"     : 0.3,
                        "color"     : "#FF8C1A",
                        "smooth_sigma": 0,
                        "exclude_body": True,
                        "phase_mask": "water",
                    },
                ],
            },
        })

        # Top-down camera auto-fitted to the pool
        cam = top_down_camera_config(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.zmin, self.zmax,
            overshoot=1,
            max_width=3840, max_height=2160,
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

    def _extra_run_patch(self):
        r, g, b = self.sky_color
        return (
            f"_m.night_sky=lambda mjcf_model:mjcf_model.asset.add("
            f"'texture',name='skybox',type='skybox',"
            f"builtin='flat',rgb1=[{r},{g},{b}],rgb2=[{r},{g},{b}],width=8,height=8);"
        )


if __name__ == "__main__":
    SimConfig().run()
