
import os
from farms_core.model.options import SpawnMode
from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config, side_camera_config

# z = 0 is the centre of the z domain; water fills z < 0, air above.
# The fish spawns at z=0.0 and straddles the air-water interface.
WATERLINE = 0.0


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'examples', '_1guillasim',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu                       = True
        self.use_bdim                      = True
        self.compute_sdf                   = True
        self.convexify                     = True
        # Eulerian (band integral on real pressure) is much more robust than the
        # Lagrangian surface-marker integral for this thin body at the free
        # surface on the coarse grid: uniform-800 then floats with the head out
        # instead of sinking to the floor. (Lagrangian wins only when the body
        # is well resolved.)
        self.force_method                  = "eulerian"
        # Partial-Heaviside (∂H) pressure-force readout: union-∂H density split
        # to links by a softmin partition of unity — seam-free, no hydrostatic
        # baseline leak (vs the default per-body n·δ band integral).
        self.force_submethod               = "deltaH"
        self.zero_pressure_inside          = False
        self.body_velocity_blend_eps_cells = None
        self.bdim_mu0_projection           = False
        self.bdim_body_div_correction      = True
        # rmgcg (recycled multigrid-preconditioned CG).  Previously this BLEW UP
        # ~iter 1200 while multigrid survived, because the MG V-cycle used as the
        # CG preconditioner was NON-SYMMETRIC (restriction was sum-of-8-children,
        # prolongation is trilinear, and sum-of-8 != trilinear^T), which is
        # invalid for CG -- mgcg/rmgcg then converged WORSE than plain multigrid.
        # FIXED (2026-07-15): the CG-preconditioner V-cycle now uses full-
        # weighting restriction = P^T (csrc .../multigrid_transfer.cu
        # restrict_fw_*, gated by the `variational` flag in vcycle_{2,3}d), so it
        # is symmetric to machine precision.  mgcg/rmgcg now beat multigrid at
        # equal V-cycle work (bench 80:1: mgcg-30 resid 1.4e-6 vs multigrid-30
        # 9.8e-6) and match it at ~1/3 the cost.  Standalone `multigrid` is
        # unchanged (still sum-of-8, robust) and remains the conservative
        # fallback if anything regresses.  Raise poisson_max_mgcg_cycles below for
        # an even lower per-step residual.
        #
        # NOTE: mgcg/rmgcg allocate transient GPU buffers every step, so they need
        # expandable_segments (else the caching allocator can cudaMalloc DURING the
        # pre-projection CUDA-graph capture -> intermittent 'operation failed due
        # to a previous error during capture', crashing after a few steps).  This
        # is auto-applied in the generated run.sh (base_sim_config.gen_sh_config).
        # self.poisson_method                = "rmgcg"
        self.poisson_method                = "multigrid"   # conservative fallback

        self.headless             = False
        self.smagorinsky_cs       = 0.


        # ── Two-phase free-surface flags ──────────────────────────────
        # BDIM handles all hydrodynamics; disable FARMS buoyancy/drag
        self.water_drag     = False
        self.water_buoyancy = False
        self.water_height   = WATERLINE

        # ── Animats ───────────────────────────────────────────────────
        self.animats_pars = [
            {
                "model_name"     : "1guilla",
                # Ventral-ballast SDF: identical mass/inertia/geometry to
                # 1guilla_800.sdf (still uniform rho=800, same weight + BDIM
                # buoyancy) but each link's COM lowered 3 cm, giving a passive
                # righting moment that keeps the planar gait from rolling the
                # body over (gait roll 24.5deg -> ~5deg). Regenerate/tune with
                # sdfs/1guilla/make_ballast_sdf.py.
                "sdf_name"       : "1guilla_600.sdf",
                "control_type"   : "position",
                "gains"          : [100.0, 1., 0],
                "spawn_mode"     : SpawnMode.FREE,
                "pose"           : [4.75, 0.1, -0.0, 0, 0, 0.05],
                "controller_path": "lilytorch.examples._1guillasim.experiments.controller.PositionController",
                "control_pars"   : {
                    "file_path": os.path.join(
                        # self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms004mpt003log.csv"
                        self.data_folder, "/data/andreaferrario/1guilla_experiments/swim/log/ms001mpt001log.csv"
                    ),
                },
            },
        ]

        # ── 3-D grid ─────────────────────────────────────────────────
        # self.Nx   = 600*2
        # self.Ny   = 300*2
        # self.Nz   = 52*2
        # self.xmin = 2
        # self.xmax = 6
        # self.ymin = -1
        # self.ymax = 1
        # self.zmin = -(2/300*52/2)
        # self.zmax = (2/300*52/2)


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

        # # ── 3-D grid ─────────────────────────────────────────────────
        # self.Nx   = 450
        # self.Ny   = 150
        # self.Nz   = 52
        # self.xmin = 3
        # self.xmax = 6
        # self.ymin = -0.5
        # self.ymax = 0.5
        # self.zmin = -(2/300*52/2)
        # self.zmax = (2/300*52/2)


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
        # poisson_tol is an ABSOLUTE L-inf residual, and the two-phase pressure
        # carries hydrostatics (~1.3e3 Pa here).  At 1e-8 the solve can never
        # reach it, so mgcg simply burned all 10 CG iterations every step and
        # stopped wherever it happened to be: the projection left max|div u|~5
        # and pumped ~60x too much kinetic energy into the fluid.  1e-5 is
        # reachable from the warm start below.  Measured, 160-step coupled run:
        # 4.2 -> 8.9 it/s, median max|div u| 4.9 -> 0.43, E_k 1.8e-3 -> 3.0e-5.
        self.poisson_tol             = 1.0e-5
        self.poisson_max_cycles      = 30
        self.poisson_max_mgcg_cycles = 10
        self.poisson_precond_vcycles = 1
        # Reuse the previous pressure as the Poisson guess.  This flag was a
        # silent no-op until the stale `not has_custom_coeffs` guard was removed
        # from solver.project (the streaming path always passes coefficients),
        # so the solve was cold-starting and rebuilding the whole hydrostatic
        # field from zero every step.
        self.poisson_warm_start      = True
        self.poisson_smoother        = "jacobi"
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "neumann"
        self.empty_cache_every       = 10**9
        # self.coupling = {
        #     "scheme"     : "implicit",
        #     "accelerator": "iqn-ils",    # or "aitken" / "constant"
        #     "reuse"      : 2,
        #     "tol"        : 1e-4,
        #     "max_iter"   : 30,
        # }

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
        self.wall_alpha        = 0.
        self.floor_color       = None
        self.water_alpha       = 0.2
        self.grid_spacing      = None                #0.5*(self.ymax - self.ymin)


    # ── Two-phase BDIM extension ──────────────────────────────────────
    def _bdim_extension(self, output_folder):
        bdim_ext                      = super()._bdim_extension(output_folder)
        solver                        = bdim_ext["config"]["bdim_yaml"]["solver"]
        # solver["graph_capture_debug"] = True

        # solver["gravity"] = [0, 0, -9.81]
        # solver["two_phase"] = {
        #     "alpha_init"             : f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
        #     "rho_water"              : 1000.0,
        #     "rho_air"                : 1.2,                                             # 80:1 stability cap
        #     "nu_water"               : self.nu,
        #     "nu_air"                 : 1.5e-5,
        #     "alpha_exclude_body"     : True,
        #     "alpha_volume_compensate": True,
        #     "air_transparent_body"   : False,
        #     # "consistent_momentum"    : True,
        # }

        solver["gravity"] = [0, 0, -9.81]
        solver["two_phase"] = {
            "alpha_init"             : f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
            "rho_water"              : 1000.0,
            # 80:1 stability cap.  This MUST NOT be the physical 1.2 (=833:1):
            # the projection coeff is dt*mu0/rho, so in air the solver converts
            # any residual pressure error into velocity with a gain of
            # rho_water/rho_air.  The variable-density V-cycle contracts only
            # ~0.99/cycle on this operator, so the Poisson never fully converges
            # and a sub-1% residual at 833:1 blows the air velocity up (NaN ->
            # MuJoCo BADQACC) around iter 1200-1400.  12.5 (80:1) amplifies the
            # same residual 66x less and the air velocity saturates instead.
            "rho_air"                : 1.5,
            "nu_water"               : self.nu,
            "nu_air"                 : 1.5e-5,
            "alpha_exclude_body"     : True,
            "alpha_volume_compensate": True,
            "air_transparent_body"   : False,
            "consistent_momentum"    : False,  # requires solver_method='python' (not kernel)
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

        # reduce overall scene light: lower diffuse and ambient
        extensions.append({
            "loader": "lilytorch.integration.light_modifier.LightModifier",
            "config": {
                # dim the main light
                "diffuse": [0.6, 0.6, 0.6],
                # reduce ambient/global illumination
                "ambient": [0.1, 0.1, 0.1],
            },
        })

        # # Air/water interface + vorticity wake visualisation
        # extensions.append({
        #     "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
        #     "config": {
        #         "update_every"      : 1,
        #         "max_vertices"      : 20 * self.Nx * self.Ny,
        #         "crop_boundary"     : 0,
        #         "debug_force_visible": False,
        #         "fields": [
        #             # {
        #             #     "field"     : "interface",
        #             #     "iso_value" : 0.5,
        #             #     "alpha"     : 0.45,
        #             #     "color"     : "#3399FF",
        #             #     "smooth_sigma": 0,
        #             #     "exclude_body": False,
        #             #     "reflective": True,
        #             # },
        #             # {
        #             #     "field"     : "omega_mag",
        #             #     "iso_value" : 10.0,
        #             #     "alpha"     : 0.3,
        #             #     "color"     : "#FF8C1A",
        #             #     "smooth_sigma": 0,
        #             #     "exclude_body": True,
        #             #     "phase_mask": "water",
        #             # },
        #         ],
        #     },
        # })

        # # Top-down camera auto-fitted to the pool
        # cam = top_down_camera_config(
        #     self.xmin, self.xmax,
        #     self.ymin, self.ymax,
        #     self.zmin, self.zmax,
        #     overshoot=1,
        #     max_width=3840, max_height=2160,
        # )
        # extensions.append({
        #     "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
        #     "config": {
        #         "path"            : os.path.join(output_folder, "output", "video.mp4"),
        #         "animat_id"       : None,
        #         "fps"             : 30,
        #         "speed"           : 1.0,
        #         "angular_velocity": 0,
        #         **cam,
        #     },
        # })

        # # Side camera auto-fitted to the pool
        # cam_side = side_camera_config(
        #     self.xmin, self.xmax,
        #     self.ymin, self.ymax,
        #     self.zmin, self.zmax,
        #     overshoot=1,
        #     max_width=3840, max_height=2160,
        # )
        # extensions.append({
        #     "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
        #     "config": {
        #         "path"            : os.path.join(output_folder, "output", "video_side.mp4"),
        #         "animat_id"       : None,
        #         "fps"             : 30,
        #         "speed"           : 1.0,
        #         "angular_velocity": 0,
        #         **cam_side,
        #     },
        # })


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
