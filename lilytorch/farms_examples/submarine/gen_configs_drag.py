"""Submarine simulation in MuJoCo using the FARMS drag model.

This example runs a single rigid-body submarine in a 3-D pool, with
hydrodynamic forces approximated by the FARMS drag model
(``use_bdim = False`` => ``use_drag = True`` in :class:`BaseSimConfig`).

It is intentionally written as a *template* mirroring the layout of the
other examples in ``lilytorch/farms_examples`` (see
``_1guillasim/gen_configs_one_free_3d.py`` or ``salamander/gen_configs_swim_3d.py``):
all BDIM / fluid-coupling knobs are present and grouped, but disabled, so
that switching to a coupled fluid simulation later only requires flipping
``self.use_bdim = True`` (and tuning the relevant grid / solver fields).

Run with::

    python -m lilytorch.farms_examples.submarine.gen_configs_drag
"""

import os

from farms_core.model.options import SpawnMode

from lilytorch.util.paths import lilytorch_repo_root
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from lilytorch.integration.camera import top_down_camera_config


class SimConfig(BaseSimConfig):

    def __init__(self):
        super().__init__()

        self.data_folder = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'submarine',
        )

        # ── Hardware ──────────────────────────────────────────────────
        self.use_gpu  = True
        self.headless = False

        # ── Simulation flags ──────────────────────────────────────────
        # Drag-only: the FARMS drag model is enabled automatically when
        # use_bdim is False (see BaseSimConfig.use_drag).  Flip this to
        # True (and tune the BDIM section below) to couple with the
        # fluid solver.
        self.use_bdim = False

        # ── Animats ───────────────────────────────────────────────────
        # Submarine with a propeller revolute joint.  The propeller is
        # driven at a constant angular velocity ``omega`` by
        # PropellerController, and two pitched blade links (with
        # anisotropic drag coefficients set in customize_morphology_links)
        # convert tangential drag into axial thrust.
        self.animats_pars = [
            {
                "model_name"  : "submarine",
                "sdf_name"    : "submarine.sdf",
                "control_type": "torque",
                "gains"       : [0.0, 0.0, 0.0],
                "controller_config": {
                    "path": "lilytorch.farms_examples.submarine."
                            "propeller_controller.PropellerController",
                    "tau" : 0.1,   # constant torque on the propeller joint (Nm)
                },
                "spawn_mode"  : SpawnMode.FREE,
                # Spawn the sub at mid-depth in the pool, oriented along +X.
                "pose"        : [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ]

        # ── 3-D grid (used by BDIM and to size the pool arena) ───────
        self.Nx   = 256
        self.Ny   = 128
        self.Nz   = 64
        self.xmin = -1.0
        self.xmax =  1.0
        self.ymin = -0.5
        self.ymax =  0.5
        self.zmin = -0.25
        self.zmax =  0.25

        # ── Physics ───────────────────────────────────────────────────
        # Submarine is neutrally buoyant: body density matches the water.
        self.rho_body          = 1000.0  # matches the SDF inertia mass
        self.rho               = 1000.0  # water density
        self.timestep          = 0.001
        self.n_iterations      = 4001
        self.save_every        = 50
        self.num_sub_steps     = 1

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 1.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        # Pool walls auto-sized from the grid extents, water filled up
        # to the top of the pool.  Drag and buoyancy are driven by the
        # FARMS drag model when use_bdim=False.
        self.wall_thickness = 0.01
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        # (water_drag / water_buoyancy default to use_drag, i.e. True)

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
        self.poisson_compile         = True
        self.poisson_nsmoothing      = 5
        self.poisson_bc_type         = "free"
        self.compile_adv_diff        = True

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

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_morphology_links(self, links_list, animat_i, animat_pars, index):
        # Anisotropic drag on the blades: high normal-to-face (blade X,
        # the thickness axis), low along chord (Z) and span (Y).  Together
        # with the blade pitch this turns tangential rotation into +X thrust.
        for link in links_list:
            if link['name'] in ('blade_0', 'blade_1'):
                link['drag_coefficients'] = [
                    [-0.5, -0.01, -0.05],
                    [0.0, 0.0, 0.0],
                ]

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
