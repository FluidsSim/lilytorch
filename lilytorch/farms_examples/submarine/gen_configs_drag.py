"""Submarine simulation coupled to the 3-D BDIM fluid solver.

This example runs a fully free submarine in a 3-D pool with two-way fluid
coupling handled by lilytorch's BDIM solver. The default propulsion setup
keeps the single propeller plus tail fins so the wake is resolved by the
fluid model instead of approximated by anisotropic drag coefficients.

The file still keeps the drag-only link overrides used in the legacy
``use_bdim = False`` path, but the default configuration below is the
fluid-coupled one.

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
        self.use_bdim = True
        # With BDIM enabled, links do not automatically opt into fluid
        # forces because BaseSimConfig otherwise ties fluid interaction
        # to use_drag. Force it on explicitly for the submarine bodies.
        self.animat_fluid_interaction = True

        submarine_sdf_dir = os.path.join(
            lilytorch_repo_root, 'farms_examples', 'sdfs', 'submarine',
        )

        # In free 3-D, a single propeller injects a steady reaction torque
        # into the hull. Use the contra-rotating layout by default so the
        # submarine remains torque-balanced without relying on artificial
        # roll constraints.
        propulsion_variant = 'contra_rotating'
        if propulsion_variant == 'single_propeller':
            sdf_file = os.path.join(submarine_sdf_dir, 'submarine.sdf')
            controller_config = {
                "path": "lilytorch.farms_examples.submarine."
                        "propeller_controller.PropellerController",
                "tau": 0.1,
            }
        elif propulsion_variant == 'tail_fins':
            sdf_file = os.path.join(submarine_sdf_dir, 'submarine_tail_fins.sdf')
            controller_config = {
                "path": "lilytorch.farms_examples.submarine."
                        "propeller_controller.PropellerController",
                "tau": 0.1,
            }
        elif propulsion_variant == 'contra_rotating':
            shaft_tau = 0.05
            sdf_file = os.path.join(submarine_sdf_dir, 'submarine_contra.sdf')
            controller_config = {
                "path": "lilytorch.farms_examples.submarine."
                        "propeller_controller.PropellerController",
                "joint_torques": {
                    "joint_propeller_front": shaft_tau,
                    "joint_propeller_rear": -shaft_tau,
                },
            }
        else:
            raise ValueError(f"Unsupported propulsion_variant: {propulsion_variant}")

        # ── Animats ───────────────────────────────────────────────────
        # The propeller blades use anisotropic drag coefficients set in
        # customize_morphology_links so their tangential motion produces
        # axial thrust.
        self.animats_pars = [
            {
                "model_name"  : "submarine",
                "sdf_file"    : sdf_file,
                "control_type": "torque",
                "gains"       : [0.0, 0.0, 0.0],
                "controller_config": controller_config,
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
        self.n_iterations      = 20001
        self.save_every        = 50
        self.num_sub_steps     = 1

        # ── MuJoCo ───────────────────────────────────────────────────
        self.visual_scale = 1.0
        self.extent       = 10.0

        # ── Arena ────────────────────────────────────────────────────
        # Pool walls auto-sized from the grid extents, water filled up
        # to the top of the pool. In the BDIM path the fluid solver
        # provides the hydrodynamic forces, so the legacy water drag /
        # buoyancy toggles remain disabled.
        self.wall_thickness = 0.01
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = False
        # (water_drag / water_buoyancy default to use_drag, i.e. True)

        # ── BDIM solver ──────────────────────────────────────────────
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

        # Body sampling for BDIM. The submarine SDFs are primitive-based,
        # so no explicit mesh interpolation grid is needed here; if mesh
        # collisions are added later, the 3-D body path will auto-size the
        # per-axis sampling from the fluid grid spacing.
        self.contour_mask          = True
        self.force_scaling         = 1.0
        self.interp_data_subfolder = "interp_data_3d"

    # ── Hooks ─────────────────────────────────────────────────────────

    def customize_morphology_links(self, links_list, animat_i, animat_pars, index):
        del animat_i, animat_pars, index

        # These drag coefficients are only used in the legacy drag-only
        # fallback (`use_bdim=False`). The BDIM configuration above ignores
        # them and takes hydrodynamic forces from the fluid solver instead.
        for link in links_list:
            # Hull: give it realistic skin-friction roll drag about its own
            # long axis. Without this the cylinder is free-spinning and any
            # steady shaft reaction torque spins it up to a large terminal
            # omega regardless of hull inertia. Pitch/yaw angular drag are
            # less critical but set to a sane nonzero value too.
            if link['name'] == 'base_link':
                link['drag_coefficients'] = [
                    [-0.1, -5.0, -5.0],
                    [-0.05, -0.5, -0.5],
                ]
                # The 3-D BDIM path applies buoyancy explicitly as a force.
                # Add a small positive metacentric offset so the buoyancy
                # force acts above the base-link CG and generates a restoring
                # roll moment instead of allowing free 360-degree roll.
                link['extras']['metacentric_offset'] = [0.0, 0.0, 0.03]

            if link['name'] in (
                    'blade_0', 'blade_1',
                    'front_blade_0', 'front_blade_1',
                    'rear_blade_0', 'rear_blade_1',
            ):
                link['drag_coefficients'] = [
                    [-0.5, -0.1, -0.3],
                    [0.0001, 0.01, 0.01],
                ]

            # Tail fins only add passive roll damping. They do not cancel
            # steady propeller reaction torque, but they do resist hull roll.
            if link['name'] in ('fin_top', 'fin_bottom'):
                link['drag_coefficients'] = [
                    [-0.3, -1.5, -0.3],
                    [0.0001, 0.01, 0.01],
                ]
            if link['name'] in ('fin_port', 'fin_starboard'):
                link['drag_coefficients'] = [
                    [-0.3, -0.3, -1.5],
                    [0.0001, 0.01, 0.01],
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
