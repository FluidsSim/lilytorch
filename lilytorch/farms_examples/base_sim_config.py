"""Base simulation configuration class for FARMS experiments.

All gen_config scripts inherit from this class and override only the
parameters and hooks that differ from the defaults.

Usage example::

    from lilytorch.farms_examples.base_sim_config import BaseSimConfig

    class SimConfig(BaseSimConfig):
        def __init__(self):
            super().__init__()
            # override only what differs
            self.Nx = 1024
            ...

    if __name__ == "__main__":
        SimConfig().run()
"""

from math import inf
import os
import subprocess
import sys
import numpy as np

from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode
from farms_core.io.sdf import ModelSDF

from lilytorch.util.paths import (
    lilytorch_repo_root, sdfs_path, gen_new_folder, save_path,
)
from lilytorch.integration.gen_pool_sdf import create_pool_sdf, create_water_sdf


class BaseSimConfig:
    """Base simulation configuration.

    Subclass and override ``__init__`` (calling ``super().__init__()``
    first) to set simulation-specific parameters. The ``gen_*`` methods
    and ``single_run`` are shared across all configurations.

    Override hook methods for custom behaviour:

    * ``customize_animat``          – per-animat / per-index tweaks
    * ``customize_morphology_links`` – override per-link morphology fields
    * ``customize_joint_initials``  – override joint initial positions
    * ``extra_simulation_extensions`` – add FlowViewer, CameraRecording, etc.
    """

    def __init__(self):
        # ── Paths ─────────────────────────────────────────────────────────
        self.stack_folder      = save_path
        self.data_folder       = None  # MUST be set by subclass
        self.bdim_handler_path = "lilytorch.integration.BDIMhandler.BDIMhandler"
        self.interp_data_subfolder = "interp_data"

        # ── Hardware ──────────────────────────────────────────────────────
        self.nthreads = 24
        self.use_gpu  = True

        # ── Simulation flags ──────────────────────────────────────────────
        self.use_bdim = True
        self.headless = False
        self.fast     = False

        # ── Drag ──────────────────────────────────────────────────────────
        self._use_drag_override = None   # None → auto (not use_bdim)
        self.constant_drags = [
            [-0.1, -5.0, -5.0],
            [-0.001, -0.001, -0.001],
        ]

        # ── Animats ───────────────────────────────────────────────────────
        self.animats_pars = []
        self.filter_fixed_joints = True  # skip joints with type=="fixed"

        # ── Grid (2-D by default; set Nz/zmin/zmax for 3-D) ──────────────
        self.Nx   = 512
        self.Ny   = 128
        self.Nz   = None
        self.xmin = -0.9
        self.xmax = 1.5
        self.ymin = -0.3
        self.ymax = 0.3
        self.zmin = None
        self.zmax = None

        # ── Fluid / body physics ──────────────────────────────────────────
        self.rho_body          = 1000.0
        self.rho               = 1000.0
        self.nu                = 1.0e-6
        self.timestep          = 0.001
        self.convection_method = "quick"
        self.n_iterations      = 10001

        # ── Output ────────────────────────────────────────────────────────
        self.save_frames = True
        self.save_every  = 200
        self.save        = False   # save fields (u, v, [w], p, sdf) to HDF5
        self.vmin        = -10.0
        self.vmax        = 10.0
        self.plot_specs  = ["curl", "pressure"]
        self.iso_3d_specs = ["omega_mag", "vel_mag"]
        self.iso_3d_value = None

        # ── Arena ─────────────────────────────────────────────────────────
        self.generate_pool  = True   # generate pool/water SDFs; False → use flat arena
        self.wall_thickness = None   # None → auto for 3-D, 0.3 for 2-D
        self.wall_height    = None   # None → 0.3 for 2-D (ignored for 3-D)
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = None   # None → use_drag
        self.water_buoyancy = None   # None → use_drag
        self.water_height   = 0
        self.ground_height  = 0.0

        # ── Animat fluid interaction ──────────────────────────────────────
        self.animat_fluid_interaction = None  # None → use_drag

        # ── MuJoCo ────────────────────────────────────────────────────────
        self.num_sub_steps = 1
        self.cb_sub_steps  = 1
        self.visual_scale  = 1.0
        self.extent        = 400.0
        self.shadow_size   = 0
        self.viewer_native_body_colors = True
        self.viewer_body_alpha = None
        self.viewer_hide_collision_geoms_with_visuals = True

        # ── BDIM solver ───────────────────────────────────────────────────
        self.bdim_dt                 = 0.0001
        self.bdim_nt                 = 800000
        self.poisson_method          = None
        self.poisson_tol             = 1.0e-7
        self.poisson_max_cycles      = 5
        self.poisson_max_mgcg_cycles = 3
        self.poisson_precond_vcycles = None
        self.poisson_warm_start      = None
        self.poisson_smoother        = None
        self.poisson_nsmoothing      = 10
        self.poisson_verbose         = False
        self.poisson_bc_type         = "neumann"
        self.poisson_compile         = False
        self.compile_adv_diff        = False
        self.compile_forces          = False
        self.compile_sdf             = False
        self.smagorinsky_cs          = 0.0
        self.carreau                 = None   # dict with keys: nu_0, nu_inf, lam, n
        self.sponge                  = None   # dict with keys: width, strength
        self.yield_damping           = None   # dict with keys: gamma_c, strength (auto-derived from carreau.tau_y if None)
        self.jacobi_weight           = 0.7
        self.dtype                   = None
        self.zero_pressure_inside    = None
        self.force_method            = None
        self.time_integration        = None
        self.streaming_sdf_3d        = None
        self.streaming_forces_3d     = None
        self.force_shared_union      = None
        self.mu_normals_union        = None
        self.bdim_union              = None
        self.force_narrow_batch      = None
        # Body-SDF sampling method for the streaming kernels:
        #   "trilinear" (default) | "triquadratic"
        self.sdf_interp_method       = None
        # Fluid-explosion guard (forwarded into bdim_yaml.solver)
        self.vmax_abort              = None   # m/s; None = auto

        # ── BDIM boundary conditions ──────────────────────────────────────
        # 2-D: 4 entries;  3-D: 6 entries
        self.bc_type_u   = ["N", "N", "N", "N"]
        self.bc_values_u = [0, 0, 0, 0]
        self.bc_type_v   = ["N", "N", "N", "N"]
        self.bc_values_v = [0, 0, 0, 0]
        self.bc_type_w   = None   # 3-D only
        self.bc_values_w = None

        # ── BDIM body config ──────────────────────────────────────────────
        self.n_samples      = None
        self.contour_mask   = None
        self.convexify      = False
        self.force_scaling  = None
        self.compute_sdf    = False
        self.suit           = 0.0

        # ── Global MuJoCo contact tweaks (also forwarded into BDIM YAML) ───
        self.bdim_physics = None

        # ── Lock the configuration ────────────────────────────────────────
        self._config_frozen = True

    def __setattr__(self, key, value):
        if key == "_config_frozen":
            super().__setattr__(key, value)
            return

        if getattr(self, "_config_frozen", False) and not hasattr(self, key):
            raise AttributeError(
                f"Attribute '{key}' involves no existing configuration parameter "
                f"in {self.__class__.__name__}.\n"
                f"Did you mean to modify an existing parameter? Check for typos.\n"
                f" allowed keys are: {sorted(self.__dict__.keys())}"
            )
        super().__setattr__(key, value)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_3d(self):
        return self.Nz is not None

    @property
    def use_drag(self):
        if self._use_drag_override is not None:
            return self._use_drag_override
        return not self.use_bdim

    @use_drag.setter
    def use_drag(self, value):
        self._use_drag_override = value

    @property
    def _fluid_interaction(self):
        if self.animat_fluid_interaction is not None:
            return self.animat_fluid_interaction
        return self.use_drag

    @property
    def _water_drag(self):
        return self.water_drag if self.water_drag is not None else self.use_drag

    @property
    def _water_buoyancy(self):
        return self.water_buoyancy if self.water_buoyancy is not None else self.use_drag

    # ── Hooks (override in subclasses) ────────────────────────────────────

    def customize_animat(self, animat_i, animat_pars, n_joints, index):
        """Called for each animat before building the config dict.

        Override to modify *animat_pars* per-index (e.g. set oscillator
        frequency or initial state).
        """

    def customize_morphology_links(self, links_list, animat_i, animat_pars, index):
        """Called after the links list is built.

        Override to customize per-link morphology fields such as friction,
        density, fluid interaction, or drag coefficients.
        """

    def customize_joint_initials(self, joints_list):
        """Called after the joints list is built.

        Override to set initial positions for specific joints
        (e.g. leg joints for salamanders).
        """

    def extra_simulation_extensions(self, output_folder):
        """Return a list of extra extension dicts to append after BDIM.

        Override to add FlowViewer, CameraRecording, CameraFollower, etc.
        """
        return []

    # ── Config generators ─────────────────────────────────────────────────

    def gen_animat_config(self, output_folder, index=0):
        for animat_i, animat_pars in enumerate(self.animats_pars):

            # Resolve SDF path
            if "sdf_file" in animat_pars:
                sdf_file = animat_pars["sdf_file"]
            else:
                sdf_file = os.path.join(
                    sdfs_path,
                    animat_pars["model_name"],
                    animat_pars["sdf_name"],
                )

            model_sdf   = ModelSDF.read(sdf_file)[0]
            link_names  = [link.name for link in model_sdf.links]
            if self.filter_fixed_joints:
                joint_names = [j.name for j in model_sdf.joints
                               if j.type != "fixed"]
            else:
                joint_names = [j.name for j in model_sdf.joints]
            nlinks   = len(link_names)
            n_joints = len(joint_names)

            # Per-index hook
            self.customize_animat(animat_i, animat_pars, n_joints, index)

            # Resolve controller extension
            if "controller_path" in animat_pars:
                ext_loader = animat_pars["controller_path"]
                ext_config = animat_pars["control_pars"]
            elif "muscle_loader" in animat_pars:
                ext_loader = animat_pars["muscle_loader"]
                ext_config = animat_pars["muscle_config"]
            elif "controller_config" in animat_pars:
                ext_loader = animat_pars["controller_config"]["path"]
                ext_config = animat_pars["controller_config"]
            else:
                ext_loader = ""
                ext_config = {}

            control_type = animat_pars["control_type"]
            gains        = animat_pars["gains"]
            spawn_mode   = animat_pars["spawn_mode"]
            pose         = animat_pars["pose"]

            animat_extensions = []
            if ext_loader:
                animat_extensions.append({
                    "loader": ext_loader,
                    "config": ext_config,
                })

            drag_coefficients = [self.constant_drags for _ in range(nlinks)]

            # == Build animat dict ==
            animat_dict = {
                "spawn": {
                    'loader'  : 0,
                    'mode'    : spawn_mode,
                    'pose'    : pose,
                    'velocity': [0, 0, 0, 0, 0, 0],
                    'extras'  : {},
                },
                "sdf": sdf_file,
                "morphology": {
                    "links": [
                        {
                            'name'             : ln,
                            'collisions'       : True,
                            'friction'         : [0.2, 0, 0],
                            'extras'           : {},
                            'fluid_interaction': self._fluid_interaction,
                            'density'          : self.rho_body,
                        } for ln in link_names
                    ],
                    "joints": [
                        {
                            'name'     : jn,
                            'initial'  : [0, 0],
                            'limits'   : [[-inf, inf], [-inf, inf]],
                            'stiffness': 0,
                            'springref': 0,
                            'damping'  : 0,
                            'extras'   : {},
                        } for jn in joint_names
                    ],
                    "self_collisions": [],
                },
                "control": {
                    "sensors": {
                        "links"    : link_names,
                        "joints"   : joint_names,
                        "contacts" : [(ln, '') for ln in link_names],
                        "xfrc"     : link_names,
                        "muscles"  : [],
                        "adhesions": [],
                        "visuals"  : [],
                    },
                    "motors": [
                        {
                            'joint_name'   : jn,
                            'control_types': [control_type],
                            'limits_torque': [-inf, inf],
                            'gains'        : list(gains),
                        } for jn in joint_names
                    ],
                },
                "extensions": animat_extensions,
            }

            # Add drag coefficients when using drag model
            if self.use_drag:
                for i, link in enumerate(animat_dict["morphology"]["links"]):
                    link["drag_coefficients"] = drag_coefficients[i]

            self.customize_morphology_links(
                animat_dict["morphology"]["links"],
                animat_i,
                animat_pars,
                index,
            )

            # Joint initial overrides hook
            self.customize_joint_initials(
                animat_dict["morphology"]["joints"]
            )

            # Swimming extension for drag-based sims
            if self.use_drag:
                animat_dict["extensions"].append({
                    "loader": "farms_mujoco.swimming.extension.SwimmingExtension",
                    "config": {"water_properties": None},
                })

            pyobject2yaml(
                os.path.join(output_folder,
                             f"animat_config_{animat_i}.yaml"),
                animat_dict,
            )

    def gen_arena_config(self, output_folder, index=0):
        if self.generate_pool:
            if self.is_3d:
                pool_dims = [self.xmax - self.xmin,
                             self.ymax - self.ymin,
                             self.zmax - self.zmin]
                wt = self.wall_thickness
                if wt is None:
                    wt = max(round(0.08 * min(pool_dims), 4), 0.01)
                create_pool_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    zmin=self.zmin, zmax=self.zmax,
                    wall_thickness=wt, plotting=False,
                )
                water_sdf = create_water_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    zmin=self.zmin, zmax=self.zmax,
                    water_height=self.zmax,
                )
                water_h = self.water_height if self.water_height != 0 else self.zmax
            else:
                wt = self.wall_thickness if self.wall_thickness is not None else 0.3
                wh = self.wall_height if self.wall_height is not None else 0.3
                create_pool_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    wall_thickness=wt, wall_height=wh, plotting=False,
                )
                water_sdf = create_water_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    water_height=self.water_height,
                    wall_height=wh,
                )
                water_h = self.water_height
            arena_sdf = os.path.join(sdfs_path, "pool", "sdf", "pool.sdf")
        else:
            arena_sdf = os.path.join(sdfs_path, "arena_flat_v0", "sdf", "arena_flat.sdf")
            water_sdf = os.path.join(sdfs_path, "arena_water_v0", "sdf", "arena_water.sdf")
            water_h = self.water_height

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
                "height"   : water_h,
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

    def gen_experiment_config(self, output_folder, index=0):
        n = len(self.animats_pars)
        experiment_dict = {
            "simulation": "simulation_config.yaml",
            "arenas"    : ["arena_config.yaml"],
            "animats"   : [f"animat_config_{i}.yaml" for i in range(n)],
            "loaders"   : {
                "simulation_options": "farms_core.simulation.options.SimulationOptions",
                "animats_options"   : ["farms_core.model.options.AnimatOptions"
                                       for _ in range(n)],
                "arenas_options"    : ["farms_core.model.options.ArenaOptions"],
                "experiment_data"   : "farms_core.experiment.data.ExperimentData",
                "animats_data"      : ["farms_core.model.data.AnimatData"
                                       for _ in range(n)],
            },
        }
        pyobject2yaml(
            os.path.join(output_folder, 'experiment_config.yaml'),
            experiment_dict,
        )

    def gen_simulation_config(self, output_folder, index=0):
        simulation_dict = {
            "units": {
                "length": "meter",
                "mass"  : "kilogram",
                "time"  : "second",
            },
            "runtime": {
                "n_iterations" : self.n_iterations,
                "buffer_size"  : self.n_iterations,
                "play"         : True,
                "rtl"          : 1.0,
                "fast"         : self.fast,
                "headless"     : self.headless,
                "show_progress": True,
            },
            "physics": {
                "timestep"      : self.timestep,
                "gravity"       : [0, 0, -9.81],
                "num_sub_steps" : self.num_sub_steps,
                "cb_sub_steps"  : self.cb_sub_steps,
                "n_solver_iters": 50,
            },
            "mujoco": {
                "cone"             : "elliptic",
                "solver"           : "CG",
                "integrator"       : "implicitfast",
                "impratio"         : 10,
                "ccd_iterations"   : 1000,
                "ccd_tolerance"    : 1e-6,
                "noslip_iterations": 1000,
                "noslip_tolerance" : 1e-6,
                "viewer"           : "MuJoCo",
                "texture_repeat"   : 1,
                "shadow_size"      : self.shadow_size,
                "visual_scale"     : self.visual_scale,
                "extent"           : self.extent,
            },
            "extensions": [
                {
                    "loader": "farms_core.simulation.extensions.ExperimentLogger",
                    "config": {
                        "log_path": os.path.join(output_folder, "output"),
                        "skip": 0,
                    },
                },
                {
                    "loader": "farms_mujoco.simulation.extensions.MjcfSaver",
                    "config": {
                        "path": os.path.join(
                            output_folder, "output", "simulation_mjcf.xml"
                        ),
                    },
                },
                {
                    "loader": "lilytorch.integration.extensions.DataLogger",
                    "config": {
                        "log_path": os.path.join(
                            output_folder, "output", "nn_data.hdf5"
                        ),
                    },
                },
            ],
        }

        if self.viewer_native_body_colors:
            simulation_dict["extensions"].append({
                "loader": "lilytorch.integration.native_body_colors.NativeBodyColors",
                "config": {
                    "alpha": self.viewer_body_alpha,
                    "hide_collision_geoms_with_visuals": self.viewer_hide_collision_geoms_with_visuals,
                },
            })

        if self.bdim_physics is not None:
            simulation_dict["extensions"].append({
                "loader": "lilytorch.integration.extensions.PhysicsOptionsExtension",
                "config": {
                    "physics_options": self.bdim_physics,
                },
            })

        if self.use_bdim:
            simulation_dict["extensions"].append(
                self._bdim_extension(output_folder)
            )

        # Extra extensions (FlowViewer, CameraRecording, etc.)
        simulation_dict["extensions"] += self.extra_simulation_extensions(
            output_folder
        )

        pyobject2yaml(
            os.path.join(output_folder, 'simulation_config.yaml'),
            simulation_dict,
        )

    def gen_sh_config(self, output_folder, index=0):
        sh_str = (
            "#!/bin/bash\n"
            "set -e\n"
            f'"{sys.executable}" -c "from farms_sim._bootstrap import main; main()" '
            '--experiment_config experiment_config.yaml "$@"\n'
        )
        with open(os.path.join(output_folder, 'run.sh'), 'w') as f:
            f.write(sh_str)

    # ── Orchestration ─────────────────────────────────────────────────────

    def single_run(self, index=0):
        output_folder = gen_new_folder(self.stack_folder)
        os.makedirs(output_folder, exist_ok=True)
        print("Saving configs to folder:", output_folder)

        self.gen_animat_config(output_folder, index)
        self.gen_arena_config(output_folder, index)
        self.gen_simulation_config(output_folder, index)
        self.gen_experiment_config(output_folder, index)
        self.gen_sh_config(output_folder, index)
        os.chdir(output_folder)
        subprocess.run(['bash', 'run.sh'], check=True)

    def run(self):
        """Run all configurations. Override for multi-run sweeps."""
        self.single_run()

    # ── Private helpers ───────────────────────────────────────────────────

    def _bdim_extension(self, output_folder):
        solver = {
            "use_gpu"                : self.use_gpu,
            "nthreads"               : self.nthreads,
            "Nx"                     : self.Nx,
            "Ny"                     : self.Ny,
            "xmin"                   : self.xmin,
            "xmax"                   : self.xmax,
            "ymin"                   : self.ymin,
            "ymax"                   : self.ymax,
            "convection_method"      : self.convection_method,
            "dt"                     : self.bdim_dt,
            "nt"                     : self.bdim_nt,
            "nu"                     : self.nu,
            "rho"                    : self.rho,
            "poisson_tol"            : self.poisson_tol,
            "poisson_max_cycles"     : self.poisson_max_cycles,
            "poisson_max_mgcg_cycles": self.poisson_max_mgcg_cycles,
            "jacobi_weight"          : self.jacobi_weight,
            "poisson_nsmoothing"     : self.poisson_nsmoothing,
            "poisson_verbose"        : self.poisson_verbose,
            "poisson_folder"         : os.path.join(self.data_folder, "data"),
            "rho_body"               : self.rho_body,
            "smagorinsky_cs"         : self.smagorinsky_cs,
        }

        if self.carreau is not None:
            solver["carreau"] = self.carreau

        if self.sponge is not None:
            solver["sponge"] = self.sponge

        if self.yield_damping is not None:
            solver["yield_damping"] = self.yield_damping

        if self.is_3d:
            solver["Nz"]   = self.Nz
            solver["zmin"] = self.zmin
            solver["zmax"] = self.zmax

        # Optional solver keys – only included when set
        for key, val in [
            ("poisson_method",          self.poisson_method),
            ("poisson_precond_vcycles", self.poisson_precond_vcycles),
            ("poisson_warm_start",      self.poisson_warm_start),
            ("poisson_smoother",        self.poisson_smoother),
            ("poisson_compile",         self.poisson_compile),
            ("poisson_bc_type",         self.poisson_bc_type),
            ("compile_adv_diff",        self.compile_adv_diff),
            ("compile_forces",          self.compile_forces),
            ("compile_sdf",             self.compile_sdf),
            ("dtype",                   self.dtype),
            ("zero_pressure_inside",    self.zero_pressure_inside),
            ("force_method",            self.force_method),
            ("time_integration",        self.time_integration),
            ("streaming_sdf_3d",        self.streaming_sdf_3d),
            ("streaming_forces_3d",     self.streaming_forces_3d),
            ("force_shared_union",      self.force_shared_union),
            ("mu_normals_union",        self.mu_normals_union),
            ("bdim_union",              self.bdim_union),
            ("force_narrow_batch",      self.force_narrow_batch),
            ("sdf_interp_method",       self.sdf_interp_method),
            ("vmax_abort",              self.vmax_abort),
        ]:
            if val is not None:
                solver[key] = val

        bc = {
            "BC_type_u"  : self.bc_type_u,
            "BC_values_u": self.bc_values_u,
            "BC_type_v"  : self.bc_type_v,
            "BC_values_v": self.bc_values_v,
        }
        if self.is_3d and self.bc_type_w is not None:
            bc["BC_type_w"]  = self.bc_type_w
            bc["BC_values_w"] = self.bc_values_w

        translation = [None, None, None] if self.is_3d else [None, None]
        body = {
            "type"           : "multi_animat",
            "sdf_folder"     : None,
            "plotting"       : False,
            "compute_interp" : self.compute_sdf,
            "plotting_meshes": False,
            "save_folder"    : os.path.join(
                self.data_folder, self.interp_data_subfolder
            ),
            "update_maps"    : {
                "rotation"   : "None",
                "translation": translation,
            },
            "suit"     : self.suit,
            "convexify": self.convexify,
            "scale"    : 1,
        }
        if self.n_samples is not None:
            body["n_samples"] = self.n_samples
        if self.force_scaling is not None:
            body["force_scaling"] = self.force_scaling
        if self.contour_mask is not None:
            body["contour_mask"] = self.contour_mask

        output = {
            "save_path"      : "",
            "existing_folder": output_folder,
            "save_frames"    : self.save_frames,
            "save_every"     : self.save_every,
            "vmin"           : self.vmin,
            "vmax"           : self.vmax,
            "plot_specs"     : None if self.plot_specs is None else list(self.plot_specs),
            "iso_3d_specs"   : None if self.iso_3d_specs is None else list(self.iso_3d_specs),
            "iso_3d_value"   : self.iso_3d_value,
            "save"           : self.save,
        }

        bdim_yaml = {
            "solver"             : solver,
            "boundary_conditions": bc,
            "body"               : body,
            "output"             : output,
        }
        if self.bdim_physics is not None:
            bdim_yaml["physics"] = self.bdim_physics

        return {
            "loader": "lilytorch.integration.extensions.FluidExtension",
            "config": {
                "handler_path": self.bdim_handler_path,
                "bdim_yaml"   : bdim_yaml,
            },
        }
