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
    * ``customize_joint_initials``  – override joint initials / damping / stiffness / limits
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
        # Either a single drag entry applied to every link:
        #   [linear_3vec, quadratic_3vec]  e.g. [[-0.1,-5,-5],[-0.001,-0.001,-0.001]]
        # or a per-link list of length == number-of-links, each element being
        # one such entry.  All entries must be identical when constant.
        self.constant_drags = [
            [-0.1, -5.0, -5.0],
            [-0.001, -0.001, -0.001],
        ]

        # ── Animats ───────────────────────────────────────────────────────
        self.animats_pars = []
        self.filter_fixed_joints = True  # skip joints with type=="fixed"

        # ── Joint defaults (applied to every non-fixed joint) ─────────────
        # Defaults baked into each joint dict in ``gen_animat_config``.
        # Set ``joint_damping > 0`` to stabilise *passive* DOFs that are
        # driven by fluid forces: the FARMS↔BDIM coupling is an explicit
        # staggered scheme, so light / neutrally-buoyant passive links are
        # prone to the added-mass instability — a little joint damping is
        # the cheapest mitigation.  Override per-joint in
        # ``customize_joint_initials`` for finer control.
        self.joint_damping   = 0.0
        self.joint_stiffness = 0.0
        self.joint_springref = 0.0

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
        self.save_drags  = False   # save drag/force records to drags.h5
        # FlowDiagnostics cadence: compute energy/enstrophy/max-div/CFL and warn
        # on blow-up (E_k>10x initial) / CFL>0.5 every N steps. 0 = disabled.
        # 100 catches Poisson under-convergence early at ~1% step overhead.
        self.diagnostics_every = 100
        self.vmin        = -10.0
        self.vmax        = 10.0
        self.plot_specs  = ["curl", "pressure"]
        self.iso_3d_specs = ["omega_mag", "vel_mag"]
        self.iso_3d_value = None

        # ── Arena ─────────────────────────────────────────────────────────
        self.generate_pool  = True   # generate pool/water SDFs; False → use flat arena
        self.grid_spacing   = None   # None or 0 → no floor grid; positive float → white grid line spacing (m)
        self.wall_thickness = None   # None → auto for 3-D, 0.3 for 2-D
        self.wall_height    = None   # None → 0.3 for 2-D (ignored for 3-D)
        self.wall_alpha     = None   # None → keep default (0.3); 0.0=transparent, 1.0=opaque
        self.water_alpha    = None   # None → keep default (0.18); 0.0=transparent, 1.0=opaque
        self.floor_color    = None   # None → black; [r, g, b] in [0,1] for background ground plane
        self.sky_color      = None   # None → black; [r, g, b] in [0,1] for skybox
        self.arena_pose     = [0, 0, 0, 0, 0, 0]
        self.water_drag     = None   # None → not use_bdim
        self.water_buoyancy = None   # None → not use_bdim
        self.water_height   = 0
        self.ground_height  = 0.0

        # ── Animat fluid interaction ──────────────────────────────────────
        self.animat_fluid_interaction = None  # None → not use_bdim

        # ── MuJoCo ────────────────────────────────────────────────────────
        self.num_sub_steps = 1
        self.cb_sub_steps  = 1
        self.visual_scale  = 1.0
        self.extent        = 400.0
        self.camera_dist   = 3.0
        self.shadow_size   = 0
        self.viewer_body_color = None  # None → keep mesh colours; [r,g,b] / [r,g,b,a] / "#rrggbb" → override

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
        self.compile_adv_diff        = False
        self.smagorinsky_cs          = 0.0
        self.carreau                 = None   # dict with keys: nu_0, nu_inf, lam, n
        self.sponge                  = None   # dict with keys: width, strength
        self.yield_damping           = None   # dict with keys: gamma_c, strength (auto-derived from carreau.tau_y if None)
        self.jacobi_weight           = 0.7
        # Floating-point precision used by FluidSolver and BDIMhandler.
        #   None         → solver default ("float32")
        #   "float32"    → single precision (recommended on consumer GPUs)
        #   "float64"    → double precision (recommended for sphere-sedimentation
        #                  validation and convergence studies)
        # Both the pure-PyTorch path and the C++/CUDA streaming kernels honour
        # this setting via ``AT_DISPATCH_FLOATING_TYPES`` in the kernel source.
        self.dtype                   = None
        self.zero_pressure_inside    = None
        self.force_method            = None
        # Distance (units of length) to offset the sample point from the
        # body surface along the outward normal when integrating
        # Lagrangian surface forces.  0 (default) samples exactly at the
        # triangle centroid / contour marker — but on BDIM-immersed
        # bodies that point sits in the smoothed band where ε is
        # computed from the blended velocity (≈ ½·fluid value) and
        # ``zero_pressure_inside`` zeros adjacent interior cells, so
        # forces are systematically under-estimated.  Setting this to
        # roughly ``eps`` (BDIM half-bandwidth ≈ 2·h) or larger moves
        # the sample into pure fluid and recovers a value comparable to
        # the Eulerian volume integral.  Recommended: sweep
        # ``{0, eps, 1.5·eps, 2·eps}`` and pick the smallest value that
        # matches the Eulerian on a steady benchmark.  Too large can
        # land the sample in a neighbouring link (concave multibody
        # geometry) or off-grid → instability — start small.
        self.lagrangian_sample_offset = None   # None → solver default (0.0)
        # Smooth body-velocity blend in the overlap band (width in cells).
        # Overlapping links (e.g. convexify=True) hard-switch the imposed
        # solid velocity at the inter-link seam under the running-min SDF
        # union, injecting grid-scale divergence → pressure spike → explicit
        # coupling blow-up.  Setting this to a few cells replaces the hard
        # switch with an SDF-weighted average  Σ w_i v_i / Σ w_i ,
        # w_i = σ(-φ_i/ε_w), continuous across the seam and exact for a
        # single (non-overlapping) body.  None/0 → legacy winner-take-all.
        self.body_velocity_blend_eps_cells = None
        self.time_integration        = None
        # Solver method: ``"python"`` | ``"kernel"``.
        # See :class:`FluidSolver` for what each method does. ``None``
        # → solver default (``"kernel"``).
        self.solver_method           = None
        # DEPRECATED.  Kept for backward compatibility — superseded by
        # ``solver_method``.  When set, they are mapped onto
        # ``solver_method`` inside :class:`FluidSolver`.
        self.use_kernels             = None
        # Body-SDF sampling method for the streaming kernels:
        #   "trilinear" (default) | "triquadratic"
        self.sdf_interp_method       = None
        # Fluid-explosion guard (forwarded into bdim_yaml.solver)
        self.vmax_abort              = None   # m/s; None = auto
        # BDIM interface-thickness multiplier: eps = eps_multiplier * h.
        # Default 2.0 in the solver.  For mesh bodies set to 1.5 or lower
        # (min ~1.0) to reduce the effective body size and drag inflation.
        self.eps_multiplier          = None   # None → solver default (2.0)
        # BDIM-σ (Lauber et al. 2022): per-body Poisson-coefficient shift
        # so thin bodies (r < eps) reach mu0_poisson = 0 inside and the
        # pressure BC is correctly enforced.  None → solver default (False).
        self.apply_bdim_sigma        = None   # None → solver default (False)
        # BDIM2 mu0-weighted Poisson coefficient (dt*mu0/rho_eff).  None →
        # solver default (True).  Set False to use the plain dt/rho_eff
        # coefficient, which keeps the projection non-degenerate — needed for
        # stable multibody swimmers (inter-link velocity seams).
        self.bdim_mu0_projection     = None   # None → solver default (True)
        # Maertens–Weymouth body-velocity-divergence RHS correction: subtract
        # (1-mu0)∇·u_b from the Poisson RHS so the mu0-weighted projection
        # stays consistent for OVERLAPPING links (e.g. convexify=True). No-op
        # for single rigid bodies. Lets you keep bdim_mu0_projection=True.
        self.bdim_body_div_correction = None  # None → solver default (False)
        # Poisson degenerate-cell freeze threshold (|diagonal| < jcap_tol →
        # frozen). Raise above the default 1e-12 to also freeze the near-
        # degenerate mu0-weighted band for overlapping bodies. None → 1e-12.
        self.poisson_jcap_tol = None
        # Delta-function order for force integration:
        #   1 (default) – first-order smoothed delta
        #   2           – Towers (2008) correction δ/|∇φ|, recommended for
        #                 mesh bodies where |∇SDF| ≠ 1 near joints/corners.
        self.force_delta_order       = None   # None → solver default (1)

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
        # Temporal under-relaxation of fluid->body force feedback.
        # F_applied^{n+1} = β · F_lag^{n+1} + (1-β) · F_applied^{n}
        # β=1.0 (None): no filtering. β<1.0: damp explicit-coupling oscillation
        # at the cost of slower transient response. DC (time-average) preserved.
        # Useful for stabilising Lagrangian forces in multibody swimmer coupling
        # without halving the physical force (which zero_pressure_inside hack did).
        self.force_relaxation = None
        # Strong (implicit) FSI coupling: set to a dict, e.g.
        #   {"scheme": "implicit", "accelerator": "iqn-ils", "reuse": 2,
        #    "tol": 1e-4, "max_iter": 30}
        # to replace the explicit force push with a quasi-Newton fixed point
        # (see lilytorch/integration/STRONG_COUPLING_FARMS_DESIGN.md).
        # None (default) -> explicit coupling.
        self.coupling = None
        self.compute_sdf    = False
        self.suit           = 0.0

        # ── Global MuJoCo contact tweaks (also forwarded into BDIM YAML) ───
        self.bdim_physics = None

        # Populated by gen_simulation_config so single_run can prepare any
        # launch-time environment needed by configured extensions.
        self._generated_simulation_extensions = []

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
    def _fluid_interaction(self):
        if self.animat_fluid_interaction is not None:
            return self.animat_fluid_interaction
        return not self.use_bdim

    @property
    def _water_drag(self):
        return self.water_drag if self.water_drag is not None else not self.use_bdim

    @property
    def _water_buoyancy(self):
        return self.water_buoyancy if self.water_buoyancy is not None else not self.use_bdim

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
        """Called after the joints list is built (mutates ``joints_list`` in place).

        Each entry is the full joint dict, so override to set any field
        per joint — not just the initial position:

        * ``initial`` – initial position/velocity (e.g. leg joints for
          salamanders).
        * ``damping`` / ``stiffness`` / ``springref`` – make a joint
          passive / compliant.  Adding damping to fluid-driven passive
          joints stabilises the explicit FARMS↔BDIM coupling (mitigates
          the added-mass instability for light / neutrally-buoyant links).
        * ``limits`` – per-joint position/velocity limits.

        For a single value applied to *every* joint, set ``joint_damping``
        / ``joint_stiffness`` / ``joint_springref`` in ``__init__`` instead.
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

            _drags = self.constant_drags
            if isinstance(_drags[0][0], (list, tuple)):
                # Per-link list: validate length then use directly
                if len(_drags) != nlinks:
                    raise ValueError(
                        f"constant_drags has {len(_drags)} entries but the "
                        f"model has {nlinks} links; provide either a single "
                        f"entry or exactly one entry per link."
                    )
                drag_coefficients = list(_drags)
            else:
                # Single constant entry: replicate for every link
                drag_coefficients = [_drags for _ in range(nlinks)]

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
                            'friction'         : [0., 0, 0],
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
                            'stiffness': self.joint_stiffness,
                            'springref': self.joint_springref,
                            'damping'  : self.joint_damping,
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
            if not self.use_bdim:
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
            if not self.use_bdim:
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
                    wall_alpha=self.wall_alpha,
                    grid_spacing=self.grid_spacing,
                    floor_color=self.floor_color,
                )
                water_sdf = create_water_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    zmin=self.zmin, zmax=self.zmax,
                    water_height=self.zmax,
                    water_alpha=self.water_alpha,
                )
                water_h = self.water_height if self.water_height != 0 else self.zmax
            else:
                wt = self.wall_thickness if self.wall_thickness is not None else 0.3
                wh = self.wall_height if self.wall_height is not None else 0.3
                create_pool_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    wall_thickness=wt, wall_height=wh, plotting=False,
                    wall_alpha=self.wall_alpha,
                    grid_spacing=self.grid_spacing,
                    floor_color=self.floor_color,
                )
                water_sdf = create_water_sdf(
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    water_height=self.water_height,
                    wall_height=wh,
                    water_alpha=self.water_alpha,
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

        if self.viewer_body_color is not None:
            simulation_dict["extensions"].append({
                "loader": "lilytorch.integration.body_color_override.BodyColorOverride",
                "config": {"color": self.viewer_body_color},
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
        self._generated_simulation_extensions = list(simulation_dict["extensions"])

        pyobject2yaml(
            os.path.join(output_folder, 'simulation_config.yaml'),
            simulation_dict,
        )

    def _extra_run_patch(self):
        """Return extra Python one-liner code injected into run.sh before main().

        Subclasses can override to patch farms_mujoco internals (e.g. night_sky)
        without modifying the farms_mujoco package.
        """
        return ""

    def gen_sh_config(self, output_folder, index=0):
        camera_dist = getattr(self, 'camera_dist', 3.0)

        # Write the offscreen-buffer fix to a helper file so we avoid
        # shell-quoting issues with complex multi-line Python inside -c "...".
        # It patches setup_mjcf_xml to widen MuJoCo's offscreen framebuffer
        # to match the largest CameraRecording resolution requested.
        with open(os.path.join(output_folder, '_offscreen_patch.py'), 'w') as f:
            f.write(
                "import farms_mujoco.simulation.mjcf as _m\n"
                "_orig_smx = _m.setup_mjcf_xml\n"
                "def _patched_smx(experiment_options, **kw):\n"
                "    r = _orig_smx(experiment_options=experiment_options, **kw)\n"
                "    g = r[0].visual.get_children('global')\n"
                "    ow, oh = g.offwidth, g.offheight\n"
                "    for e in experiment_options.simulation.extensions:\n"
                "        if 'CameraRecording' in e.loader:\n"
                "            res = e.config.get('resolution', [640, 480])\n"
                "            ow = max(ow, res[0])\n"
                "            oh = max(oh, res[1])\n"
                "    g.offwidth = ow\n"
                "    g.offheight = oh\n"
                "    return r\n"
                "_m.setup_mjcf_xml = _patched_smx\n"
            )

        # Build a one-liner that optionally monkey-patches add_cameras before
        # starting farms_sim, so that ] in the viewer uses the right distance.
        patch = (
            f"import farms_mujoco.simulation.mjcf as _m;"
            f"_o=_m.add_cameras;"
            f"_m.add_cameras=lambda link,dist={camera_dist!r},rot=None,simulation_options=None:_o(link,dist=dist,rot=rot,simulation_options=simulation_options);"
            f"exec(open('_offscreen_patch.py').read());"
        )
        patch += self._extra_run_patch()
        sh_str = (
            "#!/bin/bash\n"
            "set -e\n"
            f'"{sys.executable}" -c "{patch}from farms_sim._bootstrap import main; main()" '
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
        from lilytorch.integration.flow_viewer_2d_gpu import prepare_flow_viewer_2d_gpu_env
        from lilytorch.integration.flow_iso_gl_viewer import prepare_iso_gl_hook_env

        env = prepare_flow_viewer_2d_gpu_env(
            os.environ.copy(), self._generated_simulation_extensions,
        )
        env = prepare_iso_gl_hook_env(env, self._generated_simulation_extensions)
        subprocess.run(['bash', 'run.sh'], check=True, env=env)

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
            "diagnostics_every"      : self.diagnostics_every,
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
            ("poisson_bc_type",         self.poisson_bc_type),
            ("compile_adv_diff",        self.compile_adv_diff),
            ("solver_method",           self.solver_method),
            ("dtype",                   self.dtype),
            ("zero_pressure_inside",    self.zero_pressure_inside),
            ("force_method",            self.force_method),
            ("lagrangian_sample_offset", self.lagrangian_sample_offset),
            ("body_velocity_blend_eps_cells", self.body_velocity_blend_eps_cells),
            ("time_integration",        self.time_integration),
            ("use_kernels",             self.use_kernels),
            ("sdf_interp_method",       self.sdf_interp_method),
            ("vmax_abort",              self.vmax_abort),
            ("eps_multiplier",          self.eps_multiplier),
            ("apply_bdim_sigma",        self.apply_bdim_sigma),
            ("bdim_mu0_projection",     self.bdim_mu0_projection),
            ("bdim_body_div_correction", self.bdim_body_div_correction),
            ("poisson_jcap_tol",         self.poisson_jcap_tol),
            ("force_delta_order",       self.force_delta_order),
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
            "plotting"       : True,
            "compute_interp" : self.compute_sdf,
            "plotting_meshes": True,
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
        if self.force_relaxation is not None:
            body["force_relaxation"] = self.force_relaxation
        if self.coupling is not None:
            body["coupling"] = self.coupling
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
            "save_drags"     : self.save_drags,
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
