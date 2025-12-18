
from cmath import inf
import os
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode

sim_sir             = "lilytorch/farms_examples/1guillasim/1guilla_test_configs/"
handler_path         = "lilytorch.farms_examples.1guillasim.BDIMhandler.BDIMhandler"
sdf_folder           = "../../sdfs/1guilla/"
sdf_path             = '../../sdfs/1guilla/1guilla.sdf'

fluid_extension_path = "lilytorch.integration.extensions.FluidExtension"

control_type = "position"
controller_path      = "lilytorch.farms_examples.1guillasim.pd_controller.PositionController"
gains = [50.0, 5.0, 0]
control_pars = {'freq': 1.0, 'twl': 14, 'amp': 20.0}

# control_type    = "torque"
# controller_path = "lilytorch.farms_examples.1guillasim.torque_controller.WaveController"
# control_pars = {'freq': 1.0, 'twl': 0.8, 'amp': 0.4, 'bias': 0.0}

os.makedirs(
    sim_sir, exist_ok=True
)

nlinks = 9
njoints = 8

spawn_mode = SpawnMode.TRANSVERSE

density = 1000.0


link_names  = ["link" + str(i) for i in range(nlinks)]
joint_names = ["joint_body_" + str(i).zfill(2) for i in range(njoints)]


def gen_animat_config():

    animat_dict = {}

    animat_dict = {
        "spawn": {},
        "sdf"  : "",
        "morphology": {},
        "control": {
            "sensors": {},
            "motors" : []
        },
        "extensions": []
    }

    # == Spawn ==
    animat_dict["spawn"] = {
        'loader'  : 0,
        'mode'    : spawn_mode,
        'pose'    : [0, 0, 0, 0, 0, 3.141592653589793],
        'velocity': [0, 0, 0, 0, 0, 0],
        'extras'  : {}
    }
    animat_dict["sdf"] = sdf_path



    # == Morphology ==
    animat_dict["morphology"]["links"] = [
        {
            'name'             : link_name,
            'collisions'       : True,
            'friction'         : [0.2, 0, 0],
            'extras'           : {},
            'fluid_interaction': True,
            'density'          : density
        } for link_name in link_names
    ]
    animat_dict["morphology"]["joints"] = [
        {
            'name'     : joint_name,
            'initial'  : [0,0],
            'limits'   : [[-inf, inf], [-inf, inf]],
            'stiffness': 0,
            'springref': 0,
            'damping'  : 0,
            'extras'   : {}
        } for joint_name in joint_names
    ]
    animat_dict["morphology"]["self_collisions"] = []


    # == Control ==
    animat_dict["control"]["sensors"]["links"] = link_names
    animat_dict["control"]["sensors"]["joints"] = joint_names
    animat_dict["control"]["sensors"]["contacts"] = []
    animat_dict["control"]["sensors"]["xfrc"] = link_names
    animat_dict["control"]["sensors"]["muscles"] = []
    animat_dict["control"]["sensors"]["adhesions"] = []
    animat_dict["control"]["sensors"]["visuals"] = []
    animat_dict["control"]["motors"] = [
        {
            'joint_name'   : joint_name,
            'control_types': [control_type],
            'limits_torque': [-inf, inf],
            'gains'        : [50.0, 5.0, 0]
        } for joint_name in joint_names
    ]

    animat_dict["extensions"] = [
        {
            "loader": controller_path,
            "config":  control_pars
        }
    ]

    pyobject2yaml(
        sim_sir + 'animat_config.yaml',
        animat_dict
    )

def gen_arena_config():

    arena_dict = {
       "sdf": "../../sdfs/arena_flat_v0/sdf/arena_flat.sdf",
       "spawn": {
            "loader": 0,
            "mode": SpawnMode.FREE,
            "pose": [0, 0, 0, 0, 0, 0],
            "velocity": [0, 0, 0, 0, 0, 0],
            "extras": {}
        },
        "water": {
            "sdf": "../../sdfs/arena_water_v0/sdf/arena_water.sdf",
            "drag": False,
            "buoyancy": False,
            "height": 0,
            "velocity": [0, 0, 0],
            "viscosity": 1.0,
            "density": 1000.0,
            "maps": ["", ""],
        },
        "ground_height": -1.0,
    }
    pyobject2yaml(
        sim_sir + 'arena_config.yaml',
        arena_dict
    )

def gen_experiment_config():

    experiment_dict = {
        "simulation": "simulation_config.yaml",
        "animats": [
            "animat_config.yaml"
        ],
        "arenas": [
            "arena_config.yaml"
        ],
        "loaders": {
            "simulation_options": "farms_core.simulation.options.SimulationOptions",
            "animats_options": [
                "farms_core.model.options.AnimatOptions"
            ],
            "arenas_options": [
                "farms_core.model.options.ArenaOptions"
            ],
            "experiment_data": "farms_amphibious.data.data.ExperimentData",
            "animats_data": [
                "farms_core.model.data.AnimatData"
            ]
        },

    }
    pyobject2yaml(
        sim_sir + 'experiment_config.yaml',
        experiment_dict
    )

def gen_simulation_config():

    simulation_dict = {
        "units": {
            "length": "meter",
            "mass": "kilogram",
            "time": "second"
        },
        "runtime": {
            "n_iterations": 150001,
            "buffer_size": 150001,
            "play": True,
            "rtl": 1.0,
            "fast": False,
            "headless": True,
            "show_progress": True
        }
        ,
        "physics": {
            "timestep": 0.0001,
            "gravity": [0, 0, -9.81],
            "num_sub_steps": 1,
            "cb_sub_steps": 2,
            "n_solver_iters": 50
        },
        "mujoco": {
            "cone": "elliptic",
            "solver": "CG",
            "integrator": "implicitfast",
            "impratio": 10,
            "ccd_iterations": 1000,
            "ccd_tolerance": 1e-6,
            "noslip_iterations": 1000,
            "noslip_tolerance": 1e-6,
            "viewer": "MuJoCo",
            "texture_repeat": 1,
            "shadow_size": 1024,
            "visual_scale": 1.0,
            "extent": 100.0
        },
        "pybullet": {
            "opengl2": False,
            "lcp": "dantzig",
            "cfm": 1.0e-10,
            "erp": 0,
            "contact_erp": 0,
            "friction_erp": 0,
            "residual_threshold": 1.0e-06,
            "max_num_cmd_per_1ms": 100000000,
            "report_solver_analytics": 0
        },
        "extensions": [
            {
            "loader": fluid_extension_path,
            "config": {
                "handler_path": handler_path,
                "bdim_yaml": {
                "solver": {
                    "use_gpu": True,
                    "nthreads": 16,
                    "Nx": 2048,
                    "Ny": 512,
                    "xmin": -0.9,
                    "xmax": 5.1,
                    "ymin": -0.75,
                    "ymax": 0.75,
                    "convection_method": "abdquickest",
                    "dt": 0.0001,
                    "nt": 800000,
                    "nu": 1.0e-6,
                    "rho": 1.0e+3,
                    "poisson_tol": 1.0e-7,
                    "poisson_max_cycles": 5,
                    "poisson_max_mgcg_cycles": 3,
                    "jacobi_weight": 0.6,
                    "poisson_nsmoothing": 5,
                    "poisson_verbose": False,
                    "poisson_folder": "data/"
                },
                "boundary_conditions": {
                    "BC_type_u": ["N", "N", "N", "N"],
                    "BC_values_u": [0, 0, 0, 0],
                    "BC_type_v": ["N", "N", "N", "N"],
                    "BC_values_v": [0, 0, 0, 0]
                },
                "body": {
                    "type": "multi_animat",
                    "sdf_folder": sdf_folder,
                    "plotting": False,
                    "compute_interp": False,
                    "plotting_meshes": False,
                    "save_folder": "../",
                    "n_samples": (2000, 2000),
                    "update_maps": {
                    "rotation": "None",
                    "translation": [None, None]
                    },
                    "suit": 0.0,
                    "convexify": True,
                    "scale": 1
                },
                "output": {
                    "save_path": "/data/andreaferrario/ns_data/",
                    "save_frames": True,
                    "save_every": 500,
                    "vmin": -10,
                    "vmax": 10,
                    "save_uv": True
                }
                }
            }
            },
            {
            "loader": "farms_core.simulation.extensions.ExperimentLogger",
            "config": {
                "log_path": "output",
                "skip": 0
            }
            }
        ]}

    pyobject2yaml(
        sim_sir + 'simulation_config.yaml',
        simulation_dict
    )

def gen_sh_config():

    sh_str = f"""#!/bin/bash
    farmsim --experiment_config experiment_config.yaml "$@"
    """
    with open(
        sim_sir + 'run.sh',
        'w'
    ) as f:
        f.write(sh_str)


if __name__ == "__main__":
    gen_animat_config()
    gen_arena_config()
    gen_experiment_config()
    gen_simulation_config()
    gen_sh_config()

