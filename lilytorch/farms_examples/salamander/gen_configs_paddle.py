
from cmath import inf
import os
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode
from farms_core.io.sdf import ModelSDF
from pathlib import Path

current_file_path = Path(__file__).parent.absolute()
lilytorch_root =  str(current_file_path.parent.parent)

handler_path = "lilytorch.farms_examples.salamander.BDIMhandler.BDIMhandler"
fluid_extension_path = "lilytorch.integration.extensions.FluidExtension"

sim_sir      = lilytorch_root+"/farms_examples/salamander/paddle/"

sdf_folder   = lilytorch_root+"/farms_examples/sdfs/salamander_v5/"
sdf_path     = sdf_folder+"sdf/salamander.sdf"


control_type    = "position"
controller_path = "lilytorch.farms_examples.salamander.pd_controller_paddle.PositionController"
gains        = [0.001, .00002, 0]
control_pars = {'freq': 2.0, 'twl': 10, 'amp': 200, 'limb_pose1':-0.35*3.141592653589793, 'limb_pose2':-0.2*3.141592653589793}

use_fluid    = False
headless     = False

sdf = ModelSDF.read(sdf_path)[0] # this is the sdf content


link_names  = [link.name for link in sdf.links]
joints      = [joint for joint in sdf.joints if joint.type != "fixed"]
joint_names = [joint.name for joint in joints]

initial_joint_pos = []
for joint in joints:
    initial_joint_pos.append([0, 0])

timestep = 0.0001
spawn_mode = SpawnMode.TRANSVERSE
density = 1000.0

# link_names  = ["link_body_" + str(i) for i in range(9)]
# link_names += ["link_leg_0_L_0", "link_leg_0_L_1","link_leg_0_R_0"]

# # for i in range(2): # Front - back
# #     for j in range(4): # joints per side
# #         link_names += [f"link_leg_{i}_L_{j}", f"link_leg_{i}_R_{j}"]
# # link_names += ["foot_0_0", "foot_0_1", "foot_1_0", "foot_1_1"]

# joint_names = ["joint_body_" + str(i) for i in range(8)]
# joint_names += ["joint_leg_0_L_0", "joint_leg_0_R_0"] #, "joint_leg_1_L_0", "joint_leg_1_R_0"]



os.makedirs(
    sim_sir, exist_ok=True
)


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
            'initial'  : initial_joint_pos[i],
            'limits'   : [[-inf, inf], [-inf, inf]],
            'stiffness': 0,
            'springref': 0,
            'damping'  : 0,
            'extras'   : {}
        } for i, joint_name in enumerate(joint_names)
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
            'gains'        : gains
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
            "headless": headless,
            "show_progress": True
        }
        ,
        "physics": {
            "timestep": timestep,
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
            "visual_scale": 10.0,
            "extent": 10.0
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
            "loader": "farms_core.simulation.extensions.ExperimentLogger",
            "config": {
                "log_path": "output",
                "skip": 0
            }
            },
            {
                "loader": "farms_mujoco.simulation.extensions.MjcfSaver",
                "config": {
                    "path": "output/simulation_mjcf.xml"
                }
            }
        ]
    }

    if use_fluid:
        simulation_dict["extensions"] += [
            {
            "loader": fluid_extension_path,
            "config": {
                "handler_path": handler_path,
                "bdim_yaml": {
                "solver": {
                    "use_gpu": True,
                    "nthreads": 16,

                    # ## small tank
                    # N                 : 256
                    # xmin              : -0.02
                    # xmax              : 0.006
                    # ymin              : -0.013
                    # ymax              : 0.013

                    # intermediate tank
                    "Nx": 1024,
                    "Ny": 512,
                    "xmin": -0.11,
                    "xmax": 0.09,
                    "ymin": -0.05,
                    "ymax": 0.05,

                    # ## large tank
                    # N                 : 2048
                    # xmin              : -0.02
                    # xmax              : 0.188
                    # ymin              : -0.104
                    # ymax              : 0.104

                    "convection_method": "implicit",
                    "dt": 0.001,
                    "nt": 800000,
                    "nu": 1.0e-6,
                    "rho": 1.0e+3,
                    "poisson_tol": 1.0e-7,
                    "poisson_max_cycles": 5,
                    "poisson_max_mgcg_cycles": 3,
                    "jacobi_weight": 0.7,
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
                    "convexify": False,
                    "scale": 1
                },
                "output": {
                    "save_path": "/data/andreaferrario/ns_data/",
                    "save_frames": True,
                    "save_every": 20,
                    "vmin": -50,
                    "vmax": 50,
                    "save_uv": False
                }
                }
            }
            },
        ]

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

