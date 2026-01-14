
from cmath import inf
import os
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode
from farms_core.io.sdf import ModelSDF
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path, output_folder, todaystr, save_path
from lilytorch.integration.gen_pool_sdf import create_pool_sdf
import subprocess

data_folder = os.path.join(lilytorch_repo_root, 'farms_examples', '_1guillasim')

bdim_handler_path = "lilytorch.farms_examples._1guillasim.BDIMhandler.BDIMhandler"

os.makedirs(
    output_folder, exist_ok=True
)
print(
    "Saving configs to folder:", output_folder
)

use_bdim = True
headless = False

use_drag = not use_bdim

constant_drags = [
            [-0.1, -5.0, -5.0],
            [-0.001, -0.001, -0.001]
        ]

animats_pars = [
    {
    "model_name"   : "1guilla",
    "sdf_name"     : "1guilla.sdf",
    "control_type" : "torque",
    "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
    "muscle_config": {
        'load_controller': 'lilytorch.farms_examples._1guillasim.network.PAOscillatorController',
        'method'         : 'implicit',
        'muscle_pars'    : os.path.join(data_folder, 'muscle_params.csv'),
    },
    "gains": [0, 0, 0],
    "spawn_mode": SpawnMode.TRANSVERSE,
    "pose": [0, 0, 0, 0, 0, 3.141592653589793],
    },

    {
    "model_name"   : "1guilla",
    "sdf_name"     : "1guilla.sdf",
    "control_type" : "torque",
    "muscle_loader": "farms_ekeberg.src.ekeberg.EkebergMuscleController",
    "muscle_config": {
        'load_controller': 'lilytorch.farms_examples._1guillasim.network.PAOscillatorController',
        'method'         : 'implicit',
        'muscle_pars'    : os.path.join(data_folder, 'muscle_params.csv'),
    },
    "gains": [0, 0, 0],
    "spawn_mode": SpawnMode.TRANSVERSE,
    "pose": [1, 0, 0, 0, 0, 3.141592653589793],
    },

]

# control_type = "position"
# controller_path      = "lilytorch.farms_examples._1guillasim.pd_controller.PositionController"
# gains = [20.0, 3, 0]
# control_pars = {'freq': 1, 'twl': 12, 'amp': 20.0}



# # approximate the slime pool (check find_h_solution.py)
# "Nx": 512,
# "Ny": 256,
# "xmin": -0.85,
# "xmax": 1.71,
# "ymin": -0.64,
# "ymax": 0.64,



Nx = 2048
Ny = 512
xmin = -0.9
xmax = 5.1
ymin = -0.75
ymax = 0.75


density = 800.0
nu      = 500.0e-6
# nu    = 1.0e-6


# timestep     = 0.001
# fluid_method = "implicit"
# save_every   = 500

timestep     = 0.001
fluid_method = "abdquickest"
save_every   = 100

n_iterations = 20001


def gen_animat_config():

    for animat_i, animat_pars in enumerate(animats_pars):

        model_name    = animat_pars.pop("model_name")
        sdf_name      = animat_pars.pop("sdf_name")
        control_type  = animat_pars.pop("control_type")
        muscle_loader = animat_pars.pop("muscle_loader")
        muscle_config = animat_pars.pop("muscle_config")
        gains         = animat_pars.pop("gains")
        spawn_mode    = animat_pars.pop("spawn_mode")
        pose          = animat_pars.pop("pose")

        sdf_file   = os.path.join(sdfs_path, model_name, sdf_name)

        model_sdf   = ModelSDF.read(sdf_file)[0]
        link_names  = [link.name for link in model_sdf.links]
        joint_names = [joint.name for joint in model_sdf.joints]
        nlinks      = len(link_names)

        drag_coefficients = [
            constant_drags for _ in range(nlinks)
        ]

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
            'pose'    : pose,
            'velocity': [0, 0, 0, 0, 0, 0],
            'extras'  : {}
        }
        animat_dict["sdf"] = sdf_file



        # == Morphology ==
        animat_dict["morphology"]["links"] = [
            {
                'name'             : link_name,
                'collisions'       : True,
                'friction'         : [0.2, 0, 0],
                'extras'           : {},
                'fluid_interaction': use_drag,
                'density'          : density
            } for link_name in link_names
        ]
        if use_drag:
            for i, link in enumerate(animat_dict["morphology"]["links"]):
                link["drag_coefficients"] = drag_coefficients[i]

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
        animat_dict["control"]["sensors"]["contacts"] = [
            (link_name,'') for link_name in link_names
        ]
        animat_dict["control"]["sensors"]["xfrc"] = link_names
        animat_dict["control"]["sensors"]["muscles"] = []
        animat_dict["control"]["sensors"]["adhesions"] = []
        animat_dict["control"]["sensors"]["visuals"] = []
        animat_dict["control"]["motors"] = [
            {
                'joint_name'   : joint_name,
                'control_types': [control_type],
                'limits_torque': [-inf, inf],
                'gains'        : list(gains)
            } for joint_name in joint_names
        ]

        animat_dict["extensions"] = [
            {
                "loader": muscle_loader,
                "config": muscle_config
            }
        ]

        if use_drag:
            animat_dict["extensions"] += [
                {
                    "loader": "farms_mujoco.swimming.extension.SwimmingExtension",
                    "config": {
                        "water_properties" : None,
                    }
                }
            ]

        pyobject2yaml(
            os.path.join(output_folder, "animat_config_"+str(animat_i)+".yaml"),
            animat_dict
        )

def gen_arena_config():

    create_pool_sdf(xmin, xmax, ymin, ymax, wall_thickness=0.3, wall_height=0.3)

    arena_dict = {
       "sdf": os.path.join(sdfs_path, "pool", "sdf", "pool.sdf"),
       "spawn": {
            "loader"  : 0,
            "mode"    : SpawnMode.FREE,
            "pose"    : [0, 0, 0, 0, 0, 0],
            "velocity": [0, 0, 0, 0, 0, 0],
            "extras"  : {}
        },
        "water": {
            "sdf"      : os.path.join(sdfs_path, "arena_water_v0", "sdf", "arena_water.sdf"),
            "drag"     : use_drag,
            "buoyancy" : use_drag,
            "height"   : 0,
            "velocity" : [0, 0, 0],
            "viscosity": 1.0,
            "density"  : density,
            "maps"     : ["", ""],
        },
        "ground_height": 0.2,
    }
    pyobject2yaml(
        os.path.join(output_folder, 'arena_config.yaml'),
        arena_dict
    )

def gen_experiment_config():

    experiment_dict               = {}
    experiment_dict["simulation"] = "simulation_config.yaml"
    experiment_dict["arenas"]     = ["arena_config.yaml"]
    experiment_dict["animats"]    = ["animat_config_"+str(i)+".yaml" for i in range(len(animats_pars))]
    experiment_dict["loaders"]    = {}
    experiment_dict["loaders"]["simulation_options"] = "farms_core.simulation.options.SimulationOptions"
    experiment_dict["loaders"]["animats_options"]    = ["farms_core.model.options.AnimatOptions" for _ in range(len(animats_pars))]
    experiment_dict["loaders"]["arenas_options"]     = ["farms_core.model.options.ArenaOptions"]
    experiment_dict["loaders"]["experiment_data"]    = "farms_core.experiment.data.ExperimentData"
    experiment_dict["loaders"]["animats_data"]       = ["farms_core.model.data.AnimatData" for _ in range(len(animats_pars))]

    pyobject2yaml(
        os.path.join(output_folder, 'experiment_config.yaml'),
        experiment_dict
    )

def gen_simulation_config():

    simulation_dict = {
        "units": {
            "length": "meter",
            "mass"  : "kilogram",
            "time"  : "second"
        },
        "runtime": {
            "n_iterations" : n_iterations,
            "buffer_size"  : n_iterations,
            "play"         : True,
            "rtl"          : 1.0,
            "fast"         : False,
            "headless"     : headless,
            "show_progress": True
        }
        ,
        "physics": {
            "timestep"      : timestep,
            "gravity"       : [0, 0, -9.81],
            "num_sub_steps" : 1,
            "cb_sub_steps"  : 1,
            "n_solver_iters": 50
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
            "shadow_size"      : 1024,
            "visual_scale"     : 1.0,
            "extent"           : 100.0
        },
        "pybullet": {
            "opengl2"                : False,
            "lcp"                    : "dantzig",
            "cfm"                    : 1.0e-10,
            "erp"                    : 0,
            "contact_erp"            : 0,
            "friction_erp"           : 0,
            "residual_threshold"     : 1.0e-06,
            "max_num_cmd_per_1ms"    : 100000000,
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

    if use_bdim:

        simulation_dict["extensions"] += [
            {
            "loader": "lilytorch.integration.extensions.FluidExtension",
            "config": {
                "handler_path": bdim_handler_path,
                "bdim_yaml": {
                "solver": {
                    "use_gpu"                : True,
                    "nthreads"               : 16,
                    "Nx"                     : Nx,
                    "Ny"                     : Ny,
                    "xmin"                   : xmin,
                    "xmax"                   : xmax,
                    "ymin"                   : ymin,
                    "ymax"                   : ymax,
                    "convection_method"      : fluid_method,
                    "dt"                     : 0.0001,
                    "nt"                     : 800000,
                    "nu"                     : nu,
                    "rho"                    : 1.0e+3,
                    "poisson_tol"            : 1.0e-7,
                    "poisson_max_cycles"     : 5,
                    "poisson_max_mgcg_cycles": 3,
                    "jacobi_weight"          : 0.7,
                    "poisson_nsmoothing"     : 10,
                    "poisson_verbose"        : False,
                    "poisson_folder"         : os.path.join(data_folder, "data")
                },
                "boundary_conditions": {
                    "BC_type_u"  : ["N", "N", "N", "N"],
                    "BC_values_u": [0, 0, 0, 0],
                    "BC_type_v"  : ["N", "N", "N", "N"],
                    "BC_values_v": [0, 0, 0, 0]
                },
                "body": {
                    "type"           : "multi_animat",
                    "sdf_folder"     : None,
                    "plotting"       : False,
                    "compute_interp" : False,
                    "plotting_meshes": False,
                    "save_folder"    : os.path.join(data_folder, "interp_data"),
                    "n_samples"      : (2000, 2000),
                    "update_maps"    : {
                        "rotation"   : "None",
                        "translation": [None, None]
                    },
                    "suit"     : 0.0,
                    "convexify": True,
                    "scale"    : 1
                },
                "output": {
                    "save_path"      : save_path,
                    "existing_folder": todaystr,
                    "save_frames"    : True,
                    "save_every"     : save_every,
                    "vmin"           : -10,
                    "vmax"           : 10,
                    "save_uv"        : False
                }
                }
            }
            },
        ]

    simulation_dict["extensions"] += [
    {
        "loader": "lilytorch.integration.extensions.DataLogger",
        "config": {
            "log_path": "output/nn_data.hdf5",
        }
    }
    ]

    pyobject2yaml(
        os.path.join(output_folder, 'simulation_config.yaml'),
        simulation_dict
    )

def gen_sh_config():

    sh_str = f"""#!/bin/bash
    farmsim --experiment_config experiment_config.yaml "$@"
    """
    with open(
        os.path.join(output_folder, 'run.sh'),
        'w'
    ) as f:
        f.write(sh_str)

def gen_run():
    gen_animat_config()
    gen_arena_config()
    gen_simulation_config()
    gen_experiment_config()
    gen_sh_config()

    os.chdir(output_folder)
    subprocess.run(['bash', 'run.sh'])

if __name__ == "__main__":

    gen_run()