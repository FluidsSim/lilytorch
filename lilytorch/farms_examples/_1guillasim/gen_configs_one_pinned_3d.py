
from cmath import inf
import os
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode
from farms_core.io.sdf import ModelSDF
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path, gen_new_folder, save_path
from lilytorch.integration.gen_pool_sdf import create_pool_sdf
import subprocess
import numpy as np

stack_folder      = save_path
data_folder       = os.path.join(lilytorch_repo_root, 'farms_examples', '_1guillasim')
bdim_handler_path = "lilytorch.integration.BDIMhandler.BDIMhandler"


nthreads = 16
use_gpu  = True
use_bdim = True
headless = False  # must be False for FlowViewer (MuJoCo GUI)
fast     = False

use_drag = not use_bdim
constant_drags = [
            [-0.1, -5.0, -5.0],
            [-0.001, -0.001, -0.001]
]

animats_pars = [
    {
    "model_name"     : "1guilla",
    "sdf_name"       : "1guilla.sdf",
    "control_type"   : "position",
    "gains"          : [100.0, 4.0, 0],
    "spawn_mode"     : SpawnMode.ROTZ,
    "pose"           : [-0.65, 0, 0., 0, 0, 0],
    "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller.PositionController",
    "control_pars"   : {'freq': 1, 'twl': 12, 'amp': 30.0},
    },
]

u_inlet = 0.215971

# ---- 3-D grid --------------------------------------------------------
Nx           = 512
Ny           = 128
Nz           = 128
xmin         = -0.9
xmax         = 1.5
ymin         = -0.3
ymax         = 0.3
zmin         = -0.3
zmax         = 0.3

density = 800.0
nu    = 1.0e-6

timestep     = 0.0005
convection_method = "quick"
save_frames  = True
save_every   = 200
n_iterations = 201
save_uv      = False

# note - to compare the experimental values that were normalized
vmin = -40*u_inlet/0.85
vmax = 40*u_inlet/0.85

def gen_animat_config(output_folder):

    for animat_i, animat_pars in enumerate(animats_pars):

        model_name    = animat_pars["model_name"]
        sdf_name      = animat_pars["sdf_name"]
        control_type    = animat_pars["control_type"]
        controller_path = animat_pars["controller_path"]
        control_pars    = animat_pars["control_pars"]
        gains         = animat_pars["gains"]
        spawn_mode    = animat_pars["spawn_mode"]
        pose          = animat_pars["pose"]

        sdf_file = os.path.join(sdfs_path, model_name, sdf_name)

        model_sdf   = ModelSDF.read(sdf_file)[0]
        link_names  = [link.name for link in model_sdf.links]
        joint_names = [joint.name for joint in model_sdf.joints if joint.type != "fixed"]
        nlinks      = len(link_names)

        n_joints = len(joint_names)

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
                "loader": controller_path,
                "config": control_pars
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

def gen_arena_config(output_folder):

    create_pool_sdf(xmin, xmax, ymin, ymax, wall_thickness=0.3, wall_height=0.3, plotting=False)

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

def gen_experiment_config(output_folder):

    experiment_dict                                  = {}
    experiment_dict["simulation"]                    = "simulation_config.yaml"
    experiment_dict["arenas"]                        = ["arena_config.yaml"]
    experiment_dict["animats"]                       = ["animat_config_"+str(i)+".yaml" for i in range(len(animats_pars))]
    experiment_dict["loaders"]                       = {}
    experiment_dict["loaders"]["simulation_options"] = "farms_core.simulation.options.SimulationOptions"
    experiment_dict["loaders"]["animats_options"]    = ["farms_core.model.options.AnimatOptions" for _ in range(len(animats_pars))]
    experiment_dict["loaders"]["arenas_options"]     = ["farms_core.model.options.ArenaOptions"]
    experiment_dict["loaders"]["experiment_data"]    = "farms_core.experiment.data.ExperimentData"
    experiment_dict["loaders"]["animats_data"]       = ["farms_core.model.data.AnimatData" for _ in range(len(animats_pars))]

    pyobject2yaml(
        os.path.join(output_folder, 'experiment_config.yaml'),
        experiment_dict
    )

def gen_simulation_config(output_folder):

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
            "fast"         : fast,
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
            "extent"           : 400.0
        },
        "extensions": [
            {
                "loader": "farms_core.simulation.extensions.ExperimentLogger",
                "config": {
                    "log_path": os.path.join(output_folder, "output"),
                    "skip": 0
                }
            },
            {
                "loader": "farms_mujoco.simulation.extensions.MjcfSaver",
                "config": {
                    "path": os.path.join(output_folder, "output", "simulation_mjcf.xml")
                }
            },
            {
                "loader": "lilytorch.integration.extensions.DataLogger",
                "config": {
                    "log_path": os.path.join(output_folder, "output", "nn_data.hdf5"),
                }
            },
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
                    "use_gpu"                : use_gpu,
                    "nthreads"               : nthreads,
                    "Nx"                     : Nx,
                    "Ny"                     : Ny,
                    "Nz"                     : Nz,
                    "xmin"                   : xmin,
                    "xmax"                   : xmax,
                    "ymin"                   : ymin,
                    "ymax"                   : ymax,
                    "zmin"                   : zmin,
                    "zmax"                   : zmax,
                    "convection_method"      : convection_method,
                    "dt"                     : 0.0001,
                    "nt"                     : 800000,
                    "nu"                     : nu,
                    "rho"                    : 1.0e+3,
                    "poisson_tol"            : 1.0e-4,
                    "poisson_max_cycles"     : 30,
                    "poisson_max_mgcg_cycles": 3,
                    "jacobi_weight"          : 0.7,
                    "poisson_nsmoothing"     : 5,
                    "poisson_verbose"        : False,
                    "poisson_folder"         : os.path.join(data_folder, "data"),
                    "rho_body"               : 800.0,
                },
                "boundary_conditions": {
                    "BC_type_u"  : ["D", "D", "N", "N", "N", "N"],
                    "BC_values_u": [u_inlet, u_inlet, 0, 0, 0, 0],
                    "BC_type_v"  : ["N", "N", "D", "D", "N", "N"],
                    "BC_values_v": [0, 0, 0, 0, 0, 0],
                    "BC_type_w"  : ["N", "N", "N", "N", "D", "D"],
                    "BC_values_w": [0, 0, 0, 0, 0, 0],
                },
                "body": {
                    "type"           : "multi_animat",
                    "sdf_folder"     : None,
                    "plotting"       : False,
                    "compute_interp" : False,
                    "plotting_meshes": False,
                    "save_folder"    : os.path.join(data_folder, "interp_data_3d"),
                    "update_maps"    : {
                        "rotation"   : "None",
                        "translation": [None, None, None]
                    },
                    "suit"          : 0.0,
                    "convexify"     : True,
                    "scale"         : 1,
                    "force_scaling" : 1.0,
                },
                "output": {
                    "save_path"      : "",
                    "existing_folder": output_folder,
                    "save_frames"    : save_frames,
                    "save_every"     : save_every,
                    "vmin"           : vmin,
                    "vmax"           : vmax,
                    "save_uv"        : save_uv
                }
                }
            }
            }
        ]

        # ── Flow visualisation in MuJoCo viewer (requires headless=False) ──
        simulation_dict["extensions"] += [
            {
                "loader": "lilytorch.integration.flow_viewer.FlowViewer",
                "config": {
                    "field"        : "omega_z",
                    "max_spheres"  : 4000,
                    "iso_fraction" : 0.15,
                    "smooth_sigma" : 2.5,
                    "crop_boundary": 3,
                    "sphere_size"  : 0.02,
                    "update_every" : None,
                }
            }
        ]

    # ── CameraRecording MUST come last so it captures frames
    # ── after FluidExtension + FlowViewer have updated the scene ──
    simulation_dict["extensions"] += [
        {
            "loader": "farms_mujoco.sensors.camera.CameraRecording",
            "config": {
                "path": os.path.join(output_folder, "output", "video.mp4"),
                "animat_id": None,
                "fps": 30,
                "speed": 1.0,
                "azimuth": 0,
                "elevation": -90,
                "distance": 2,
                "angular_velocity": 0,
                "offset": [0, 0, 0.0],
                "resolution": [1280, 720]
            }
        }
    ]

    pyobject2yaml(
        os.path.join(output_folder, 'simulation_config.yaml'),
        simulation_dict
    )

def gen_sh_config(output_folder):

    sh_str = f"""#!/bin/bash
    farmsim --experiment_config experiment_config.yaml "$@"
    """
    with open(
        os.path.join(output_folder, 'run.sh'),
        'w'
    ) as f:
        f.write(sh_str)


def single_run():

    output_folder = gen_new_folder(stack_folder)

    os.makedirs(
        output_folder, exist_ok=True
    )
    print(
        "Saving configs to folder:", output_folder
    )

    gen_animat_config(output_folder)
    gen_arena_config(output_folder)
    gen_simulation_config(output_folder)
    gen_experiment_config(output_folder)
    gen_sh_config(output_folder)
    os.chdir(output_folder)
    subprocess.run(['bash', 'run.sh'])


if __name__ == "__main__":

    single_run()
