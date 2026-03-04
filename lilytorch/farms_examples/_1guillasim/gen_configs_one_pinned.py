
from cmath import inf
import os
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode
from farms_core.io.sdf import ModelSDF
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path, gen_new_folder, save_path
from lilytorch.integration.gen_pool_sdf import create_pool_sdf
import subprocess
import numpy as np

stack_folder      = os.path.join(save_path, "2guilla","fb_on")
stack_folder      = save_path
data_folder       = os.path.join(lilytorch_repo_root, 'farms_examples', '_1guillasim')
bdim_handler_path = "lilytorch.farms_examples._1guillasim.BDIMhandler.BDIMhandler"


nthreads = 16
use_gpu  = True
use_bdim = True
use_precice = False   # Set True to couple with OpenFOAM via preCICE (3D)
headless = False
fast     = False

precice_config_path = os.path.join(data_folder, "precice-config.xml")
openfoam_case_path  = os.path.join(data_folder, "openfoam_case")
stl_folder          = os.path.join(sdfs_path, "1guilla", "meshes")

use_drag = not use_bdim and not use_precice
constant_drags = [
            [-0.1, -5.0, -5.0],
            [-0.001, -0.001, -0.001]
]

animats_pars = [
    {
    "model_name"     : "1guilla",
    "sdf_name"       : "1guilla_link1_base.sdf",
    "control_type"   : "position",
    "gains"          : [50.0, .4, 0],
    "spawn_mode"     : SpawnMode.ROTZ,
    "pose"           : [-0.65, 0, 0.2, 0, 0, 0],
    "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller_fixed_neck.PositionController",
    "control_pars"   : {'freq': 1, 'twl': 0.571429*14, 'amp': 15.0},
    },
]


# u_inlet = 0.115971

u_inlet = 0.215971

# control_type = "position"
# controller_path      = "lilytorch.farms_examples._1guillasim.pd_controller.PositionController"
# gains = [20.0, 3, 0]
# control_pars = {'freq': 1, 'twl': 12, 'amp': 20.0}



Nx           = 1024
Ny           = 256
xmin         = -0.9
xmax         = 1.5
ymin         = -0.3
ymax         = 0.3

density = 800.0
# nu      = 500.0e-6
nu    = 1.0e-6


# timestep     = 0.01
# fluid_method = "implicit"
# save_every   = 50
# n_iterations = 1001

timestep     = 0.0005
fluid_method = "abdquickest"
save_frames  = False
save_every   = 200
n_iterations = 20001


save_frames  = True
save_uv      = False

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
        # animat_dict["control"]["sensors"]["contacts"] = []
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
    #    "sdf": os.path.join(sdfs_path, "arena_flat_v0", "sdf", "arena_flat.sdf"),
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
            # {
            # "loader": "farms_mujoco.simulation.extensions.CameraFollower",
            # "config": {
            #     "animat_id": 0,
            #     "distance": 1.0,
            #     "azimuth": 0,
            #     "elevation": -90,
            #     "angular_velocity": 0
            # }
            # },
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
            }            # {
            #     "loader": "farms_mujoco.simulation.extensions.TrailCoMViewer",
            #     "config": {
            #         "width": 0.1,
            #         "rgba" : [1.0, 0.0, 0.0, 1.0]
            #     }
            # }
        ]
    }

    if use_precice:

        simulation_dict["extensions"] += [
            {
            "loader": "lilytorch.farms_examples._1guillasim.precice_extension.PreCICEExtension",
            "config": {
                "precice_config": precice_config_path,
                "stl_folder"    : stl_folder,
                "rho_fluid"     : 1000.0,
            }
            }
        ]

    elif use_bdim:

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
                    "BC_type_u"  : ["D", "D", "N", "N"],
                    "BC_values_u": [u_inlet, u_inlet, 0],
                    "BC_type_v"  : ["N", "N", "D", "D"],
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
                    "save_path"      : "",
                    "existing_folder": output_folder,
                    "save_frames"    : save_frames,
                    "save_every"     : save_every,
                    "vmin"           : -40,
                    "vmax"           : 40,
                    "save_uv"        : save_uv
                }
                }
            }
            }
        ]

    pyobject2yaml(
        os.path.join(output_folder, 'simulation_config.yaml'),
        simulation_dict
    )

def gen_sh_config(output_folder):

    if use_precice:
        # For preCICE coupling, generate a script that:
        #   1. Launches OpenFOAM in the background
        #   2. Runs FARMS in the foreground
        sh_str = f"""#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OF_CASE="$SCRIPT_DIR/openfoam_case"

echo "=== Launching preCICE-coupled FARMS + OpenFOAM ==="

# Start OpenFOAM participant in background
echo "Starting OpenFOAM (pimpleFoam)..."
cd "$OF_CASE"
source /usr/lib/openfoam/openfoam2312/etc/bashrc
pimpleFoam &
OF_PID=$!

# Start FARMS participant
echo "Starting FARMS..."
cd "$SCRIPT_DIR"
farmsim --experiment_config experiment_config.yaml "$@"

# Wait for OpenFOAM to finish
wait $OF_PID
echo "=== Coupling complete ==="
"""
    else:
        sh_str = f"""#!/bin/bash
    farmsim --experiment_config experiment_config.yaml "$@"
    """
    with open(
        os.path.join(output_folder, 'run.sh'),
        'w'
    ) as f:
        f.write(sh_str)

def gen_precice_case(output_folder):
    """Copy the OpenFOAM case template and precice-config.xml into the output folder."""
    import shutil

    # Copy OpenFOAM case
    src_of_case = openfoam_case_path
    dst_of_case = os.path.join(output_folder, "openfoam_case")
    if os.path.exists(dst_of_case):
        shutil.rmtree(dst_of_case)
    shutil.copytree(src_of_case, dst_of_case)

    # Copy precice-config.xml
    src_precice_xml = precice_config_path
    dst_precice_xml = os.path.join(output_folder, "precice-config.xml")
    shutil.copy2(src_precice_xml, dst_precice_xml)

    # Also copy into the openfoam_case dir (OpenFOAM adapter looks for ../precice-config.xml)
    shutil.copy2(src_precice_xml, os.path.join(dst_of_case, "precice-config.xml"))

    print(f"  Copied OpenFOAM case to: {dst_of_case}")
    print(f"  Copied preCICE config to: {dst_precice_xml}")

    # Prepare the mesh
    print("  Preparing OpenFOAM mesh (blockMesh + snappyHexMesh)...")
    result = subprocess.run(
        ['bash', 'prepare_mesh.sh', stl_folder],
        cwd=dst_of_case,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: Mesh preparation failed:\n{result.stderr}")
    else:
        print("  OpenFOAM mesh prepared successfully.")


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

    if use_precice:
        gen_precice_case(output_folder)

    os.chdir(output_folder)
    subprocess.run(['bash', 'run.sh'])

    # import sys
    # from farms_sim.farmsim import main
    # sys.argv = ['farmsim', '--experiment_config', 'experiment_config.yaml']
    # main()

if __name__ == "__main__":

    single_run()
