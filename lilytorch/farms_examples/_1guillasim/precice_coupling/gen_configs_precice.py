"""
Config generator for preCICE-OpenFOAM coupled simulation.

Generates FARMS configs with the PreCICEExtension instead of the BDIM
FluidExtension. The OpenFOAM case is copied into the output folder and
meshed automatically.

Usage:
    python gen_configs_precice.py
"""

from cmath import inf
import os
import shutil
import subprocess
import numpy as np
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode
from farms_core.io.sdf import ModelSDF
from lilytorch.util.paths import lilytorch_repo_root, sdfs_path, gen_new_folder, save_path
from lilytorch.integration.gen_pool_sdf import create_pool_sdf

# ============================================================================
#  Paths
# ============================================================================
stack_folder        = save_path
data_folder         = os.path.join(lilytorch_repo_root, 'farms_examples', '_1guillasim')
coupling_folder     = os.path.join(data_folder, 'precice_coupling')
precice_config_path = os.path.join(coupling_folder, 'precice-config.xml')
openfoam_case_path  = os.path.join(coupling_folder, 'openfoam_case')
stl_folder          = os.path.join(sdfs_path, '1guilla', 'meshes')

# ============================================================================
#  Simulation parameters
# ============================================================================
headless     = False
fast         = False
density      = 800.0
timestep     = 0.0005
n_iterations = 20001
u_inlet      = 0.215971

# Domain (matching BDIM for comparison)
xmin, xmax = -0.9, 1.5
ymin, ymax = -0.3, 0.3

# ============================================================================
#  Animat parameters
# ============================================================================
animats_pars = [
    {
        "model_name"     : "1guilla",
        "sdf_name"       : "1guilla_link1_base.sdf",
        "control_type"   : "position",
        "gains"          : [50.0, 0.4, 0],
        "spawn_mode"     : SpawnMode.ROTZ,
        "pose"           : [-0.65, 0, 0.2, 0, 0, 0],
        "controller_path": "lilytorch.farms_examples._1guillasim.pd_controller_fixed_neck.PositionController",
        "control_pars"   : {'freq': 1, 'twl': 0.571429 * 14, 'amp': 15.0},
    },
]

# ============================================================================
#  Config generators
# ============================================================================

def gen_animat_config(output_folder):
    for animat_i, ap in enumerate(animats_pars):
        sdf_file    = os.path.join(sdfs_path, ap["model_name"], ap["sdf_name"])
        model_sdf   = ModelSDF.read(sdf_file)[0]
        link_names  = [link.name for link in model_sdf.links]
        joint_names = [j.name for j in model_sdf.joints if j.type != "fixed"]

        animat_dict = {
            "spawn": {
                'loader'  : 0,
                'mode'    : ap["spawn_mode"],
                'pose'    : ap["pose"],
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
                        'fluid_interaction': False,
                        'density'          : density,
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
                        'control_types': [ap["control_type"]],
                        'limits_torque': [-inf, inf],
                        'gains'        : list(ap["gains"]),
                    } for jn in joint_names
                ],
            },
            "extensions": [
                {
                    "loader": ap["controller_path"],
                    "config": ap["control_pars"],
                }
            ],
        }

        pyobject2yaml(
            os.path.join(output_folder, f"animat_config_{animat_i}.yaml"),
            animat_dict,
        )


def gen_arena_config(output_folder):
    create_pool_sdf(xmin, xmax, ymin, ymax,
                    wall_thickness=0.3, wall_height=0.3, plotting=False)

    arena_dict = {
        "sdf": os.path.join(sdfs_path, "pool", "sdf", "pool.sdf"),
        "spawn": {
            "loader"  : 0,
            "mode"    : SpawnMode.FREE,
            "pose"    : [0, 0, 0, 0, 0, 0],
            "velocity": [0, 0, 0, 0, 0, 0],
            "extras"  : {},
        },
        "water": {
            "sdf"      : os.path.join(sdfs_path, "arena_water_v0", "sdf", "arena_water.sdf"),
            "drag"     : False,
            "buoyancy" : False,
            "height"   : 0,
            "velocity" : [0, 0, 0],
            "viscosity": 1.0,
            "density"  : density,
            "maps"     : ["", ""],
        },
        "ground_height": 0.2,
    }
    pyobject2yaml(os.path.join(output_folder, 'arena_config.yaml'), arena_dict)


def gen_experiment_config(output_folder):
    n = len(animats_pars)
    experiment_dict = {
        "simulation"                  : "simulation_config.yaml",
        "arenas"                      : ["arena_config.yaml"],
        "animats"                     : [f"animat_config_{i}.yaml" for i in range(n)],
        "loaders": {
            "simulation_options": "farms_core.simulation.options.SimulationOptions",
            "animats_options"   : ["farms_core.model.options.AnimatOptions"] * n,
            "arenas_options"    : ["farms_core.model.options.ArenaOptions"],
            "experiment_data"   : "farms_core.experiment.data.ExperimentData",
            "animats_data"      : ["farms_core.model.data.AnimatData"] * n,
        },
    }
    pyobject2yaml(os.path.join(output_folder, 'experiment_config.yaml'), experiment_dict)


def gen_simulation_config(output_folder):
    simulation_dict = {
        "units": {"length": "meter", "mass": "kilogram", "time": "second"},
        "runtime": {
            "n_iterations" : n_iterations,
            "buffer_size"  : n_iterations,
            "play"         : True,
            "rtl"          : 1.0,
            "fast"         : fast,
            "headless"     : headless,
            "show_progress": True,
        },
        "physics": {
            "timestep"      : timestep,
            "gravity"       : [0, 0, -9.81],
            "num_sub_steps" : 1,
            "cb_sub_steps"  : 1,
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
            "shadow_size"      : 1024,
            "visual_scale"     : 1.0,
            "extent"           : 400.0,
        },
        "extensions": [
            {
                "loader": "farms_core.simulation.extensions.ExperimentLogger",
                "config": {"log_path": os.path.join(output_folder, "output"), "skip": 0},
            },
            {
                "loader": "farms_mujoco.simulation.extensions.MjcfSaver",
                "config": {"path": os.path.join(output_folder, "output", "simulation_mjcf.xml")},
            },
            {
                "loader": "lilytorch.integration.extensions.DataLogger",
                "config": {"log_path": os.path.join(output_folder, "output", "nn_data.hdf5")},
            },
            {
                "loader": "farms_mujoco.sensors.camera.CameraRecording",
                "config": {
                    "path"            : os.path.join(output_folder, "output", "video.mp4"),
                    "animat_id"       : None,
                    "fps"             : 30,
                    "speed"           : 1.0,
                    "azimuth"         : 0,
                    "elevation"       : -90,
                    "distance"        : 2,
                    "angular_velocity": 0,
                    "offset"          : [0, 0, 0.0],
                    "resolution"      : [1280, 720],
                },
            },
            # ---- preCICE coupling ----
            {
                "loader": "lilytorch.farms_examples._1guillasim.precice_coupling.precice_extension.PreCICEExtension",
                "config": {
                    "precice_config": os.path.join(output_folder, "precice-config.xml"),
                    "stl_folder"    : stl_folder,
                    "rho_fluid"     : 1000.0,
                },
            },
        ],
    }

    pyobject2yaml(os.path.join(output_folder, 'simulation_config.yaml'), simulation_dict)


def gen_sh_config(output_folder):
    sh_str = """#!/bin/bash
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
    with open(os.path.join(output_folder, 'run.sh'), 'w') as f:
        f.write(sh_str)


def gen_precice_case(output_folder):
    """Copy the OpenFOAM case template and precice-config.xml into the output folder."""

    # Copy OpenFOAM case
    dst_of_case = os.path.join(output_folder, "openfoam_case")
    if os.path.exists(dst_of_case):
        shutil.rmtree(dst_of_case)
    shutil.copytree(openfoam_case_path, dst_of_case)

    # Copy precice-config.xml
    dst_xml = os.path.join(output_folder, "precice-config.xml")
    shutil.copy2(precice_config_path, dst_xml)
    # OpenFOAM adapter looks for ../precice-config.xml
    shutil.copy2(precice_config_path, os.path.join(dst_of_case, "precice-config.xml"))

    print(f"  Copied OpenFOAM case  → {dst_of_case}")
    print(f"  Copied preCICE config → {dst_xml}")

    # Prepare the mesh
    print("  Preparing OpenFOAM mesh (blockMesh + snappyHexMesh)...")
    result = subprocess.run(
        ['bash', 'prepare_mesh.sh', stl_folder],
        cwd=dst_of_case, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: Mesh preparation failed:\n{result.stderr}")
    else:
        print("  OpenFOAM mesh prepared successfully.")


# ============================================================================
#  Main
# ============================================================================

def single_run():
    output_folder = gen_new_folder(stack_folder)
    os.makedirs(output_folder, exist_ok=True)
    print("Saving configs to folder:", output_folder)

    gen_animat_config(output_folder)
    gen_arena_config(output_folder)
    gen_simulation_config(output_folder)
    gen_experiment_config(output_folder)
    gen_sh_config(output_folder)
    gen_precice_case(output_folder)

    os.chdir(output_folder)
    subprocess.run(['bash', 'run.sh'])


if __name__ == "__main__":
    single_run()
