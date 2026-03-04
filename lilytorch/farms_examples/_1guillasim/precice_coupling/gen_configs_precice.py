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
headless     = True
fast         = True
density      = 800.0
timestep     = 0.001
n_iterations = 2001
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
    # Embed source paths so run.sh can find plot scripts at runtime
    _coupling_src = coupling_folder  # absolute path known at generation time

    sh_str = f"""#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OF_CASE="$SCRIPT_DIR/openfoam_case"
PLOT_SRC="{_coupling_src}"
PLOTTED_TIMES_FILE="$SCRIPT_DIR/.plotted_times"
touch "$PLOTTED_TIMES_FILE"

echo "=== Launching preCICE-coupled FARMS + OpenFOAM ==="

# ---------------------------------------------------------------
# Live-plot monitor: runs in background, watches for new timesteps
# in processor0, reconstructs them, and generates 2D + 3D plots.
# ---------------------------------------------------------------
live_monitor() {{
    echo "[monitor] Started — polling every 30s for new timesteps"
    while true; do
        sleep 30

        # Discover timestep dirs that processor0 has written (exclude 0 and constant)
        NEW_TIMES=""
        if [ -d "$OF_CASE/processor0" ]; then
            for TD in "$OF_CASE/processor0"/*/; do
                TNAME=$(basename "$TD")
                # Skip non-numeric, 0, constant
                [[ "$TNAME" =~ ^[0-9] ]] || continue
                [[ "$TNAME" == "0" ]] && continue
                # Skip already done
                grep -qxF "$TNAME" "$PLOTTED_TIMES_FILE" && continue
                # Check it has U (i.e., complete write)
                [ -f "$TD/U" ] || continue
                NEW_TIMES="$NEW_TIMES $TNAME"
            done
        fi

        [ -z "$NEW_TIMES" ] && continue

        echo "[monitor] New timesteps detected:$NEW_TIMES"

        # Reconstruct only the new times
        bash -c "
            source /usr/lib/openfoam/openfoam2312/etc/bashrc
            for T in $NEW_TIMES; do
                reconstructPar -case \\"$OF_CASE\\" -time \\$T > /dev/null 2>&1
            done
        "

        # Generate plots for just these times
        TIMES_ARGS=""
        for T in $NEW_TIMES; do
            TIMES_ARGS="$TIMES_ARGS $T"
        done

        if [ -f "$PLOT_SRC/plot_3d.py" ]; then
            echo "[monitor] Vorticity top-view for:$TIMES_ARGS"
            python3 "$PLOT_SRC/plot_3d.py" "$SCRIPT_DIR" --times $TIMES_ARGS 2>&1 | tail -5
        fi

        # Clean up reconstructed data to save disk space (plots already saved)
        for T in $NEW_TIMES; do
            rm -rf "$OF_CASE/$T"
        done

        # Mark these times as plotted
        for T in $NEW_TIMES; do
            echo "$T" >> "$PLOTTED_TIMES_FILE"
        done
    done
}}

# Start monitor in background
live_monitor &
MONITOR_PID=$!
echo "Live-plot monitor running (PID $MONITOR_PID)"

# Start OpenFOAM participant in a SEPARATE subshell so its environment
# does not pollute the FARMS/Python process.
# Both participants run from SCRIPT_DIR so preCICE sockets are in the same dir.
echo "Starting OpenFOAM (pimpleFoam) on 24 MPI ranks..."
bash -c "
    source /usr/lib/openfoam/openfoam2312/etc/bashrc
    export FOAM_SIGFPE=false
    cd \\"$SCRIPT_DIR\\"
    mpirun -np 24 --oversubscribe pimpleFoam -parallel -case \\"$OF_CASE\\" > \\"$OF_CASE/log.pimpleFoam\\" 2>&1
" &
OF_PID=$!

# Give OpenFOAM a moment to initialise preCICE and start listening
sleep 3

# Verify OpenFOAM is running
if ! kill -0 $OF_PID 2>/dev/null; then
    echo "ERROR: pimpleFoam died. Check $OF_CASE/log.pimpleFoam"
    cat "$OF_CASE/log.pimpleFoam" 2>/dev/null | tail -20
    kill $MONITOR_PID 2>/dev/null
    exit 1
fi
echo "OpenFOAM running (PID $OF_PID)"

# Start FARMS participant (inherits the original Python-friendly environment)
echo "Starting FARMS..."
cd "$SCRIPT_DIR"
MUJOCO_GL=egl farmsim --experiment_config experiment_config.yaml "$@"
FARMS_EXIT=$?

# Wait for OpenFOAM to finish
wait $OF_PID || true
echo "=== Coupling complete (FARMS exit code: $FARMS_EXIT) ==="

# Stop the live monitor
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
echo "Live monitor stopped."

# ---- Final pass: plot any remaining timesteps the monitor missed ----
echo "=== Final post-processing pass ==="
bash -c "
    source /usr/lib/openfoam/openfoam2312/etc/bashrc
    reconstructPar -case \\"$OF_CASE\\" -newTimes 2>&1 | tail -5
"

if [ -f "$PLOT_SRC/plot_3d.py" ]; then
    echo "  -> Final vorticity renders (if any remaining)..."
    python3 "$PLOT_SRC/plot_3d.py" "$SCRIPT_DIR" 2>&1 | tail -10
fi

# Clean up any leftover reconstructed timestep dirs
for D in "$OF_CASE"/*/; do
    DNAME=$(basename "$D")
    [[ "$DNAME" =~ ^[0-9] ]] || continue
    [[ "$DNAME" == "0" ]] && continue
    rm -rf "$D"
done

echo "=== All done. Figures in $SCRIPT_DIR/figures/ ==="
"""
    with open(os.path.join(output_folder, 'run.sh'), 'w') as f:
        f.write(sh_str)


def gen_precice_case(output_folder):
    """Copy the OpenFOAM case template and precice-config.xml into the output folder.

    The mesh (blockMesh + snappyHexMesh) is prepared once in the *template*
    openfoam_case/ directory and cached there.  Subsequent runs detect the
    existing polyMesh and skip the expensive meshing step entirely.
    """

    # --- Prepare mesh in template (cached) --------------------------------
    template_polyMesh = os.path.join(openfoam_case_path, "constant", "polyMesh")
    if not os.path.isdir(template_polyMesh) or not os.path.isfile(
        os.path.join(template_polyMesh, "owner")
    ):
        print("  Preparing OpenFOAM mesh in template (one-time)...")
        result = subprocess.run(
            ['bash', 'prepare_mesh.sh', stl_folder],
            cwd=openfoam_case_path, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: Mesh preparation failed:\n{result.stderr}")
        else:
            print("  Template mesh prepared and cached.")
    else:
        print("  Using cached mesh from template (skipping blockMesh/snappyHexMesh).")

    # --- Copy the (now-meshed) template into the output folder -------------
    dst_of_case = os.path.join(output_folder, "openfoam_case")
    if os.path.exists(dst_of_case):
        shutil.rmtree(dst_of_case)
    shutil.copytree(openfoam_case_path, dst_of_case)

    # Copy precice-config.xml
    dst_xml = os.path.join(output_folder, "precice-config.xml")
    shutil.copy2(precice_config_path, dst_xml)
    shutil.copy2(precice_config_path, os.path.join(dst_of_case, "precice-config.xml"))

    # Ensure 0/ has the boundary conditions (not just snappy leftovers)
    dst_0 = os.path.join(dst_of_case, "0")
    dst_0orig = os.path.join(dst_of_case, "0.orig")
    if os.path.isdir(dst_0orig):
        for f in os.listdir(dst_0orig):
            shutil.copy2(os.path.join(dst_0orig, f), os.path.join(dst_0, f))

    # --- Decompose mesh for parallel run --------------------------------
    print("  Decomposing mesh for parallel run...")
    result = subprocess.run(
        ['bash', '-c',
         'source /usr/lib/openfoam/openfoam2312/etc/bashrc && '
         f'decomposePar -case {dst_of_case} -force'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: decomposePar failed:\n{result.stderr[-500:]}")
    else:
        print("  Mesh decomposed into 24 subdomains.")

    print(f"  Copied OpenFOAM case  → {dst_of_case}")
    print(f"  Copied preCICE config → {dst_xml}")


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
    print(f"\n=== To run manually: cd {output_folder} && bash run.sh ===\n")
    subprocess.run(['bash', 'run.sh'])


if __name__ == "__main__":
    single_run()
