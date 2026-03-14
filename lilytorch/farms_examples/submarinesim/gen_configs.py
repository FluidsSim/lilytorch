"""Generate simulation configuration files for the submarine model.

Submarine description
---------------------
The submarine is modelled as 5 cylindrical body segments connected by 4
revolute joints whose axes are aligned with the world z-axis (yaw).  A
sinusoidal travelling wave of lateral displacement propagates from head to
tail, generating forward thrust via reactive hydrodynamic forces
(carangiform / BCF locomotion) – the same mechanism used by many elongated
fish.

Reynolds number
---------------
    Re = v * L / nu
       = 0.01 [m/s] * 0.1 [m] / 1e-6 [m^2/s]
       = 1000

At Re ~ 1000 the flow is laminar / transitional, which is well within the
tractable range for viscous fluid solvers (LBM, BDIM, Navier-Stokes FVM).
Keeping the body length short (0.1 m) and the swimming speed low (~0.01 m/s)
ensures that Re stays comfortably below the turbulent transition threshold.

Physical parameters (per segment, cylinder r = 0.01 m, L = 0.02 m)
--------------------------------------------------------------------
    V   = pi * r^2 * L              = 6.283e-6  m^3
    rho = 1000 kg/m^3               (neutral buoyancy)
    m   = rho * V                   = 6.283e-3  kg
    Ixx = m * r^2 / 2               = 3.14e-7   kg*m^2  (roll)
    Iyy = Izz = m*(3r^2+L^2)/12    = 3.67e-7   kg*m^2  (pitch / yaw)

Usage
-----
    python gen_configs.py
"""

from math import inf
import os
from pathlib import Path
from farms_core.io.yaml import pyobject2yaml
from farms_core.model.options import SpawnMode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
current_file_path = Path(__file__).parent.absolute()
lilytorch_root = str(current_file_path.parent.parent)

sim_dir = str(current_file_path / "sim") + "/"
# Absolute path resolved at config-generation time; FARMS resolves this at runtime.
sdf_path = lilytorch_root + "/farms_examples/sdfs/submarine_v1/sdf/submarine.sdf"

os.makedirs(sim_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Controller / simulation settings
# ---------------------------------------------------------------------------
control_type = "position"
controller_path = "lilytorch.farms_examples.submarinesim.pd_controller.PositionController"
gains = [0.001, 0.00002, 0]

# Swimming gait parameters
# freq  : tail-beat frequency [Hz]
# twl   : travelling wavelength [body segments]
# amp   : peak joint amplitude [degrees] – kept moderate for Re ~ 1000
control_pars = {"freq": 1.0, "twl": 8, "amp": 20}

headless = True
save_frames = False

# ---------------------------------------------------------------------------
# Model topology (matches submarine.sdf)
# ---------------------------------------------------------------------------
nlinks = 5
njoints = 4
link_names = ["link_body_" + str(i) for i in range(nlinks)]
joint_names = ["joint_body_" + str(i) for i in range(njoints)]

spawn_mode = SpawnMode.TRANSVERSE
density = 1000.0  # kg/m^3 – neutral buoyancy

# Simulation timestep – chosen to resolve the tail-beat period well
# (at 1 Hz, 100 steps/cycle; at Re 1000 the viscous time scale nu/L^2 ~ 0.1 s)
timestep = 0.01


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def gen_animat_config():
    """Animat (submarine body) configuration."""
    animat_dict = {
        "spawn": {
            "loader": 0,
            "mode": spawn_mode,
            # Spawn slightly above the arena floor; facing +x
            "pose": [0, 0, 0.05, 0, 0, 0],
            "velocity": [0, 0, 0, 0, 0, 0],
            "extras": {},
        },
        "sdf": sdf_path,
        "morphology": {
            "links": [
                {
                    "name": link_name,
                    "collisions": True,
                    "friction": [0.2, 0, 0],
                    "extras": {},
                    "fluid_interaction": True,
                    "density": density,
                }
                for link_name in link_names
            ],
            "joints": [
                {
                    "name": joint_name,
                    "initial": [0, 0],
                    "limits": [[-inf, inf], [-inf, inf]],
                    "stiffness": 0,
                    "springref": 0,
                    "damping": 0,
                    "extras": {},
                }
                for joint_name in joint_names
            ],
            "self_collisions": [],
        },
        "control": {
            "sensors": {
                "links": link_names,
                "joints": joint_names,
                "contacts": [],
                "xfrc": link_names,
                "muscles": [],
                "adhesions": [],
                "visuals": [],
            },
            "motors": [
                {
                    "joint_name": joint_name,
                    "control_types": [control_type],
                    "limits_torque": [-inf, inf],
                    "gains": gains,
                }
                for joint_name in joint_names
            ],
        },
        "extensions": [
            {
                "loader": controller_path,
                "config": control_pars,
            }
        ],
    }
    pyobject2yaml(sim_dir + "animat_config.yaml", animat_dict)


def gen_arena_config():
    """Arena configuration.

    Buoyancy is enabled so that the neutrally buoyant submarine (density =
    1000 kg/m^3) experiences zero net vertical force in water.  Drag is
    also enabled to provide realistic resistance at Re ~ 1000.
    """
    arena_dict = {
        "sdf": "../../sdfs/arena_flat_v0/sdf/arena_flat.sdf",
        "spawn": {
            "loader": 0,
            "mode": SpawnMode.FREE,
            "pose": [0, 0, 0, 0, 0, 0],
            "velocity": [0, 0, 0, 0, 0, 0],
            "extras": {},
        },
        "water": {
            "sdf": "../../sdfs/arena_water_v0/sdf/arena_water.sdf",
            # Enable drag and buoyancy for realistic 3-D underwater dynamics
            "drag": True,
            "buoyancy": True,
            # Water surface at z = 0.2 m (submarine spawns at z = 0.05 m)
            "height": 0.2,
            "velocity": [0, 0, 0],
            "viscosity": 1.0,
            "density": 1000.0,
            "maps": ["", ""],
        },
        "ground_height": -0.5,
    }
    pyobject2yaml(sim_dir + "arena_config.yaml", arena_dict)


def gen_experiment_config():
    """Experiment configuration."""
    experiment_dict = {
        "simulation": "simulation_config.yaml",
        "animats": ["animat_config.yaml"],
        "arenas": ["arena_config.yaml"],
        "loaders": {
            "simulation_options": "farms_core.simulation.options.SimulationOptions",
            "animats_options": ["farms_core.model.options.AnimatOptions"],
            "arenas_options": ["farms_core.model.options.ArenaOptions"],
            "experiment_data": "farms_amphibious.data.data.ExperimentData",
            "animats_data": ["farms_core.model.data.AnimatData"],
        },
    }
    pyobject2yaml(sim_dir + "experiment_config.yaml", experiment_dict)


def gen_simulation_config():
    """Simulation configuration (MuJoCo, 3-D, with gravity and water physics).

    The timestep of 0.01 s gives 100 steps per tail-beat at 1 Hz and is
    small enough to resolve the body dynamics accurately.  The gravity
    vector is set to [0, 0, -9.81] m/s^2; buoyancy in the arena config
    provides the upward counterforce for the neutrally buoyant submarine.
    """
    simulation_dict = {
        "units": {
            "length": "meter",
            "mass": "kilogram",
            "time": "second",
        },
        "runtime": {
            "n_iterations": 50001,
            "buffer_size": 50001,
            "play": True,
            "rtl": 1.0,
            "fast": False,
            "headless": headless,
            "show_progress": True,
        },
        "physics": {
            # 3-D simulation with full gravity; buoyancy compensates for
            # the neutrally buoyant submarine body
            "timestep": timestep,
            "gravity": [0, 0, -9.81],
            "num_sub_steps": 1,
            "cb_sub_steps": 2,
            "n_solver_iters": 1000,
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
            "visual_scale": 100.0,
            "extent": 10.0,
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
            "report_solver_analytics": 0,
        },
        "extensions": [
            {
                "loader": "farms_core.simulation.extensions.ExperimentLogger",
                "config": {"log_path": "output", "skip": 0},
            },
            {
                "loader": "farms_mujoco.simulation.extensions.MjcfSaver",
                "config": {"path": "output/simulation_mjcf.xml"},
            },
        ],
    }
    pyobject2yaml(sim_dir + "simulation_config.yaml", simulation_dict)


def gen_sh_config():
    """Shell script to launch the simulation."""
    sh_str = """#!/bin/bash
farmsim --experiment_config experiment_config.yaml "$@"
"""
    with open(sim_dir + "run.sh", "w") as f:
        f.write(sh_str)


if __name__ == "__main__":
    gen_animat_config()
    gen_arena_config()
    gen_experiment_config()
    gen_simulation_config()
    gen_sh_config()
    print("Submarine simulation configs written to:", sim_dir)
