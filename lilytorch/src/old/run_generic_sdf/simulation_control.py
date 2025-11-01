#!/usr/bin/env python3
"""Run salamander simulation"""

from typing import Union
from farms_core import pylog
pylog.set_level('warning')
from farms_amphibious.control.network import AnimatNetwork
from farms_mujoco.sensors.camera import CameraCallback
from farms_amphibious.model.options import AmphibiousMotorOptions, AmphibiousLinkOptions
from farms_sim.simulation import (
    setup_from_clargs,
    run_simulation,
)
from farms_core.io.sdf import ModelSDF

from farms_amphibious.model.options import GenericOptions
from farms_amphibious.data.data import (
    GenericData,
    AmphibiousKinematicsData,
    get_amphibious_data,
)
from farms_amphibious.control.kinematics import KinematicsController
from farms_amphibious.control.amphibious import (
    GenericController,
    get_generic_controller,
)
from farms_core.model.options import JointOptions, LinkOptions
import numpy as np
import sys
import os

class EmptyController:

    """Test controller"""
    def __init__(self, n_iterations, timestep, njoints):
        self.timestep = timestep
        self.njoints  = njoints
        self.times    = np.linspace(0, n_iterations*timestep, n_iterations)
        self.state    = np.zeros((n_iterations, njoints*2)) # state array for recording all the variables

    def step(self):
        pass
        # return self.state[iteration,:]


class generic_animat_controller(AnimatNetwork):
    def __init__(self, data, n_iterations):
        super().__init__(data=data, n_iterations=n_iterations)

    def step(self, iteration, time, timestep):
        """Control step"""

def run_experiment(prepath, sdf_path):

    """Main"""
    args = [
        '--simulator',
        'MUJOCO',
        '--simulation_config', prepath+'simulation.yaml',
        '--animat_config', prepath+'animat.yaml',
        '--arena_config', prepath+'arena.yaml',
        '--log_path', 'output',
        '--profile', prepath+'output/simulation.profile',
    ]
    sys.argv += args

    # Setup
    pylog.info('Loading options from clargs')
    (
        clargs,
        animat_options,
        sim_options,
        arena_options,
        simulator,
    ) = setup_from_clargs(animat_options_loader=GenericOptions)
    animat_options.sdf = sdf_path

    model_sdf=ModelSDF.read(filename=os.path.expandvars(animat_options.sdf))[0]
    njoints = len(model_sdf.joints)

    # modify the position controller file to follow 0 positions
    tstop = sim_options["timestep"]*sim_options["n_iterations"]
    times = np.expand_dims(np.arange(0,tstop,sim_options["timestep"]),axis=1)
    data  = np.hstack((times,np.zeros((sim_options["n_iterations"],njoints))))
    path = os.path.join(os.path.dirname(__file__), "positions.csv")
    np.savetxt(path, data,delimiter=',')
    animat_options["control"]["kinematics_sampling"] = sim_options["timestep"]
    animat_options["control"]["kinematics_indices"] = list(range(1, njoints+1))
    animat_options["control"]["kinematics_time_index"] = 0
    animat_options["control"]["kinematics_invert"] = False
    animat_options["control"]["kinematics_degrees"] = False
    animat_options["control"]["kinematics_start"] = 0.0
    animat_options["control"]["kinematics_end"] = tstop

    joints=[]
    for joint in model_sdf.joints:
        if joint.name.startswith('joint_body_'):
            joints.append({
                'name': joint.name,
                'initial': [0, 0],
                'limits': [[-np.inf, np.inf], [-np.inf, np.inf]],
                'stiffness': 0,
                'springref': 0,
                'damping': 0,
                'extras': {},
            })
    animat_options["morphology"]["joints"] = [JointOptions(**joint) for joint in joints]

    links=[]
    for link in model_sdf.links:
        links.append({
            'name': link.name,
            'collisions': 'false',
            # the following are dummy parameters that I need to include
            'friction': [0.0, 0.0, 0.0],
            'extras': {
                'restitution': 0.0,
                'linearDamping': 0,
                'angularDamping': 0,
            },
            'density': 800.0,
            'swimming': False,
            'drag_coefficients': [
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
            ],
            'mass_multiplier': 1,
        })
    # from IPython import embed; embed()
    animat_options["morphology"]["links"] = [AmphibiousLinkOptions(**link) for link in links]
    animat_options["morphology"]["n_joints_body"] = len(animat_options["morphology"]["joints"])

    animat_options.control.contacts = [
        (link.name,'0')
        for link in model_sdf.links
    ]

    animat_options.control.sensors.links = [link.name for link in model_sdf.links]
    animat_options.control.sensors.joints = [joint.name for joint in model_sdf.joints]

    motors = [
        {
            'joint_name': joint.name,
            'control_types': ['position'],
            'limits_torque': [-np.inf, np.inf],
            'gains': [1.0e-01, 1.0e-03],
            'equation': 'position',
            'transform': {
                'gain': 1,
                'bias': 0,
            },
            # the following are dummy parameters that I need to include to run farms_amphibious
            'offsets': {
                'gain': 0,
                'bias': 0,
                'low': 1,
                'high': 5,
                'saturation': 0,
                'rate': 2,
            },
            'passive': {
                'is_passive': False,
                'stiffness_coefficient': 0,
                'damping_coefficient': 0,
                'friction_coefficient': 0,
            },
        } for joint in model_sdf.joints
    ]

    animat_options.control.motors = [AmphibiousMotorOptions(**motor) for motor in motors]


    # load animat data
    animat_data: Union[GenericData, AmphibiousKinematicsData] = (
        get_amphibious_data(
            animat_options=animat_options,
            simulation_options=sim_options,
        )
    )


    controller = EmptyController(sim_options.n_iterations, sim_options.timestep, njoints)

    # Additional engine-specific options
    options = {}
    options['callbacks'] = []


    animat_network = generic_animat_controller(animat_data, sim_options.n_iterations)

    if sim_options.video:
        camera = CameraCallback(
            camera_id=sim_options.camera_id,
            timestep=sim_options.timestep,
            n_iterations=sim_options.n_iterations,
            fps=sim_options.video_fps,
            speed=sim_options.video_speed,
            width=sim_options.video_resolution[0],
            height=sim_options.video_resolution[1],
            skips=sim_options.skips
        )
        options['callbacks'] += [camera]

    # Generic controller (OK)
    animat_controller: Union[GenericController, KinematicsController] = (
        get_generic_controller(
            animat_data=animat_data,
            animat_network=animat_network,
            animat_options=animat_options,
            sim_options=sim_options,
        )
    )

    # Simulation
    pylog.info('Creating simulation environment')
    options["handle_exceptions"] = False # handle the exception of physics error

    from IPython import embed; embed()

    controller.mujoco_error = False
    run_simulation(
        animat_data=animat_data,
        animat_options=animat_options,
        animat_controller=animat_controller,
        simulation_options=sim_options,
        arena_options=arena_options,
        simulator=simulator,
        save_mjcf=True,
    )



if __name__ == '__main__':
    yaml_dir = ""
    # run_experiment(yaml_dir,"models/1guilla_v1/sdf/1guilla.sdf")
    run_experiment(yaml_dir,"models/salamander_v5/sdf/salamander.sdf")

