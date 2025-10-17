#!/usr/bin/env python3
"""Run salamander simulation"""

from typing import Union
import util.update_pars as util
import numpy as np

from farms_core import pylog
pylog.set_level('warning')

from farms_core.simulation.options import Simulator
from farms_core.model.options import SpawnMode
from farms_mujoco.sensors.camera import CameraCallback
from farms_sim.simulation import (
    setup_from_clargs,
    run_simulation,
)
import util.callbacks as callbacks

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
from util.rw import Dict2Class

from controllers.wave_controller import WaveController
from controllers.empty_controller import EmptyController
from util.controller import *

from dm_control.rl.control import PhysicsError

import sys


def run_experiment(pars):

    """Main"""
    prepath = pars.yaml_path
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

    # modify simulation options according to the user pars
    sim_options ["timestep"]      = pars.timestep
    sim_options ["n_iterations"]  = pars.n_iterations
    sim_options ["buffer_size"]   = pars.n_iterations
    sim_options ["video"]         = pars.video_record
    sim_options ["headless"]      = pars.headless
    sim_options ["fast"]          = pars.fast
    sim_options ["video_type"]    = pars.video_type
    sim_options ["video_speed"]   = pars.video_speed
    sim_options ["video_fps"]     = pars.video_fps
    sim_options ["record_path"]   = pars.log_path+pars.video_name
    sim_options ["camera_id"]     = pars.camera_id
    sim_options ["show_progress"] = pars.show_progress
    sim_options ["gravity"]       = pars.gravity
    sim_options["video_fps"]      = 10
    sim_options["skips"]          = 10

    animat_options["spawn"]         = Dict2Class(pars.spawn)
    animat_options.spawn.mode = SpawnMode(pars.spawn["mode"])

    # update muscle and drag parameters
    control_type=animat_options["control"]["motors"][0]["control_types"][0]
    if control_type=="torque":
        util.update_muscle_param(animat_options)
    elif control_type=="position":
        util.update_pd_gains(animat_options)

    if pars.swimming_mode=="drag":
        util.update_drag_param(animat_options)
    # util.update_swimming_mode(arena_options,swimming_mode=swimming_mode)


    if pars.random_spine==True:
        joints = animat_options["morphology"]["joints"]
        for joint in joints:
            joint["initial"]=[0.2*np.random.randn(),0]

    # load animat data
    animat_data: Union[GenericData, AmphibiousKinematicsData] = (
        get_amphibious_data(
            animat_options=animat_options,
            simulation_options=sim_options,
        )
    )

    if pars.controller == "sine" or pars.controller == "square":
        controller = WaveController(pars)
    elif pars.controller == "empty":
        controller = EmptyController(pars)
    else:
        raise ValueError(f"Unknown controller: {pars.controller}")

    # Additional engine-specific options
    options = {}
    camera=None

    options = {}
    options['callbacks'] = []
    camera=None
    assert simulator == Simulator.MUJOCO
    if pars.swimming_mode=="-":
        print('Using swimming mode: -')
    elif pars.swimming_mode=="drag":
        print('Using drag swimming mode')
        options['callbacks'] += [
                callbacks.DragCallback(animat_options, arena_options),
            ]
        if control_type=="torque":
            animat_network = DragMuscleController(animat_data, controller, sim_options.n_iterations)
        elif control_type=="position":
            animat_network = DragPositionController(animat_data, controller, sim_options.n_iterations)
        else:
            raise ValueError(f"Unknown control type: {control_type}, need position or torque")
    elif pars.swimming_mode=="bdim":
        print('Using bdim swimming mode')
        animat_network = BDIMController(animat_data, sim_options, controller, controller.pars.yaml_file, sim_options.n_iterations, control_type)
        options['callbacks'] += [
                callbacks.FluidCallback(animat_options, arena_options, animat_network),
            ]
    else:
        raise ValueError(f"Unknown swimming mode: {pars.swimming_mode}")


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

    controller.mujoco_error = False
    try:
        run_simulation(
            animat_data=animat_data,
            animat_options=animat_options,
            animat_controller=animat_controller,
            simulation_options=sim_options,
            arena_options=arena_options,
            simulator=simulator,
            **options,
        )
    except PhysicsError:
        controller.mujoco_error = True

    if sim_options.video:
        print("Saving video in "f'{sim_options.record_path}')
        if sim_options.video_type=="gif":
            camera.save(
                filename=f'{sim_options.record_path}.gif',
                iteration=sim_options.n_iterations,
                writer='pillow',
            )
        elif sim_options.video_type=="html":
            camera.save(
                filename=f'{sim_options.record_path}.html',
                iteration=sim_options.n_iterations,
                writer='html',
            )
        else:
            camera.save(
                filename=f'{sim_options.record_path}.mp4',
                iteration=sim_options.n_iterations,
                writer='ffmpeg',
            )

    return animat_data, controller

