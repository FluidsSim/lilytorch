"""Diagnostic controller: hold all joints at zero (no undulation).

Subclasses the gait PositionController but replaces the recorded kinematics
with a constant zero trajectory, so the body floats straight under buoyancy
alone. Used to separate gait-driven tumbling from passive instability.
"""

import numpy as np

from lilytorch.examples._1guillasim.experiments.controller import (
    PositionController,
)


class StraightController(PositionController):

    def load_positions(self, file_path, goal=True, plot=False):
        # 8 motors, two time samples spanning a long horizon, all angles 0.
        times = np.array([0.0, 1e6])
        thetas = np.zeros((2, 8))
        return np.column_stack([times, thetas])
