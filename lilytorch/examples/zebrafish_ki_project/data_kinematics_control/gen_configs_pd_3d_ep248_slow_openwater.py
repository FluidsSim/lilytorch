"""ep248 slow config, IDENTICAL to gen_configs_pd_3d_ep248_slow_implicit.py
except for the SPAWN POSE.

Hypothesis under test (2026-07-21): the sim's residual turn (+53 deg net body
yaw vs the real fish's ~0) is a HYDRODYNAMIC WALL INTERACTION, not an
open-loop instability of the finless body.

The production config spawns the fish at yaw 134.93 deg, i.e. heading
-45.07 deg in world (heading = yaw - 180), aimed diagonally at the ymin wall
of a 0.10 x 0.05 x 0.0125 m arena.  With BL = 0.018 m the arena is only
2.8 BL wide, and the fish crosses it: lateral wall clearance falls
monotonically 1.0 BL -> 0.20 BL, and the whole turn happens over exactly that
interval (heading flat within +-10 deg while clearance > 0.6 BL, then ramping
to +48 deg).  A sustained lateral force appears at the same time
(+-8 uN mean while clearance > 0.5 BL, +34..39 uN once below 0.33 BL).  The
final heading is ~parallel to that wall -- the classic near-wall
parallel-alignment response of a swimmer.

This config removes the confound WITHOUT touching the grid, resolution,
physics, gait, controller or coupling: the fish is spawned heading +x, along
the long axis of the box, on the centreline y = 0.  Lateral clearance is then
~1.17 BL and CONSTANT for the whole run instead of decaying to 0.20 BL.

    heading = yaw - 180 deg  ->  yaw = pi gives heading = 0 (+x).
    The model CoM sits 0.00707 m BEHIND the spawn origin along the heading,
    so origin x = +0.005 puts the CoM at x = -0.002: ~0.53 BL of clearance
    behind the tail at t = 0 and ~0.69 BL ahead of the head at t_end.
    Fore/aft walls are LEFT-RIGHT SYMMETRIC about the fish midline and so
    cannot bias the yaw; only the lateral clearance matters here.

Prediction: if the turn is wall-driven, net body yaw collapses from +53 deg to
a few degrees (matching the real fish) with no other change.  If the turn is
intrinsic to the finless open-loop body, it survives unchanged.

Usage
-----
    python gen_configs_pd_3d_ep248_slow_openwater.py
"""

from __future__ import annotations

import math

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow_implicit import (
    SimConfig as _SlowImplicitConfig,
)


class SimConfig(_SlowImplicitConfig):

    def __init__(self):
        super().__init__()

        # Heading +x along the long axis, on the y centreline.
        self.animats_pars[0]["pose"] = [0.005, 0, 0, 0, 0, math.pi]


if __name__ == "__main__":
    SimConfig().single_run()
