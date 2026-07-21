"""ep248 slow config, IDENTICAL to gen_configs_pd_3d_ep248_slow_implicit.py
except that the GAIT IS MIRRORED: ``kinematics_invert: True`` makes
lilytorch.integration.kinematics negate every commanded joint angle
(``kinematics *= -1``), i.e. the exact left/right reflection of the prescribed
swimming motion.

Left/right symmetry test (2026-07-21).  Every run of this episode drifts in the
SAME direction (+40..+50 deg of slow yaw drift, always to the same side).  Two
things could produce that:

  1. the drift is a deterministic response to the prescribed gait, and the
     body + solver are left/right symmetric;
  2. the body mesh, its staggered SDF, or the solver carries a left/right
     asymmetry that biases the yaw regardless of the gait.

Mirroring the gait separates them, because for a symmetric model the mirrored
gait must produce the exactly mirrored trajectory:

    slow drift -> about -40 deg (mirror of baseline)  =>  case 1, model is
        symmetric and the drift is gait-determined; the open question is then
        purely why the real fish arrests the drift and the sim does not.
    slow drift stays positive, or is strongly asymmetric in magnitude
        (e.g. -15 deg against the baseline's +41)      =>  case 2, a genuine
        left/right asymmetry in the model or solver -- a bug, not biology.

Compare heading(t) of this run against MINUS heading(t) of the baseline
(2026-07-20T12:17:59) with analyze_heading_divergence.py.

Usage
-----
    python gen_configs_pd_3d_ep248_slow_invert.py
"""

from __future__ import annotations

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow_implicit import (
    SimConfig as _SlowImplicitConfig,
)


class SimConfig(_SlowImplicitConfig):

    def __init__(self):
        super().__init__()

        self.animats_pars[0]["control_pars"]["kinematics_invert"] = True


if __name__ == "__main__":
    SimConfig().single_run()
