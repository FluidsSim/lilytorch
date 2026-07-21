"""ep248 slow config: STABLE kp=0.2 servo + reference PRE-EMPHASIS.

The over-damped kp=0.2 servo realizes only ~86% of the commanded joint
amplitude (a near-uniform per-joint attenuation + a uniform ~2.5 ms lag that
does not affect amplitude). Instead of stiffening the servo (which hits a
BADQACC stability wall at kp~=0.4), we pre-multiply the reference angles by
1/tracking_ratio so the *realized* angles match the intended (real-fish)
trajectory. Servo gains and timestep are unchanged -> no new instability.

Per-joint gains measured from the kp=0.2 baseline run (T12:17), 0-0.40 s.

Usage
-----
    python gen_configs_pd_3d_ep248_slow_preemph.py
"""

from __future__ import annotations

import os

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow_implicit import (
    SimConfig as _SlowImplicitConfig,
)

# 1/tracking_ratio per joint (kp=0.2 baseline); last joint is passive -> 1.0.
_PREEMPH = [1.212, 1.213, 1.159, 1.154, 1.189, 1.145, 1.149,
            1.126, 1.109, 1.128, 1.145, 1.128, 1.269, 1.309, 1.0]

_HEADLESS = os.environ.get("PREEMPH_HEADLESS", "1") == "1"


class SimConfig(_SlowImplicitConfig):

    def __init__(self):
        super().__init__()

        # Keep the stable default servo; only pre-emphasise the reference.
        self.animats_pars[0]["control_pars"]["reference_gain"] = _PREEMPH

        if _HEADLESS:
            self.headless = True
            self.save = True
            self.save_frames = False

    def extra_simulation_extensions(self, output_folder):
        if _HEADLESS:
            return []
        return super().extra_simulation_extensions(output_folder)


if __name__ == "__main__":
    SimConfig().single_run()
