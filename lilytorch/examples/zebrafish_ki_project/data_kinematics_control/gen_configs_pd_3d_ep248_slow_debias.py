"""ep248 slow config: straight-swim gait DE-BIASING (Issue 2 / turning).

RESULT (2026-07-20, see HANDOFF_sim_real_mismatch.md): this did NOT work.
Zeroing the gait's net curvature makes the sim turn MORE, not less
(max lateral 0.66 -> 2.76 BL, forward progress +3.4 -> -0.7 BL), because the
turn is emergent open-loop hydrodynamics, not the prescribed -3.9 deg bias.
Kept as a documented negative-result diagnostic. Do not re-chase this lever.

Identical to gen_configs_pd_3d_ep248_slow_implicit.py except the reference
joint trajectory is de-biased: each joint's time-mean is subtracted so the
mean posture is perfectly straight (net body curvature 0 deg).

Rationale (HANDOFF_sim_real_mismatch.md, Open Q2): the extracted ep248 gait
carries a net body curvature of -3.9 deg. Over ~9 tail beats that mean is
within ~1 SE of zero (mean/SE ~ -0.28) — it is the residual of a symmetric
oscillation with std ~42 deg, NOT a real steady turning command. The real
fish swam straight (0.1 deg net heading) carrying that same gait. But the sim
body is open-loop (no fins, no heading hold), so it integrates the residual
bias into a spurious net turn (~+50 deg). Zeroing the per-joint means removes
the phantom steady bias while preserving the full oscillation amplitude.

Diagnostic A/B against the implicit T12:17 run (net turn ~+53 deg): how much
of the sim's turn is the integrated gait bias vs. genuine open-loop drift.

Usage
-----
    python gen_configs_pd_3d_ep248_slow_debias.py
"""

from __future__ import annotations

import os

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow_implicit import (
    SimConfig as _SlowImplicitConfig,
)

_HEADLESS = os.environ.get("DEBIAS_HEADLESS", "1") == "1"
# "net"  -> remove only the summed net curvature (uniform tiny shift, keeps
#           per-joint camber): the faithful test of Open Q2 (0 deg net curvature).
# "true" -> per-joint demean (straight mean posture at every joint).
_DEBIAS_MODE = os.environ.get("DEBIAS_MODE", "net")


class SimConfig(_SlowImplicitConfig):

    def __init__(self):
        super().__init__()

        # Straight-swim de-biasing (see module docstring).
        self.animats_pars[0]["control_pars"]["debias_reference"] = (
            "net" if _DEBIAS_MODE == "net" else True
        )

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
