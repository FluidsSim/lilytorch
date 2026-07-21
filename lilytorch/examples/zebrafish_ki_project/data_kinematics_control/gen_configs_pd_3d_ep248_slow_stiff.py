"""ep248 slow config with a STIFFER (higher-bandwidth) PD position servo, to
recover the ~12% recoil / body-wave amplitude the soft servo attenuates.

Root cause (measured): the default servo is kp=0.2, kv=0.001 -> heavily
over-damped -> closed-loop bandwidth ~kp/kv = 200 rad/s = 32 Hz, which rolls
off the 20-45 Hz recoil band. kp=0.2 already sits near the *explicit* stability
edge (the position spring is integrated explicitly by MuJoCo's implicitfast),
held stable only by the large kv, so simply raising kp (or lowering kv) blows
up. To raise bandwidth AND keep stability at fixed joint inertia we buy
headroom two ways:
  1. Halve the timestep (0.00025 -> 0.000125): 4x the explicit-spring ceiling.
  2. Strong (implicit) FSI coupling: removes the added-mass lag that makes a
     faster-tracking body destabilise the explicit fluid coupling.
Then push the servo to kp=0.6, kv=0.0008 -> bandwidth ~750 rad/s = 120 Hz.

Set STIFF_TEST_STEPS below for a quick stability probe; None -> full 0.55 s.

Usage
-----
    python gen_configs_pd_3d_ep248_slow_stiff.py
"""

from __future__ import annotations

import os

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow_implicit import (
    SimConfig as _SlowImplicitConfig,
)

# Tunable via environment for quick ladders:
#   STIFF_KP, STIFF_KV, STIFF_DT, STIFF_STEPS
_KP    = float(os.environ.get("STIFF_KP", "0.6"))
_KV    = float(os.environ.get("STIFF_KV", "0.0008"))
_DT    = float(os.environ.get("STIFF_DT", "0.000125"))
_STEPS = int(os.environ.get("STIFF_STEPS", "600"))


class SimConfig(_SlowImplicitConfig):

    def __init__(self):
        super().__init__()

        # ── Stiffer, higher-bandwidth position servo ────────────────
        # gains = [kp, kv, ki]; bandwidth ~ kp/kv.
        self.animats_pars[0]["gains"] = [_KP, _KV, 0]

        # ── Smaller timestep for explicit-spring stability headroom ──
        self.timestep = _DT
        self.bdim_dt = self.timestep

        self.n_iterations = _STEPS
        self.bdim_nt = self.n_iterations + 1

        # ── Cheap/headless: no video, no GL overlays ────────────────
        self.headless = True
        self.save = True
        self.save_frames = False

    def extra_simulation_extensions(self, output_folder):
        # Drop the video recorders / GL viewer / light+sky modifiers so the
        # probe runs headless and fast; keep only the solver + logging.
        return []


if __name__ == "__main__":
    SimConfig().single_run()
