"""ep248 slow config with FREQUENCY-DEPENDENT servo pre-emphasis (lead
compensator), identical to gen_configs_pd_3d_ep248_slow_implicit.py otherwise.

Fixes the body-wave HARMONIC deficit (2026-07-21).  The position servo is a
first-order lag with its corner at kp/kv = 0.2/0.001 = 200 rad/s = 31.83 Hz.
Measured realized/commanded joint-angle ratio per tail-beat harmonic:

    run                             1f0     2f0     3f0     4f0
    baseline kp=0.2               0.892   0.707   0.550   0.434
    + scalar pre-emphasis 1.15x   1.029   0.829   0.646   0.512

Fitting |H| = 1/sqrt(1+(f/fc)^2) to each harmonic independently gives
fc = 32.3 / 32.7 / 32.3 / 31.5 Hz -- the plant is first order with the corner
exactly at kp/kv, and 2f0 lands on the -3 dB point.  The scalar gain lifts
every frequency equally, so it fixes the fundamental and leaves the harmonics
17/35/49% short.  That deficit is what remains of the sim-vs-real 20-45 Hz
amplitude gap once the tracking's midline-length artifact is accounted for.

The lead compensator inverts the known plant exactly:

    ref_preemph(t) = ref(t) + (1/wc) * d(ref)/dt      wc = 2*pi*31.83

RESULT (2026-07-21): the correction WORKS at the joint level and CONFIRMS the
diagnosis, but NO variant produced a usable episode.  **Do not use this config
for production runs.**

    variant                       steps   1f0    2f0    3f0    4f0
    baseline kp=0.2                2200  0.892  0.707  0.550  0.434
    scalar pre-emphasis 1.15x      2199  1.029  0.829  0.646  0.512
    all joints, cap 2.5             579  0.972  0.978  0.972  0.698   CRASHED
    all joints, cap 1.5             820  1.014  0.987  0.892  0.725   CRASHED
    skip 3 tail joints, cap 1.5    2200  0.992  0.962  0.772  0.599   ran, but
                                                                      PATHOLOGICAL

The joint-angle tracking is corrected exactly as predicted (2f0 0.707 -> 0.96),
which proves the first-order-lag diagnosis.  But:
  * all-joint variants die with mjWARN_BADQACC at the posterior DOFs
    (~1e-13 kg m^2, no armature -- FARMS does not plumb it), the same
    fragility that pins kp at 0.2;
  * the skip-tail variant runs to completion but the FREE-BODY response is
    pathological -- mean CoM speed 32.9 BL/s with only 1.05 BL of net
    progress, versus the baseline's 6.67 BL/s and 3.45 BL.  The fish thrashes
    in place.  Boosting joints 0-11 while leaving 12-14 on the raw reference
    almost certainly kinks the travelling wave at the junction.
    **A numerically clean run is not a physically valid one -- always check
    net displacement, not just that the episode completed.**

The blocker is the model, not the compensator: the posterior DOFs cannot
absorb any increase in commanded high-frequency content.  The real fix is
joint ARMATURE (absent from the MJCF -- verified, no `armature` attribute
anywhere, tail diaginertia ~1e-13).  It is reachable by wrapping
``farms_mujoco.simulation.mjcf.setup_mjcf_xml`` the way ``_offscreen_patch.py``
already does, but choosing the value changes the passive dynamics and is a
modelling decision, not a bug fix.

``reference_gain`` is deliberately NOT set: the correction already returns the
fundamental to unity, and stacking both would over-drive it.

NB the first attempt used the textbook time-domain lead, ref + (1/wc)*dref/dt.
That is an unbounded high-pass and it DIED at ~iter 450 with mjWARN_BADQACC at
the tail DOFs (~1e-12 kg m^2, no armature) -- the same failure mode as the
kp>=0.4 probes.  The shipped version is a CAPPED, ZERO-PHASE magnitude
correction (``servo_preemphasis_cap``, default 2.5, which exceeds the 2.29
needed at 4f0 so the harmonics of interest are untouched).

Usage
-----
    python gen_configs_pd_3d_ep248_slow_lead.py
"""

from __future__ import annotations

import math
import os

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow_implicit import (
    SimConfig as _SlowImplicitConfig,
)

# Boost cap.  Required boost per harmonic: 1f0 1.125, 2f0 1.430, 3f0 1.835,
# 4f0 2.289.  cap=2.5 corrects all four but DIED at iter 579 (BADQACC, tail
# DOFs).  cap=1.5 is the smallest that still fully corrects 1f0 AND 2f0 --
# and 2f0 = 32.7 Hz is the whole 20-45 Hz band in question -- while demanding
# far less of the servo above that.  Override with LEAD_CAP.
LEAD_CAP = float(os.environ.get("LEAD_CAP", 1.5))

# Number of TRAILING joints left uncorrected.  The posterior DOFs (~1e-13
# kg m^2, no armature) are where mjWARN_BADQACC fires; cap=1.5 on all 15
# joints died at iter 820.  LEAD_SKIP_TAIL=3 leaves joints 12-14 alone.
LEAD_SKIP_TAIL = int(os.environ.get("LEAD_SKIP_TAIL", 0))


class SimConfig(_SlowImplicitConfig):

    def __init__(self):
        super().__init__()

        kp, kv, _ = self.animats_pars[0]["gains"]
        pars = self.animats_pars[0]["control_pars"]
        pars["servo_corner_hz"] = kp / (2 * math.pi * kv)
        pars["servo_preemphasis_cap"] = LEAD_CAP
        if LEAD_SKIP_TAIL > 0:
            pars["servo_preemphasis_joints"] = list(range(15 - LEAD_SKIP_TAIL))


if __name__ == "__main__":
    SimConfig().single_run()
