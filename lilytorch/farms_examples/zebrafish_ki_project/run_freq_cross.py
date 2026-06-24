"""Frequency-cross study for the position-controlled zebrafish.

Runs the PD-controlled zebrafish under four kinematics, the 2x2 cross of
{slow shape, fast shape} x {slow frequency, fast frequency}:

    slow_slow   slow shape (joints_positions_slow_sigmoid.xlsx)  @ f_slow
    fast_fast   fast shape (joints_positions_fast.xlsx)          @ f_fast
    slow_fast   slow shape                                        @ f_fast   (new)
    fast_slow   fast shape                                        @ f_slow   (new)

The recorded trajectories play at their *natural* frequency when sampled at
``S0 = 0.00025 s`` (verified: slow tail beat ~= 11.4 Hz, fast ~= 20.0 Hz,
matching controller_parameters.py). Because the controller maps each xlsx
sample onto a real-time grid of ``n_samples * kinematics_sampling`` seconds,
the playback frequency is inversely proportional to ``kinematics_sampling``::

    f_played = f_natural * (S0 / sampling)  =>  sampling = S0 * f_natural / f_target

So retargeting a recorded gait to a new frequency needs no new CSV, only a
rescaled ``kinematics_sampling``.

Each case is launched headless (no video), with FlowDiagnostics enabled so the
fluid kinetic-energy time-series is written to ``diagnostics.h5`` next to the
FARMS ``output/simulation.hdf5``. Runs land under::

    /data/andreaferrario/ns_data/zebrafish_freq_cross/<case>/<timestamp>/

Usage
-----
    python run_freq_cross.py              # run all four cases sequentially
    python run_freq_cross.py slow_fast    # run a single case
"""

import sys

from controller_parameters import (
    SLOW_SWIMMING_CONTROLLER_PARAMETERS as _SLOW,
    FAST_SWIMMING_CONTROLLER_PARAMETERS as _FAST,
)
from gen_configs_pd_3d_slow_fast import SimConfig as ZFConfig

# Natural tail-beat frequencies of the two recorded gaits at sampling S0.
F_SLOW = _SLOW["frequency"]   # ~= 11.417 Hz
F_FAST = _FAST["frequency"]   # ~= 19.971 Hz
S0 = 0.00025                  # base kinematics_sampling at which the gaits are natural


def _sampling_for(f_natural, f_target):
    """kinematics_sampling that makes a gait of natural freq *f_natural* play at *f_target*."""
    return S0 * f_natural / f_target


# (shape mode, target frequency, kinematics_sampling)
CASES = {
    "slow_slow": ("slow", F_SLOW, _sampling_for(F_SLOW, F_SLOW)),  # == S0
    "fast_fast": ("fast", F_FAST, _sampling_for(F_FAST, F_FAST)),  # == S0
    "slow_fast": ("slow", F_FAST, _sampling_for(F_SLOW, F_FAST)),  # slow shape sped up
    "fast_slow": ("fast", F_SLOW, _sampling_for(F_FAST, F_SLOW)),  # fast shape slowed down
}

STACK_ROOT = "zebrafish_freq_cross"


class FreqCrossConfig(ZFConfig):
    """ZF position-control config pinned to one (shape, frequency) case."""

    def __init__(self, case):
        super().__init__()
        mode, _f_target, sampling = CASES[case]

        # ── Retarget the recorded gait: pick the xlsx (mode) + playback rate.
        cpars = self.animats_pars[0]["control_pars"]
        cpars["mode"] = mode
        cpars["kinematics_sampling"] = sampling

        # ── Persist the fluid kinetic-energy time-series to diagnostics.h5.
        #    save=True (with a huge save_every) only exists to create the
        #    solver save_path; save_frames stays off so no field frames dump.
        self.save = True
        self.save_frames = False
        self.save_every = 10_000_000
        self.diagnostics_every = 2

        # ── Each case to its own stack folder so metrics can find the output.
        self.stack_folder = f"{STACK_ROOT}/{case}"

        # ── Fast-dynamics cases need a halved timestep for stability.
        if _f_target == F_FAST:
            self.timestep /= 2
            self.bdim_dt = self.timestep

    # NOTE: extra_simulation_extensions is inherited from the parent SimConfig
    # so each case records the same videos (top-down / follow / back) and shows
    # the FlowIsoGLViewer, exactly like gen_configs_pd_3d_slow_fast.py.


def _run_case(case):
    print(f"\n=== freq-cross case: {case}  "
          f"(mode={CASES[case][0]}, f_target={CASES[case][1]:.4f} Hz, "
          f"sampling={CASES[case][2]:.6e} s) ===")
    FreqCrossConfig(case).run()


def main():
    cases = sys.argv[1:] or list(CASES)
    for case in cases:
        if case not in CASES:
            raise SystemExit(f"Unknown case {case!r}. Choose from: {', '.join(CASES)}")
    for case in cases:
        _run_case(case)


if __name__ == "__main__":
    main()
