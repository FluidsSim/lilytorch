"""ep248 slow config, IDENTICAL to gen_configs_pd_3d_ep248_slow.py except that
the FARMS<->BDIM coupling is STRONG (implicit) instead of explicit.

Diagnostic A/B test: does the whole-body forward surge (present in the real
fish, absent in the explicit-coupling sim) reappear once the fluid force and
the free-body acceleration are converged within each step?  Explicit coupling
can under-represent the acceleration-reaction (added-mass) force, so the body
never feels the per-stroke thrust impulse — implicit coupling removes that lag.

Accelerator is Aitken (Irons-Tuck) rather than the default IQN-ILS: IQN-ILS
reuse (default 2 windows) is a known free-swimmer blow-up mode; Aitken carries
no cross-step history, so it is the robust choice for this test.

Usage
-----
    python gen_configs_pd_3d_ep248_slow_implicit.py
"""

from __future__ import annotations

from lilytorch.examples.zebrafish_ki_project.data_kinematics_control.gen_configs_pd_3d_ep248_slow import (
    SimConfig as _SlowExplicitConfig,
)


class SimConfig(_SlowExplicitConfig):

    def __init__(self):
        super().__init__()

        # ── Strong (implicit) FSI coupling ───────────────────────────
        # Converge the fluid load <-> body acceleration fixed point every
        # step. force_relaxation is ignored under implicit coupling.
        self.coupling = {
            "scheme":      "implicit",
            "accelerator": "aitken",   # avoid IQN-ILS reuse poisoning (free swimmer)
            "tol":         1.0e-4,     # relative interface-residual tolerance
            "max_iter":    30,         # coupling sweeps per step
        }

        # ── Per-link hydro force logging ─────────────────────────────
        # Writes output/drags.h5 with per-link viscous + pressure force
        # (world frame, N) and torque about each link COM (N m), full
        # time history. Needed to check the anterior lateral-force / yaw-
        # torque balance behind the head over-wag (2.7x real). The record
        # buffers are already filled every step; this only enables the
        # end-of-episode write, so the runtime cost is nil.
        self.save_drags = True


if __name__ == "__main__":
    SimConfig().single_run()
