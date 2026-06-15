"""Attitude stabiliser: a virtual keel/fin that keeps the body upright.

Applies a PD righting moment that drives the base link's "up" axis back to
vertical (kills roll + pitch) while leaving surge/sway/heave/yaw free, so a
planar-recorded swimming gait plays out without the body rolling over. The
torque is purely horizontal in world frame, so it does NOT fight yaw.

Implemented as a generalised force on the base free joint's rotational DOFs
(qfrc_applied), overwritten every step (no accumulation). Independent of the
BDIM hydro loads, which go through xfrc_applied.

Config: kp (righting stiffness, N·m/rad), kd (angular damping, N·m·s),
        base_body_match (default "link0").
"""

import numpy as np
from scipy.spatial.transform import Rotation

from farms_core.simulation.extensions import TaskExtension


class AttitudeStabilizer(TaskExtension):
    def __init__(self, experiment_options, kp=0.5, kd=0.05,
                 base_body_match="link0"):
        super().__init__()
        self.experiment_options = experiment_options
        self.kp = float(kp)
        self.kd = float(kd)
        self.base_body_match = base_body_match
        self._body_id = None
        self._dofadr = None

    @classmethod
    def from_options(cls, config, experiment_options):
        return cls(
            experiment_options=experiment_options,
            kp=config.get("kp", 0.5),
            kd=config.get("kd", 0.05),
            base_body_match=config.get("base_body_match", "link0"),
        )

    def _resolve(self, physics):
        import mujoco
        model = physics.model
        bid = next((i for i in range(model.nbody)
                    if self.base_body_match in model.body(i).name), 1)
        self._body_id = bid
        # the free joint attached to that body
        dofadr = None
        for j in range(model.njnt):
            if (model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                    and model.jnt_bodyid[j] == bid):
                dofadr = int(model.jnt_dofadr[j])
                break
        self._dofadr = dofadr
        print(f"[AttitudeStabilizer] body id={bid} "
              f"'{model.body(bid).name}' free-joint dofadr={dofadr} "
              f"kp={self.kp} kd={self.kd}")

    def initialize_episode(self, task, physics):
        self._resolve(physics)

    def before_step(self, task, action, physics):
        if self._dofadr is None:
            return
        d = physics.data
        bid, dof = self._body_id, self._dofadr
        R = Rotation.from_quat(
            np.asarray(d.xquat[bid], dtype=float)[[1, 2, 3, 0]]).as_matrix()
        up_body = R[:, 2]                          # body z-axis in world
        # restoring torque: rotate body-up toward world-up; horizontal => no yaw
        tau_restore = self.kp * np.cross(up_body, np.array([0.0, 0.0, 1.0]))
        # damping on the horizontal (roll/pitch) angular velocity only
        omega_local = np.asarray(d.qvel[dof + 3: dof + 6], dtype=float)
        omega_world = R @ omega_local
        tau_damp = -self.kd * np.array([omega_world[0], omega_world[1], 0.0])
        tau_world = tau_restore + tau_damp
        tau_local = R.T @ tau_world
        d.qfrc_applied[dof:dof + 3] = 0.0          # no applied translational force
        d.qfrc_applied[dof + 3: dof + 6] = tau_local
