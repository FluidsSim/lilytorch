"""Diagnostic extension: log the animat base-link world pose every step.

Additive integration helper (not core source). Records iteration, sim time,
the base link world position (x,y,z) and its orientation as roll/pitch/yaw
(extrinsic XYZ, radians) to a CSV so we can see whether the body pitches
(head-sinks) or rolls before it tumbles.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from farms_core.simulation.extensions import TaskExtension


class OrientationLogger(TaskExtension):
    def __init__(self, experiment_options, log_path, base_body_match="link0"):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path
        self.base_body_match = base_body_match
        self._body_id = None
        self._rows = []

    @classmethod
    def from_options(cls, config, experiment_options):
        return cls(
            experiment_options=experiment_options,
            log_path=config.get("log_path", "orientation_log.csv"),
            base_body_match=config.get("base_body_match", "link0"),
        )

    def _resolve_body(self, physics):
        model = physics.model
        n = model.nbody
        match = None
        for i in range(n):
            name = model.body(i).name
            if self.base_body_match in name:
                match = i
                break
        if match is None:
            # fall back: first body that is not 'world'
            match = 1 if n > 1 else 0
        self._body_id = match
        print(f"[OrientationLogger] logging body id={match} "
              f"name='{model.body(match).name}' -> {self.log_path}")
        # one-time mass / gravity report so fz (fluid) can be compared to weight
        g = np.asarray(model.opt.gravity, dtype=float)
        link_mass = 0.0
        for i in range(n):
            nm = model.body(i).name
            m = float(model.body(i).mass[0]) if hasattr(model.body(i), "mass") \
                else float(model.body_mass[i])
            if "link" in nm:
                link_mass += m
            print(f"[OrientationLogger]   body[{i}] '{nm}' mass={m:.6g}")
        print(f"[OrientationLogger] gravity={g}  sum(link masses)={link_mass:.6g}  "
              f"weight_z(MuJoCo units)={link_mass * g[2]:.6g}")

    def initialize_episode(self, task, physics):
        self._resolve_body(physics)
        self._rows = []

    def before_step(self, task, action, physics):
        if self._body_id is None:
            self._resolve_body(physics)
        d = physics.data
        pos = np.asarray(d.xpos[self._body_id], dtype=float).copy()
        quat_wxyz = np.asarray(d.xquat[self._body_id], dtype=float)  # (w,x,y,z)
        rpy = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler("xyz")
        it = int(getattr(task, "sim_iteration", len(self._rows)))
        t = float(getattr(d, "time", it))
        # net external (fluid) force on the whole system: xfrc_applied is
        # (nbody x 6) [fx,fy,fz, tx,ty,tz]; FARMS buoyancy/drag are disabled so
        # this is the BDIM hydro load. fz>0 is upward (buoyancy).
        xfrc = np.asarray(d.xfrc_applied, dtype=float)
        fz = float(xfrc[:, 2].sum())
        fx = float(xfrc[:, 0].sum())
        self._rows.append([it, t, pos[0], pos[1], pos[2],
                           rpy[0], rpy[1], rpy[2], fx, fz])
        # periodic flush so we keep data even if the run blows up / is killed
        if len(self._rows) % 50 == 0:
            self._flush()

    def end_episode(self, task, physics):
        self._flush()

    def _flush(self):
        if not self._rows:
            return
        arr = np.asarray(self._rows, dtype=float)
        header = "iter,time,x,y,z,roll,pitch,yaw,fx,fz"
        np.savetxt(self.log_path, arr, delimiter=",", header=header, comments="")
