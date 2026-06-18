"""Headless swim-speed logger for the submerged-vs-surface diagnostic.

Logs the base-link (head) position and root-joint velocity every step to a CSV
so the steady-state forward swim speed can be recovered as the slope of x(t)
(robust to the spawn mode) or read directly from the surge DOF velocity.

Used by the body-in-air speed-bias investigation (HP5b): an A/B of two
``SpawnMode.TRANSVERSE`` runs that differ ONLY in spawn depth — surface
(dorsal surface in air) vs fully submerged (whole body in water).  Both lock
heave/roll/pitch, so the only difference is dorsal air-exposure; a speed drop
from surface->submerged isolates the body-in-air drag-unloading effect.

Plugs into the FARMS ``TaskExtension`` hook chain like ``DataLogger`` /
``RealtimeMonitor`` (``from_options(config, experiment_options)`` +
``before_step``).  No-ops gracefully if the base body can't be found.
"""

import os

from farms_core.simulation.extensions import TaskExtension


class SpeedLogger(TaskExtension):
    """Append ``it, t, x, y, z, vx_fd, qv0, qv1, qv2`` to a CSV every step.

    ``vx_fd`` is the finite-difference forward velocity of the base link
    (``dx/dt`` over one physics step) — the spawn-mode-agnostic speed.
    ``qv0..2`` are the first three root-joint velocities (for TRANSVERSE:
    surge, sway, yaw-rate); written as NaN when the root has no free DOFs.
    """

    def __init__(self, experiment_options, log_path, base_body_match="link0"):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path
        self.base_body_match = base_body_match
        self._fh = None
        self._bid = None
        self._dt = None
        self._prev_x = None
        self._it = 0

    @classmethod
    def from_options(cls, config, experiment_options):
        return cls(
            experiment_options=experiment_options,
            log_path=config.get("log_path", "/tmp/speed_log.csv"),
            base_body_match=config.get("base_body_match", "link0"),
        )

    def initialize_episode(self, task, physics):
        del task
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._fh = open(self.log_path, "w")
        self._fh.write("it,t,x,y,z,vx_fd,qv0,qv1,qv2\n")
        self._fh.flush()
        self._dt = float(physics.model.opt.timestep)
        self._prev_x = None
        self._it = 0
        # Resolve the base body id by name substring match.
        self._bid = None
        m = physics.model
        for i in range(m.nbody):
            name = m.id2name(i, "body")
            if name and self.base_body_match in name:
                self._bid = i
                break
        if self._bid is None:
            print(f"[SpeedLogger] WARNING: no body matching "
                  f"'{self.base_body_match}'; logging root only.", flush=True)

    def before_step(self, task, action, physics):
        del task, action
        d = physics.data
        x = y = z = float("nan")
        if self._bid is not None:
            xpos = d.xpos[self._bid]
            x, y, z = float(xpos[0]), float(xpos[1]), float(xpos[2])
        vx = float("nan")
        if self._prev_x is not None and self._dt > 0:
            vx = (x - self._prev_x) / self._dt
        self._prev_x = x
        qv = d.qvel
        qv0 = float(qv[0]) if qv.shape[0] > 0 else float("nan")
        qv1 = float(qv[1]) if qv.shape[0] > 1 else float("nan")
        qv2 = float(qv[2]) if qv.shape[0] > 2 else float("nan")
        t = self._it * self._dt
        self._fh.write(f"{self._it},{t:.5f},{x:.6f},{y:.6f},{z:.6f},"
                       f"{vx:.6f},{qv0:.6f},{qv1:.6f},{qv2:.6f}\n")
        if self._it % 100 == 0:
            self._fh.flush()
            print(f"[SpeedLogger] it={self._it} x={x:.4f} vx_fd={vx:.4f} "
                  f"qv0={qv0:.4f}", flush=True)
        self._it += 1

    def end_episode(self, task, physics):
        del task, physics
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
