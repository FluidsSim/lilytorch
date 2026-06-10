"""Logs the boat base_link world-z and pitch each step (headless verification).

Pitch = bow up/down tilt about the world-y axis (the "front heavier than back"
trim the user is debugging).  The hull long axis (mesh-x) maps to world-x, so
the world-z component of the body's local x-axis gives the trim directly:
    pitch_deg = asin(z-component of body x-axis)   (+ = bow up, - = bow down).
"""
import math
import mujoco
from farms_core.simulation.extensions import TaskExtension


class BoatZLogger(TaskExtension):
    def __init__(self, experiment_options, log_path, print_every=50):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path
        self.print_every = int(print_every)
        self._it = 0; self._bid = None; self._rows = []

    @classmethod
    def from_options(cls, config, experiment_options):
        return cls(experiment_options, config.get("log_path", "boat_z.csv"),
                   config.get("print_every", 50))

    def initialize_episode(self, task, physics):
        m = physics.model.ptr
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if "base_link" in nm:
                self._bid = b; print(f"[zlog] tracking '{nm}' id={b}"); break

    def _pitch_deg(self, physics):
        # xmat is row-major 3x3; body local x-axis in world = (m0, m3, m6).
        xmat = physics.data.xmat[self._bid]
        axz = max(-1.0, min(1.0, float(xmat[6])))
        return math.degrees(math.asin(axz))

    def before_step(self, task, action, physics):
        if self._bid is None: return
        x = float(physics.data.xpos[self._bid][0])
        y = float(physics.data.xpos[self._bid][1])
        z = float(physics.data.xpos[self._bid][2]); t = float(physics.data.time)
        pitch = self._pitch_deg(physics)
        # yaw = heading of the body x-axis in the world x-y plane
        xmat = physics.data.xmat[self._bid]
        yaw = math.degrees(math.atan2(float(xmat[3]), float(xmat[0])))
        # applied hydrodynamic force/torque on the hull (world frame)
        fx = float(physics.data.xfrc_applied[self._bid][0])
        fz = float(physics.data.xfrc_applied[self._bid][2])
        ty = float(physics.data.xfrc_applied[self._bid][4])
        self._rows.append((self._it, t, x, z, pitch))
        if self._it % self.print_every == 0 or self._it < 5:
            print(f"[zlog] it={self._it:5d} t={t:.3f}s x={x:+.4f} y={y:+.4f} z={z:.4f} "
                  f"pitch={pitch:+.2f}deg yaw={yaw:+.2f}deg Fx={fx:+.1f}N "
                  f"Fz={fz:+.1f}N Ty={ty:+.1f}Nm",
                  flush=True)
        self._it += 1

    def end_episode(self, task, physics):
        with open(self.log_path, "w") as f:
            f.write("iteration,time,x,z,pitch_deg\n")
            for it, t, x, z, p in self._rows:
                f.write(f"{it},{t:.6f},{x:.6f},{z:.6f},{p:.6f}\n")
        print(f"[zlog] wrote {len(self._rows)} rows", flush=True)
