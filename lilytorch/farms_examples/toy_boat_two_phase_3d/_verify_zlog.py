"""Logs the boat base_link world-z each step (headless verification)."""
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

    def before_step(self, task, action, physics):
        if self._bid is None: return
        z = float(physics.data.xpos[self._bid][2]); t = float(physics.data.time)
        self._rows.append((self._it, t, z))
        if self._it % self.print_every == 0:
            print(f"[zlog] it={self._it:5d} t={t:.3f}s z={z:.4f}", flush=True)
        self._it += 1

    def end_episode(self, task, physics):
        with open(self.log_path, "w") as f:
            f.write("iteration,time,z\n")
            for it, t, z in self._rows: f.write(f"{it},{t:.6f},{z:.6f}\n")
        print(f"[zlog] wrote {len(self._rows)} rows", flush=True)
