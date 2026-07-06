import os
import sys

# Put THIS directory on the child sim's PYTHONPATH so the generated simulation
# (run from a timestamped output folder) can import the local ``_verify_zlog``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from gen_configs import SimConfig

class VerifyConfig(SimConfig):
    def __init__(self):
        super().__init__()
        self.headless = True
        self.n_iterations = int(os.environ.get("VERIFY_NITER", "600"))
        self.bdim_nt = self.n_iterations
        self.save_frames = False
        self.save = False

    def extra_simulation_extensions(self, output_folder):
        return [{"loader": "_verify_zlog.BoatZLogger",
                 "config": {"log_path": os.path.join(output_folder, "boat_z.csv"),
                            "print_every": 50}}]

if __name__ == "__main__":
    print("=== VERIFY DSYHS boat ===", flush=True)
    VerifyConfig().run()
