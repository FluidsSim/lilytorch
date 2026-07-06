import os, sys
from gen_configs_swim_2d import SimConfig as Sal2D

DOWNSCALE = int(os.environ.get("STUDY_DOWNSCALE", "2"))
N_ITERS = int(os.environ.get("STUDY_NITERS", "800"))
RESULTS_DIR = "/data/andreaferrario/lilytorch/_overlap_study"

CONDITIONS = {
    "sal2d_noblend": dict(convexify=True, force_method="lagrangian",
                          body_velocity_blend_eps_cells=None),
    "sal2d_blend":   dict(convexify=True, force_method="lagrangian",
                          body_velocity_blend_eps_cells=2.0),
    "sal2d_nat_noblend": dict(convexify=True, body_velocity_blend_eps_cells=None),
    "sal2d_nat_blend":   dict(convexify=True, body_velocity_blend_eps_cells=2.0),
}


class StudyConfig(Sal2D):
    def __init__(self, ov):
        super().__init__()
        self.headless = True; self.save = False; self.save_frames = False
        self.Nx //= DOWNSCALE; self.Ny //= DOWNSCALE
        self.n_iterations = N_ITERS; self.bdim_nt = self.n_iterations + 1
        self.save_every = 100000
        for k, v in ov.items():
            setattr(self, k, v)

    def extra_simulation_extensions(self, output_folder):
        return []

    def _extra_run_patch(self):
        return "import lilytorch.integration._overlap_diag as _d;_d.install();"


def main():
    name = sys.argv[1]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["LILY_DIAG_CSV"] = os.path.join(RESULTS_DIR, f"{name}.csv")
    print(f"=== {name}: {CONDITIONS[name]} ===")
    StudyConfig(CONDITIONS[name]).run()


if __name__ == "__main__":
    main()
