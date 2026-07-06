"""2-D kernel-mode validation of the velocity-blend fix (1guilla pinned).

convexify=True + lagrangian forces, kernel mode, blend OFF vs ON.
"""
import os
import sys

from gen_configs_one_pinned_2d import SimConfig as Pinned2D

DOWNSCALE = int(os.environ.get("STUDY_DOWNSCALE", "2"))
N_ITERS = int(os.environ.get("STUDY_NITERS", "800"))
RESULTS_DIR = "/data/andreaferrario/lilytorch/_overlap_study"

CONDITIONS = {
    "p2d_noblend": dict(convexify=True, force_method="lagrangian",
                        force_relaxation=1.0, zero_pressure_inside=True,
                        body_velocity_blend_eps_cells=None),
    "p2d_blend":   dict(convexify=True, force_method="lagrangian",
                        force_relaxation=1.0, zero_pressure_inside=True,
                        body_velocity_blend_eps_cells=2.0),
    # native-settings smoke: confirm the 2-D blend kernel path runs cleanly
    "p2d_native_noblend": dict(body_velocity_blend_eps_cells=None),
    "p2d_native_blend":   dict(body_velocity_blend_eps_cells=2.0),
}


class StudyConfig(Pinned2D):
    def __init__(self, overrides):
        super().__init__()
        self.headless = True
        self.save = False
        self.save_frames = False
        self.Nx = self.Nx // DOWNSCALE
        self.Ny = self.Ny // DOWNSCALE
        self.n_iterations = N_ITERS
        self.bdim_nt = self.n_iterations + 1
        self.save_every = 100000
        for k, v in overrides.items():
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
