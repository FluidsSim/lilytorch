"""Cross-model overlap test on pleurodeles (normally convexify=False, stable).

Toggles convexify ON/OFF with NO force_relaxation to test whether mesh overlap
destabilises the coupling for a different model too (i.e. the instability is a
property of overlap, not of the 1guilla model).
"""
import os
import sys

from gen_configs_swim_3d import SimConfig as PleuroConfig

DOWNSCALE = int(os.environ.get("STUDY_DOWNSCALE", "2"))
N_ITERS = int(os.environ.get("STUDY_NITERS", "800"))
RESULTS_DIR = "/data/andreaferrario/lilytorch/_overlap_study"

CONDITIONS = {
    # recompute the SDF (compute_sdf=True) so convexify actually takes effect;
    # redirect the cache subfolder so the user's cached interp_data_3d is untouched.
    "pleuro_rc_cvx0": dict(convexify=False, force_relaxation=1.0,
                           compute_sdf=True, interp_data_subfolder="interp_data_3d_study_cvx0"),
    "pleuro_rc_cvx1": dict(convexify=True,  force_relaxation=1.0,
                           compute_sdf=True, interp_data_subfolder="interp_data_3d_study_cvx1"),
}


class StudyConfig(PleuroConfig):
    def __init__(self, overrides):
        super().__init__()
        self.headless = True
        self.save = False
        self.save_frames = False
        self.Nx = self.Nx // DOWNSCALE
        self.Ny = self.Ny // DOWNSCALE
        self.Nz = max(self.Nz // DOWNSCALE, 8)
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
    overrides = CONDITIONS[name]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["LILY_DIAG_CSV"] = os.path.join(RESULTS_DIR, f"{name}.csv")
    print(f"=== {name}: {overrides} ===")
    StudyConfig(overrides).run()


if __name__ == "__main__":
    main()
