"""Overlap / force_relaxation instability study driver.

Reproduces the full-pool 1guilla coupled sim under a matrix of toggles to
test the hypothesis that *convexify-induced mesh overlap* is what destabilises
the explicit rigid-fluid coupling (forcing force_relaxation<1).

Each condition is launched as its own FARMS subprocess (via SimConfig.run());
per-step stability diagnostics are written to an absolute CSV by the injected
``_overlap_diag`` monkeypatch.

Usage:
    python study_overlap.py <condition_name>

Conditions are defined in CONDITIONS below.  Grid is downscaled by DOWNSCALE
from the production full-pool grid (same physical extents → overlap geometry
is preserved) to keep runtime/memory modest while a fair A/B is retained.
"""
import os
import sys

from gen_config_full_pool import SimConfig as FullPoolConfig

DOWNSCALE = int(os.environ.get("STUDY_DOWNSCALE", "2"))
N_ITERS = int(os.environ.get("STUDY_NITERS", "800"))
RESULTS_DIR = "/data/andreaferrario/lilytorch/_overlap_study"


# name -> dict of overrides
CONDITIONS = {
    # convexify, force_relaxation, zero_pressure_inside
    "cvx1_relax1_zpi1": dict(convexify=True,  force_relaxation=1.0, zero_pressure_inside=True),
    "cvx0_relax1_zpi1": dict(convexify=False, force_relaxation=1.0, zero_pressure_inside=True),
    "cvx1_relax03_zpi1": dict(convexify=True, force_relaxation=0.3, zero_pressure_inside=True),
    "cvx1_relax1_zpi0": dict(convexify=True,  force_relaxation=1.0, zero_pressure_inside=False),
    "cvx0_relax1_zpi0": dict(convexify=False, force_relaxation=1.0, zero_pressure_inside=False),
    # dt-halving probe: convexify+overlap, no relax, but half the coupling dt.
    # If this stabilises, the blow-up is the explicit added-mass/coupling-dt
    # mode that overlap aggravates (explains why pleurodeles' dt=5e-4 tolerates
    # the same overlap that destabilises 1guilla at dt=1e-3).
    "cvx1_relax1_zpi1_dt0005": dict(convexify=True, force_relaxation=1.0,
                                    zero_pressure_inside=True, timestep=0.0005),
    # ---- velocity-blend fix validation (Python solver path) ----
    "py_cvx1_noblend": dict(convexify=True, force_relaxation=1.0,
                            zero_pressure_inside=True, solver_method="python"),
    "py_cvx1_blend": dict(convexify=True, force_relaxation=1.0,
                          zero_pressure_inside=True, solver_method="python",
                          body_velocity_blend_eps_cells=2.0),
    # ---- KERNEL-mode blend validation (the real target) ----
    # Same config as cvx1_relax1_zpi1 (which exploded @ ~step 745) but with
    # the CUDA velocity blend on. Expect: stays bounded, no force_relaxation.
    "cvx1_blend2_kernel": dict(convexify=True, force_relaxation=1.0,
                               zero_pressure_inside=True,
                               body_velocity_blend_eps_cells=2.0),
}


class StudyConfig(FullPoolConfig):
    def __init__(self, overrides):
        super().__init__()

        # ── headless, no GL viewers / camera ──────────────────────────
        self.headless = True
        self.save = False
        self.save_frames = False

        # ── downscale grid (physical extents unchanged) ───────────────
        self.Nx = self.Nx // DOWNSCALE
        self.Ny = self.Ny // DOWNSCALE
        self.Nz = max(self.Nz // DOWNSCALE, 8)

        # ── short run ─────────────────────────────────────────────────
        self.n_iterations = N_ITERS
        self.bdim_nt = self.n_iterations + 1
        self.save_every = 100000  # effectively never

        # ── apply condition toggles ───────────────────────────────────
        base_dt = self.timestep
        for k, v in overrides.items():
            setattr(self, k, v)

        # Keep coupling dt in sync, and run the same PHYSICAL time when dt
        # is reduced (so a slower-growing instability still has a fair chance
        # to manifest within the run).
        if self.timestep != base_dt:
            self.bdim_dt = self.timestep
            self.n_iterations = int(round(N_ITERS * base_dt / self.timestep))
            self.bdim_nt = self.n_iterations + 1

    def extra_simulation_extensions(self, output_folder):
        # Strip FlowIsoGLViewer / CameraRecording / sky / light — pure physics.
        return []

    def _extra_run_patch(self):
        # Inject per-step stability diagnostics into the subprocess.
        return (
            "import lilytorch.integration._overlap_diag as _d;_d.install();"
        )


def main():
    name = sys.argv[1]
    overrides = CONDITIONS[name]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    diag_csv = os.path.join(RESULTS_DIR, f"{name}.csv")
    os.environ["LILY_DIAG_CSV"] = diag_csv
    print(f"=== condition {name}: {overrides} → {diag_csv} ===")
    StudyConfig(overrides).run()


if __name__ == "__main__":
    main()
