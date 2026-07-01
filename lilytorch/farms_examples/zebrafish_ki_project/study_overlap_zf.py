"""Validate the velocity-blend fix on the zebrafish_ki model (the user's case).

convexify=True (overlapping links) in kernel mode, blend OFF (expect explode)
vs blend ON (expect stable). compute_sdf recomputes the SDF so convexify takes
effect; interp cache redirected so the user's data is untouched.
"""
import os
import sys

from gen_configs_pd_3d_slow_fast import SimConfig as ZFConfig

DOWNSCALE = int(os.environ.get("STUDY_DOWNSCALE", "2"))
N_ITERS = int(os.environ.get("STUDY_NITERS", "800"))
RESULTS_DIR = "/data/andreaferrario/lilytorch/_overlap_study"

CONDITIONS = {
    "zf_cvx1_noblend": dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=None),
    "zf_cvx1_blend":   dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=2.0),
    "zf_cvx0_noblend": dict(convexify=False, force_relaxation=1.0,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf0",
                            body_velocity_blend_eps_cells=None),
    "zf_cvx1_blend4":  dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=4.0),
    "zf_cvx1_blend8":  dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=8.0),
    "zf_cvx1_blend_relax03": dict(convexify=True, force_relaxation=0.3,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=2.0),
    "zf_cvx1_relax03_noblend": dict(convexify=True, force_relaxation=0.3,
                            compute_sdf=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=None),
    # bdim_mu0_projection=False → non-degenerate Poisson coeff (dt/rho) so the
    # projection can remove the seam divergence in the body interior/overlap.
    "zf_cvx1_mu0p0_noblend": dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True, bdim_mu0_projection=False,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=None),
    "zf_cvx1_mu0p0_blend": dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True, bdim_mu0_projection=False,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=2.0),
    # Maertens-Weymouth RHS correction WITH the correct mu0-weighted operator.
    "zf_cvx1_mwcorr": dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True, bdim_body_div_correction=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=None),
    # FULL BDIM per user: non-degenerate operator (mu0_proj=False) + M&W RHS term
    "zf_cvx1_mu0p0_mwcorr": dict(convexify=True, force_relaxation=1.0,
                            compute_sdf=True, bdim_mu0_projection=False,
                            bdim_body_div_correction=True,
                            interp_data_subfolder="interp_data_study_zf",
                            body_velocity_blend_eps_cells=None),
    # faithful M&W (mu0_proj=True + correction) + raised degenerate-freeze tol
    "zf_jcap8": dict(convexify=True, force_relaxation=1.0, compute_sdf=True,
                     bdim_body_div_correction=True, poisson_jcap_tol=1e-8,
                     interp_data_subfolder="interp_data_study_zf",
                     body_velocity_blend_eps_cells=None),
    "zf_jcap7": dict(convexify=True, force_relaxation=1.0, compute_sdf=True,
                     bdim_body_div_correction=True, poisson_jcap_tol=1e-7,
                     interp_data_subfolder="interp_data_study_zf",
                     body_velocity_blend_eps_cells=None),
    # faithful M&W + HUGE solver budget — tests under-convergence hypothesis
    "zf_bigsolve": dict(convexify=True, force_relaxation=1.0, compute_sdf=True,
                     bdim_body_div_correction=True,
                     poisson_tol=1e-9, poisson_max_mgcg_cycles=400,
                     poisson_max_cycles=80,
                     interp_data_subfolder="interp_data_study_zf",
                     body_velocity_blend_eps_cells=None),
}


class StudyConfig(ZFConfig):
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
        return ("import lilytorch.integration._overlap_diag as _d;_d.install();"
                "import lilytorch.integration._region_diag as _rd;_rd.install();")


def main():
    name = sys.argv[1]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["LILY_DIAG_CSV"] = os.path.join(RESULTS_DIR, f"{name}.csv")
    os.environ["LILY_REGION_CSV"] = os.path.join(RESULTS_DIR, f"{name}_region.csv")
    print(f"=== {name}: {CONDITIONS[name]} ===")
    StudyConfig(CONDITIONS[name]).run()


if __name__ == "__main__":
    main()
