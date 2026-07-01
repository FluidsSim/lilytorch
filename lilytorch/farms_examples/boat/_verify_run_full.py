import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from gen_configs import SimConfig


class VerifyFull(SimConfig):
    """Headless full-model run with pitch logging (no viewer / video).

    Tests the PYTHON path with two-phase ``consistent_momentum`` (conservative
    rho*u transport) — the interface stabiliser that the fused kernel lacks,
    targeting the 833x waterline blow-up.
    """

    def __init__(self):
        super().__init__()
        self.headless = True
        self.n_iterations = int(os.environ.get("VERIFY_NITER", "1500"))
        self.save_frames = False
        self.save = False
        # Env-driven knobs so several mitigations can be A/B'd without edits.
        # VERIFY_SOLVER unset -> inherit whatever gen_configs.py sets.
        solver = os.environ.get("VERIFY_SOLVER", "")
        if solver:
            self.solver_method = solver
        dt_scale = float(os.environ.get("VERIFY_DT_SCALE", "1.0"))
        if dt_scale != 1.0:
            self.timestep *= dt_scale
            self.n_iterations = int(self.n_iterations / dt_scale)
        self.bdim_dt = self.timestep
        self.bdim_nt = self.n_iterations
        sdf_override = os.environ.get("VERIFY_SDF", "")
        if sdf_override:
            self.animats_pars[0]["sdf_file"] = os.path.join(
                self.data_folder, sdf_override)
        tau = os.environ.get("VERIFY_TAU", "")
        if tau:
            self.animats_pars[0]["controller_config"]["tau"] = float(tau)
        damping = os.environ.get("VERIFY_DAMPING", "")
        if damping:
            self.joint_damping = float(damping)
        # Optional grid refinement test: VERIFY_DOMAIN="xmin,xmax,ymin,ymax,zmin,zmax"
        # + VERIFY_GRID="Nx,Ny,Nz" override the config domain/resolution together.
        domain = os.environ.get("VERIFY_DOMAIN", "")
        if domain:
            (self.xmin, self.xmax, self.ymin, self.ymax,
             self.zmin, self.zmax) = (float(v) for v in domain.split(","))
        grid = os.environ.get("VERIFY_GRID", "")
        if grid:
            self.Nx, self.Ny, self.Nz = (int(v) for v in grid.split(","))

    def _bdim_extension(self, output_folder):
        ext = super()._bdim_extension(output_folder)
        tp = ext["config"]["bdim_yaml"]["solver"]["two_phase"]
        if os.environ.get("VERIFY_NOCARVE", "0") == "1":
            tp["alpha_exclude_body"] = False
        if os.environ.get("VERIFY_CM", "0") == "1":
            tp["consistent_momentum"] = True
        rho_solid = os.environ.get("VERIFY_RHO_SOLID", "")
        if rho_solid:
            tp["rho_solid"] = float(rho_solid)
        return ext

    def extra_simulation_extensions(self, output_folder):
        return [{"loader": "_verify_zlog.BoatZLogger",
                 "config": {"log_path": os.path.join(output_folder, "boat_z_full.csv"),
                            "print_every": 25}}]


if __name__ == "__main__":
    print("=== VERIFY FULL boat headless (pitch log) ===", flush=True)
    VerifyFull().run()
