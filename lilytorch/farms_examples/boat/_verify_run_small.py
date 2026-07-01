import os
import sys

# Put THIS directory on sys.path AND on the child sim's PYTHONPATH so the
# generated simulation (which runs from a timestamped output folder) can import
# the local ``_verify_zlog`` extension loader.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from gen_configs_small import SmallSimConfig


class VerifySmall(SmallSimConfig):
    def __init__(self):
        super().__init__()
        # VERIFY_VIEWER=1 opens the MuJoCo viewer with the live flow overlay;
        # default stays headless so the automated stability tests keep working.
        self.headless = os.environ.get("VERIFY_VIEWER", "0") != "1"
        self.n_iterations = int(os.environ.get("VERIFY_NITER", "1500"))
        self.bdim_nt = self.n_iterations
        self.save_frames = False
        self.save = False
        solver = os.environ.get("VERIFY_SOLVER", "")
        if solver:
            self.solver_method = solver
        if os.environ.get("VERIFY_SPONGE", "0") == "1":
            self.sponge = {
                "width":    float(os.environ.get("VERIFY_SPONGE_W", "0.02")),
                "strength": float(os.environ.get("VERIFY_SPONGE_S", "500.0")),
                "axes":     ["x", "y", "z"],
            }

    def _bdim_extension(self, output_folder):
        ext = super()._bdim_extension(output_folder)
        tp = ext["config"]["bdim_yaml"]["solver"]["two_phase"]
        if os.environ.get("VERIFY_CM", "0") == "1":
            tp["consistent_momentum"] = True
        rho_solid = os.environ.get("VERIFY_RHO_SOLID", "")
        if rho_solid:
            tp["rho_solid"] = float(rho_solid)
        if os.environ.get("VERIFY_MU0FREE", "0") == "1":
            tp.pop("rho_solid", None)            # mutually exclusive
            tp["mu0_free_coeff"] = True
        atb = os.environ.get("VERIFY_ATB", "")
        if atb != "":
            tp["air_transparent_body"] = (atb == "1")
        return ext

    def extra_simulation_extensions(self, output_folder):
        exts = [{"loader": "_verify_zlog.BoatZLogger",
                 "config": {"log_path": os.path.join(output_folder, "boat_z.csv"),
                            "print_every": 50}}]
        if os.environ.get("VERIFY_VIEWER", "0") == "1":
            # Live overlay in the MuJoCo viewer: air/water interface (VOF alpha
            # iso 0.5, translucent blue) + vorticity-magnitude shell (the wake).
            exts.append({
                "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
                "config": {
                    "update_every": 1,
                    "max_vertices": 800000,
                    "crop_boundary": 0,
                    "debug_force_visible": False,
                    "fields": [
                        {"field": "interface", "iso_value": 0.5, "alpha": 0.45,
                         "color": "#3399FF", "smooth_sigma": 0, "exclude_body": False},
                        {"field": "omega_mag", "iso_fraction": 0.8, "alpha": 0.3,
                         "color": "#FF8C1A", "smooth_sigma": 1.5, "exclude_body": True},
                    ],
                },
            })
        return exts


if __name__ == "__main__":
    print("=== VERIFY SMALL (S=0.1) boat ===", flush=True)
    VerifySmall().run()
