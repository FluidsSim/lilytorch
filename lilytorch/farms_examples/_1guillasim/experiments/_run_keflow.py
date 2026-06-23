"""Diagnostic: does GRAVITY change the FLOW or only the force READOUT?

Uniform-density (rho_air=rho_water) submerged eel with the root PINNED
(SpawnMode.FIXED) so the body kinematics are IDENTICAL with/without gravity
(prescribed gait, no force feedback moving the root).  The ONLY difference
between the two runs is the fluid gravity body force.  We log the fluid kinetic
energy every step; if KE(grav) == KE(nograv) the velocity field is unchanged ->
gravity is a pure-gradient READOUT effect (cheap fix: hydrostatic-field
subtraction).  If KE diverges -> gravity perturbs the FLOW (the variable-coeff
projection near the body) -> no readout fix works.

  KEFLOW_NOGRAV=0 KEFLOW_TAG=ke_grav    python _run_keflow.py
  KEFLOW_NOGRAV=1 KEFLOW_TAG=ke_nograv  python _run_keflow.py
"""
import os

from farms_core.model.options import SpawnMode
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from gen_config_surface_pool import SimConfig as _PoolConfig

_OUT = "/data/andreaferrario/lilytorch/_submerged_diag"
os.makedirs(_OUT, exist_ok=True)


def _write_flat_gait(path, t_end):
    """Write a flat (zero-amplitude) gait CSV in the ms004 log format so the
    KinematicsController holds every joint straight -> the body is RIGID (no
    undulation).  Two metadata header rows + the long column header + data rows
    (40 cols; only cols 0,1=time and 2:10=8 goal angles are read; all zero)."""
    cols = (
        "absReceived_ts[s],absReceived_ts[us]," +
        ",".join(f"{i}GoalPosition[rad]" for i in range(8)) + "," +
        ",".join(f"{i}FbckPosition[rad]" for i in range(8)) + "," +
        ",".join(f"{i}FbckCurrent[mA]" for i in range(8)) + "," +
        ",".join(f"{i}FbckVoltage[volts]" for i in range(8)) +
        ",ax,ay,az,Wx,Wy,Wz"
    )
    n = max(8, int(t_end) + 4)
    with open(path, "w") as fh:
        fh.write("NumMotors,positionAmplitude[deg],Lambda,Frequency[Hz],\n")
        fh.write("8,0,12,0.25\n")
        fh.write(cols + "\n")
        zeros = ",".join("0" for _ in range(38))   # 40 cols - 2 time cols
        for i in range(n):
            fh.write(f"{1000000000 + i},0,{zeros}\n")  # 1 s spacing, all angles 0


class SimConfig(_PoolConfig):

    def __init__(self):
        super().__init__()
        self.headless          = True
        self.save              = False
        self.n_iterations      = int(os.environ.get("KEFLOW_N", 800))
        self.bdim_nt           = self.n_iterations + 1
        self.diagnostics_every = 1                      # log KE every step
        # force_scaling=0 -> the fluid exerts NO force on the body, so the body
        # kinematics depend only on MuJoCo (gait + its own gravity), i.e. are
        # IDENTICAL whether or not the FLUID has gravity.  TRANSVERSE (heave
        # locked at z=-0.07, surge/sway/yaw free) keeps the body translating so
        # the fluid stays stable (a fully-pinned body blows up the CFL).
        self.force_scaling = 0.0
        self.animats_pars[0]["spawn_mode"] = SpawnMode.TRANSVERSE
        # KEFLOW_Z: heave-locked spawn depth.  Default -0.07 = fully submerged
        # (all prior quantitative tests).  Surface case = -0.0115 (the eel's real
        # swimming draft, body straddling the 833:1 interface) -- §6 of the
        # hydrostatic_gravity handoff: the hydrostatic kink sits right at the body.
        self.animats_pars[0]["pose"][2]    = float(os.environ.get("KEFLOW_Z", -0.07))
        # ── STEADY-GLIDE / STATIC modes (decisive unsteady-vs-static test) ──
        # KEFLOW_GLIDE=1: zero the gait (rigid straight body) and give the body
        #   a constant surge velocity (KEFLOW_VX, default -0.13 m/s = forward at
        #   the real swim speed) so it TRANSLATES through the fluid WITHOUT
        #   undulating.  With force_scaling=0 + TRANSVERSE (heave locked) and no
        #   actuation the body coasts at constant velocity -> exercises the
        #   convective flux but NOT the unsteady undulation term.
        # KEFLOW_STATIC=1: zero the gait, NO surge velocity -> the body sits
        #   still (pure static-pressure readout reference).
        # The ∂H partition readout needs the per-body SDFs maintained only on the
        # python force path (the kernel-streaming path keeps the union SDF only).
        if any(os.environ.get(k, "0") == "1" for k in
               ("KEFLOW_PHEAVI", "KEFLOW_PYTHON")):
            self.solver_method = "python"
        if (os.environ.get("KEFLOW_GLIDE", "0") == "1"
                or os.environ.get("KEFLOW_STATIC", "0") == "1"):
            flat = os.path.join(_OUT, "flat_gait.csv")
            _write_flat_gait(flat, self.n_iterations * self.timestep + 8)
            self.animats_pars[0]["control_pars"]["file_path"] = flat
        # KEFLOW_PTOL: tighten the Poisson solve to test whether the spurious
        # gravity flow is an under-resolved hydrostatic gradient (loose default
        # tol=1e-4 on the anisotropic grid).
        if "KEFLOW_PTOL" in os.environ:
            self.poisson_tol             = float(os.environ["KEFLOW_PTOL"])
            self.poisson_method          = os.environ.get("KEFLOW_PMETHOD", "mgcg")
            self.poisson_max_cycles      = int(os.environ.get("KEFLOW_PCYC", 300))
            self.poisson_max_mgcg_cycles = int(os.environ.get("KEFLOW_PCYC", 300))

    def _bdim_extension(self, output_folder):
        # KEFLOW_SINGLEPHASE=1: plain single-phase FluidSolver (no two_phase/
        # alpha/air handling) but WITH fluid gravity -> isolates whether the
        # spurious gravity-driven flow is the BDIM band (present here too) or the
        # two-phase alpha/air machinery (absent here).
        if os.environ.get("KEFLOW_SINGLEPHASE", "0") == "1":
            ext = BaseSimConfig._bdim_extension(self, output_folder)
            if os.environ.get("KEFLOW_NOGRAV", "0") != "1":
                ext["config"]["bdim_yaml"]["solver"]["gravity"] = [0, 0, -9.81]
            ext["loader"] = (
                "lilytorch.farms_examples._1guillasim.experiments._ke_ext."
                "KEFluidExtension")
            return ext
        ext = super()._bdim_extension(output_folder)
        solver = ext["config"]["bdim_yaml"]["solver"]
        tp = solver["two_phase"]
        # Uniform density by default; KEFLOW_REALAIR=1 keeps the real 833:1 jump
        # (rho_air=1.2) -> the stiff variable-coeff Poisson the production run has.
        if os.environ.get("KEFLOW_REALAIR", "0") != "1":
            tp["rho_air"] = 1000.0
            tp["nu_air"]  = self.nu
        elif "KEFLOW_RHOAIR" in os.environ:
            # Override the air density to probe the density-ratio dependence of
            # the gravity-on interface flow (parasitic-current hypothesis).
            tp["rho_air"] = float(os.environ["KEFLOW_RHOAIR"])
        # KEFLOW_PHEAVI=1: partial-Heaviside (∂H) pressure-force readout
        # (union-∂H force density split to links by a softmin partition of unity;
        # the seam-free, SBP-clean weight).  Native-kernel equivalent:
        # solver.force_submethod = "deltaH".
        tp["partial_heaviside_forces"] = os.environ.get("KEFLOW_PHEAVI", "0") == "1"
        # Toggle the fluid gravity body force.
        if os.environ.get("KEFLOW_NOGRAV", "0") == "1":
            solver.pop("gravity", None)
        # Swap in the KE-dumping FluidExtension (this module is importable as
        # _run_keflow when run from the experiments dir).
        ext["loader"] = (
            "lilytorch.farms_examples._1guillasim.experiments._ke_ext."
            "KEFluidExtension")
        return ext

    def extra_simulation_extensions(self, output_folder):
        return []                                       # headless, no viewers


if __name__ == "__main__":
    SimConfig().run()
