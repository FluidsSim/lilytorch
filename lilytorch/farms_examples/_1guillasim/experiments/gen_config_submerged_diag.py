"""Body-in-air speed-bias diagnostic (HP5b).

Rigorous A/B to isolate the two-phase surface speed bias: run the SAME eel,
gait, solver, and grid in ``SpawnMode.TRANSVERSE`` (root = slide-x, slide-y,
hinge-z -> surge/sway/yaw FREE, heave/roll/pitch LOCKED) at two spawn depths:

    DIAG_SPAWNZ=-0.0115  surface   -> dorsal surface in air  (expect ~0.22 m/s)
    DIAG_SPAWNZ=-0.07    submerged -> whole body in water     (expect ~0.15 m/s)

Because the buoyant (lighter-than-water) eel would otherwise float to the
surface, the locked vertical DOF is what holds it under for the submerged run.
Both runs lock heave/roll/pitch, so the ONLY difference is dorsal air-exposure;
a speed drop surface->submerged confirms the body-in-air drag-unloading
hypothesis (and submerged should match the single-phase underwater ~0.15).

Headless, no viewers; logs head x(t) + surge velocity via SpeedLogger.

Env knobs:
    DIAG_TAG     output label                 (default "submerged")
    DIAG_N       n_iterations                 (default 6000)
    DIAG_SPAWNZ  base spawn z (m)             (default -0.07, submerged)
    DIAG_OUT     CSV output directory         (default /data/.../_submerged_diag)

Run from this directory, e.g.:
    DIAG_TAG=submerged DIAG_SPAWNZ=-0.07   python gen_config_submerged_diag.py
    DIAG_TAG=surface   DIAG_SPAWNZ=-0.0115 python gen_config_submerged_diag.py
"""

import os

from farms_core.model.options import SpawnMode
from lilytorch.farms_examples.base_sim_config import BaseSimConfig
from gen_config_surface_pool import SimConfig as _PoolConfig, WATERLINE

_OUT = os.environ.get(
    "DIAG_OUT", "/data/andreaferrario/lilytorch/_submerged_diag")
os.makedirs(_OUT, exist_ok=True)
_TAG = os.environ.get("DIAG_TAG", "submerged")


class SimConfig(_PoolConfig):

    def __init__(self):
        super().__init__()
        self.headless     = True
        self.save         = False
        self.n_iterations = int(os.environ.get("DIAG_N", 6000))
        self.bdim_nt      = self.n_iterations + 1
        # The partition-∂H force readout needs the per-body SDFs maintained only
        # on the python force path (the kernel-streaming path keeps the union SDF).
        if (os.environ.get("DIAG_PHEAVI_PART", "0") == "1"
                or os.environ.get("DIAG_PYTHON", "0") == "1"):
            self.solver_method = "python"
        # DIAG_NO_ZPI=1: disable zero_pressure_inside so the single-phase force
        # band quadrature keeps the FULL band (interior half too), matching the
        # two-phase treatment -> isolates the band-interior quadrature difference.
        if os.environ.get("DIAG_NO_ZPI", "0") == "1":
            self.zero_pressure_inside = False

        # TRANSVERSE (slide-x, slide-y, hinge-z): frees surge/sway/YAW, locks
        # only heave/roll/pitch.  Yaw stays free so the anguilliform head-yaw
        # is natural (TRANSVERSE0 locked it -> stiff, unphysical kinematics);
        # the locked vertical still holds the buoyant body submerged.  The
        # submerged added-mass blow-up is cured by force_relaxation, NOT by
        # locking yaw.  DIAG_SPAWNMODE overrides.
        self.animats_pars[0]["spawn_mode"] = getattr(
            SpawnMode, os.environ.get("DIAG_SPAWNMODE", "TRANSVERSE"))
        # Spawn depth: submerged by default; surface control via DIAG_SPAWNZ.
        zspawn = float(os.environ.get("DIAG_SPAWNZ", -0.07))
        self.animats_pars[0]["pose"][2] = zspawn
        # Spawn farther from the inlet wall (x=xmin) so the fast straight-line
        # swimmer has runway to reach a steady plateau mid-domain before wall
        # proximity contaminates the speed.
        if "DIAG_SPAWNX" in os.environ:
            self.animats_pars[0]["pose"][0] = float(os.environ["DIAG_SPAWNX"])

        # Temporal under-relaxation of the fluid->body force feedback (FSI
        # stability).  At steady state the cycle-mean force is preserved, so
        # the swim speed is unbiased; it only damps the explicit-coupling
        # added-mass oscillation that blows up the fully-submerged case.
        # 1.0 = off.  Apply the SAME value to both A/B runs for fairness.
        self.force_relaxation = float(os.environ.get("DIAG_FRELAX", 1.0))

        # DIAG_COUPLING=aitken: implicit (strong) FSI coupling — cures the
        # added-mass blow-up the explicit path hits with FREE DOFs fully
        # submerged (force_relaxation is ignored when implicit).
        if os.environ.get("DIAG_COUPLING", "explicit") == "aitken":
            self.coupling = {"scheme": "implicit", "accelerator": "aitken",
                             "tol": 1e-3, "max_iter": 15}

        # DIAG_FREESLIP_TOP=1: open/free-slip top (z+, face index 5) instead of
        # the no-slip lid -> mimics the two-phase free surface for the
        # single-phase confinement test.  Frees tangential (u,v) AND normal (w)
        # at the top so displaced water can vent upward instead of recirculating
        # against a rigid sealed lid.
        if os.environ.get("DIAG_FREESLIP_TOP", "0") == "1":
            for bc in (self.bc_type_u, self.bc_type_v, self.bc_type_w):
                bc[5] = "N"

    def _bdim_extension(self, output_folder):
        # DIAG_SINGLEPHASE=1: plain single-phase FluidSolver (no two_phase
        # block, no gravity/free-surface) -> the body swims in infinite water.
        # With z locked (TRANSVERSE0) buoyancy is irrelevant, so this is the
        # clean "single-phase underwater" control for the same DOF-locked eel.
        if os.environ.get("DIAG_SINGLEPHASE", "0") == "1":
            # Skip the _PoolConfig two-phase/gravity injection: go straight to
            # the base builder -> the BDIMhandler instantiates a plain
            # FluidSolver (no solver.two_phase block).
            ext = BaseSimConfig._bdim_extension(self, output_folder)
            # DIAG_GRAVITY=1: add fluid gravity to the single-phase path (the
            # two-phase path has it; isolates whether gravity explains the gap).
            if os.environ.get("DIAG_GRAVITY", "0") == "1":
                ext["config"]["bdim_yaml"]["solver"]["gravity"] = [0, 0, -9.81]
            return ext
        ext = super()._bdim_extension(output_folder)
        # Self-contained: gen_config_surface_pool may have its two_phase/gravity
        # block commented out (production toggling).  Inject defaults here so this
        # diagnostic harness is INDEPENDENT of that file's current state.
        solver = ext["config"]["bdim_yaml"]["solver"]
        if "two_phase" not in solver:
            solver["gravity"] = [0, 0, -9.81]
            solver["two_phase"] = {
                "alpha_init"             : f"lambda X, Y, Z: (Z < {WATERLINE}).double()",
                "rho_water"              : 1000.0,
                "rho_air"                : 1.2,
                "nu_water"               : self.nu,
                "nu_air"                 : 1.5e-5,
                "alpha_exclude_body"     : True,
                "alpha_volume_compensate": True,
                "air_transparent_body"   : False,
            }
            # The pressure force readout is the SBP-clean union-∂H partition
            # ("deltaH"), inherited from the _PoolConfig solver block
            # (self.force_submethod); no per-step gauge anchor needed.
        # DIAG_RHO_AIR: override the two-phase air density.  Set = rho_water
        # (1000) to remove the density interface entirely -> two-phase code path
        # + gravity but UNIFORM density (isolates the interface from gravity).
        if "DIAG_RHO_AIR" in os.environ:
            tp = ext["config"]["bdim_yaml"]["solver"]["two_phase"]
            tp["rho_air"] = float(os.environ["DIAG_RHO_AIR"])
            if float(os.environ["DIAG_RHO_AIR"]) >= 1000.0:
                tp["nu_air"] = self.nu          # fully uniform fluid
        # DIAG_TP_NOGRAVITY=1: strip fluid gravity from the two-phase path ->
        # no hydrostatic field -> isolates whether the hydrostatic force-integral
        # leak (not the flow) is what separates two-phase from single-phase.
        if os.environ.get("DIAG_TP_NOGRAVITY", "0") == "1":
            ext["config"]["bdim_yaml"]["solver"].pop("gravity", None)
        # DIAG_PHEAVI_PART=1: run the union-∂H partition readout on the PYTHON
        # force path (parity check vs the default native "deltaH"); needs
        # solver_method='python' (set in __init__).
        if os.environ.get("DIAG_PHEAVI_PART", "0") == "1":
            tp = ext["config"]["bdim_yaml"]["solver"]["two_phase"]
            tp["partial_heaviside_forces"] = True
        return ext

    def extra_simulation_extensions(self, output_folder):
        # Strip the GUI/camera/flow viewers (need a display); log speed only.
        return [{
            "loader": "lilytorch.integration.speed_logger.SpeedLogger",
            "config": {
                "log_path": os.path.join(_OUT, f"speed_{_TAG}.csv"),
                "base_body_match": "link0",
            },
        }]


if __name__ == "__main__":
    SimConfig().run()
