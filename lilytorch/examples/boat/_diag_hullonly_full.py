#!/usr/bin/env python3
"""Seam-isolation diagnostic for the tau-independent two-phase blow-up (B).

Runs the EXACT exploding configuration (same grid h, kernel mode, 833:1 air,
tau=0, two-phase, coupling) but against a FULL-SCALE *hull-only* model
(``toy_boat_hullonly_full.sdf`` — keel/rudder/propeller deleted).  The fluid
therefore sees a SINGLE body, so there are NO multi-body convexify SDF seams.

Discriminator:
  * STABLE  -> the multi-body SDF seams are the blow-up source (suspect ii).
  * EXPLODES -> seams exonerated; look at the two-phase scheme / waterline /
                coupling (suspects i, iii, iv).

Watch the FlowDiagnostics output (energy / max-divergence / CFL) and the
viewer.  The full sim blows up around iter ~90, so a few hundred iters suffice.

Run::
    cd lilytorch/examples/boat
    python3 _diag_hullonly_full.py
"""
import os
import xml.etree.ElementTree as ET

from gen_configs import SimConfig


def _scale_sdf_mass(src_sdf, factor, dst_sdf):
    """Write a copy of *src_sdf* with every link mass + inertia x *factor*.

    Pure body-mass change: geometry, pose, SDF and the whole coupling are
    untouched.  Added-mass instability is stable iff m_body > m_added, so a
    large factor MUST stabilise the EXPLICIT run if (and only if) the blow-up
    is the added-mass coupling -- nothing else is cured by mass alone.
    """
    tree = ET.parse(src_sdf)
    for inertial in tree.iter("inertial"):
        m = inertial.find("mass")
        if m is not None:
            m.text = repr(float(m.text) * factor)
        inertia = inertial.find("inertia")
        if inertia is not None:
            for comp in inertia:
                inertia_val = comp.text
                if inertia_val is not None and inertia_val.strip():
                    comp.text = repr(float(inertia_val) * factor)
    tree.write(dst_sdf)
    return dst_sdf


# ── Env-var knobs for the isolation tests ──────────────────────────────────
#   RHO_AIR=100   Test i  : density ratio 833->10 (kernel)        [scheme]
#   TP_FIX=1      Test iii: python + consistent_momentum           [scheme cure]
#   POISSON_HARD=1  crank V-cycles + tol -> tests Poisson under-convergence
#                   stable -> 4 V-cycles don't converge at h=0.05 (residual div)
#   SUBMERGED=1   alpha_init = all water (waterline above the domain) -> NO free
#                 surface / triple line, hull fully submerged in uniform density.
#                   stable   -> the body-waterline triple line is the blow-up
#                   explodes -> it's the BDIM body coupling itself (density-uniform)
# (A bare body-free tank cannot be done by spawning the body out of the grid:
#  the kernel streaming path computes its work region from the body AABB and
#  crashes on a fully-outside body.  Use a standalone two-phase script instead.)
#   IMPLICIT=1    switch the FSI coupling explicit -> implicit (added-mass cure;
#                 the articulated-joint caveat does not apply to the jointless hull)
#   FRELAX=<f>    explicit coupling with force_relaxation=<f> low-pass (e.g. 0.1)
#   MASSX=<f>     ** the added-mass PROOF **: scale body mass+inertia x<f> with
#                 EXPLICIT coupling, no relaxation, nothing else changed.
#                 heavier -> later/no blow-up (monotonic) PROVES added mass;
#                 unchanged -> it is NOT added mass.
_RHO_AIR      = float(os.environ.get("RHO_AIR", "1.2"))
_TP_FIX       = os.environ.get("TP_FIX", "0") == "1"
_POISSON_HARD = os.environ.get("POISSON_HARD", "0") == "1"
_SUBMERGED    = os.environ.get("SUBMERGED", "0") == "1"
_IMPLICIT     = os.environ.get("IMPLICIT", "0") == "1"
_FRELAX       = os.environ.get("FRELAX", "")
_FULL         = os.environ.get("FULL", "0") == "1"   # keep full boat (keel/rudder/prop)
_MASSX        = os.environ.get("MASSX", "")
#   NOCONVEX=1    build the fluid SDF from the REAL (concave) mesh, not the 4.11x
#                 convex hull -> ~4x LESS displaced/added mass at the SAME 1400 kg.
#                 stable -> the convex-hull volume inflation IS what drives the
#                           added-mass ratio above 1 (user's hypothesis confirmed).
_NOCONVEX     = os.environ.get("NOCONVEX", "0") == "1"


class HullOnlyDiag(SimConfig):
    def __init__(self):
        super().__init__()

        # ── Body model: hull-only (default) or the full boat (FULL=1) ────
        if _FULL:
            boat_sdf = os.path.join(self.data_folder, 'toy_boat.sdf')
            model_name = "toy_boat"
        else:
            boat_sdf = os.path.join(self.data_folder, 'toy_boat_hullonly_full.sdf')
            model_name = "toy_boat_hullonly_full"
            # No revolute propeller joint -> drop the propeller controller so
            # the launcher does not try to actuate a joint that is gone.
            self.animats_pars[0].pop("controller_config", None)

        # ── MASSX (the added-mass PROOF): scale body mass+inertia, explicit
        #    coupling untouched.  Write the scaled SDF next to the original.
        if _MASSX:
            scaled = os.path.join(self.data_folder, f'_massx_{model_name}.sdf')
            _scale_sdf_mass(boat_sdf, float(_MASSX), scaled)
            boat_sdf = scaled

        self.animats_pars[0]["sdf_file"]   = boat_sdf
        self.animats_pars[0]["model_name"] = model_name

        # ── Test iii (candidate FIX): leave kernel mode for the python path,
        #    which is the only place the high-density-ratio cure lives.
        if _TP_FIX:
            self.solver_method = "python"

        # ── NOCONVEX: real-volume fluid SDF (tests the convex-inflation cause).
        if _NOCONVEX:
            self.convexify = False

        # ── Added-mass cures (the leading hypothesis) ────────────────────
        if _IMPLICIT:
            self.coupling = dict(self.coupling, scheme="implicit")
        if _FRELAX:
            self.force_relaxation = float(_FRELAX)

        # ── POISSON_HARD: drive the projection to tight convergence to test
        #    whether 4 V-cycles simply fail to converge at h=0.05.
        if _POISSON_HARD:
            self.poisson_tol             = 1.0e-9
            self.poisson_max_cycles      = 40
            self.poisson_max_mgcg_cycles = 200
            self.poisson_precond_vcycles = 2
            self.poisson_nsmoothing      = 8

        # ── Keep EVERYTHING else identical to the exploding run, but make the
        #    blow-up easy to see: short run, frequent diagnostics, no frames.
        self.n_iterations    = int(os.environ.get("NITER", "400"))
        self.bdim_nt         = self.n_iterations
        self.diagnostics_every = 5
        self.save_frames     = False
        self.save            = False
        # Headless: the GL viewer / camera need a display; skip them so the
        # run is driven purely by the FlowDiagnostics output.
        self.headless        = True

    def extra_simulation_extensions(self, output_folder):
        # Drop the FlowIsoGLViewer + StreamingCameraRecording (GL/display).
        return []

    def _bdim_extension(self, output_folder):
        """Inject the two-phase-scheme test overrides into the VOF block."""
        bdim_ext = super()._bdim_extension(output_folder)
        tp = bdim_ext["config"]["bdim_yaml"]["solver"]["two_phase"]
        # Test i: lower the density ratio.
        tp["rho_air"] = _RHO_AIR
        # Test iii: enable the conservative momentum transport (python only).
        if _TP_FIX:
            tp["consistent_momentum"] = True
        # SUBMERGED: fill the whole domain with water -> no air, no interface.
        if _SUBMERGED:
            tp["alpha_init"] = "lambda X, Y, Z: torch.ones_like(Z).double()"
        return bdim_ext


if __name__ == "__main__":
    print("=== SEAM-ISOLATION DIAG: full-scale HULL-ONLY, tau=0 ===", flush=True)
    print("    stable -> seams are the cause;  explodes -> not the seams",
          flush=True)
    HullOnlyDiag().run()
