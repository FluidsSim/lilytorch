"""Live arbitration of the force readouts on the real zebrafish swim.

§10 measured the readouts on a FROZEN field: the eulerian reports ~4x the
viscous drag the lagrangian does, hence ~0.41x the net thrust, hence a slower
swim.  That says which readouts DISAGREE, not which is RIGHT.  The arbiter for
"right" is the energy budget: dE_k/dt = P_act - dissipation.  A readout that
over-reads drag steals momentum from the body that the fluid never receives, so
its budget should close worse.

Runs the production ``gen_configs_pd_3d_slow_fast`` config headless (same grid,
gait, physics; viewers stripped) into a {stack}/{case} layout, so
``verify_energy_balance.py --stack ... --case ...`` can read it directly.

  ZFISH_CASE=lagr_off0 ZFISH_FORCE_METHOD=lagrangian ZFISH_LAGR_OFFSET=0.0 \
      python -m lilytorch.validation.force_readout_oracle.gen_zfish_readout_arbitration

Env: ZFISH_CASE, ZFISH_FORCE_METHOD, ZFISH_LAGR_OFFSET (metres, or "" for the
solver default), ZFISH_DELTA_ORDER, ZFISH_NITER.
"""
from __future__ import annotations

import os

from lilytorch.examples.zebrafish_ki_project.gen_configs_pd_3d_slow_fast import (
    SimConfig as _ProdConfig,
)

STACK = "zfish_readout_arbitration"
CASE = os.environ.get("ZFISH_CASE", "lagr_off0")
FORCE_METHOD = os.environ.get("ZFISH_FORCE_METHOD", "lagrangian")
LAGR_OFFSET = os.environ.get("ZFISH_LAGR_OFFSET", "")
DELTA_ORDER = int(os.environ.get("ZFISH_DELTA_ORDER", 1))
NITER = int(os.environ.get("ZFISH_NITER", 1201))       # dt=5e-4 -> 0.6 s


class SimConfig(_ProdConfig):

    def __init__(self):
        super().__init__()
        self.stack_folder = os.path.join(self.stack_folder, STACK, CASE)
        self.headless = True
        # save=True ONLY to make the solver create ``save_path`` — that is what
        # gates diagnostics.h5 / drags.h5 at end_episode (solver.py:641).  It
        # does NOT dump fields here: nothing in the FARMS-coupled path calls
        # ``save_results`` (the step loop is MuJoCo-driven), so no field HDF5 is
        # written.  Without this, verify_energy_balance has no diagnostics.h5.
        self.save = True
        self.save_frames = False
        self.save_drags = True         # per-link force records -> drags.h5
        self.force_method = FORCE_METHOD
        self.force_delta_order = DELTA_ORDER
        if LAGR_OFFSET != "":
            self.lagrangian_sample_offset = float(LAGR_OFFSET)
        self.n_iterations = NITER
        self.bdim_nt = self.n_iterations + 1
        # verify_energy_balance differentiates E_k on the diagnostics grid; the
        # production cadence of 100 leaves too few samples over 0.3 s.
        self.diagnostics_every = 10

    def extra_simulation_extensions(self, output_folder):
        return []                      # no viewers/recorders: need a display


if __name__ == "__main__":
    print(f"[arb] case={CASE} force_method={FORCE_METHOD} "
          f"lagr_offset={LAGR_OFFSET or 'default(0.0)'} "
          f"delta_order={DELTA_ORDER} n_iter={NITER}")
    SimConfig().single_run()
