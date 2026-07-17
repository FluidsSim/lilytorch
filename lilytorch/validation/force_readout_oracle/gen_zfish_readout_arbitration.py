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

STACK = os.environ.get("ZFISH_STACK", "zfish_readout_arbitration")
CASE = os.environ.get("ZFISH_CASE", "lagr_off0")
FORCE_METHOD = os.environ.get("ZFISH_FORCE_METHOD", "lagrangian")
LAGR_OFFSET = os.environ.get("ZFISH_LAGR_OFFSET", "")
DELTA_ORDER = int(os.environ.get("ZFISH_DELTA_ORDER", 1))
NITER = int(os.environ.get("ZFISH_NITER", 1201))       # dt=5e-4 -> 0.6 s

# Grid refinement for convergence studies.  REFINE=2 halves h in all three
# directions with the DOMAIN HELD FIXED, and halves dt to keep CFL — so it is a
# pure resolution change.  (The 1024x256x128 block commented out in the
# production config also halves the y extent, which would confound resolution
# with a narrower tank.)  n_iterations scales so the physical time is unchanged.
# Float: 2 halves h, 0.5 doubles it — a 3-point sequence lets the convergence
# ORDER be measured rather than assumed.
REFINE = float(os.environ.get("ZFISH_REFINE", 1))
# Offset in CELLS, so the recommendation "off ~= h" tracks the grid: an offset
# fixed in metres would mean a different thing on a finer grid.
LAGR_OFFSET_CELLS = os.environ.get("ZFISH_LAGR_OFFSET_CELLS", "")


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
        self.n_iterations = NITER

        if REFINE != 1:
            self.Nx = int(round(self.Nx * REFINE))
            self.Ny = int(round(self.Ny * REFINE))
            self.Nz = int(round(self.Nz * REFINE))
            self.timestep = self.timestep / REFINE
            self.bdim_dt = self.timestep
            self.n_iterations = int(round((self.n_iterations - 1) * REFINE)) + 1

        h = (self.xmax - self.xmin) / self.Nx
        if LAGR_OFFSET_CELLS != "":
            self.lagrangian_sample_offset = float(LAGR_OFFSET_CELLS) * h
        elif LAGR_OFFSET != "":
            self.lagrangian_sample_offset = float(LAGR_OFFSET)
        self.bdim_nt = self.n_iterations + 1
        # verify_energy_balance differentiates E_k on the diagnostics grid; the
        # production cadence of 100 leaves too few samples over 0.3 s.
        self.diagnostics_every = 10

    def extra_simulation_extensions(self, output_folder):
        return []                      # no viewers/recorders: need a display


if __name__ == "__main__":
    c = SimConfig()
    h = (c.xmax - c.xmin) / c.Nx
    print(f"[arb] case={CASE} force_method={FORCE_METHOD} "
          f"offset={c.lagrangian_sample_offset} delta_order={DELTA_ORDER}")
    print(f"[arb] grid {c.Nx}x{c.Ny}x{c.Nz} = {c.Nx*c.Ny*c.Nz/1e6:.1f} M cells  "
          f"h={h*1e3:.4f} mm  dt={c.timestep}  n_iter={c.n_iterations}  "
          f"t_end={c.n_iterations*c.timestep:.3f} s  "
          f"R/h={0.559e-3/h:.2f} (thin direction)")
    c.single_run()
