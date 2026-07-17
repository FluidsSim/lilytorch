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
      python -m lilytorch.examples.force_benchmarks.gen_zfish_readout_arbitration

Env: ZFISH_CASE, ZFISH_FORCE_METHOD, ZFISH_DELTA_ORDER, ZFISH_NITER.

Sampling location — two families, the split one wins if both are set:

* ``ZFISH_OFF_P_CELLS`` / ``ZFISH_OFF_F_CELLS`` (cells) → the per-channel
  ``sample_offset_{pressure,friction}_cells``, applied to BOTH readouts.  This
  is the only way to express Verma et al. 2017's ``(0, 2h)`` — p read at the
  true surface, sigma on a surface lifted 2h — and hence the only way to pin
  the two readouts to identical sampling locations.  Either may be set alone.
* ``ZFISH_LAGR_OFFSET_CELLS`` / ``ZFISH_LAGR_OFFSET`` (cells / metres) → the
  LEGACY single ``lagrangian_sample_offset``, which moves p and sigma together
  and touches the lagrangian only.  Kept so the §10 runs stay reproducible.

All unset → every readout keeps its legacy default (eulerian ``(0, eps)``,
lagrangian ``(0, 0)``).
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
# Per-channel offsets in CELLS, applied to BOTH readouts.  These supersede the
# legacy knob above: they can express (0, 2h) = Verma et al. 2017, which the
# single knob cannot, and they eliminate sampling location as a free variable
# in any eulerian-vs-lagrangian comparison.
OFF_P_CELLS = os.environ.get("ZFISH_OFF_P_CELLS", "")
OFF_F_CELLS = os.environ.get("ZFISH_OFF_F_CELLS", "")


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
        # Per-channel knobs last: they apply to both readouts, so they must win
        # over the legacy single knob if a run sets both.
        if OFF_P_CELLS != "":
            self.sample_offset_pressure_cells = float(OFF_P_CELLS)
        if OFF_F_CELLS != "":
            self.sample_offset_friction_cells = float(OFF_F_CELLS)
        self.bdim_nt = self.n_iterations + 1
        # verify_energy_balance differentiates E_k on the diagnostics grid; the
        # production cadence of 100 leaves too few samples over 0.3 s.
        self.diagnostics_every = 10

    def extra_simulation_extensions(self, output_folder):
        return []                      # no viewers/recorders: need a display


def _resolved_offsets_cells(c, h):
    """(off_p, off_f) in cells that the ACTIVE readout will actually use.

    Mirrors FluidSolver.__init__ (solver.py ~446-464).  Printed rather than
    inferred because "which offset did that run use" has been the single most
    confounding question in this investigation — every §10 cross-method number
    is untrustworthy for exactly this reason.
    """
    eps_mult = 2.0 if c.eps_multiplier is None else float(c.eps_multiplier)
    lso_cells = 0.0 if c.lagrangian_sample_offset is None else \
        float(c.lagrangian_sample_offset) / h
    if c.sample_offset_pressure_cells is None:
        off_p = 0.0 if FORCE_METHOD == "eulerian" else lso_cells
    else:
        off_p = float(c.sample_offset_pressure_cells)
    if c.sample_offset_friction_cells is None:
        off_f = eps_mult if FORCE_METHOD == "eulerian" else lso_cells
    else:
        off_f = float(c.sample_offset_friction_cells)
    return off_p, off_f


if __name__ == "__main__":
    c = SimConfig()
    h = (c.xmax - c.xmin) / c.Nx
    off_p, off_f = _resolved_offsets_cells(c, h)
    pinned = (c.sample_offset_pressure_cells is not None
              or c.sample_offset_friction_cells is not None)
    print(f"[arb] case={CASE} force_method={FORCE_METHOD} "
          f"delta_order={DELTA_ORDER}")
    print(f"[arb] SAMPLING  (off_p, off_f) = ({off_p:g}h, {off_f:g}h)  "
          f"= ({off_p*h*1e3:.4f}, {off_f*h*1e3:.4f}) mm  "
          f"[{'split knobs, both readouts pinned' if pinned else 'LEGACY per-readout default'}]"
          f"{'  <- Verma et al. 2017' if (off_p, off_f) == (0.0, 2.0) else ''}")
    print(f"[arb] legacy lagrangian_sample_offset={c.lagrangian_sample_offset}")
    print(f"[arb] grid {c.Nx}x{c.Ny}x{c.Nz} = {c.Nx*c.Ny*c.Nz/1e6:.1f} M cells  "
          f"h={h*1e3:.4f} mm  dt={c.timestep}  n_iter={c.n_iterations}  "
          f"t_end={c.n_iterations*c.timestep:.3f} s  "
          f"R/h={0.559e-3/h:.2f} (thin direction)")
    c.single_run()
