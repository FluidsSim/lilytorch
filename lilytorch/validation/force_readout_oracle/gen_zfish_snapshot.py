"""Run the zebrafish gait headless and dump ONE frozen force-readout scene.

Reuses the production ``gen_configs_pd_3d_slow_fast.SimConfig`` unchanged except
for what makes it cheap and reproducible here: headless, no cameras/video, no
GL iso viewer, and a run that stops the moment the snapshot is on disk.  Keeping
the physics/grid/gait identical to production is the whole point — the snapshot
must be a scene the real run actually visits.

    ZFISH_SNAP_STEP=300 ZFISH_SNAP_OUT=/path/snap.pt \
        python -m lilytorch.validation.force_readout_oracle.gen_zfish_snapshot

Then sweep it offline with ``shift_sweep_3d.py``.
"""
from __future__ import annotations

import os

from lilytorch.examples.zebrafish_ki_project.gen_configs_pd_3d_slow_fast import (
    SimConfig as _ProdConfig,
)

SNAP_STEP = int(os.environ.get("ZFISH_SNAP_STEP", 300))
SNAP_OUT = os.environ.get("ZFISH_SNAP_OUT", "/data/andreaferrario/ns_data/"
                                            "zfish_force_snapshot/snap.pt")


class SimConfig(_ProdConfig):

    def __init__(self):
        super().__init__()
        self.headless = True
        self.save = False
        self.save_frames = False
        # The hook wraps forces_method2_3d, so the run must take the eulerian
        # branch; the sweep re-derives the lagrangian side offline anyway.
        self.force_method = "eulerian"
        # Enough steps to reach SNAP_STEP; the hook aborts the run there.
        self.n_iterations = SNAP_STEP + 10
        self.bdim_nt = self.n_iterations + 1

    def extra_simulation_extensions(self, output_folder):
        # Drop every viewer/recorder: they need a display and dominate runtime.
        return []

    def _extra_run_patch(self):
        return (
            "import lilytorch.validation.force_readout_oracle."
            "zfish_snapshot_hook as _z; "
            f"_z.install(step={SNAP_STEP!r}, out={SNAP_OUT!r}, stop=True);"
        )


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SNAP_OUT), exist_ok=True)
    SimConfig().single_run()
