"""``src_warp.forces`` — hydrodynamic force readout (native for now).

Two force readouts exist in ``lilytorch.src.forces``:

* **Eulerian** ``streaming_sdf_forces_post_{2,3}d`` — the n·δ viscous+pressure
  band integral.  This is the **one native op without a Warp port** (a
  block-reduction + ``atomic_add`` scatter; tractable as a ``warp_forces.py``,
  same class as ``warp_lagrangian.py`` — see §F option "port it first").
* **Lagrangian** ``lagrangian_forces_{2,3}d`` — IS ported to Warp
  (``warp_poc/warp_lagrangian.py``, parity ≤1.7e-16, CPU==GPU), but its Warp
  wrapper takes a *decomposed* arg list (eps_xx…, tri_centroid/normal/area…),
  not the native fused call convention of ``forces.py``.

So the recommended SU1 routing (§4 of the task brief) is to run the validation
cases with ``force_method="lagrangian"`` on the native op, which is the most
accurate readout for fully-resolved bodies anyway.  Wiring the Warp lagrangian
wrapper behind the native ``forces.py`` call site (or porting the Eulerian
readout) is the remaining force-path work.  This module is the native readout
under the Warp namespace.
"""
from lilytorch.src.forces import *  # noqa: F401,F403
