"""CPU-only workarounds needed by the Step-5 probe on this branch.

These patches are confined to the probe and do not modify the source tree.
They exist because:

1. The available CPU-only torch build (nightly 2.12.0) has a broken
   ``torch.compile`` (CSE typing error) when invoked on the solver's
   advection/diffusion stencils.  We replace ``torch.compile`` with the
   identity wrapper so the python reference path is exercised without
   compilation.

2. ``PoissonSolver._dispatch_vcycle`` (poisson_mult.py:903) unconditionally
   dispatches to CUDA-only native ops even when ``solver_method='python'``.
   We route CPU tensors through the pure-python ``_vcycle`` that already
   exists in the same module.

3. ``CompositeBodyAnalytical.update`` (body.py:1253) does not populate
   ``_sdf_sparse``; this attribute is required by ``forces_method2``
   (forces.py:360-371) when ``compute_forces=True``.  In the FARMS path
   ``BDIMhandler._update_2d`` populates it; the standalone analytical-body
   path used by ``cylinder_drag_2d`` does not.  We synthesise the minimal
   no-AABB sparse form so forces can be integrated for numerical
   equivalence checks.

These patches are *not* relevant to the Step-5 hypothesis itself; they are
solely about getting the analytical-body harness to run end-to-end on CPU.
"""

import torch as _t


def install_cpu_patches():
    _t.compile = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))

    from lilytorch.src import poisson_mult as _pm
    _orig_dispatch = _pm.PoissonSolver._dispatch_vcycle

    def _patched_dispatch(self, f, p, face_arrs):
        if f.device.type != "cpu":
            return _orig_dispatch(self, f, p, face_arrs)
        return self._vcycle(f, p, face_arrs)

    _pm.PoissonSolver._dispatch_vcycle = _patched_dispatch

    from lilytorch.src import body as _b
    _orig_cba_update = _b.CompositeBodyAnalytical.update

    def _patched_cba_update(self, t, iteration, dt=1):
        _orig_cba_update(self, t, iteration, dt=dt)
        self._sdf_sparse = [(None, body.sdf_val) for body in self.bodies]

    _b.CompositeBodyAnalytical.update = _patched_cba_update
