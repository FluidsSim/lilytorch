"""Determinism test for _MujocoCheckpoint (the Option-A foundation).

A restored checkpoint must reproduce the integration exactly: predicting
from sⁿ under fixed inputs must give the same s̃ every time, and restoring
must return the state bit-for-bit.  This is what lets the implicit loop run
throwaway prediction steps and lets the runtime reproduce the committed
prediction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lilytorch.integration.BDIMhandler import _MujocoCheckpoint


_XML = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.05 0.02" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


def _physics():
    m = mujoco.MjModel.from_xml_string(_XML)
    d = mujoco.MjData(m)
    # a non-trivial state: spin + drift + an applied wrench
    d.qvel[:3] = [0.4, -0.2, 0.1]
    d.qvel[3:6] = [0.3, -0.5, 0.2]
    d.xfrc_applied[1, :3] = [1.0, -2.0, 3.0]
    mujoco.mj_forward(m, d)
    return SimpleNamespace(model=SimpleNamespace(ptr=m), data=SimpleNamespace(ptr=d)), m, d


def test_restore_is_exact():
    physics, m, d = _physics()
    ckpt = _MujocoCheckpoint(physics)
    s0 = ckpt.save()
    qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()

    ckpt.integrate(3)                      # perturb
    assert not np.allclose(d.qpos, qpos0)  # state actually changed

    ckpt.restore(s0)                       # should bring everything back
    assert np.allclose(d.qpos, qpos0, atol=0, rtol=0)
    assert np.allclose(d.qvel, qvel0, atol=0, rtol=0)
    # xfrc_applied is part of mjSTATE_INTEGRATION
    assert np.allclose(d.xfrc_applied[1, :3], [1.0, -2.0, 3.0])


def test_prediction_is_reproducible_under_restore():
    """Predict from sⁿ twice (with a restore between) -> identical s̃.

    This is the property Option A relies on: the prediction the implicit
    loop commits is exactly what re-integrating from sⁿ under the same
    inputs produces."""
    physics, m, d = _physics()
    ckpt = _MujocoCheckpoint(physics)
    s0 = ckpt.save()

    ckpt.integrate(1)
    qpos_a, qvel_a = d.qpos.copy(), d.qvel.copy()

    ckpt.restore(s0)
    ckpt.integrate(1)
    qpos_b, qvel_b = d.qpos.copy(), d.qvel.copy()

    assert np.allclose(qpos_a, qpos_b, atol=0, rtol=0)
    assert np.allclose(qvel_a, qvel_b, atol=0, rtol=0)


def test_state_includes_xfrc_size():
    physics, m, d = _physics()
    ckpt = _MujocoCheckpoint(physics)
    # nq+nv+na = 7+6+0 = 13; INTEGRATION adds time/warmstart/ctrl/qfrc/xfrc/...
    assert ckpt.n > m.nq + m.nv + m.na
