"""Tests for BDIMhandler's ``_pose_source`` (strong-coupling data path).

Two layers:

1. **Convention tests** (runnable anywhere with mujoco) — validate the exact
   MuJoCo fields/conventions ``_gather_data_physics`` relies on, mirroring
   ``farms_mujoco.simulation.physics.physics2data``:
     * ``framelinvel`` / ``frameangvel`` (objtype xbody, global frame)
       sensordata == ``mj_objectVelocity`` (lin / ang),
     * ``xquat[[1,2,3,0]]`` (mujoco wxyz -> xyzw) -> rotation == ``xmat``,
     * ``xpos`` / ``xipos`` are the frame-origin / COM world positions.

2. **Equivalence test** (skipped unless a live FARMS coupled sim is
   available) — at the start-of-step pose, ``gather_data`` with
   ``_pose_source="sensors"`` must equal ``_pose_source="physics"``.

Run: ``pytest lilytorch/integration/test_pose_source.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from scipy.spatial.transform import Rotation


_MODEL_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="link" pos="0.3 -0.2 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.05 0.02" pos="0.04 0 0" density="1000"/>
    </body>
  </worldbody>
  <sensor>
    <framelinvel name="lv" objtype="xbody" objname="link"/>
    <frameangvel name="av" objtype="xbody" objname="link"/>
  </sensor>
</mujoco>
"""


def _stepped_model():
    m = mujoco.MjModel.from_xml_string(_MODEL_XML)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link")
    # free joint: qpos = [x,y,z, qw,qx,qy,qz], qvel = [vx,vy,vz, wx,wy,wz]
    quat = Rotation.from_euler("xyz", [0.3, -0.4, 0.7]).as_quat()  # xyzw
    d.qpos[:3] = [0.3, -0.2, 0.5]
    d.qpos[3:7] = quat[[3, 0, 1, 2]]                                # -> wxyz
    d.qvel[:3] = [0.5, -0.2, 0.1]                                   # linear
    d.qvel[3:6] = [0.05, -0.1, 0.2]                                 # angular
    mujoco.mj_forward(m, d)
    return m, d, bid


def test_framevel_sensors_match_object_velocity():
    """framelinvel/frameangvel sensordata == mj_objectVelocity (world frame)."""
    m, d, bid = _stepped_model()
    lv_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "lv")]
    av_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "av")]
    lv = d.sensordata[lv_adr:lv_adr + 3].copy()
    av = d.sensordata[av_adr:av_adr + 3].copy()

    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, vel6, 0)  # 0 = global
    ang_ref, lin_ref = vel6[:3], vel6[3:]

    assert np.allclose(lv, lin_ref, atol=1e-9), (lv, lin_ref)
    assert np.allclose(av, ang_ref, atol=1e-9), (av, ang_ref)


def test_quat_wxyz_to_xyzw_matches_xmat():
    """xquat[[1,2,3,0]] -> Rotation matrix == data.xmat (the convention used
    by both gather_data and _gather_data_physics)."""
    m, d, bid = _stepped_model()
    quat_xyzw = d.xquat[bid][[1, 2, 3, 0]]
    R_from_quat = Rotation.from_quat(quat_xyzw).as_matrix()
    R_xmat = d.xmat[bid].reshape(3, 3)
    assert np.allclose(R_from_quat, R_xmat, atol=1e-9)


def test_xpos_xipos_are_origin_and_com():
    """xpos = body-frame origin; xipos = body COM (offset by the geom pos)."""
    m, d, bid = _stepped_model()
    # geom is offset +0.04 along the body x-axis from the frame origin, so
    # xipos != xpos and the offset, rotated to world, has magnitude 0.04.
    offset = d.xipos[bid] - d.xpos[bid]
    assert np.isclose(np.linalg.norm(offset), 0.04, atol=1e-6)


# ---------------------------------------------------------------------------
#  Equivalence test (needs a live FARMS coupled sim) — skipped otherwise
# ---------------------------------------------------------------------------
def assert_gather_data_sources_agree(handler, task, physics, atol=1e-6):
    """Reusable assertion: at the current (synced) physics state, the
    "sensors" and "physics" pose sources must return matching kinematics.

    Call this from a live FARMS run right after ``task.update_sensors`` (so
    the sensor buffer and physics.data describe the same pose) and before
    any internal prediction step.
    """
    handler._task = task
    handler._physics = physics
    # The MuJoCo/FARMS access now lives behind the rigid-body backend (AP1);
    # bind the live task/physics there so the "physics" pose source reads them.
    handler._backend.bind_step(task, physics)

    handler._pose_source = "sensors"
    s_com, s_urdf, s_R, s_lin, s_ang = handler.gather_data(handler.iteration)
    handler._pose_source = "physics"
    p_com, p_urdf, p_R, p_lin, p_ang = handler.gather_data(handler.iteration)
    handler._pose_source = "sensors"

    for a, b, name in [
        (s_com, p_com, "com_pos"), (s_urdf, p_urdf, "urdf_pos"),
        (s_R, p_R, "R"), (s_lin, p_lin, "lin_vel"), (s_ang, p_ang, "ang_vel"),
    ]:
        for i, (ai, bi) in enumerate(zip(a, b)):
            assert np.allclose(np.asarray(ai), np.asarray(bi), atol=atol), (
                f"{name} mismatch on animat {i}:\n sensors={ai}\n physics={bi}"
            )


@pytest.mark.skip(
    reason="Requires a live FARMS coupled sim (task+physics+AnimatData). "
           "Call assert_gather_data_sources_agree(handler, task, physics) "
           "from within a real coupled run (e.g. a sphere-drop step) to "
           "verify the two pose sources agree at the start-of-step pose."
)
def test_gather_data_physics_matches_sensors_in_farms():
    pass
