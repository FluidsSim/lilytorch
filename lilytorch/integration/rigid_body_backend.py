"""Rigid-body engine adapter for :class:`BDIMhandler`.

``BDIMhandler`` (the FARMS<->lilytorch FSI bridge) does not import FARMS: it
speaks MuJoCo directly and pulls index-maps / unit scales from a FARMS ``task``
object.  This module isolates that *entire* engine-specific surface behind a
small adapter so the handler's numerics stay engine-agnostic.

This is the AP1 seam of the portability plan (drop FARMS, support pluggable
rigid-body engines).  Today the only implementation is
:class:`FarmsMujocoBackend` (MuJoCo state + FARMS ``task`` maps/units); a raw
``mujoco.MjModel/MjData`` backend (AP2) and an Isaac Lab backend (AP5) can later
implement the same :class:`RigidBodyBackend` surface without touching the
handler.

Contract: all quantities crossing the adapter are **SI** and (in 3-D) full
three-component; the 2-D reduction (``lin_axes`` / ``_2d_ang_ax`` slicing) is
applied inside the backend's pose/velocity return so it exactly reproduces the
pre-adapter handler behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation


class MujocoCheckpoint:
    """Save/restore the full MuJoCo integration state for sub-iteration.

    ``mjSTATE_INTEGRATION`` bundles qpos, qvel, act, time, qacc_warmstart,
    ctrl, qfrc_applied, xfrc_applied, mocap and eq_active -- everything
    needed to reproduce an integration step deterministically across a
    checkpoint/restore.  Used by the implicit (strongly-coupled) step to
    run throwaway prediction integrations that are each undone before the
    next coupling sweep.  See STRONG_COUPLING_FARMS_DESIGN.md section 5.
    """

    def __init__(self, physics):
        import mujoco
        self._mj = mujoco
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.m = physics.model.ptr
        self.d = physics.data.ptr
        self.n = mujoco.mj_stateSize(self.m, self.spec)
        self._buf = np.zeros(self.n, dtype=np.float64)

    def save(self):
        self._mj.mj_getState(self.m, self.d, self._buf, self.spec)
        return self._buf.copy()

    def restore(self, state):
        self._mj.mj_setState(self.m, self.d, state, self.spec)
        self._mj.mj_forward(self.m, self.d)

    def integrate(self, nstep=1):
        """Advance ``nstep`` MuJoCo steps in place, then refresh derived
        quantities (xpos/xipos/xquat/sensordata) for the new state so the
        fluid's ``physics`` pose source reads the predicted pose."""
        self._mj.mj_step(self.m, self.d, nstep)
        self._mj.mj_forward(self.m, self.d)


class RigidBodyBackend(ABC):
    """Minimal surface :class:`BDIMhandler` needs from a rigid-body engine.

    All poses/velocities are returned in SI units; all applied wrenches are
    given in SI units.  Per-animat lists are ordered to match the handler's
    ``self.data`` (the FARMS ``AnimatData`` list).
    """

    @abstractmethod
    def bind_step(self, task, physics):
        """Bind the live per-step engine handles (``task`` carries the FARMS
        index maps + unit scales; ``physics`` the MuJoCo state)."""

    @property
    @abstractmethod
    def gravity_z(self) -> float:
        """Gravity along the world z-axis [m/s^2] (e.g. -9.81)."""

    @abstractmethod
    def set_contact_params(self, solref, solimp):
        """Optionally override engine contact-solver parameters (init-time)."""

    @abstractmethod
    def get_body_poses_velocities(self, source, iteration):
        """Return ``(com_poses, urdf_poses, Rs, lin_vels, ang_vels)`` per animat.

        ``source`` selects the read path: ``"sensors"`` (FARMS ``AnimatData``
        buffers at ``iteration``) or ``"physics"`` (live engine state).
        """

    @abstractmethod
    def get_body_mass_radius(self, animat_id, link_id):
        """Return ``(mass, max_rbound)`` for one link, in SI units."""

    @abstractmethod
    def apply_xfrc(self, body_ids, lin_xfrc_idx, ang_xfrc_idx,
                   lin_total, ang_total, buoyancy, buoy_xidx):
        """Write per-body external linear/angular forces to the engine.

        ``lin_total`` / ``ang_total`` are SI loads shaped ``(D, B)`` / ``(Nt, B)``;
        ``buoyancy`` is a per-body SI scalar added on the ``buoy_xidx`` linear
        axis (``buoy_xidx is None`` => no buoyancy term).
        """

    @abstractmethod
    def checkpoint(self):
        """Return a fresh checkpoint object (``save``/``restore``/``integrate``)
        for the implicit-coupling sub-iteration."""


class FarmsMujocoBackend(RigidBodyBackend):
    """MuJoCo state + FARMS ``task`` (index maps / unit scales) backend.

    Holds the (reused) ``physics`` instance from construction; the live
    ``task`` (and a defensive ``physics`` re-bind) are refreshed each step via
    :meth:`bind_step`.
    """

    def __init__(self, physics, data, ndim, lin_axes, _2d_ang_ax, dtype_np):
        self._physics = physics
        self.data = data
        self.ndim = ndim
        self.lin_axes = lin_axes
        self._2d_ang_ax = _2d_ang_ax
        self.dtype_np = dtype_np
        self._task = None

    # -- per-step binding -------------------------------------------------
    def bind_step(self, task, physics):
        self._task = task
        if physics is not None:
            self._physics = physics

    # -- static engine queries -------------------------------------------
    @property
    def gravity_z(self) -> float:
        return float(self._physics.model.opt.gravity[2])

    def set_contact_params(self, solref, solimp):
        model = self._physics.model
        if solref is not None:
            model.geom_solref[:, 0] = solref[0]
            model.geom_solref[:, 1] = solref[1]
        if solimp is not None:
            model.geom_solimp[:, 0] = solimp[0]
            model.geom_solimp[:, 1] = solimp[1]
            model.geom_solimp[:, 2] = solimp[2]
            model.geom_solimp[:, 3] = solimp[3]
            model.geom_solimp[:, 4] = solimp[4]

    def get_body_mass_radius(self, animat_id, link_id):
        model = self._physics.model
        ind = self._task.maps[animat_id]["sensors"]["data2xfrc"][link_id]
        mass = float(model.body_mass[ind])
        # Max bounding-sphere radius among geoms on this body (FARMS
        # SwimmingHandler logic).
        max_rbound = 0.0
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) == ind:
                rb = float(model.geom_rbound[gi])
                if rb > max_rbound:
                    max_rbound = rb
        return mass, max_rbound

    # -- pose / velocity reads -------------------------------------------
    def get_body_poses_velocities(self, source, iteration):
        if source == "physics":
            return self._gather_physics()
        return self._gather_sensors(iteration)

    def _gather_sensors(self, iteration):
        """Read FARMS ``AnimatData`` sensor buffers at ``iteration``."""
        com_poses, urdf_poses, Rs, lin_vels, ang_vels = [], [], [], [], []
        for exp_data in self.data:
            sen = exp_data.sensors.links

            com = np.asarray(sen.com_positions()[iteration, :], dtype=self.dtype_np)
            urdf = np.asarray(sen.urdf_positions()[iteration, :], dtype=self.dtype_np)
            R = Rotation.from_quat(sen.urdf_orientations()[iteration, :]).as_matrix().astype(self.dtype_np)
            lin = np.asarray(sen.com_lin_velocities()[iteration, :], dtype=self.dtype_np)
            nlinks = len(sen.names)
            if self.ndim == 2:
                com = com[:, self.lin_axes]
                urdf = urdf[:, self.lin_axes]
                R = R[:, self.lin_axes, :][:, :, self.lin_axes]
                lin = lin[:, self.lin_axes]
                ang = np.asarray(
                    [sen.com_ang_velocity(iteration, lk)[self._2d_ang_ax]
                        for lk in range(nlinks)],
                    dtype=self.dtype_np,
                )
            else:
                ang = np.stack([
                    np.asarray(sen.com_ang_velocity(iteration, lk), dtype=self.dtype_np)
                    for lk in range(nlinks)
                ])
            com_poses.append(com)
            urdf_poses.append(urdf)
            Rs.append(R)
            lin_vels.append(lin)
            ang_vels.append(ang)
        return com_poses, urdf_poses, Rs, lin_vels, ang_vels

    def _gather_physics(self):
        """Read live ``physics.data`` (strong coupling), mirroring
        ``farms_mujoco.simulation.physics.physics2data`` field-for-field:

        * ``urdf_pos`` <- ``data.xpos[xpos2data]   / units.meters``
        * ``com_pos``  <- ``data.xipos[xipos2data] / units.meters``
        * ``R``        <- ``Rotation.from_quat(data.xquat[xquat2data][:, [1,2,3,0]])``
        * ``lin_vel``  <- ``data.sensordata[framelinvel2data] / units.velocity``
        * ``ang_vel``  <- ``data.sensordata[frameangvel2data] / units.angular_velocity``
        """
        task = self._task
        units = task.units
        d = self._physics.data

        com_poses, urdf_poses, Rs, lin_vels, ang_vels = [], [], [], [], []
        for animat_i, _exp in enumerate(self.data):
            sm = task.maps[animat_i]["sensors"]

            urdf = np.asarray(d.xpos[sm["xpos2data"]], dtype=self.dtype_np) / units.meters
            com = np.asarray(d.xipos[sm["xipos2data"]], dtype=self.dtype_np) / units.meters
            quat = np.asarray(d.xquat[sm["xquat2data"]], dtype=self.dtype_np)[:, [1, 2, 3, 0]]
            R = Rotation.from_quat(quat).as_matrix().astype(self.dtype_np)
            lin = np.asarray(d.sensordata[sm["framelinvel2data"]], dtype=self.dtype_np) / units.velocity
            ang = np.asarray(d.sensordata[sm["frameangvel2data"]], dtype=self.dtype_np) / units.angular_velocity

            if self.ndim == 2:
                com = com[:, self.lin_axes]
                urdf = urdf[:, self.lin_axes]
                R = R[:, self.lin_axes, :][:, :, self.lin_axes]
                lin = lin[:, self.lin_axes]
                ang = ang[:, self._2d_ang_ax]

            com_poses.append(com)
            urdf_poses.append(urdf)
            Rs.append(R)
            lin_vels.append(lin)
            ang_vels.append(ang)
        return com_poses, urdf_poses, Rs, lin_vels, ang_vels

    # -- force application ------------------------------------------------
    def apply_xfrc(self, body_ids, lin_xfrc_idx, ang_xfrc_idx,
                   lin_total, ang_total, buoyancy, buoy_xidx):
        task = self._task
        physics = self._physics
        units_N = task.units.newtons

        for body_i, (animat_id, link_id) in enumerate(body_ids):
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            for d, xidx in enumerate(lin_xfrc_idx):
                val = lin_total[d][body_i] * units_N
                if xidx == buoy_xidx:
                    val += buoyancy[body_i] * units_N
                physics.data.xfrc_applied[ind, xidx] = val

            for d, xidx in enumerate(ang_xfrc_idx):
                physics.data.xfrc_applied[ind, xidx] = ang_total[d][body_i] * units_N

    # -- implicit-coupling stepping --------------------------------------
    def checkpoint(self):
        return MujocoCheckpoint(self._physics)
