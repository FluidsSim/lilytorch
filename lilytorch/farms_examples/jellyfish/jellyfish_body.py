"""Free-swimming 3-D jellyfish body for lilytorch's standalone FluidSolver.

The jellyfish geometry still follows the analytical WaterLily bell model:
a thin spherical shell clipped by a horizontal plane and pulsed through the
time-dependent maps used in ``ThreeD_Jelly.jl``. The difference is that the
body now also carries its own rigid state: a 6D Newton-Euler model advances
the jellyfish centre of mass and orientation from the hydrodynamic force and
torque computed by the solver.

This keeps the actuation local to the bell deformation while the global pose
is solved in the standalone fluid loop, without relying on MuJoCo.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from lilytorch.src.body import Body


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class JellyfishParams:
    """Geometric, actuation, and rigid-body parameters of the jellyfish.

    Defaults are a metric rescaling of the WaterLily ``jelly(L=2^5)``
    reference case:

        L_grid = 32,  R_grid = 2L/3 ≈ 21.3,  t_grid = 1,  h_grid = 4L−2R

    Mapping grid units to metres with a cell size of about ``dx``, so
    that ``R`` is some "physical" reference length and ``t`` is 4.7 %·R
    (= (3/64)·R).
    """

    # --- geometry (metres) --------------------------------------------------
    R: float = 0.05          # mean shell radius
    t: float = 0.05 * 3.0 / 64.0   # shell half-thickness (≈ 2.34 mm)
    h: float = 0.0           # cutting-plane height (sphere-centre frame)

    # --- kinematics --------------------------------------------------------
    U: float = 1.0           # reference speed used to set ω
    pulse_amplitude: float = 0.1   # lateral-scale modulation (1/10 in WaterLily)
    axial_amplitude: float = 0.25  # z-shift amplitude as fraction of R (R/4)

    # --- lab-frame placement ------------------------------------------------
    # Initial sphere-centre placement in the lab frame.
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    # --- rigid-body model ---------------------------------------------------
    # These values come from the generated jellyfish SDF and are treated as a
    # fixed reference inertia model for the pulsing bell.
    mass: float = 0.07546768732
    inertia_diag: tuple[float, float, float] = (
        7.893193369e-05,
        7.893193369e-05,
        1.262487591e-04,
    )
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.025037438901)
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    initial_orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def omega(self) -> float:
        return 2.0 * self.U / self.R

    @classmethod
    def from_solver_config(cls, pars: dict[str, Any] | None) -> "JellyfishParams":
        cfg = {} if pars is None else pars.get("jellyfish", {})

        origin = cfg.get("origin")
        if origin is None:
            origin = [cfg.get("x0", cls.x0), cfg.get("y0", cls.y0), cfg.get("z0", cls.z0)]

        return cls(
            R=cfg.get("R", cls.R),
            t=cfg.get("t", cls.t),
            h=cfg.get("h", cls.h),
            U=cfg.get("U", cls.U),
            pulse_amplitude=cfg.get("pulse_amplitude", cls.pulse_amplitude),
            axial_amplitude=cfg.get("axial_amplitude", cls.axial_amplitude),
            x0=origin[0],
            y0=origin[1],
            z0=origin[2],
            mass=cfg.get("mass", cls.mass),
            inertia_diag=tuple(cfg.get("inertia_diag", cls.inertia_diag)),
            com_offset=tuple(cfg.get("com_offset", cls.com_offset)),
            gravity=tuple(cfg.get("gravity", cls.gravity)),
            linear_damping=cfg.get("linear_damping", cls.linear_damping),
            angular_damping=cfg.get("angular_damping", cls.angular_damping),
            initial_orientation_rpy=tuple(
                cfg.get("initial_orientation_rpy", cls.initial_orientation_rpy)
            ),
            initial_linear_velocity=tuple(
                cfg.get("initial_linear_velocity", cls.initial_linear_velocity)
            ),
            initial_angular_velocity=tuple(
                cfg.get("initial_angular_velocity", cls.initial_angular_velocity)
            ),
        )


def _cross_components(ax, ay, az, bx, by, bz):
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion)


def _quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind()
    rw, rx, ry, rz = rhs.unbind()
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quaternion_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    half = 0.5 * rpy
    cr, cp, cy = torch.cos(half)
    sr, sp, sy = torch.sin(half)
    return _normalize_quaternion(
        torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            )
        )
    )


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternion(quaternion)
    qw, qx, qy, qz = q.unbind()

    two_qx = 2.0 * qx
    two_qy = 2.0 * qy
    two_qz = 2.0 * qz
    two_qwqx = two_qx * qw
    two_qwqy = two_qy * qw
    two_qwqz = two_qz * qw
    two_qxqx = two_qx * qx
    two_qxqy = two_qy * qx
    two_qxqz = two_qz * qx
    two_qyqy = two_qy * qy
    two_qyqz = two_qz * qy
    two_qzqz = two_qz * qz

    return torch.stack(
        (
            torch.stack((1.0 - (two_qyqy + two_qzqz), two_qxqy - two_qwqz, two_qxqz + two_qwqy)),
            torch.stack((two_qxqy + two_qwqz, 1.0 - (two_qxqx + two_qzqz), two_qyqz - two_qwqx)),
            torch.stack((two_qxqz - two_qwqy, two_qyqz + two_qwqx, 1.0 - (two_qxqx + two_qyqy))),
        )
    )


def _rotate_components(rotation: torch.Tensor, x, y, z):
    return (
        rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z,
        rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * z,
        rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z,
    )


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


class JellyfishBody(Body):
    """WaterLily jellyfish bell plus a rigid 6D free-swimming state."""

    def __init__(self, device, x, y, z, eps=0.05, grids=None,
                 params: JellyfishParams | None = None):
        super().__init__(device, x, y, z=z, eps=eps, grids=grids)
        assert self.ndim == 3, "JellyfishBody is 3-D only"

        self.params = params if params is not None else JellyfishParams()
        self.mass = torch.tensor(self.params.mass, device=device, dtype=self.dtype)
        self.inertia_diag = torch.tensor(
            self.params.inertia_diag, device=device, dtype=self.dtype,
        )
        self.inv_inertia_diag = 1.0 / self.inertia_diag
        self.com_offset_body = torch.tensor(
            self.params.com_offset, device=device, dtype=self.dtype,
        )
        self.gravity = torch.tensor(
            self.params.gravity, device=device, dtype=self.dtype,
        )
        self.linear_damping = torch.tensor(
            self.params.linear_damping, device=device, dtype=self.dtype,
        )
        self.angular_damping = torch.tensor(
            self.params.angular_damping, device=device, dtype=self.dtype,
        )

        initial_rpy = torch.tensor(
            self.params.initial_orientation_rpy, device=device, dtype=self.dtype,
        )
        self.rigid_quaternion = _quaternion_from_rpy(initial_rpy)
        self.rigid_rotation = _quaternion_to_matrix(self.rigid_quaternion)
        self.rigid_rotation_t = self.rigid_rotation.transpose(0, 1)
        initial_origin = torch.tensor(
            [self.params.x0, self.params.y0, self.params.z0],
            device=device, dtype=self.dtype,
        )
        self.rigid_com_pos = initial_origin + self.rigid_rotation @ self.com_offset_body
        self.rigid_lin_vel = torch.tensor(
            self.params.initial_linear_velocity, device=device, dtype=self.dtype,
        )
        self.rigid_ang_vel_body = torch.tensor(
            self.params.initial_angular_velocity, device=device, dtype=self.dtype,
        )
        self.rigid_ang_vel_world = self.rigid_rotation @ self.rigid_ang_vel_body
        self.origin_world = initial_origin.clone()
        self.last_force_world = torch.zeros(3, device=device, dtype=self.dtype)
        self.last_torque_world = torch.zeros(3, device=device, dtype=self.dtype)

        # Solver iterates over self.bodies (e.g. for force bookkeeping);
        # we expose a tiny proxy sub-body whose ``com_pos`` is 1-D (shape
        # (ndim,)) and whose ``eps`` matches ours — that's all the force
        # kernel reads off individual bodies.  The composite's own
        # ``com_pos`` (see below) stays 2-D (shape (nbodies, ndim)) to
        # satisfy ``FluidSolver.check_termination`` / ``inside``.
        self.body = self
        _sub_com = torch.tensor(
            [self.params.x0, self.params.y0, self.params.z0],
            device=device, dtype=self.dtype,
        )
        self._sub_body = SimpleNamespace(com_pos=_sub_com, eps=self.eps)
        self.bodies = [self._sub_body]
        self.nbodies = 1

        # Minimal contour stubs (3-D path of the solver does not use
        # contours, but some plotting hooks look them up).
        zero3 = torch.zeros(1, device=device, dtype=self.dtype)
        self.cnt = torch.zeros((3, 1), device=device, dtype=self.dtype)
        self.cnt_update = self.cnt.clone().detach()
        self.curv_coord = torch.tensor([0.0, 1.0], device=device, dtype=self.dtype)
        self.ds = self.curv_coord[1] - self.curv_coord[0]
        self.cnt_u = zero3.clone()
        self.cnt_v = zero3.clone()
        self.cnt_w = zero3.clone()
        self.cnt_f_u = zero3.clone()
        self.cnt_f_v = zero3.clone()
        self.cnt_f_w = zero3.clone()
        self.cnt_int_f_u = zero3.clone()
        self.cnt_int_f_v = zero3.clone()
        self.cnt_int_f_w = zero3.clone()
        self.mask = torch.arange(1, device=device)
        # Shape (nbodies, ndim) — solver.inside() expects a 2-D tensor
        # when iterating over bodies.
        self.com_pos = torch.tensor(
            [[self.rigid_com_pos[0], self.rigid_com_pos[1], self.rigid_com_pos[2]]],
            device=device, dtype=self.dtype,
        )

        self.clear_history()

        # Prime SDF / body-velocity fields so that the solver's initial
        # set-up (_recompute_mu_normals, plotting) finds valid tensors.
        self.update(torch.tensor(0.0, device=device, dtype=self.dtype), 0)

    def clear_history(self):
        self.iteration_history: list[int] = []
        self.time_history: list[float] = []
        self.origin_history: list[list[float]] = []
        self.com_history: list[list[float]] = []
        self.linear_velocity_history: list[list[float]] = []
        self.angular_velocity_body_history: list[list[float]] = []
        self.quaternion_history: list[list[float]] = []
        self.force_history: list[list[float]] = []
        self.torque_history: list[list[float]] = []

    def _record_state(self, t: torch.Tensor, iteration: int):
        self.iteration_history.append(int(iteration))
        self.time_history.append(float(t.item()))
        self.origin_history.append(self.origin_world.detach().cpu().tolist())
        self.com_history.append(self.rigid_com_pos.detach().cpu().tolist())
        self.linear_velocity_history.append(self.rigid_lin_vel.detach().cpu().tolist())
        self.angular_velocity_body_history.append(
            self.rigid_ang_vel_body.detach().cpu().tolist()
        )
        self.quaternion_history.append(self.rigid_quaternion.detach().cpu().tolist())

    def save_state_history(self, output_dir: str):
        if not self.time_history:
            return

        csv_path = os.path.join(output_dir, "jellyfish_rigid_state.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "iteration",
                    "time",
                    "origin_x", "origin_y", "origin_z",
                    "com_x", "com_y", "com_z",
                    "vel_x", "vel_y", "vel_z",
                    "omega_body_x", "omega_body_y", "omega_body_z",
                    "quat_w", "quat_x", "quat_y", "quat_z",
                    "force_x", "force_y", "force_z",
                    "torque_x", "torque_y", "torque_z",
                ]
            )
            for index, time_value in enumerate(self.time_history):
                force = self.force_history[index] if index < len(self.force_history) else [0.0, 0.0, 0.0]
                torque = self.torque_history[index] if index < len(self.torque_history) else [0.0, 0.0, 0.0]
                writer.writerow(
                    [
                        self.iteration_history[index],
                        time_value,
                        *self.origin_history[index],
                        *self.com_history[index],
                        *self.linear_velocity_history[index],
                        *self.angular_velocity_body_history[index],
                        *self.quaternion_history[index],
                        *force,
                        *torque,
                    ]
                )

    def apply_force_feedback(self, force, torque, iteration, time, dt, solver=None):
        del iteration, time, solver

        if force.ndim == 2:
            total_force = force.sum(dim=0)
        else:
            total_force = force

        if torque.ndim == 2:
            total_torque_world = torque.sum(dim=0)
        else:
            raise ValueError("JellyfishBody expects a 3-D torque vector.")

        total_force = total_force + self.mass * self.gravity - self.linear_damping * self.rigid_lin_vel
        total_torque_body = self.rigid_rotation_t @ total_torque_world - self.angular_damping * self.rigid_ang_vel_body

        linear_acc = total_force / self.mass
        angular_momentum_body = self.inertia_diag * self.rigid_ang_vel_body
        gyroscopic = torch.linalg.cross(self.rigid_ang_vel_body, angular_momentum_body)
        angular_acc_body = (total_torque_body - gyroscopic) * self.inv_inertia_diag

        self.rigid_lin_vel = self.rigid_lin_vel + dt * linear_acc
        self.rigid_com_pos = self.rigid_com_pos + dt * self.rigid_lin_vel
        self.rigid_ang_vel_body = self.rigid_ang_vel_body + dt * angular_acc_body

        omega_quat = torch.cat(
            (
                torch.zeros(1, device=self.device, dtype=self.dtype),
                self.rigid_ang_vel_body,
            )
        )
        quat_dot = 0.5 * _quaternion_multiply(self.rigid_quaternion, omega_quat)
        self.rigid_quaternion = _normalize_quaternion(self.rigid_quaternion + dt * quat_dot)
        self._refresh_rigid_kinematics()

        self.last_force_world = total_force
        self.last_torque_world = total_torque_world
        self.force_history.append(total_force.detach().cpu().tolist())
        self.torque_history.append(total_torque_world.detach().cpu().tolist())

    def _refresh_rigid_kinematics(self):
        self.rigid_rotation = _quaternion_to_matrix(self.rigid_quaternion)
        self.rigid_rotation_t = self.rigid_rotation.transpose(0, 1)
        self.rigid_ang_vel_world = self.rigid_rotation @ self.rigid_ang_vel_body
        self.origin_world = self.rigid_com_pos - self.rigid_rotation @ self.com_offset_body

    # ------------------------------------------------------------------
    #   Analytical SDF pieces (in the sphere body-frame coordinates)
    # ------------------------------------------------------------------

    def _sdf_sphere_shell(self, x, y, z):
        """Signed distance to the spherical shell of mean radius R, half
        thickness t.  Negative inside the shell."""
        p = self.params
        r = torch.sqrt(x * x + y * y + z * z)
        return torch.abs(r - p.R) - p.t

    def _sdf_plane_upper(self, z):
        """SDF of the half-space z >= h (negative inside, i.e. z > h)."""
        return self.params.h - z  # note: negative where z > h (inside)

    def _sdf_bell(self, xb, yb, zb, zp):
        """Body SDF = sphere shell ∩ upper half-space.

        The sphere is evaluated at sphere-frame coordinates ``(xb, yb,
        zb) = A·x_lab + B + C`` and the plane at plane-frame
        coordinates ``zp = z_lab + C_z``.
        """
        sdf_s = self._sdf_sphere_shell(xb, yb, zb)
        sdf_p = self._sdf_plane_upper(zp)
        # CSG intersection: max(sdf_A, sdf_B).  ``sdf_p`` is already
        # negative inside the half-space z > h, so we just take the
        # max.  (This matches WaterLily's ``sphere − plane`` which is
        # implemented there as an intersection with −plane.)
        return torch.maximum(sdf_s, sdf_p)

    # ------------------------------------------------------------------
    #   Kinematic maps  (A(t), B(t), C(t) and their time derivatives)
    # ------------------------------------------------------------------

    def _kinematics(self, t: torch.Tensor):
        """Return the scalar factors (Ax, Az, Bz, Cz) and their time
        derivatives for time ``t``.

        WaterLily:
            A(t) = (1 − cos(ωt)/10, 1 − cos(ωt)/10, 1)
            B(t) = (0, 0, (cos(ωt)−1) R/4 − h)
            C(t) = (0, 0, sin(ωt) R/4)
        """
        p = self.params
        w = p.omega()
        cw = torch.cos(w * t)
        sw = torch.sin(w * t)

        Ax = 1.0 - p.pulse_amplitude * cw                       # also Ay
        Az = torch.ones_like(Ax)
        Bz = (cw - 1.0) * p.axial_amplitude * p.R - p.h
        Cz = sw * p.axial_amplitude * p.R

        # Time derivatives
        dAx = p.pulse_amplitude * w * sw
        dAz = torch.zeros_like(dAx)
        dBz = -w * sw * p.axial_amplitude * p.R
        dCz = w * cw * p.axial_amplitude * p.R

        return Ax, Az, Bz, Cz, dAx, dAz, dBz, dCz

    # ------------------------------------------------------------------
    #   Material velocity of the body at lab-frame grid points
    # ------------------------------------------------------------------
    # For the sphere part, a material point is fixed in the *sphere
    # frame* where coordinates are X_b = A · (x_lab − c0) + B + C, with
    # c0 = (x0, y0, z0) the constant lab-frame offset of the sphere
    # centre.  Differentiating X_b = const gives
    #
    #     ẋ_lab = − (1/A) · (Ȧ · (x_lab − c0) + Ḃ + Ċ).
    #
    # That is the Eulerian body velocity at the lab-frame point x_lab:
    # exactly what BDIM requires to enforce no-slip.  The plane cuts the
    # body but carries no material — we use the sphere's velocity
    # everywhere, which is correct on the shell surface (the only place
    # where the plane actually clips material).

    def _body_velocity(self, X, Y, Z,
                       Yx, Yy, Yz,
                       Ax, Az, dAx, dAz, dBz, dCz):
        # Internal deformation velocity in the body frame, plus rigid COM
        # translation and rigid-body rotation in the lab frame.
        du_x_local = -(dAx * Yx) / Ax
        du_y_local = -(dAx * Yy) / Ax
        du_z_local = -(dAz * Yz + dBz + dCz) / Az

        du_x_world, du_y_world, du_z_world = _rotate_components(
            self.rigid_rotation,
            du_x_local,
            du_y_local,
            du_z_local,
        )

        rx = X - self.rigid_com_pos[0]
        ry = Y - self.rigid_com_pos[1]
        rz = Z - self.rigid_com_pos[2]
        omega_x, omega_y, omega_z = self.rigid_ang_vel_world
        rigid_x, rigid_y, rigid_z = _cross_components(
            omega_x, omega_y, omega_z,
            rx, ry, rz,
        )

        return (
            self.rigid_lin_vel[0] + rigid_x + du_x_world,
            self.rigid_lin_vel[1] + rigid_y + du_y_world,
            self.rigid_lin_vel[2] + rigid_z + du_z_world,
        )

    # ------------------------------------------------------------------
    #   Main update – called every solver time step
    # ------------------------------------------------------------------

    def update(self, t, iteration, dt=1):
        device = self.device
        dtype = self.dtype
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)

        self._refresh_rigid_kinematics()
        Ax, Az, Bz, Cz, dAx, dAz, dBz, dCz = self._kinematics(t)

        # Helper: evaluate the combined SDF on an arbitrary (X,Y,Z)
        # lab-frame grid.  The sphere and plane maps both shift z by Cz
        # (C(t) is common), but the sphere additionally rescales laterally
        # by A(t) and translates by Bz in z.
        def _sdf_on(X, Y, Z):
            rx = X - self.origin_world[0]
            ry = Y - self.origin_world[1]
            rz = Z - self.origin_world[2]
            Yx, Yy, Yz = _rotate_components(self.rigid_rotation_t, rx, ry, rz)
            # Sphere-frame coords: X_b = A · r_lab + B + C
            xb = Ax * Yx
            yb = Ax * Yy
            zb = Az * Yz + Bz + Cz
            # Plane-frame z: z_p = z_lab + C_z  (map from WaterLily,
            # "x .+ C(t)").  The plane SDF is (h − z_p).
            zp = Yz + Cz
            return self._sdf_bell(xb, yb, zb, zp)

        # ------------------------------------------------------------
        # Cell-centred & MAC staggered SDFs
        # ------------------------------------------------------------
        self.sdf_val   = _sdf_on(self.X,       self.Y,       self.Z_grid)
        self.sdf_val_u = _sdf_on(self.Xu_stag, self.Yu_stag, self.Zu_stag)
        self.sdf_val_v = _sdf_on(self.Xv_stag, self.Yv_stag, self.Zv_stag)
        self.sdf_val_w = _sdf_on(self.Xw_stag, self.Yw_stag, self.Zw_stag)

        # Aliases so the single-body solver code that expects
        # ``body.sdf_u / sdf_v / sdf_w`` / ``body.sdf_val`` also works.
        self.sdf_u = self.sdf_val_u
        self.sdf_v = self.sdf_val_v
        self.sdf_w = self.sdf_val_w

        # ------------------------------------------------------------
        # Material velocities on each staggered grid
        # ------------------------------------------------------------
        xu = self.Xu_stag - self.origin_world[0]
        yu = self.Yu_stag - self.origin_world[1]
        zu = self.Zu_stag - self.origin_world[2]
        Yux, Yuy, Yuz = _rotate_components(self.rigid_rotation_t, xu, yu, zu)

        xv = self.Xv_stag - self.origin_world[0]
        yv = self.Yv_stag - self.origin_world[1]
        zv = self.Zv_stag - self.origin_world[2]
        Yvx, Yvy, Yvz = _rotate_components(self.rigid_rotation_t, xv, yv, zv)

        xw = self.Xw_stag - self.origin_world[0]
        yw = self.Yw_stag - self.origin_world[1]
        zw = self.Zw_stag - self.origin_world[2]
        Ywx, Ywy, Ywz = _rotate_components(self.rigid_rotation_t, xw, yw, zw)

        bu, _, _ = self._body_velocity(
            self.Xu_stag, self.Yu_stag, self.Zu_stag,
            Yux, Yuy, Yuz,
            Ax, Az, dAx, dAz, dBz, dCz)
        _, bv, _ = self._body_velocity(
            self.Xv_stag, self.Yv_stag, self.Zv_stag,
            Yvx, Yvy, Yvz,
            Ax, Az, dAx, dAz, dBz, dCz)
        _, _, bw = self._body_velocity(
            self.Xw_stag, self.Yw_stag, self.Zw_stag,
            Ywx, Ywy, Ywz,
            Ax, Az, dAx, dAz, dBz, dCz)

        self.body_u = bu
        self.body_v = bv
        self.body_w = bw

        # ------------------------------------------------------------
        # Per-body SDF stack expected by the force-computation path.
        # ``FluidSolver.forces_method2_3d`` iterates over ``comp.bodies``
        # and reads ``comp.sdf_vals[i]`` when no sparse storage is set.
        # We stack our single SDF along a leading "body" dimension.
        # ``_sdf_sparse`` is intentionally left unset so the solver
        # falls back to this dense path.
        # ------------------------------------------------------------
        self.sdf_vals = self.sdf_val.unsqueeze(0)

        # Shape (nbodies, ndim) — used by solver.inside() / check_termination
        self.com_pos = self.rigid_com_pos.unsqueeze(0)
        # 1-D shape (ndim,) — used by forces_method2_3d per sub-body
        self._sub_body.com_pos = self.rigid_com_pos

        self._record_state(t, iteration)
