"""Strongly-coupled (implicit) FSI driver for a standalone rigid body.

This turns lilytorch's *explicit* per-step force push into a converged
fixed-point iteration using the accelerators in :mod:`fsi_coupling`.  It
targets the standalone ``apply_force_feedback`` path (e.g.
:class:`JellyfishBody`, or any analytic rigid body) — **not** the
FARMS/MuJoCo path, where the structure integrator is owned by MuJoCo and
sub-iterating requires re-stepping ``mj_step`` (a larger change, sketched
at the bottom of this file).

The per-step algorithm (compare :meth:`FluidSolver.step_`)::

    snapshot fluid fields (u0,v0,w0,p0) and body state at start of step n
    x  <- extrapolated guess of the end-of-step body coupling state
    repeat:
        restore fluid to the start-of-step snapshot      # same x_n each sweep
        impose candidate body state x  (SDF + no-slip velocity)
        advance the fluid one step      -> hydrodynamic loads (F, T)
        integrate the body from state_n under (F, T)  -> x_tilde
        if ||x_tilde - x|| < tol: break
        x <- accelerator.relax(x, x_tilde)               # Aitken / IQN-ILS
    commit converged fluid fields and body state
    accelerator.finalize_timestep()

Restoring the fluid every sweep is essential: each coupling iteration must
re-solve the *same* time step from the *same* initial fluid state, varying
only the imposed interface motion.  Without it you would be iterating a
moving target and the quasi-Newton secants would be meaningless.

The driver is written against the small :class:`RigidBodyCoupling`
protocol below so it can be unit-tested with a mock fluid/body (see
``test_strong_coupling.py``) and bound to the real solver via
:class:`FluidSolverAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from fsi_coupling import CouplingAccelerator, IQNILS


# ======================================================================
#  Protocols the driver is written against
# ======================================================================
@runtime_checkable
class RigidBodyCoupling(Protocol):
    """A rigid body that can be (a) frozen to a candidate end-of-step
    state for the fluid to see, and (b) implicitly integrated under loads
    without committing.

    The *coupling vector* is a flat array parametrising the interface;
    for a free rigid body ``[com_pos(3), lin_vel(3), ang_vel(3)]`` is the
    natural choice (orientation can be folded in for tumbling bodies).
    """

    def get_state(self) -> np.ndarray:
        """Return the committed start-of-step coupling vector."""

    def set_coupling_state(self, x: np.ndarray) -> None:
        """Impose candidate end-of-step kinematics, then refresh the SDF
        and no-slip body-velocity fields the fluid solver reads."""

    def predict(self, state_n: np.ndarray, force, torque, dt: float) -> np.ndarray:
        """Newton–Euler integrate from ``state_n`` under ``(force, torque)``
        over ``dt`` and return the resulting coupling vector *without*
        mutating committed state."""

    def commit(self, x: np.ndarray) -> None:
        """Accept ``x`` as the new committed state for the next step."""


@runtime_checkable
class FluidStepper(Protocol):
    """The minimal fluid interface the driver needs."""

    def snapshot(self):
        """Return an opaque checkpoint of the mutable fluid state."""

    def restore(self, ckpt) -> None:
        """Reset the fluid to a checkpoint produced by :meth:`snapshot`."""

    def advance_and_read_loads(self, iteration: int, t: float, dt: float):
        """Advance the fluid one step (advect→project→correct), compute the
        loads on the body, and return ``(force, torque)`` as flat arrays."""


# ======================================================================
#  The driver
# ======================================================================
@dataclass
class StrongCoupledFSI:
    fluid: FluidStepper
    body: RigidBodyCoupling
    accelerator: CouplingAccelerator = field(default_factory=lambda: IQNILS(reuse=2))
    tol: float = 1e-6
    max_iter: int = 30
    # relative tolerance scale: residual is compared against tol*(1+||x||)
    relative: bool = True

    # diagnostics from the last step
    last_iters: int = field(default=0, init=False)
    last_residual: float = field(default=0.0, init=False)
    iters_history: list = field(default_factory=list, init=False)

    def step(self, iteration: int, t: float, dt: float) -> bool:
        """Advance one implicitly-coupled time step.

        Returns ``True`` if the coupling converged, ``False`` otherwise.
        """
        fluid, body, acc = self.fluid, self.body, self.accelerator

        ckpt = fluid.snapshot()
        state_n = np.asarray(body.get_state(), dtype=np.float64)

        # First guess for the end-of-step state: same as start of step.
        # (A constant or linear extrapolation could be plugged in here.)
        x = state_n.copy()
        x_tilde = x
        converged = False

        for k in range(1, self.max_iter + 1):
            # Re-solve the SAME step from the SAME fluid state every sweep.
            fluid.restore(ckpt)
            body.set_coupling_state(x)

            force, torque = fluid.advance_and_read_loads(iteration, t, dt)
            x_tilde = body.predict(state_n, force, torque, dt)

            res = acc.residual_norm(x, x_tilde)
            scale = (1.0 + np.linalg.norm(x)) if self.relative else 1.0
            self.last_iters, self.last_residual = k, res
            if res < self.tol * scale:
                converged = True
                break
            if not np.isfinite(res):
                break
            x = np.asarray(acc.relax(x, x_tilde), dtype=np.float64)

        # Commit the converged interface state; the fluid is already at the
        # last (converged) solve because we restored+advanced inside the
        # loop and did not restore again afterwards.
        body.commit(x_tilde)
        acc.finalize_timestep()
        self.iters_history.append(self.last_iters)
        return converged


# ======================================================================
#  Adapter that binds the driver to the real FluidSolver
# ======================================================================
class FluidSolverAdapter:
    """Wrap a :class:`FluidSolver` so :class:`StrongCoupledFSI` can drive it.

    This reproduces the body-facing half of :meth:`FluidSolver.step_`
    (``solver.py``) but (1) without committing and (2) returning the loads
    instead of pushing them through ``apply_force_feedback`` — the driver
    owns the structure integration so it can sub-iterate.
    """

    def __init__(self, fluid_solver):
        self.fs = fluid_solver
        self.ndim = fluid_solver.ndim

    # -- checkpoint / restore the mutable fluid state ------------------
    def snapshot(self):
        fs = self.fs
        fields = {"u0": fs.u0.clone(), "v0": fs.v0.clone(), "p0": fs.p0.clone()}
        if self.ndim == 3:
            fields["w0"] = fs.w0.clone()
        return fields

    def restore(self, ckpt):
        fs = self.fs
        fs.u0 = ckpt["u0"].clone()
        fs.v0 = ckpt["v0"].clone()
        fs.p0 = ckpt["p0"].clone()
        if self.ndim == 3:
            fs.w0 = ckpt["w0"].clone()

    # -- one fluid sweep + load read-out -------------------------------
    def advance_and_read_loads(self, iteration, t, dt):
        import torch

        fs = self.fs
        # The body's SDF / no-slip velocity fields were just refreshed by
        # body.set_coupling_state(); refresh BDIM weighting if needed.
        if not fs._use_kernels:
            fs._recompute_mu_normals()

        if self.ndim == 2:
            u, v, p = fs.fluid_step(fs.u0, fs.v0, fs.p0, dt)
            fs.u0, fs.v0, fs.p0 = u, v, p
            if fs.force_method == "lagrangian":
                fs.forces_lagrangian_2d(u, v, p, iteration)
            else:
                fs.forces_method2(u, v, p, iteration)
            fx = fs.friction_force_lin_x + fs.pressure_force_x
            fy = fs.friction_force_lin_y + fs.pressure_force_y
            tz = fs.friction_force_ang_z + fs.pressure_force_ang_z
            force = torch.stack((fx, fy)).sum(dim=-1).detach().cpu().numpy().ravel()
            torque = np.atleast_1d(float(tz.sum()))
        else:
            u, v, w, p = fs.fluid_step(fs.u0, fs.v0, fs.w0, fs.p0, dt)
            fs.u0, fs.v0, fs.w0, fs.p0 = u, v, w, p
            if fs.force_method == "lagrangian":
                fs.forces_lagrangian_3d(u, v, w, p, iteration)
            else:
                fs.forces_method2_3d(u, v, w, p, iteration)
            fx = fs.friction_force_lin_x + fs.pressure_force_x
            fy = fs.friction_force_lin_y + fs.pressure_force_y
            fz = fs.friction_force_lin_z + fs.pressure_force_z
            tx = fs.friction_force_ang_x + fs.pressure_force_ang_x
            ty = fs.friction_force_ang_y + fs.pressure_force_ang_y
            tz = fs.friction_force_ang_z + fs.pressure_force_ang_z
            force = torch.stack((fx, fy, fz)).sum(dim=-1).detach().cpu().numpy().ravel()
            torque = torch.stack((tx, ty, tz)).sum(dim=-1).detach().cpu().numpy().ravel()
        return force, torque


# ----------------------------------------------------------------------
#  Note on the FARMS/MuJoCo path
# ----------------------------------------------------------------------
# For the coupled FARMS path (BDIMhandler), the structure integrator is
# MuJoCo, advanced by the FARMS task pipeline *outside* this driver.  To
# sub-iterate there you would, inside one coupling iteration:
#   1. mj_saveState  (checkpoint MuJoCo) once at step start,
#   2. restore fluid + MuJoCo each sweep,
#   3. write the candidate loads to xfrc_applied and call mj_step1/mj_step2
#      (or mj_forward + integrator) to get the predicted body pose,
#   4. feed that pose back into BDIMhandler.update() and re-solve the fluid.
# The accelerator and convergence logic are identical; only snapshot/
# restore and the structure-predict call change.  Recommended next step
# once this standalone prototype is validated on the sphere-drop case.
