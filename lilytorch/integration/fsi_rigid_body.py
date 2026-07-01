"""A free rigid 2-D circle for end-to-end strong-coupling validation.

This is the minimal solver-compatible *free* body needed to drive
:class:`strong_coupling.StrongCoupledFSI` against a real
:class:`~lilytorch.src.solver.FluidSolver`.  It reuses the proven
``BodyAnalytical`` / staggered-grid machinery (the same path
``flow_past_circle_2d`` uses) but replaces the prescribed motion with a
free rigid state ``[x, y, vx, vy]`` driven by the coupling loads.

Pieces
------
* :class:`RigidCircle2D`     — circle SDF; position set from the *candidate*
  coupling state via an overridden ``rototranslate_points`` (velocity comes
  out of the motion-map slope through the existing autograd path).
* :class:`SingleBodyComposite` — wraps one body so ``FluidSolver`` sees the
  usual ``composite_body`` interface, and refreshes ``com_pos`` each step.
* :class:`RigidCircleCoupling` — the :class:`RigidBodyCoupling` adapter:
  ``get_state`` / ``set_coupling_state`` / ``predict`` / ``commit``.
* :func:`build_rigid_circle_fsi` — assembles a ``FluidSolver`` + driver.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lilytorch.src.body import BodyAnalytical, Body, rotate_grid_2d


class RigidCircle2D(BodyAnalytical):
    """Circle whose pose is set externally (the coupling candidate state).

    ``rototranslate_points`` is overridden to place the body at the stored
    ``_pos`` / ``_rot``; the inherited ``update`` still derives the no-slip
    body velocity from the motion maps, which we set to constant-velocity
    lines so their autograd slope equals the stored linear velocity.
    """

    def __init__(self, device, x, y, radius, pos0, vel0, eps=0.05):
        # Candidate kinematics (read by the overridden rototranslate_points
        # and by the motion maps); set BEFORE super().__init__ so the maps,
        # which close over self, are valid immediately.
        self._pos = torch.tensor(pos0, device=device, dtype=x.dtype)
        self._rot = torch.eye(2, device=device, dtype=x.dtype)
        self._vx = float(vel0[0])
        self._vy = float(vel0[1])

        radius = float(radius)
        sdf = lambda X, Y: torch.sqrt(X * X + Y * Y) - radius
        # Velocity maps: slope = (vx, vy); only their d/dt is used.
        maps = (
            lambda tt: 0.0 * tt,                       # theta(t): no spin
            [lambda tt: self._vx * tt, lambda tt: self._vy * tt],
        )
        # pre_update=False: the composite drives the first update with grids.
        super().__init__(device, x, y, sdf, maps, eps=eps, pre_update=False)

    def rototranslate_points(self, t):
        # Place the body at the candidate position; orientation fixed.
        self.com_pos = self._pos
        return (self._pos, self._rot)


class SingleBodyComposite(Body):
    """Minimal ``composite_body`` wrapping a single free body.

    Mirrors ``CompositeBodyAnalytical.update`` for one body, and (unlike it)
    refreshes ``com_pos`` each step so the torque integration uses the
    current centre of mass.
    """

    def __init__(self, device, x, y, body):
        super().__init__(device, x, y, eps=body.eps)
        self._setup_grids()
        self.bodies = [body]
        self.nbodies = 1
        self.mu_funcs = body.mu_funcs
        self.com_pos = torch.zeros((1, self.ndim), device=device, dtype=self.dtype)
        self.update(torch.tensor(0.0, device=device, dtype=self.dtype), 0)

    def update(self, t, iteration, dt=1):
        b = self.bodies[0]
        b.update(t, iteration, dt=dt, grids=self._grids)
        self.sdf_val   = b.sdf_val
        self.sdf_val_u = b.sdf_u
        self.sdf_val_v = b.sdf_v
        self.body_u    = b.body_u
        self.body_v    = b.body_v
        self.com_pos[0, 0] = b.com_pos[0]
        self.com_pos[0, 1] = b.com_pos[1]


class RigidCircleCoupling:
    """:class:`strong_coupling.RigidBodyCoupling` for the free circle.

    Coupling vector ``x = [px, py, vx, vy]``.  The structure model is a
    point mass under the net fluid force plus a constant external force
    ``f_ext`` (e.g. net weight − buoyancy, or a prescribed push).
    """

    def __init__(self, composite: SingleBodyComposite, mass: float,
                 f_ext=(0.0, 0.0)):
        self.comp = composite
        self.body = composite.bodies[0]
        self.mass = float(mass)
        self.f_ext = np.asarray(f_ext, dtype=np.float64)
        # committed state
        self.pos = self.body._pos.detach().cpu().numpy().astype(np.float64)
        self.vel = np.array([self.body._vx, self.body._vy], dtype=np.float64)

    def get_state(self):
        return np.concatenate([self.pos, self.vel])

    def set_coupling_state(self, x):
        x = np.asarray(x, dtype=np.float64)
        dev, dt_ = self.body.device, self.body.dtype
        self.body._pos = torch.tensor(x[:2], device=dev, dtype=dt_)
        self.body._vx = float(x[2])
        self.body._vy = float(x[3])

    def predict(self, state_n, force, torque, dt):
        # net fluid force on the single body (sum over bodies if (B,2))
        f = np.asarray(force, dtype=np.float64).reshape(-1, 2).sum(axis=0)
        a = (f + self.f_ext) / self.mass
        v = state_n[2:] + dt * a
        p = state_n[:2] + dt * v
        return np.concatenate([p, v])

    def commit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.pos = x[:2].copy()
        self.vel = x[2:].copy()
        self.set_coupling_state(x)


def build_rigid_circle_fsi(pars, radius, pos0, vel0, mass, f_ext=(0.0, 0.0),
                           accelerator=None, tol=1e-5, max_iter=30):
    """Build a ``FluidSolver`` with an injected free circle + a driver.

    ``pars`` is a solver config dict (as from ``yaml2pyobject``); it should
    select ``solver_method='python'`` and ``compute_forces`` is forced on.
    Returns ``(driver, coupling, fluid_solver)``.
    """
    from lilytorch.src.solver import FluidSolver
    from strong_coupling import StrongCoupledFSI, FluidSolverAdapter
    from fsi_coupling import IQNILS

    fs = FluidSolver(pars, compute_forces=True)

    body = RigidCircle2D(fs.device, fs.x, fs.y, radius, pos0, vel0,
                         eps=pars["solver"].get("eps", 0.05))
    comp = SingleBodyComposite(fs.device, fs.x, fs.y, body)

    # Inject our free body in place of the yaml-built one (single body, so
    # n_bodies / force records already match).
    fs.composite_body = comp
    fs.n_bodies = 1

    coupling = RigidCircleCoupling(comp, mass=mass, f_ext=f_ext)
    if accelerator is None:
        accelerator = IQNILS(omega_init=0.1, reuse=2)

    driver = StrongCoupledFSI(
        fluid=FluidSolverAdapter(fs),
        body=coupling,
        accelerator=accelerator,
        tol=tol,
        max_iter=max_iter,
    )
    return driver, coupling, fs
