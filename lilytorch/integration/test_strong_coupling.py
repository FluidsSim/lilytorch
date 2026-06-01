"""End-to-end test of the StrongCoupledFSI driver with a mock fluid/body.

We model a 1-DOF body (vertical position + velocity) falling/settling in a
dense fluid.  The "fluid" returns a load that is dominated by added mass
(``F = -m_added * a_body``) plus buoyancy and drag — the same physics that
destabilises explicit coupling for a solid in water.  The mock exercises
the real driver code path: snapshot/restore each sweep, candidate-state
imposition, structure prediction, accelerator, convergence, commit.
"""

from __future__ import annotations

import numpy as np

from fsi_coupling import ConstantUnderRelaxation, IQNILS
from strong_coupling import StrongCoupledFSI


class MockFluid:
    """Fluid whose load on the body is added-mass + drag + buoyancy.

    Coupling vector x = [pos, vel].  The added-mass load depends on the
    *imposed* body acceleration a = (vel - vel_n)/dt, exactly the term that
    makes the partitioned map expansive when m_added > m_struct.
    """

    def __init__(self, m_added=4.0, drag=0.8, buoyancy=3.0, dt=1e-2):
        self.m_added, self.drag, self.buoyancy, self.dt = m_added, drag, buoyancy, dt
        self._state_n = np.array([0.0, 0.0])     # committed start-of-step
        self._imposed = np.array([0.0, 0.0])     # candidate state to solve at
        self.n_solves = 0

    # checkpoint here is trivial (no field history) but must round-trip.
    def snapshot(self):
        return self._state_n.copy()

    def restore(self, ckpt):
        self._state_n = ckpt.copy()

    def set_imposed(self, x):
        self._imposed = np.asarray(x, dtype=np.float64)

    def advance_and_read_loads(self, iteration, t, dt):
        self.n_solves += 1
        vel = self._imposed[1]
        vel_n = self._state_n[1]
        a = (vel - vel_n) / dt
        f = -self.m_added * a - self.drag * vel + self.buoyancy
        return np.array([f]), np.array([0.0])

    def finalize(self, iteration):
        # once-per-step fluid tail; nothing to do for the mock
        self.n_finalize = getattr(self, "n_finalize", 0) + 1


class MockBody:
    """1-DOF Newton-Euler body coupled through the mock fluid.

    Structure: m a + 0 = f + gravity.  State = [pos, vel].
    """

    def __init__(self, fluid: MockFluid, mass=1.0, gravity=-9.81):
        self.fluid = fluid
        self.mass, self.gravity = mass, gravity
        self.state = np.array([0.0, 0.0])   # committed [pos, vel]

    def get_state(self):
        return self.state.copy()

    def set_coupling_state(self, x):
        # tell the fluid which candidate end-of-step kinematics to solve at
        self.fluid.set_imposed(x)

    def predict(self, state_n, force, torque, dt):
        f = float(force[0]) + self.mass * self.gravity
        a = f / self.mass
        vel = state_n[1] + dt * a
        pos = state_n[0] + dt * vel
        return np.array([pos, vel])

    def commit(self, x):
        self.state = np.asarray(x, dtype=np.float64).copy()


def _run(accelerator, n_steps=50, **fluid_kw):
    dt = fluid_kw.pop("dt", 1e-2)
    fluid = MockFluid(dt=dt, **fluid_kw)
    body = MockBody(fluid)
    drv = StrongCoupledFSI(fluid=fluid, body=body, accelerator=accelerator,
                           tol=1e-9, max_iter=80)
    traj = []
    for it in range(n_steps):
        fluid._state_n = body.state.copy()   # sync committed state into fluid
        ok = drv.step(it, it * dt, dt)
        if not ok:
            return body, traj, False
        traj.append(body.state.copy())
    return body, traj, True


def test_driver_protocol_conformance():
    fluid = MockFluid()
    body = MockBody(fluid)
    from strong_coupling import FluidStepper, RigidBodyCoupling
    assert isinstance(fluid, FluidStepper)
    assert isinstance(body, RigidBodyCoupling)


def test_driver_converges_with_iqnils_heavy_added_mass():
    body, traj, ok = _run(IQNILS(omega_init=0.1, reuse=2), m_added=4.0)
    assert ok
    # body should be moving (settling under gravity+buoyancy), finite state
    assert np.all(np.isfinite(body.state))
    assert len(traj) == 50


def test_driver_explicit_diverges_heavy_added_mass():
    # omega=1 explicit push -> the added-mass instability should blow it up
    body, traj, ok = _run(ConstantUnderRelaxation(omega=1.0), m_added=4.0)
    assert not ok


def test_checkpoint_restore_is_exercised_each_sweep():
    """Every coupling sweep must re-solve from the same start state."""
    fluid = MockFluid(m_added=4.0)
    body = MockBody(fluid)
    drv = StrongCoupledFSI(fluid=fluid, body=body,
                           accelerator=IQNILS(omega_init=0.1),
                           tol=1e-9, max_iter=80)
    fluid._state_n = body.state.copy()
    drv.step(0, 0.0, 1e-2)
    # more than one fluid solve means we actually sub-iterated
    assert fluid.n_solves > 1
    # converged residual recorded
    assert drv.last_residual < 1e-8


def test_finalize_called_once_per_step():
    """The once-per-step fluid tail must run exactly once per step,
    regardless of how many coupling sweeps were needed."""
    fluid = MockFluid(m_added=4.0)
    body = MockBody(fluid)
    drv = StrongCoupledFSI(fluid=fluid, body=body,
                           accelerator=IQNILS(omega_init=0.1),
                           tol=1e-9, max_iter=80)
    n_steps = 5
    for it in range(n_steps):
        fluid._state_n = body.state.copy()
        drv.step(it, it * 1e-2, 1e-2)
    assert fluid.n_solves > n_steps          # sub-iterated
    assert getattr(fluid, "n_finalize", 0) == n_steps  # tail once per step


def test_iqnils_terminal_velocity_matches_analytic():
    """Steady state: drag balances net weight -> v_term = (m g + buoy)/drag.

    (At steady state added-mass term vanishes since a -> 0.)"""
    m, g, drag, buoy = 1.0, -9.81, 0.8, 3.0
    # relaxation time (m+m_added)/drag = 6.25 s; integrate ~25 time constants
    body, traj, ok = _run(IQNILS(omega_init=0.1, reuse=3),
                          n_steps=32000, m_added=4.0, drag=drag,
                          buoyancy=buoy, dt=5e-3)
    assert ok
    v_term = (m * g + buoy) / drag
    assert abs(body.state[1] - v_term) < 1e-3
