"""Tests + runnable demo for the FSI coupling accelerators.

Run as a test::

    pytest lilytorch/integration/test_fsi_coupling.py

Run as a demo (prints the added-mass instability table)::

    python lilytorch/integration/test_fsi_coupling.py

The added-mass model problem below is the standard partitioned-FSI
stability benchmark (Causin–Gerbeau–Nobile / Förster–Wall–Ramm): a
spring–mass structure whose only fluid load is the *added mass* of the
incompressible fluid it must accelerate.  The composed fixed-point map has
gain ``~ -m_added / m_struct``, so the explicit scheme diverges as soon as
``m_added > m_struct`` — exactly what happens to a light/neutrally-buoyant
solid in water.  This is a faithful, dependency-free proxy for the
blow-ups seen in the sphere-drop and swimmer cases.
"""

from __future__ import annotations

import numpy as np

from fsi_coupling import (
    AitkenRelaxation,
    ConstantUnderRelaxation,
    IQNILS,
    make_accelerator,
)


# ======================================================================
#  Added-mass model problem
# ======================================================================
class AddedMassFSI:
    """One-DOF spring-mass driven only by incompressible added mass.

    Structure (implicit, Newmark-like with a=(s-2 s_n+s_nm1)/dt^2)::
        m a + k s = f   ->   S(f) = [f + m/dt^2 (2 s_n - s_nm1)] / (m/dt^2 + k)

    Fluid (reduced incompressible load)::
        F(s) = -m_added a(s) - c_added v(s)

    Coupling variable x = s (interface displacement).  One sweep
    H(s) = S(F(s)); the explicit gain is dH/ds ~ -m_added/m for small k,
    so the plain iteration diverges once m_added/m > 1.
    """

    def __init__(self, m=1.0, k=10.0, m_added=5.0, c_added=0.5, dt=1e-2):
        self.m, self.k, self.m_added, self.c_added, self.dt = m, k, m_added, c_added, dt
        self.s_n = 1.0      # current displacement (start displaced)
        self.s_nm1 = 1.0    # previous step

    def fluid(self, s):
        dt = self.dt
        a = (s - 2 * self.s_n + self.s_nm1) / dt**2
        v = (s - self.s_n) / dt
        return -self.m_added * a - self.c_added * v

    def structure(self, f):
        dt = self.dt
        c = self.m / dt**2
        return (f + c * (2 * self.s_n - self.s_nm1)) / (c + self.k)

    def sweep(self, s):
        """One composed solver sweep: x -> x_tilde."""
        return self.structure(self.fluid(s))

    def exact_step(self):
        """Closed-form converged displacement for the current step."""
        dt = self.dt
        # H(s) = alpha*s + beta ;  s* = beta/(1-alpha)
        s0 = self.sweep(0.0)
        s1 = self.sweep(1.0)
        alpha = s1 - s0
        beta = s0
        return beta / (1.0 - alpha)

    def advance(self, s_star):
        self.s_nm1 = self.s_n
        self.s_n = s_star


def _solve_step(model, acc, tol=1e-10, max_iter=200):
    """Iterate one coupled time step with accelerator ``acc``.

    Returns (converged_displacement, n_iterations, diverged?).
    """
    x = np.array([model.s_n], dtype=np.float64)  # initial guess = last step
    for it in range(1, max_iter + 1):
        x_tilde = np.array([model.sweep(float(x[0]))])
        res = acc.residual_norm(x, x_tilde)
        if res < tol:
            acc.finalize_timestep()
            return float(x_tilde[0]), it, False
        if not np.isfinite(res) or res > 1e12:
            return float(x_tilde[0]), it, True
        x = acc.relax(x, x_tilde)
    return float(x[0]), max_iter, True


# ----------------------------------------------------------------------
#  Tests
# ----------------------------------------------------------------------
def test_explicit_diverges_under_added_mass():
    """The plain explicit push (omega=1) must blow up when m_added > m."""
    model = AddedMassFSI(m=1.0, m_added=5.0, k=10.0, c_added=0.0)
    _, _, diverged = _solve_step(model, ConstantUnderRelaxation(omega=1.0))
    assert diverged, "explicit coupling should diverge at m_added/m = 5"


def test_aitken_converges_under_added_mass():
    model = AddedMassFSI(m=1.0, m_added=5.0, k=10.0, c_added=0.5)
    s, n, diverged = _solve_step(model, AitkenRelaxation(omega_init=0.1))
    assert not diverged
    assert abs(s - model.exact_step()) < 1e-8
    assert n < 60


def test_iqnils_converges_fast_under_added_mass():
    model = AddedMassFSI(m=1.0, m_added=5.0, k=10.0, c_added=0.5)
    s, n, diverged = _solve_step(model, IQNILS(omega_init=0.1))
    assert not diverged
    assert abs(s - model.exact_step()) < 1e-8
    # A scalar problem is a 1-secant Newton step -> converges almost instantly.
    assert n <= 3


def test_iqnils_handles_extreme_mass_ratio():
    """m_added/m = 100: hopeless for explicit, trivial for quasi-Newton."""
    model = AddedMassFSI(m=1.0, m_added=100.0, k=5.0, c_added=0.2)
    s, n, diverged = _solve_step(model, IQNILS(omega_init=0.05))
    assert not diverged
    assert abs(s - model.exact_step()) < 1e-8


def test_iqnils_multistep_with_reuse():
    """Run several steps; reuse should keep iteration counts low."""
    model = AddedMassFSI(m=1.0, m_added=8.0, k=20.0, c_added=0.5, dt=5e-3)
    acc = IQNILS(omega_init=0.1, reuse=3)
    counts = []
    for _ in range(20):
        s_star, n, diverged = _solve_step(model, acc)
        assert not diverged
        assert abs(s_star - model.exact_step()) < 1e-7
        model.advance(s_star)
        counts.append(n)
    assert max(counts) < 20


def test_iqnils_solves_divergent_linear_system():
    """Vector check: IQN-ILS solves a fixed point whose Picard map is
    expansive (spectral radius > 1), where plain iteration cannot."""
    rng = np.random.default_rng(0)
    n = 6
    G = rng.standard_normal((n, n))
    G *= 1.8 / max(np.abs(np.linalg.eigvals(G)))   # spectral radius 1.8 > 1
    b = rng.standard_normal(n)
    x_exact = np.linalg.solve(np.eye(n) - G, b)     # fixed point of x=Gx+b

    acc = IQNILS(omega_init=0.3)
    x = np.zeros(n)
    converged = False
    for _ in range(200):
        x_tilde = G @ x + b
        if np.linalg.norm(x_tilde - x) < 1e-10:
            converged = True
            break
        x = acc.relax(x, x_tilde)
    assert converged
    assert np.allclose(x, x_exact, atol=1e-7)


def test_make_accelerator_factory():
    assert isinstance(make_accelerator("iqn-ils", reuse=2), IQNILS)
    assert isinstance(make_accelerator("aitken"), AitkenRelaxation)
    assert isinstance(make_accelerator("constant", omega=0.4), ConstantUnderRelaxation)


# ----------------------------------------------------------------------
#  Demo
# ----------------------------------------------------------------------
def _demo():
    print("\nAdded-mass partitioned-FSI stability demo")
    print("(spring-mass in incompressible fluid; gain ~ -m_added/m_struct)\n")
    header = f"{'m_added/m':>10} | {'explicit w=1':>14} | {'constant w=0.5':>15} | {'Aitken':>10} | {'IQN-ILS':>10}"
    print(header)
    print("-" * len(header))
    for ratio in (0.5, 2.0, 5.0, 20.0, 100.0):
        cells = [f"{ratio:>10.1f}"]
        for acc in (
            ConstantUnderRelaxation(omega=1.0),
            ConstantUnderRelaxation(omega=0.5),
            AitkenRelaxation(omega_init=0.1),
            IQNILS(omega_init=0.1),
        ):
            model = AddedMassFSI(m=1.0, m_added=ratio, k=10.0, c_added=0.5)
            _, n, diverged = _solve_step(model, acc)
            cells.append(f"{'DIVERGED':>14}" if diverged else f"{n:>5} iters     ")
        # widths differ per column; just join with separators
        print(f"{cells[0]} | {cells[1]:>14} | {cells[2]:>15} | {cells[3]:>10} | {cells[4]:>10}")
    print("\n('iters' = coupling iterations to reach residual 1e-10 in one time step)")


if __name__ == "__main__":
    _demo()
