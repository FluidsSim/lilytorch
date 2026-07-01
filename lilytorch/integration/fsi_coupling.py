"""Strong (implicit) fluid–structure coupling accelerators.

Background
----------
Lilytorch currently couples the fluid to immersed bodies with a *weakly
partitioned* (explicit) scheme: the fluid is advanced once, the loads are
read off, and they are pushed onto the body integrator (FARMS/MuJoCo, or
the standalone ``apply_force_feedback`` path).  The only stabiliser is a
constant temporal low-pass on the force (``force_relaxation`` in
:mod:`BDIMhandler`).

Explicit coupling is provably unstable when the added (displaced-fluid)
mass is comparable to or larger than the body mass — i.e. *exactly* the
regime of a solid body in water (``rho_body ~ rho_fluid``).  A constant
under-relaxation factor only buys a little margin and smears the true
transient.  This is the root cause of the sphere-drop blow-ups and of the
case-by-case ``force_relaxation`` tuning.

This module implements the **preCICE family of interface accelerators**
that turn the per-step explicit push into a *strongly coupled*
fixed-point iteration that converges regardless of the mass ratio:

* :class:`ConstantUnderRelaxation` — baseline (what we have today).
* :class:`AitkenRelaxation`        — Irons–Tuck adaptive scalar relaxation.
* :class:`IQNILS`                  — Interface Quasi-Newton with Inverse
  Least Squares (Degroote et al. 2009), the preCICE workhorse.

The "coupling variable" ``x`` is whatever vector parametrises the
interface — for a rigid body the natural choice is its end-of-step
kinematic state ``[com_pos(3), lin_vel(3), ang_vel(3), ...]``.  Because a
rigid body has only a handful of DOFs, the linear algebra is tiny and we
keep everything in ``float64`` NumPy; there is no benefit to GPU tensors
here.

Fixed-point formulation
-----------------------
Let ``H`` be the composed solver: impose interface state ``x`` in the
fluid, solve the fluid, read the loads, integrate the structure over
``dt`` to obtain a new state ``x̃ = H(x)``.  We seek the fixed point
``x* = H(x*)``, equivalently the root of the residual ``r(x) = H(x) - x``.

Each accelerator takes the pair ``(x, x̃)`` produced by one solver sweep
and returns the next input ``x``.  When the residual norm drops below the
tolerance the step is converged; call :meth:`finalize_timestep` and move
on.

References
----------
J. Degroote, K.-J. Bathe, J. Vierendeels, *Performance of a new
partitioned procedure versus a monolithic procedure in fluid–structure
interaction*, Computers & Structures 87 (2009) 793–801.

B. Uekermann et al., *The preCICE coupling library* (quasi-Newton
acceleration), https://precice.org/couple-your-code-configuration-acceleration.html
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "CouplingAccelerator",
    "ConstantUnderRelaxation",
    "AitkenRelaxation",
    "IQNILS",
    "make_accelerator",
]


def _as_vec(x) -> np.ndarray:
    """Flatten anything array-like into a contiguous float64 1-D vector."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).reshape(-1)


class CouplingAccelerator:
    """Common interface for interface coupling accelerators.

    A single coupling iteration is::

        x_next = acc.relax(x, x_tilde)

    where ``x`` is the input that was imposed on the fluid this sweep and
    ``x_tilde`` is the structure state that came back out.  When the step
    has converged, call :meth:`finalize_timestep` exactly once.
    """

    def relax(self, x, x_tilde) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def residual_norm(self, x, x_tilde) -> float:
        """L2 norm of the interface residual ``x_tilde - x``."""
        return float(np.linalg.norm(_as_vec(x_tilde) - _as_vec(x)))

    def finalize_timestep(self) -> None:
        """Hook called once when the current time step has converged."""

    def reset(self) -> None:
        """Forget all history (e.g. when restarting a simulation)."""


@dataclass
class ConstantUnderRelaxation(CouplingAccelerator):
    """``x_{k+1} = x_k + omega * (x_tilde_k - x_k)``.

    The classic staggered relaxation.  ``omega == 1`` reproduces the plain
    explicit (Gauss–Seidel) push that the codebase uses today.  It is here
    as the baseline to beat — it diverges once the added-mass ratio is
    large enough that the fixed-point map is expansive.
    """

    omega: float = 0.5

    def relax(self, x, x_tilde) -> np.ndarray:
        x = _as_vec(x)
        x_tilde = _as_vec(x_tilde)
        return x + self.omega * (x_tilde - x)


@dataclass
class AitkenRelaxation(CouplingAccelerator):
    """Irons–Tuck adaptive under-relaxation (preCICE ``aitken``).

    The relaxation factor is recomputed every iteration from the last two
    residuals::

        omega_k = -omega_{k-1} * <r_{k-1}, r_k - r_{k-1}> / ||r_k - r_{k-1}||^2

    It needs no per-problem tuning and is dramatically more robust than a
    fixed ``omega``, while staying a one-liner.  The last factor is reused
    as the initial guess for the next time step (a preCICE default).
    """

    omega_init: float = 0.1
    omega_max: float = 1.0

    _r_prev: np.ndarray | None = field(default=None, repr=False)
    _omega: float = field(default=0.0, repr=False)

    def relax(self, x, x_tilde) -> np.ndarray:
        x = _as_vec(x)
        x_tilde = _as_vec(x_tilde)
        r = x_tilde - x

        if self._r_prev is None:
            self._omega = self.omega_init
        else:
            dr = r - self._r_prev
            denom = float(dr @ dr)
            if denom > 1e-30:
                self._omega = -self._omega * float(self._r_prev @ dr) / denom
            # keep the magnitude sane; preserve sign
            mag = min(abs(self._omega), self.omega_max)
            self._omega = float(np.sign(self._omega) * mag) if mag > 0 else self.omega_init

        self._r_prev = r
        return x + self._omega * r

    def finalize_timestep(self) -> None:
        # Carry the converged omega into the next step; drop the residual
        # history (a new step starts a fresh secant sequence).
        self._r_prev = None

    def reset(self) -> None:
        self._r_prev = None
        self._omega = 0.0


@dataclass
class IQNILS(CouplingAccelerator):
    """Interface Quasi-Newton with Inverse Least Squares (Degroote 2009).

    This is the preCICE ``IQN-ILS`` post-processing.  Over the iterations
    of a time step it collects, for ``k >= 1``::

        V = [ Δr_k , Δr_{k-1} , ... ]   with  Δr_i = r_i - r_{i-1}
        W = [ Δx̃_k , Δx̃_{k-1} , ... ]  with  Δx̃_i = x̃_i - x̃_{i-1}

    and approximates the (inverse) interface Jacobian by the multi-secant
    relation ``W ≈ J^{-1} V``.  The next input is::

        solve   min_c || V c + r_k ||           (least squares, via QR)
        x_{k+1} = x̃_k + W c

    With enough columns this is a Newton step on ``r(x) = 0`` and converges
    super-linearly, *independently of the mass ratio* — which is precisely
    why it cures the added-mass instability.

    Parameters
    ----------
    omega_init:
        Under-relaxation used only on the very first sweep of the very
        first time step, when no secant information exists yet.
    reuse:
        Number of *previous* time steps whose ``(V, W)`` columns are reused
        to warm-start the least-squares system (preCICE ``time-windows-reused``).
        ``0`` disables reuse.
    filter_eps:
        QR1 column filter: drop a column whose orthogonalised contribution
        is below ``filter_eps`` times the column norm (controls the
        conditioning of the least-squares problem).
    """

    omega_init: float = 0.1
    reuse: int = 0
    filter_eps: float = 1e-8

    # per-time-step history
    _x_tilde_prev: np.ndarray | None = field(default=None, repr=False)
    _r_prev: np.ndarray | None = field(default=None, repr=False)
    _Vcols: list = field(default_factory=list, repr=False)
    _Wcols: list = field(default_factory=list, repr=False)
    # cross-time-step reuse (deque of (Vcols, Wcols) per past step)
    _reuse_store: deque = field(default=None, repr=False)

    def __post_init__(self):
        self._reuse_store = deque(maxlen=max(int(self.reuse), 0))

    # -- column bookkeeping --------------------------------------------
    def _all_columns(self):
        """Return (V, W) as (n, m) arrays from this step + reused steps."""
        Vcols = list(self._Vcols)
        Wcols = list(self._Wcols)
        for Vold, Wold in self._reuse_store:
            Vcols.extend(Vold)
            Wcols.extend(Wold)
        if not Vcols:
            return None, None
        # newest columns first (already inserted at front this step)
        V = np.stack(Vcols, axis=1)
        W = np.stack(Wcols, axis=1)
        return V, W

    # -- the quasi-Newton least-squares solve --------------------------
    @staticmethod
    def _qr_filter_solve(V, W, r, filter_eps):
        """Solve ``min_c ||V c + r||`` with QR + column filtering, return
        the quasi-Newton increment ``W c``.

        Columns whose orthogonal contribution falls below
        ``filter_eps * ||column||`` are dropped (QR1 filter); this keeps the
        triangular factor well conditioned when iterates become collinear.
        """
        n, m = V.shape
        keep = []
        # Modified Gram-Schmidt with on-the-fly filtering.
        Q = np.zeros((n, m), dtype=np.float64)
        Rt = np.zeros((m, m), dtype=np.float64)
        nkeep = 0
        for j in range(m):
            v = V[:, j].copy()
            col_norm = np.linalg.norm(v)
            for i in range(nkeep):
                rij = Q[:, i] @ v
                Rt[i, nkeep] = rij
                v = v - rij * Q[:, i]
            rho = np.linalg.norm(v)
            if col_norm > 0 and rho <= filter_eps * col_norm:
                # linearly dependent on already-kept columns -> drop it
                continue
            Q[:, nkeep] = v / rho if rho > 0 else v
            Rt[nkeep, nkeep] = rho
            keep.append(j)
            nkeep += 1

        if nkeep == 0:
            return None  # no usable secant information

        Qk = Q[:, :nkeep]
        Rk = Rt[:nkeep, :nkeep]
        # Solve R c = -Q^T r  (least squares normal form, upper-triangular).
        rhs = -(Qk.T @ r)
        c = np.linalg.solve(Rk, rhs)
        Wk = W[:, keep]
        return Wk @ c

    # -- public API ----------------------------------------------------
    def relax(self, x, x_tilde) -> np.ndarray:
        x = _as_vec(x)
        x_tilde = _as_vec(x_tilde)
        r = x_tilde - x

        # Append the new secant pair (needs a previous iterate this step).
        if self._x_tilde_prev is not None:
            self._Vcols.insert(0, r - self._r_prev)
            self._Wcols.insert(0, x_tilde - self._x_tilde_prev)

        self._x_tilde_prev = x_tilde
        self._r_prev = r

        V, W = self._all_columns()
        if V is None:
            # No secant data at all (very first sweep, no reuse): relax.
            return x + self.omega_init * r

        increment = self._qr_filter_solve(V, W, r, self.filter_eps)
        if increment is None:
            return x + self.omega_init * r
        return x_tilde + increment

    def finalize_timestep(self) -> None:
        # Push this step's columns into the reuse store, then clear the
        # per-step history so the next step starts a fresh secant sequence.
        if self._reuse_store.maxlen and self._Vcols:
            self._reuse_store.appendleft((list(self._Vcols), list(self._Wcols)))
        self._Vcols = []
        self._Wcols = []
        self._x_tilde_prev = None
        self._r_prev = None

    def reset(self) -> None:
        self._Vcols = []
        self._Wcols = []
        self._x_tilde_prev = None
        self._r_prev = None
        self._reuse_store.clear()


def make_accelerator(name: str, **kwargs) -> CouplingAccelerator:
    """Factory so configs can select an accelerator by string.

    >>> make_accelerator("iqn-ils", reuse=2)            # doctest: +ELLIPSIS
    IQNILS(...)
    """
    key = name.strip().lower().replace("_", "-")
    if key in ("constant", "fixed", "under-relaxation"):
        return ConstantUnderRelaxation(**kwargs)
    if key in ("aitken",):
        return AitkenRelaxation(**kwargs)
    if key in ("iqn-ils", "iqnils", "quasi-newton", "qn"):
        return IQNILS(**kwargs)
    raise ValueError(f"unknown coupling accelerator {name!r}")
