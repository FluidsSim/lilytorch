"""Flow diagnostics — energy, enstrophy, divergence, CFL monitoring.

Extracted from :mod:`lilytorch.src.solver` to keep that module focused on
the solver itself.  :class:`FlowDiagnostics` is a lightweight, dependency-free
monitor that the solver instantiates (when ``diagnostics_every > 0``) and
calls once per ``check_every`` steps from ``FluidSolver.finalize_step``.
"""

import logging
import os
import warnings

import h5py
import torch

logger = logging.getLogger(__name__)


class FlowDiagnostics:
    """Lightweight monitor for kinetic energy, enstrophy, max-divergence,
    and CFL number.  Records scalar time-series and optionally warns when
    energy grows beyond a user-specified factor of its initial value.

    Parameters
    ----------
    nt : int
        Total number of time steps (for pre-allocation).
    ndim : int
        Spatial dimension (2 or 3).
    h : float or Tensor
        Uniform grid spacing.
    device, dtype
        Torch device / dtype for the record arrays.
    check_every : int
        Diagnostics are computed every *check_every* steps.  1 = every step.
    energy_growth_factor : float
        Issue a warning when E_k exceeds *energy_growth_factor* × E_k(0).
        Set to ``None`` or ``inf`` to disable the energy blow-up check.
    """

    def __init__(self, nt, ndim, h, device, dtype,
                 check_every=1, energy_growth_factor=10.0):
        self.nt    = nt
        self.ndim  = ndim
        self.h     = float(h)
        self.hd    = self.h ** ndim          # cell volume  h^d
        self.device = device
        self.dtype  = dtype

        self.check_every = max(1, int(check_every))
        self.energy_growth_factor = energy_growth_factor

        # Pre-allocated record arrays (filled with NaN so uncomputed slots
        # are visually obvious when plotted).
        self.kinetic_energy = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.enstrophy      = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.max_divergence  = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.cfl_number      = torch.full((nt,), float('nan'), device=device, dtype=dtype)

        self._ek0 = None   # E_k at the first computed step (baseline)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, iteration, u, v, p, dt, nu, divergence_fn, vorticity_fn,
               w=None):
        """Compute and record diagnostics for the current step.

        Parameters
        ----------
        iteration : int
            Current time-step index.
        u, v : Tensor
            Velocity components.  *w* is ``None`` in 2-D.
        p : Tensor
            Pressure (unused for now; kept for future pressure-energy).
        dt : float or Tensor
            Time step size (for CFL).
        nu : float or Tensor
            Kinematic viscosity (for CFL).
        divergence_fn : callable(u, v, w=None) -> Tensor
            Solver's divergence method.
        vorticity_fn  : callable(u, v, w=None) -> Tensor
            Solver's vorticity method (returns scalar in 2-D,
            magnitude in 3-D).
        w : Tensor or None
            z-velocity component (3-D only).
        """
        if iteration % self.check_every != 0:
            return

        h  = self.h
        hd = self.hd
        dt_val = float(dt)
        nu_val = float(nu)

        # ---- kinetic energy  E_k = 0.5 * h^d * Σ(u² + v² [+ w²]) ----
        ke = u.square() + v.square()
        if w is not None:
            ke = ke + w.square()
        ek = 0.5 * hd * ke.sum()
        self.kinetic_energy[iteration] = ek

        # ---- enstrophy  Z = 0.5 * h^d * Σ ω² ----
        omega = vorticity_fn(u, v, w)
        enst = 0.5 * hd * omega.square().sum()
        self.enstrophy[iteration] = enst

        # ---- max |div(u)| ----
        div = divergence_fn(u, v, w=w)
        self.max_divergence[iteration] = div.abs().max()

        # ---- CFL = u_max * dt / h ----
        vel_max = u.abs().max()
        vel_max = max(vel_max, v.abs().max())
        if w is not None:
            vel_max = max(vel_max, w.abs().max())
        self.cfl_number[iteration] = float(vel_max) * dt_val / h

        # ---- energy blow-up warning ----
        if self._ek0 is None:
            self._ek0 = float(ek) if float(ek) > 0 else 1.0
        if (self.energy_growth_factor is not None
                and float(ek) > self.energy_growth_factor * self._ek0):
            warnings.warn(
                f"[FlowDiagnostics] E_k = {float(ek):.6e} at iter {iteration} "
                f"exceeds {self.energy_growth_factor}x initial "
                f"({self._ek0:.6e}).  Possible blow-up.",
                RuntimeWarning, stacklevel=2,
            )

        # ---- CFL warning ----
        cfl_val = float(self.cfl_number[iteration])
        if cfl_val > 0.5:
            warnings.warn(
                f"[FlowDiagnostics] CFL = {cfl_val:.3f} > 0.5 at iter {iteration}",
                RuntimeWarning, stacklevel=2,
            )

    # ------------------------------------------------------------------
    def save_h5(self, path, lock):
        """Write diagnostics to ``<path>/diagnostics.h5``."""
        h5_path = os.path.join(path, "diagnostics.h5")
        data = {
            "kinetic_energy": self.kinetic_energy.cpu().numpy().copy(),
            "enstrophy":      self.enstrophy.cpu().numpy().copy(),
            "max_divergence":  self.max_divergence.cpu().numpy().copy(),
            "cfl_number":      self.cfl_number.cpu().numpy().copy(),
        }
        with lock:
            with h5py.File(h5_path, "w") as f:
                for name, arr in data.items():
                    f.create_dataset(name, data=arr)
        logger.info("Saved flow diagnostics to %s", h5_path)
