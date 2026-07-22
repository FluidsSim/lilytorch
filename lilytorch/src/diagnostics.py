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

from lilytorch.src import operations as ops

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
        # Viscous dissipation *rate* (per unit density): nu * h^d * Σ(mu0·|S|²),
        # |S| = sqrt(2 S_ij S_ij).  Multiply by rho for watts, integrate over
        # time for the energy irreversibly lost to the fluid.  mu0 masks the
        # solid interior so the BDIM band does not contaminate the integral.
        self.dissipation_rate = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        # Unmasked counterpart: ν·h^d·Σ|S|² integrated over the *whole* grid
        # (body interior + transition band included).  Captures any boundary-layer
        # dissipation that the mu0 mask clips; equals dissipation_rate when no
        # mask is applied.  Useful as an upper bound for the energy-balance check.
        self.dissipation_rate_unmasked = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        # Pressure-Poisson convergence.  ``poisson_residual`` is the L-inf
        # residual the solve stopped at; compare it against poisson_tol to see
        # whether the solve converged or simply ran out its cycle cap.
        # ``poisson_iters`` is the iteration count (CG iterations, or V-cycles
        # for standalone multigrid).  It is only informative when poisson_tol
        # >= 0: a negative tol disables the early-exit test so the solve stays
        # host-sync-free and graph-capturable, and the loop then always runs
        # its full cycle cap.
        self.poisson_residual = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.poisson_iters    = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        # max|div u| restricted to cells that are essentially pure fluid
        # (mu0 > 0.99).  See the note where this is filled in update().
        self.max_divergence_fluid = torch.full((nt,), float('nan'), device=device, dtype=dtype)

        self._ek0 = None   # E_k at the first computed step (baseline)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, iteration, u, v, p, dt, nu, divergence_fn, vorticity_fn,
               w=None, sdf_cc=None, mu_fn=None, poisson_solver=None):
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
        sdf_cc : Tensor or None
            Cell-centred union signed-distance field (>0 in fluid).  When given
            together with *mu_fn*, the viscous-dissipation integral is masked to
            the fluid region so the BDIM transition band inside the body does
            not contaminate it.  ``None`` ⇒ dissipation integrated over the
            whole grid (body interior included).
        mu_fn : callable(d) -> (mu0, mu1) or None
            Smooth-Heaviside builder (the body's ``mu_funcs``); ``mu0`` is the
            fluid fraction (≈1 in fluid, ≈0 inside the body).
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

        # ---- viscous dissipation rate  ε = ν * h^d * Σ(mu0·|S|²) ----
        # |S| = sqrt(2 S_ij S_ij), so |S|² = 2 S_ij S_ij and the dissipation
        # density is Φ = 2μ S:S = μ|S|² = ρν|S|².  We store the per-density rate
        # ν·h^d·Σ(mu0·|S|²) (parallel to the rho-free kinetic_energy); multiply
        # by rho downstream for watts.
        vel = (u, v) if w is None else (u, v, w)
        smag = ops.strain_rate_magnitude(vel, h, self.ndim)   # |S| on CC grid
        phi = smag.square()                                   # |S|² = 2 S:S
        # Unmasked dissipation over the whole grid (always recorded).
        self.dissipation_rate_unmasked[iteration] = nu_val * hd * phi.sum()
        phi_masked = phi
        if sdf_cc is not None and mu_fn is not None:
            try:
                mu0_cc, _ = mu_fn(sdf_cc)
                if mu0_cc.shape == phi.shape:
                    phi_masked = phi * mu0_cc
            except Exception:
                pass
        self.dissipation_rate[iteration] = nu_val * hd * phi_masked.sum()

        # ---- max |div(u)| ----
        div = divergence_fn(u, v, w=w)
        self.max_divergence[iteration] = div.abs().max()

        # Same maximum taken over the fluid only.  With a BDIM body the
        # projection enforces div(u) = (1-mu0)*div(u_body), not div(u) = 0, so
        # cells inside/straddling the body are legitimately non-solenoidal --
        # and with overlapping links (convexify) that term is large.  The
        # whole-grid maximum above is therefore dominated by the body and says
        # little about solver health; this one is the number to watch.
        if sdf_cc is not None and mu_fn is not None:
            try:
                mu0_div, _ = mu_fn(sdf_cc)
                if mu0_div.shape == div.shape:
                    self.max_divergence_fluid[iteration] = (
                        div.abs()[mu0_div > 0.99].max()
                    )
            except Exception:
                pass

        # ---- CFL = u_max * dt / h ----
        vel_max = u.abs().max()
        vel_max = max(vel_max, v.abs().max())
        if w is not None:
            vel_max = max(vel_max, w.abs().max())
        self.cfl_number[iteration] = float(vel_max) * dt_val / h

        # ---- pressure-Poisson convergence ----
        if poisson_solver is not None:
            resid = getattr(poisson_solver, "_last_resid", None)
            if resid is not None:
                self.poisson_residual[iteration] = resid
            self.poisson_iters[iteration] = getattr(poisson_solver, "_last_niter", -1)

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
            "dissipation_rate": self.dissipation_rate.cpu().numpy().copy(),
            "dissipation_rate_unmasked": self.dissipation_rate_unmasked.cpu().numpy().copy(),
            "poisson_residual": self.poisson_residual.cpu().numpy().copy(),
            "poisson_iters":    self.poisson_iters.cpu().numpy().copy(),
            "max_divergence_fluid": self.max_divergence_fluid.cpu().numpy().copy(),
        }
        with lock:
            with h5py.File(h5_path, "w") as f:
                for name, arr in data.items():
                    f.create_dataset(name, data=arr)
        logger.info("Saved flow diagnostics to %s", h5_path)
