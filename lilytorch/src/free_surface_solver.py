"""One-fluid free-surface solver (ghost-fluid-style), SEPARATE from the
two-phase VOF model — :class:`~lilytorch.src.two_phase_solver.TwoPhaseSolver`
is left untouched.

Motivation
----------
The two-phase VOF solver represents the air as a *real* light fluid. For a thin
surface-swimming body the air develops coupling-driven noisy motion whose
pressure leaks spurious force into the body-force band integral (see the
air-vs-water force decomposition diagnostic). A one-fluid free surface removes
this at the source: the **air is not a fluid**, it is a constant-pressure void.

Method
------
Only the water is governed by Navier-Stokes. The air is imposed as
``p = p_atm = 0`` (a Dirichlet free-surface boundary condition), which the
multigrid/MGCG Poisson solver already supports through its ``dirichlet_mask``
(it pins masked cells to zero and restricts the mask across coarse levels).
Concretely this solver:

  * uses a **uniform water density** (constant-coefficient projection) instead
    of the 833:1 variable-density blend — so there is no density-jump
    instability and no air momentum;
  * pins ``p = 0`` in the air cells (``alpha < 0.5``) every projection, so the
    water carries the correct hydrostatic pressure (``p=0`` at the surface,
    ``rho_w g`` below) and **buoyancy emerges from the free-surface BC** rather
    than from a density jump;
  * **extends the water velocity into the air** (harmonic) so the void has no
    independent dynamics and the interface advects smoothly;
  * turns OFF the gauge anchor (the ``p=0`` BC already pins the pressure datum)
    and the air-transparent-body blend (no light air to be transparent to).

It REUSES the two-phase machinery for interface tracking (the Weymouth-Yue VOF
``alpha``), gravity, BDIM body coupling, and the eulerian force integral — only
the projection / air treatment changes. Because the air pressure is a smooth
``p=0`` (not a noisy real-air field), the body-force band integral is clean.

Status: validated incrementally — see ``validation/free_surface/`` (hydrostatic
first). Not a drop-in for the production eel until validated there.
"""

import torch

from lilytorch.src.two_phase_solver import TwoPhaseSolver


class FreeSurfaceSolver(TwoPhaseSolver):

    def __init__(self, pars, dtype=None, custom_update=None, compute_forces=True):
        super().__init__(pars, dtype=dtype, custom_update=custom_update,
                         compute_forces=compute_forces)

        # ---- one-fluid: collapse the phase contrast to a single (water) fluid.
        # recip_density_cc/face then return 1/rho_water uniformly => the
        # variable-density projection degenerates to a constant-coefficient one.
        self.two_phase.rho_air = self.two_phase.rho_water
        self.two_phase.nu_air  = self.two_phase.nu_water

        # The p=0 free-surface BC pins the pressure datum; the gauge anchor and
        # the air-transparent blend are two-phase devices that must be off here.
        self._gauge_anchor_forces = False
        self._air_transparent_body = False

        fs = pars["solver"].get("free_surface", {})
        # number of harmonic sweeps used to extend the water velocity into the
        # air void each step (covers the near-surface band; ~5-15 is plenty).
        self._fs_extend_iters = int(fs.get("extend_iters", 10))
        self._fs_air_mask_cache = None
        self._wrap_poisson_air_dirichlet()
        print(f"[FreeSurfaceSolver] one-fluid free surface: p=0 in air "
              f"(dirichlet mask), uniform rho={self.two_phase.rho_water}, "
              f"velocity-extend iters={self._fs_extend_iters}")

    # ------------------------------------------------------------------
    #  p = 0 in the air via the Poisson dirichlet_mask
    # ------------------------------------------------------------------
    def _air_mask_interior(self):
        """Interior-shaped boolean mask of air cells (alpha<0.5), matching the
        shape the Poisson solver pins via ``p[inner].masked_fill_(mask, 0)``."""
        a = self.two_phase.alpha
        inner = tuple(slice(1, -1) for _ in range(self.ndim))
        return (a[inner] < 0.5).contiguous()

    def _wrap_poisson_air_dirichlet(self):
        """Inject the current air mask as the Poisson ``dirichlet_mask`` right
        before each solve. ``project`` resets the mask to ``None`` at its top,
        so we set it here (after that reset, before the actual solve)."""
        ps = self.poisson_solver
        for name in ("solve_multigrid", "solve_mgcg", "solve_rmgcg"):
            orig = getattr(ps, name, None)
            if orig is None or getattr(orig, "_fs_wrapped", False):
                continue

            def wrapped(*a, _orig=orig, **k):
                ps.dirichlet_mask = self._fs_air_mask_cache
                return _orig(*a, **k)
            wrapped._fs_wrapped = True
            setattr(ps, name, wrapped)

    def project(self, *args, **kwargs):
        # Cache the air mask for the wrapped Poisson solve, then run the
        # (now constant-coefficient, since rho_air==rho_water) projection.
        self._fs_air_mask_cache = self._air_mask_interior()
        out = super().project(*args, **kwargs)
        # Hard-pin the air void to p=0: the V-cycle preconditioner pins it, but
        # CG (mgcg) iterates can re-pollute it; the air is dynamically a void, so
        # zeroing its pressure post-solve keeps the body-force band clean.
        p = out[-1]
        p[self.two_phase.alpha < 0.5] = 0.0
        return out

    # ------------------------------------------------------------------
    #  velocity extension into the air void (harmonic, holds water fixed)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extend_velocity_into_air(self, *vels):
        """Harmonically extend each velocity component from water into the air
        so the void has no independent dynamics (water cells held fixed). In
        place. A cell-centred water mask is reused for every (staggered)
        component by trimming to the component shape — adequate for the thin
        near-surface band that the interface advection samples."""
        if self._fs_extend_iters <= 0:
            return
        a = self.two_phase.alpha
        water_cc = (a >= 0.5)
        nd = self.ndim
        for c in vels:
            if c is None:
                continue
            sl = tuple(slice(0, s) for s in c.shape)
            w = water_cc[sl]
            if w.shape != c.shape:
                continue
            held = c.clone()
            for _ in range(self._fs_extend_iters):
                s = torch.zeros_like(c)
                for d in range(nd):
                    s = s + torch.roll(c, 1, d) + torch.roll(c, -1, d)
                s = s / (2 * nd)
                c[...] = torch.where(w, held, s)

    def finalize_step(self, u, v, p, iteration, w_vel=None):
        # Slave the air velocity to the water before the interface (VOF)
        # transport in the base finalize_step.
        self._extend_velocity_into_air(u, v, w_vel)
        return super().finalize_step(u, v, p, iteration, w_vel=w_vel)
