"""Ad-hoc 3-D jellyfish body for lilytorch's FluidSolver.

The jellyfish geometry is the analytical SDF used in WaterLily-jl's
``examples/ThreeD_Jelly.jl``: a thin spherical shell of mean radius ``R``
and half-thickness ``t``, intersected with the half-space ``z >= h`` so
that only the upper "bell" remains::

    sdf_sphere(x) = | |x|  − R | − t
    sdf_plane (x) = x_3 − h
    sdf_body  (x) = max(sdf_sphere, −sdf_plane)          (CSG "sphere ∖ plane")

WaterLily animates the jellyfish with prescribed, time-varying *maps*
``x_body = A(t) · x_lab + B(t) + C(t)`` for the sphere and
``x_body = x_lab + C(t)`` for the plane, with

    ω     = 2 U / R                                       (pulse freq.)
    A(t)  = (1 − cos(ωt)/10,  1 − cos(ωt)/10,  1)         (lateral pulse)
    B(t)  = (0, 0, (cos(ωt) − 1) R/4 − h)                 (bell-tip shift)
    C(t)  = (0, 0, sin(ωt) R/4)                           (common shift)

The ``cos(ωt)/10`` factor makes the bell periodically contract/expand in
the horizontal plane (the "pulse"), while ``B + C`` shifts the bell
vertically so that the effective propulsion is in the -z direction — the
full 6-coordinate motion of a free-swimming jellyfish (translation +
orientation + bell deformation).

Because this is a prescribed-kinematics body, no separate rigid-body
integrator is required: the state is a closed-form function of time, and
the *only* ad-hoc solver needed is the update routine below that feeds
the FluidSolver with (sdf_val, body_u, body_v, body_w) on every step.
MuJoCo is not involved at all — this is what "run the SDF updating as
WaterLily" means.

This module is self-contained: it does not touch any lilytorch source
outside ``farms_examples/jellyfish/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from lilytorch.src.body import Body


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class JellyfishParams:
    """Geometric / kinematic parameters of the jellyfish.

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
    # Additional constant offset of the sphere centre in the lab frame.
    # The bell opens toward -z, so placing the sphere centre high in the
    # domain makes the jellyfish swim downward under its own propulsion.
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def omega(self) -> float:
        return 2.0 * self.U / self.R


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


class JellyfishBody(Body):
    """Single-body composite that exposes the WaterLily jellyfish on the
    MAC staggered grid.

    The class is plugged into :class:`lilytorch.src.solver.FluidSolver`
    by replacing ``solver.composite_body`` *after* the solver has been
    constructed — this is the pattern the solver's ``step_`` already
    supports (it only calls ``composite_body.update`` and reads a
    well-defined set of attributes).

    Attributes filled on every :meth:`update` call
    ---------------------------------------------
    sdf_val, sdf_val_u, sdf_val_v, sdf_val_w
        Signed distance on the cell-centred grid and on each MAC
        staggered grid.
    body_u, body_v, body_w
        Eulerian material velocity of the body at every grid point,
        i.e. the lab-frame velocity of the body point that currently
        occupies that location.  Needed by BDIM to impose the no-slip
        condition.
    bodies
        One-element list exposing ``self`` so that downstream code that
        iterates over composite sub-bodies (e.g. force reporting) keeps
        working.
    """

    def __init__(self, device, x, y, z, eps=0.05, grids=None,
                 params: JellyfishParams | None = None):
        super().__init__(device, x, y, z=z, eps=eps, grids=grids)
        assert self.ndim == 3, "JellyfishBody is 3-D only"

        self.params = params if params is not None else JellyfishParams()

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
            [[self.params.x0, self.params.y0, self.params.z0]],
            device=device, dtype=self.dtype,
        )

        # Prime SDF / body-velocity fields so that the solver's initial
        # set-up (_recompute_mu_normals, plotting) finds valid tensors.
        self.update(torch.tensor(0.0, device=device, dtype=self.dtype), 0)

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
                       Ax, Az, dAx, dAz, dBz, dCz):
        p = self.params
        # Local coordinates relative to the sphere centre
        rx = X - p.x0
        ry = Y - p.y0
        rz = Z - p.z0
        # Note: Ay = Ax by construction.
        u = -(dAx * rx) / Ax
        v = -(dAx * ry) / Ax
        w = -(dAz * rz + dBz + dCz) / Az
        return u, v, w

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

        p = self.params
        Ax, Az, Bz, Cz, dAx, dAz, dBz, dCz = self._kinematics(t)

        # Helper: evaluate the combined SDF on an arbitrary (X,Y,Z)
        # lab-frame grid.  The sphere and plane maps both shift z by Cz
        # (C(t) is common), but the sphere additionally rescales laterally
        # by A(t) and translates by Bz in z.
        def _sdf_on(X, Y, Z):
            rx = X - p.x0
            ry = Y - p.y0
            rz = Z - p.z0
            # Sphere-frame coords: X_b = A · r_lab + B + C
            xb = Ax * rx
            yb = Ax * ry
            zb = Az * rz + Bz + Cz
            # Plane-frame z: z_p = z_lab + C_z  (map from WaterLily,
            # "x .+ C(t)").  The plane SDF is (h − z_p).
            zp = rz + Cz
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
        bu, _, _ = self._body_velocity(
            self.Xu_stag, self.Yu_stag, self.Zu_stag,
            Ax, Az, dAx, dAz, dBz, dCz)
        _, bv, _ = self._body_velocity(
            self.Xv_stag, self.Yv_stag, self.Zv_stag,
            Ax, Az, dAx, dAz, dBz, dCz)
        _, _, bw = self._body_velocity(
            self.Xw_stag, self.Yw_stag, self.Zw_stag,
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

        # Track COM position (useful for plotting / logging).  The
        # sphere-centre's lab-frame position is the fixed offset c0
        # minus the inverse image of (B+C) through A; since A is
        # purely a scale on (x,y) and 1 on z, the lab-frame centre in
        # z moves by -(Bz + Cz).
        com_z = p.z0 - float(Bz + Cz) / float(Az)
        com_xyz = torch.tensor(
            [p.x0, p.y0, com_z], device=device, dtype=dtype,
        )
        # Shape (nbodies, ndim) — used by solver.inside() / check_termination
        self.com_pos = com_xyz.unsqueeze(0)
        # 1-D shape (ndim,) — used by forces_method2_3d per sub-body
        self._sub_body.com_pos = com_xyz
