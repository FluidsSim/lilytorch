"""Force-and-torque computation for immersed bodies.

Extracted from ``solver.py`` (item #8 of the HIGH PRIORITY backlog).

The eulerian readout (``forces_method2`` / ``forces_method2_3d``) is
native-only: it runs the streaming-SDF force op and returns.  Standalone
python bodies are no longer supported (use the FARMS/BDIMhandler path).
The lagrangian readout keeps its module-level helpers (``_viscous_stress_tensor``,
``_marker_aabb_slab``) so they can be wrapped by :func:`torch.compile`.

The ``forces_method*`` functions take ``self`` (a ``FluidSolver``) as their
first argument and are bound to ``FluidSolver`` as methods at the bottom
of ``solver.py`` -- they are NOT directly callable from this module's
public API.  This keeps the public API of :class:`FluidSolver` unchanged
while moving ~1500 LoC of force code out of ``solver.py``.
"""
from __future__ import annotations

import math

import torch

# The Lagrangian surface-integral force readout is a native CUDA / C++ op
# (``native.lagrangian_forces_{2,3}d``) with an ``at::parallel_for`` CPU twin.
from lilytorch.src.native import (
    lagrangian_forces_2d as _lagrangian_forces_2d_kernel,
    lagrangian_forces_3d as _lagrangian_forces_3d_kernel,
    streaming_sdf_forces_post_2d,
    streaming_sdf_forces_post_3d,
)


# ======================================================================
# Streaming metadata for solver-driven (handler-less) analytical bodies
# ======================================================================
#
# The eulerian force kernels read their per-body descriptors from
# ``comp._kernel_static_{2,3}d`` / ``comp._kernel_step``, which the BDIM handler
# packs every step from the FARMS pose buffers.  A ``composite_analytical`` body
# driven straight from the solver (the two-phase validation scripts, the force
# oracles) never goes through the handler, so those attributes were absent and
# the force functions used to ``return`` silently -- leaving every load at
# exactly zero with no warning.  That is what broke
# ``validation/two_phase_2d/run_floating_cylinder.py`` (F_y = 0, 100 % error),
# which is the case the "floating cylinder 0.91x Archimedes" claim rests on.
#
# Analytical bodies can supply everything the kernels need: the SDF is a
# callable, ``local_aabb`` bounds it, and ``BodyAnalytical.update`` now caches
# the pose (``_pose_rot`` / ``_pose_transl``) and velocities.  So build the same
# packed layout here instead of bailing out.  The handler path is untouched: if
# it has already populated the attributes, these helpers never run.


def _analytical_stream_static(self, comp, D):
    """Pack the static per-body descriptors (``F_flat`` / ``F_offsets`` /
    ``body_shapes`` / ``body_meta``) for analytical bodies.

    Mirrors ``BDIMhandler._stream_static_pack``; the per-body local SDF template
    is sampled from the callable on a grid derived from ``local_aabb``, exactly
    as ``BDIMhandler._build_stream_meta`` does for its analytical branch.
    """
    attr = '_kernel_static_3d' if D == 3 else '_kernel_static_2d'
    cached = getattr(comp, attr, None)
    if cached is not None:
        return cached
    h = float(self.h)
    anames = ('x', 'y', 'z')[:D]
    for body in comp.bodies:
        if getattr(body, '_stream_meta', None) is not None:
            continue
        sdf_callable = getattr(body, 'sdf', None)
        local_aabb = getattr(body, 'local_aabb', None)
        if sdf_callable is None or local_aabb is None:
            return None
        # Pad the template box by the full BDIM band before sampling.
        # ``BodyAnalytical.local_aabb`` is fitted TIGHT to the surface contour
        # (body.py builds it from cnt min/max with no margin), but the kernel
        # queries the template at every cell of the force AABB, which reaches
        # eps + 4h beyond the surface.  Unpadded, those queries clamp to the
        # box edge and read a body-local SDF that is far too small, so the
        # partition of unity drops the cell and its share of the force is
        # simply lost.  The handler's mesh branch pads for the same reason
        # (``local_aabb_lo = cnt.min - (eps + 4h)``).
        pad = float(comp.eps) + 4.0 * h
        lo = local_aabb[0].cpu() - pad
        hi = local_aabb[1].cpu() + pad
        # h/2 sampling, matching the handler's analytical branch.
        sizes = [max(2, int(round(float(hi[i] - lo[i]) / (0.5 * h))) + 1)
                 for i in range(D)]
        axes = tuple(
            torch.linspace(float(lo[i]), float(hi[i]), sizes[i],
                           dtype=self.dtype, device=self.device)
            for i in range(D)
        )
        with torch.no_grad():
            F = sdf_callable(*torch.meshgrid(*axes, indexing='ij')).contiguous()
        inv = [1.0 / float(a[1] - a[0]) if a.numel() > 1 else 1.0 for a in axes]
        meta = {'F': F}
        for i, a in enumerate(anames):
            meta[f'b{a}0'] = float(axes[i][0])
            meta[f'b{a}_last'] = float(axes[i][-1])
            meta[f'inv_d{a}'] = inv[i]
        meta['inv_vol'] = float(np_prod(inv))
        body._stream_meta = meta

    F_chunks, F_off, shapes, meta_rows = [], [0], [], []
    for body in comp.bodies:
        m = body._stream_meta
        F_chunks.append(m['F'].flatten())
        F_off.append(F_off[-1] + m['F'].numel())
        shapes.append([int(s) for s in m['F'].shape])
        meta_rows.append(
            [m[f'b{a}0'] for a in anames]
            + [m[f'b{a}_last'] for a in anames]
            + [m[f'inv_d{a}'] for a in anames]
            + [m['inv_vol']]
        )
    F_flat = torch.cat(F_chunks).contiguous()
    for b, body in enumerate(comp.bodies):
        body._stream_meta['F'] = F_flat[F_off[b]:F_off[b + 1]].view(*shapes[b])
    sm = {
        'F_flat': F_flat,
        'F_offsets': torch.tensor(F_off, dtype=torch.int64, device=self.device),
        'body_shapes': torch.tensor(shapes, dtype=torch.int64, device=self.device),
        'body_meta': torch.tensor(meta_rows, dtype=self.dtype, device=self.device),
    }
    setattr(comp, attr, sm)
    return sm


def np_prod(vals):
    out = 1.0
    for v in vals:
        out *= v
    return out


def _analytical_stream_step(self, comp, D):
    """Pack the per-step ``kin`` / AABB tensors for analytical bodies.

    ``kin`` row layout is fixed by the kernels and must match
    ``BDIMhandler._update_streaming_multi`` exactly::

        [R^T (D*D) | body_pos (D) | com_pos (D) | lin_vel (D) | ang (3|1)]
    """
    B = len(comp.bodies)
    anames = ('x', 'y', 'z')[:D]
    g_1d = tuple(getattr(comp, f'g{a}_1d', None) for a in anames)
    if any(g is None for g in g_1d):
        g_1d = tuple(getattr(comp, a) for a in anames)
    gs = tuple(int(g.numel()) for g in g_1d)

    wRT = D * D
    ang_w = 3 if D == 3 else 1
    kin = torch.zeros((B, wRT + 3 * D + ang_w), dtype=self.dtype,
                      device=self.device)
    i_lo = torch.zeros((B, D), dtype=torch.int64)
    i_hi = torch.zeros((B, D), dtype=torch.int64)
    h = float(self.h)
    band = float(comp.eps) + 4.0 * h

    for b, body in enumerate(comp.bodies):
        rot = getattr(body, '_pose_rot', None)
        transl = getattr(body, '_pose_transl', None)
        if rot is None or transl is None:
            return None
        R = rot.to(dtype=self.dtype, device=self.device)
        kin[b, 0:wRT] = R.T.reshape(-1)
        pos = torch.as_tensor([float(transl[i]) for i in range(D)],
                              dtype=self.dtype, device=self.device)
        o = wRT
        kin[b, o:o + D] = pos; o += D
        com = getattr(body, 'com_pos', None)
        kin[b, o:o + D] = (com[:D].to(self.dtype) if com is not None
                           and com.numel() >= D else pos)
        o += D
        lv = getattr(body, '_pose_lin_vel', None)
        if lv is not None:
            for i in range(min(D, len(lv))):
                kin[b, o + i] = float(lv[i])
        o += D
        av = getattr(body, '_pose_ang_vel', None)
        if av is not None and ang_w == 1:
            kin[b, o] = float(av)

        # World AABB: local box corners rotated into world, plus the BDIM band.
        la = getattr(body, 'local_aabb', None)
        lo_l = la[0].to(self.dtype).cpu()
        hi_l = la[1].to(self.dtype).cpu()
        corners = torch.cartesian_prod(
            *[torch.stack([lo_l[i], hi_l[i]]) for i in range(D)])
        corners = corners.reshape(-1, D).to(self.dtype)
        world = (R.cpu() @ corners.T).T + pos.cpu()
        wlo = world.min(dim=0).values - band
        whi = world.max(dim=0).values + band
        for i in range(D):
            g = g_1d[i]
            g0 = float(g[0])
            i_lo[b, i] = int(max(0, math.floor((float(wlo[i]) - g0) / h)))
            i_hi[b, i] = int(min(gs[i], math.ceil((float(whi[i]) - g0) / h) + 1))
    i_hi = torch.maximum(i_hi, i_lo)
    dims = i_hi - i_lo
    kstep = {
        'kin': kin,
        'aabb_lo': i_lo.to(self.device),
        'aabb_dim': dims.to(self.device),
        'max_vol': int(dims.prod(dim=1).max()) if B > 0 else 0,
    }
    for i, a in enumerate(anames):
        kstep[f'g{a}'] = g_1d[i]
    comp._kernel_step = kstep
    return kstep


def _require_stream_meta(self, comp, D):
    """Return ``(static, step)`` for the eulerian force kernels, building them
    for handler-less analytical bodies. Raises rather than silently returning
    zero loads (the lagrangian path raises on its own missing contract too)."""
    attr = '_kernel_static_3d' if D == 3 else '_kernel_static_2d'
    step = getattr(comp, '_kernel_step', None)
    static = getattr(comp, attr, None)
    if step is not None and static is not None:
        return static, step
    static = _analytical_stream_static(self, comp, D)
    step = _analytical_stream_step(self, comp, D) if static is not None else None
    if static is None or step is None:
        raise RuntimeError(
            f"force_method='eulerian' needs per-body streaming metadata "
            f"({attr} / _kernel_step). The BDIM handler supplies it for mesh "
            "bodies; analytical bodies need a callable `sdf` and `local_aabb`, "
            "and must have been update()d at least once so the pose is cached. "
            f"Body types present: "
            f"{[type(b).__name__ for b in comp.bodies]}. "
            "Use force_method='lagrangian' if the body cannot provide these."
        )
    return static, step


# ======================================================================
# Methods bound to FluidSolver (take ``self`` as first argument).
# ======================================================================


def forces_method2(self, u, v, p, iteration):
    comp = self.composite_body
    B = len(comp.bodies)

    sm, _stream_step = _require_stream_meta(self, comp, 2)

    out2d = getattr(self, '_kernel_post_out_buf_2d', None)
    _fresh_out = out2d is None or out2d.shape != (B, 6)
    if _fresh_out:
        out2d = torch.zeros((B, 6), dtype=torch.float64, device=self.device)
        self._kernel_post_out_buf_2d = out2d

    if self.use_variable_viscosity:
        nu_rho_field = self._compute_nu_rho_for_forces(u, v)
    else:
        nu_rho_scalar = getattr(self, '_kernel_post_nu_rho_scalar_2d', None)
        if nu_rho_scalar is None:
            nu_rho_scalar = torch.empty(
                (1,), device=self.device, dtype=self.dtype,
            )
            # nu/rho are constant for the run (variable viscosity takes the
            # branch above): fill ONCE — float(gpu_tensor) is a device sync.
            nu_rho_scalar.fill_(self._cached_float('nu', self.nu)
                                * self._cached_float('rho', self.rho))
            self._kernel_post_nu_rho_scalar_2d = nu_rho_scalar
        nu_rho_field = nu_rho_scalar

    # Scalar kernel params must be python floats: passing a 0-d GPU tensor
    # forces a host-side conversion per launch (= a hidden device sync).
    # eps is the single BDIM half-width authority (solver.eps = eps_multiplier * h).
    eps_body = self._cached_float('eps', comp.eps)
    interp_method = int(getattr(self, '_sdf_interp_method', 0))
    # 0 = union ndelta (default), 2 = per-body analytic normal (sm2).
    _fsm = int(getattr(self, 'force_submethod', 0))
    # Both readouts split the union force to links by the body-velocity blend
    # partition, so the ph_tau slot carries that blend width (in metres).
    _ph_tau = (float(getattr(self, '_body_vel_blend_cells', 0.0))
               * self._cached_float('h', self.h))

    if not _fresh_out:
        out2d.zero_()
    streaming_sdf_forces_post_2d(
        sm['F_flat'], sm['F_offsets'],
        sm['body_shapes'], sm['body_meta'], _stream_step['kin'],
        _stream_step['aabb_lo'], _stream_step['aabb_dim'],
        _stream_step['gx'], _stream_step['gy'],
        self._cached_float('h', self.h), _stream_step['max_vol'],
        comp.sdf_val,
        interp_method,
        u.contiguous(), v.contiguous(), p.contiguous(),
        nu_rho_field,
        eps_body,
        self._cached_float('off_p', self.eul_sample_offset_pressure),
        self._cached_float('off_f', self.eul_sample_offset_friction),
        self._cached_float('h2', self.h2),
        self.force_delta_order,
        out2d,
        _fsm, _ph_tau,
    )

    out_s = out2d if out2d.dtype == u.dtype else out2d.to(u.dtype)
    self.viscous_drag_record[:B, 0, iteration]  = out_s[:, 0]
    self.viscous_drag_record[:B, 1, iteration]  = out_s[:, 1]
    self.pressure_drag_record[:B, 0, iteration] = out_s[:, 3]
    self.pressure_drag_record[:B, 1, iteration] = out_s[:, 4]
    self.friction_force_lin_x = self.viscous_drag_record[:B, 0, iteration]
    self.friction_force_lin_y = self.viscous_drag_record[:B, 1, iteration]
    self.friction_force_ang_z = out_s[:, 2].clone()
    self.pressure_force_x     = self.pressure_drag_record[:B, 0, iteration]
    self.pressure_force_y     = self.pressure_drag_record[:B, 1, iteration]
    self.pressure_force_ang_z = out_s[:, 5].clone()
    self.xstress_tensor = None
    self.ystress_tensor = None
    self.pforce_x = None
    self.pforce_y = None
    return


# ==================================================================
# 3-D force computation  (volume-integral with smoothed delta)
# ==================================================================

def forces_method2_3d(self, u, v, w, p, iteration):
    """Compute viscous and pressure forces/torques on each body in 3-D.

    Uses the same smoothed-delta volume-integration approach as the 2-D
    ``forces_method2`` but extended to three dimensions:

    Viscous force::

    F_visc = ∫ (σ · n) δ_ε(d - ε) dV

    where σ_{ij} = ν ρ (∂u_i/∂x_j + ∂u_j/∂x_i) is the viscous stress.

    Pressure force::

    F_pres = -∫ p n δ_ε(d) dV

    Torques are computed about each body's centre of mass via r × f.
    """
    comp = self.composite_body

    # ============================================================
    # Kernel path: post-fluid-step native force pass.  The update stage
    # only maintains union geometry/winning density; forces are computed
    # here from the current fluid fields and current union normals, with
    # per-body deltas obtained by on-demand body-local SDF sampling.
    # ============================================================
    _stream_static, _stream_step = _require_stream_meta(self, comp, 3)
    B = len(comp.bodies)
    out = getattr(self, '_kernel_post_out_buf_3d', None)
    _fresh_out = out is None or out.shape != (B, 12)
    if _fresh_out:
        out = torch.zeros((B, 12), dtype=torch.float64, device=self.device)
        self._kernel_post_out_buf_3d = out

    if self.use_variable_viscosity:
        nu_rho_field = self._compute_nu_rho_for_forces(u, v, w)
    else:
        nu_rho_scalar = getattr(self, '_kernel_post_nu_rho_scalar_3d', None)
        if nu_rho_scalar is None:
            nu_rho_scalar = torch.empty((1,), device=self.device, dtype=self.dtype)
            # nu/rho are constant for the run (variable viscosity takes the
            # branch above): fill ONCE — float(gpu_tensor) is a device sync.
            nu_rho_scalar.fill_(self._cached_float('nu', self.nu)
                                * self._cached_float('rho', self.rho))
            self._kernel_post_nu_rho_scalar_3d = nu_rho_scalar
        nu_rho_field = nu_rho_scalar

    # Scalar kernel params must be python floats — see the 2-D twin.
    # eps is the single BDIM half-width authority (solver.eps = eps_multiplier * h).
    eps_body = self._cached_float('eps', comp.eps)
    # 0 = union ndelta (default), 2 = per-body analytic normal (sm2).  Both
    # split the union force to links by the SAME partition of unity the
    # streaming body-velocity blend uses, so the ph_tau slot carries
    # ``body_velocity_blend_eps_cells`` (<=0 -> hard nearest-body winner).
    _fsm = int(getattr(self, 'force_submethod', 0))
    _ph_tau = (float(getattr(self, '_body_vel_blend_cells', 0.0))
               * self._cached_float('h', self.h))
    if not _fresh_out:
        out.zero_()
    streaming_sdf_forces_post_3d(
        _stream_static['F_flat'], _stream_static['F_offsets'],
        _stream_static['body_shapes'], _stream_static['body_meta'],
        _stream_step['kin'], _stream_step['aabb_lo'], _stream_step['aabb_dim'],
        _stream_step['gx'], _stream_step['gy'], _stream_step['gz'],
        self._cached_float('h', self.h), _stream_step['max_vol'],
        comp.sdf_val,
        getattr(self, '_sdf_interp_method', 0),
        u.contiguous(), v.contiguous(), w.contiguous(), p.contiguous(),
        nu_rho_field,
        eps_body,
        self._cached_float('off_p', self.eul_sample_offset_pressure),
        self._cached_float('off_f', self.eul_sample_offset_friction),
        self._cached_float('h3', self.h3),
        self.force_delta_order, out,
        _fsm, _ph_tau,
    )
    out_s = out if out.dtype == u.dtype else out.to(u.dtype)
    self.viscous_drag_record[:B, :, iteration]    = out_s[:, 0:3]
    self.viscous_torque_record[:B, :, iteration]  = out_s[:, 3:6]
    self.pressure_drag_record[:B, :, iteration]   = out_s[:, 6:9]
    self.pressure_torque_record[:B, :, iteration] = out_s[:, 9:12]
    self.friction_force_lin_x = self.viscous_drag_record[:B, 0, iteration]
    self.friction_force_lin_y = self.viscous_drag_record[:B, 1, iteration]
    self.friction_force_lin_z = self.viscous_drag_record[:B, 2, iteration]
    self.friction_force_ang_x = self.viscous_torque_record[:B, 0, iteration]
    self.friction_force_ang_y = self.viscous_torque_record[:B, 1, iteration]
    self.friction_force_ang_z = self.viscous_torque_record[:B, 2, iteration]
    self.pressure_force_x     = self.pressure_drag_record[:B, 0, iteration]
    self.pressure_force_y     = self.pressure_drag_record[:B, 1, iteration]
    self.pressure_force_z     = self.pressure_drag_record[:B, 2, iteration]
    self.pressure_force_ang_x = self.pressure_torque_record[:B, 0, iteration]
    self.pressure_force_ang_y = self.pressure_torque_record[:B, 1, iteration]
    self.pressure_force_ang_z = self.pressure_torque_record[:B, 2, iteration]
    self.xstress_tensor = None
    self.ystress_tensor = None
    self.zstress_tensor = None
    self.pforce_x = None
    self.pforce_y = None
    self.pforce_z = None
    return


# ======================================================================
# Lagrangian (surface-integral) force methods
# ======================================================================
#
# These functions implement the surface-integral form of the BDIM force
# computation introduced in Uhlmann (2005), Vanella & Balaras (2009) and
# Kempe & Fröhlich (2012):
#
#   F = ∮_∂Ω (σ·n - p n) dS
#   τ = ∮_∂Ω (r-x_com) × (σ·n - p n) dS
#
# instead of the volumetric ``Σ stress·δ_ε(φ)·h^D`` quadrature used by
# the eulerian path (``forces_method2`` / ``forces_method2_3d``).
#
# The surface is sampled at Lagrangian markers carried by each body:
#
#   * 2-D: contour points ``body.cnt_update`` (2, M) with arc-length
#     coordinate ``body.curv_coord`` (M,).  The outward unit normal at
#     each marker is computed on the fly from the local tangent
#     ``t = ∂cnt/∂s`` via ``n = (t_y, -t_x)`` (CCW convention).
#   * 3-D: triangulation with
#     ``body.tri_centroid_world`` (3, T),
#     ``body.tri_normal_world``   (3, T), and
#     ``body.tri_area``           (T,).  See ``body._build_surface_3d``.
#
# For both dims:
#   1. Compute the CC viscous-stress tensor σ_ij and the CC pressure
#      ``p`` on (a sub-block of) the Eulerian grid -- the **same shared
#      kernel** the eulerian path already builds.  Crop to the union
#      AABB when the streaming sparse-SDF layout is available, otherwise
#      use the full grid.
#   2. For each body, interpolate σ_ij and p at the Lagrangian markers
#      (bilinear / trilinear via ``RegularGridInterpolator``).
#   3. Contract σ·n at the marker and integrate over the surface
#      (trapezoidal in 2-D, simple sum in 3-D since triangle areas already
#      carry the quadrature weight).
#
# These functions are dispatched from ``solver.step_`` when
# ``solver.force_method == "lagrangian"``.  They are method-bound to
# ``FluidSolver`` in ``solver.py`` at the bottom of the class body.

def _viscous_stress_tensor(vels, h):
    """Compute σ_ij = ν⁻¹ρ⁻¹ × (∂u_i/∂x_j + ∂u_j/∂x_i) at CC.

    Returns
    -------
    stress : tensor of shape (D, D, *spatial)
        Symmetric viscous strain-rate tensor (without ν·ρ); caller
        multiplies by ν·ρ to obtain σ_ij.
    """
    D = len(vels)
    rng = range(D)

    # u_i interpolated to CC along its stagger dim i (used for cross
    # derivatives).  Diagonal entries use the compact MAC stencil.
    cc_vels = []
    for i in rng:
        u_i = vels[i]
        u_cc = torch.empty_like(u_i)
        sl_lo = [slice(None)] * D; sl_lo[i] = slice(0, -1); sl_lo = tuple(sl_lo)
        sl_hi = [slice(None)] * D; sl_hi[i] = slice(1, None); sl_hi = tuple(sl_hi)
        sl_last = [slice(None)] * D; sl_last[i] = -1; sl_last = tuple(sl_last)
        u_cc[sl_lo] = 0.5 * (u_i[sl_lo] + u_i[sl_hi])
        u_cc[sl_last] = u_i[sl_last]
        cc_vels.append(u_cc)

    grad = [[None] * D for _ in rng]
    for i in rng:
        u_i = vels[i]
        d_ii = torch.empty_like(u_i)
        sl_lo = [slice(None)] * D; sl_lo[i] = slice(0, -1); sl_lo = tuple(sl_lo)
        sl_hi = [slice(None)] * D; sl_hi[i] = slice(1, None); sl_hi = tuple(sl_hi)
        sl_last = [slice(None)] * D; sl_last[i] = -1; sl_last = tuple(sl_last)
        sl_secondlast = [slice(None)] * D; sl_secondlast[i] = -2
        sl_secondlast = tuple(sl_secondlast)
        d_ii[sl_lo] = (u_i[sl_hi] - u_i[sl_lo]) / h
        d_ii[sl_last] = d_ii[sl_secondlast]
        grad[i][i] = d_ii
        for j in rng:
            if j != i:
                grad[i][j] = torch.gradient(
                    cc_vels[i], spacing=h, dim=j, edge_order=2,
                )[0]

    # Symmetric strain-rate ε_ij = (∂u_i/∂x_j + ∂u_j/∂x_i).  For the
    # surface traction t_i = σ_ij n_j = ν·ρ·ε_ij n_j (Newtonian fluid,
    # no isotropic part — the isotropic pressure is added separately).
    eps_ij = [[None] * D for _ in rng]
    for i in rng:
        for j in rng:
            eps_ij[i][j] = grad[i][j] + grad[j][i]
    return eps_ij


def _marker_aabb_slab(coords_flat, axes, sample_offset, inv_h, halo_extra=5):
    """Grid-index slab tightly enclosing all surface markers (+ halo).

    The Lagrangian force kernels only sample the strain-rate / pressure
    fields at a body's surface markers, yet the legacy path built those
    fields over the *whole* domain every step (the dominant cost).  This
    helper returns the index window that the marker sampling actually
    touches so the caller can crop the (bandwidth-heavy) field build to a
    small slab around the union of all bodies — mirroring how the
    Eulerian streaming path crops to per-body AABBs.

    Parameters
    ----------
    coords_flat : tensor ``(D, M_total)``
        World-frame coordinates of every marker of every body, packed
        along the marker axis (``cnt_flat`` in 2-D, ``tri_centroid`` in
        3-D).
    axes : tuple of ``D`` 1-D tensors
        Cell-centred coordinate vectors (``comp.x, comp.y[, comp.z]``).
    sample_offset : float
        Largest outward normal sampling offset over all channels (world
        units); widens the slab.  Callers with independent pressure /
        friction offsets must pass ``max(|off_p|, |off_f|)`` so the slab
        still contains every query point.
    inv_h : float
        ``1 / h`` (uniform grid spacing).
    halo_extra : int
        Extra cells beyond the marker extent to keep every sampling /
        strain stencil in the slab *interior*, so cropped field values
        match the full-grid build exactly.  Covers the biquadratic
        sampler (±1), the strain stencil (±2), and slack.

    Returns
    -------
    list[tuple[int, int]] or None
        Per-dimension ``(lo, hi)`` half-open index ranges, or ``None``
        when there are no markers (caller falls back to the full grid).
    """
    D = coords_flat.shape[0]
    M = coords_flat.shape[1]
    if M == 0:
        return None
    # One host sync for the whole AABB (vs. a full-grid gradient build).
    mn = coords_flat.amin(dim=1).tolist()
    mx = coords_flat.amax(dim=1).tolist()
    halo = int(math.ceil(abs(float(sample_offset)) * inv_h)) + int(halo_extra)
    ranges = []
    for d in range(D):
        n = int(axes[d].numel())
        a0 = float(axes[d][0])
        lo = int(math.floor((mn[d] - a0) * inv_h)) - halo
        hi = int(math.ceil((mx[d] - a0) * inv_h)) + halo + 1
        if lo < 0:
            lo = 0
        if hi > n:
            hi = n
        # Too thin for the edge_order=2 strain stencil: bail to full grid.
        if hi - lo < 3:
            return None
        ranges.append((lo, hi))
    return ranges


def forces_lagrangian_2d(self, u, v, p, iteration):
    """Surface-integral viscous + pressure forces in 2-D (rigid bodies).

    Fused, multi-body native kernel path -- builds the CC strain-rate
    tensor on the full grid once per step, packs every body's contour
    into a single ``(2, M_total)`` tensor, and dispatches a single
    ``lagrangian_forces_2d`` call (CPU OpenMP or CUDA atomicAdd) that
    integrates ``∮ (ν·ρ·ε·n - p n) ds`` per body in one launch.

    Output channels match the legacy 2-D 6-channel layout:
        [fv_x, fv_y, t_v, fp_x, fp_y, t_p]
    """
    comp = self.composite_body
    B = len(comp.bodies)
    h = self.h
    inv_dx = 1.0 / h; inv_dy = 1.0 / h
    off_p = float(self.lagr_sample_offset_pressure)
    off_f = float(self.lagr_sample_offset_friction)
    # The slab must cover whichever channel reaches furthest out.
    soff_max = max(abs(off_p), abs(off_f))

    # Pack per-body contours into a single (2, M_total) tensor with int64
    # offsets.  Skip degenerate bodies (M <= 1) by emitting an empty slice
    # for them -- the kernel zeros their output row.  ``cnt_update`` is
    # already in the solver dtype/device (refreshed each step by the
    # BDIM handler), so ``.to`` is a no-op fast path here.
    cnts = []
    offs = [0]
    for body in comp.bodies:
        c = body.cnt_update
        if c is None or c.shape[1] <= 1:
            cnts.append(torch.empty(2, 0, dtype=p.dtype, device=p.device))
        else:
            cnts.append(c.to(dtype=p.dtype, device=p.device))
        offs.append(offs[-1] + cnts[-1].shape[1])
    cnt_flat    = torch.cat(cnts, dim=1) if cnts else torch.empty(2, 0, dtype=p.dtype, device=p.device)
    cnt_offsets = torch.tensor(offs, dtype=torch.int64, device=p.device)

    com_pos = torch.stack(
        [body.com_pos.to(dtype=p.dtype, device=p.device) for body in comp.bodies],
        dim=0,
    ) if B > 0 else torch.empty(0, 2, dtype=p.dtype, device=p.device)

    # ---- Crop the strain-rate / pressure build to the marker AABB -----
    # The kernel only samples the fields at the surface markers, so build
    # them on a small slab around the union of all contours instead of the
    # whole domain (the dominant cost).  Field values in the slab interior
    # match the full-grid build exactly (the halo keeps every sampling /
    # strain stencil away from the slab edge), so results are unchanged.
    nu_rho = self._compute_nu_rho_for_forces(u, v)
    slab = _marker_aabb_slab(cnt_flat, (comp.x, comp.y), soff_max, inv_dx)
    if slab is not None:
        (i0, i1), (j0, j1) = slab
        sl = (slice(i0, i1), slice(j0, j1))
        eps_ij = _viscous_stress_tensor((u[sl].contiguous(), v[sl].contiguous()), h)
        p_field = p[sl].contiguous()
        bx0 = float(comp.x[i0]); by0 = float(comp.y[j0])
        Mx = i1 - i0; My = j1 - j0
        nu_rho_crop = sl
    else:
        eps_ij = _viscous_stress_tensor((u, v), h)
        p_field = p
        bx0 = float(comp.x[0]); by0 = float(comp.y[0])
        Mx = int(comp.x.numel()); My = int(comp.y.numel())
        nu_rho_crop = None

    # nu_rho may be a Python float / scalar (constant viscosity) or a CC
    # tensor (variable viscosity).  Reuse a 1-element device buffer for
    # the scalar case to avoid a per-step allocation.
    if torch.is_tensor(nu_rho) and nu_rho.ndim == 2:
        nu_rho_field = nu_rho[nu_rho_crop].contiguous() if nu_rho_crop is not None else nu_rho
    else:
        buf = getattr(self, '_lagr_nu_rho_buf_2d', None)
        if buf is None:
            buf = torch.empty(1, dtype=p.dtype, device=p.device)
            self._lagr_nu_rho_buf_2d = buf
        buf.fill_(float(nu_rho))
        nu_rho_field = buf

    # Persistent (B, 6) float64 output buffer (zeroed by the kernel).
    out = getattr(self, '_lagr_out_buf_2d', None)
    if out is None or out.shape[0] != B:
        out = torch.zeros((B, 6), dtype=torch.float64, device=p.device)
        self._lagr_out_buf_2d = out

    _lagrangian_forces_2d_kernel(
        eps_ij[0][0], eps_ij[0][1], eps_ij[1][1],
        p_field, nu_rho_field,
        cnt_flat, cnt_offsets, com_pos,
        bx0, by0, inv_dx, inv_dy,
        Mx, My, method="linear",
        sample_offset_pressure=off_p,
        sample_offset_friction=off_f,
        out=out,
    )

    # Vectorised scatter of the (B, 6) kernel output into the per-body
    # force caches and per-iteration records (no Python per-body loop).
    out_d = out.to(self.dtype)
    self.friction_force_lin_x[:B] = out_d[:, 0]
    self.friction_force_lin_y[:B] = out_d[:, 1]
    self.friction_force_ang_z[:B] = out_d[:, 2]
    self.pressure_force_x[:B]     = out_d[:, 3]
    self.pressure_force_y[:B]     = out_d[:, 4]
    self.pressure_force_ang_z[:B] = out_d[:, 5]

    self.viscous_drag_record[:B, 0, iteration]  = out_d[:, 0]
    self.viscous_drag_record[:B, 1, iteration]  = out_d[:, 1]
    self.pressure_drag_record[:B, 0, iteration] = out_d[:, 3]
    self.pressure_drag_record[:B, 1, iteration] = out_d[:, 4]

    # Lagrangian path doesn't expose volumetric stress / pforce tensors.
    self.xstress_tensor = None
    self.ystress_tensor = None
    self.pforce_x = None
    self.pforce_y = None


def forces_lagrangian_3d(self, u, v, w, p, iteration):
    """Surface-integral viscous + pressure forces in 3-D (rigid bodies).

    Fused, multi-body native kernel path.  Builds the CC strain-rate
    tensor once per step, packs every body's triangulation
    (``tri_centroid_world`` / ``tri_normal_world`` / ``tri_area``;
    see ``body._build_surface_3d``) into single flat tensors with
    int64 per-body offsets, and dispatches a single
    ``lagrangian_forces_3d`` call (CPU OpenMP or CUDA atomicAdd) to
    integrate ``Σ_T (ν·ρ·ε·n - p n) A_T`` per body.

    Output channels match the legacy 3-D 12-channel layout:
        [fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
         fp_x, fp_y, fp_z, tp_x, tp_y, tp_z]
    """
    comp = self.composite_body
    B = len(comp.bodies)
    h = self.h
    inv_dx = 1.0 / h; inv_dy = 1.0 / h; inv_dz = 1.0 / h
    off_p = float(self.lagr_sample_offset_pressure)
    off_f = float(self.lagr_sample_offset_friction)
    # The slab must cover whichever channel reaches furthest out.
    soff_max = max(abs(off_p), abs(off_f))

    # Pack per-body triangulation into flat (3, T_total) / (T_total,)
    # tensors with int64 offsets.  Raise the same friendly error as the
    # legacy path when a body is missing the triangulation contract.
    tri_cs, tri_ns, tri_as = [], [], []
    offs = [0]
    for bi, body in enumerate(comp.bodies):
        tri_c = getattr(body, 'tri_centroid_world', None)
        tri_n = getattr(body, 'tri_normal_world', None)
        tri_a = getattr(body, 'tri_area', None)
        if tri_c is None or tri_n is None or tri_a is None:
            raise RuntimeError(
                f"Body {bi} ({type(body).__name__}) does not expose "
                "tri_centroid_world/tri_normal_world/tri_area; cannot use "
                "force_method='lagrangian' in 3-D.  Use 'eulerian' or "
                "add a surface triangulation in the body's constructor."
            )
        tri_cs.append(tri_c.to(dtype=p.dtype, device=p.device))
        tri_ns.append(tri_n.to(dtype=p.dtype, device=p.device))
        tri_as.append(tri_a.to(dtype=p.dtype, device=p.device))
        offs.append(offs[-1] + tri_cs[-1].shape[1])
    tri_centroid = torch.cat(tri_cs, dim=1) if tri_cs else torch.empty(3, 0, dtype=p.dtype, device=p.device)
    tri_normal   = torch.cat(tri_ns, dim=1) if tri_ns else torch.empty(3, 0, dtype=p.dtype, device=p.device)
    tri_area     = torch.cat(tri_as, dim=0) if tri_as else torch.empty(0, dtype=p.dtype, device=p.device)
    tri_offsets  = torch.tensor(offs, dtype=torch.int64, device=p.device)

    com_pos = torch.stack(
        [body.com_pos.to(dtype=p.dtype, device=p.device) for body in comp.bodies],
        dim=0,
    ) if B > 0 else torch.empty(0, 3, dtype=p.dtype, device=p.device)

    # ---- Crop the strain-rate / pressure build to the triangle AABB ---
    # See ``forces_lagrangian_2d``: the kernel only samples the fields at
    # triangle centroids, so build the 6 strain components + pressure on a
    # slab around the union of all triangulations instead of the full
    # (Nx·Ny·Nz) domain.  This is the dominant 3-D cost (6 gradient fields).
    nu_rho = self._compute_nu_rho_for_forces(u, v, w)
    slab = _marker_aabb_slab(tri_centroid, (comp.x, comp.y, comp.z), soff_max, inv_dx)
    if slab is not None:
        (i0, i1), (j0, j1), (k0, k1) = slab
        sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
        eps_ij = _viscous_stress_tensor(
            (u[sl].contiguous(), v[sl].contiguous(), w[sl].contiguous()), h)
        p_field = p[sl].contiguous()
        bx0 = float(comp.x[i0]); by0 = float(comp.y[j0]); bz0 = float(comp.z[k0])
        Mx = i1 - i0; My = j1 - j0; Mz = k1 - k0
        nu_rho_crop = sl
    else:
        eps_ij = _viscous_stress_tensor((u, v, w), h)
        p_field = p
        bx0 = float(comp.x[0]); by0 = float(comp.y[0]); bz0 = float(comp.z[0])
        Mx = int(comp.x.numel()); My = int(comp.y.numel()); Mz = int(comp.z.numel())
        nu_rho_crop = None

    if torch.is_tensor(nu_rho) and nu_rho.ndim == 3:
        nu_rho_field = nu_rho[nu_rho_crop].contiguous() if nu_rho_crop is not None else nu_rho
    else:
        buf = getattr(self, '_lagr_nu_rho_buf_3d', None)
        if buf is None:
            buf = torch.empty(1, dtype=p.dtype, device=p.device)
            self._lagr_nu_rho_buf_3d = buf
        buf.fill_(float(nu_rho))
        nu_rho_field = buf

    # Persistent (B, 12) float64 output buffer (zeroed by the kernel).
    out = getattr(self, '_lagr_out_buf_3d', None)
    if out is None or out.shape[0] != B:
        out = torch.zeros((B, 12), dtype=torch.float64, device=p.device)
        self._lagr_out_buf_3d = out

    _lagrangian_forces_3d_kernel(
        eps_ij[0][0], eps_ij[1][1], eps_ij[2][2],
        eps_ij[0][1], eps_ij[0][2], eps_ij[1][2],
        p_field, nu_rho_field,
        tri_centroid, tri_normal, tri_area,
        tri_offsets, com_pos,
        bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
        Mx, My, Mz, method="linear",
        # Tunable per config (``sample_offset_{pressure,friction}_cells``,
        # or the legacy single ``lagrangian_sample_offset``).  See solver.py.
        sample_offset_pressure=off_p,
        sample_offset_friction=off_f,
        out=out,
    )

    # Vectorised scatter of the (B, 12) kernel output (no per-body loop).
    out_d = out.to(self.dtype)
    self.friction_force_lin_x[:B] = out_d[:, 0]
    self.friction_force_lin_y[:B] = out_d[:, 1]
    self.friction_force_lin_z[:B] = out_d[:, 2]
    self.friction_force_ang_x[:B] = out_d[:, 3]
    self.friction_force_ang_y[:B] = out_d[:, 4]
    self.friction_force_ang_z[:B] = out_d[:, 5]
    self.pressure_force_x[:B]     = out_d[:, 6]
    self.pressure_force_y[:B]     = out_d[:, 7]
    self.pressure_force_z[:B]     = out_d[:, 8]
    self.pressure_force_ang_x[:B] = out_d[:, 9]
    self.pressure_force_ang_y[:B] = out_d[:, 10]
    self.pressure_force_ang_z[:B] = out_d[:, 11]

    self.viscous_drag_record[:B, 0, iteration]   = out_d[:, 0]
    self.viscous_drag_record[:B, 1, iteration]   = out_d[:, 1]
    self.viscous_drag_record[:B, 2, iteration]   = out_d[:, 2]
    self.viscous_torque_record[:B, 0, iteration] = out_d[:, 3]
    self.viscous_torque_record[:B, 1, iteration] = out_d[:, 4]
    self.viscous_torque_record[:B, 2, iteration] = out_d[:, 5]
    self.pressure_drag_record[:B, 0, iteration]   = out_d[:, 6]
    self.pressure_drag_record[:B, 1, iteration]   = out_d[:, 7]
    self.pressure_drag_record[:B, 2, iteration]   = out_d[:, 8]
    self.pressure_torque_record[:B, 0, iteration] = out_d[:, 9]
    self.pressure_torque_record[:B, 1, iteration] = out_d[:, 10]
    self.pressure_torque_record[:B, 2, iteration] = out_d[:, 11]

    self.xstress_tensor = None
    self.ystress_tensor = None
    self.zstress_tensor = None
    self.pforce_x = None
    self.pforce_y = None
    self.pforce_z = None


