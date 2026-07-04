"""Force-and-torque computation for immersed bodies.

Extracted from ``solver.py`` (item #8 of the HIGH PRIORITY backlog).

Module-level kernels (``_forces_shared_*`` / ``_forces_body_*``) are pure
tensor-in / tensor-out functions kept module-level so they can be wrapped
by :func:`torch.compile`.

The ``forces_method*`` functions take ``self`` (a ``FluidSolver``) as their
first argument and are bound to ``FluidSolver`` as methods at the bottom
of ``solver.py`` -- they are NOT directly callable from this module's
public API.  This keeps the public API of :class:`FluidSolver` unchanged
while moving ~1500 LoC of force code out of ``solver.py``.
"""
from __future__ import annotations

import math

import torch

from lilytorch.src import operations as ops
# Warp force ops are imported from the leaf kernel modules (not from facade) so
# this module stays upstream of facade in the import graph.  The Eulerian readout
# (``streaming_sdf_forces_post_*_warp``) is defined in the merged Warp section at
# the end of this file and consumed directly by force_method2 below.
from lilytorch.src.lagrangian import (
    lagrangian_forces_2d_warp as _lagrangian_forces_2d_kernel,
    lagrangian_forces_3d_warp as _lagrangian_forces_3d_kernel,
)


# ======================================================================
# Compilable force-computation kernels  (module-level, for torch.compile)
# ======================================================================

def _forces_shared(vels, p, normals, nu_rho, h):
    """Dim-agnostic shared stress·n and pressure force density.

    Inputs are pure tensors / scalars — no ``self`` access — so this
    function is safe for ``torch.compile(mode='reduce-overhead')``.  The
    Python-level loops over ``range(D)`` unroll at trace time and Dynamo
    guards on ``len(vels)``, so calling with D=2 and D=3 produces two
    fully shape-specialized compiled artifacts (no per-D wrappers needed).

    * **Diagonal** (∂u_i/∂x_i) uses the compact MAC stencil — exact at CC.
    * **Cross** (∂u_i/∂x_j, i≠j) interpolates u_i to CC along its stagger
      dim then takes ``torch.gradient`` along x_j.

    Stagger convention: u_i is staggered at -h/2 along its own dim i.

    Parameters
    ----------
    vels    : tuple of D MAC-staggered velocity tensors.
    p       : CC pressure tensor.
    normals : tuple of D CC unit-normal components.
    nu_rho  : scalar or CC tensor (variable viscosity).
    h       : grid spacing.

    Returns
    -------
    ``(stresses, pforces)`` — each a length-D tuple of CC tensors.
    """
    D = len(vels)
    rng = range(D)

    # u_i interpolated to CC along its stagger dim i (used for cross derivatives).
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

    # grad[i][j] = ∂u_i/∂x_j at CC.
    grad = [[None] * D for _ in rng]
    for i in rng:
        # diagonal: compact stencil from the staggered field directly.
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
        # cross: central diff of CC-interpolated u_i along x_j (j ≠ i).
        for j in rng:
            if j != i:
                grad[i][j] = torch.gradient(
                    cc_vels[i], spacing=h, dim=j, edge_order=2,
                )[0]

    # Viscous stress·n_j summed over j:  (σ·n)_i = ν ρ ( 2 ∂u_i/∂x_i n_i
    #                                       + Σ_{j≠i} (∂u_i/∂x_j + ∂u_j/∂x_i) n_j )
    stresses = []
    for i in rng:
        s = 2.0 * grad[i][i] * normals[i]
        for j in rng:
            if j != i:
                s = s + (grad[i][j] + grad[j][i]) * normals[j]
        stresses.append(nu_rho * s)

    # Pressure force density: -p n_i (NO mu0 masking — see note below).
    # The smoothed delta δ_ε(φ) is normalised so that ∫δ_ε dφ = 1 over the
    # full support [-ε, +ε].  Masking p by mu0 (the smooth Heaviside, which
    # equals 0.5 at the surface) halves the integral:
    #     ∫ mu0(φ) δ_ε(φ) dφ = 0.5  (WRONG)
    # vs  ∫        δ_ε(φ) dφ = 1.0  (CORRECT)
    # The Poisson solve produces a continuous pressure field across the
    # interface, so both sides contribute correctly to the surface integral.
    pforces = [-p * normals[i] for i in rng]

    return tuple(stresses), tuple(pforces)


def _forces_body_integrate_3d(
    xstress, ystress, zstress,
    pforce_x, pforce_y, pforce_z,
    sdf_i, eps_body, eps_solver,
    com_x, com_y, com_z,
    X, Y, Z, h3,
    sdf_grad_mag=None,
):
    """Integrate viscous + pressure force / torque for ONE body.

    Inlines ``body.phi`` (smoothed delta) and ``cross_product_3d`` for
    maximal fusion under ``torch.compile``.

    Parameters
    ----------
    sdf_grad_mag : (Ni, Nj, Nk) or None
        |∇SDF| for this body.  When provided (Towers 2nd-order), deltas
        are divided by this to correct for non-unit SDF gradients.

    Returns 18 scalars: (fv_x,fv_y,fv_z, tv_x,tv_y,tv_z,
                          fp_x,fp_y,fp_z, tp_x,tp_y,tp_z).
    """
    # smoothed delta — viscous (shifted by eps_solver)
    d_visc = sdf_i - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # smoothed delta — pressure
    delta_pres = torch.where(
        torch.abs(sdf_i) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_i / eps_body)) / (2.0 * eps_body),
        0.0,
    )

    # Towers (2008) 2nd-order correction: δ_S = δ_ε(φ) / |∇φ|
    if sdf_grad_mag is not None:
        inv_grad = 1.0 / sdf_grad_mag.clamp(min=1e-3)
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    # viscous forces — cast to float64 before .sum() for deterministic
    # accumulation order (GPU tree-reduction vs CPU sequential).
    fvisc_x = xstress * delta_visc
    fvisc_y = ystress * delta_visc
    fvisc_z = zstress * delta_visc
    _d = torch.float64
    fv_x = fvisc_x.to(_d).sum().to(fvisc_x.dtype) * h3
    fv_y = fvisc_y.to(_d).sum().to(fvisc_y.dtype) * h3
    fv_z = fvisc_z.to(_d).sum().to(fvisc_z.dtype) * h3

    # moment arms from CoM
    rx = X - com_x
    ry = Y - com_y
    rz = Z - com_z

    # torque  r × f_visc
    tv_x = (ry * fvisc_z - rz * fvisc_y).to(_d).sum().to(fvisc_x.dtype) * h3
    tv_y = (rz * fvisc_x - rx * fvisc_z).to(_d).sum().to(fvisc_x.dtype) * h3
    tv_z = (rx * fvisc_y - ry * fvisc_x).to(_d).sum().to(fvisc_x.dtype) * h3

    # pressure forces
    fpres_x = pforce_x * delta_pres
    fpres_y = pforce_y * delta_pres
    fpres_z = pforce_z * delta_pres
    fp_x = fpres_x.to(_d).sum().to(fpres_x.dtype) * h3
    fp_y = fpres_y.to(_d).sum().to(fpres_y.dtype) * h3
    fp_z = fpres_z.to(_d).sum().to(fpres_z.dtype) * h3

    # torque  r × f_pres
    tp_x = (ry * fpres_z - rz * fpres_y).to(_d).sum().to(fpres_x.dtype) * h3
    tp_y = (rz * fpres_x - rx * fpres_z).to(_d).sum().to(fpres_x.dtype) * h3
    tp_z = (rx * fpres_y - ry * fpres_x).to(_d).sum().to(fpres_x.dtype) * h3

    return (fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
            fp_x, fp_y, fp_z, tp_x, tp_y, tp_z)


def _forces_body_batch(
    stresses, pforces, sdf_all, eps_body, eps_solver,
    com, grids, h_pow, sdf_grad_mag=None,
):
    """Dim-agnostic batched per-body force/torque integration.

    Returns lists ``(fv, tv, fp, tp)``.  Length of ``fv``/``fp`` is D; length
    of ``tv``/``tp`` is 1 in 2-D (single out-of-plane torque) and 3 in 3-D.

    Torques use the origin-decomposition trick that avoids materializing
    ``(B, *spatial)`` moment-arm tensors:
        τ_i = (Σ X_j·f_k)·h^D - com_j·F_k  -  (Σ X_k·f_j)·h^D + com_k·F_j
    where (i, j, k) is a cyclic permutation of the spatial axes.
    """
    D = len(stresses)
    spatial = tuple(range(1, D + 1))
    _d = torch.float64
    _dt = stresses[0].dtype

    # smoothed deltas at CC (broadcast over B)
    d_visc = sdf_all - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        torch.zeros_like(sdf_all),
    )
    delta_pres = torch.where(
        torch.abs(sdf_all) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_all / eps_body)) / (2.0 * eps_body),
        torch.zeros_like(sdf_all),
    )
    # Towers (2008) 2nd-order correction: δ_S = δ_ε(φ) / |∇φ|
    if sdf_grad_mag is not None:
        inv_grad = 1.0 / sdf_grad_mag.clamp(min=1e-3)
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    fvisc = [stresses[i].unsqueeze(0) * delta_visc for i in range(D)]
    fpres = [pforces[i].unsqueeze(0) * delta_pres for i in range(D)]

    fv = [f.to(_d).sum(dim=spatial).to(_dt) * h_pow for f in fvisc]
    fp = [f.to(_d).sum(dim=spatial).to(_dt) * h_pow for f in fpres]

    Xu = [g.unsqueeze(0) for g in grids]

    # Cyclic axis pairs: τ_i = r_j × f_k.  In 2-D the only torque is τ_z
    # (single out-of-plane component); in 3-D, τ_x, τ_y, τ_z.
    if D == 2:
        cycl = [(0, 1)]
    else:
        cycl = [(1, 2), (2, 0), (0, 1)]

    tv, tp = [], []
    for (j, k) in cycl:
        raw_v_jk = (Xu[j] * fvisc[k]).to(_d).sum(dim=spatial).to(_dt) * h_pow
        raw_v_kj = (Xu[k] * fvisc[j]).to(_d).sum(dim=spatial).to(_dt) * h_pow
        tv.append(raw_v_jk - com[j] * fv[k] - raw_v_kj + com[k] * fv[j])
        raw_p_jk = (Xu[j] * fpres[k]).to(_d).sum(dim=spatial).to(_dt) * h_pow
        raw_p_kj = (Xu[k] * fpres[j]).to(_d).sum(dim=spatial).to(_dt) * h_pow
        tp.append(raw_p_jk - com[j] * fp[k] - raw_p_kj + com[k] * fp[j])

    return tuple(fv), tuple(tv), tuple(fp), tuple(tp)


# ======================================================================
# Methods bound to FluidSolver (take ``self`` as first argument).
# ======================================================================

def forces_method1(self, u, v, p, iteration):

    # ---- CC normals (reuse cached values when available) ---------
    normal_x = getattr(self, 'normal_x', None)
    if normal_x is None:
        normal_x, normal_y = self.composite_body.compute_normals(
            self.composite_body.sdf_val
        )
        self.normal_x = normal_x
        self.normal_y = normal_y
    else:
        normal_y = self.normal_y

    # Build a co-located CC traction field before contour interpolation.
    # The legacy method1 mixed x/y traction components sampled from
    # different staggered grids, which produced noisy force/torque pairs
    # at moving contour points and destabilized MuJoCo coupling.
    nu_rho = self._compute_nu_rho_for_forces(u, v)
    stresses, pforces = self._forces_shared_compiled(
        (u, v), p, (normal_x, normal_y), nu_rho, self.h,
    )
    xstress, ystress = stresses
    pforce_x, pforce_y = pforces

    self.xstress_tensor = xstress
    self.ystress_tensor = ystress
    self.pforce_x = pforce_x
    self.pforce_y = pforce_y


    for i, body in enumerate(self.composite_body.bodies[:]):

              mask       = body.mask
              curv_coord = body.curv_coord[mask].to(torch.float64)

              moment_arm = [
                  body.cnt_update[0]-body.com_pos[0],
                  body.cnt_update[1]-body.com_pos[1],
              ]

              self.interp_utility.F = self.xstress_tensor
              f_x = self.interp_utility(body.cnt_update[0], body.cnt_update[1])
              self.interp_utility.F = self.ystress_tensor
              f_y = self.interp_utility(body.cnt_update[0], body.cnt_update[1])

              visc_torque = ops.cross_product_2d(
                      moment_arm[0][mask],
                      moment_arm[1][mask],
                      f_x[mask],
                      f_y[mask]
                      ).to(torch.float64)

              self.friction_force_lin_x[i] = torch.trapz(
                  f_x[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.friction_force_lin_y[i] = torch.trapz(
                  f_y[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.friction_force_ang_z[i] = torch.trapz(
                  visc_torque, curv_coord
              ).to(self.dtype)

              moment_arm = [
                  body.cnt_update[0]-body.com_pos[0],
                  body.cnt_update[1]-body.com_pos[1],
              ]

              self.interp_utility.F = self.pforce_x
              f_x = self.interp_utility(body.cnt_update[0], body.cnt_update[1])
              self.interp_utility.F = self.pforce_y
              f_y = self.interp_utility(body.cnt_update[0], body.cnt_update[1])

              pres_torque = ops.cross_product_2d(
                      moment_arm[0][mask],
                      moment_arm[1][mask],
                      f_x[mask],
                      f_y[mask]
                      ).to(torch.float64)

              self.pressure_force_x[i] = torch.trapz(
                  f_x[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.pressure_force_y[i] = torch.trapz(
                  f_y[mask].to(torch.float64), curv_coord
              ).to(self.dtype)
              self.pressure_force_ang_z[i] = torch.trapz(
                  pres_torque, curv_coord
              ).to(self.dtype)

              self.viscous_drag_record[i,0,iteration] = self.friction_force_lin_x[i]
              self.viscous_drag_record[i,1,iteration] = self.friction_force_lin_y[i]

              self.pressure_drag_record[i,0,iteration] = self.pressure_force_x[i]
              self.pressure_drag_record[i,1,iteration] = self.pressure_force_y[i]


def forces_method2(self, u, v, p, iteration):
    comp = self.composite_body
    # 2-D update-time force caches are intentionally ignored here.
    # The current-step force pass below recomputes forces from the same
    # streamed geometry using post-fluid-step u/v/p.
    if getattr(comp, '_combined_forces_out', None) is not None:
        comp._combined_forces_out = None

    B = len(comp.bodies)

    _stream_step = getattr(comp, '_kernel_step', None)
    _have_sparse_2d = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )
    _use_kernel_post_forces_2d = (
        not _have_sparse_2d
        and _stream_step is not None
        and getattr(comp, '_kernel_static_2d', None) is not None
    )

    if _use_kernel_post_forces_2d:

        sm = comp._kernel_static_2d

        out2d = getattr(self, '_kernel_post_out_buf_2d', None)
        if out2d is None or out2d.shape != (B, 6):
            out2d = torch.zeros((B, 6), dtype=torch.float64, device=self.device)
            self._kernel_post_out_buf_2d = out2d
        else:
            out2d.zero_()

        if self.use_variable_viscosity:
            nu_rho_field = self._compute_nu_rho_for_forces(u, v)
        else:
            nu_rho_scalar = getattr(self, '_kernel_post_nu_rho_scalar_2d', None)
            if nu_rho_scalar is None:
                nu_rho_scalar = torch.empty(
                    (1,), device=self.device, dtype=self.dtype,
                )
                self._kernel_post_nu_rho_scalar_2d = nu_rho_scalar
            nu_rho_scalar.fill_(float(self.nu) * float(self.rho))
            nu_rho_field = nu_rho_scalar

        eps_body = comp.bodies[0].eps
        interp_method = int(getattr(self, '_sdf_interp_method', 0))
        _fsm = 1 if getattr(self, 'force_submethod', 'ndelta') == 'deltaH' else 0
        _ph_tau = float(getattr(self, 'force_ph_blend_cells', 1.5)) * self.h

        streaming_sdf_forces_post_2d(
            sm['F_flat'], sm['F_offsets'],
            sm['body_shapes'], sm['body_meta'], _stream_step['kin'],
            _stream_step['aabb_lo'], _stream_step['aabb_dim'],
            _stream_step['gx'], _stream_step['gy'],
            self.h, _stream_step['max_vol'],
            comp.sdf_val,
            interp_method,
            u.contiguous(), v.contiguous(), p.contiguous(),
            nu_rho_field,
            eps_body, self.eps, self.h2,
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

    # ---- CC normals ------------------------------------------------
    # Python path: cached by _recompute_mu_normals (called before step).
    # Kernel path: no persistent mu/normal buffers exist, so compute once
    # here and store on self — _release_bdim_fields clears them after the
    # step.  This avoids a redundant torch.gradient call on every implicit
    # coupling sub-iteration.
    normal_x = getattr(self, 'normal_x', None)
    normal_y = getattr(self, 'normal_y', None)
    if normal_x is None or normal_y is None:
        normal_x, normal_y = comp.compute_normals(comp.sdf_val)
        self.normal_x = normal_x
        self.normal_y = normal_y

    # ==============================================================
    # BATCHED path — all bodies in one fused call
    # ==============================================================
    # Shared part: stress + pressure force density.  In the streaming
    # force path, crop this bandwidth-heavy tensor work to the union of
    # body AABBs (with a halo for derivative stencils) and let the sparse
    # force kernel index relative to that cropped slab.
    u_aabb = None

    nu_rho = self._compute_nu_rho_for_forces(u, v)
    if u_aabb is not None:
        u_i0, u_i1, u_j0, u_j1 = u_aabb
        usl = (slice(u_i0, u_i1), slice(u_j0, u_j1))
        nu_rho_sub = (
            nu_rho[usl].contiguous()
            if torch.is_tensor(nu_rho) and nu_rho.ndim == 2
            else nu_rho
        )
        stresses, pforces = self._forces_shared_compiled(
            (u[usl].contiguous(), v[usl].contiguous()),
            p[usl].contiguous(),
            (normal_x[usl].contiguous(), normal_y[usl].contiguous()),
            nu_rho_sub, self.h,
        )
    else:
        u_i0, u_j0 = 0, 0
        stresses, pforces = self._forces_shared_compiled(
            (u, v), p, (normal_x, normal_y), nu_rho, self.h,
        )
    xstress, ystress = stresses
    pforce_x, pforce_y = pforces

    # Cache for post-processing
    self.xstress_tensor = xstress
    self.ystress_tensor = ystress
    self.pforce_x = pforce_x
    self.pforce_y = pforce_y

    eps_body = comp.bodies[0].eps


    # Towers (2008) 2nd-order: compute per-body |∇SDF| on CC grid  (B,Ni,Nj)
    # ``comp.sdf_vals`` was the legacy dense per-body SDF stack.  The new
    # Python-style 2-D body updates populate ``comp._sdf_sparse``
    # (per-body AABB-cropped slabs) instead — mirroring 3-D.  Reconstruct
    # a dense (B, Ni, Nj) tensor here when the sparse storage is present.
    _have_sparse_2d = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )
    if _have_sparse_2d:
        _FAR = 1e6
        Ni, Nj = comp.sdf_val.shape
        sdf_vals = torch.full(
            (B, Ni, Nj), _FAR,
            device=self.device, dtype=self.dtype,
        )
        for bi in range(B):
            aabb_i, sdf_sub_i = comp._sdf_sparse[bi]
            if aabb_i is not None:
                i0, i1, j0, j1 = aabb_i
                sdf_vals[bi, i0:i1, j0:j1] = sdf_sub_i
            else:
                sdf_vals[bi] = sdf_sub_i
    elif hasattr(comp, 'sdf_vals') and comp.sdf_vals is not None:
        sdf_vals = comp.sdf_vals  # legacy dense path
    else:
        # Analytical composite (e.g. CompositeBodyAnalytical) keeps the
        # current per-body CC SDF on each sub-body as ``body.sdf_val``
        # (already rotated/translated in body.update), but populates
        # neither ``_sdf_sparse`` nor a dense ``sdf_vals`` stack.  Rebuild
        # the (B, Ni, Nj) stack from the per-body fields so the standalone
        # (non-kernel) force path works for analytical bodies too.
        sdf_vals = torch.stack(
            [b.sdf_val for b in comp.bodies], dim=0,
        )

    sdf_grad_mag_2d = None
    if self.force_delta_order == 2:
        gx = torch.gradient(sdf_vals, spacing=self.h, dim=1, edge_order=2)[0]
        gy = torch.gradient(sdf_vals, spacing=self.h, dim=2, edge_order=2)[0]
        sdf_grad_mag_2d = torch.sqrt(gx**2 + gy**2)

    # Build the 2-D cell-centred meshgrid on the fly — comp._grids may
    # not be allocated in kernel mode (memory optimisation; see
    # body.py:_StaggeredGrids).  Small for 2-D grids.
    _Xcc, _Ycc = torch.meshgrid(comp.x, comp.y, indexing='ij')
    fv, tv, fp, tp = self._forces_body_batch_compiled(
        (xstress, ystress),
        (pforce_x, pforce_y),
        sdf_vals,
        eps_body, self.eps,
        (comp.com_pos[:, 0], comp.com_pos[:, 1]),
        (_Xcc, _Ycc), self.h2,
        sdf_grad_mag_2d,
    )
    fv_x, fv_y = fv
    tv_z,      = tv
    fp_x, fp_y = fp
    tp_z,      = tp

    for i in range(B):
        self.friction_force_lin_x[i] = fv_x[i]
        self.friction_force_lin_y[i] = fv_y[i]
        self.friction_force_ang_z[i] = tv_z[i]
        self.pressure_force_x[i] = fp_x[i]
        self.pressure_force_y[i] = fp_y[i]
        self.pressure_force_ang_z[i] = tp_z[i]
        self.viscous_drag_record[i, 0, iteration]  = fv_x[i]
        self.viscous_drag_record[i, 1, iteration]  = fv_y[i]
        self.pressure_drag_record[i, 0, iteration] = fp_x[i]
        self.pressure_drag_record[i, 1, iteration] = fp_y[i]


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
    _stream_step = getattr(comp, '_kernel_step', None)
    _stream_static = getattr(comp, '_kernel_static_3d', None)
    _use_kernel_post = (
        _stream_step is not None
        and _stream_static is not None
    )
    if _use_kernel_post:
        B = len(comp.bodies)
        out = getattr(self, '_kernel_post_out_buf_3d', None)
        if out is None or out.shape != (B, 12):
            out = torch.zeros((B, 12), dtype=torch.float64, device=self.device)
            self._kernel_post_out_buf_3d = out
        else:
            out.zero_()

        if self.use_variable_viscosity:
            nu_rho_field = self._compute_nu_rho_for_forces(u, v, w)
        else:
            nu_rho_scalar = getattr(self, '_kernel_post_nu_rho_scalar_3d', None)
            if nu_rho_scalar is None:
                nu_rho_scalar = torch.empty((1,), device=self.device, dtype=self.dtype)
                self._kernel_post_nu_rho_scalar_3d = nu_rho_scalar
            nu_rho_scalar.fill_(float(self.nu) * float(self.rho))
            nu_rho_field = nu_rho_scalar

        eps_body = comp.bodies[0].eps
        _fsm = 1 if getattr(self, 'force_submethod', 'ndelta') == 'deltaH' else 0
        _ph_tau = float(getattr(self, 'force_ph_blend_cells', 1.5)) * self.h
        streaming_sdf_forces_post_3d(
            _stream_static['F_flat'], _stream_static['F_offsets'],
            _stream_static['body_shapes'], _stream_static['body_meta'],
            _stream_step['kin'], _stream_step['aabb_lo'], _stream_step['aabb_dim'],
            _stream_step['gx'], _stream_step['gy'], _stream_step['gz'],
            self.h, _stream_step['max_vol'],
            comp.sdf_val,
            getattr(self, '_sdf_interp_method', 0),
            u.contiguous(), v.contiguous(), w.contiguous(), p.contiguous(),
            nu_rho_field,
            eps_body, self.eps, self.h3, self.force_delta_order, out,
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

    nu_rho = self._compute_nu_rho_for_forces(u, v, w)
    h      = self.h
    h3     = self.h3

    # ---- CC normals (reuse cached values when available) ---------
    # Kernel path: compute once and cache on self so implicit coupling
    # sub-iterations don't repeat the torch.gradient call.
    nx = getattr(self, 'normal_x', None)
    if nx is None:
        nx, ny, nz = comp.compute_normals(comp.sdf_val)
        self.normal_x, self.normal_y, self.normal_z = nx, ny, nz
    else:
        ny, nz = self.normal_y, self.normal_z

    # ============================================================
    # Phase F (fused per-body forces) was removed; the path below
    # uses the streaming forces / shared-stress kernels (Phase D).
    # ============================================================
    # ---- decide whether to crop shared kernel to union AABB ------
    # Only worthwhile when narrow-band path is on and sparse SDFs are
    # available.  We compute the union AABB across all bodies and run
    # the bandwidth-bound _forces_shared only on that sub-block.
    # Per-body integration then uses indices RELATIVE to the union.
    _have_sparse_for_union = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )
    # Determine union AABB (only if all bodies have AABBs; if any
    # body uses full-grid evaluation we fall back to the full kernel).
    u_aabb = None
    if _have_sparse_for_union:
        u_i0, u_j0, u_k0 = 1 << 30, 1 << 30, 1 << 30
        u_i1, u_j1, u_k1 = -1, -1, -1
        for aabb_i, _ in comp._sdf_sparse:
            if aabb_i is None:
                u_i0 = -1
                break
            i0, i1, j0, j1, k0, k1 = aabb_i
            if i0 < u_i0: u_i0 = i0
            if j0 < u_j0: u_j0 = j0
            if k0 < u_k0: u_k0 = k0
            if i1 > u_i1: u_i1 = i1
            if j1 > u_j1: u_j1 = j1
            if k1 > u_k1: u_k1 = k1
        if u_i0 != -1:
            # 1-cell halo on each side for gradient stencil safety;
            # AABB indices are already padded by ``_body_aabb_indices``
            # but the cropped boundary cell of the shared kernel
            # uses a one-sided diff, so add an extra halo of 2.
            Ni, Nj, Nk = u.shape
            halo = 2
            u_i0 = max(0, u_i0 - halo); u_i1 = min(Ni, u_i1 + halo)
            u_j0 = max(0, u_j0 - halo); u_j1 = min(Nj, u_j1 + halo)
            u_k0 = max(0, u_k0 - halo); u_k1 = min(Nk, u_k1 + halo)
            u_aabb = (u_i0, u_i1, u_j0, u_j1, u_k0, u_k1)

    if u_aabb is not None:
        # ---- shared kernel on cropped sub-block ------------------
        ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
        usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))
        # Variable-viscosity (Smagorinsky/Carreau) returns a full-grid
        # tensor; crop it to the union sub-block to match the other
        # cropped inputs.  Constant viscosity stays a scalar.
        nu_rho_sub = nu_rho[usl].contiguous() if torch.is_tensor(nu_rho) and nu_rho.ndim == 3 else nu_rho
        stresses, pforces = self._forces_shared_dyn_compiled(
            (u[usl].contiguous(), v[usl].contiguous(), w[usl].contiguous()),
            p[usl].contiguous(),
            (nx[usl].contiguous(), ny[usl].contiguous(), nz[usl].contiguous()),
            nu_rho_sub, h,
        )
    else:
        # ---- full-grid shared kernel (default path) --------------
        stresses, pforces = self._forces_shared_compiled(
            (u, v, w), p, (nx, ny, nz), nu_rho, h,
        )
    xstress, ystress, zstress = stresses
    pforce_x, pforce_y, pforce_z = pforces

    # Cache stress / pforce for post-processing if needed
    self.xstress_tensor = xstress
    self.ystress_tensor = ystress
    self.zstress_tensor = zstress
    self.pforce_x = pforce_x
    self.pforce_y = pforce_y
    self.pforce_z = pforce_z

    # ---- per-body integration -----------------------------------
    X   = self.composite_body.X
    Y   = self.composite_body.Y
    Z   = self.composite_body.Z_grid
    comp = self.composite_body
    B = len(comp.bodies)

    _use_sparse = hasattr(comp, '_sdf_sparse') and comp._sdf_sparse[0] is not None
    for i, body in enumerate(comp.bodies):
        eps_body = body.eps

        if _use_sparse:
            aabb_i, sdf_sub_i = comp._sdf_sparse[i]
            if aabb_i is not None:
                i0, i1, j0, j1, k0, k1 = aabb_i
                sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
                grad_mag_i = None
                if self.force_delta_order == 2:
                    gx = torch.gradient(sdf_sub_i, spacing=h, dim=0, edge_order=2)[0]
                    gy = torch.gradient(sdf_sub_i, spacing=h, dim=1, edge_order=2)[0]
                    gz = torch.gradient(sdf_sub_i, spacing=h, dim=2, edge_order=2)[0]
                    grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)
                (fv_x, fv_y, fv_z,
                 tv_x, tv_y, tv_z,
                 fp_x, fp_y, fp_z,
                 tp_x, tp_y, tp_z) = _forces_body_integrate_3d(
                    xstress[sl], ystress[sl], zstress[sl],
                    pforce_x[sl], pforce_y[sl], pforce_z[sl],
                    sdf_sub_i, eps_body, self.eps,
                    body.com_pos[0], body.com_pos[1], body.com_pos[2],
                    X[sl], Y[sl], Z[sl], h3,
                    grad_mag_i,
                )
            else:
                # Full-grid SDF (rare: body covers most of grid)
                grad_mag_i = None
                if self.force_delta_order == 2:
                    gx = torch.gradient(sdf_sub_i, spacing=h, dim=0, edge_order=2)[0]
                    gy = torch.gradient(sdf_sub_i, spacing=h, dim=1, edge_order=2)[0]
                    gz = torch.gradient(sdf_sub_i, spacing=h, dim=2, edge_order=2)[0]
                    grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)
                (fv_x, fv_y, fv_z,
                 tv_x, tv_y, tv_z,
                 fp_x, fp_y, fp_z,
                 tp_x, tp_y, tp_z) = _forces_body_integrate_3d(
                    xstress, ystress, zstress,
                    pforce_x, pforce_y, pforce_z,
                    sdf_sub_i, eps_body, self.eps,
                    body.com_pos[0], body.com_pos[1], body.com_pos[2],
                    X, Y, Z, h3,
                    grad_mag_i,
                )
        else:
            # Legacy dense path (comp.sdf_vals exists)
            sdf_i = comp.sdf_vals[i]
            grad_mag_i = None
            if self.force_delta_order == 2:
                gx = torch.gradient(sdf_i, spacing=h, dim=0, edge_order=2)[0]
                gy = torch.gradient(sdf_i, spacing=h, dim=1, edge_order=2)[0]
                gz = torch.gradient(sdf_i, spacing=h, dim=2, edge_order=2)[0]
                grad_mag_i = torch.sqrt(gx**2 + gy**2 + gz**2)
            (fv_x, fv_y, fv_z,
             tv_x, tv_y, tv_z,
             fp_x, fp_y, fp_z,
             tp_x, tp_y, tp_z) = self._forces_body_compiled(
                xstress, ystress, zstress,
                pforce_x, pforce_y, pforce_z,
                sdf_i, eps_body, self.eps,
                body.com_pos[0], body.com_pos[1], body.com_pos[2],
                X, Y, Z, h3,
                grad_mag_i,
            )

        self.friction_force_lin_x[i] = fv_x
        self.friction_force_lin_y[i] = fv_y
        self.friction_force_lin_z[i] = fv_z
        self.friction_force_ang_x[i] = tv_x
        self.friction_force_ang_y[i] = tv_y
        self.friction_force_ang_z[i] = tv_z
        self.pressure_force_x[i] = fp_x
        self.pressure_force_y[i] = fp_y
        self.pressure_force_z[i] = fp_z
        self.pressure_force_ang_x[i] = tp_x
        self.pressure_force_ang_y[i] = tp_y
        self.pressure_force_ang_z[i] = tp_z
        self.viscous_drag_record[i, 0, iteration]  = fv_x
        self.viscous_drag_record[i, 1, iteration]  = fv_y
        self.viscous_drag_record[i, 2, iteration]  = fv_z
        self.pressure_drag_record[i, 0, iteration] = fp_x
        self.pressure_drag_record[i, 1, iteration] = fp_y
        self.pressure_drag_record[i, 2, iteration] = fp_z
        self.viscous_torque_record[i, 0, iteration]  = tv_x
        self.viscous_torque_record[i, 1, iteration]  = tv_y
        self.viscous_torque_record[i, 2, iteration]  = tv_z
        self.pressure_torque_record[i, 0, iteration] = tp_x
        self.pressure_torque_record[i, 1, iteration] = tp_y
        self.pressure_torque_record[i, 2, iteration] = tp_z


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
        Outward normal sampling offset (world units); widens the slab.
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
    soff = float(self.lagrangian_sample_offset)

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
    slab = _marker_aabb_slab(cnt_flat, (comp.x, comp.y), soff, inv_dx)
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
        sample_offset=soff,
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
    soff = float(self.lagrangian_sample_offset)

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
    slab = _marker_aabb_slab(tri_centroid, (comp.x, comp.y, comp.z), soff, inv_dx)
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
        # Tunable per config (``lagrangian_sample_offset``).  See the
        # 2-D path / solver.py docstring.
        sample_offset=soff,
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


# =====================================================================
# Warp Eulerian surface-force readout  (merged from former src/kernels/forces.py)
# ---------------------------------------------------------------------
# n.delta viscous+pressure band integral (+ deltaH second pass); the single
# production Eulerian force path (GPU and CPU).
# =====================================================================
from typing import Any

import warp as wp
import torch

wp.init()

from lilytorch.src.streaming_sdf import sdf_sample_off_2d
from lilytorch.src.streaming_sdf import trilinear_sample_off


_PI = 3.141592653589793


# ─────────────────────────────────────────────────────────────────────────────
#  3-D SDF sampling with F-offset (faithful port of sdf_sample_dispatch).
#  trilinear comes from warp_kernels (F_off-aware); triquadratic ported here.
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _triquadratic_sample_off_3d(
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    zero = type(bx0)(0.0)
    one = type(bx0)(1.0)
    half = type(bx0)(0.5)
    tx = wp.clamp((xq - bx0) * inv_dx, zero, type(bx0)(Mx - 1))
    ty = wp.clamp((yq - by0) * inv_dy, zero, type(bx0)(My - 1))
    tz = wp.clamp((zq - bz0) * inv_dz, zero, type(bx0)(Mz - 1))

    ix = wp.min(int(tx), Mx - 2)
    iy = wp.min(int(ty), My - 2)
    iz = wp.min(int(tz), Mz - 2)

    if ix < 1 or iy < 1 or iz < 1 or Mx < 3 or My < 3 or Mz < 3:
        return trilinear_sample_off(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                    inv_dx, inv_dy, inv_dz, xq, yq, zq)

    fx = tx - type(bx0)(ix)
    fy = ty - type(bx0)(iy)
    fz = tz - type(bx0)(iz)

    wxm = half * fx * (fx - one)
    wx0 = one - fx * fx
    wxp = half * fx * (fx + one)
    wym = half * fy * (fy - one)
    wy0 = one - fy * fy
    wyp = half * fy * (fy + one)
    wzm = half * fz * (fz - one)
    wz0 = one - fz * fz
    wzp = half * fz * (fz + one)

    s2 = Mz
    s1 = My * Mz
    base = F_off + (ix - 1) * s1 + (iy - 1) * s2 + (iz - 1)

    out = zero
    for dx in range(3):
        wx = wxm
        if dx == 1:
            wx = wx0
        if dx == 2:
            wx = wxp
        b0 = base + dx * s1
        plane = zero
        for dy in range(3):
            wy = wym
            if dy == 1:
                wy = wy0
            if dy == 2:
                wy = wyp
            b1 = b0 + dy * s2
            rrow = wzm * F[b1] + wz0 * F[b1 + 1] + wzp * F[b1 + 2]
            plane += wy * rrow
        out += wx * plane
    return out


@wp.func
def _sdf_sample_off_3d(
    interp_method: int,
    F:      wp.array(dtype=Any),
    F_off:  int,
    Mx: int, My: int, Mz: int,
    bx0: Any, by0: Any, bz0: Any,
    inv_dx: Any, inv_dy: Any, inv_dz: Any,
    xq: Any, yq: Any, zq: Any,
):
    if interp_method == 1:
        return _triquadratic_sample_off_3d(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                           inv_dx, inv_dy, inv_dz, xq, yq, zq)
    return trilinear_sample_off(F, F_off, Mx, My, Mz, bx0, by0, bz0,
                                inv_dx, inv_dy, inv_dz, xq, yq, zq)


# ═════════════════════════════════════════════════════════════════════════════
#  2-D main kernel — n·δ viscous (+ optional pressure) band integral
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def forces_post_2d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    u_prev:      wp.array(dtype=Any),
    v_prev:      wp.array(dtype=Any),
    p_prev:      wp.array(dtype=Any),
    nu_rho_field: wp.array(dtype=Any),
    nu_rho_field_size: wp.int32,
    inv_h: Any, eps_body: Any, eps_solver: Any, h2: Any,
    delta_order: wp.int32,
    with_pressure: wp.int32,
    max_vol: wp.int32,
    out: wp.array(dtype=wp.float64),   # (B*6,)
):
    tid = wp.tid()
    b = tid // max_vol
    local = tid - b * max_vol
    Ai = wp.int32(aabb_dim[b * 2 + 0])
    Aj = wp.int32(aabb_dim[b * 2 + 1])
    vol = Ai * Aj
    if local >= vol:
        return

    zero = type(inv_h)(0.0)
    one = type(inv_h)(1.0)
    half = type(inv_h)(0.5)
    two = type(inv_h)(2.0)

    di = local // Aj
    dj = local - di * Aj
    i = wp.int32(aabb_lo[b * 2 + 0]) + di
    j = wp.int32(aabb_lo[b * 2 + 1]) + dj
    g_idx = i * Ngy + j

    F_off = wp.int32(F_offsets[b])
    Mx = wp.int32(body_shapes[b * 2 + 0])
    My = wp.int32(body_shapes[b * 2 + 1])

    bx0 = body_meta[b * 7 + 0]
    by0 = body_meta[b * 7 + 1]
    idx_ = body_meta[b * 7 + 4]
    idy_ = body_meta[b * 7 + 5]

    r00 = kin[b * 11 + 0]
    r01 = kin[b * 11 + 1]
    r10 = kin[b * 11 + 2]
    r11 = kin[b * 11 + 3]
    bp_x = kin[b * 11 + 4]
    bp_y = kin[b * 11 + 5]
    cm_x = kin[b * 11 + 6]
    cm_y = kin[b * 11 + 7]

    xc = gx[i]
    yc = gy[j]
    dx_w = xc - bp_x
    dy_w = yc - bp_y
    bxq = r00 * dx_w + r01 * dy_w
    byq = r10 * dx_w + r11 * dy_w

    s_cc_body = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                  bx0, by0, idx_, idy_, bxq, byq)

    band_lo = wp.min(eps_solver - eps_body, -eps_body)
    band_hi = wp.max(eps_solver + eps_body, eps_body)
    if s_cc_body <= band_lo or s_cc_body >= band_hi:
        return

    nu_rho_val = nu_rho_field[0]
    if nu_rho_field_size != 1:
        nu_rho_val = nu_rho_field[g_idx]

    # ── union normal n = ∇sdf_cc / |∇sdf_cc| (central / 2nd-order one-sided) ──
    dsdx_union = zero
    if Ngx >= 3:
        if i == 0:
            dsdx_union = (type(inv_h)(-3.0) * sdf_cc[j]
                          + type(inv_h)(4.0) * sdf_cc[Ngy + j]
                          - sdf_cc[2 * Ngy + j]) * half * inv_h
        elif i == Ngx - 1:
            dsdx_union = (type(inv_h)(3.0) * sdf_cc[(Ngx - 1) * Ngy + j]
                          - type(inv_h)(4.0) * sdf_cc[(Ngx - 2) * Ngy + j]
                          + sdf_cc[(Ngx - 3) * Ngy + j]) * half * inv_h
        else:
            dsdx_union = (sdf_cc[(i + 1) * Ngy + j]
                          - sdf_cc[(i - 1) * Ngy + j]) * half * inv_h
    elif Ngx == 2:
        dsdx_union = (sdf_cc[Ngy + j] - sdf_cc[j]) * inv_h

    dsdy_union = zero
    row = i * Ngy
    if Ngy >= 3:
        if j == 0:
            dsdy_union = (type(inv_h)(-3.0) * sdf_cc[row]
                          + type(inv_h)(4.0) * sdf_cc[row + 1]
                          - sdf_cc[row + 2]) * half * inv_h
        elif j == Ngy - 1:
            dsdy_union = (type(inv_h)(3.0) * sdf_cc[row + (Ngy - 1)]
                          - type(inv_h)(4.0) * sdf_cc[row + (Ngy - 2)]
                          + sdf_cc[row + (Ngy - 3)]) * half * inv_h
        else:
            dsdy_union = (sdf_cc[row + (j + 1)]
                          - sdf_cc[row + (j - 1)]) * half * inv_h
    elif Ngy == 2:
        dsdy_union = (sdf_cc[i * Ngy + 1] - sdf_cc[i * Ngy]) * inv_h

    union_norm = wp.sqrt(dsdx_union * dsdx_union + dsdy_union * dsdy_union)
    union_inv_norm = zero
    if union_norm > zero:
        union_inv_norm = one / union_norm
    nx = dsdx_union * union_inv_norm
    ny = dsdy_union * union_inv_norm

    # ── clamped neighbour indices ────────────────────────────────────────────
    im1 = i - 1
    if i <= 0:
        im1 = 0
    ip1 = i + 1
    if i + 1 >= Ngx:
        ip1 = i
    im2 = i - 2
    if i <= 1:
        im2 = 0
    ip2 = i + 2
    if i + 2 >= Ngx:
        ip2 = Ngx - 1
    jm1 = j - 1
    if j <= 0:
        jm1 = 0
    jp1 = j + 1
    if j + 1 >= Ngy:
        jp1 = j
    jm2 = j - 2
    if j <= 1:
        jm2 = 0
    jp2 = j + 2
    if j + 2 >= Ngy:
        jp2 = Ngy - 1

    # ── velocity gradients (staggered → cell-centred) ────────────────────────
    if i + 1 < Ngx:
        dudx = (u_prev[ip1 * Ngy + j] - u_prev[i * Ngy + j]) * inv_h
    else:
        dudx = (u_prev[i * Ngy + j] - u_prev[im1 * Ngy + j]) * inv_h

    if j + 1 < Ngy:
        dvdy = (v_prev[i * Ngy + jp1] - v_prev[i * Ngy + j]) * inv_h
    else:
        dvdy = (v_prev[i * Ngy + j] - v_prev[i * Ngy + jm1]) * inv_h

    u_cc_jm2 = half * (u_prev[i * Ngy + jm2] + u_prev[ip1 * Ngy + jm2])
    u_cc_jm1 = half * (u_prev[i * Ngy + jm1] + u_prev[ip1 * Ngy + jm1])
    u_cc_j0 = half * (u_prev[i * Ngy + j] + u_prev[ip1 * Ngy + j])
    u_cc_jp1 = half * (u_prev[i * Ngy + jp1] + u_prev[ip1 * Ngy + jp1])
    u_cc_jp2 = half * (u_prev[i * Ngy + jp2] + u_prev[ip1 * Ngy + jp2])

    if Ngy >= 3:
        if j == 0:
            dudy = (type(inv_h)(-3.0) * u_cc_j0 + type(inv_h)(4.0) * u_cc_jp1
                    - u_cc_jp2) * half * inv_h
        elif j == Ngy - 1:
            dudy = (type(inv_h)(3.0) * u_cc_j0 - type(inv_h)(4.0) * u_cc_jm1
                    + u_cc_jm2) * half * inv_h
        else:
            dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h
    else:
        dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h

    v_cc_im2 = half * (v_prev[im2 * Ngy + j] + v_prev[im2 * Ngy + jp1])
    v_cc_im1 = half * (v_prev[im1 * Ngy + j] + v_prev[im1 * Ngy + jp1])
    v_cc_i0 = half * (v_prev[i * Ngy + j] + v_prev[i * Ngy + jp1])
    v_cc_ip1 = half * (v_prev[ip1 * Ngy + j] + v_prev[ip1 * Ngy + jp1])
    v_cc_ip2 = half * (v_prev[ip2 * Ngy + j] + v_prev[ip2 * Ngy + jp1])

    if Ngx >= 3:
        if i == 0:
            dvdx = (type(inv_h)(-3.0) * v_cc_i0 + type(inv_h)(4.0) * v_cc_ip1
                    - v_cc_ip2) * half * inv_h
        elif i == Ngx - 1:
            dvdx = (type(inv_h)(3.0) * v_cc_i0 - type(inv_h)(4.0) * v_cc_im1
                    + v_cc_im2) * half * inv_h
        else:
            dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h
    else:
        dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h

    xs = nu_rho_val * (two * dudx * nx + (dudy + dvdx) * ny)
    ys = nu_rho_val * ((dvdx + dudy) * nx + two * dvdy * ny)

    p_c = p_prev[g_idx]
    pxv = -p_c * nx
    pyv = -p_c * ny

    # ── smoothed delta weights ───────────────────────────────────────────────
    inv_2eps = half / eps_body
    pi_ov_eb = type(inv_h)(_PI) / eps_body

    delta_visc = zero
    delta_pres = zero
    d_visc = s_cc_body - eps_solver
    if d_visc > -eps_body and d_visc < eps_body:
        delta_visc = (one + wp.cos(pi_ov_eb * d_visc)) * inv_2eps
    if s_cc_body > -eps_body and s_cc_body < eps_body:
        delta_pres = (one + wp.cos(pi_ov_eb * s_cc_body)) * inv_2eps
    if with_pressure == 0:
        delta_pres = zero

    if delta_order == 2 and (delta_visc > zero or delta_pres > zero):
        h_grid = one / inv_h
        s_xp = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq + r00 * h_grid, byq + r10 * h_grid)
        s_xm = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq - r00 * h_grid, byq - r10 * h_grid)
        s_yp = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq + r01 * h_grid, byq + r11 * h_grid)
        s_ym = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                 bx0, by0, idx_, idy_,
                                 bxq - r01 * h_grid, byq - r11 * h_grid)
        dsdx = (s_xp - s_xm) * half * inv_h
        dsdy = (s_yp - s_ym) * half * inv_h
        grad_mag = wp.sqrt(dsdx * dsdx + dsdy * dsdy)
        min_grad = type(inv_h)(1e-3)
        if grad_mag < min_grad:
            grad_mag = min_grad
        inv_grad = one / grad_mag
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    arm_x = xc - cm_x
    arm_y = yc - cm_y

    fv_x = wp.float64(xs * delta_visc)
    fv_y = wp.float64(ys * delta_visc)
    fp_x = wp.float64(pxv * delta_pres)
    fp_y = wp.float64(pyv * delta_pres)
    ax = wp.float64(arm_x)
    ay = wp.float64(arm_y)
    h2_d = wp.float64(h2)

    o = b * 6
    wp.atomic_add(out, o + 0, fv_x * h2_d)
    wp.atomic_add(out, o + 1, fv_y * h2_d)
    wp.atomic_add(out, o + 2, (ax * fv_y - ay * fv_x) * h2_d)
    wp.atomic_add(out, o + 3, fp_x * h2_d)
    wp.atomic_add(out, o + 4, fp_y * h2_d)
    wp.atomic_add(out, o + 5, (ax * fp_y - ay * fp_x) * h2_d)


# ── 2-D deltaH (∂H) pressure second pass (force_submethod == 1) ──────────────

@wp.func
def _heaviside_smooth_2d(phi: Any, inv_eps: Any):
    pi = type(phi)(_PI)
    one = type(phi)(1.0)
    half = type(phi)(0.5)
    x = phi * inv_eps
    x = wp.clamp(x, -one, one)
    return half * (one + x + wp.sin(pi * x) / pi)


@wp.kernel
def forces_post_deltaH_pressure_2d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    p_prev:      wp.array(dtype=Any),
    inv_h: Any, inv_eps: Any, inv_tau: Any, h2: Any,
    B: wp.int32,
    uli0: wp.int32, ulj0: wp.int32, ULi: wp.int32, ULj: wp.int32,
    out: wp.array(dtype=wp.float64),
):
    local = wp.tid()
    uvol = ULi * ULj
    if local >= uvol:
        return
    di = local // ULj
    dj = local - di * ULj
    i = uli0 + di
    j = ulj0 + dj
    if i < 0 or i >= Ngx or j < 0 or j >= Ngy:
        return

    zero = type(inv_h)(0.0)
    half = type(inv_h)(0.5)
    one = type(inv_h)(1.0)

    # ∇H of the smoothed union Heaviside (central / 2nd-order one-sided).
    gHx = zero
    if Ngx >= 3:
        if i == 0:
            gHx = (type(inv_h)(-3.0) * _heaviside_smooth_2d(sdf_cc[0 * Ngy + j], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[1 * Ngy + j], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[2 * Ngy + j], inv_eps)) * half * inv_h
        elif i == Ngx - 1:
            gHx = (type(inv_h)(3.0) * _heaviside_smooth_2d(sdf_cc[(Ngx - 1) * Ngy + j], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[(Ngx - 2) * Ngy + j], inv_eps)
                   + _heaviside_smooth_2d(sdf_cc[(Ngx - 3) * Ngy + j], inv_eps)) * half * inv_h
        else:
            gHx = (_heaviside_smooth_2d(sdf_cc[(i + 1) * Ngy + j], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[(i - 1) * Ngy + j], inv_eps)) * half * inv_h
    elif Ngx == 2:
        gHx = (_heaviside_smooth_2d(sdf_cc[1 * Ngy + j], inv_eps)
               - _heaviside_smooth_2d(sdf_cc[0 * Ngy + j], inv_eps)) * inv_h

    gHy = zero
    rr = i * Ngy
    if Ngy >= 3:
        if j == 0:
            gHy = (type(inv_h)(-3.0) * _heaviside_smooth_2d(sdf_cc[rr + 0], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[rr + 1], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[rr + 2], inv_eps)) * half * inv_h
        elif j == Ngy - 1:
            gHy = (type(inv_h)(3.0) * _heaviside_smooth_2d(sdf_cc[rr + (Ngy - 1)], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_2d(sdf_cc[rr + (Ngy - 2)], inv_eps)
                   + _heaviside_smooth_2d(sdf_cc[rr + (Ngy - 3)], inv_eps)) * half * inv_h
        else:
            gHy = (_heaviside_smooth_2d(sdf_cc[rr + (j + 1)], inv_eps)
                   - _heaviside_smooth_2d(sdf_cc[rr + (j - 1)], inv_eps)) * half * inv_h
    elif Ngy == 2:
        gHy = (_heaviside_smooth_2d(sdf_cc[rr + 1], inv_eps)
               - _heaviside_smooth_2d(sdf_cc[rr + 0], inv_eps)) * inv_h

    if gHx == zero and gHy == zero:
        return

    g = i * Ngy + j
    p_c = p_prev[g]
    fdx = -p_c * gHx
    fdy = -p_c * gHy
    xc = gx[i]
    yc = gy[j]
    sdfu = sdf_cc[g]

    # softmin partition of unity Z = Σ_b exp(-(s_b - s_union)/tau)
    Z = zero
    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 2 + 0])
        j0 = wp.int32(aabb_lo[b * 2 + 1])
        Ai = wp.int32(aabb_dim[b * 2 + 0])
        Aj = wp.int32(aabb_dim[b * 2 + 1])
        if i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj:
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 2 + 0])
            My = wp.int32(body_shapes[b * 2 + 1])
            dx_w = xc - kin[b * 11 + 4]
            dy_w = yc - kin[b * 11 + 5]
            bxq = kin[b * 11 + 0] * dx_w + kin[b * 11 + 1] * dy_w
            byq = kin[b * 11 + 2] * dx_w + kin[b * 11 + 3] * dy_w
            s_b = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                    body_meta[b * 7 + 0], body_meta[b * 7 + 1],
                                    body_meta[b * 7 + 4], body_meta[b * 7 + 5],
                                    bxq, byq)
            Z += wp.exp(-(s_b - sdfu) * inv_tau)
    if Z <= zero:
        return
    inv_Z = one / Z
    h2_d = wp.float64(h2)

    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 2 + 0])
        j0 = wp.int32(aabb_lo[b * 2 + 1])
        Ai = wp.int32(aabb_dim[b * 2 + 0])
        Aj = wp.int32(aabb_dim[b * 2 + 1])
        if i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj:
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 2 + 0])
            My = wp.int32(body_shapes[b * 2 + 1])
            dx_w = xc - kin[b * 11 + 4]
            dy_w = yc - kin[b * 11 + 5]
            bxq = kin[b * 11 + 0] * dx_w + kin[b * 11 + 1] * dy_w
            byq = kin[b * 11 + 2] * dx_w + kin[b * 11 + 3] * dy_w
            s_b = sdf_sample_off_2d(interp_method, F_flat, F_off, Mx, My,
                                    body_meta[b * 7 + 0], body_meta[b * 7 + 1],
                                    body_meta[b * 7 + 4], body_meta[b * 7 + 5],
                                    bxq, byq)
            wb = wp.exp(-(s_b - sdfu) * inv_tau) * inv_Z
            fbx = wb * fdx
            fby = wb * fdy
            ax = xc - kin[b * 11 + 6]
            ay = yc - kin[b * 11 + 7]
            o = b * 6
            wp.atomic_add(out, o + 3, wp.float64(fbx) * h2_d)
            wp.atomic_add(out, o + 4, wp.float64(fby) * h2_d)
            wp.atomic_add(out, o + 5,
                          (wp.float64(ax) * wp.float64(fby)
                           - wp.float64(ay) * wp.float64(fbx)) * h2_d)


# Register float32 + float64 specialisations (generic args only).
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(forces_post_2d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "sdf_cc": _A, "u_prev": _A, "v_prev": _A, "p_prev": _A,
        "nu_rho_field": _A,
        "inv_h": _dt, "eps_body": _dt, "eps_solver": _dt, "h2": _dt,
    })
    wp.overload(forces_post_deltaH_pressure_2d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A,
        "sdf_cc": _A, "p_prev": _A,
        "inv_h": _dt, "inv_eps": _dt, "inv_tau": _dt, "h2": _dt,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  3-D main kernel — n·δ viscous (+ optional pressure) band integral
# ═════════════════════════════════════════════════════════════════════════════

@wp.kernel
def forces_post_3d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    gz:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    u_prev:      wp.array(dtype=Any),
    v_prev:      wp.array(dtype=Any),
    w_prev:      wp.array(dtype=Any),
    p_prev:      wp.array(dtype=Any),
    nu_rho_field: wp.array(dtype=Any),
    nu_rho_field_size: wp.int32,
    inv_h: Any, eps_body: Any, eps_solver: Any, h3: Any,
    delta_order: wp.int32,
    with_pressure: wp.int32,
    max_vol: wp.int32,
    out: wp.array(dtype=wp.float64),   # (B*12,)
):
    tid = wp.tid()
    b = tid // max_vol
    local = tid - b * max_vol
    Ai = wp.int32(aabb_dim[b * 3 + 0])
    Aj = wp.int32(aabb_dim[b * 3 + 1])
    Ak = wp.int32(aabb_dim[b * 3 + 2])
    vol = Ai * Aj * Ak
    if local >= vol:
        return

    zero = type(inv_h)(0.0)
    one = type(inv_h)(1.0)
    half = type(inv_h)(0.5)
    two = type(inv_h)(2.0)

    di = local // (Aj * Ak)
    rem = local - di * (Aj * Ak)
    dj = rem // Ak
    dk = rem - dj * Ak
    i = wp.int32(aabb_lo[b * 3 + 0]) + di
    j = wp.int32(aabb_lo[b * 3 + 1]) + dj
    k = wp.int32(aabb_lo[b * 3 + 2]) + dk
    g_idx = (i * Ngy + j) * Ngz + k

    F_off = wp.int32(F_offsets[b])
    Mx = wp.int32(body_shapes[b * 3 + 0])
    My = wp.int32(body_shapes[b * 3 + 1])
    Mz = wp.int32(body_shapes[b * 3 + 2])
    bx0 = body_meta[b * 10 + 0]
    by0 = body_meta[b * 10 + 1]
    bz0 = body_meta[b * 10 + 2]
    idx_ = body_meta[b * 10 + 6]
    idy_ = body_meta[b * 10 + 7]
    idz_ = body_meta[b * 10 + 8]

    r00 = kin[b * 21 + 0]
    r01 = kin[b * 21 + 1]
    r02 = kin[b * 21 + 2]
    r10 = kin[b * 21 + 3]
    r11 = kin[b * 21 + 4]
    r12 = kin[b * 21 + 5]
    r20 = kin[b * 21 + 6]
    r21 = kin[b * 21 + 7]
    r22 = kin[b * 21 + 8]
    bp_x = kin[b * 21 + 9]
    bp_y = kin[b * 21 + 10]
    bp_z = kin[b * 21 + 11]
    cm_x = kin[b * 21 + 12]
    cm_y = kin[b * 21 + 13]
    cm_z = kin[b * 21 + 14]

    xc = gx[i]
    yc = gy[j]
    zc = gz[k]
    dx_w = xc - bp_x
    dy_w = yc - bp_y
    dz_w = zc - bp_z
    bxq = r00 * dx_w + r01 * dy_w + r02 * dz_w
    byq = r10 * dx_w + r11 * dy_w + r12 * dz_w
    bzq = r20 * dx_w + r21 * dy_w + r22 * dz_w

    s_cc_body = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz,
                                   bx0, by0, bz0, idx_, idy_, idz_,
                                   bxq, byq, bzq)

    band_lo = wp.min(eps_solver - eps_body, -eps_body)
    band_hi = wp.max(eps_solver + eps_body, eps_body)
    if s_cc_body <= band_lo or s_cc_body >= band_hi:
        return

    nu_rho_val = nu_rho_field[0]
    if nu_rho_field_size != 1:
        nu_rho_val = nu_rho_field[g_idx]

    # ── union normal n = ∇sdf_cc / |∇sdf_cc| ─────────────────────────────────
    sij = i * Ngy * Ngz + j * Ngz + k
    dsdx_union = zero
    if Ngx >= 3:
        if i == 0:
            dsdx_union = (type(inv_h)(-3.0) * sdf_cc[(0 * Ngy + j) * Ngz + k]
                          + type(inv_h)(4.0) * sdf_cc[(1 * Ngy + j) * Ngz + k]
                          - sdf_cc[(2 * Ngy + j) * Ngz + k]) * half * inv_h
        elif i == Ngx - 1:
            dsdx_union = (type(inv_h)(3.0) * sdf_cc[((Ngx - 1) * Ngy + j) * Ngz + k]
                          - type(inv_h)(4.0) * sdf_cc[((Ngx - 2) * Ngy + j) * Ngz + k]
                          + sdf_cc[((Ngx - 3) * Ngy + j) * Ngz + k]) * half * inv_h
        else:
            dsdx_union = (sdf_cc[((i + 1) * Ngy + j) * Ngz + k]
                          - sdf_cc[((i - 1) * Ngy + j) * Ngz + k]) * half * inv_h
    elif Ngx == 2:
        dsdx_union = (sdf_cc[(1 * Ngy + j) * Ngz + k]
                      - sdf_cc[(0 * Ngy + j) * Ngz + k]) * inv_h

    dsdy_union = zero
    if Ngy >= 3:
        if j == 0:
            dsdy_union = (type(inv_h)(-3.0) * sdf_cc[(i * Ngy + 0) * Ngz + k]
                          + type(inv_h)(4.0) * sdf_cc[(i * Ngy + 1) * Ngz + k]
                          - sdf_cc[(i * Ngy + 2) * Ngz + k]) * half * inv_h
        elif j == Ngy - 1:
            dsdy_union = (type(inv_h)(3.0) * sdf_cc[(i * Ngy + (Ngy - 1)) * Ngz + k]
                          - type(inv_h)(4.0) * sdf_cc[(i * Ngy + (Ngy - 2)) * Ngz + k]
                          + sdf_cc[(i * Ngy + (Ngy - 3)) * Ngz + k]) * half * inv_h
        else:
            dsdy_union = (sdf_cc[(i * Ngy + (j + 1)) * Ngz + k]
                          - sdf_cc[(i * Ngy + (j - 1)) * Ngz + k]) * half * inv_h
    elif Ngy == 2:
        dsdy_union = (sdf_cc[(i * Ngy + 1) * Ngz + k]
                      - sdf_cc[(i * Ngy + 0) * Ngz + k]) * inv_h

    dsdz_union = zero
    if Ngz >= 3:
        if k == 0:
            dsdz_union = (type(inv_h)(-3.0) * sdf_cc[sij - k + 0]
                          + type(inv_h)(4.0) * sdf_cc[sij - k + 1]
                          - sdf_cc[sij - k + 2]) * half * inv_h
        elif k == Ngz - 1:
            dsdz_union = (type(inv_h)(3.0) * sdf_cc[sij]
                          - type(inv_h)(4.0) * sdf_cc[sij - 1]
                          + sdf_cc[sij - 2]) * half * inv_h
        else:
            dsdz_union = (sdf_cc[sij + 1] - sdf_cc[sij - 1]) * half * inv_h
    elif Ngz == 2:
        dsdz_union = (sdf_cc[(i * Ngy + j) * Ngz + 1]
                      - sdf_cc[(i * Ngy + j) * Ngz + 0]) * inv_h

    union_norm = wp.sqrt(dsdx_union * dsdx_union + dsdy_union * dsdy_union
                         + dsdz_union * dsdz_union)
    inv_norm = zero
    if union_norm > zero:
        inv_norm = one / union_norm
    nx = dsdx_union * inv_norm
    ny = dsdy_union * inv_norm
    nz = dsdz_union * inv_norm

    # ── clamped neighbour indices ────────────────────────────────────────────
    im1 = i - 1
    if i <= 0:
        im1 = 0
    ip1 = i + 1
    if i + 1 >= Ngx:
        ip1 = i
    im2 = i - 2
    if i <= 1:
        im2 = 0
    ip2 = i + 2
    if i + 2 >= Ngx:
        ip2 = Ngx - 1
    jm1 = j - 1
    if j <= 0:
        jm1 = 0
    jp1 = j + 1
    if j + 1 >= Ngy:
        jp1 = j
    jm2 = j - 2
    if j <= 1:
        jm2 = 0
    jp2 = j + 2
    if j + 2 >= Ngy:
        jp2 = Ngy - 1
    km1 = k - 1
    if k <= 0:
        km1 = 0
    kp1 = k + 1
    if k + 1 >= Ngz:
        kp1 = k
    km2 = k - 2
    if k <= 1:
        km2 = 0
    kp2 = k + 2
    if k + 2 >= Ngz:
        kp2 = Ngz - 1

    # flat index helper g(i,j,k) = (i*Ngy+j)*Ngz+k
    # ── normal derivatives (forward; backward at upper boundary) ─────────────
    if i + 1 < Ngx:
        dudx = (u_prev[(ip1 * Ngy + j) * Ngz + k] - u_prev[(i * Ngy + j) * Ngz + k]) * inv_h
    else:
        dudx = (u_prev[(i * Ngy + j) * Ngz + k] - u_prev[(im1 * Ngy + j) * Ngz + k]) * inv_h
    if j + 1 < Ngy:
        dvdy = (v_prev[(i * Ngy + jp1) * Ngz + k] - v_prev[(i * Ngy + j) * Ngz + k]) * inv_h
    else:
        dvdy = (v_prev[(i * Ngy + j) * Ngz + k] - v_prev[(i * Ngy + jm1) * Ngz + k]) * inv_h
    if k + 1 < Ngz:
        dwdz = (w_prev[(i * Ngy + j) * Ngz + kp1] - w_prev[(i * Ngy + j) * Ngz + k]) * inv_h
    else:
        dwdz = (w_prev[(i * Ngy + j) * Ngz + k] - w_prev[(i * Ngy + j) * Ngz + km1]) * inv_h

    # dudy (u CC = 0.5*(u[i,j,k]+u[i+1,j,k]))
    u_cc_jm2 = half * (u_prev[(i * Ngy + jm2) * Ngz + k] + u_prev[(ip1 * Ngy + jm2) * Ngz + k])
    u_cc_jm1 = half * (u_prev[(i * Ngy + jm1) * Ngz + k] + u_prev[(ip1 * Ngy + jm1) * Ngz + k])
    u_cc_j0 = half * (u_prev[(i * Ngy + j) * Ngz + k] + u_prev[(ip1 * Ngy + j) * Ngz + k])
    u_cc_jp1 = half * (u_prev[(i * Ngy + jp1) * Ngz + k] + u_prev[(ip1 * Ngy + jp1) * Ngz + k])
    u_cc_jp2 = half * (u_prev[(i * Ngy + jp2) * Ngz + k] + u_prev[(ip1 * Ngy + jp2) * Ngz + k])
    if Ngy >= 3:
        if j == 0:
            dudy = (type(inv_h)(-3.0) * u_cc_j0 + type(inv_h)(4.0) * u_cc_jp1 - u_cc_jp2) * half * inv_h
        elif j == Ngy - 1:
            dudy = (type(inv_h)(3.0) * u_cc_j0 - type(inv_h)(4.0) * u_cc_jm1 + u_cc_jm2) * half * inv_h
        else:
            dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h
    else:
        dudy = (u_cc_jp1 - u_cc_jm1) * half * inv_h

    # dudz
    u_cc_km2 = half * (u_prev[(i * Ngy + j) * Ngz + km2] + u_prev[(ip1 * Ngy + j) * Ngz + km2])
    u_cc_km1 = half * (u_prev[(i * Ngy + j) * Ngz + km1] + u_prev[(ip1 * Ngy + j) * Ngz + km1])
    u_cc_k0 = half * (u_prev[(i * Ngy + j) * Ngz + k] + u_prev[(ip1 * Ngy + j) * Ngz + k])
    u_cc_kp1 = half * (u_prev[(i * Ngy + j) * Ngz + kp1] + u_prev[(ip1 * Ngy + j) * Ngz + kp1])
    u_cc_kp2 = half * (u_prev[(i * Ngy + j) * Ngz + kp2] + u_prev[(ip1 * Ngy + j) * Ngz + kp2])
    if Ngz >= 3:
        if k == 0:
            dudz = (type(inv_h)(-3.0) * u_cc_k0 + type(inv_h)(4.0) * u_cc_kp1 - u_cc_kp2) * half * inv_h
        elif k == Ngz - 1:
            dudz = (type(inv_h)(3.0) * u_cc_k0 - type(inv_h)(4.0) * u_cc_km1 + u_cc_km2) * half * inv_h
        else:
            dudz = (u_cc_kp1 - u_cc_km1) * half * inv_h
    else:
        dudz = (u_cc_kp1 - u_cc_km1) * half * inv_h

    # dvdx (v CC = 0.5*(v[i,j,k]+v[i,j+1,k]))
    v_cc_im2 = half * (v_prev[(im2 * Ngy + j) * Ngz + k] + v_prev[(im2 * Ngy + jp1) * Ngz + k])
    v_cc_im1 = half * (v_prev[(im1 * Ngy + j) * Ngz + k] + v_prev[(im1 * Ngy + jp1) * Ngz + k])
    v_cc_i0 = half * (v_prev[(i * Ngy + j) * Ngz + k] + v_prev[(i * Ngy + jp1) * Ngz + k])
    v_cc_ip1 = half * (v_prev[(ip1 * Ngy + j) * Ngz + k] + v_prev[(ip1 * Ngy + jp1) * Ngz + k])
    v_cc_ip2 = half * (v_prev[(ip2 * Ngy + j) * Ngz + k] + v_prev[(ip2 * Ngy + jp1) * Ngz + k])
    if Ngx >= 3:
        if i == 0:
            dvdx = (type(inv_h)(-3.0) * v_cc_i0 + type(inv_h)(4.0) * v_cc_ip1 - v_cc_ip2) * half * inv_h
        elif i == Ngx - 1:
            dvdx = (type(inv_h)(3.0) * v_cc_i0 - type(inv_h)(4.0) * v_cc_im1 + v_cc_im2) * half * inv_h
        else:
            dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h
    else:
        dvdx = (v_cc_ip1 - v_cc_im1) * half * inv_h

    # dvdz
    v_cc_km2 = half * (v_prev[(i * Ngy + j) * Ngz + km2] + v_prev[(i * Ngy + jp1) * Ngz + km2])
    v_cc_km1 = half * (v_prev[(i * Ngy + j) * Ngz + km1] + v_prev[(i * Ngy + jp1) * Ngz + km1])
    v_cc_k0 = half * (v_prev[(i * Ngy + j) * Ngz + k] + v_prev[(i * Ngy + jp1) * Ngz + k])
    v_cc_kp1 = half * (v_prev[(i * Ngy + j) * Ngz + kp1] + v_prev[(i * Ngy + jp1) * Ngz + kp1])
    v_cc_kp2 = half * (v_prev[(i * Ngy + j) * Ngz + kp2] + v_prev[(i * Ngy + jp1) * Ngz + kp2])
    if Ngz >= 3:
        if k == 0:
            dvdz = (type(inv_h)(-3.0) * v_cc_k0 + type(inv_h)(4.0) * v_cc_kp1 - v_cc_kp2) * half * inv_h
        elif k == Ngz - 1:
            dvdz = (type(inv_h)(3.0) * v_cc_k0 - type(inv_h)(4.0) * v_cc_km1 + v_cc_km2) * half * inv_h
        else:
            dvdz = (v_cc_kp1 - v_cc_km1) * half * inv_h
    else:
        dvdz = (v_cc_kp1 - v_cc_km1) * half * inv_h

    # dwdx (w CC = 0.5*(w[i,j,k]+w[i,j,k+1]))
    w_cc_im2 = half * (w_prev[(im2 * Ngy + j) * Ngz + k] + w_prev[(im2 * Ngy + j) * Ngz + kp1])
    w_cc_im1 = half * (w_prev[(im1 * Ngy + j) * Ngz + k] + w_prev[(im1 * Ngy + j) * Ngz + kp1])
    w_cc_i0 = half * (w_prev[(i * Ngy + j) * Ngz + k] + w_prev[(i * Ngy + j) * Ngz + kp1])
    w_cc_ip1 = half * (w_prev[(ip1 * Ngy + j) * Ngz + k] + w_prev[(ip1 * Ngy + j) * Ngz + kp1])
    w_cc_ip2 = half * (w_prev[(ip2 * Ngy + j) * Ngz + k] + w_prev[(ip2 * Ngy + j) * Ngz + kp1])
    if Ngx >= 3:
        if i == 0:
            dwdx = (type(inv_h)(-3.0) * w_cc_i0 + type(inv_h)(4.0) * w_cc_ip1 - w_cc_ip2) * half * inv_h
        elif i == Ngx - 1:
            dwdx = (type(inv_h)(3.0) * w_cc_i0 - type(inv_h)(4.0) * w_cc_im1 + w_cc_im2) * half * inv_h
        else:
            dwdx = (w_cc_ip1 - w_cc_im1) * half * inv_h
    else:
        dwdx = (w_cc_ip1 - w_cc_im1) * half * inv_h

    # dwdy
    w_cc_jm2 = half * (w_prev[(i * Ngy + jm2) * Ngz + k] + w_prev[(i * Ngy + jm2) * Ngz + kp1])
    w_cc_jm1 = half * (w_prev[(i * Ngy + jm1) * Ngz + k] + w_prev[(i * Ngy + jm1) * Ngz + kp1])
    w_cc_j0 = half * (w_prev[(i * Ngy + j) * Ngz + k] + w_prev[(i * Ngy + j) * Ngz + kp1])
    w_cc_jp1 = half * (w_prev[(i * Ngy + jp1) * Ngz + k] + w_prev[(i * Ngy + jp1) * Ngz + kp1])
    w_cc_jp2 = half * (w_prev[(i * Ngy + jp2) * Ngz + k] + w_prev[(i * Ngy + jp2) * Ngz + kp1])
    if Ngy >= 3:
        if j == 0:
            dwdy = (type(inv_h)(-3.0) * w_cc_j0 + type(inv_h)(4.0) * w_cc_jp1 - w_cc_jp2) * half * inv_h
        elif j == Ngy - 1:
            dwdy = (type(inv_h)(3.0) * w_cc_j0 - type(inv_h)(4.0) * w_cc_jm1 + w_cc_jm2) * half * inv_h
        else:
            dwdy = (w_cc_jp1 - w_cc_jm1) * half * inv_h
    else:
        dwdy = (w_cc_jp1 - w_cc_jm1) * half * inv_h

    xs = nu_rho_val * (two * dudx * nx + (dudy + dvdx) * ny + (dudz + dwdx) * nz)
    ys = nu_rho_val * ((dvdx + dudy) * nx + two * dvdy * ny + (dvdz + dwdy) * nz)
    zs = nu_rho_val * ((dwdx + dudz) * nx + (dwdy + dvdz) * ny + two * dwdz * nz)

    p_c = p_prev[g_idx]
    pxv = -p_c * nx
    pyv = -p_c * ny
    pzv = -p_c * nz

    inv_2eps = half / eps_body
    pi_ov_eb = type(inv_h)(_PI) / eps_body
    delta_visc = zero
    delta_pres = zero
    d_visc = s_cc_body - eps_solver
    if d_visc > -eps_body and d_visc < eps_body:
        delta_visc = (one + wp.cos(pi_ov_eb * d_visc)) * inv_2eps
    if s_cc_body > -eps_body and s_cc_body < eps_body:
        delta_pres = (one + wp.cos(pi_ov_eb * s_cc_body)) * inv_2eps
    if with_pressure == 0:
        delta_pres = zero

    if delta_order == 2 and (delta_visc > zero or delta_pres > zero):
        h_grid = one / inv_h
        s_xp = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq + r00 * h_grid, byq + r10 * h_grid, bzq + r20 * h_grid)
        s_xm = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq - r00 * h_grid, byq - r10 * h_grid, bzq - r20 * h_grid)
        s_yp = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq + r01 * h_grid, byq + r11 * h_grid, bzq + r21 * h_grid)
        s_ym = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq - r01 * h_grid, byq - r11 * h_grid, bzq - r21 * h_grid)
        s_zp = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq + r02 * h_grid, byq + r12 * h_grid, bzq + r22 * h_grid)
        s_zm = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz, bx0, by0, bz0,
                                  idx_, idy_, idz_, bxq - r02 * h_grid, byq - r12 * h_grid, bzq - r22 * h_grid)
        dxx = s_xp - s_xm
        dyy = s_yp - s_ym
        dzz = s_zp - s_zm
        grad_mag = wp.sqrt((dxx * dxx + dyy * dyy + dzz * dzz) * type(inv_h)(0.25) * inv_h * inv_h)
        min_grad = type(inv_h)(1e-3)
        if grad_mag < min_grad:
            grad_mag = min_grad
        inv_grad = one / grad_mag
        delta_visc = delta_visc * inv_grad
        delta_pres = delta_pres * inv_grad

    ax = wp.float64(xc - cm_x)
    ay = wp.float64(yc - cm_y)
    az = wp.float64(zc - cm_z)
    fv_x = wp.float64(xs * delta_visc)
    fv_y = wp.float64(ys * delta_visc)
    fv_z = wp.float64(zs * delta_visc)
    fp_x = wp.float64(pxv * delta_pres)
    fp_y = wp.float64(pyv * delta_pres)
    fp_z = wp.float64(pzv * delta_pres)
    h3_d = wp.float64(h3)

    o = b * 12
    wp.atomic_add(out, o + 0, fv_x * h3_d)
    wp.atomic_add(out, o + 1, fv_y * h3_d)
    wp.atomic_add(out, o + 2, fv_z * h3_d)
    wp.atomic_add(out, o + 3, (ay * fv_z - az * fv_y) * h3_d)
    wp.atomic_add(out, o + 4, (az * fv_x - ax * fv_z) * h3_d)
    wp.atomic_add(out, o + 5, (ax * fv_y - ay * fv_x) * h3_d)
    wp.atomic_add(out, o + 6, fp_x * h3_d)
    wp.atomic_add(out, o + 7, fp_y * h3_d)
    wp.atomic_add(out, o + 8, fp_z * h3_d)
    wp.atomic_add(out, o + 9, (ay * fp_z - az * fp_y) * h3_d)
    wp.atomic_add(out, o + 10, (az * fp_x - ax * fp_z) * h3_d)
    wp.atomic_add(out, o + 11, (ax * fp_y - ay * fp_x) * h3_d)


# ── 3-D deltaH (∂H) pressure second pass ─────────────────────────────────────

@wp.func
def _heaviside_smooth_3d(phi: Any, inv_eps: Any):
    pi = type(phi)(_PI)
    one = type(phi)(1.0)
    half = type(phi)(0.5)
    x = phi * inv_eps
    x = wp.clamp(x, -one, one)
    return half * (one + x + wp.sin(pi * x) / pi)


@wp.kernel
def forces_post_deltaH_pressure_3d_kernel(
    F_flat:      wp.array(dtype=Any),
    F_offsets:   wp.array(dtype=wp.int64),
    body_shapes: wp.array(dtype=wp.int64),
    body_meta:   wp.array(dtype=Any),
    kin:         wp.array(dtype=Any),
    aabb_lo:     wp.array(dtype=wp.int64),
    aabb_dim:    wp.array(dtype=wp.int64),
    gx:          wp.array(dtype=Any),
    gy:          wp.array(dtype=Any),
    gz:          wp.array(dtype=Any),
    Ngx: wp.int32, Ngy: wp.int32, Ngz: wp.int32,
    sdf_cc:      wp.array(dtype=Any),
    interp_method: wp.int32,
    p_prev:      wp.array(dtype=Any),
    inv_h: Any, inv_eps: Any, inv_tau: Any, h3: Any,
    B: wp.int32,
    uli0: wp.int32, ulj0: wp.int32, ulk0: wp.int32,
    ULi: wp.int32, ULj: wp.int32, ULk: wp.int32,
    out: wp.array(dtype=wp.float64),
):
    localt = wp.tid()
    uvol = ULi * ULj * ULk
    if localt >= uvol:
        return
    di = localt // (ULj * ULk)
    rem = localt - di * (ULj * ULk)
    dj = rem // ULk
    dk = rem - dj * ULk
    i = uli0 + di
    j = ulj0 + dj
    k = ulk0 + dk
    if i < 0 or i >= Ngx or j < 0 or j >= Ngy or k < 0 or k >= Ngz:
        return

    zero = type(inv_h)(0.0)
    one = type(inv_h)(1.0)
    half = type(inv_h)(0.5)

    gHx = zero
    if Ngx >= 3:
        if i == 0:
            gHx = (type(inv_h)(-3.0) * _heaviside_smooth_3d(sdf_cc[(0 * Ngy + j) * Ngz + k], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[(1 * Ngy + j) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[(2 * Ngy + j) * Ngz + k], inv_eps)) * half * inv_h
        elif i == Ngx - 1:
            gHx = (type(inv_h)(3.0) * _heaviside_smooth_3d(sdf_cc[((Ngx - 1) * Ngy + j) * Ngz + k], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[((Ngx - 2) * Ngy + j) * Ngz + k], inv_eps)
                   + _heaviside_smooth_3d(sdf_cc[((Ngx - 3) * Ngy + j) * Ngz + k], inv_eps)) * half * inv_h
        else:
            gHx = (_heaviside_smooth_3d(sdf_cc[((i + 1) * Ngy + j) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[((i - 1) * Ngy + j) * Ngz + k], inv_eps)) * half * inv_h
    elif Ngx == 2:
        gHx = (_heaviside_smooth_3d(sdf_cc[(1 * Ngy + j) * Ngz + k], inv_eps)
               - _heaviside_smooth_3d(sdf_cc[(0 * Ngy + j) * Ngz + k], inv_eps)) * inv_h

    gHy = zero
    if Ngy >= 3:
        if j == 0:
            gHy = (type(inv_h)(-3.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + 0) * Ngz + k], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + 1) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[(i * Ngy + 2) * Ngz + k], inv_eps)) * half * inv_h
        elif j == Ngy - 1:
            gHy = (type(inv_h)(3.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + (Ngy - 1)) * Ngz + k], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[(i * Ngy + (Ngy - 2)) * Ngz + k], inv_eps)
                   + _heaviside_smooth_3d(sdf_cc[(i * Ngy + (Ngy - 3)) * Ngz + k], inv_eps)) * half * inv_h
        else:
            gHy = (_heaviside_smooth_3d(sdf_cc[(i * Ngy + (j + 1)) * Ngz + k], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[(i * Ngy + (j - 1)) * Ngz + k], inv_eps)) * half * inv_h
    elif Ngy == 2:
        gHy = (_heaviside_smooth_3d(sdf_cc[(i * Ngy + 1) * Ngz + k], inv_eps)
               - _heaviside_smooth_3d(sdf_cc[(i * Ngy + 0) * Ngz + k], inv_eps)) * inv_h

    gHz = zero
    base_ij = (i * Ngy + j) * Ngz
    if Ngz >= 3:
        if k == 0:
            gHz = (type(inv_h)(-3.0) * _heaviside_smooth_3d(sdf_cc[base_ij + 0], inv_eps)
                   + type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[base_ij + 1], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[base_ij + 2], inv_eps)) * half * inv_h
        elif k == Ngz - 1:
            gHz = (type(inv_h)(3.0) * _heaviside_smooth_3d(sdf_cc[base_ij + (Ngz - 1)], inv_eps)
                   - type(inv_h)(4.0) * _heaviside_smooth_3d(sdf_cc[base_ij + (Ngz - 2)], inv_eps)
                   + _heaviside_smooth_3d(sdf_cc[base_ij + (Ngz - 3)], inv_eps)) * half * inv_h
        else:
            gHz = (_heaviside_smooth_3d(sdf_cc[base_ij + (k + 1)], inv_eps)
                   - _heaviside_smooth_3d(sdf_cc[base_ij + (k - 1)], inv_eps)) * half * inv_h
    elif Ngz == 2:
        gHz = (_heaviside_smooth_3d(sdf_cc[base_ij + 1], inv_eps)
               - _heaviside_smooth_3d(sdf_cc[base_ij + 0], inv_eps)) * inv_h

    if gHx == zero and gHy == zero and gHz == zero:
        return

    g = (i * Ngy + j) * Ngz + k
    p_c = p_prev[g]
    fdx = -p_c * gHx
    fdy = -p_c * gHy
    fdz = -p_c * gHz
    xc = gx[i]
    yc = gy[j]
    zc = gz[k]
    sdfu = sdf_cc[g]

    Z = zero
    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 3 + 0])
        j0 = wp.int32(aabb_lo[b * 3 + 1])
        k0 = wp.int32(aabb_lo[b * 3 + 2])
        Ai = wp.int32(aabb_dim[b * 3 + 0])
        Aj = wp.int32(aabb_dim[b * 3 + 1])
        Ak = wp.int32(aabb_dim[b * 3 + 2])
        if (i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj
                and k >= k0 and k < k0 + Ak):
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 3 + 0])
            My = wp.int32(body_shapes[b * 3 + 1])
            Mz = wp.int32(body_shapes[b * 3 + 2])
            dx_w = xc - kin[b * 21 + 9]
            dy_w = yc - kin[b * 21 + 10]
            dz_w = zc - kin[b * 21 + 11]
            bxq = kin[b * 21 + 0] * dx_w + kin[b * 21 + 1] * dy_w + kin[b * 21 + 2] * dz_w
            byq = kin[b * 21 + 3] * dx_w + kin[b * 21 + 4] * dy_w + kin[b * 21 + 5] * dz_w
            bzq = kin[b * 21 + 6] * dx_w + kin[b * 21 + 7] * dy_w + kin[b * 21 + 8] * dz_w
            s_b = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz,
                                     body_meta[b * 10 + 0], body_meta[b * 10 + 1],
                                     body_meta[b * 10 + 2], body_meta[b * 10 + 6],
                                     body_meta[b * 10 + 7], body_meta[b * 10 + 8],
                                     bxq, byq, bzq)
            Z += wp.exp(-(s_b - sdfu) * inv_tau)
    if Z <= zero:
        return
    inv_Z = one / Z
    h3_d = wp.float64(h3)

    for b in range(B):
        i0 = wp.int32(aabb_lo[b * 3 + 0])
        j0 = wp.int32(aabb_lo[b * 3 + 1])
        k0 = wp.int32(aabb_lo[b * 3 + 2])
        Ai = wp.int32(aabb_dim[b * 3 + 0])
        Aj = wp.int32(aabb_dim[b * 3 + 1])
        Ak = wp.int32(aabb_dim[b * 3 + 2])
        if (i >= i0 and i < i0 + Ai and j >= j0 and j < j0 + Aj
                and k >= k0 and k < k0 + Ak):
            F_off = wp.int32(F_offsets[b])
            Mx = wp.int32(body_shapes[b * 3 + 0])
            My = wp.int32(body_shapes[b * 3 + 1])
            Mz = wp.int32(body_shapes[b * 3 + 2])
            dx_w = xc - kin[b * 21 + 9]
            dy_w = yc - kin[b * 21 + 10]
            dz_w = zc - kin[b * 21 + 11]
            bxq = kin[b * 21 + 0] * dx_w + kin[b * 21 + 1] * dy_w + kin[b * 21 + 2] * dz_w
            byq = kin[b * 21 + 3] * dx_w + kin[b * 21 + 4] * dy_w + kin[b * 21 + 5] * dz_w
            bzq = kin[b * 21 + 6] * dx_w + kin[b * 21 + 7] * dy_w + kin[b * 21 + 8] * dz_w
            s_b = _sdf_sample_off_3d(interp_method, F_flat, F_off, Mx, My, Mz,
                                     body_meta[b * 10 + 0], body_meta[b * 10 + 1],
                                     body_meta[b * 10 + 2], body_meta[b * 10 + 6],
                                     body_meta[b * 10 + 7], body_meta[b * 10 + 8],
                                     bxq, byq, bzq)
            wb = wp.exp(-(s_b - sdfu) * inv_tau) * inv_Z
            fbx = wb * fdx
            fby = wb * fdy
            fbz = wb * fdz
            ax = wp.float64(xc - kin[b * 21 + 12])
            ay = wp.float64(yc - kin[b * 21 + 13])
            az = wp.float64(zc - kin[b * 21 + 14])
            fbxd = wp.float64(fbx)
            fbyd = wp.float64(fby)
            fbzd = wp.float64(fbz)
            o = b * 12
            wp.atomic_add(out, o + 6, fbxd * h3_d)
            wp.atomic_add(out, o + 7, fbyd * h3_d)
            wp.atomic_add(out, o + 8, fbzd * h3_d)
            wp.atomic_add(out, o + 9, (ay * fbzd - az * fbyd) * h3_d)
            wp.atomic_add(out, o + 10, (az * fbxd - ax * fbzd) * h3_d)
            wp.atomic_add(out, o + 11, (ax * fbyd - ay * fbxd) * h3_d)


# Register float32 + float64 specialisations for the 3-D kernels.
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(forces_post_3d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A, "gz": _A,
        "sdf_cc": _A, "u_prev": _A, "v_prev": _A, "w_prev": _A, "p_prev": _A,
        "nu_rho_field": _A,
        "inv_h": _dt, "eps_body": _dt, "eps_solver": _dt, "h3": _dt,
    })
    wp.overload(forces_post_deltaH_pressure_3d_kernel, {
        "F_flat": _A, "body_meta": _A, "kin": _A, "gx": _A, "gy": _A, "gz": _A,
        "sdf_cc": _A, "p_prev": _A,
        "inv_h": _dt, "inv_eps": _dt, "inv_tau": _dt, "h3": _dt,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  Host wrappers (mirror the native ops.py signatures)
# ═════════════════════════════════════════════════════════════════════════════

def _wp_dtype(t):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _torch_dtype(t):
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _fast_flat(x, wpf, tdt, wdev):
    """Low-overhead zero-copy flat Warp view (~2× cheaper than ``wp.from_torch``
    on the common contiguous/right-dtype path; safe fallback otherwise)."""
    if x.dtype == tdt and x.is_contiguous():
        return wp.array(ptr=x.data_ptr(), dtype=wpf, shape=(x.numel(),),
                        device=wdev)
    return wp.from_torch(x.reshape(-1).contiguous().to(tdt))


def _fast_flat_i64(x, wdev):
    """Low-overhead zero-copy flat int64 Warp view."""
    if x.dtype == torch.int64 and x.is_contiguous():
        return wp.array(ptr=x.data_ptr(), dtype=wp.int64, shape=(x.numel(),),
                        device=wdev)
    return wp.from_torch(x.reshape(-1).contiguous().to(torch.int64))


def streaming_sdf_forces_post_2d_warp(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy,
        h_grid, max_vol_per_body, sdf_cc, interp_method,
        u_prev, v_prev, p_prev, nu_rho_field,
        eps_body, eps_solver, h2, delta_order,
        out, force_submethod=0, ph_tau=0.0):
    """Warp port of ``streaming_sdf_forces_post_2d``.  Writes ``out`` (B*6 or
    (B,6) float64) in place — the native wrapper does NOT zero ``out`` (the
    caller's persistent buffer is pre-zeroed each step), so neither do we."""
    B = int(aabb_dim.shape[0])
    if B <= 0 or int(max_vol_per_body) <= 0:
        return
    wdev = "cuda:0" if F_flat.device.type == "cuda" else "cpu"
    wpf = _wp_dtype(F_flat)
    tdt = _torch_dtype(F_flat)
    Ngx = int(gx.numel())
    Ngy = int(gy.numel())
    with_pressure = 1 if int(force_submethod) == 0 else 0

    def f(x):
        return _fast_flat(x, wpf, tdt, wdev)

    def fi(x):
        return _fast_flat_i64(x, wdev)

    out_w = wp.from_torch(out.reshape(-1))
    wp.launch(
        forces_post_2d_kernel, dim=B * int(max_vol_per_body),
        inputs=[
            f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta), f(kin),
            fi(aabb_lo), fi(aabb_dim), f(gx), f(gy),
            Ngx, Ngy, f(sdf_cc), int(interp_method),
            f(u_prev), f(v_prev), f(p_prev), f(nu_rho_field),
            int(nu_rho_field.numel()),
            wpf(1.0 / h_grid), wpf(eps_body), wpf(eps_solver), wpf(h2),
            int(delta_order), int(with_pressure), int(max_vol_per_body),
            out_w,
        ],
        device=wdev)

    if int(force_submethod) != 0:
        lo = aabb_lo.to("cpu").to(torch.int64)
        dim = aabb_dim.to("cpu").to(torch.int64)
        ulo = [Ngx, Ngy]
        uhi = [0, 0]
        for b in range(B):
            for d in range(2):
                a0 = int(lo[b, d])
                a1 = a0 + int(dim[b, d])
                ulo[d] = min(ulo[d], a0)
                uhi[d] = max(uhi[d], a1)
        Ng = [Ngx, Ngy]
        halo = 2
        for d in range(2):
            ulo[d] = max(ulo[d] - halo, 0)
            uhi[d] = min(uhi[d] + halo, Ng[d])
        ULi = uhi[0] - ulo[0]
        ULj = uhi[1] - ulo[1]
        if ULi > 0 and ULj > 0:
            tau = ph_tau if ph_tau > 0.0 else 1e-9
            wp.launch(
                forces_post_deltaH_pressure_2d_kernel, dim=ULi * ULj,
                inputs=[
                    f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta),
                    f(kin), fi(aabb_lo), fi(aabb_dim), f(gx), f(gy),
                    Ngx, Ngy, f(sdf_cc), int(interp_method), f(p_prev),
                    wpf(1.0 / h_grid), wpf(1.0 / eps_body), wpf(1.0 / tau),
                    wpf(h2), int(B),
                    int(ulo[0]), int(ulo[1]), int(ULi), int(ULj),
                    out_w,
                ],
                device=wdev)
    # No wp.synchronize(): the caller reads ``out`` through torch, which orders
    # after the Warp launch on the (legacy null) default stream — the explicit
    # full-device sync was a per-call latency floor, not a correctness need.


def streaming_sdf_forces_post_3d_warp(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, gz,
        h_grid, max_vol_per_body, sdf_cc, interp_method,
        u_prev, v_prev, w_prev, p_prev, nu_rho_field,
        eps_body, eps_solver, h3, delta_order,
        out, force_submethod=0, ph_tau=0.0):
    """Warp port of ``streaming_sdf_forces_post_3d``.  Accumulates into ``out``
    (B*12 or (B,12) float64) in place (native does not zero it)."""
    B = int(aabb_dim.shape[0])
    if B <= 0 or int(max_vol_per_body) <= 0:
        return
    wdev = "cuda:0" if F_flat.device.type == "cuda" else "cpu"
    wpf = _wp_dtype(F_flat)
    tdt = _torch_dtype(F_flat)
    Ngx = int(gx.numel())
    Ngy = int(gy.numel())
    Ngz = int(gz.numel())
    with_pressure = 1 if int(force_submethod) == 0 else 0

    def f(x):
        return _fast_flat(x, wpf, tdt, wdev)

    def fi(x):
        return _fast_flat_i64(x, wdev)

    out_w = wp.from_torch(out.reshape(-1))
    wp.launch(
        forces_post_3d_kernel, dim=B * int(max_vol_per_body),
        inputs=[
            f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta), f(kin),
            fi(aabb_lo), fi(aabb_dim), f(gx), f(gy), f(gz),
            Ngx, Ngy, Ngz, f(sdf_cc), int(interp_method),
            f(u_prev), f(v_prev), f(w_prev), f(p_prev), f(nu_rho_field),
            int(nu_rho_field.numel()),
            wpf(1.0 / h_grid), wpf(eps_body), wpf(eps_solver), wpf(h3),
            int(delta_order), int(with_pressure), int(max_vol_per_body),
            out_w,
        ],
        device=wdev)

    if int(force_submethod) != 0:
        lo = aabb_lo.to("cpu").to(torch.int64)
        dim = aabb_dim.to("cpu").to(torch.int64)
        ulo = [Ngx, Ngy, Ngz]
        uhi = [0, 0, 0]
        for b in range(B):
            for d in range(3):
                a0 = int(lo[b, d])
                a1 = a0 + int(dim[b, d])
                ulo[d] = min(ulo[d], a0)
                uhi[d] = max(uhi[d], a1)
        Ng = [Ngx, Ngy, Ngz]
        halo = 2
        for d in range(3):
            ulo[d] = max(ulo[d] - halo, 0)
            uhi[d] = min(uhi[d] + halo, Ng[d])
        ULi = uhi[0] - ulo[0]
        ULj = uhi[1] - ulo[1]
        ULk = uhi[2] - ulo[2]
        if ULi > 0 and ULj > 0 and ULk > 0:
            tau = ph_tau if ph_tau > 0.0 else 1e-9
            wp.launch(
                forces_post_deltaH_pressure_3d_kernel, dim=ULi * ULj * ULk,
                inputs=[
                    f(F_flat), fi(F_offsets), fi(body_shapes), f(body_meta),
                    f(kin), fi(aabb_lo), fi(aabb_dim), f(gx), f(gy), f(gz),
                    Ngx, Ngy, Ngz, f(sdf_cc), int(interp_method), f(p_prev),
                    wpf(1.0 / h_grid), wpf(1.0 / eps_body), wpf(1.0 / tau),
                    wpf(h3), int(B),
                    int(ulo[0]), int(ulo[1]), int(ulo[2]),
                    int(ULi), int(ULj), int(ULk),
                    out_w,
                ],
                device=wdev)
    # No wp.synchronize(): the caller reads ``out`` through torch, which orders
    # after the Warp launch on the (legacy null) default stream — the explicit
    # full-device sync was a per-call latency floor, not a correctness need.


# In-module aliases (historical short names) used by the force_method2 call
# sites above.
streaming_sdf_forces_post_2d = streaming_sdf_forces_post_2d_warp
streaming_sdf_forces_post_3d = streaming_sdf_forces_post_3d_warp
