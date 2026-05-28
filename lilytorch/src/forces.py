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
import torch

from lilytorch.src import operations as ops
from lilytorch.src.kernels import (
    lagrangian_forces_2d as _lagrangian_forces_2d_kernel,
    lagrangian_forces_3d as _lagrangian_forces_3d_kernel,
    streaming_sdf_forces_post_2d,
    streaming_sdf_forces_post_3d,
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

    # ---- CC normals (computed on-the-fly, not cached on self) ------
    normal_x, normal_y = self.composite_body.compute_normals(
        self.composite_body.sdf_val
    )

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
    _use_legacy_sparse_forces_2d = False
    _use_kernel_post_forces_2d = (
        self._use_kernels
        and not _have_sparse_2d
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
    # The 2-D update path recomputes and caches these in
    # ``_recompute_mu_normals``.  Reuse them here instead of launching
    # another full-grid normal computation for the force pass.
    normal_x = getattr(self, 'normal_x', None)
    normal_y = getattr(self, 'normal_y', None)
    if normal_x is None or normal_y is None:
        normal_x, normal_y = comp.compute_normals(comp.sdf_val)

    # ==============================================================
    # BATCHED path — all bodies in one fused call
    # ==============================================================
    # Shared part: stress + pressure force density.  In the streaming
    # force path, crop this bandwidth-heavy tensor work to the union of
    # body AABBs (with a halo for derivative stencils) and let the sparse
    # force kernel index relative to that cropped slab.
    u_aabb = None
    if (
        _use_legacy_sparse_forces_2d
        and self._use_kernels
    ):
        u_i0, u_j0 = 1 << 30, 1 << 30
        u_i1, u_j1 = -1, -1
        for aabb_i, _ in comp._sdf_sparse:
            if aabb_i is None:
                u_i0 = -1
                break
            i0, i1, j0, j1 = aabb_i
            if i0 < u_i0: u_i0 = i0
            if j0 < u_j0: u_j0 = j0
            if i1 > u_i1: u_i1 = i1
            if j1 > u_j1: u_j1 = j1
        if u_i0 != -1:
            Ni, Nj = u.shape
            halo = 2
            u_i0 = max(0, u_i0 - halo); u_i1 = min(Ni, u_i1 + halo)
            u_j0 = max(0, u_j0 - halo); u_j1 = min(Nj, u_j1 + halo)
            if u_i1 > u_i0 and u_j1 > u_j0:
                u_aabb = (u_i0, u_i1, u_j0, u_j1)

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
    if _use_legacy_sparse_forces_2d:
        self.xstress_tensor = None
        self.ystress_tensor = None
        self.pforce_x = None
        self.pforce_y = None
    else:
        self.xstress_tensor = xstress
        self.ystress_tensor = ystress
        self.pforce_x = pforce_x
        self.pforce_y = pforce_y

    eps_body = comp.bodies[0].eps



    # Towers (2008) 2nd-order: compute per-body |∇SDF| on CC grid  (B,Ni,Nj)
    # ``comp.sdf_vals`` was the legacy dense per-body SDF stack.  The new
    # 2-D Python update (``_update_2d``) populates ``comp._sdf_sparse``
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
    else:
        sdf_vals = comp.sdf_vals  # legacy dense path

    sdf_grad_mag_2d = None
    if self.force_delta_order == 2:
        gx = torch.gradient(sdf_vals, spacing=self.h, dim=1, edge_order=2)[0]
        gy = torch.gradient(sdf_vals, spacing=self.h, dim=2, edge_order=2)[0]
        sdf_grad_mag_2d = torch.sqrt(gx**2 + gy**2)

    fv, tv, fp, tp = self._forces_body_batch_compiled(
        (xstress, ystress),
        (pforce_x, pforce_y),
        sdf_vals,
        eps_body, self.eps,
        (comp.com_pos[:, 0], comp.com_pos[:, 1]),
        (comp._grids.X, comp._grids.Y), self.h2,
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

    Viscous force:
        F_visc = ∫ (σ · n) δ_ε(d - ε) dV
    where σ_{ij} = ν ρ (∂u_i/∂x_j + ∂u_j/∂x_i) is the viscous stress.

    Pressure force:
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
        self._use_kernels
        and _stream_step is not None
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
    nx = getattr(self, 'normal_x', None)
    if nx is None:
        nx, ny, nz = comp.compute_normals(comp.sdf_val)
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
        self._use_kernels
        and hasattr(comp, '_sdf_sparse')
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


def _contour_normal_2d(cnt_update):
    """Outward unit normal at each 2-D contour point.

    Computes ``n = (t_y, -t_x)`` where ``t = ∂c/∂s`` is the unit tangent
    obtained from central differences on the (assumed-ordered) contour
    point list.

    The CCW orientation convention is fixed by the analytical / mesh
    body initialisers: marching-squares walks counter-clockwise around
    the zero level set, so ``(t_y, -t_x)`` is the outward normal.
    """
    cx = cnt_update[0]
    cy = cnt_update[1]
    # central diff; cnt is closed → wrap with roll
    tx = (torch.roll(cx, -1, 0) - torch.roll(cx, 1, 0)) * 0.5
    ty = (torch.roll(cy, -1, 0) - torch.roll(cy, 1, 0)) * 0.5
    L = torch.sqrt(tx * tx + ty * ty).clamp_min(1e-30)
    tx = tx / L
    ty = ty / L
    # outward normal (CCW orientation ⇒ outward = +90° rotation of tangent)
    nx = ty
    ny = -tx
    return nx, ny


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
    nu_rho = self._compute_nu_rho_for_forces(u, v)
    eps_ij = _viscous_stress_tensor((u, v), h)

    # Pack per-body contours into a single (2, M_total) tensor with int64
    # offsets.  Skip degenerate bodies (M <= 1) by emitting an empty slice
    # for them -- the kernel zeros their output row.
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

    # nu_rho may be a Python float (constant viscosity) or a CC tensor.
    if torch.is_tensor(nu_rho) and nu_rho.ndim == 2:
        nu_rho_field = nu_rho
    else:
        nu_rho_field = torch.tensor(
            [float(nu_rho)], dtype=p.dtype, device=p.device)

    grids = comp._grids
    Mx = int(grids.x.numel())
    My = int(grids.y.numel())
    bx0 = float(grids.x[0]); by0 = float(grids.y[0])
    inv_dx = 1.0 / h; inv_dy = 1.0 / h

    out = _lagrangian_forces_2d_kernel(
        eps_ij[0][0], eps_ij[0][1], eps_ij[1][1],
        p, nu_rho_field,
        cnt_flat, cnt_offsets, com_pos,
        bx0, by0, inv_dx, inv_dy,
        Mx, My, method="linear",
    )

    # Scatter the (B, 6) kernel output into the legacy per-body force
    # caches and per-iteration records.
    for bi in range(B):
        fv_x = out[bi, 0].to(self.dtype)
        fv_y = out[bi, 1].to(self.dtype)
        tv_t = out[bi, 2].to(self.dtype)
        fp_x = out[bi, 3].to(self.dtype)
        fp_y = out[bi, 4].to(self.dtype)
        tp_t = out[bi, 5].to(self.dtype)

        self.friction_force_lin_x[bi] = fv_x
        self.friction_force_lin_y[bi] = fv_y
        self.friction_force_ang_z[bi] = tv_t
        self.pressure_force_x[bi]     = fp_x
        self.pressure_force_y[bi]     = fp_y
        self.pressure_force_ang_z[bi] = tp_t

        self.viscous_drag_record[bi, 0, iteration]  = fv_x
        self.viscous_drag_record[bi, 1, iteration]  = fv_y
        self.pressure_drag_record[bi, 0, iteration] = fp_x
        self.pressure_drag_record[bi, 1, iteration] = fp_y

    # Lagrangian path doesn't expose volumetric stress / pforce tensors.
    self.xstress_tensor = None
    self.ystress_tensor = None
    self.pforce_x = None
    self.pforce_y = None
    comp = self.composite_body
    B = len(comp.bodies)
    h = self.h
    nu_rho = self._compute_nu_rho_for_forces(u, v)
    # Use scalar viscosity when constant; lagrangian path doesn't need a
    # full nu_rho field on the band -- it's sampled at the marker points.
    nu_rho_const = (
        not (torch.is_tensor(nu_rho) and nu_rho.ndim == 2)
    )

    # Symmetric strain-rate tensor ε_ij at CC; viscous stress σ = ν·ρ·ε.
    eps_ij = _viscous_stress_tensor((u, v), h)

    # Build per-axis interpolators (RegularGridInterpolator handles
    # 2-D/3-D and runs on CPU or CUDA via the existing dispatched
    # ``interp_2d`` / ``interp_3d`` ops).
    from lilytorch.src.kernels import RegularGridInterpolator
    grids = comp._grids
    axes_2d = (grids.x, grids.y)

    # Bilinear interpolation of σ_ij and p at every contour marker of
    # every body.  We share one interpolator instance per field by
    # reassigning ``.F`` between bodies -- this avoids re-baking metadata.
    interp = RegularGridInterpolator(axes_2d, p, method="linear")

    def _sample(field, qx, qy):
        interp.F = field.contiguous()
        return interp(qx, qy)

    # If nu_rho is a CC tensor, sample it at markers; else broadcast scalar.
    for bi, body in enumerate(comp.bodies):
        cnt = body.cnt_update                    # (2, M)
        if cnt.shape[1] <= 1:
            # Degenerate body (no surface contour); skip.
            self.friction_force_lin_x[bi] = 0
            self.friction_force_lin_y[bi] = 0
            self.friction_force_ang_z[bi] = 0
            self.pressure_force_x[bi] = 0
            self.pressure_force_y[bi] = 0
            self.pressure_force_ang_z[bi] = 0
            continue

        qx = cnt[0]
        qy = cnt[1]
        nx, ny = _contour_normal_2d(cnt)

        # Sample ε_ij at markers
        e_xx = _sample(eps_ij[0][0], qx, qy)
        e_xy = _sample(eps_ij[0][1], qx, qy)
        e_yy = _sample(eps_ij[1][1], qx, qy)
        if nu_rho_const:
            nu_rho_m = nu_rho  # scalar
        else:
            nu_rho_m = _sample(nu_rho, qx, qy)

        # Viscous traction t_v = ν·ρ·(ε_ij n_j)
        tvx = nu_rho_m * (e_xx * nx + e_xy * ny)
        tvy = nu_rho_m * (e_xy * nx + e_yy * ny)

        # Pressure traction t_p = -p n
        p_m = _sample(p, qx, qy)
        tpx = -p_m * nx
        tpy = -p_m * ny

        # Arc-length integration over the (closed) contour.  ``ds`` from
        # consecutive cnt distances; trapz with the wrap-around endpoint.
        dx = torch.roll(qx, -1, 0) - qx
        dy = torch.roll(qy, -1, 0) - qy
        ds_seg = torch.sqrt(dx * dx + dy * dy)  # (M,)
        # midpoint quadrature: F = Σ_seg 0.5*(f_i + f_{i+1}) * ds_seg
        def _line_integral(f):
            return 0.5 * ((f + torch.roll(f, -1, 0)) * ds_seg).sum()

        fv_x = _line_integral(tvx)
        fv_y = _line_integral(tvy)
        fp_x = _line_integral(tpx)
        fp_y = _line_integral(tpy)

        # Torques about com
        com = body.com_pos
        rx = qx - com[0]
        ry = qy - com[1]
        tv_torque = _line_integral(rx * tvy - ry * tvx)
        tp_torque = _line_integral(rx * tpy - ry * tpx)

        self.friction_force_lin_x[bi] = fv_x.to(self.dtype)
        self.friction_force_lin_y[bi] = fv_y.to(self.dtype)
        self.friction_force_ang_z[bi] = tv_torque.to(self.dtype)
        self.pressure_force_x[bi] = fp_x.to(self.dtype)
        self.pressure_force_y[bi] = fp_y.to(self.dtype)
        self.pressure_force_ang_z[bi] = tp_torque.to(self.dtype)

        self.viscous_drag_record[bi, 0, iteration]  = self.friction_force_lin_x[bi]
        self.viscous_drag_record[bi, 1, iteration]  = self.friction_force_lin_y[bi]
        self.pressure_drag_record[bi, 0, iteration] = self.pressure_force_x[bi]
        self.pressure_drag_record[bi, 1, iteration] = self.pressure_force_y[bi]

    # Lagrangian path doesn't expose volumetric stress / pforce tensors.
    self.xstress_tensor = None
    self.ystress_tensor = None
    self.pforce_x = None
    self.pforce_y = None


def _forces_lagrangian_2d_python_ref(self, u, v, p, iteration):
    """Pure-PyTorch reference implementation (kept for tests / debugging).

    The production path :func:`forces_lagrangian_2d` dispatches to the
    fused ``lagrangian_forces_2d`` C++/CUDA kernel.  This reference
    mirrors that kernel one-to-one and is what was used to validate it.
    """

    # Build per-axis interpolators (RegularGridInterpolator handles
    # 2-D/3-D and runs on CPU or CUDA via the existing dispatched
    # ``interp_2d`` / ``interp_3d`` ops).
    from lilytorch.src.kernels import RegularGridInterpolator
    grids = comp._grids
    axes_2d = (grids.x, grids.y)

    # Bilinear interpolation of σ_ij and p at every contour marker of
    # every body.  We share one interpolator instance per field by
    # reassigning ``.F`` between bodies -- this avoids re-baking metadata.
    interp = RegularGridInterpolator(axes_2d, p, method="linear")

    def _sample(field, qx, qy):
        interp.F = field.contiguous()
        return interp(qx, qy)

    # If nu_rho is a CC tensor, sample it at markers; else broadcast scalar.
    for bi, body in enumerate(comp.bodies):
        cnt = body.cnt_update                    # (2, M)
        if cnt.shape[1] <= 1:
            # Degenerate body (no surface contour); skip.
            self.friction_force_lin_x[bi] = 0
            self.friction_force_lin_y[bi] = 0
            self.friction_force_ang_z[bi] = 0
            self.pressure_force_x[bi] = 0
            self.pressure_force_y[bi] = 0
            self.pressure_force_ang_z[bi] = 0
            continue

        qx = cnt[0]
        qy = cnt[1]
        nx, ny = _contour_normal_2d(cnt)

        # Sample ε_ij at markers
        e_xx = _sample(eps_ij[0][0], qx, qy)
        e_xy = _sample(eps_ij[0][1], qx, qy)
        e_yy = _sample(eps_ij[1][1], qx, qy)
        if nu_rho_const:
            nu_rho_m = nu_rho  # scalar
        else:
            nu_rho_m = _sample(nu_rho, qx, qy)

        # Viscous traction t_v = ν·ρ·(ε_ij n_j)
        tvx = nu_rho_m * (e_xx * nx + e_xy * ny)
        tvy = nu_rho_m * (e_xy * nx + e_yy * ny)

        # Pressure traction t_p = -p n
        p_m = _sample(p, qx, qy)
        tpx = -p_m * nx
        tpy = -p_m * ny

        # Arc-length integration over the (closed) contour.  ``ds`` from
        # consecutive cnt distances; trapz with the wrap-around endpoint.
        dx = torch.roll(qx, -1, 0) - qx
        dy = torch.roll(qy, -1, 0) - qy
        ds_seg = torch.sqrt(dx * dx + dy * dy)  # (M,)
        # midpoint quadrature: F = Σ_seg 0.5*(f_i + f_{i+1}) * ds_seg
        def _line_integral(f):
            return 0.5 * ((f + torch.roll(f, -1, 0)) * ds_seg).sum()

        fv_x = _line_integral(tvx)
        fv_y = _line_integral(tvy)
        fp_x = _line_integral(tpx)
        fp_y = _line_integral(tpy)

        # Torques about com
        com = body.com_pos
        rx = qx - com[0]
        ry = qy - com[1]
        tv_torque = _line_integral(rx * tvy - ry * tvx)
        tp_torque = _line_integral(rx * tpy - ry * tpx)

        self.friction_force_lin_x[bi] = fv_x.to(self.dtype)
        self.friction_force_lin_y[bi] = fv_y.to(self.dtype)
        self.friction_force_ang_z[bi] = tv_torque.to(self.dtype)
        self.pressure_force_x[bi] = fp_x.to(self.dtype)
        self.pressure_force_y[bi] = fp_y.to(self.dtype)
        self.pressure_force_ang_z[bi] = tp_torque.to(self.dtype)

        self.viscous_drag_record[bi, 0, iteration]  = self.friction_force_lin_x[bi]
        self.viscous_drag_record[bi, 1, iteration]  = self.friction_force_lin_y[bi]
        self.pressure_drag_record[bi, 0, iteration] = self.pressure_force_x[bi]
        self.pressure_drag_record[bi, 1, iteration] = self.pressure_force_y[bi]

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
    nu_rho = self._compute_nu_rho_for_forces(u, v, w)
    eps_ij = _viscous_stress_tensor((u, v, w), h)

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

    if torch.is_tensor(nu_rho) and nu_rho.ndim == 3:
        nu_rho_field = nu_rho
    else:
        nu_rho_field = torch.tensor(
            [float(nu_rho)], dtype=p.dtype, device=p.device)

    grids = comp._grids
    Mx = int(grids.x.numel()); My = int(grids.y.numel()); Mz = int(grids.z.numel())
    bx0 = float(grids.x[0]); by0 = float(grids.y[0]); bz0 = float(grids.z[0])
    inv_dx = 1.0 / h; inv_dy = 1.0 / h; inv_dz = 1.0 / h

    out = _lagrangian_forces_3d_kernel(
        eps_ij[0][0], eps_ij[1][1], eps_ij[2][2],
        eps_ij[0][1], eps_ij[0][2], eps_ij[1][2],
        p, nu_rho_field,
        tri_centroid, tri_normal, tri_area,
        tri_offsets, com_pos,
        bx0, by0, bz0, inv_dx, inv_dy, inv_dz,
        Mx, My, Mz, method="linear",
    )

    # Scatter (B, 12) into legacy per-body caches and per-iteration records.
    for bi in range(B):
        fv_x = out[bi, 0].to(self.dtype); fv_y = out[bi, 1].to(self.dtype); fv_z = out[bi, 2].to(self.dtype)
        tv_x = out[bi, 3].to(self.dtype); tv_y = out[bi, 4].to(self.dtype); tv_z = out[bi, 5].to(self.dtype)
        fp_x = out[bi, 6].to(self.dtype); fp_y = out[bi, 7].to(self.dtype); fp_z = out[bi, 8].to(self.dtype)
        tp_x = out[bi, 9].to(self.dtype); tp_y = out[bi, 10].to(self.dtype); tp_z = out[bi, 11].to(self.dtype)

        self.friction_force_lin_x[bi] = fv_x; self.friction_force_lin_y[bi] = fv_y; self.friction_force_lin_z[bi] = fv_z
        self.friction_force_ang_x[bi] = tv_x; self.friction_force_ang_y[bi] = tv_y; self.friction_force_ang_z[bi] = tv_z
        self.pressure_force_x[bi] = fp_x;     self.pressure_force_y[bi] = fp_y;     self.pressure_force_z[bi] = fp_z
        self.pressure_force_ang_x[bi] = tp_x; self.pressure_force_ang_y[bi] = tp_y; self.pressure_force_ang_z[bi] = tp_z

        self.viscous_drag_record[bi, 0, iteration]   = fv_x
        self.viscous_drag_record[bi, 1, iteration]   = fv_y
        self.viscous_drag_record[bi, 2, iteration]   = fv_z
        self.viscous_torque_record[bi, 0, iteration] = tv_x
        self.viscous_torque_record[bi, 1, iteration] = tv_y
        self.viscous_torque_record[bi, 2, iteration] = tv_z
        self.pressure_drag_record[bi, 0, iteration]   = fp_x
        self.pressure_drag_record[bi, 1, iteration]   = fp_y
        self.pressure_drag_record[bi, 2, iteration]   = fp_z
        self.pressure_torque_record[bi, 0, iteration] = tp_x
        self.pressure_torque_record[bi, 1, iteration] = tp_y
        self.pressure_torque_record[bi, 2, iteration] = tp_z

    self.xstress_tensor = None
    self.ystress_tensor = None
    self.zstress_tensor = None
    self.pforce_x = None
    self.pforce_y = None
    self.pforce_z = None


def _forces_lagrangian_3d_python_ref(self, u, v, w, p, iteration):
    """Pure-PyTorch reference implementation (kept for tests / debugging).

    The production path :func:`forces_lagrangian_3d` dispatches to the
    fused ``lagrangian_forces_3d`` C++/CUDA kernel.
    """
    comp = self.composite_body
    B = len(comp.bodies)
    h = self.h
    nu_rho = self._compute_nu_rho_for_forces(u, v, w)
    nu_rho_const = not (torch.is_tensor(nu_rho) and nu_rho.ndim == 3)

    eps_ij = _viscous_stress_tensor((u, v, w), h)

    from lilytorch.src.kernels import RegularGridInterpolator
    grids = comp._grids
    axes_3d = (grids.x, grids.y, grids.z)
    interp = RegularGridInterpolator(axes_3d, p, method="linear")

    def _sample(field, qx, qy, qz):
        interp.F = field.contiguous()
        return interp(qx, qy, qz)

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
        qx = tri_c[0]; qy = tri_c[1]; qz = tri_c[2]
        nx = tri_n[0]; ny = tri_n[1]; nz = tri_n[2]
        area = tri_a

        e_xx = _sample(eps_ij[0][0], qx, qy, qz)
        e_yy = _sample(eps_ij[1][1], qx, qy, qz)
        e_zz = _sample(eps_ij[2][2], qx, qy, qz)
        e_xy = _sample(eps_ij[0][1], qx, qy, qz)
        e_xz = _sample(eps_ij[0][2], qx, qy, qz)
        e_yz = _sample(eps_ij[1][2], qx, qy, qz)
        if nu_rho_const:
            nu_rho_m = nu_rho
        else:
            nu_rho_m = _sample(nu_rho, qx, qy, qz)

        tvx = nu_rho_m * (e_xx * nx + e_xy * ny + e_xz * nz)
        tvy = nu_rho_m * (e_xy * nx + e_yy * ny + e_yz * nz)
        tvz = nu_rho_m * (e_xz * nx + e_yz * ny + e_zz * nz)

        p_m = _sample(p, qx, qy, qz)
        tpx = -p_m * nx; tpy = -p_m * ny; tpz = -p_m * nz

        fv_x = (tvx * area).sum(); fv_y = (tvy * area).sum(); fv_z = (tvz * area).sum()
        fp_x = (tpx * area).sum(); fp_y = (tpy * area).sum(); fp_z = (tpz * area).sum()

        com = body.com_pos
        rx = qx - com[0]; ry = qy - com[1]; rz = qz - com[2]
        tv_x = ((ry * tvz - rz * tvy) * area).sum()
        tv_y = ((rz * tvx - rx * tvz) * area).sum()
        tv_z = ((rx * tvy - ry * tvx) * area).sum()
        tp_x = ((ry * tpz - rz * tpy) * area).sum()
        tp_y = ((rz * tpx - rx * tpz) * area).sum()
        tp_z = ((rx * tpy - ry * tpx) * area).sum()

        self.friction_force_lin_x[bi] = fv_x.to(self.dtype)
        self.friction_force_lin_y[bi] = fv_y.to(self.dtype)
        self.friction_force_lin_z[bi] = fv_z.to(self.dtype)
        self.friction_force_ang_x[bi] = tv_x.to(self.dtype)
        self.friction_force_ang_y[bi] = tv_y.to(self.dtype)
        self.friction_force_ang_z[bi] = tv_z.to(self.dtype)
        self.pressure_force_x[bi] = fp_x.to(self.dtype)
        self.pressure_force_y[bi] = fp_y.to(self.dtype)
        self.pressure_force_z[bi] = fp_z.to(self.dtype)
        self.pressure_force_ang_x[bi] = tp_x.to(self.dtype)
        self.pressure_force_ang_y[bi] = tp_y.to(self.dtype)
        self.pressure_force_ang_z[bi] = tp_z.to(self.dtype)

        self.viscous_drag_record[bi, 0, iteration]   = self.friction_force_lin_x[bi]
        self.viscous_drag_record[bi, 1, iteration]   = self.friction_force_lin_y[bi]
        self.viscous_drag_record[bi, 2, iteration]   = self.friction_force_lin_z[bi]
        self.viscous_torque_record[bi, 0, iteration] = self.friction_force_ang_x[bi]
        self.viscous_torque_record[bi, 1, iteration] = self.friction_force_ang_y[bi]
        self.viscous_torque_record[bi, 2, iteration] = self.friction_force_ang_z[bi]
        self.pressure_drag_record[bi, 0, iteration]   = self.pressure_force_x[bi]
        self.pressure_drag_record[bi, 1, iteration]   = self.pressure_force_y[bi]
        self.pressure_drag_record[bi, 2, iteration]   = self.pressure_force_z[bi]
        self.pressure_torque_record[bi, 0, iteration] = self.pressure_force_ang_x[bi]
        self.pressure_torque_record[bi, 1, iteration] = self.pressure_force_ang_y[bi]
        self.pressure_torque_record[bi, 2, iteration] = self.pressure_force_ang_z[bi]

    self.xstress_tensor = None
    self.ystress_tensor = None
    self.zstress_tensor = None
    self.pforce_x = None
    self.pforce_y = None
    self.pforce_z = None
