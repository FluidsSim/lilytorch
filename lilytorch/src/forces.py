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


# ======================================================================
# Compilable force-computation kernels  (module-level, for torch.compile)
# ======================================================================

def _forces_shared(vels, p, normals, nu_rho, h):
    """Dim-agnostic shared stress·n and pressure force density.

    Inputs are pure tensors / scalars — no ``self`` access — so this
    function is safe for ``torch.compile(mode='reduce-overhead')``.  The
    Python-level loops over ``range(D)`` unroll at trace time because the
    per-D wrappers (:func:`_forces_shared_2d`, :func:`_forces_shared_3d`)
    fix the component count when compiled, so each compiled artifact still
    sees a fully unrolled, shape-specialized graph.

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
    Flat tuple ``(stress_0, …, stress_{D-1}, pforce_0, …, pforce_{D-1})``.
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

    return tuple(stresses) + tuple(pforces)


def _forces_shared_3d(u, v, w, p, sdf_val, nx, ny, nz, nu_rho, h):
    """3-D wrapper around :func:`_forces_shared` (kept for compile shape
    specialization).  ``sdf_val`` is unused; preserved for API stability."""
    s0, s1, s2, p0, p1, p2 = _forces_shared(
        (u, v, w), p, (nx, ny, nz), nu_rho, h)
    return s0, s1, s2, p0, p1, p2


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

    return fv, tv, fp, tp


def _forces_body_batch_3d(
    xstress, ystress, zstress,
    pforce_x, pforce_y, pforce_z,
    sdf_all, eps_body, eps_solver,
    com_x, com_y, com_z,
    X, Y, Z, h3,
    sdf_grad_mag=None,
):
    """3-D wrapper around :func:`_forces_body_batch`.  Returns 12 tensors
    of shape (B,): fv_xyz, tv_xyz, fp_xyz, tp_xyz."""
    fv, tv, fp, tp = _forces_body_batch(
        [xstress, ystress, zstress], [pforce_x, pforce_y, pforce_z],
        sdf_all, eps_body, eps_solver,
        [com_x, com_y, com_z], [X, Y, Z], h3, sdf_grad_mag,
    )
    return (fv[0], fv[1], fv[2], tv[0], tv[1], tv[2],
            fp[0], fp[1], fp[2], tp[0], tp[1], tp[2])


# ======================================================================
# 2-D wrappers — keep distinct symbols so torch.compile produces a
# shape-specialized graph per dimension.
# ======================================================================

def _forces_shared_2d(u, v, p, sdf_val, nx, ny, nu_rho, h):
    """2-D wrapper around :func:`_forces_shared`.  ``sdf_val`` is unused;
    preserved for API stability."""
    s0, s1, p0, p1 = _forces_shared((u, v), p, (nx, ny), nu_rho, h)
    return s0, s1, p0, p1


def _forces_body_batch_2d(
    xstress, ystress,
    pforce_x, pforce_y,
    sdf_vals_cc,
    eps_body, eps_solver,
    com_x, com_y,
    X, Y, h2,
    sdf_grad_mag=None,
):
    """2-D wrapper around :func:`_forces_body_batch`.  Returns 6 tensors of
    shape (B,): fv_x, fv_y, tv_z, fp_x, fp_y, tp_z."""
    fv, tv, fp, tp = _forces_body_batch(
        [xstress, ystress], [pforce_x, pforce_y],
        sdf_vals_cc, eps_body, eps_solver,
        [com_x, com_y], [X, Y], h2, sdf_grad_mag,
    )
    return fv[0], fv[1], tv[0], fp[0], fp[1], tp[0]

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
    (xstress, ystress,
     pforce_x, pforce_y) = self._forces_shared_2d_compiled(
        u, v, p, self.composite_body.sdf_val,
        normal_x, normal_y,
        nu_rho, self.h,
    )

    if self._compile_forces:
        xstress = xstress.clone()
        ystress = ystress.clone()
        pforce_x = pforce_x.clone()
        pforce_y = pforce_y.clone()

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
        from lilytorch.src.kernels import streaming_sdf_forces_post_2d

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
    # ``_recompute_mu_normals_2d``.  Reuse them here instead of launching
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
        (xstress, ystress,
         pforce_x, pforce_y) = self._forces_shared_2d_compiled(
            u[usl].contiguous(), v[usl].contiguous(),
            p[usl].contiguous(), comp.sdf_val[usl].contiguous(),
            normal_x[usl].contiguous(), normal_y[usl].contiguous(),
            nu_rho_sub, self.h,
        )
    else:
        u_i0, u_j0 = 0, 0
        (xstress, ystress,
         pforce_x, pforce_y) = self._forces_shared_2d_compiled(
            u, v, p, comp.sdf_val,
            normal_x, normal_y,
            nu_rho, self.h,
        )

    if self._compile_forces:
        xstress  = xstress.clone()
        ystress  = ystress.clone()
        pforce_x = pforce_x.clone()
        pforce_y = pforce_y.clone()

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

    (fv_x, fv_y, tv_z,
     fp_x, fp_y, tp_z) = self._forces_body_batch_2d_compiled(
        xstress, ystress,
        pforce_x, pforce_y,
        sdf_vals,
        eps_body, self.eps,
        comp.com_pos[:, 0], comp.com_pos[:, 1],
        self.grids.X, self.grids.Y, self.h2,
        sdf_grad_mag_2d,
    )

    if self._compile_forces:
        fv_x = fv_x.clone(); fv_y = fv_y.clone(); tv_z = tv_z.clone()
        fp_x = fp_x.clone(); fp_y = fp_y.clone(); tp_z = tp_z.clone()

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

    When ``compile_forces=True``, the heavy tensor work is delegated to
    two ``torch.compile``-d kernels (``_forces_shared_3d`` and
    ``_forces_body_integrate_3d``) that fuse ~40 CUDA kernels into one
    or two CUDA-graph launches, giving ~6× wall-clock speedup.
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
        from lilytorch.src.kernels import streaming_sdf_forces_post_3d
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
    # the bandwidth-bound _forces_shared_3d only on that sub-block.
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
        (xstress, ystress, zstress,
         pforce_x, pforce_y, pforce_z) = self._forces_shared_dyn_compiled(
            u[usl].contiguous(), v[usl].contiguous(), w[usl].contiguous(),
            p[usl].contiguous(), comp.sdf_val[usl].contiguous(),
            nx[usl].contiguous(), ny[usl].contiguous(), nz[usl].contiguous(),
            nu_rho_sub, h,
        )
    else:
        # ---- full-grid shared kernel (default path) --------------
        (xstress, ystress, zstress,
         pforce_x, pforce_y, pforce_z) = self._forces_shared_compiled(
            u, v, w, p, comp.sdf_val, nx, ny, nz,
            nu_rho, h,
        )

    # When compiled with CUDA graphs (reduce-overhead), the returned
    # tensors live in the graph's replay buffer and will be overwritten
    # by the next compiled call.  Clone them here so the body-integrate
    # kernel can safely read them.
    #
    if self._compile_forces:
        xstress  = xstress.clone()
        ystress  = ystress.clone()
        zstress  = zstress.clone()
        pforce_x = pforce_x.clone()
        pforce_y = pforce_y.clone()
        pforce_z = pforce_z.clone()

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

    # Prefer the narrow-band per-body path whenever sparse SDFs are
    # available: it integrates each body only over its AABB sub-block
    # (~50³ cells × B) instead of scattering to (B, Nx, Ny, Nz) and
    # reducing over the full volume.  This is ~50–100× less work for
    # typical thin-body geometries and removes the large Triton kernel
    # shape-dependent cliff seen on 512×128×128 grids.
    _have_sparse = (
        hasattr(comp, '_sdf_sparse')
        and len(comp._sdf_sparse) > 0
        and comp._sdf_sparse[0] is not None
    )




    if self._compile_forces:
        # ---- BATCHED path (compiled CUDA graph) ----------------
        # Process all bodies in one fused kernel launch.
        # Reconstruct dense sdf_vals from sparse storage if needed.
        if hasattr(comp, '_sdf_sparse') and comp._sdf_sparse[0] is not None:
            _FAR = 1e6
            sdf_all = torch.full(
                (B, *self.grid_shape), _FAR,
                device=self.device, dtype=self.dtype,
            )
            for bi in range(B):
                aabb_i, sdf_sub_i = comp._sdf_sparse[bi]
                if aabb_i is not None:
                    i0, i1, j0, j1, k0, k1 = aabb_i
                    sdf_all[bi, i0:i1, j0:j1, k0:k1] = sdf_sub_i
                else:
                    sdf_all[bi] = sdf_sub_i
        else:
            sdf_all = comp.sdf_vals                # legacy dense path
        eps_body = comp.bodies[0].eps

        if not hasattr(self, '_com_buf_x'):
            self._com_buf_x = torch.empty(B, device=self.device, dtype=self.dtype)
            self._com_buf_y = torch.empty(B, device=self.device, dtype=self.dtype)
            self._com_buf_z = torch.empty(B, device=self.device, dtype=self.dtype)
        for i, body in enumerate(comp.bodies):
            self._com_buf_x[i] = body.com_pos[0]
            self._com_buf_y[i] = body.com_pos[1]
            self._com_buf_z[i] = body.com_pos[2]

        # Towers (2008) 2nd-order: per-body |∇SDF|  (B, Ni, Nj, Nk)
        sdf_grad_mag_3d = None
        if self.force_delta_order == 2:
            gx = torch.gradient(sdf_all, spacing=h, dim=1, edge_order=2)[0]
            gy = torch.gradient(sdf_all, spacing=h, dim=2, edge_order=2)[0]
            gz = torch.gradient(sdf_all, spacing=h, dim=3, edge_order=2)[0]
            sdf_grad_mag_3d = torch.sqrt(gx**2 + gy**2 + gz**2)

        (fv_x, fv_y, fv_z,
         tv_x, tv_y, tv_z,
         fp_x, fp_y, fp_z,
         tp_x, tp_y, tp_z) = self._forces_body_batch_compiled(
            xstress, ystress, zstress,
            pforce_x, pforce_y, pforce_z,
            sdf_all, eps_body, self.eps,
            self._com_buf_x, self._com_buf_y, self._com_buf_z,
            X, Y, Z, h3,
            sdf_grad_mag_3d,
        )

        # Clone outputs (CUDA graph buffer reuse)
        fv_x = fv_x.clone(); fv_y = fv_y.clone(); fv_z = fv_z.clone()
        tv_x = tv_x.clone(); tv_y = tv_y.clone(); tv_z = tv_z.clone()
        fp_x = fp_x.clone(); fp_y = fp_y.clone(); fp_z = fp_z.clone()
        tp_x = tp_x.clone(); tp_y = tp_y.clone(); tp_z = tp_z.clone()

        for i in range(B):
            self.friction_force_lin_x[i] = fv_x[i]
            self.friction_force_lin_y[i] = fv_y[i]
            self.friction_force_lin_z[i] = fv_z[i]
            self.friction_force_ang_x[i] = tv_x[i]
            self.friction_force_ang_y[i] = tv_y[i]
            self.friction_force_ang_z[i] = tv_z[i]
            self.pressure_force_x[i] = fp_x[i]
            self.pressure_force_y[i] = fp_y[i]
            self.pressure_force_z[i] = fp_z[i]
            self.pressure_force_ang_x[i] = tp_x[i]
            self.pressure_force_ang_y[i] = tp_y[i]
            self.pressure_force_ang_z[i] = tp_z[i]
            self.viscous_drag_record[i, 0, iteration]  = fv_x[i]
            self.viscous_drag_record[i, 1, iteration]  = fv_y[i]
            self.viscous_drag_record[i, 2, iteration]  = fv_z[i]
            self.pressure_drag_record[i, 0, iteration] = fp_x[i]
            self.pressure_drag_record[i, 1, iteration] = fp_y[i]
            self.pressure_drag_record[i, 2, iteration] = fp_z[i]
            self.viscous_torque_record[i, 0, iteration]  = tv_x[i]
            self.viscous_torque_record[i, 1, iteration]  = tv_y[i]
            self.viscous_torque_record[i, 2, iteration]  = tv_z[i]
            self.pressure_torque_record[i, 0, iteration] = tp_x[i]
            self.pressure_torque_record[i, 1, iteration] = tp_y[i]
            self.pressure_torque_record[i, 2, iteration] = tp_z[i]
    else:
        # ---- PER-BODY path (un-compiled fallback) --------------
        # Use sparse SDF storage when available: integrate forces
        # only over the body's AABB sub-block (much less memory
        # and faster than iterating the full grid).
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
