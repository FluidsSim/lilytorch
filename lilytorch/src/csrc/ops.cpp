// =====================================================================
//  lilytorch_kernels: native ops for BDIM-IB CFD.
//
//  This file provides:
//    - the empty `_C` Python module (so `import lilytorch.src._C`
//      forces the .so to load and runs the static TORCH_LIBRARY blocks
//      below);
//    - the operator schemas under the `lilytorch_kernels` namespace.
//
//  CPU implementations live in `streaming_sdf_cpu.cpp` (OpenMP),
//  CUDA implementations in `cuda/streaming_sdf.cu`. Each backend
//  registers its own dispatch table via TORCH_LIBRARY_IMPL.
// =====================================================================

#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/util/ArrayRef.h>

extern "C" {
PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace lilytorch_kernels {

// ----------------------- schemas -----------------------
TORCH_LIBRARY(lilytorch_kernels, m) {
    // Phase-I 3-D streaming SDF / face velocity update.  Kernel B
    // computes rho_eff from mu0 in registers, so no per-cell
    // winning-density tensor or rho_bodies input is needed.
    //
    // key_cc_t / key_u_t / key_v_t / key_w_t are caller-allocated int64
    // scratch buffers of size >= Ngx*Ngy*Ngz (one per stagger) used to
    // pack/unpack per-cell winning-body keys.  Moving them out of the
    // kernel avoids a 4× empty allocation per call.
    m.def(
        "streaming_sdf_stag_3d_multi("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, Tensor gz, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v, Tensor(d!) sdf_w,"
        " Tensor(e!) body_u, Tensor(f!) body_v, Tensor(g!) body_w,"
        " Tensor(h!) key_cc_t, Tensor(i!) key_u_t, Tensor(j!) key_v_t, Tensor(k!) key_w_t,"
        " int interp_method,"
        " int dirty_i0, int dirty_j0, int dirty_k0,"
        " int dirty_Ai, int dirty_Aj, int dirty_Ak,"
        " Tensor(l!) num_u, Tensor(m!) num_v, Tensor(n!) num_w,"
        " Tensor(o!) den_u, Tensor(p!) den_v, Tensor(q!) den_w,"
        " float blend_eps"
        ") -> ()");

    // Phase-I fused BDIM2 + variable-density Poisson coefficient kernel.
    // Reads advdiff outputs + Kernel-A face SDFs / body velocities, writes
    // the persistent velocity fields u0/v0/w0 and Poisson coefficients
    // ch/cv/cw inside the dirty AABB.  mu0, mu1 and unit normals are
    // computed in CUDA thread registers and never stored globally.
    m.def(
        "bdim_coeff_3d("
        "Tensor u_prime, Tensor v_prime, Tensor w_prime,"
        " Tensor sdf_u, Tensor sdf_v, Tensor sdf_w,"
        " Tensor body_u, Tensor body_v, Tensor body_w,"
        " Tensor(a!) u0, Tensor(b!) v0, Tensor(c!) w0,"
        " Tensor(d!) ch, Tensor(e!) cv, Tensor(f!) cw,"
        " float eps, float rho_f, float dt,"
        " float h_grid,"
        " int dirty_i0, int dirty_j0, int dirty_k0,"
        " int dirty_Ai, int dirty_Aj, int dirty_Ak,"
        " int mu0_projection"
        ") -> ()");

    // BDIM-σ variant of bdim_coeff_3d (Lauber et al. 2022).
    // For each cell the Poisson coefficient is evaluated with mu0 of a
    // shifted SDF phi - sigma_shifts[body_id] (lookup via key_u/v/w), so
    // thin bodies (r < eps) reach mu0_poisson = 0 inside the body and the
    // pressure BC is correctly enforced.  The velocity BDIM fields
    // (phi_out) are unchanged — only the Poisson coefficient line uses
    // the shifted mu0.
    m.def(
        "bdim_coeff_sigma_3d("
        "Tensor u_prime, Tensor v_prime, Tensor w_prime,"
        " Tensor sdf_u, Tensor sdf_v, Tensor sdf_w,"
        " Tensor body_u, Tensor body_v, Tensor body_w,"
        " Tensor(a!) u0, Tensor(b!) v0, Tensor(c!) w0,"
        " Tensor(d!) ch, Tensor(e!) cv, Tensor(f!) cw,"
        " Tensor key_u, Tensor key_v, Tensor key_w,"
        " Tensor sigma_shifts,"
        " float eps, float rho_f, float dt,"
        " float h_grid,"
        " int dirty_i0, int dirty_j0, int dirty_k0,"
        " int dirty_Ai, int dirty_Aj, int dirty_Ak,"
        " int mu0_projection"
        ") -> ()");

    m.def(
        "streaming_sdf_forces_post_3d("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, Tensor gz, float h_grid,"
        " int max_vol_per_body,"
        " Tensor sdf_cc,"
        " int interp_method,"
        " Tensor u, Tensor v, Tensor w, Tensor p,"
        " Tensor nu_rho_field,"
        " float eps_body, float eps_solver, float h3,"
        " int delta_order, int force_submethod, float ph_tau,"
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "apply_bcs_3d("
        "Tensor(a!) u, Tensor(b!) v, Tensor(c!) w,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " Tensor ref_desc, Tensor ref_val,"
        " int max_dim0, int max_dim1"
        ") -> ()");

    // Phase-I 2-D streaming SDF / face velocity update.
    //
    // key_cc_t / key_u_t / key_v_t are caller-allocated int64 scratch buffers
    // of size >= Ngx*Ngy used to pack/unpack per-cell winning-body keys.
    // Mirrors the 3-D version; required by BDIM-σ so the keys can be read
    // by Kernel B (bdim_coeff_sigma_2d) after Kernel A populates them.
    m.def(
        "streaming_sdf_stag_2d_multi("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v,"
        " Tensor(d!) body_u, Tensor(e!) body_v,"
        " Tensor(f!) key_cc_t, Tensor(g!) key_u_t, Tensor(h!) key_v_t,"
        " int interp_method,"
        " int dirty_i0, int dirty_j0, int dirty_Ai, int dirty_Aj,"
        " Tensor(l!) num_u, Tensor(m!) num_v,"
        " Tensor(o!) den_u, Tensor(p!) den_v,"
        " float blend_eps"
        ") -> ()");

    // Phase-I fused BDIM2 + variable-density Poisson coefficient kernel (2-D).
    m.def(
        "bdim_coeff_2d("
        "Tensor u_prime, Tensor v_prime,"
        " Tensor sdf_u, Tensor sdf_v,"
        " Tensor body_u, Tensor body_v,"
        " Tensor(a!) u0, Tensor(b!) v0,"
        " Tensor(c!) ch, Tensor(d!) cv,"
        " float eps, float rho_f, float dt,"
        " float h_grid,"
        " int dirty_i0, int dirty_j0,"
        " int dirty_Ai, int dirty_Aj,"
        " int mu0_projection"
        ") -> ()");

    // BDIM-σ variant of bdim_coeff_2d (Lauber et al. 2022).  See the
    // 3-D variant above for documentation.
    m.def(
        "bdim_coeff_sigma_2d("
        "Tensor u_prime, Tensor v_prime,"
        " Tensor sdf_u, Tensor sdf_v,"
        " Tensor body_u, Tensor body_v,"
        " Tensor(a!) u0, Tensor(b!) v0,"
        " Tensor(c!) ch, Tensor(d!) cv,"
        " Tensor key_u, Tensor key_v,"
        " Tensor sigma_shifts,"
        " float eps, float rho_f, float dt,"
        " float h_grid,"
        " int dirty_i0, int dirty_j0,"
        " int dirty_Ai, int dirty_Aj,"
        " int mu0_projection"
        ") -> ()");

    // ---- Lagrangian (surface-integral) force kernels ------------------
    //
    // ``lagrangian_forces_{2d,3d}``: fused per-body surface integration
    // of ``∮ (ν·ρ·ε·n - p n) dS`` and the associated torque about
    // ``com_pos``.  Replaces the per-body Python loop in
    // ``forces.forces_lagrangian_{2d,3d}`` with a single kernel launch
    // (rigid-body only — the markers / triangles must be in the
    // world frame and are precomputed by the caller).
    //
    // 2-D inputs:
    //   * ``eps_xx``, ``eps_xy``, ``eps_yy`` — CC strain-rate components,
    //     each shape ``(Mx, My)`` (built once per step by
    //     ``forces._viscous_stress_tensor``).
    //   * ``p`` — CC pressure ``(Mx, My)``.
    //   * ``nu_rho_field`` — either a 0-D / 1-element tensor (constant
    //     viscosity ⇒ scalar ν·ρ) or a full CC field of shape ``(Mx, My)``.
    //   * ``cnt_flat`` — concatenated contours of all bodies, shape
    //     ``(2, M_total)`` (rows are x and y).
    //   * ``cnt_offsets`` — int64 prefix offsets, shape ``(B+1,)``.
    //     Body ``b`` owns markers ``[cnt_offsets[b], cnt_offsets[b+1])``;
    //     within each body the marker list is CCW-oriented and *closed*
    //     (segment ``i → i+1`` wraps at the end of the body's slice).
    //   * ``com_pos`` — per-body COM, shape ``(B, 2)``.
    //   * Grid metadata: ``bx0, by0, inv_dx, inv_dy, Mx, My``.
    //   * ``interp_method`` — 0=bilinear, 1=biquadratic.
    //   * ``out`` — preallocated ``(B, 6)`` float64; column layout is
    //     ``[fv_x, fv_y, t_v, fp_x, fp_y, t_p]``.  Writes are
    //     accumulated atomically (CUDA) or reduced per-body (CPU).
    m.def(
        "lagrangian_forces_2d("
        "Tensor eps_xx, Tensor eps_xy, Tensor eps_yy,"
        " Tensor p, Tensor nu_rho_field,"
        " Tensor cnt_flat, Tensor cnt_offsets,"
        " Tensor com_pos,"
        " float bx0, float by0,"
        " float inv_dx, float inv_dy,"
        " int Mx, int My,"
        " int interp_method,"
        " float sample_offset,"
        " Tensor(a!) out"
        ") -> ()");

    // 3-D inputs mirror the 2-D op with the symmetric 6 strain-rate
    // components + per-triangle (centroid, normal, area) packed across
    // all bodies.  Output is ``(B, 12)`` float64 with column layout
    //   ``[fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
    //      fp_x, fp_y, fp_z, tp_x, tp_y, tp_z]``.
    m.def(
        "lagrangian_forces_3d("
        "Tensor eps_xx, Tensor eps_yy, Tensor eps_zz,"
        " Tensor eps_xy, Tensor eps_xz, Tensor eps_yz,"
        " Tensor p, Tensor nu_rho_field,"
        " Tensor tri_centroid, Tensor tri_normal, Tensor tri_area,"
        " Tensor tri_offsets, Tensor com_pos,"
        " float bx0, float by0, float bz0,"
        " float inv_dx, float inv_dy, float inv_dz,"
        " int Mx, int My, int Mz,"
        " int interp_method,"
        " float sample_offset,"
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "streaming_sdf_forces_post_2d("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, float h_grid,"
        " int max_vol_per_body,"
        " Tensor sdf_cc,"
        " int interp_method,"
        " Tensor u_prev, Tensor v_prev, Tensor p_prev,"
        " Tensor nu_rho_field,"
        " float eps_body, float eps_solver, float h2,"
        " int delta_order, int force_submethod, float ph_tau,"
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "apply_bcs_2d("
        "Tensor(a!) u, Tensor(b!) v,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " Tensor ref_desc, Tensor ref_val,"
        " int max_line_dim"
        ") -> ()");

    // ---- Fused advection flux kernel (T2a) ----------------------------
    // advect_flux_add: accumulates dt_dh*(F_left - F_right) into rhs for
    // one (velocity component, spatial direction) pair.  Replaces the
    // _flux → F[:-1]-F[1:] → rhs.add_() chain in _solve_convective.
    //
    // fv  : face velocities  (Nfd-1, Nt1[, Nt2]) — non-contiguous slice OK
    // p   : field values     (Nfd,   Nt1[, Nt2]) — non-contiguous slice OK
    // rhs : accumulator      (Nfd-2, Nt1[, Nt2]) — C-contiguous, mutated
    // dt_dh      : dt / h_d (scalar)
    // C          : max Courant number for ABDQUICKEST; ignored by other schemes
    // scheme_id  : 0=QUICK 1=ABDQUICKEST 2=vanLeer 3=CDS 4=CUBISTA
    // face_dim   : which tensor dimension the faces lie along (0, 1, or 2)
    m.def(
        "advect_flux_add("
        "Tensor fv, Tensor p, Tensor(a!) rhs,"
        " float dt_dh, float C,"
        " int scheme_id, int face_dim"
        ") -> ()");

    // ---- Weymouth & Yue conservative-VOF sweep (MP10 / T2d) -----------
    // cvof_sweep: one bounded conservative-VOF directional sweep, replacing
    // the Python ``_cvof_sweep`` shift/limit/flux/divergence-correction
    // chain in TwoPhase._cvof_sweep with a single launch.  Mirrors
    //   out[i] = a[i] + cfl*(F(i) - F(i+1) + a[i]*(u[i+1]-u[i]))
    // with the W&Y van-Leer-limited, Courant-corrected donor face value.
    //
    // a   : volume fraction (full grid, ghost-padded, Neumann)
    // u_d : MAC velocity component along face_dim (full grid, strided OK)
    // cfl : dt / h (scalar)
    // face_dim : sweep direction (0, 1, or 2)
    // out : preallocated = a.clone(); kernel overwrites interior-along-d
    m.def(
        "cvof_sweep("
        "Tensor a, Tensor u_d, float cfl, int face_dim,"
        " Tensor(a!) out"
        ") -> ()");

    // ---- Multigrid RBGS sweeper kernels --------------------------------
    // rbgs_sweep_2d: tiled 2-D RBGS smoother (all nsmoothing sweeps fused).
    //   p is updated in-place (includes Neumann BC application).
    //   f, cp0, cm0, cp1, cm1 are the interior-cell RHS and face coefficients.
    m.def(
        "rbgs_sweep_2d("
        "Tensor(a!) p, Tensor f,"
        " Tensor cp0, Tensor cm0, Tensor cp1, Tensor cm1,"
        " float jcap_tol, int nsmoothing"
        ") -> ()");

    // rbgs_sweep_3d: thread-per-cell 3-D RBGS smoother.
    //   p is updated in-place; cp2/cm2 are the z-face coefficients.
    m.def(
        "rbgs_sweep_3d("
        "Tensor(a!) p, Tensor f,"
        " Tensor cp0, Tensor cm0, Tensor cp1, Tensor cm1,"
        " Tensor cp2, Tensor cm2,"
        " float jcap_tol, int nsmoothing"
        ") -> ()");

    // ---- Multigrid Jacobi sweeper kernels ------------------------------
    // jacobi_sweep_2d: tiled 2-D weighted Jacobi smoother.
    //   p is updated in-place.  w is the relaxation weight (w=1: plain Jacobi).
    m.def(
        "jacobi_sweep_2d("
        "Tensor(a!) p, Tensor f,"
        " Tensor cp0, Tensor cm0, Tensor cp1, Tensor cm1,"
        " float jcap_tol, float w, int nsmoothing"
        ") -> ()");

    // jacobi_sweep_3d: double-buffer 3-D weighted Jacobi smoother.
    //   p is updated in-place via an internal temp buffer.
    m.def(
        "jacobi_sweep_3d("
        "Tensor(a!) p, Tensor f,"
        " Tensor cp0, Tensor cm0, Tensor cp1, Tensor cm1,"
        " Tensor cp2, Tensor cm2,"
        " float jcap_tol, float w, int nsmoothing"
        ") -> ()");

    // ---- Multigrid residual kernels ------------------------------------
    // mg_residual_2d / mg_residual_3d: compute
    //   r = (f - A(p)) * (|J| >= jcap_tol)    where A(p) = sum - J*p
    // with J and the active mask in registers only (no global allocations).
    // r must be a pre-allocated interior-shape tensor (no ghost cells);
    // p is ghost-padded.  Used to replace the J/active/sum/addcmul_/neg_
    // chain inside the multigrid V-cycle so neither J (~64 MB) nor active
    // (~16 MB bool) is ever materialised as a tensor.
    m.def(
        "mg_residual_2d("
        "Tensor p, Tensor f,"
        " Tensor cp0, Tensor cm0, Tensor cp1, Tensor cm1,"
        " float jcap_tol,"
        " Tensor(a!) r"
        ") -> ()");

    m.def(
        "mg_residual_3d("
        "Tensor p, Tensor f,"
        " Tensor cp0, Tensor cm0, Tensor cp1, Tensor cm1,"
        " Tensor cp2, Tensor cm2,"
        " float jcap_tol,"
        " Tensor(a!) r"
        ") -> ()");

    // ---- Scattered-point interpolation --------------------------------
    // Reusable bilinear/biquadratic (2-D) and trilinear/triquadratic (3-D)
    // samplers backed by the same device functions as streaming_sdf.
    // interp_method: 0 = linear, 1 = quadratic.
    // F must be contiguous and row-major: F[ix, iy] or F[ix, iy, iz].
    // G is pre-allocated by the caller, same dtype and device as F.
    m.def(
        "interp_2d("
        "Tensor F, Tensor xq, Tensor yq,"
        " float bx0, float by0,"
        " float inv_dx, float inv_dy,"
        " int Mx, int My,"
        " int interp_method,"
        " Tensor(a!) G"
        ") -> ()");
    m.def(
        "interp_3d("
        "Tensor F, Tensor xq, Tensor yq, Tensor zq,"
        " float bx0, float by0, float bz0,"
        " float inv_dx, float inv_dy, float inv_dz,"
        " int Mx, int My, int Mz,"
        " int interp_method,"
        " Tensor(a!) G"
        ") -> ()");

    // ---- Multigrid grid-transfer kernels --------------------------------
    // Residual restriction: sum of 4 (2-D) or 8 (3-D) fine children into rc.
    m.def("restrict_residual_2d(Tensor r, Tensor(a!) rc) -> ()");
    m.def("restrict_residual_3d(Tensor r, Tensor(a!) rc) -> ()");
    // Face restriction (WaterLily): stride-2 in face_dim, sum-of-pairs in
    // transverse, single 0.5 factor. face_dim in {0,1} (2-D) or {0,1,2} (3-D).
    m.def("restrict_face_2d(Tensor src, Tensor(a!) dst, int face_dim) -> ()");
    m.def("restrict_face_3d(Tensor src, Tensor(a!) dst, int face_dim) -> ()");
    // Prolongation + in-place correction: bilinear/trilinear align_corners=False
    // interpolation of ec[interior] added into p[interior] (both ghost-padded).
    m.def("prolongate_add_2d(Tensor ec, Tensor(a!) p) -> ()");
    m.def("prolongate_add_3d(Tensor ec, Tensor(a!) p) -> ()");

    // ---- Monolithic multigrid Poisson drivers --------------------------
    // Run the full multi-V-cycle solve in C++: scale f by h², N V-cycles
    // (with L∞ early-exit at ``tol``), final float64 mean subtraction.
    // ``p`` is ghost-padded and mutated in place; returns the final residual.
    // ``smoother_id``: 0 = RBGS, 1 = weighted Jacobi (uses ``w``).
    m.def(
        "poisson_solve_multigrid_2d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_vcycles, float tol, int smoother_id"
        ") -> Tensor");
    m.def(
        "poisson_solve_multigrid_3d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv, Tensor cw,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_vcycles, float tol, int smoother_id"
        ") -> Tensor");

    // MGCG (multigrid-preconditioned conjugate gradient).
    m.def(
        "poisson_solve_mgcg_2d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_cycles, int precond_vcycles,"
        " float tol, int smoother_id"
        ") -> Tensor");
    m.def(
        "poisson_solve_mgcg_3d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv, Tensor cw,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_cycles, int precond_vcycles,"
        " float tol, int smoother_id"
        ") -> Tensor");

    // RMGCG (recycled / deflated MGCG).  U,W carry the B-orthonormal recycle
    // basis (kdef may be 0 → plain MGCG); returns (residual, harvested last-k
    // search directions D, iterations performed).
    m.def(
        "poisson_solve_rmgcg_2d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv,"
        " Tensor U, Tensor W, int harvest_k,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_cycles, int precond_vcycles,"
        " float tol, int smoother_id"
        ") -> (Tensor, Tensor, int)");
    m.def(
        "poisson_solve_rmgcg_3d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv, Tensor cw,"
        " Tensor U, Tensor W, int harvest_k,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_cycles, int precond_vcycles,"
        " float tol, int smoother_id"
        ") -> (Tensor, Tensor, int)");
}

}  // namespace lilytorch_kernels
