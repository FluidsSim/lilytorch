// =====================================================================
//  lilytorch_kernels: native ops for BDIM-IB CFD.
//
//  This file provides:
//    - the empty `_C` Python module (so `import lilytorch.src._C`
//      forces the .so to load and runs the static TORCH_LIBRARY blocks
//      below);
//    - the operator schemas under the `lilytorch_kernels` namespace.
//
//  CPU implementations live in `ops_3d.cpp` / `ops_2d.cpp` (at::parallel_for),
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
    // Phase-I fused BDIM2 + variable-density Poisson coefficient kernel.
    // Reads advdiff outputs + Kernel-A face SDFs / body velocities, writes
    // the persistent velocity fields u0/v0/w0 and Poisson coefficients
    // ch/cv/cw inside the dirty AABB.  mu0, mu1 and unit normals are
    // computed in CUDA thread registers and never stored globally.

    // NOTE (2.4): the BDIM-σ ops (bdim_coeff_sigma_{2,3}d) were removed with
    // the packed-key union path — their body-id input existed only as the
    // packed-key winner.  If BDIM-σ (Lauber et al. 2022) is revived, have the
    // Regime-B resolve kernel emit a per-cell winner body-id field instead.
    // NOTE (CL2/CLx): the bdim_coeff_{2,3}d ops were removed — superseded by
    // bdim_apply_{2,3}d which adds the Maertens-Weymouth body-divergence
    // correction.  The live BDIM ops live in bdim_apply.cu / bdim_apply.cpp.

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
        " float eps_body,"
        " float sample_offset_pressure, float sample_offset_friction,"
        " float h3,"
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

    // ---- Regime-B per-body private buffers + resolve kernels -------------
    // streaming_sdf_stag_2d_resolve: two-stage streaming SDF for overlapping
    // bodies.  Min-stage writes raw SDF + staggered body velocity to
    // per-body private flat buffers (priv_*); resolve-stage iterates the
    // union dirty AABB, reads each covering body's private buffer, picks
    // the minimum SDF (no atomics, single thread per global cell), and
    // writes the winner to the global output tensors.  Full fp64 precision
    // (no packed key).  priv_offsets is int64 [B+1] cumulative body_vol.
    m.def(
        "streaming_sdf_stag_2d_resolve("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v,"
        " Tensor(d!) body_u, Tensor(e!) body_v,"
        " int interp_method,"
        " int dirty_i0, int dirty_j0, int dirty_Ai, int dirty_Aj,"
        " Tensor priv_offsets,"
        " Tensor(f!) priv_sdf_cc, Tensor(g!) priv_sdf_u, Tensor(h!) priv_sdf_v,"
        " Tensor(i!) priv_body_u, Tensor(j!) priv_body_v,"
        " float blend_eps"
        ") -> ()");

    m.def(
        "streaming_sdf_stag_3d_resolve("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, Tensor gz, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v, Tensor(d!) sdf_w,"
        " Tensor(e!) body_u, Tensor(f!) body_v, Tensor(g!) body_w,"
        " int interp_method,"
        " int dirty_i0, int dirty_j0, int dirty_k0,"
        " int dirty_Ai, int dirty_Aj, int dirty_Ak,"
        " Tensor priv_offsets,"
        " Tensor(h!) priv_sdf_cc, Tensor(i!) priv_sdf_u, Tensor(j!) priv_sdf_v, Tensor(k!) priv_sdf_w,"
        " Tensor(l!) priv_body_u, Tensor(m!) priv_body_v, Tensor(n!) priv_body_w,"
        " float blend_eps"
        ") -> ()");

    // bdim_apply_3d: static full-grid BDIM2 velocity + Poisson coefficients
    // + Maertens–Weymouth body-divergence correction.  Launches over the FULL
    // grid (pose-independent → CUDA-graph-capturable); the per-step dirty
    // AABB lives in the device-resident int32 rect tensor [i0,j0,k0,Ai,Aj,Ak].
    // Threads inside the AABB compute BDIM2 (u0 = mu0*(u'-body)+body+mu1*nd)
    // and write the face-grid Poisson coefficient; outside, u0 = u_prime
    // (pass-through) and ch/cv/cw stay at their persistent dt/rho prefill.
    // mw_on != 0 additionally computes (1-mu0_cc)*div(u_body) into div_corr.
    m.def(
        "bdim_apply_3d("
        "Tensor u_prime, Tensor v_prime, Tensor w_prime,"
        " Tensor sdf_u, Tensor sdf_v, Tensor sdf_w,"
        " Tensor body_u, Tensor body_v, Tensor body_w,"
        " Tensor(a!) u0, Tensor(b!) v0, Tensor(c!) w0,"
        " Tensor(d!) ch, Tensor(e!) cv, Tensor(f!) cw,"
        " Tensor sdf_cc, Tensor(g!) div_corr,"
        " Tensor rect,"
        " float eps, float rho_f, float dt, float h_grid,"
        " float eps_mw, float inv_dx, float inv_dy, float inv_dz,"
        " int mu0_projection, int mw_on"
        ") -> ()");

    // bdim_apply_2d: 2-D analogue of bdim_apply_3d.
    m.def(
        "bdim_apply_2d("
        "Tensor u_prime, Tensor v_prime,"
        " Tensor sdf_u, Tensor sdf_v,"
        " Tensor body_u, Tensor body_v,"
        " Tensor(a!) u0, Tensor(b!) v0,"
        " Tensor(c!) ch, Tensor(d!) cv,"
        " Tensor sdf_cc, Tensor(e!) div_corr,"
        " Tensor rect,"
        " float eps, float rho_f, float dt, float h_grid,"
        " float eps_mw, float inv_dx, float inv_dy,"
        " int mu0_projection, int mw_on"
        ") -> ()");

    // ---- Fused semi-Lagrangian advection kernels (item 8.D) ------------
    // sl_advect_2d: fused RK2 midpoint semi-Lagrangian back-trace,
    // one launch for both staggered components, writing persistent out_*.
    m.def(
        "sl_advect_2d("
        "Tensor u, Tensor v,"
        " Tensor gxu, Tensor gyu, Tensor gxv, Tensor gyv,"
        " Tensor(a!) out_u, Tensor(b!) out_v,"
        " float u_bx0, float u_by0, float u_idx, float u_idy,"
        " float v_bx0, float v_by0, float v_idx, float v_idy,"
        " float dt"
        ") -> ()");

    // sl_advect_3d: 3-D analogue of sl_advect_2d.
    m.def(
        "sl_advect_3d("
        "Tensor u, Tensor v, Tensor w,"
        " Tensor gxu, Tensor gyu, Tensor gzu,"
        " Tensor gxv, Tensor gyv, Tensor gzv,"
        " Tensor gxw, Tensor gyw, Tensor gzw,"
        " Tensor(a!) out_u, Tensor(b!) out_v, Tensor(c!) out_w,"
        " float u_bx0, float u_by0, float u_bz0, float u_idx, float u_idy, float u_idz,"
        " float v_bx0, float v_by0, float v_bz0, float v_idx, float v_idy, float v_idz,"
        " float w_bx0, float w_by0, float w_bz0, float w_idx, float w_idy, float w_idz,"
        " float dt"
        ") -> ()");

    // diffuse_add: in-place explicit-diffusion Laplacian accumulate.
    // The caller snapshots target → copy_buf before calling; this kernel
    // reads the stencil from copy_buf and accumulates into target.
    // nu_eff can be a 1-element tensor (constant ν·dt) or a full field.
    // scale_constant: pre-computed ν·dt (used only when nu_eff.numel()≤1).
    m.def(
        "diffuse_add("
        "Tensor(a!) target, Tensor copy_buf, Tensor nu_eff,"
        " float dt, int ndim,"
        " float dh0, float dh1, float dh2,"
        " float scale_constant"
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
    //   * ``sample_offset_pressure`` / ``sample_offset_friction`` — distance
    //     (metres) along the outward normal at which ``p`` and the strain
    //     tensor are respectively sampled.  Independent knobs: the two
    //     channels are contaminated differently by the BDIM band, and pinning
    //     them to a common value is what makes an eulerian/lagrangian
    //     comparison like-for-like.  Both default to 0 at the Python layer
    //     (sample exactly on the marker).  Passing the same value for both
    //     reproduces the legacy single ``sample_offset`` knob.
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
        " float sample_offset_pressure, float sample_offset_friction,"
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
        " float sample_offset_pressure, float sample_offset_friction,"
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
        " float eps_body,"
        " float sample_offset_pressure, float sample_offset_friction,"
        " float h2,"
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

    // ---- Fused per-cell flux accumulate (13c) --------------------------
    // advect_flux_accumulate: the fused flux-add + interior-accumulate
    // pair.  One launch per velocity component: computes face velocities
    // from the original staggered fields on the fly and accumulates
    // dst[cell] += Σ_d dt_dh_d*(F_L - F_R) over the interior of the
    // FULL-GRID output (which already holds vel + diffusion increment).
    //
    // phi_src : full-grid copy of vel[comp_i] (flux stencil source)
    // dst     : full-grid output buffer, mutated in the interior
    // u, v, w : original velocity fields (w ignored in 2-D — pass u)
    // comp_i  : which velocity component is being solved for
    // dt_dh*  : dt / h_d per direction (dt_dh2 ignored in 2-D)
    // C       : max Courant number for ABDQUICKEST; ignored otherwise
    // scheme_id : 0=QUICK 1=ABDQUICKEST 2=vanLeer 3=CDS 4=CUBISTA
    m.def(
        "advect_flux_accumulate("
        "Tensor phi_src, Tensor(a!) dst,"
        " Tensor u, Tensor v, Tensor w,"
        " int comp_i,"
        " float dt_dh0, float dt_dh1, float dt_dh2,"
        " float C, int scheme_id"
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

    // ---- Strain-rate magnitude ----------------------------------------
    // strain_rate_magnitude: |S̄| = sqrt(2·S_ij·S_ij) at every grid point,
    // for the Smagorinsky eddy viscosity, the Carreau / yield-damping
    // viscosity field and the flow diagnostics.  Reproduces the reference
    // ``torch.gradient(edge_order=2) + _stag_to_cc`` path (4 full-grid
    // temporaries in 2-D, 9 in 3-D) in a single launch, registers only.
    //
    // u, v, w  : MAC velocity components (full grid, same shape, strided OK).
    //            w is required in 3-D and ignored in 2-D.
    // h        : uniform grid spacing.
    // out      : preallocated, shaped like u; fully overwritten.
    m.def(
        "strain_rate_magnitude("
        "Tensor u, Tensor v, Tensor? w, float h,"
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
    //   r = (f - A(p)) * (|J| >= jcap_tol)
    // where A(p) is evaluated as sum_faces c_face*(p_neighbor - p_cell)
    // to avoid cancellation of absolute-pressure terms in float32.  J and the
    // active mask remain in registers only (no global allocations).
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

    // ---- Raw V-cycle (MGCG preconditioner primitive) --------------------
    // ``n_vcycles`` V-cycles with NO gauge fix — no ghost-ring Neumann pass
    // and no mean subtraction, unlike the whole-solve drivers below.  This is
    // the V-cycle a PCG preconditioner needs (the one the native MGCG driver
    // applies to ``z`` internally), and it is what keeps MGCG / RMGCG alive on
    // the CPU, where the whole-solve twins are stubs and the CG loop runs in
    // Python (``PoissonSolver._cg_core``).
    //
    // ``f`` is the raw smoother RHS — ALREADY h²-scaled by the caller (no
    // internal rescale, unlike the drivers below).  ``p`` is ghost-padded and
    // mutated in place; the interior residual is returned.
    m.def(
        "mg_vcycle_2d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv,"
        " float jcap_tol, float w, int nsmoothing, int n_vcycles,"
        " int smoother_id"
        ") -> Tensor");
    m.def(
        "mg_vcycle_3d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv, Tensor cw,"
        " float jcap_tol, float w, int nsmoothing, int n_vcycles,"
        " int smoother_id"
        ") -> Tensor");

    // ---- Monolithic multigrid Poisson drivers --------------------------
    // Run the full multi-V-cycle solve in C++: scale f by h², N V-cycles
    // (with L∞ early-exit at ``tol``), final float64 mean subtraction.
    // ``p`` is ghost-padded and mutated in place; returns the final residual.
    // ``smoother_id``: 0 = RBGS, 1 = weighted Jacobi (uses ``w``).
    // Returns (residual, V-cycles performed).  With ``tol < 0`` there is no
    // early exit, so the count is always ``max_vcycles`` and carries no
    // information — see the note on the graph-capture path in poisson_solve.cu.
    m.def(
        "poisson_solve_multigrid_2d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_vcycles, float tol, int smoother_id"
        ") -> (Tensor, int)");
    m.def(
        "poisson_solve_multigrid_3d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv, Tensor cw,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_vcycles, float tol, int smoother_id"
        ") -> (Tensor, int)");

    // MGCG (multigrid-preconditioned conjugate gradient).
    // Returns (residual, CG iterations performed); same ``tol < 0`` caveat.
    m.def(
        "poisson_solve_mgcg_2d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_cycles, int precond_vcycles,"
        " float tol, int smoother_id"
        ") -> (Tensor, int)");
    m.def(
        "poisson_solve_mgcg_3d("
        "Tensor(a!) p, Tensor f, Tensor ch, Tensor cv, Tensor cw,"
        " float h2, float jcap_tol, float w,"
        " int nsmoothing, int max_cycles, int precond_vcycles,"
        " float tol, int smoother_id"
        ") -> (Tensor, int)");

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
