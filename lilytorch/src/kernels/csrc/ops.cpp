// =====================================================================
//  lilytorch_kernels: native ops for BDIM-IB CFD.
//
//  This file provides:
//    - the empty `_C` Python module (so `import lilytorch.src.kernels._C`
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
    m.def(
        "streaming_sdf_min_3d("
        "Tensor F, Tensor bx, Tensor by, Tensor bz,"
        " float bx0, float by0, float bz0,"
        " float bx_last, float by_last, float bz_last,"
        " float inv_dx, float inv_dy, float inv_dz, float inv_vol,"
        " float[] R_T, float[] body_pos, float[] com_pos,"
        " float[] lin_vel, float[] ang_vel,"
        " Tensor gx, Tensor gy, Tensor gz, float h_grid,"
        " int i0, int i1, int j0, int j1, int k0, int k1,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v, Tensor(d!) sdf_w,"
        " Tensor(e!) body_u, Tensor(f!) body_v, Tensor(g!) body_w,"
        " Tensor(h!) sparse_cc,"
        " int interp_method=0"
        ") -> ()");
    m.def(
        "streaming_sdf_min_3d_multi("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor bx_flat, Tensor bx_offsets,"
        " Tensor by_flat, Tensor by_offsets,"
        " Tensor bz_flat, Tensor bz_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim, Tensor cell_offsets,"
        " Tensor gx, Tensor gy, Tensor gz, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v, Tensor(d!) sdf_w,"
        " Tensor(e!) body_u, Tensor(f!) body_v, Tensor(g!) body_w,"
        " Tensor(h!) sparse_cc_flat,"
        " int interp_method=0"
        ") -> ()");
    // Forces kernel: reads the per-body cc-SDF cached in `sparse_cc_flat`
    // (populated by streaming_sdf_min_3d_multi), so it no longer needs the
    // body SDF grid / axis tensors / body_meta. `kin` is still passed for
    // the per-body COM (rows 12..14 of each 21-stride block).
    m.def(
        "bdim_forces_3d_multi("
        "Tensor sparse_cc_flat, Tensor cell_offsets,"
        " Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, Tensor gz,"
        " int u_i0, int u_j0, int u_k0, int Sj, int Sk,"
        " Tensor xs, Tensor ys, Tensor zs,"
        " Tensor px, Tensor py, Tensor pz,"
        " float eps_body, float eps_solver, float h3,"
        " int max_vol_per_body,"
        " int delta_order,"
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "apply_bcs_3d("
        "Tensor(a!) u, Tensor(b!) v, Tensor(c!) w,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " int max_plane_dim"
        ") -> ()");

    // ---------------- 2-D analogues ----------------
    // Single-body 2-D streaming SDF / face-velocity update.
    // R_T is column-major 2x2 (4 floats); body_pos/com_pos/lin_vel are
    // 2-element; angular velocity is the scalar omega (out-of-plane).
    // bx/by axis tables and bx_last/by_last/inv_vol are accepted for
    // signature symmetry with the 3-D op but unused by the kernel,
    // which infers corner weights analytically on uniform body grids.
    m.def(
        "streaming_sdf_min_2d("
        "Tensor F, Tensor bx, Tensor by,"
        " float bx0, float by0,"
        " float bx_last, float by_last,"
        " float inv_dx, float inv_dy, float inv_vol,"
        " float[] R_T, float[] body_pos, float[] com_pos,"
        " float[] lin_vel, float omega,"
        " Tensor gx, Tensor gy, float h_grid,"
        " int i0, int i1, int j0, int j1,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v,"
        " Tensor(d!) body_u, Tensor(e!) body_v,"
        " Tensor(f!) sparse_cc,"
        " int interp_method=0"
        ") -> ()");
    // Multi-body 2-D streaming SDF / face-velocity update.
    //   body_shapes : int64 [B,2]   (Mx, My)
    //   body_meta   : float [B,7]   (bx0, by0, bxL, byL, inv_dx, inv_dy, inv_vol)
    //   kin         : float [B,11]  (R_T[0..3], bp_xy, cm_xy, lv_xy, omega)
    //   aabb_lo     : int64 [B,2]   (i0, j0)
    //   aabb_dim    : int64 [B,2]   (Ai, Aj)
    //   cell_offsets: int64 [B+1]
    m.def(
        "streaming_sdf_min_2d_multi("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor bx_flat, Tensor bx_offsets,"
        " Tensor by_flat, Tensor by_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim, Tensor cell_offsets,"
        " Tensor gx, Tensor gy, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v,"
        " Tensor(d!) body_u, Tensor(e!) body_v,"
        " Tensor(f!) sparse_cc_flat,"
        " int interp_method=0"
        ") -> ()");

    // 2-D forces kernel — analogue of bdim_forces_3d_multi.  Reads the
    // per-body cc-SDF cached in ``sparse_cc_flat`` (populated by
    // streaming_sdf_min_2d_multi).  ``out`` is float64 and has 8
    // channels per body:
    //   [fv_x, fv_y, t_v, fp_x, fp_y, t_p, 0, 0]
    // where t_* are the scalar out-of-plane torques and the trailing
    // two slots are reserved (always written as 0).
    m.def(
        "bdim_forces_2d_multi("
        "Tensor sparse_cc_flat, Tensor cell_offsets,"
        " Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy,"
        " int u_i0, int u_j0, int Sj,"
        " Tensor xs, Tensor ys,"
        " Tensor px, Tensor py,"
        " float eps_body, float eps_solver, float h2,"
        " int max_vol_per_body,"
        " int delta_order,"
        " Tensor(a!) out"
        ") -> ()");

    // 2-D fused boundary-condition writes — analogue of apply_bcs_3d.
    //   shapes  : int64 [2,2]    -> (Nx, Ny) per component (u, v)
    //   neu_desc: int32 [N_neu, 3] -> (comp, axis, side)
    //   dir_desc: int32 [N_dir, 3] -> (comp, axis, offset)
    //   dir_val : float[N_dir]
    m.def(
        "apply_bcs_2d("
        "Tensor(a!) u, Tensor(b!) v,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " int max_line_dim"
        ") -> ()");

    // ---- Scattered-point interpolation --------------------------------
    // Reusable bilinear/biquadratic (2-D) and trilinear/triquadratic (3-D)
    // samplers backed by the same device functions as streaming_sdf.
    // interp_method: 0 = linear, 1 = quadratic.
    // F must be contiguous and row-major: F[ix, iy] or F[ix, iy, iz].
    // G is pre-allocated by the caller, same dtype and device as F.
    m.def(
        "interpolate_2d("
        "Tensor F, Tensor xq, Tensor yq,"
        " float bx0, float by0,"
        " float inv_dx, float inv_dy,"
        " int Mx, int My,"
        " int interp_method,"
        " Tensor(a!) G"
        ") -> ()");
    m.def(
        "interpolate_3d("
        "Tensor F, Tensor xq, Tensor yq, Tensor zq,"
        " float bx0, float by0, float bz0,"
        " float inv_dx, float inv_dy, float inv_dz,"
        " int Mx, int My, int Mz,"
        " int interp_method,"
        " Tensor(a!) G"
        ") -> ()");
}

}  // namespace lilytorch_kernels
