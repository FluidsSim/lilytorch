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
        "streaming_sdf_min_rho_3d_multi("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, Tensor gz, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v, Tensor(d!) sdf_w,"
        " Tensor(e!) body_u, Tensor(f!) body_v, Tensor(g!) body_w,"
        " int interp_method,"
        " Tensor rho_bodies,"
        " Tensor(h!) winning_rho_cc,"
        " int dirty_i0, int dirty_j0, int dirty_k0,"
        " int dirty_Ai, int dirty_Aj, int dirty_Ak"
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
        " int delta_order,"
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "apply_bcs_3d("
        "Tensor(a!) u, Tensor(b!) v, Tensor(c!) w,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " int max_dim0, int max_dim1"
        ") -> ()");

    m.def(
        "streaming_sdf_min_rho_2d_multi("
        "Tensor F_flat, Tensor F_offsets,"
        " Tensor body_shapes, Tensor body_meta, Tensor kin,"
        " Tensor aabb_lo, Tensor aabb_dim,"
        " Tensor gx, Tensor gy, float h_grid,"
        " int max_vol_per_body,"
        " Tensor(a!) sdf_cc, Tensor(b!) sdf_u, Tensor(c!) sdf_v,"
        " Tensor(d!) body_u, Tensor(e!) body_v,"
        " int interp_method,"
        " Tensor rho_bodies,"
        " Tensor(f!) winning_rho_cc,"
        " int dirty_i0, int dirty_j0, int dirty_Ai, int dirty_Aj"
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
        " int delta_order,"
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "apply_bcs_2d("
        "Tensor(a!) u, Tensor(b!) v,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " int max_line_dim"
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
