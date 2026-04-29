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
        " Tensor(h!) sparse_cc"
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
        " Tensor(h!) sparse_cc_flat"
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
        " Tensor(a!) out"
        ") -> ()");

    m.def(
        "apply_bcs_3d("
        "Tensor(a!) u, Tensor(b!) v, Tensor(c!) w,"
        " Tensor shapes, Tensor neu_desc, Tensor dir_desc, Tensor dir_val,"
        " int max_plane_dim"
        ") -> ()");
}

}  // namespace lilytorch_kernels
