// =====================================================================
//  advection_schemes.h — the five convective face-value schemes, shared
//  by the CUDA kernels (advection_flux.cu) and their CPU twins.
//
//  Convention: u = upstream (far), c = center (upwind), d = downstream.
//  The caller handles the positive/negative flow direction swap.
//
//  Single source for the five limiters, shared by the CUDA kernels and
//  their CPU twins.
//  (_scheme_quick .. _scheme_cubista) exactly — same tiny-denominator
//  guard (1e-30), same operation order.
//
//  Scheme IDs (must match _CUDA_SCHEME_IDS in advection.py):
//      0 = QUICK
//      1 = ABDQUICKEST
//      2 = vanLeer
//      3 = CDS
//      4 = CUBISTA
// =====================================================================
#pragma once

#if defined(__CUDACC__)
#define LT_SCHEME_FN __host__ __device__ __forceinline__
#else
#define LT_SCHEME_FN inline
#endif

namespace lilytorch_kernels {
namespace schemes {

template <typename T>
LT_SCHEME_FN T sabs(T x) { return x < T(0) ? -x : x; }

template <typename T>
LT_SCHEME_FN T smax(T a, T b) { return a > b ? a : b; }

template <typename T>
LT_SCHEME_FN T smin(T a, T b) { return a < b ? a : b; }

template <typename T>
LT_SCHEME_FN T median3(T a, T b, T c) {
    return smax(smin(a, b), smin(smax(a, b), c));
}

template <typename T>
LT_SCHEME_FN T scheme_quick(T u, T c, T d) {
    T inner = median3(T(10) * c - T(9) * u, c, d);
    T outer = (T(5) * c + T(2) * d - u) / T(6);
    return median3(outer, c, inner);
}

template <typename T>
LT_SCHEME_FN T scheme_abdquickest(T u, T c, T d, T C) {
    T denom = d - c;
    if (sabs(denom) < T(1e-30)) return c;
    T rf      = (c - u) / denom;
    T C2      = C * C;
    T C_upper = T(2) * (T(1) - C);
    T scale   = (T(1) - C2) / (T(3) - T(3) * C);
    T offset  = (T(2) + C2 - T(3) * C) / (T(3) - T(3) * C);
    T psi     = smin(rf * scale + offset, C_upper);
    psi       = smin(psi, rf * C_upper);
    psi       = smax(psi, T(0));
    return c + T(0.5) * denom * psi;
}

template <typename T>
LT_SCHEME_FN T scheme_van_leer(T u, T c, T d) {
    T denom = d - c;
    if (sabs(denom) < T(1e-30)) return c;
    T rf     = (c - u) / denom;
    T abs_rf = sabs(rf);
    T psi    = (rf + abs_rf) / (T(1) + abs_rf);
    return c + T(0.5) * denom * psi;
}

template <typename T>
LT_SCHEME_FN T scheme_cds(T /*u*/, T c, T d) {
    return T(0.5) * (c + d);
}

template <typename T>
LT_SCHEME_FN T scheme_cubista(T u, T c, T d) {
    T denom = d - c;
    if (sabs(denom) < T(1e-30)) return c;
    T rf  = (c - u) / denom;
    T psi = smin(T(0.75) * rf + T(0.25), T(1.5));
    psi   = smin(psi, rf * T(1.5));
    psi   = smax(psi, T(0));
    return c + T(0.5) * denom * psi;
}

// Compile-time dispatch to the appropriate scheme.
template <typename T, int sid>
LT_SCHEME_FN T apply_scheme(T u, T c, T d, T C) {
    if constexpr (sid == 0) return scheme_quick<T>(u, c, d);
    if constexpr (sid == 1) return scheme_abdquickest<T>(u, c, d, C);
    if constexpr (sid == 2) return scheme_van_leer<T>(u, c, d);
    if constexpr (sid == 3) return scheme_cds<T>(u, c, d);
    if constexpr (sid == 4) return scheme_cubista<T>(u, c, d);
    return c;
}

}  // namespace schemes
}  // namespace lilytorch_kernels

#undef LT_SCHEME_FN
