// =====================================================================
//  poisson_scratch.h — pointer‑stable persistent scratch buffers for
//  the native Poisson drivers (mgcg, rmgcg, multigrid, vcycle).
//
//  Motivation: the mgcg/rmgcg drivers allocate ~8 full‑grid tensors per
//  solve call.  That churn forces PyTorch's caching allocator to
//  occasionally grow the pool (cudaMalloc) mid‑step, which invalidates a
//  concurrent CUDA‑graph capture.  By caching the scratch tensors after
//  the first (eager) allocation, warmed‑up solves become zero‑allocation
//  and the pool never grows inside a capture.
//
//  Conventions (must be observed by every call site):
//   1. Callers MUST zero a cached buffer (.zero_()) if they need zeros —
//      scratch() returns UN-INITIALISED memory.
//   2. Two simultaneously‑live buffers must use DIFFERENT tags even when
//      they have the same shape (otherwise they alias and clobber each
//      other).
//   3. Once cached, a slot's data_ptr is permanent.  Never resize_ or
//      reassign a cached tensor except on a genuine grid‑size change.
//   4. The cache is process‑global and NOT thread‑safe.  Current usage is
//      single‑threaded per process.
//   5. Memory is held for process life (acceptable for reusable scratch).
// =====================================================================
#pragma once

#include <ATen/ATen.h>
#include <string>
#include <unordered_map>

namespace lilytorch_kernels {
namespace poisson_scratch {

namespace detail {

// Build a compact string key:  "tag|Nx|Ny|[Nz]|dtype_id|device_idx"
inline std::string make_key(const std::string& tag,
                            at::IntArrayRef shape,
                            const at::TensorOptions& opts) {
    std::string key = tag;
    for (auto s : shape) {
        key += "|";
        key += std::to_string(s);
    }
    key += "|d" + std::to_string(static_cast<int>(opts.dtype().toScalarType()));
    key += "|dev" + std::to_string(opts.device().index());
    return key;
}

}  // namespace detail

// ---------------------------------------------------------------------
// Returns a reference to a pointer‑stable tensor of the requested shape /
// dtype / device.  Allocated on first access, reused forever after.
// The buffer is UNINITIALISED — caller must .zero_() if it needs zeros.
// ---------------------------------------------------------------------
inline at::Tensor& scratch(const std::string& tag,
                           at::IntArrayRef shape,
                           const at::TensorOptions& opts) {
    static std::unordered_map<std::string, at::Tensor> pool;
    std::string key = detail::make_key(tag, shape, opts);

    auto it = pool.find(key);
    if (it == pool.end() || !it->second.defined()) {
        pool[key] = at::empty(shape, opts);
    }
    return pool[key];
}

// ---------------------------------------------------------------------
// Returns a zero‑filled persistent tensor.  Allocated once (at::zeros),
// NEVER written by anyone after — it is a pure read‑only zero source.
// Used to replace the wasteful per‑solve ``f_zero`` allocation.
// ---------------------------------------------------------------------
inline at::Tensor& zero_buffer(at::IntArrayRef shape,
                               const at::TensorOptions& opts) {
    static std::unordered_map<std::string, at::Tensor> pool;
    std::string key = detail::make_key("_zero_", shape, opts);

    auto it = pool.find(key);
    if (it == pool.end() || !it->second.defined()) {
        pool[key] = at::zeros(shape, opts);
    }
    return pool[key];
}

}  // namespace poisson_scratch
}  // namespace lilytorch_kernels
