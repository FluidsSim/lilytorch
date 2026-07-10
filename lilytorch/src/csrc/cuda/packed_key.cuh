// =====================================================================
//  packed_key.cuh
//
//  Packed (sdf, body_id) 64-bit key helpers shared by the 2-D and 3-D
//  multi-body SDF-write CUDA kernels (``streaming_sdf_stag_*_multi``).
//
//  These let those kernels run with a single launch fanned across
//  ``gridDim.y == B`` instead of B sequential per-body launches.  The
//  per-cell multi-field compare-swap
//      if (s < sdf_*[g]) { sdf_*[g] = s; bU/bV/bW[g] = ...; }
//  is replaced by a single ``atomicMin(uint64_t* key, packed_key)``,
//  followed by a cheap decode pass that reconstructs the SDF and
//  recomputes the linked velocity / density from the winning body
//  index alone — no need to touch ``bU/bV/bW`` during phase 1.
//
//  Encoding (IEEE-754 fp32 → sortable uint32, then pack with body id):
//      f2u_sortable(+f) flips the sign bit       → high half of u32 range
//      f2u_sortable(-f) flips all bits           → low  half of u32 range
//  giving a strict monotone ordering: f1 < f2  ⇔  f2u(f1) < f2u(f2).
//
//  ``key = (f2u(s) << 32) | (uint32_t)body_id``.  The SDF lives in the
//  high half so it dominates ordering; the body id breaks ties
//  deterministically (lower id wins).  The sentinel ``body_id = B``
//  marks "no body has touched this cell" and lets the decode pass
//  preserve any pre-existing value of ``sdf_*[g]``.
//
//  All scalar dtypes (fp16/fp32/fp64) are quantised to fp32 for the
//  key.  Final values written back to ``sdf_*[g]`` are recovered from
//  the key (cast back to scalar_t); the precision loss for fp64 union
//  SDFs is bounded by ~1 ulp in fp32 — well below the geometric error
//  of the upstream SDF sampler at typical body resolutions.
// =====================================================================

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lilytorch_kernels {

__device__ __forceinline__ uint32_t f2u_sortable(float f)
{
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ float u2f_sortable(uint32_t u)
{
    return __uint_as_float((u & 0x80000000u) ? (u ^ 0x80000000u) : ~u);
}

template <typename scalar_t>
__device__ __forceinline__ uint64_t pack_sdf_body_key(scalar_t s, int b)
{
    return ((uint64_t)f2u_sortable((float)s) << 32) | (uint32_t)b;
}

__device__ __forceinline__ uint32_t unpack_body_id(uint64_t key)
{
    return (uint32_t)(key & 0xFFFFFFFFull);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t unpack_sdf(uint64_t key)
{
    return (scalar_t)u2f_sortable((uint32_t)(key >> 32));
}

}  // namespace lilytorch_kernels
