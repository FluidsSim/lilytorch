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
//  Encoding (IEEE-754 fp64 → sortable uint64, then pack with body id):
//      f2u_sortable(+f) flips the sign bit       → high half of u64 range
//      f2u_sortable(-f) flips all bits           → low  half of u64 range
//  giving a strict monotone ordering: f1 < f2  ⇔  f2u(f1) < f2u(f2).
//
//  ``key = (f2u64(s) & ~0xFFFF) | (uint16_t)body_id``.  The SDF occupies
//  the top 48 bits so it dominates ordering; the low 16 bits carry the
//  body id, which breaks ties deterministically (lower id wins).  The
//  sentinel ``body_id = B`` marks "no body has touched this cell" and
//  lets the decode pass preserve any pre-existing value of ``sdf_*[g]``
//  (requires ``B <= 0xFFFF`` — far beyond any realistic link count).
//
//  The SDF is carried in the fp64 sortable domain and only the low 16
//  mantissa bits are dropped to make room for the body id.  For an
//  fp32-origin value those 16 bits are exactly zero, so fp32 union SDFs
//  round-trip **bit-exactly**; for fp64 the relative loss is ~2^-36
//  (~1.5e-11) across the geometrically meaningful SDF band (|s| ≲ O(1)),
//  well under the 1e-9 native-vs-Warp parity gate.  (The former key
//  quantised everything to fp32, flooring fp64 parity at ~4e-9.)
// =====================================================================

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lilytorch_kernels {

// fp64 → order-preserving uint64 (and back).
__device__ __forceinline__ uint64_t f2u_sortable(double f)
{
    uint64_t u = __double_as_longlong(f);
    return (u & 0x8000000000000000ull) ? ~u : (u ^ 0x8000000000000000ull);
}

__device__ __forceinline__ double u2f_sortable(uint64_t u)
{
    return __longlong_as_double(
        (u & 0x8000000000000000ull) ? (u ^ 0x8000000000000000ull) : ~u);
}

// Low 16 bits reserved for the body id; top 48 bits hold the sortable SDF.
static constexpr uint64_t KEY_SDF_MASK  = 0xFFFFFFFFFFFF0000ull;
static constexpr uint64_t KEY_BODY_MASK = 0x000000000000FFFFull;

template <typename scalar_t>
__device__ __forceinline__ uint64_t pack_sdf_body_key(scalar_t s, int b)
{
    return (f2u_sortable((double)s) & KEY_SDF_MASK)
           | ((uint64_t)b & KEY_BODY_MASK);
}

__device__ __forceinline__ uint32_t unpack_body_id(uint64_t key)
{
    return (uint32_t)(key & KEY_BODY_MASK);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t unpack_sdf(uint64_t key)
{
    return (scalar_t)u2f_sortable(key & KEY_SDF_MASK);
}

}  // namespace lilytorch_kernels
