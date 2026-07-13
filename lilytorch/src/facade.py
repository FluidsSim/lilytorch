"""Body-update marshalling bridges — **native** implementation.

The two bridges below adapt the historical positional ``body_update_{2,3}d``
call convention (a flat body-table layout, inherited from the retired
hand-written CUDA/C++ ``_C.so`` extension) into ``native.streaming_sdf_stag_*_multi``
calls.  They stream the immersed-body SDF + staggered body velocities using the
native CUDA/C++ kernels (with CPU twins for CPU runs).

Bridges:

* ``body_update_{2,3}d`` — streaming SDF + staggered body velocities (formerly
  "Kernel A"), flat-table layout → native streaming_sdf_stag_{2,3}d_multi;
  dtype-generic (f32+f64).  Per-bridge CUDA-graph capture is deferred (native
  eager is fast enough); the per-step region is graph-captured at the whole-step
  level by ``graph_capture.NativeWholeStepGraphRunner``.
"""

import torch

from lilytorch.src import native

# Streaming SDF "far" sentinel — atomsic-min needs untouched cells pre-filled to +FAR.
_FAR = 1e4


def _aabbs_are_disjoint(aabb_lo, aabb_dim):
    """Return True if every pair of body AABBs is disjoint (no overlap on any axis).

    AABBs are half-open ``[lo, lo+dim)``.  Two AABBs are disjoint when at least
    one axis has no overlap; they intersect only when ALL axes overlap.
    Regime A (direct-write) is safe exactly when bodies are pairwise disjoint.
    """
    if aabb_lo is None or aabb_dim is None:
        return False
    B = aabb_lo.shape[0]
    if B <= 1:
        return True
    # Vectorised pairwise overlap: boxes a,b intersect iff on EVERY axis
    # lo_a < hi_b and lo_b < hi_a.  Done on-device in one shot with a SINGLE
    # host sync at the end (the old per-element `.item()` triple loop cost
    # ~B^2*ndim device syncs per call — catastrophic in a pipelined sim).
    lo = aabb_lo.to(torch.int64)
    hi = lo + aabb_dim.to(torch.int64)
    ov = (lo[:, None, :] < hi[None, :, :]) & (lo[None, :, :] < hi[:, None, :])
    ov = ov.all(dim=2)               # [B,B] pairwise overlap
    ov.fill_diagonal_(False)
    return not bool(ov.any().item())  # one sync


# Grow-only persistent per-body private-buffer cache for the Regime-B resolve.
# Keyed on (device, dtype, ndim); reused across steps so the streaming path does
# NO per-call allocation and NO host sync (the killer was `total_vol.item()`).
_priv_cache: dict = {}


def _regime_b_priv(B, max_vol, dtype, device, ndim):
    """Sync-free private buffers + offsets for the Regime-B resolve.

    Uses a UNIFORM per-body stride of ``max_vol`` (host-known: `B` from
    `aabb_dim.shape[0]`, `max_vol` a python int), so:
      * offsets = arange(B+1)*max_vol — no cumsum, no `.item()`, no D→H sync;
      * buffers sized to `B*max_vol` (≥ Σ body_vol), grow-only + reused.
    Body `b` owns the disjoint slice `[b*max_vol, b*max_vol+body_vol[b])`, which
    is what the min/resolve kernels index via `priv_offsets[b]+local`.
    """
    need = int(B) * int(max_vol)
    n_bufs = 5 if ndim == 2 else 7
    key = (device, dtype, ndim)
    ent = _priv_cache.get(key)
    if ent is None or ent["cap"] < need:
        cap = max(need, (ent["cap"] if ent else 0))
        ent = {"cap": cap,
               "bufs": [torch.empty(cap, dtype=dtype, device=device)
                        for _ in range(n_bufs)]}
        _priv_cache[key] = ent
    offsets = torch.arange(B + 1, dtype=torch.int64, device=device) * int(max_vol)
    return offsets, ent["bufs"]


def _native_body_update_2d(
    F_flat, F_offsets, body_shapes, body_meta, kin,
    aabb_lo, aabb_dim, gx, gy, h, max_vol,
    sdf_cc, sdf_u, sdf_v, body_u, body_v,
    interp_method,
    dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
    num_u, num_v, den_u, den_v, blend_eps,
    key_cc, key_u, key_v,
    use_graph=False,  # ignored — native eager only for now
):
    """Native 2-D streaming SDF body update (eager path)."""
    if int(dirty_Ai) * int(dirty_Aj) <= 0:
        return

    # Regime A: direct-write kernel when all bodies have disjoint AABBs
    if _aabbs_are_disjoint(aabb_lo, aabb_dim):
        native.streaming_sdf_stag_2d_direct(
            F_flat, F_offsets,
            body_shapes, body_meta, kin,
            aabb_lo, aabb_dim,
            gx, gy, float(h), int(max_vol),
            sdf_cc, sdf_u, sdf_v, body_u, body_v,
            int(interp_method),
            int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
        )
        return

    # Regime B (overlapping bodies): per-body private buffers + one-thread-per-
    # cell resolve = deterministic true min, full fp64, no atomics/race.
    # Validated byte-identical vs `_multi` (fp32) / ≤1e-9 (fp64) + GPU==CPU twin.
    # (The direct kernel must NOT run here — its non-atomic gridDim.y=B write
    # races overlapping bodies.)  Per-call priv-buffer alloc: eager path only,
    # not yet graph-captured (see the 2.x graph-key note).
    # The resolve kernel does not carry the softmin velocity blend (num/den);
    # fall back to `_multi` (which does) when blending is on, so overlap+blend
    # configs don't silently lose it. TODO(2.4): add blend to the resolve.
    if float(blend_eps) > 0.0:
        native.streaming_sdf_stag_2d_multi(
            F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
            gx, gy, float(h), int(max_vol), sdf_cc, sdf_u, sdf_v, body_u, body_v,
            key_cc, key_u, key_v, int(interp_method),
            int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
            num_u, num_v, den_u, den_v, float(blend_eps))
        return
    priv_offsets, pb = _regime_b_priv(
        body_shapes.size(0), max_vol, sdf_cc.dtype, sdf_cc.device, 2)
    native.streaming_sdf_stag_2d_resolve(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, body_u, body_v, int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
        priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4],
    )


def _native_body_update_3d(
    F_flat, F_offsets, body_shapes, body_meta, kin,
    aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
    sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
    interp_method,
    dirty_i0, dirty_j0, dirty_k0,
    dirty_Ai, dirty_Aj, dirty_Ak,
    num_u, num_v, num_w, den_u, den_v, den_w, blend_eps,
    key_cc, key_u, key_v, key_w,
    use_graph=False,  # ignored — native eager only for now
):
    """Native 3-D streaming SDF body update (eager path)."""
    if int(dirty_Ai) * int(dirty_Aj) * int(dirty_Ak) <= 0:
        return

    # Regime A: direct-write kernel when all bodies have disjoint AABBs
    if _aabbs_are_disjoint(aabb_lo, aabb_dim):
        native.streaming_sdf_stag_3d_direct(
            F_flat, F_offsets,
            body_shapes, body_meta, kin,
            aabb_lo, aabb_dim,
            gx, gy, gz, float(h), int(max_vol),
            sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
            int(interp_method),
            int(dirty_i0), int(dirty_j0), int(dirty_k0),
            int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        )
        return

    # Regime B (overlapping bodies): per-body private buffers + one-thread-per-
    # cell resolve = deterministic true min, full fp64, no atomics/race.
    # Validated byte-identical vs `_multi` (fp32) / ≤1e-9 (fp64) + GPU==CPU twin.
    # Per-call priv-buffer alloc: eager path only, not yet graph-captured.
    # Blend not carried by the resolve kernel — fall back to `_multi` (see 2-D).
    if float(blend_eps) > 0.0:
        native.streaming_sdf_stag_3d_multi(
            F_flat, F_offsets, body_shapes, body_meta, kin, aabb_lo, aabb_dim,
            gx, gy, gz, float(h), int(max_vol),
            sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
            key_cc, key_u, key_v, key_w, int(interp_method),
            int(dirty_i0), int(dirty_j0), int(dirty_k0),
            int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
            num_u, num_v, num_w, den_u, den_v, den_w, float(blend_eps))
        return
    priv_offsets, pb = _regime_b_priv(
        body_shapes.size(0), max_vol, sdf_cc.dtype, sdf_cc.device, 3)
    native.streaming_sdf_stag_3d_resolve(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, gz, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w, int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6],
    )

# ── body_update (2-D) → NATIVE (marshalling bridge) ─────────────────────────
# body_update runs on native CUDA kernels + CPU twins at f32 AND f64.

class _BodyUpdate2DBridge:
    """Native 2-D streaming SDF body update bridge.

    Allocates persistent per-grid int64 key scratch buffers on first call
    (keyed on grid dimensions).  The SDF→FAR / body→0 resets are performed
    by the caller (BDIMhandler) before this bridge is invoked; this matches
    the contract the Warp bridge had (the fills were folded into the Warp
    captured graph but the caller still pre-filled for the CPU path)."""

    def __init__(self):
        self._key_cache = {}  # (Ngx, Ngy, device_id, dtype_str) → (key_cc, key_u, key_v)

    def _get_keys(self, Ngx, Ngy, device, dtype):
        cache_key = (Ngx, Ngy, device.index if device.type == 'cuda' else -1,
                     str(dtype))
        if cache_key not in self._key_cache:
            N = Ngx * Ngy
            self._key_cache[cache_key] = (
                torch.empty(N, dtype=torch.int64, device=device),
                torch.empty(N, dtype=torch.int64, device=device),
                torch.empty(N, dtype=torch.int64, device=device),
            )
        return self._key_cache[cache_key]

    def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                 aabb_lo, aabb_dim, gx, gy, h, max_vol,
                 sdf_cc, sdf_u, sdf_v, body_u, body_v,
                 interp_method,
                 dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
                 num_u, num_v, den_u, den_v, blend_eps,
                 use_graph=False):
        Ngx, Ngy = int(sdf_cc.shape[0]), int(sdf_cc.shape[1])
        key_cc, key_u, key_v = self._get_keys(Ngx, Ngy, sdf_cc.device,
                                               sdf_cc.dtype)
        _native_body_update_2d(
            F_flat, F_offsets, body_shapes, body_meta, kin,
            aabb_lo, aabb_dim, gx, gy, h, max_vol,
            sdf_cc, sdf_u, sdf_v, body_u, body_v,
            interp_method,
            dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
            num_u, num_v, den_u, den_v, blend_eps,
            key_cc, key_u, key_v,
            use_graph=use_graph,
        )


body_update_2d = _BodyUpdate2DBridge()


# ── body_update (3-D) → NATIVE (marshalling bridge) ─────────────────────────

class _BodyUpdate3DBridge:
    """Native 3-D streaming SDF body update bridge.

    3-D analogue of :class:`_BodyUpdate2DBridge`."""

    def __init__(self):
        self._key_cache = {}  # (Ngx, Ngy, Ngz, device_id, dtype_str) → keys

    def _get_keys(self, Ngx, Ngy, Ngz, device, dtype):
        cache_key = (Ngx, Ngy, Ngz,
                     device.index if device.type == 'cuda' else -1,
                     str(dtype))
        if cache_key not in self._key_cache:
            N = Ngx * Ngy * Ngz
            self._key_cache[cache_key] = (
                torch.empty(N, dtype=torch.int64, device=device),
                torch.empty(N, dtype=torch.int64, device=device),
                torch.empty(N, dtype=torch.int64, device=device),
                torch.empty(N, dtype=torch.int64, device=device),
            )
        return self._key_cache[cache_key]

    def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                 aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
                 sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
                 interp_method,
                 dirty_i0, dirty_j0, dirty_k0,
                 dirty_Ai, dirty_Aj, dirty_Ak,
                 num_u, num_v, num_w, den_u, den_v, den_w, blend_eps,
                 use_graph=False):
        Ngx, Ngy, Ngz = (int(sdf_cc.shape[0]), int(sdf_cc.shape[1]),
                         int(sdf_cc.shape[2]))
        key_cc, key_u, key_v, key_w = self._get_keys(Ngx, Ngy, Ngz,
                                                      sdf_cc.device,
                                                      sdf_cc.dtype)
        _native_body_update_3d(
            F_flat, F_offsets, body_shapes, body_meta, kin,
            aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
            interp_method,
            dirty_i0, dirty_j0, dirty_k0,
            dirty_Ai, dirty_Aj, dirty_Ak,
            num_u, num_v, num_w, den_u, den_v, den_w, blend_eps,
            key_cc, key_u, key_v, key_w,
            use_graph=use_graph,
        )


body_update_3d = _BodyUpdate3DBridge()
