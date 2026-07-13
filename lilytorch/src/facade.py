"""Body-update marshalling bridges — **native** implementation.

The two bridges below adapt the historical positional ``body_update_{2,3}d``
call convention (a flat body-table layout, inherited from the retired
hand-written CUDA/C++ ``_C.so`` extension) onto the per-body streaming-SDF
kernels.  They stream the immersed-body SDF + staggered body velocities using
the native CUDA/C++ kernels (with CPU twins for CPU runs).

Regime dispatch (per_body_key_buffers, item 2.4 — the union-AABB packed-key
``_multi`` path is DELETED):

* pairwise-disjoint body AABBs → ``streaming_sdf_stag_{2,3}d_direct``
  (single writer per cell, no keys, no atomics);
* overlapping AABBs → ``streaming_sdf_stag_{2,3}d_resolve`` (per-body private
  buffers + one-thread-per-cell per-stagger min resolve; full fp64, no
  atomics; softmin velocity blend in-kernel when ``blend_eps > 0``).

Per-bridge CUDA-graph capture is deferred (native eager is fast enough); the
per-step region is graph-captured at the whole-step level by
``graph_capture.NativeWholeStepGraphRunner``.
"""

import torch

from lilytorch.src import native


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


def body_update_2d(
    F_flat, F_offsets, body_shapes, body_meta, kin,
    aabb_lo, aabb_dim, gx, gy, h, max_vol,
    sdf_cc, sdf_u, sdf_v, body_u, body_v,
    interp_method,
    dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
    blend_eps=0.0,
    use_graph=False,  # ignored — native eager only for now
):
    """Native 2-D streaming SDF body update (eager path).

    The caller (BDIMhandler) pre-fills ``sdf_*`` to +FAR and ``body_*`` to 0
    before every call — the kernels only write body-covered cells."""
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
    # cell PER-STAGGER min resolve = deterministic true min, full fp64, no
    # atomics/race.  (The direct kernel must NOT run here — its non-atomic
    # gridDim.y=B write races overlapping bodies.)  blend_eps > 0: softmin
    # velocity blend accumulated in registers inside the resolve kernel
    # (deterministic ordered sum).  Per-call priv-buffer offsets: eager path
    # only, not yet graph-captured (see the 2.x graph-key note).
    priv_offsets, pb = _regime_b_priv(
        body_shapes.size(0), max_vol, sdf_cc.dtype, sdf_cc.device, 2)
    native.streaming_sdf_stag_2d_resolve(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, body_u, body_v, int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
        priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4],
        blend_eps=float(blend_eps),
    )


def body_update_3d(
    F_flat, F_offsets, body_shapes, body_meta, kin,
    aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
    sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
    interp_method,
    dirty_i0, dirty_j0, dirty_k0,
    dirty_Ai, dirty_Aj, dirty_Ak,
    blend_eps=0.0,
    use_graph=False,  # ignored — native eager only for now
):
    """Native 3-D streaming SDF body update (eager path).  See 2-D."""
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

    # Regime B (overlapping bodies) — see the 2-D note.
    priv_offsets, pb = _regime_b_priv(
        body_shapes.size(0), max_vol, sdf_cc.dtype, sdf_cc.device, 3)
    native.streaming_sdf_stag_3d_resolve(
        F_flat, F_offsets, body_shapes, body_meta, kin,
        aabb_lo, aabb_dim, gx, gy, gz, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w, int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6],
        blend_eps=float(blend_eps),
    )


# Back-compat aliases (the bridge classes collapsed into plain functions when
# the _multi key-scratch buffers were deleted in item 2.4).
_native_body_update_2d = body_update_2d
_native_body_update_3d = body_update_3d
