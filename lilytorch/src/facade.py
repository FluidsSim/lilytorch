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
    """Native 2-D streaming SDF body update (eager path).

    ``key_cc/key_u/key_v`` are persistent int64 scratch buffers
    (size >= Ngx*Ngy), allocated once by the bridge.
    ``num_u/num_v/den_u/den_v`` are 1-element dummies when blend_eps==0
    (the native wrapper expands them to full-grid zeros)."""
    if int(dirty_Ai) * int(dirty_Aj) <= 0:
        return
    B = int(body_shapes.reshape(-1).numel() // 2)

    native.streaming_sdf_stag_2d_multi(
        F_flat, F_offsets,
        body_shapes.reshape(-1), body_meta.reshape(-1), kin.reshape(-1),
        aabb_lo.reshape(-1), aabb_dim.reshape(-1),
        gx, gy, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, body_u, body_v,
        key_cc, key_u, key_v,
        int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_Ai), int(dirty_Aj),
        num_u, num_v, den_u, den_v, float(blend_eps),
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

    native.streaming_sdf_stag_3d_multi(
        F_flat, F_offsets,
        body_shapes.reshape(-1), body_meta.reshape(-1), kin.reshape(-1),
        aabb_lo.reshape(-1), aabb_dim.reshape(-1),
        gx, gy, gz, float(h), int(max_vol),
        sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
        key_cc, key_u, key_v, key_w,
        int(interp_method),
        int(dirty_i0), int(dirty_j0), int(dirty_k0),
        int(dirty_Ai), int(dirty_Aj), int(dirty_Ak),
        num_u, num_v, num_w, den_u, den_v, den_w, float(blend_eps),
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
