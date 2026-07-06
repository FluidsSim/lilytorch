"""Body-update marshalling bridges — **Warp** implementation.

The two bridges below adapt the historical positional ``body_update_{2,3}d``
call convention (a flat body-table layout, inherited from the retired
hand-written CUDA/C++ ``_C.so`` extension) into :class:`WarpStreamingSDF` /
:class:`WarpStreamingSDF2D`.  They stream the immersed-body SDF + staggered body
velocities on a single-source ``@wp.kernel`` (the same source runs on CPU and
CUDA); there is no native fallback — Warp is a hard dependency.

This module used to be a full kernel-API aggregator that re-exported every Warp
op behind the native positional convention.  With Warp the sole backend those
pass-throughs added nothing, so consumers now import each op straight from its
source module (``bdim``, ``cvof``, ``advection``, ``interpolation``, ``forces``,
``lagrangian``, ``poisson``).  Only the body-update bridges — which carry real
marshalling logic — remain here.

Bridges:

* ``body_update_{2,3}d`` — streaming SDF + staggered body velocities (formerly
  "Kernel A"), flat-table layout → ``WarpStreamingSDF``; dtype-generic (f32+f64),
  optional CUDA-graph fast path.
"""

import torch

# Streaming SDF "far" sentinel — Warp body_update's atomic-min needs untouched cells
# pre-filled to +FAR.  Used by the CUDA-graph fast path, which folds the reset
# into the captured graph (Warp memset) instead of a per-step torch fill.
_FAR = 1e4

# ── body_update (2-D) → WARP (marshalling bridge) ──────────────────────────────────
# body_update runs on Warp at f32 AND f64 (dtype-generic port).

from lilytorch.src.streaming_sdf import WarpStreamingSDF2D as _WarpSDF2D
import warp as _wp


def _wdev(t):
    return "cuda:0" if t.device.type == "cuda" else "cpu"


class _BodyUpdate2DBridge:
    """Adapt the native ``body_update_2d`` positional call into
    :class:`WarpStreamingSDF2D` (flat-table layout → setup/update/run).

    The static body table (``F_flat``/``body_meta``/``gx``/``gy``…) is cached
    and re-setup only when its identity / grid / dtype / interp / blend
    toggles; per-step ``kin``/``aabb`` go through ``update_kinematics``.
    Outputs are wrapped zero-copy from the caller's torch tensors (which the
    solver step pre-fills to ``+FAR``/0, exactly as ``wp.atomic_min`` needs).
    Generic in dtype: f32 and f64 both run on Warp."""

    def __init__(self):
        self._w = None
        self._key = None
        self._graph = None     # (out-ptr key, captured graph)
        self._seen = {}

    def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                 aabb_lo, aabb_dim, gx, gy, h, max_vol,
                 sdf_cc, sdf_u, sdf_v, body_u, body_v,
                 interp_method,
                 dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
                 num_u, num_v, den_u, den_v, blend_eps,
                 use_graph=False):
        if int(dirty_Ai) * int(dirty_Aj) <= 0:
            return
        wpf = _wp.float64 if sdf_cc.dtype == torch.float64 else _wp.float32
        Ngx, Ngy = int(sdf_cc.shape[0]), int(sdf_cc.shape[1])
        B = int(body_shapes.reshape(-1).numel() // 2)
        blend_on = float(blend_eps) > 0.0
        key = (id(F_flat), Ngx, Ngy, str(wpf), int(interp_method), blend_on)
        if key != self._key:
            w = _WarpSDF2D(Ngx, Ngy, device=_wdev(sdf_cc), dtype=wpf)
            # sm['F_offsets'] is length B+1 (start offsets + total); the Warp
            # class wants the B start offsets (it never reads the end).
            w.setup(F_flat, F_offsets[:B], body_shapes, body_meta, gx, gy,
                    float(h), int(max_vol), interp_method=int(interp_method),
                    blend_eps=float(blend_eps))
            self._w, self._key = w, key
            self._graph = None
        w = self._w
        # Kinematics update stays OUTSIDE any captured graph — it copies the
        # new body pose into the persistent w._kin/_aabb the graph reads.
        if int(max_vol) > w._max_vol:
            self._graph = None
        w.update_kinematics(kin, aabb_lo, aabb_dim, max_vol=int(max_vol))

        def f(t):
            return _wp.from_torch(t.reshape(-1))

        # CUDA-graph fast path (default on CUDA): replaces the ~2× eager
        # ``wp.launch`` host floor (~70–170 µs) with one graph replay (~3 µs).
        # Requires stable output buffers (the solver override reuses persistent
        # temporaries under the same flag); a churn guard captures only on the
        # 2nd sighting of a given output-pointer signature, so transient
        # buffers stay eager.  The SDF→FAR / body→0 resets are folded INTO the
        # graph (Warp memsets) so the solver override does not pay the
        # ~35 µs/step of size-independent torch fill launches — this is what
        # makes body_update beat native at small grids too.
        if use_graph and sdf_cc.is_cuda and not blend_on:
            gkey = (sdf_cc.data_ptr(), sdf_u.data_ptr(), sdf_v.data_ptr(),
                    body_u.data_ptr(), body_v.data_ptr())
            if self._graph is not None and self._graph[0] == gkey:
                _wp.capture_launch(self._graph[1])
                return
            n = self._seen.get(gkey, 0) + 1
            self._seen[gkey] = n
            args = (f(sdf_cc), f(sdf_u), f(sdf_v), f(body_u), f(body_v))

            def _fill_launch():
                args[0].fill_(_FAR); args[1].fill_(_FAR); args[2].fill_(_FAR)
                args[3].zero_(); args[4].zero_()
                w._launch_fanned(*args)

            if n >= 2:
                _fill_launch()  # warm-up / JIT
                with _wp.ScopedCapture(device=_wdev(sdf_cc)) as cap:
                    _fill_launch()
                self._graph = (gkey, cap.graph)
                return
            _fill_launch()      # 1st sighting → eager (bridge owns the fills)
            return

        w.run_fanned_eager(f(sdf_cc), f(sdf_u), f(sdf_v),
                           f(body_u), f(body_v))


body_update_2d = _BodyUpdate2DBridge()

# ── body_update (3-D) → WARP (marshalling bridge) ──────────────────────────────────
from lilytorch.src.streaming_sdf import WarpStreamingSDF as _WarpSDF3D


class _BodyUpdate3DBridge:
    """3-D analogue of :class:`_BodyUpdate2DBridge`.  Adapts the native
    ``body_update_3d`` positional call into
    :class:`WarpStreamingSDF` (z axis: ``aabb_*``/``body_shapes`` are ``B*3``,
    ``kin`` is ``B*21``; adds ``gz``/``sdf_w``/``bW``).  Generic in dtype:
    f32 and f64 both run on Warp."""

    def __init__(self):
        self._w = None
        self._key = None
        self._graph = None
        self._seen = {}

    def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                 aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
                 sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
                 interp_method,
                 dirty_i0, dirty_j0, dirty_k0,
                 dirty_Ai, dirty_Aj, dirty_Ak,
                 num_u, num_v, num_w, den_u, den_v, den_w, blend_eps,
                 use_graph=False):
        if int(dirty_Ai) * int(dirty_Aj) * int(dirty_Ak) <= 0:
            return
        wpf = _wp.float64 if sdf_cc.dtype == torch.float64 else _wp.float32
        Ngx, Ngy, Ngz = (int(sdf_cc.shape[0]), int(sdf_cc.shape[1]),
                         int(sdf_cc.shape[2]))
        B = int(body_shapes.reshape(-1).numel() // 3)
        blend_on = float(blend_eps) > 0.0
        key = (id(F_flat), Ngx, Ngy, Ngz, str(wpf), blend_on)
        if key != self._key:
            w = _WarpSDF3D(Ngx, Ngy, Ngz, device=_wdev(sdf_cc), dtype=wpf)
            w.setup(F_flat, F_offsets[:B], body_shapes, body_meta,
                    gx, gy, gz, float(h), int(max_vol),
                    blend_eps=float(blend_eps))
            self._w, self._key = w, key
            self._graph = None
        w = self._w
        # Growth of the per-body AABB volume must invalidate the fanned
        # graph too (its launch dims are frozen at capture).
        if int(max_vol) > w._max_vol:
            self._graph = None
        w.update_kinematics(kin, aabb_lo, aabb_dim, max_vol=int(max_vol))

        def f(t):
            return _wp.from_torch(t.reshape(-1))

        # CUDA-graph fast path (default on CUDA) — see the 2-D bridge.  The
        # SDF→FAR / body→0 resets are folded into the captured graph (Warp
        # memsets), so the override pays no per-step torch fills.
        if use_graph and sdf_cc.is_cuda and not blend_on:
            gkey = (sdf_cc.data_ptr(), sdf_u.data_ptr(), sdf_v.data_ptr(),
                    sdf_w.data_ptr(), body_u.data_ptr(), body_v.data_ptr(),
                    body_w.data_ptr())
            if self._graph is not None and self._graph[0] == gkey:
                _wp.capture_launch(self._graph[1])
                return
            n = self._seen.get(gkey, 0) + 1
            self._seen[gkey] = n
            args = (f(sdf_cc), f(sdf_u), f(sdf_v), f(sdf_w),
                    f(body_u), f(body_v), f(body_w))

            def _fill_launch():
                args[0].fill_(_FAR); args[1].fill_(_FAR); args[2].fill_(_FAR)
                args[3].fill_(_FAR)
                args[4].zero_(); args[5].zero_(); args[6].zero_()
                w._launch_fanned(*args)

            if n >= 2:
                _fill_launch()  # warm-up / JIT
                with _wp.ScopedCapture(device=_wdev(sdf_cc)) as cap:
                    _fill_launch()
                self._graph = (gkey, cap.graph)
                return
            _fill_launch()      # 1st sighting → eager (bridge owns the fills)
            return

        w.run_fanned_eager(f(sdf_cc), f(sdf_u), f(sdf_v), f(sdf_w),
                           f(body_u), f(body_v), f(body_w))


body_update_3d = _BodyUpdate3DBridge()
