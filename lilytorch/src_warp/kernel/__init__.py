"""Unified kernel-backend API — **Warp** implementation.

Exposes the same names as :mod:`lilytorch.src_cuda.kernel` (the contract is
documented there).  Ops whose Warp wrapper has an identical call convention to
the native op are wired to the single-source ``@wp.kernel`` ports in
:mod:`lilytorch.warp_poc`; everything else falls back to the native op so this
tree is always runnable end-to-end.

Wired to Warp (``WARP_BACKED``):

* ``advect_flux_add`` — :func:`warp_poc.warp_advection.advect_flux_add_warp`
  (identical signature ``(fv, p, rhs, dt_dh, C, scheme_id, face_dim)``;
  in-place ``rhs += dt_dh·(F_L−F_R)``).  Bit-exact vs native (all 5 schemes,
  2-D+3-D), single-source CPU==GPU.
* ``cvof_sweep`` — :func:`warp_poc.warp_cvof.cvof_sweep_warp`
  (identical signature ``(a, u_d, cfl, face_dim, out)``; writes ``out``
  interior in place).  Bit-exact vs native, 2-D+3-D.

Native fallback (Warp ports exist + are parity-clean in ``warp_poc`` but need a
marshalling / driver-assembly bridge to drop into the live solver — see §F):

* Kernel A/B ``streaming_sdf_stag_*`` / ``bdim_coeff_*`` — the Warp wrappers
  take the POC per-body scene layout, not the native ``F_flat``/``F_offsets``/
  ``body_meta``/``kin`` flat-table marshalling. (Marshalling bridge = remaining.)
* Poisson ``K`` — the native ``poisson_solve_*`` is a monolithic C++ driver;
  Warp routing means assembling the mgcg/multigrid outer loop from the Warp
  smoother + transfer ops (``WarpVCycle`` shows it converges). (Driver assembly.)
* ``apply_bcs_*`` — Warp wrapper uses a single ``max_face_dim`` vs the native
  ``(max_dim0, max_dim1)``; safe only on cubic faces. (Kept native.)
* forces — Eulerian ``streaming_sdf_forces_post_*`` is unported; the Warp
  ``lagrangian_forces_*`` IS ported but has a decomposed (eps_xx…tri_*) call
  convention, not the native fused arg list.  SU1 routing uses
  ``force_method=lagrangian`` on the native op for now. (See ``forces.py``.)
"""

import torch

# Native ops are the fallback for everything not yet wired to Warp.
from lilytorch.src.kernels import _C as _C  # noqa: F401  (registers torch.ops)
from lilytorch.src.kernels import ops as _ops

BACKEND = "warp"

_warp_backed = set()

# ── Kernel A / Kernel B (3-D) — native fallback (3-D bridge = remaining) ────
streaming_sdf_stag_3d_multi = _ops.streaming_sdf_stag_3d_multi
bdim_coeff_3d = _ops.bdim_coeff_3d
bdim_coeff_sigma_3d = _ops.bdim_coeff_sigma_3d

# ── Kernel A / Kernel B (2-D) → WARP (marshalling bridge) ───────────────────
# Kernel A + Kernel B both run on Warp at f32 AND f64 (dtype-generic ports).
# The native handles below are used ONLY if Warp is unavailable (import except),
# plus the σ Kernel B, which still needs the packed key_* the Warp streaming
# bridge does not yet emit (see TASK_remaining_warp_wiring.md — σ-key emission).
_native_streaming_sdf_2d = _ops.streaming_sdf_stag_2d_multi
_native_bdim_coeff_2d = _ops.bdim_coeff_2d
bdim_coeff_sigma_2d = _ops.bdim_coeff_sigma_2d  # σ keys unported → still native

try:
    from lilytorch.warp_poc.warp_kernels_2d import WarpStreamingSDF2D as _WarpSDF2D
    from lilytorch.warp_poc.warp_bdim_2d import bdim_coeff_2d_warp as _bdim2d_warp
    import warp as _wp

    def _wdev(t):
        return "cuda:0" if t.device.type == "cuda" else "cpu"

    class _KernelA2DBridge:
        """Adapt the native ``streaming_sdf_stag_2d_multi`` positional call into
        :class:`WarpStreamingSDF2D` (flat-table layout → setup/update/run).

        The static body table (``F_flat``/``body_meta``/``gx``/``gy``…) is cached
        and re-setup only when its identity / grid / dtype / interp / blend
        toggles; per-step ``kin``/``aabb`` go through ``update_kinematics``.
        Outputs are wrapped zero-copy from the caller's torch tensors (which the
        solver step pre-fills to ``+FAR``/0, exactly as ``wp.atomic_min`` needs).
        Generic in dtype: f32 and f64 both run on Warp.  The σ ``key_*`` outputs
        are not emitted (the σ Kernel B stays native — gated in the solver)."""

        def __init__(self):
            self._w = None
            self._key = None

        def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                     aabb_lo, aabb_dim, gx, gy, h, max_vol,
                     sdf_cc, sdf_u, sdf_v, body_u, body_v,
                     key_cc, key_u, key_v, interp_method,
                     dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
                     num_u, num_v, den_u, den_v, blend_eps):
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
            w = self._w
            w.update_kinematics(kin, aabb_lo, aabb_dim)

            def f(t):
                return _wp.from_torch(t.reshape(-1))

            w.run_fanned_eager(f(sdf_cc), f(sdf_u), f(sdf_v), f(body_u), f(body_v))

    streaming_sdf_stag_2d_multi = _KernelA2DBridge()
    _warp_backed.add("streaming_sdf_stag_2d_multi")

    # Signature-identical drop-in; the Warp port is now dtype-generic (f32+f64),
    # so this runs on Warp at either precision (no native fallback).
    bdim_coeff_2d = _bdim2d_warp
    _warp_backed.add("bdim_coeff_2d")
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    streaming_sdf_stag_2d_multi = _native_streaming_sdf_2d
    bdim_coeff_2d = _native_bdim_coeff_2d

# ── advection flux → WARP ──────────────────────────────────────────────────
try:
    from lilytorch.warp_poc.warp_advection import advect_flux_add_warp as _advect_warp

    def advect_flux_add(fv, p, rhs, dt_dh, C_courant, scheme_id, face_dim):
        """Warp ``advect_flux_add`` (in-place ``rhs += dt_dh·(F_L−F_R)``).

        Signature-identical drop-in for ``torch.ops.lilytorch_kernels.advect_flux_add``.
        """
        _advect_warp(fv, p, rhs, dt_dh, C_courant, scheme_id, face_dim)

    _warp_backed.add("advect_flux_add")
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    advect_flux_add = torch.ops.lilytorch_kernels.advect_flux_add

# ── two-phase VOF → WARP ───────────────────────────────────────────────────
try:
    from lilytorch.warp_poc.warp_cvof import cvof_sweep_warp as _cvof_warp

    def cvof_sweep(a, u_d, cfl, face_dim, out):
        """Warp ``cvof_sweep`` (writes ``out`` interior in place).

        Signature-identical drop-in for ``torch.ops.lilytorch_kernels.cvof_sweep``.
        """
        _cvof_warp(a, u_d, cfl, face_dim, out)

    _warp_backed.add("cvof_sweep")
except Exception:  # pragma: no cover
    cvof_sweep = torch.ops.lilytorch_kernels.cvof_sweep

# ── BC / interp — native (apply_bcs face-dim caveat) ──────────────────────
apply_bcs_2d = _ops.apply_bcs_2d
apply_bcs_3d = _ops.apply_bcs_3d
interp_2d = _ops.interp_2d
interp_3d = _ops.interp_3d

# ── forces — native (Eulerian unported; lagrangian routed in forces.py) ────
lagrangian_forces_2d = _ops.lagrangian_forces_2d
lagrangian_forces_3d = _ops.lagrangian_forces_3d
streaming_sdf_forces_post_2d = _ops.streaming_sdf_forces_post_2d
streaming_sdf_forces_post_3d = _ops.streaming_sdf_forces_post_3d

# ── Poisson driver handle — native (driver assembly = remaining) ───────────
K = _ops

#: Ops actually executing on Warp single-source kernels in this backend.
WARP_BACKED = frozenset(_warp_backed)
