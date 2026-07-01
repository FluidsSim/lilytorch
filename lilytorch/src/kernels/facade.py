"""Unified kernel-backend API — **Warp** implementation.

Exposes the same names as :mod:`lilytorch.src_cuda.kernel` (the contract is
documented there).  Ops whose Warp wrapper has an identical call convention to
the native op are wired to the single-source ``@wp.kernel`` ports in
:mod:`lilytorch.src.kernels`; everything else falls back to the native op so this
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
* forces — both readouts run on Warp: Lagrangian ``lagrangian_forces_*`` and
  Eulerian ``streaming_sdf_forces_post_*`` (dtype-generic shims here, routed by
  the ``src_warp.solver`` ``forces_lagrangian_*`` / ``forces_method2*``
  overrides).  No native custom force kernel remains. (See ``forces.py``.)
"""

import torch

# Native ops are the fallback for everything not yet wired to Warp.

BACKEND = "warp"

# Streaming SDF "far" sentinel — Warp Kernel A's atomic-min needs untouched cells
# pre-filled to +FAR.  Used by the CUDA-graph fast path, which folds the reset
# into the captured graph (Warp memset) instead of a per-step torch fill.
_FAR = 1e4

_warp_backed = set()

# ── Kernel A / Kernel B (3-D) → WARP (marshalling bridge) ───────────────────
# Mirrors the 2-D bridge: Kernel A streams via WarpStreamingSDF, Kernel B is a
# signature-identical drop-in.  Both dtype-generic (f32+f64).  The σ Kernel B
# also runs on Warp (Item 5): Kernel A's bridge emits the winning body-id keys.

# ── Kernel A / Kernel B (2-D) → WARP (marshalling bridge) ───────────────────
# Kernel A + Kernel B both run on Warp at f32 AND f64 (dtype-generic ports).
# The native handles below are used ONLY if Warp is unavailable (import except).
# The σ Kernel B also runs on Warp now (Item 5): the streaming bridge emits the
# body-id key_* arrays the σ pass reads (see the bridges' emit_keys path).

try:
    from lilytorch.src.kernels.streaming_sdf_2d import WarpStreamingSDF2D as _WarpSDF2D
    from lilytorch.src.kernels.bdim_2d import bdim_coeff_2d_warp as _bdim2d_warp
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
        Generic in dtype: f32 and f64 both run on Warp.  The σ ``key_*`` arrays
        (winning body-id) are emitted on the ``emit_keys`` path (Item 5)."""

        def __init__(self):
            self._w = None
            self._key = None
            self._graph = None     # (out-ptr key, captured graph)
            self._seen = {}

        def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                     aabb_lo, aabb_dim, gx, gy, h, max_vol,
                     sdf_cc, sdf_u, sdf_v, body_u, body_v,
                     key_cc, key_u, key_v, interp_method,
                     dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
                     num_u, num_v, den_u, den_v, blend_eps,
                     emit_keys=False, use_graph=False):
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
            w.update_kinematics(kin, aabb_lo, aabb_dim)

            def f(t):
                return _wp.from_torch(t.reshape(-1))

            # BDIM-σ path: emit the winning body-id into key_u/key_v (low 32
            # bits) so the Warp σ Kernel B can read it (it masks key & 0xffffffff).
            # Sentinel = B (= n_sigma) on untouched cells → no σ shift, mirroring
            # the native ``B_sentinel``.
            if emit_keys:
                key_u.fill_(B)
                key_v.fill_(B)
                w.run_fanned_eager(f(sdf_cc), f(sdf_u), f(sdf_v),
                                   f(body_u), f(body_v),
                                   key_u=f(key_u), key_v=f(key_v), emit_keys=1)
                return

            # CUDA-graph fast path (opt-in, non-σ): replaces the ~2× eager
            # ``wp.launch`` host floor (~70–170 µs) with one graph replay (~3 µs).
            # Requires stable output buffers (the solver override reuses persistent
            # temporaries under the same flag); a churn guard captures only on the
            # 2nd sighting of a given output-pointer signature, so transient
            # buffers stay eager.  The SDF→FAR / body→0 resets are folded INTO the
            # graph (Warp memsets) so the solver override does not pay the
            # ~35 µs/step of size-independent torch fill launches — this is what
            # makes Kernel A beat native at small grids too.
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

    streaming_sdf_stag_2d_multi = _KernelA2DBridge()
    _warp_backed.add("streaming_sdf_stag_2d_multi")

    # Signature-identical drop-in; the Warp port is now dtype-generic (f32+f64),
    # so this runs on Warp at either precision (no native fallback).
    bdim_coeff_2d = _bdim2d_warp
    _warp_backed.add("bdim_coeff_2d")

    # ── Kernel A (3-D) bridge ────────────────────────────────────────────────
    from lilytorch.src.kernels.streaming_sdf import WarpStreamingSDF as _WarpSDF3D
    from lilytorch.src.kernels.bdim import bdim_coeff_3d_warp as _bdim3d_warp

    class _KernelA3DBridge:
        """3-D analogue of :class:`_KernelA2DBridge`.  Adapts the native
        ``streaming_sdf_stag_3d_multi`` positional call into
        :class:`WarpStreamingSDF` (z axis: ``aabb_*``/``body_shapes`` are ``B*3``,
        ``kin`` is ``B*21``; adds ``gz``/``sdf_w``/``bW``).  The σ ``key_*``
        arrays (winning body-id, dirty-local) are emitted on the ``emit_keys``
        path (Item 5).  Generic in dtype: f32 and f64 both run on Warp."""

        def __init__(self):
            self._w = None
            self._key = None
            self._graph = None
            self._seen = {}

        def __call__(self, F_flat, F_offsets, body_shapes, body_meta, kin,
                     aabb_lo, aabb_dim, gx, gy, gz, h, max_vol,
                     sdf_cc, sdf_u, sdf_v, sdf_w, body_u, body_v, body_w,
                     key_cc, key_u, key_v, key_w, interp_method,
                     dirty_i0, dirty_j0, dirty_k0,
                     dirty_Ai, dirty_Aj, dirty_Ak,
                     num_u, num_v, num_w, den_u, den_v, den_w, blend_eps,
                     emit_keys=False, use_graph=False):
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
            w.update_kinematics(kin, aabb_lo, aabb_dim)  # outside the graph

            def f(t):
                return _wp.from_torch(t.reshape(-1))

            # BDIM-σ: emit the winning body-id into key_{u,v,w} (low 32 bits).
            # Sentinel = B (= n_sigma) on untouched cells → no σ shift.
            if emit_keys:
                key_u.fill_(B); key_v.fill_(B); key_w.fill_(B)
                w.run_fanned_eager(f(sdf_cc), f(sdf_u), f(sdf_v), f(sdf_w),
                                   f(body_u), f(body_v), f(body_w),
                                   key_u=f(key_u), key_v=f(key_v),
                                   key_w=f(key_w), emit_keys=1,
                                   dirty=(int(dirty_i0), int(dirty_j0),
                                          int(dirty_k0), int(dirty_Aj),
                                          int(dirty_Ak)))
                return

            # CUDA-graph fast path (opt-in, non-σ) — see the 2-D bridge.  The
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

    streaming_sdf_stag_3d_multi = _KernelA3DBridge()
    _warp_backed.add("streaming_sdf_stag_3d_multi")

    # Signature-identical drop-in (dtype-generic f32+f64).
    bdim_coeff_3d = _bdim3d_warp
    _warp_backed.add("bdim_coeff_3d")

    # σ Kernel B (thin bodies): the Warp Kernel A now emits the body-id keys the
    # σ pass reads (Item 5), so the σ variants run on Warp too — native-positional
    # shims around the same dtype-generic ``bdim_coeff_{2,3}d_warp`` (keys +
    # sigma_shifts as keywords).  The solver step calls ``bdim_coeff_{2,3}d``
    # directly with the σ keywords; these named shims keep the facade contract
    # (parity with ``src_cuda.kernel``) Warp-backed instead of native.
    def bdim_coeff_sigma_2d(
            u_prime, v_prime, sdf_u, sdf_v, body_u, body_v, u0, v0, ch, cv,
            key_u, key_v, sigma_shifts, eps, rho_f, dt, h_grid,
            dirty_i0, dirty_j0, dirty_Ai, dirty_Aj, mu0_projection=1):
        return _bdim2d_warp(
            u_prime, v_prime, sdf_u, sdf_v, body_u, body_v, u0, v0, ch, cv,
            eps, rho_f, dt, h_grid, dirty_i0, dirty_j0, dirty_Ai, dirty_Aj,
            mu0_projection, key_u=key_u, key_v=key_v, sigma_shifts=sigma_shifts)

    def bdim_coeff_sigma_3d(
            u_prime, v_prime, w_prime, sdf_u, sdf_v, sdf_w,
            body_u, body_v, body_w, u0, v0, w0, ch, cv, cw,
            key_u, key_v, key_w, sigma_shifts, eps, rho_f, dt, h_grid,
            dirty_i0, dirty_j0, dirty_k0, dirty_Ai, dirty_Aj, dirty_Ak,
            mu0_projection=1):
        return _bdim3d_warp(
            u_prime, v_prime, w_prime, sdf_u, sdf_v, sdf_w,
            body_u, body_v, body_w, u0, v0, w0, ch, cv, cw,
            eps, rho_f, dt, h_grid,
            dirty_i0, dirty_j0, dirty_k0, dirty_Ai, dirty_Aj, dirty_Ak,
            mu0_projection, key_u=key_u, key_v=key_v, key_w=key_w,
            sigma_shifts=sigma_shifts)

    _warp_backed.add("bdim_coeff_sigma_2d")
    _warp_backed.add("bdim_coeff_sigma_3d")
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise

# ── advection flux → WARP ──────────────────────────────────────────────────
try:
    from lilytorch.src.kernels.advection import advect_flux_add_warp as _advect_warp

    def advect_flux_add(fv, p, rhs, dt_dh, C_courant, scheme_id, face_dim):
        """Warp ``advect_flux_add`` (in-place ``rhs += dt_dh·(F_L−F_R)``).

        Signature-identical drop-in for ``torch.ops.lilytorch_kernels.advect_flux_add``.
        """
        _advect_warp(fv, p, rhs, dt_dh, C_courant, scheme_id, face_dim)

    _warp_backed.add("advect_flux_add")
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise

# ── two-phase VOF → WARP ───────────────────────────────────────────────────
try:
    from lilytorch.src.kernels.cvof import cvof_sweep_warp as _cvof_warp

    def cvof_sweep(a, u_d, cfl, face_dim, out):
        """Warp ``cvof_sweep`` (writes ``out`` interior in place).

        Signature-identical drop-in for ``torch.ops.lilytorch_kernels.cvof_sweep``.
        """
        _cvof_warp(a, u_d, cfl, face_dim, out)

    _warp_backed.add("cvof_sweep")
except Exception:  # pragma: no cover
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise

# ── BC / interp → WARP ─────────────────────────────────────────────────────
# ``apply_bcs_{2,3}d`` (fused Neumann/Dirichlet/reflective ghost-line writes,
# dtype-generic f32+f64) is routed by the ``src_warp.advection.AdvDiffSolver``
# ``set_BCs`` override (the native ``set_BCs`` calls ``torch.ops…`` directly, so
# the module-global swap does not apply — see TASK §Item 4).  The 3-D wrapper now
# takes both face dims ``(max_dim0, max_dim1)``.  ``interp_{2,3}d`` (scattered
# bilinear/trilinear gather, marker / semi-Lagrangian path) is also dtype-generic.
try:
    from lilytorch.src.kernels.misc_2d import (
        apply_bcs_2d_warp as apply_bcs_2d,
        interp_2d_warp as interp_2d,
        ApplyBcs2DGraphRunner,
    )
    from lilytorch.src.kernels.misc_3d import (
        apply_bcs_3d_warp as apply_bcs_3d,
        interp_3d_warp as interp_3d,
        ApplyBcs3DGraphRunner,
    )
    _warp_backed |= {"apply_bcs_2d", "apply_bcs_3d", "interp_2d", "interp_3d"}
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise

# ── forces — Lagrangian → WARP; Eulerian still native ──────────────────────
# The Warp Lagrangian wrappers take the SAME positional arg list as the native
# ``ops.lagrangian_forces_{2,3}d`` (decomposed eps_xx…/tri_*…) plus the
# ``method``/``sample_offset``/``out=`` keywords, so they are drop-in shims for
# ``_lagrangian_forces_{2,3}d_kernel`` at the ``forces_lagrangian_*`` call site
# (routed there by the ``src_warp.solver`` method override).  Dtype-generic:
# per-element math runs in the field dtype, ``out`` is the float64 accumulator.
# The Eulerian readout (n·δ band integral + deltaH ∂H pass) is also on Warp:
# ``warp_forces.py`` ports ``streaming_sdf_forces_post_{2,3}d`` with the identical
# native positional signature (incl. ``force_submethod``/``ph_tau``), routed by
# the ``src_warp.solver`` ``forces_method2{,_3d}`` override.
try:
    from lilytorch.src.kernels.forces import (
        streaming_sdf_forces_post_2d_warp as streaming_sdf_forces_post_2d,
        streaming_sdf_forces_post_3d_warp as streaming_sdf_forces_post_3d,
    )
    _warp_backed.add("streaming_sdf_forces_post_2d")
    _warp_backed.add("streaming_sdf_forces_post_3d")
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise
try:
    from lilytorch.src.kernels.lagrangian import (
        lagrangian_forces_2d_warp as _lagr2d_warp,
        lagrangian_forces_3d_warp as _lagr3d_warp,
    )

    def lagrangian_forces_2d(
            eps_xx, eps_xy, eps_yy, p, nu_rho_field,
            cnt_flat, cnt_offsets, com_pos,
            bx0, by0, inv_dx, inv_dy, Mx, My,
            method="linear", sample_offset=0.0, out=None):
        """Warp ``lagrangian_forces_2d`` — drop-in for the native ops wrapper."""
        return _lagr2d_warp(
            eps_xx, eps_xy, eps_yy, p, nu_rho_field,
            cnt_flat, cnt_offsets, com_pos,
            bx0, by0, inv_dx, inv_dy, Mx, My,
            method=method, sample_offset=sample_offset, out=out)

    def lagrangian_forces_3d(
            eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
            p, nu_rho_field, tri_centroid, tri_normal, tri_area,
            tri_offsets, com_pos,
            bx0, by0, bz0, inv_dx, inv_dy, inv_dz, Mx, My, Mz,
            method="linear", sample_offset=0.0, out=None):
        """Warp ``lagrangian_forces_3d`` — drop-in for the native ops wrapper."""
        return _lagr3d_warp(
            eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
            p, nu_rho_field, tri_centroid, tri_normal, tri_area,
            tri_offsets, com_pos,
            bx0, by0, bz0, inv_dx, inv_dy, inv_dz, Mx, My, Mz,
            method=method, sample_offset=sample_offset, out=out)

    _warp_backed.add("lagrangian_forces_2d")
    _warp_backed.add("lagrangian_forces_3d")
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise

# ── Poisson driver — WARP (Python outer driver + Warp fine-level smoother) ──
# The native ``poisson_solve_*`` C++ driver is NOT used by this backend: the
# ``src_warp.poisson_mult.PoissonSolver`` subclass forces the Python outer driver
# and runs the fine-level smoother + residual on the single-source Warp kernels
# (``warp_poisson{,_2d}``).  ``K`` stays the native ops handle for symmetry, but
# the Warp Poisson path never dispatches through it.
try:
    from lilytorch.src.kernels import poisson_2d as _wp2d  # noqa: F401
    from lilytorch.src.kernels import poisson as _wp3d  # noqa: F401
    _warp_backed |= {
        "rbgs_sweep_2d", "rbgs_sweep_3d", "jacobi_sweep_2d", "jacobi_sweep_3d",
        "mg_residual_2d", "mg_residual_3d",
    }
except Exception:  # pragma: no cover - degrade to native if Warp unavailable
    # Warp backend is required (native CUDA/C++ kernels removed).
    raise

#: Ops actually executing on Warp single-source kernels in this backend.
WARP_BACKED = frozenset(_warp_backed)
