"""Dimension-agnostic diffusion operators for MAC staggered grids.

Split out of the former monolithic ``adv_diff.py`` (since removed) so the
advection schemes (:mod:`lilytorch.src.advection`) and the diffusion closures
can grow independently.

These are **pure Warp functions** — no class, no boundary conditions (the caller,
e.g. :class:`~lilytorch.src.advection.AdvDiffSolver`, owns the ghost layer via
``set_BCs``).  The single public entry point :func:`diffuse_add_` accumulates
the diffusion increment in-place on the target grid.

The :func:`diffuse_add_` entry point is the single public API — a fused,
copy+accumulate Warp kernel that is graph-capturable and works for 2-D and 3-D,
constant and variable viscosity.
"""

from __future__ import annotations

import torch
import warp as wp

wp.init()


# =====================================================================
#  Warp helpers  (same pattern as advection.py)
# =====================================================================

def _wp_device(t: torch.Tensor) -> str:
    return f"cuda:{t.device.index}" if t.is_cuda else "cpu"


def _wp_dtype(t: torch.Tensor):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def _flat(t: torch.Tensor):
    """Zero-copy flat Warp view (f32/f64) over t's storage."""
    assert t.dtype in (torch.float64, torch.float32), "warp diffusion: f32/f64 only"
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    return wp.array(ptr=t.data_ptr(), dtype=_wp_dtype(t),
                    shape=(int(remaining),), device=_wp_device(t))


_FLAT_VIEW_CACHE: dict = {}


def _flat_cached(t: torch.Tensor):
    """Cached :func:`_flat` — reuse the Warp view for a given buffer/layout."""
    elem = t.element_size()
    remaining = (t.untyped_storage().nbytes() - t.storage_offset() * elem) // elem
    key = (t.data_ptr(), int(remaining), t.dtype, _wp_device(t))
    view = _FLAT_VIEW_CACHE.get(key)
    if view is None:
        view = _flat(t)
        _FLAT_VIEW_CACHE[key] = view
    return view


# Dummy nu_eff array for constant-coefficient launches (is_variable=0).
# Warp eliminates the dead branch so the array is never actually read,
# but the unified kernel signature requires a valid wp.array argument.
_DUMMY_NU_EFF: dict = {}  # keyed by (device_str, wp_dtype)


def _get_dummy_nu_eff(device: str, wpf):
    """Return a tiny dummy ``wp.array`` to satisfy the ``nu_eff`` kernel
    parameter when running in constant-coefficient mode."""
    key = (device, wpf)
    arr = _DUMMY_NU_EFF.get(key)
    if arr is None:
        arr = wp.zeros(1, dtype=wpf, device=device)
        _DUMMY_NU_EFF[key] = arr
    return arr


# =====================================================================
#  Warp fused Laplacian-accumulate kernel  (unified 2-D / 3-D, f32 / f64)
#
#  2-D is handled as the degenerate 3-D case with Nz=1, Niz=1, s_z=0,
#  inv_dh2_z=0.0 so the z-stencil term vanishes and the single z-slice
#  iteration runs identically to a pure 2-D launch.
# =====================================================================

from typing import Any  # noqa: E402


@wp.kernel
def copy_full_grid_kernel(
    src: wp.array(dtype=Any),
    dst: wp.array(dtype=Any),
    N: int,
):
    """Flat-memory full-grid copy: ``dst[:N] = src[:N]``."""
    tid = wp.tid()
    if tid >= N:
        return
    dst[tid] = src[tid]


@wp.kernel
def fused_laplacian_accumulate_kernel(
    phi_src: wp.array(dtype=Any),        # read-only copy of phi (source for stencil)
    phi_dst: wp.array(dtype=Any),        # original grid, MODIFIED in place
    nu_eff: wp.array(dtype=Any),         # only read when is_variable=1
    scale: Any,                          # nu*dt (constant) or dt (variable)
    is_variable: int,                    # compile-time constant: 0 or 1
    Nx: int, Ny: int, Nz: int,
    Nix: int, Niy: int, Niz: int,
    s_x: int, s_y: int, s_z: int,
    inv_dh2_x: Any, inv_dh2_y: Any, inv_dh2_z: Any,
):
    """Fused diffusion: reads the stencil from *phi_src* and accumulates
    ``phi_dst[c] += scale * laplacian(phi_src)`` in a single pass.

    When ``is_variable=0``: constant-coefficient Laplacian (``scale = nu*dt``).
    When ``is_variable=1``: harmonic-mean variable-coefficient Laplacian
    using the ``nu_eff`` field (``scale = dt``).

    ``is_variable`` is a compile-time constant — zero branch overhead."""
    tid = wp.tid()
    total = Nix * Niy * Niz
    if tid >= total:
        return
    iz = tid % Niz
    ixy = tid // Niz
    iy = ixy % Niy
    ix = ixy // Niy
    gx = ix + 1
    gy = iy + 1
    gz = iz + 1
    c  = gx * s_x + gy * s_y + gz * s_z
    xp = (gx + 1) * s_x + gy * s_y + gz * s_z
    xm = (gx - 1) * s_x + gy * s_y + gz * s_z
    yp = gx * s_x + (gy + 1) * s_y + gz * s_z
    ym = gx * s_x + (gy - 1) * s_y + gz * s_z
    two = type(phi_src[0])(2.0)
    if is_variable:
        tiny = type(phi_src[0])(1e-30)
        nu_c = nu_eff[c]
        # x-direction
        nu_f = two * nu_c * nu_eff[xp] / (nu_c + nu_eff[xp] + tiny)
        nu_b = two * nu_c * nu_eff[xm] / (nu_c + nu_eff[xm] + tiny)
        lap = (nu_f * (phi_src[xp] - phi_src[c]) - nu_b * (phi_src[c] - phi_src[xm])) * inv_dh2_x
        # y-direction
        nu_f = two * nu_c * nu_eff[yp] / (nu_c + nu_eff[yp] + tiny)
        nu_b = two * nu_c * nu_eff[ym] / (nu_c + nu_eff[ym] + tiny)
        lap += (nu_f * (phi_src[yp] - phi_src[c]) - nu_b * (phi_src[c] - phi_src[ym])) * inv_dh2_y
        if Nz > 1:
            zp = gx * s_x + gy * s_y + (gz + 1) * s_z
            zm = gx * s_x + gy * s_y + (gz - 1) * s_z
            nu_f = two * nu_c * nu_eff[zp] / (nu_c + nu_eff[zp] + tiny)
            nu_b = two * nu_c * nu_eff[zm] / (nu_c + nu_eff[zm] + tiny)
            lap += (nu_f * (phi_src[zp] - phi_src[c]) - nu_b * (phi_src[c] - phi_src[zm])) * inv_dh2_z
    else:
        lap = (phi_src[xp] - two * phi_src[c] + phi_src[xm]) * inv_dh2_x
        lap += (phi_src[yp] - two * phi_src[c] + phi_src[ym]) * inv_dh2_y
        if Nz > 1:
            zp = gx * s_x + gy * s_y + (gz + 1) * s_z
            zm = gx * s_x + gy * s_y + (gz - 1) * s_z
            lap += (phi_src[zp] - two * phi_src[c] + phi_src[zm]) * inv_dh2_z
    phi_dst[c] = phi_dst[c] + scale * lap


@wp.kernel
def zero_interior_kernel(
    data: wp.array(dtype=Any),
    Nix: int, Niy: int, Niz: int,
):
    """``data[tid] = 0`` — pure Warp zero of a compacted interior buffer.
    Graph-safe replacement for ``torch.Tensor.zero_()``."""
    tid = wp.tid()
    total = Nix * Niy * Niz
    if tid >= total:
        return
    data[tid] = type(data[0])(0.0)


# Register float32 + float64 specialisations.
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(copy_full_grid_kernel, {"src": _A, "dst": _A})
    wp.overload(fused_laplacian_accumulate_kernel, {
        "phi_src": _A, "phi_dst": _A, "nu_eff": _A,
        "scale": _dt, "inv_dh2_x": _dt, "inv_dh2_y": _dt, "inv_dh2_z": _dt,
    })
    wp.overload(zero_interior_kernel, {"data": _A})


# ─────────────────────────────────────────────────────────────────────────────
#  Eager launch wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _strides_and_dims(phi_t):
    """Return (Nx,Ny,Nz, Nix,Niy,Niz, s_x,s_y,s_z) for a C-contiguous phi.

    2-D tensors are normalised to Nz=1, Niz=1, s_z=0.
    """
    ndim = phi_t.ndim
    assert ndim in (2, 3), f"diffusion: expected 2-D or 3-D field, got {ndim}-D"
    shape = phi_t.shape
    stride = phi_t.stride()

    Nx, Ny = shape[0], shape[1]
    Nz = shape[2] if ndim == 3 else 1
    Nix, Niy = Nx - 2, Ny - 2
    Niz = Nz - 2 if ndim == 3 else 1
    s_x = int(stride[0])
    s_y = int(stride[1])
    s_z = int(stride[2]) if ndim == 3 else 0
    return Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z


def _copy_full_grid_eager(src_t, dst_t):
    """Eager (non-graph) launch of :func:`copy_full_grid_kernel`:
    ``dst_t[:] = src_t[:]`` — flat full-grid copy."""
    N = src_t.numel()
    if N <= 0:
        return
    wp.launch(
        copy_full_grid_kernel,
        dim=N,
        inputs=[_flat_cached(src_t), _flat_cached(dst_t), N],
        device=_wp_device(src_t),
    )


def _fused_diffuse_add_eager(phi_src_t, phi_dst_t, scale, inv_dh2, *, nu_eff_t=None):
    """Eager launch of the unified fused Laplacian-accumulate kernel.

    Reads stencil from *phi_src_t*, accumulates ``scale*laplacian`` into
    *phi_dst_t*.

    When *nu_eff_t* is ``None``: constant-coefficient (``scale = nu*dt``).
    Otherwise: variable-coefficient (``scale = dt``, nu from *nu_eff_t*).
    """
    (Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z) = _strides_and_dims(phi_src_t)
    wpf = _wp_dtype(phi_src_t)
    dev = _wp_device(phi_src_t)
    _inv_z = wpf(float(inv_dh2[2])) if len(inv_dh2) > 2 else wpf(0.0)
    is_variable = 0 if nu_eff_t is None else 1
    _nu_eff = _flat_cached(nu_eff_t) if nu_eff_t is not None else _get_dummy_nu_eff(dev, wpf)
    wp.launch(
        fused_laplacian_accumulate_kernel,
        dim=Nix * Niy * Niz,
        inputs=[
            _flat_cached(phi_src_t), _flat_cached(phi_dst_t),
            _nu_eff,
            wpf(float(scale)),
            is_variable,
            Nx, Ny, Nz, Nix, Niy, Niz,
            s_x, s_y, s_z,
            wpf(float(inv_dh2[0])), wpf(float(inv_dh2[1])), _inv_z,
        ],
        device=dev,
    )


def _zero_interior_eager(data_t):
    """Eager (non-graph) launch of :func:`zero_interior_kernel`:
    ``data_t[:] = 0`` — pure Warp zero, no torch ops."""
    ndim = data_t.ndim
    Nix, Niy = data_t.shape[0], data_t.shape[1]
    Niz = data_t.shape[2] if ndim == 3 else 1
    wp.launch(
        zero_interior_kernel,
        dim=Nix * Niy * Niz,
        inputs=[_flat_cached(data_t), Nix, Niy, Niz],
        device=_wp_device(data_t),
    )


# =====================================================================
#  Public API  (single entry point: diffuse_add_)
# =====================================================================

def diffuse_add_(target, copy_buf, dt, *, dh, nu_eff=None, nu=None):
    """In-place, pure-Warp ``target[interior] += diffuse(target, dt, ...)``
    using a double-buffer: ``copy(target → copy_buf)``, then a fused stencil
    read from ``copy_buf`` with accumulate into ``target``.

    Bit-identical to ``target[inner] += diffuse(target, dt, dh=dh,
    nu_eff=nu_eff, nu=nu)`` but issues **only Warp launches**
    (copy + fused Laplacian-accumulate) — no torch ``mul_``/slice-add,
    no fresh allocation.  Graph-capturable by the whole-step runner;
    ``copy_buf`` is a persistent full-grid buffer (same shape/dtype/device
    as ``target``).

    The caller is responsible for pre-computing ``nu_eff = nu + nu_t``
    when using variable viscosity.
    """
    inv_dh2 = [1.0 / (float(h) * float(h)) for h in dh]
    is_variable = nu_eff is not None
    scale = float(dt) if is_variable else float(nu) * float(dt)

    # Step 1: snapshot target → copy_buf (implicit barrier before stencil reads).
    _copy_full_grid_eager(target, copy_buf)

    # Step 2: fused stencil-read from copy_buf, accumulate into target.
    _fused_diffuse_add_eager(
        copy_buf, target, scale, inv_dh2, nu_eff_t=nu_eff)
