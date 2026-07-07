"""Dimension-agnostic diffusion operators for MAC staggered grids.

Split out of the former monolithic ``adv_diff.py`` (since removed) so the
advection schemes (:mod:`lilytorch.src.advection`) and the diffusion closures
can grow independently.

These are **pure functions** — no class, no boundary conditions (the caller,
e.g. :class:`~lilytorch.src.advection.AdvDiffSolver`, owns the ghost layer via
``set_BCs``).  They return the *interior* diffusion increment, i.e. the value
to add to ``phi[inner]``.

The :func:`diffuse` entry point dispatches to a fused Warp kernel with
CUDA-graph capture on GPU and eager Warp launch on CPU.

This is the natural place for future diffusion-model expansions
(anisotropic / tensorial viscosity, additional non-Newtonian closures, etc.):
add a new operator here and a branch in :func:`diffuse`.
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


# =====================================================================
#  Warp Laplacian kernels  (unified 2-D / 3-D, f32 / f64)
#
#  2-D is handled as the degenerate 3-D case with Nz=1, Niz=1, s_z=0,
#  inv_dh2_z=0.0 so the z-stencil term vanishes and the single z-slice
#  iteration runs identically to a pure 2-D launch.
# =====================================================================

from typing import Any  # noqa: E402


@wp.kernel
def laplacian_warp_kernel(
    phi: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int,          # full grid dims (Nz=1 for 2-D)
    Nix: int, Niy: int, Niz: int,       # interior dims  (Niz=1 for 2-D)
    s_x: int, s_y: int, s_z: int,       # phi strides (s_z=0 for 2-D)
    inv_dh2_x: Any, inv_dh2_y: Any, inv_dh2_z: Any,
):
    tid = wp.tid()
    total = Nix * Niy * Niz
    if tid >= total:
        return

    # Decode flat tid → interior (ix, iy, iz); z is fastest-varying.
    iz = tid % Niz
    ixy = tid // Niz
    iy = ixy % Niy
    ix = ixy // Niy

    # Convert to full-grid indices (add 1 for ghost offset).
    gx = ix + 1
    gy = iy + 1
    gz = iz + 1

    c  = gx * s_x + gy * s_y + gz * s_z
    xp = (gx + 1) * s_x + gy * s_y + gz * s_z
    xm = (gx - 1) * s_x + gy * s_y + gz * s_z
    yp = gx * s_x + (gy + 1) * s_y + gz * s_z
    ym = gx * s_x + (gy - 1) * s_y + gz * s_z

    two = type(phi[0])(2.0)
    lap = (phi[xp] - two * phi[c] + phi[xm]) * inv_dh2_x
    lap += (phi[yp] - two * phi[c] + phi[ym]) * inv_dh2_y

    if Nz > 1:
        zp = gx * s_x + gy * s_y + (gz + 1) * s_z
        zm = gx * s_x + gy * s_y + (gz - 1) * s_z
        lap += (phi[zp] - two * phi[c] + phi[zm]) * inv_dh2_z

    out[tid] = lap


@wp.kernel
def variable_laplacian_warp_kernel(
    phi: wp.array(dtype=Any),
    nu_eff: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int,
    Nix: int, Niy: int, Niz: int,
    s_x: int, s_y: int, s_z: int,
    inv_dh2_x: Any, inv_dh2_y: Any, inv_dh2_z: Any,
):
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

    tiny = type(phi[0])(1e-30)
    two = type(phi[0])(2.0)

    nu_c = nu_eff[c]

    # x-direction
    nu_f = two * nu_c * nu_eff[xp] / (nu_c + nu_eff[xp] + tiny)
    nu_b = two * nu_c * nu_eff[xm] / (nu_c + nu_eff[xm] + tiny)
    lap = (nu_f * (phi[xp] - phi[c]) - nu_b * (phi[c] - phi[xm])) * inv_dh2_x

    # y-direction
    nu_f = two * nu_c * nu_eff[yp] / (nu_c + nu_eff[yp] + tiny)
    nu_b = two * nu_c * nu_eff[ym] / (nu_c + nu_eff[ym] + tiny)
    lap += (nu_f * (phi[yp] - phi[c]) - nu_b * (phi[c] - phi[ym])) * inv_dh2_y

    if Nz > 1:
        zp = gx * s_x + gy * s_y + (gz + 1) * s_z
        zm = gx * s_x + gy * s_y + (gz - 1) * s_z
        nu_f = two * nu_c * nu_eff[zp] / (nu_c + nu_eff[zp] + tiny)
        nu_b = two * nu_c * nu_eff[zm] / (nu_c + nu_eff[zm] + tiny)
        lap += (nu_f * (phi[zp] - phi[c]) - nu_b * (phi[c] - phi[zm])) * inv_dh2_z

    out[tid] = lap


# Register float32 + float64 specialisations for the constant-coefficient kernel.
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(laplacian_warp_kernel, {
        "phi": _A, "out": _A,
        "inv_dh2_x": _dt, "inv_dh2_y": _dt, "inv_dh2_z": _dt,
    })

# Register float32 + float64 specialisations for the variable-coefficient kernel.
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(variable_laplacian_warp_kernel, {
        "phi": _A, "nu_eff": _A, "out": _A,
        "inv_dh2_x": _dt, "inv_dh2_y": _dt, "inv_dh2_z": _dt,
    })


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


def _laplacian_warp_eager(phi_t, out_t, inv_dh2):
    """Eager (non-graph) launch of the constant-coefficient Warp Laplacian.

    Writes the raw Laplacian (no nu*dt scaling) into *out_t*.
    """
    (Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z) = _strides_and_dims(phi_t)
    wpf = _wp_dtype(phi_t)
    dev = _wp_device(phi_t)
    _inv_z = wpf(float(inv_dh2[2])) if len(inv_dh2) > 2 else wpf(0.0)
    wp.launch(
        laplacian_warp_kernel,
        dim=Nix * Niy * Niz,
        inputs=[
            _flat_cached(phi_t), _flat_cached(out_t),
            Nx, Ny, Nz, Nix, Niy, Niz,
            s_x, s_y, s_z,
            wpf(float(inv_dh2[0])), wpf(float(inv_dh2[1])), _inv_z,
        ],
        device=dev,
    )


def _variable_laplacian_warp_eager(phi_t, nu_eff_t, out_t, dh):
    """Eager (non-graph) launch of the variable-coefficient Warp Laplacian.

    Writes the raw Laplacian (no dt scaling) into *out_t*.
    """
    (Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z) = _strides_and_dims(phi_t)
    wpf = _wp_dtype(phi_t)
    dev = _wp_device(phi_t)
    inv_dh2 = [1.0 / (float(h) * float(h)) for h in dh]
    _inv_z = wpf(float(inv_dh2[2])) if len(inv_dh2) > 2 else wpf(0.0)
    wp.launch(
        variable_laplacian_warp_kernel,
        dim=Nix * Niy * Niz,
        inputs=[
            _flat_cached(phi_t), _flat_cached(nu_eff_t), _flat_cached(out_t),
            Nx, Ny, Nz, Nix, Niy, Niz,
            s_x, s_y, s_z,
            wpf(float(inv_dh2[0])), wpf(float(inv_dh2[1])), _inv_z,
        ],
        device=dev,
    )


# =====================================================================
#  CUDA-graph-cached diffusion runner
#
#  Manages persistent output buffers and captured CUDA graphs for the
#  two diffusion variants (constant / variable viscosity).  Analogous to
#  :class:`_WarpGraphRunner` in the advection module but specialised for
#  the diffusion output-buffer lifecycle.
# =====================================================================

class _DiffusionGraphRunner:
    """CUDA-graph-cached diffusion launch.

    Captures on the second sighting of a stable (phi_ptr, shape, dtype, device)
    signature.  Owns a persistent output buffer that is reused across
    replays; callers consume the returned tensor immediately (within the
    same time step) so buffer aliasing is safe.

    The dt/nu scaling is *outside* the captured graph (it is a simple
    PyTorch ``mul_``), so dt and nu are NOT part of the capture key —
    a single graph serves all timestep/viscosity values for the same grid.
    """

    __slots__ = ("_graphs", "_seen")

    def __init__(self):
        self._graphs: dict = {}   # key → (graph, out_tensor)
        self._seen: dict = {}     # key → count

    # ------------------------------------------------------------------
    #  Key builders — pointer + shape + dtype + device, no dt/nu scaling
    # ------------------------------------------------------------------
    @staticmethod
    def _key_constant(phi_t):
        return (phi_t.data_ptr(), phi_t.shape, phi_t.dtype, str(phi_t.device))

    @staticmethod
    def _key_variable(phi_t, nu_eff_t):
        return (phi_t.data_ptr(), nu_eff_t.data_ptr(),
                phi_t.shape, phi_t.dtype, str(phi_t.device))

    # ------------------------------------------------------------------
    #  Constant-coefficient  laplacian(phi)  [nu * dt * laplacian(phi)]
    # ------------------------------------------------------------------
    def _launch_constant(self, phi_t, dt, nu, inv_dh2):
        (Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z) = _strides_and_dims(phi_t)
        n_interior = Nix * Niy * Niz
        if n_interior <= 0:
            return torch.zeros(1, device=phi_t.device, dtype=phi_t.dtype)

        # CPU path: always eager Warp (no CUDA graphs on CPU).
        if phi_t.device.type != "cuda":
            interior_shape = (Nix, Niy) if Nz == 1 else (Nix, Niy, Niz)
            out_t = torch.empty(interior_shape, dtype=phi_t.dtype, device=phi_t.device)
            _laplacian_warp_eager(phi_t, out_t, inv_dh2)
            out_t.mul_(float(nu) * float(dt))
            return out_t

        key = self._key_constant(phi_t)
        ent = self._graphs.get(key)
        if ent is not None and ent[0] is not None:
            wp.capture_launch(ent[0])
            ent[1].mul_(float(nu) * float(dt))
            return ent[1]

        n = self._seen.get(key, 0) + 1
        self._seen[key] = n

        if n < 2:
            # First sighting → allocate out buffer and run eagerly.
            interior_shape = (Nix, Niy) if Nz == 1 else (Nix, Niy, Niz)
            out_t = torch.empty(interior_shape, dtype=phi_t.dtype, device=phi_t.device)
            _laplacian_warp_eager(phi_t, out_t, inv_dh2)
            out_t.mul_(float(nu) * float(dt))
            self._graphs[key] = (None, out_t)  # placeholder (graph=None)
            return out_t

        # Second sighting → capture the graph.
        self._seen.pop(key, None)
        _, out_t = self._graphs[key]
        with wp.ScopedCapture(device=_wp_device(phi_t)) as cap:
            _laplacian_warp_eager(phi_t, out_t, inv_dh2)
        self._graphs[key] = (cap.graph, out_t)
        # Stream capture RECORDS the launch without executing it — replay the
        # fresh graph once or this step would return the previous step's
        # (already-scaled) laplacian, double-scaled by the mul_ below.
        wp.capture_launch(cap.graph)
        out_t.mul_(float(nu) * float(dt))
        return out_t

    # ------------------------------------------------------------------
    #  Variable-coefficient  laplacian(phi, nu_eff)  [dt * ...]
    # ------------------------------------------------------------------
    def _launch_variable(self, phi_t, dt, nu_eff_t, dh):
        (Nx, Ny, Nz, Nix, Niy, Niz, s_x, s_y, s_z) = _strides_and_dims(phi_t)
        n_interior = Nix * Niy * Niz
        if n_interior <= 0:
            return torch.zeros(1, device=phi_t.device, dtype=phi_t.dtype)

        # CPU path: always eager Warp (no CUDA graphs on CPU).
        if phi_t.device.type != "cuda":
            interior_shape = (Nix, Niy) if Nz == 1 else (Nix, Niy, Niz)
            out_t = torch.empty(interior_shape, dtype=phi_t.dtype, device=phi_t.device)
            _variable_laplacian_warp_eager(phi_t, nu_eff_t, out_t, dh)
            out_t.mul_(float(dt))
            return out_t

        key = self._key_variable(phi_t, nu_eff_t)
        ent = self._graphs.get(key)
        if ent is not None and ent[0] is not None:
            wp.capture_launch(ent[0])
            ent[1].mul_(float(dt))
            return ent[1]

        n = self._seen.get(key, 0) + 1
        self._seen[key] = n

        if n < 2:
            # First sighting → allocate out buffer and run eagerly.
            interior_shape = (Nix, Niy) if Nz == 1 else (Nix, Niy, Niz)
            out_t = torch.empty(interior_shape, dtype=phi_t.dtype, device=phi_t.device)
            _variable_laplacian_warp_eager(phi_t, nu_eff_t, out_t, dh)
            out_t.mul_(float(dt))
            self._graphs[key] = (None, out_t)
            return out_t

        # Second sighting → capture the graph.
        self._seen.pop(key, None)
        _, out_t = self._graphs[key]
        with wp.ScopedCapture(device=_wp_device(phi_t)) as cap:
            _variable_laplacian_warp_eager(phi_t, nu_eff_t, out_t, dh)
        self._graphs[key] = (cap.graph, out_t)
        # Same as _launch_constant: capture records without executing — replay
        # once so this step's rhs is fresh, not the stale double-scaled buffer.
        wp.capture_launch(cap.graph)
        out_t.mul_(float(dt))
        return out_t


# Module-level singleton graph runner.
_diff_graph_runner: _DiffusionGraphRunner | None = None


def _get_diff_graph_runner() -> _DiffusionGraphRunner:
    global _diff_graph_runner
    if _diff_graph_runner is None:
        _diff_graph_runner = _DiffusionGraphRunner()
    return _diff_graph_runner


# =====================================================================
#  Public API
# =====================================================================

def diffuse(phi, dt, *, nu=None, nu_t=None, inv_dh2=None, dh=None):
    """Explicit forward-Euler diffusion increment over interior cells.

    Returns a **fresh** tensor (safe for in-place ``add_``) equal to:

    * ``nu * dt * laplacian(phi)``                    when ``nu_t is None``
      (constant viscosity), or
    * ``dt * variable_laplacian(phi, nu + nu_t)``     otherwise
      (Smagorinsky / variable-coefficient).

    On CUDA the computation uses a fused Warp kernel with CUDA-graph
    capture (one launch per call); on CPU the same Warp kernel runs eagerly.

    Parameters mirror the fields cached on the solver: ``inv_dh2`` is
    required for the constant path, ``dh`` for the variable path.
    """
    if nu_t is None:
        runner = _get_diff_graph_runner()
        return runner._launch_constant(phi, dt, nu, inv_dh2)
    nu_eff = nu + nu_t
    runner = _get_diff_graph_runner()
    return runner._launch_variable(phi, dt, nu_eff, dh)
