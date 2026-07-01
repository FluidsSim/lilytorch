"""Warp single-source 2-D Poisson smoothers (RBGS + weighted Jacobi) + residual.

2-D analogue of `warp_poisson.py`.  Faithful port of the native 2-D
variable-coefficient 5-point red-black Gauss-Seidel / weighted-Jacobi smoother
and the multigrid residual.  Homogeneous Neumann is folded into the stencil by
index clamping (ghost = nearest interior = self at the boundary), so NO separate
BC kernel launch is needed — identical math to native's explicit ghost refresh.

`p` is ghost-padded (Nx+2, Ny+2); `f` and the four face coefficients are
interior (Nx, Ny).  Same memory layout as the native op, so the same torch
tensors feed both via zero-copy `wp.from_torch`.

The thread-per-cell `rbgs_halfsweep`/`jacobi_sweep`/`residual` kernels run on
Warp `device="cpu"` (→ C++/OpenMP) *and* `"cuda:0"` from one source — this is the
CPU-capable smoother the multigrid driver uses on CPU (the tiled smoothers are
GPU-only).

**Precision (single source, both dtypes).**  Value arrays / float scalars are
Warp generics (`Any`); float literals are materialised in the bound element type
via `type(x)(literal)` and `wp.overload` pre-registers the float32 *and* float64
specialisations (Warp 1.14 needs them up front).  float32 codegen is unchanged
from the original concrete kernels (existing parity stays bit-identical); float64
is what an f64 solver uses.
"""
from __future__ import annotations

from typing import Any, Optional

import warp as wp
import torch

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Kernels (2-D arrays; launched over interior (Nx, Ny))
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _stencil_sum_2d(
    p: wp.array(dtype=Any), b: int, si: int,
    i: int, j: int, Nx: int, Ny: int,
    cp0: Any, cm0: Any, cp1: Any, cm1: Any,
):
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + 1]
    if j > 0:      pjm = p[b - 1]
    return cp0 * pip + cm0 * pim + cp1 * pjp + cm1 * pjm


@wp.kernel
def rbgs_halfsweep_2d(
    p:   wp.array(dtype=Any),   # flat padded (Nx+2)(Ny+2)
    f:   wp.array(dtype=Any),   # flat interior Nx·Ny
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    Nx: int, Ny: int,
    jcap_tol: Any,
    color: int,
):
    i, j = wp.tid()
    if ((i + j) & 1) != color:
        return
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    Jinv = type(jcap_tol)(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = type(jcap_tol)(1.0) / J
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    p[b] = (-f[c] + s) * Jinv


@wp.kernel
def rbgs_halfsweep_2d_compact(
    p:   wp.array(dtype=Any),
    f:   wp.array(dtype=Any),
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    Nx: int, Ny: int,
    jcap_tol: Any,
    color: int,
):
    """Color-COMPACTED flat-1-D half-sweep (even Ny): launch only the Nx·Ny/2
    cells of one color, so NO thread early-returns on the parity check.  Maps a
    flat tid in [0, Nx·Ny/2) → the t-th cell of `color`.  Bit-identical math to
    the full launch; ~10% faster (HANDOFF lesson 23)."""
    t = wp.tid()
    half = Ny // 2
    i = t // half
    r = t - i * half
    off = color
    if (i & 1) == 1:
        off = 1 - color
    j = 2 * r + off
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    Jinv = type(jcap_tol)(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = type(jcap_tol)(1.0) / J
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    p[b] = (-f[c] + s) * Jinv


@wp.kernel
def jacobi_sweep_2d(
    p:   wp.array(dtype=Any),   # flat padded (Nx+2)(Ny+2)  (read)
    p2:  wp.array(dtype=Any),   # flat padded (Nx+2)(Ny+2)  (write)
    f:   wp.array(dtype=Any),
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    Nx: int, Ny: int,
    jcap_tol: Any,
    w: Any,
):
    i, j = wp.tid()
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    Jinv = type(jcap_tol)(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = type(jcap_tol)(1.0) / J
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    p_new = (-f[c] + s) * Jinv
    p2[b] = w * p_new + (type(w)(1.0) - w) * p[b]


@wp.kernel
def residual_2d(
    p:   wp.array(dtype=Any),
    f:   wp.array(dtype=Any),
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    Nx: int, Ny: int,
    r:   wp.array(dtype=Any),
):
    """r = -f - A·p,  A·p = J·p_c - sum."""
    i, j = wp.tid()
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    r[c] = -f[c] - (J * p[b] - s)


# ── Register float32 + float64 specialisations up front ─────────────────────
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    wp.overload(rbgs_halfsweep_2d, {
        "p": _A, "f": _A, "cp0": _A, "cm0": _A, "cp1": _A, "cm1": _A,
        "jcap_tol": _dt})
    wp.overload(rbgs_halfsweep_2d_compact, {
        "p": _A, "f": _A, "cp0": _A, "cm0": _A, "cp1": _A, "cm1": _A,
        "jcap_tol": _dt})
    wp.overload(jacobi_sweep_2d, {
        "p": _A, "p2": _A, "f": _A, "cp0": _A, "cm0": _A, "cp1": _A, "cm1": _A,
        "jcap_tol": _dt, "w": _dt})
    wp.overload(residual_2d, {
        "p": _A, "f": _A, "cp0": _A, "cm0": _A, "cp1": _A, "cm1": _A, "r": _A})


# ─────────────────────────────────────────────────────────────────────────────
#  Native-signature host wrappers (drop-in for the in-solver V-cycle)
# ─────────────────────────────────────────────────────────────────────────────

def _wdev(t):
    return "cuda:0" if t.device.type == "cuda" else "cpu"


def _wpf(t):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def rbgs_sweep_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol, nsmoothing):
    """Warp port of native ``rbgs_sweep_2d``: ``nsmoothing`` red-black sweeps,
    Neumann folded into the stencil.  ``p`` (padded, contiguous) mutated in place.
    Generic in dtype (f32/f64), selected from ``p``."""
    Nx, Ny = f.shape
    wdev, wpf = _wdev(p), _wpf(p)
    fp = wp.from_torch(p.reshape(-1))
    ff = wp.from_torch(f.contiguous().reshape(-1))
    c = [wp.from_torch(x.contiguous().reshape(-1)) for x in (cp0, cm0, cp1, cm1)]
    jt = wpf(jcap_tol)
    even = (Ny % 2 == 0)
    for _ in range(int(nsmoothing)):
        for color in (0, 1):
            if even:
                wp.launch(rbgs_halfsweep_2d_compact, dim=Nx * Ny // 2,
                          inputs=[fp, ff, *c, int(Nx), int(Ny), jt, int(color)],
                          device=wdev)
            else:
                wp.launch(rbgs_halfsweep_2d, dim=(int(Nx), int(Ny)),
                          inputs=[fp, ff, *c, int(Nx), int(Ny), jt, int(color)],
                          device=wdev)


def jacobi_sweep_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol, w, nsmoothing):
    """Warp port of native ``jacobi_sweep_2d``: ``nsmoothing`` weighted-Jacobi
    sweeps (ping-pong).  ``p`` (padded, contiguous) mutated in place (the final
    result is copied back into it when ``nsmoothing`` is odd)."""
    Nx, Ny = f.shape
    wdev, wpf = _wdev(p), _wpf(p)
    a, b = p, torch.empty_like(p)
    ff = wp.from_torch(f.contiguous().reshape(-1))
    c = [wp.from_torch(x.contiguous().reshape(-1)) for x in (cp0, cm0, cp1, cm1)]
    jt, ww = wpf(jcap_tol), wpf(w)
    for _ in range(int(nsmoothing)):
        wp.launch(jacobi_sweep_2d, dim=(int(Nx), int(Ny)),
                  inputs=[wp.from_torch(a.reshape(-1)), wp.from_torch(b.reshape(-1)),
                          ff, *c, int(Nx), int(Ny), jt, ww], device=wdev)
        a, b = b, a
    if a.data_ptr() != p.data_ptr():
        p.copy_(a)
    return p


def mg_residual_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol):
    """Warp port of native ``mg_residual_2d``: returns the interior residual
    in the **native sign convention** ``r = f + A·p`` (= −1 × the ``residual_2d``
    kernel, which uses ``r = -f - A·p``) as a fresh (Nx, Ny) torch tensor.  This
    is the sign the multigrid coarse V-cycle expects for the defect correction."""
    Nx, Ny = f.shape
    wdev, wpf = _wdev(p), _wpf(p)
    r = torch.empty((Nx, Ny), dtype=p.dtype, device=p.device)
    fp = wp.from_torch(p.reshape(-1))
    ff = wp.from_torch(f.contiguous().reshape(-1))
    c = [wp.from_torch(x.contiguous().reshape(-1)) for x in (cp0, cm0, cp1, cm1)]
    wp.launch(residual_2d, dim=(int(Nx), int(Ny)),
              inputs=[fp, ff, *c, int(Nx), int(Ny), wp.from_torch(r.reshape(-1))],
              device=wdev)
    return r.neg_()


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent-array class (kept for the POC benches/tests; f32 default)
# ─────────────────────────────────────────────────────────────────────────────

class WarpRBGS2D:
    """2-D RBGS / weighted-Jacobi smoother + residual on persistent Warp arrays."""

    def __init__(self, Nx: int, Ny: int, device: str = "cuda:0",
                 jcap_tol: float = 1e-30, dtype=wp.float32):
        self.Nx, self.Ny = Nx, Ny
        self.device = device
        self.jcap_tol = jcap_tol
        self._wpf = dtype
        self._graph: Optional[wp.Graph] = None

    def setup(self, p_t, f_t, coeffs_t):
        """p_t: padded (Nx+2,Ny+2); f_t,coeffs interior (Nx,Ny) torch."""
        self._wpf = wp.float64 if p_t.dtype == torch.float64 else wp.float32
        self.p = wp.from_torch(p_t.contiguous().reshape(-1))
        self.f = wp.from_torch(f_t.contiguous().reshape(-1))
        cp0, cm0, cp1, cm1 = coeffs_t
        self.cp0 = wp.from_torch(cp0.contiguous().reshape(-1))
        self.cm0 = wp.from_torch(cm0.contiguous().reshape(-1))
        self.cp1 = wp.from_torch(cp1.contiguous().reshape(-1))
        self.cm1 = wp.from_torch(cm1.contiguous().reshape(-1))
        td = torch.float64 if self._wpf == wp.float64 else torch.float32
        self.r = wp.zeros(self.Nx * self.Ny, dtype=self._wpf, device=self.device)
        self.p2 = wp.zeros((self.Nx + 2) * (self.Ny + 2), dtype=self._wpf,
                           device=self.device)

    def _coef(self):
        return [self.cp0, self.cm0, self.cp1, self.cm1]

    def _half(self, color):
        if self.Ny % 2 == 0:
            wp.launch(rbgs_halfsweep_2d_compact, dim=self.Nx * self.Ny // 2,
                      inputs=[self.p, self.f, *self._coef(),
                              self.Nx, self.Ny, self._wpf(self.jcap_tol), color],
                      device=self.device)
        else:
            wp.launch(rbgs_halfsweep_2d, dim=(self.Nx, self.Ny),
                      inputs=[self.p, self.f, *self._coef(),
                              self.Nx, self.Ny, self._wpf(self.jcap_tol), color],
                      device=self.device)

    def sweep(self, n: int = 1):
        for _ in range(n):
            self._half(0)
            self._half(1)

    def jacobi(self, n: int = 1, w: float = 1.0):
        """n weighted-Jacobi sweeps (ping-pong); result left in self.p."""
        for _ in range(n):
            wp.launch(jacobi_sweep_2d, dim=(self.Nx, self.Ny),
                      inputs=[self.p, self.p2, self.f, *self._coef(),
                              self.Nx, self.Ny, self._wpf(self.jcap_tol),
                              self._wpf(w)],
                      device=self.device)
            self.p, self.p2 = self.p2, self.p

    def residual_norm(self) -> float:
        wp.launch(residual_2d, dim=(self.Nx, self.Ny),
                  inputs=[self.p, self.f, *self._coef(), self.Nx, self.Ny, self.r],
                  device=self.device)
        return float(wp.to_torch(self.r).norm().item())

    def capture_sweeps(self, n: int):
        with wp.ScopedCapture(device=self.device) as cap:
            self.sweep(n)
        self._graph = cap.graph

    def run_graph(self):
        if self._graph is None:
            raise RuntimeError("call capture_sweeps() first")
        wp.capture_launch(self._graph)
