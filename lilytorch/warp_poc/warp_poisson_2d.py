"""Warp single-source 2-D Poisson smoothers (RBGS + weighted Jacobi) + residual.

2-D analogue of `warp_poisson.py`.  Faithful port of the native 2-D
variable-coefficient 5-point smoothers (`multigrid_smoothers.cu`:
`rbgs_2d_tiled_kernel`, `jacobi_2d_tiled_kernel`).

Parity nuance vs native (NEW vs the 3-D case — see HANDOFF):
  * The native 2-D smoothers are **block-tiled** (8x32) and fuse all
    `nsmoothing` sweeps in shared memory using STALE inter-tile halos for
    sweeps 2..n (a multigrid block-boundary approximation).  The 3-D native
    smoothers are thread-per-cell with explicit Neumann between half-sweeps, so
    they're globally exact.  Consequences for a thread-per-cell Warp port:
      - **Jacobi, nsmoothing=1**: the tiled kernel loads p once and writes
        `w*p_new+(1-w)*p_old` from the pre-sweep neighbours → identical to a
        global Jacobi sweep on ANY grid → **bit-exact** vs native.
      - **RBGS, nsmoothing=1**: bit-exact only when the grid fits ONE tile
        (Nx<=8, Ny<=32) — otherwise native's black half-sweep reads stale red
        neighbours across tile seams.  The Warp global RBGS is the (more
        correct) un-approximated smoother; it matches native within the
        block-approx gap and converges.
  * jcap: native sets `Jinv=0` (→ writes 0) when `|J|<jcap_tol`; the port
    replicates this exactly (the 3-D Warp port instead skipped the cell — a
    benign difference since J never caps for positive face coeffs).

Neumann is folded into the stencil by index clamp (ghost = self at boundary),
bit-identical to native's pre/post `apply_neumann_bc_2d` ghost mirror for the
interior cells we compare.  Same `@wp.kernel` runs on Warp CPU and CUDA.
"""
from __future__ import annotations

import warp as wp
import torch
from typing import Optional

wp.init()


@wp.func
def _stencil_sum_2d(
    p: wp.array(dtype=wp.float32), b: int, si: int,
    i: int, j: int, Nx: int, Ny: int,
    cp0: wp.float32, cm0: wp.float32, cp1: wp.float32, cm1: wp.float32,
) -> wp.float32:
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + 1]
    if j > 0:      pjm = p[b - 1]
    return cp0 * pip + cm0 * pim + cp1 * pjp + cm1 * pjm


@wp.kernel
def rbgs_halfsweep_2d(
    p:   wp.array(dtype=wp.float32),   # flat padded (Nx+2)(Ny+2)
    f:   wp.array(dtype=wp.float32),   # flat interior Nx·Ny
    cp0: wp.array(dtype=wp.float32),
    cm0: wp.array(dtype=wp.float32),
    cp1: wp.array(dtype=wp.float32),
    cm1: wp.array(dtype=wp.float32),
    Nx: int, Ny: int,
    jcap_tol: wp.float32,
    color: int,
):
    i, j = wp.tid()
    if ((i + j) & 1) != color:
        return
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    Jinv = wp.float32(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = wp.float32(1.0) / J
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    p[b] = (-f[c] + s) * Jinv


@wp.kernel
def rbgs_halfsweep_2d_compact(
    p:   wp.array(dtype=wp.float32),
    f:   wp.array(dtype=wp.float32),
    cp0: wp.array(dtype=wp.float32),
    cm0: wp.array(dtype=wp.float32),
    cp1: wp.array(dtype=wp.float32),
    cm1: wp.array(dtype=wp.float32),
    Nx: int, Ny: int,
    jcap_tol: wp.float32,
    color: int,
):
    """Color-COMPACTED flat-1-D half-sweep (even Ny): launch only the Nx·Ny/2
    cells of one color, so NO thread early-returns on the parity check (the
    plain `rbgs_halfsweep_2d` idles ~50% of its 2-D launch).  Maps a flat tid in
    [0, Nx·Ny/2) → the t-th cell of `color`.  Bit-identical math to the full
    launch; ~10% faster (HANDOFF lesson 23)."""
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
    Jinv = wp.float32(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = wp.float32(1.0) / J
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    p[b] = (-f[c] + s) * Jinv


@wp.kernel
def jacobi_sweep_2d(
    p:   wp.array(dtype=wp.float32),   # flat padded (Nx+2)(Ny+2)  (read)
    p2:  wp.array(dtype=wp.float32),   # flat padded (Nx+2)(Ny+2)  (write)
    f:   wp.array(dtype=wp.float32),
    cp0: wp.array(dtype=wp.float32),
    cm0: wp.array(dtype=wp.float32),
    cp1: wp.array(dtype=wp.float32),
    cm1: wp.array(dtype=wp.float32),
    Nx: int, Ny: int,
    jcap_tol: wp.float32,
    w: wp.float32,
):
    i, j = wp.tid()
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    Jinv = wp.float32(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = wp.float32(1.0) / J
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    p_new = (-f[c] + s) * Jinv
    p2[b] = w * p_new + (wp.float32(1.0) - w) * p[b]


@wp.kernel
def residual_2d(
    p:   wp.array(dtype=wp.float32),
    f:   wp.array(dtype=wp.float32),
    cp0: wp.array(dtype=wp.float32),
    cm0: wp.array(dtype=wp.float32),
    cp1: wp.array(dtype=wp.float32),
    cm1: wp.array(dtype=wp.float32),
    Nx: int, Ny: int,
    r:   wp.array(dtype=wp.float32),
):
    """r = -f - A·p,  A·p = J·p_c - sum."""
    i, j = wp.tid()
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    s = _stencil_sum_2d(p, b, si, i, j, Nx, Ny, cp0[c], cm0[c], cp1[c], cm1[c])
    r[c] = -f[c] - (J * p[b] - s)


class WarpRBGS2D:
    """2-D RBGS / weighted-Jacobi smoother + residual on persistent Warp arrays."""

    def __init__(self, Nx: int, Ny: int, device: str = "cuda:0",
                 jcap_tol: float = 1e-30):
        self.Nx, self.Ny = Nx, Ny
        self.device = device
        self.jcap_tol = jcap_tol
        self._graph: Optional[wp.Graph] = None

    def setup(self, p_t, f_t, coeffs_t):
        """p_t: padded (Nx+2,Ny+2); f_t,coeffs interior (Nx,Ny) torch."""
        self.p = wp.from_torch(p_t.contiguous().reshape(-1))
        self.f = wp.from_torch(f_t.contiguous().reshape(-1))
        cp0, cm0, cp1, cm1 = coeffs_t
        self.cp0 = wp.from_torch(cp0.contiguous().reshape(-1))
        self.cm0 = wp.from_torch(cm0.contiguous().reshape(-1))
        self.cp1 = wp.from_torch(cp1.contiguous().reshape(-1))
        self.cm1 = wp.from_torch(cm1.contiguous().reshape(-1))
        self.r = wp.zeros(self.Nx * self.Ny, dtype=wp.float32, device=self.device)
        self.p2 = wp.zeros((self.Nx + 2) * (self.Ny + 2), dtype=wp.float32,
                           device=self.device)

    def _coef(self):
        return [self.cp0, self.cm0, self.cp1, self.cm1]

    def _half(self, color):
        # Color-compacted flat launch when Ny is even (no idle threads); else
        # the full 2-D launch with the parity-check early-return.  Both produce
        # bit-identical interior results.
        if self.Ny % 2 == 0:
            wp.launch(rbgs_halfsweep_2d_compact, dim=self.Nx * self.Ny // 2,
                      inputs=[self.p, self.f, *self._coef(),
                              self.Nx, self.Ny, wp.float32(self.jcap_tol), color],
                      device=self.device)
        else:
            wp.launch(rbgs_halfsweep_2d, dim=(self.Nx, self.Ny),
                      inputs=[self.p, self.f, *self._coef(),
                              self.Nx, self.Ny, wp.float32(self.jcap_tol), color],
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
                              self.Nx, self.Ny, wp.float32(self.jcap_tol),
                              wp.float32(w)],
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
