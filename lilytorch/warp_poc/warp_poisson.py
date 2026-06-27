"""Warp single-source Poisson smoother + residual (AP6 counter-demo).

Point of this file
──────────────────
Refutes the assumption that the fusible *stencil* kernels (RBGS/Jacobi smoother,
residual) must go through torch.compile rather than Warp.  They are perfectly
expressible as `@wp.kernel` — the only open question was performance vs the
hand-tiled native CUDA, which we measure in `bench_poisson.py`.

Faithful port of the native variable-coefficient 7-point red-black Gauss-Seidel
(`multigrid_smoothers.cu: rbgs_3d_halfsweep_kernel`):

    J        = cp0+cm0 + cp1+cm1 + cp2+cm2                 (diagonal)
    sum      = cp0·p[i+1] + cm0·p[i-1] + cp1·p[j+1] + cm1·p[j-1]
                                       + cp2·p[k+1] + cm2·p[k-1]
    p[i,j,k] = (-f + sum) / J            (only cells with (i+j+k)&1 == color)

`p` is ghost-padded (Nx+2, Ny+2, Nz+2); `f` and the six face coefficients are
interior (Nx, Ny, Nz).  Same memory layout as the native op, so the same torch
tensors feed both via zero-copy `wp.from_torch`.

The SAME kernels run on Warp device "cpu" (→ C++/OpenMP) and "cuda:0".
"""
from __future__ import annotations

import warp as wp
import torch
from typing import Optional

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Kernels (3-D arrays; launched over interior (Nx, Ny, Nz))
# ─────────────────────────────────────────────────────────────────────────────

#  FLAT 1-D arrays with a precomputed base index + ±stride offsets — mirrors the
#  native CUDA addressing (`p_base ± si/sj/1`).  3-D wp.array indexing recomputes
#  strides on every access and benchmarked ~1.5× slower; this closes that gap.

@wp.kernel
def rbgs_halfsweep_3d(
    p:   wp.array(dtype=wp.float32),   # flat padded (Nx+2)(Ny+2)(Nz+2)
    f:   wp.array(dtype=wp.float32),   # flat interior Nx·Ny·Nz
    cp0: wp.array(dtype=wp.float32),
    cm0: wp.array(dtype=wp.float32),
    cp1: wp.array(dtype=wp.float32),
    cm1: wp.array(dtype=wp.float32),
    cp2: wp.array(dtype=wp.float32),
    cm2: wp.array(dtype=wp.float32),
    Nx: int, Ny: int, Nz: int,
    jcap_tol: wp.float32,
    color:    int,
):
    i, j, k = wp.tid()
    if ((i + j + k) & 1) != color:
        return
    c = i * (Ny * Nz) + j * Nz + k
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    if J < jcap_tol and J > -jcap_tol:
        return
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    # Homogeneous Neumann folded in by index clamping (ghost = nearest interior
    # = self at the boundary).  Identical to native's explicit ghost refresh,
    # but needs NO separate BC kernel launch.
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc; pkp = pc; pkm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + sj]
    if j > 0:      pjm = p[b - sj]
    if k < Nz - 1: pkp = p[b + 1]
    if k > 0:      pkm = p[b - 1]
    s = (cp0[c] * pip + cm0[c] * pim
         + cp1[c] * pjp + cm1[c] * pjm
         + cp2[c] * pkp + cm2[c] * pkm)
    p[b] = (-f[c] + s) / J


@wp.kernel
def neumann_fused_3d(p: wp.array(dtype=wp.float32), Nx: int, Ny: int, Nz: int):
    """All six ghost faces in one launch (dim = (max(Ny,Nz), max(Nx,Ny,Nz), 3)),
    mirroring the native `neumann_bc_3d_fused`."""
    a, bb, face = wp.tid()
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    if face == 0:                          # x-faces: a=k, bb=j
        if a >= Nz or bb >= Ny:
            return
        jk = (bb + 1) * sj + (a + 1)
        p[jk]                = p[si + jk]
        p[(Nx + 1) * si + jk] = p[Nx * si + jk]
    elif face == 1:                        # y-faces: a=k, bb=i
        if a >= Nz or bb >= Nx:
            return
        base = (bb + 1) * si + (a + 1)
        p[base]                = p[base + sj]
        p[base + (Ny + 1) * sj] = p[base + Ny * sj]
    else:                                  # z-faces: a=j, bb=i
        if a >= Ny or bb >= Nx:
            return
        base = (bb + 1) * si + (a + 1) * sj
        p[base]          = p[base + 1]
        p[base + Nz + 1] = p[base + Nz]


@wp.kernel
def residual_3d(
    p:   wp.array(dtype=wp.float32),
    f:   wp.array(dtype=wp.float32),
    cp0: wp.array(dtype=wp.float32),
    cm0: wp.array(dtype=wp.float32),
    cp1: wp.array(dtype=wp.float32),
    cm1: wp.array(dtype=wp.float32),
    cp2: wp.array(dtype=wp.float32),
    cm2: wp.array(dtype=wp.float32),
    Nx: int, Ny: int, Nz: int,
    r:   wp.array(dtype=wp.float32),   # flat interior
):
    """r = -f - A·p,  with  A·p = J·p_c - sum  (so r→0 at convergence)."""
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc; pkp = pc; pkm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + sj]
    if j > 0:      pjm = p[b - sj]
    if k < Nz - 1: pkp = p[b + 1]
    if k > 0:      pkm = p[b - 1]
    s = (cp0[c] * pip + cm0[c] * pim
         + cp1[c] * pjp + cm1[c] * pjm
         + cp2[c] * pkp + cm2[c] * pkm)
    r[c] = -f[c] - (J * pc - s)


# ─────────────────────────────────────────────────────────────────────────────
#  Python driver (control-flow stays in Python; kernels do the work)
# ─────────────────────────────────────────────────────────────────────────────

class WarpRBGS:
    """RBGS smoother + residual on persistent Warp arrays, graph-capturable.

    Coefficients/ f are static for a given problem; p is updated in place.
    """

    def __init__(self, Nx: int, Ny: int, Nz: int, device: str = "cuda:0",
                 jcap_tol: float = 1e-30):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.device = device
        self.jcap_tol = jcap_tol
        self._graph: Optional[wp.Graph] = None

    def setup(self, p_t, f_t, coeffs_t):
        """p_t: padded (Nx+2,Ny+2,Nz+2); f_t,coeffs interior (Nx,Ny,Nz) torch.
        Flattened to 1-D Warp arrays (zero-copy views) for native-style addressing."""
        self.p = wp.from_torch(p_t.contiguous().reshape(-1))
        self.f = wp.from_torch(f_t.contiguous().reshape(-1))
        cp0, cm0, cp1, cm1, cp2, cm2 = coeffs_t
        self.cp0 = wp.from_torch(cp0.contiguous().reshape(-1)); self.cm0 = wp.from_torch(cm0.contiguous().reshape(-1))
        self.cp1 = wp.from_torch(cp1.contiguous().reshape(-1)); self.cm1 = wp.from_torch(cm1.contiguous().reshape(-1))
        self.cp2 = wp.from_torch(cp2.contiguous().reshape(-1)); self.cm2 = wp.from_torch(cm2.contiguous().reshape(-1))
        self.r = wp.zeros(self.Nx * self.Ny * self.Nz, dtype=wp.float32, device=self.device)
        # fused-Neumann launch span (matches native: span_a=max(Ny,Nz), span_b=max(Nx,Ny,Nz))
        self._bc_dim = (max(self.Ny, self.Nz), max(self.Nx, self.Ny, self.Nz), 3)

    def _coef_args(self):
        return [self.cp0, self.cm0, self.cp1, self.cm1, self.cp2, self.cm2]

    def apply_neumann(self):
        """Explicit homogeneous-Neumann ghost refresh (one fused launch).

        No longer used by `sweep()` — Neumann is now folded into the half-sweep
        by index clamping (faster, no BC launch).  Retained because non-Neumann
        BCs (e.g. Dirichlet) DO need an explicit ghost kernel like this."""
        wp.launch(neumann_fused_3d, dim=self._bc_dim,
                  inputs=[self.p, self.Nx, self.Ny, self.Nz], device=self.device)

    def _half(self, color):
        wp.launch(rbgs_halfsweep_3d, dim=(self.Nx, self.Ny, self.Nz),
                  inputs=[self.p, self.f, *self._coef_args(),
                          self.Nx, self.Ny, self.Nz,
                          wp.float32(self.jcap_tol), color],
                  device=self.device)

    def sweep(self, n: int = 1):
        """n full red-black sweeps.  Homogeneous Neumann is folded into the
        half-sweep (index clamp), so NO separate BC kernel is launched — this
        is what closes the last gap to native (which pays 2 BC launches/sweep).
        Bit-identical to native's explicit ghost refresh (ghost = self at bdry)."""
        for _ in range(n):
            self._half(0)
            self._half(1)

    def residual_norm(self) -> float:
        wp.launch(residual_3d, dim=(self.Nx, self.Ny, self.Nz),
                  inputs=[self.p, self.f, *self._coef_args(),
                          self.Nx, self.Ny, self.Nz, self.r],
                  device=self.device)
        rt = wp.to_torch(self.r)
        return float(rt.norm().item())

    # ── CUDA-graph capture of K sweeps (constant host cost) ──────────────────
    def capture_sweeps(self, n: int):
        with wp.ScopedCapture(device=self.device) as cap:
            self.sweep(n)
        self._graph = cap.graph

    def run_graph(self):
        if self._graph is None:
            raise RuntimeError("call capture_sweeps() first")
        wp.capture_launch(self._graph)
