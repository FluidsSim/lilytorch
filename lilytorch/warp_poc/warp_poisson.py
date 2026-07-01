"""Warp single-source 3-D Poisson smoother (RBGS + weighted Jacobi) + residual.

Faithful port of the native variable-coefficient 7-point red-black Gauss-Seidel
(`multigrid_smoothers.cu: rbgs_3d_halfsweep_kernel`) and the weighted Jacobi
sweep + multigrid residual:

    J        = cp0+cm0 + cp1+cm1 + cp2+cm2                 (diagonal)
    sum      = cp0·p[i+1] + cm0·p[i-1] + cp1·p[j+1] + cm1·p[j-1]
                                       + cp2·p[k+1] + cm2·p[k-1]
    p[i,j,k] = (-f + sum) / J            (RBGS: only cells with (i+j+k)&1==color)

`p` is ghost-padded (Nx+2, Ny+2, Nz+2); `f` and the six face coefficients are
interior (Nx, Ny, Nz).  Homogeneous Neumann is folded into the half-sweep /
residual by index clamping (ghost = nearest interior = self at the boundary), so
NO separate BC kernel launch is needed — identical math to native's explicit
ghost refresh.  The SAME kernels run on Warp `device="cpu"` (→ C++/OpenMP) and
`"cuda:0"`.

**Precision (single source, both dtypes).**  Value arrays / float scalars are
Warp generics (`Any`); float literals are materialised in the bound element type
via `type(x)(literal)` and `wp.overload` pre-registers float32 *and* float64.
float32 codegen is unchanged from the original concrete kernels; float64 is what
an f64 solver uses.
"""
from __future__ import annotations

from typing import Any, Optional

import warp as wp
import torch

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Kernels (3-D arrays; launched over interior (Nx, Ny, Nz))
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def _stencil_sum_3d(
    p: wp.array(dtype=Any), b: int, si: int, sj: int,
    i: int, j: int, k: int, Nx: int, Ny: int, Nz: int,
    cp0: Any, cm0: Any, cp1: Any, cm1: Any, cp2: Any, cm2: Any,
):
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc; pkp = pc; pkm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + sj]
    if j > 0:      pjm = p[b - sj]
    if k < Nz - 1: pkp = p[b + 1]
    if k > 0:      pkm = p[b - 1]
    return (cp0 * pip + cm0 * pim + cp1 * pjp + cm1 * pjm
            + cp2 * pkp + cm2 * pkm)


@wp.kernel
def rbgs_halfsweep_3d(
    p:   wp.array(dtype=Any),   # flat padded (Nx+2)(Ny+2)(Nz+2)
    f:   wp.array(dtype=Any),   # flat interior Nx·Ny·Nz
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any),
    cm2: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int,
    jcap_tol: Any,
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
    s = _stencil_sum_3d(p, b, si, sj, i, j, k, Nx, Ny, Nz,
                        cp0[c], cm0[c], cp1[c], cm1[c], cp2[c], cm2[c])
    p[b] = (-f[c] + s) / J


@wp.kernel
def jacobi_sweep_3d(
    p:   wp.array(dtype=Any),   # flat padded (read)
    p2:  wp.array(dtype=Any),   # flat padded (write)
    f:   wp.array(dtype=Any),
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any),
    cm2: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int,
    jcap_tol: Any,
    w: Any,
):
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    Jinv = type(jcap_tol)(0.0)
    if J >= jcap_tol or J <= -jcap_tol:
        Jinv = type(jcap_tol)(1.0) / J
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    s = _stencil_sum_3d(p, b, si, sj, i, j, k, Nx, Ny, Nz,
                        cp0[c], cm0[c], cp1[c], cm1[c], cp2[c], cm2[c])
    p_new = (-f[c] + s) * Jinv
    p2[b] = w * p_new + (type(w)(1.0) - w) * p[b]


@wp.kernel
def neumann_fused_3d(p: wp.array(dtype=Any), Nx: int, Ny: int, Nz: int):
    """All six ghost faces in one launch, mirroring `neumann_bc_3d_fused`."""
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
    p:   wp.array(dtype=Any),
    f:   wp.array(dtype=Any),
    cp0: wp.array(dtype=Any),
    cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any),
    cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any),
    cm2: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int,
    r:   wp.array(dtype=Any),   # flat interior
):
    """r = -f - A·p,  with  A·p = J·p_c - sum  (so r→0 at convergence)."""
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    s = _stencil_sum_3d(p, b, si, sj, i, j, k, Nx, Ny, Nz,
                        cp0[c], cm0[c], cp1[c], cm1[c], cp2[c], cm2[c])
    r[c] = -f[c] - (J * p[b] - s)


# ── Register float32 + float64 specialisations up front ─────────────────────
for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    _C6 = {"p": _A, "f": _A, "cp0": _A, "cm0": _A, "cp1": _A, "cm1": _A,
           "cp2": _A, "cm2": _A}
    wp.overload(rbgs_halfsweep_3d, {**_C6, "jcap_tol": _dt})
    wp.overload(jacobi_sweep_3d, {**_C6, "p2": _A, "jcap_tol": _dt, "w": _dt})
    wp.overload(residual_3d, {**_C6, "r": _A})
    wp.overload(neumann_fused_3d, {"p": _A})


# ─────────────────────────────────────────────────────────────────────────────
#  Native-signature host wrappers (drop-in for the in-solver V-cycle)
# ─────────────────────────────────────────────────────────────────────────────

def _wdev(t):
    return "cuda:0" if t.device.type == "cuda" else "cpu"


def _wpf(t):
    return wp.float64 if t.dtype == torch.float64 else wp.float32


def rbgs_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, nsmoothing):
    """Warp port of native ``rbgs_sweep_3d``: ``nsmoothing`` red-black sweeps,
    Neumann folded into the stencil.  ``p`` (padded, contiguous) mutated in place.
    Generic in dtype (f32/f64)."""
    Nx, Ny, Nz = f.shape
    wdev, wpf = _wdev(p), _wpf(p)
    fp = wp.from_torch(p.reshape(-1))
    ff = wp.from_torch(f.contiguous().reshape(-1))
    c = [wp.from_torch(x.contiguous().reshape(-1))
         for x in (cp0, cm0, cp1, cm1, cp2, cm2)]
    jt = wpf(jcap_tol)
    for _ in range(int(nsmoothing)):
        for color in (0, 1):
            wp.launch(rbgs_halfsweep_3d, dim=(int(Nx), int(Ny), int(Nz)),
                      inputs=[fp, ff, *c, int(Nx), int(Ny), int(Nz), jt, int(color)],
                      device=wdev)


def jacobi_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol, w,
                         nsmoothing):
    """Warp port of native ``jacobi_sweep_3d``: ``nsmoothing`` weighted-Jacobi
    sweeps (ping-pong).  ``p`` (padded, contiguous) mutated in place."""
    Nx, Ny, Nz = f.shape
    wdev, wpf = _wdev(p), _wpf(p)
    a, b = p, torch.empty_like(p)
    ff = wp.from_torch(f.contiguous().reshape(-1))
    c = [wp.from_torch(x.contiguous().reshape(-1))
         for x in (cp0, cm0, cp1, cm1, cp2, cm2)]
    jt, ww = wpf(jcap_tol), wpf(w)
    for _ in range(int(nsmoothing)):
        wp.launch(jacobi_sweep_3d, dim=(int(Nx), int(Ny), int(Nz)),
                  inputs=[wp.from_torch(a.reshape(-1)), wp.from_torch(b.reshape(-1)),
                          ff, *c, int(Nx), int(Ny), int(Nz), jt, ww], device=wdev)
        a, b = b, a
    if a.data_ptr() != p.data_ptr():
        p.copy_(a)
    return p


def mg_residual_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol):
    """Warp port of native ``mg_residual_3d``: returns the interior residual in
    the **native sign convention** ``r = f + A·p`` (= −1 × the ``residual_3d``
    kernel) as a fresh (Nx, Ny, Nz) torch tensor — the sign the multigrid coarse
    V-cycle expects for the defect correction."""
    Nx, Ny, Nz = f.shape
    wdev, wpf = _wdev(p), _wpf(p)
    r = torch.empty((Nx, Ny, Nz), dtype=p.dtype, device=p.device)
    fp = wp.from_torch(p.reshape(-1))
    ff = wp.from_torch(f.contiguous().reshape(-1))
    c = [wp.from_torch(x.contiguous().reshape(-1))
         for x in (cp0, cm0, cp1, cm1, cp2, cm2)]
    wp.launch(residual_3d, dim=(int(Nx), int(Ny), int(Nz)),
              inputs=[fp, ff, *c, int(Nx), int(Ny), int(Nz),
                      wp.from_torch(r.reshape(-1))],
              device=wdev)
    return r.neg_()


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent-array class (kept for the POC benches/tests; f32 default)
# ─────────────────────────────────────────────────────────────────────────────

class WarpRBGS:
    """RBGS smoother + residual on persistent Warp arrays, graph-capturable."""

    def __init__(self, Nx: int, Ny: int, Nz: int, device: str = "cuda:0",
                 jcap_tol: float = 1e-30, dtype=wp.float32):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.device = device
        self.jcap_tol = jcap_tol
        self._wpf = dtype
        self._graph: Optional[wp.Graph] = None

    def setup(self, p_t, f_t, coeffs_t):
        """p_t: padded (Nx+2,Ny+2,Nz+2); f_t,coeffs interior (Nx,Ny,Nz) torch."""
        self._wpf = wp.float64 if p_t.dtype == torch.float64 else wp.float32
        self.p = wp.from_torch(p_t.contiguous().reshape(-1))
        self.f = wp.from_torch(f_t.contiguous().reshape(-1))
        cp0, cm0, cp1, cm1, cp2, cm2 = coeffs_t
        self.cp0 = wp.from_torch(cp0.contiguous().reshape(-1)); self.cm0 = wp.from_torch(cm0.contiguous().reshape(-1))
        self.cp1 = wp.from_torch(cp1.contiguous().reshape(-1)); self.cm1 = wp.from_torch(cm1.contiguous().reshape(-1))
        self.cp2 = wp.from_torch(cp2.contiguous().reshape(-1)); self.cm2 = wp.from_torch(cm2.contiguous().reshape(-1))
        self.r = wp.zeros(self.Nx * self.Ny * self.Nz, dtype=self._wpf, device=self.device)
        self._bc_dim = (max(self.Ny, self.Nz), max(self.Nx, self.Ny, self.Nz), 3)

    def _coef_args(self):
        return [self.cp0, self.cm0, self.cp1, self.cm1, self.cp2, self.cm2]

    def apply_neumann(self):
        """Explicit homogeneous-Neumann ghost refresh (one fused launch)."""
        wp.launch(neumann_fused_3d, dim=self._bc_dim,
                  inputs=[self.p, self.Nx, self.Ny, self.Nz], device=self.device)

    def _half(self, color):
        wp.launch(rbgs_halfsweep_3d, dim=(self.Nx, self.Ny, self.Nz),
                  inputs=[self.p, self.f, *self._coef_args(),
                          self.Nx, self.Ny, self.Nz,
                          self._wpf(self.jcap_tol), color],
                  device=self.device)

    def sweep(self, n: int = 1):
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

    def capture_sweeps(self, n: int):
        with wp.ScopedCapture(device=self.device) as cap:
            self.sweep(n)
        self._graph = cap.graph

    def run_graph(self):
        if self._graph is None:
            raise RuntimeError("call capture_sweeps() first")
        wp.capture_launch(self._graph)
