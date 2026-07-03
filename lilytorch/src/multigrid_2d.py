"""Warp 2-D multigrid transfer ops + a converging Warp V-cycle (AP6, P4).

2-D analogue of `warp_multigrid.py`.  Ports `mg_residual_2d`,
`restrict_residual_2d` (sum-of-4), `restrict_face_2d` (0.5·sum WaterLily),
`prolongate_add_2d` (bilinear, align_corners=False) and assembles them with a
float64 RBGS smoother into `WarpVCycle2D` — demonstrating the Poisson OUTER
DRIVER (control flow) composes with Warp kernel-level ops in 2-D, exactly like
the native `poisson_mult._vcycle_rbgs_2d`.  The driver stays Python / CUDA-graph
capturable; Warp supplies only the per-cell kernels.

Reuses the native-matching double-precision `linear_weights` from the 3-D
module.  Same `@wp.kernel` runs on Warp CPU and CUDA; float64 for parity.
"""
from __future__ import annotations

import warp as wp
import torch
from typing import Optional

wp.init()

from lilytorch.src.multigrid import linear_weights, _wd, _f


@wp.kernel
def rbgs_halfsweep_2d_f64(
    p: wp.array(dtype=wp.float64),
    f: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    Nx: int, Ny: int,
    jcap_tol: wp.float64, color: int,
):
    i, j = wp.tid()
    if ((i + j) & 1) != color:
        return
    c = i * Ny + j
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    if J < jcap_tol and J > -jcap_tol:
        return
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + 1]
    if j > 0:      pjm = p[b - 1]
    s = cp0[c] * pip + cm0[c] * pim + cp1[c] * pjp + cm1[c] * pjm
    p[b] = (-f[c] + s) / J


@wp.kernel
def mg_residual_2d(
    p: wp.array(dtype=wp.float64),
    f: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    r: wp.array(dtype=wp.float64),
    Nx: int, Ny: int,
    jcap_tol: wp.float64,
):
    i, j = wp.tid()
    c = i * Ny + j
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    if J < jcap_tol and J > -jcap_tol:
        r[c] = wp.float64(0.0)
        return
    s = (cp0[c] * p[b + si] + cm0[c] * p[b - si]
         + cp1[c] * p[b + 1] + cm1[c] * p[b - 1])
    r[c] = f[c] - s + J * p[b]


@wp.kernel
def restrict_residual_2d(
    r: wp.array(dtype=wp.float64),
    rc: wp.array(dtype=wp.float64),
    Nxf: int, Nyf: int, Nxc: int, Nyc: int,
):
    # Match the native add order a+b+c+d (a=[i0,j0], b=[i1,j0], c=[i0,j1],
    # d=[i1,j1]) → bit-exact (pure additions, no FMA).
    I, J = wp.tid()
    i0 = 2 * I; i1 = i0 + 1
    j0 = 2 * J; j1 = j0 + 1
    a = r[i0 * Nyf + j0]
    b = wp.float64(0.0)
    if i1 < Nxf:
        b = r[i1 * Nyf + j0]
    c = wp.float64(0.0)
    if j1 < Nyf:
        c = r[i0 * Nyf + j1]
    d = wp.float64(0.0)
    if i1 < Nxf and j1 < Nyf:
        d = r[i1 * Nyf + j1]
    rc[I * Nyc + J] = a + b + c + d


@wp.kernel
def restrict_face_2d(
    src: wp.array(dtype=wp.float64),
    dst: wp.array(dtype=wp.float64),
    Nf0: int, Nf1: int, Nc0: int, Nc1: int,
    face_dim: int,
):
    I, J = wp.tid()
    ni = 2; nj = 2
    if face_dim == 0:
        ni = 1
    else:
        nj = 1
    s = wp.float64(0.0)
    for di in range(ni):
        ii = 2 * I + di
        if ii < Nf0:
            for dj in range(nj):
                jj = 2 * J + dj
                if jj < Nf1:
                    s = s + src[ii * Nf1 + jj]
    dst[I * Nc1 + J] = wp.float64(0.5) * s


@wp.kernel
def prolongate_add_2d(
    ec: wp.array(dtype=wp.float64),   # padded coarse (Nxc+2, Nyc+2)
    p: wp.array(dtype=wp.float64),    # padded fine   (Nxf+2, Nyf+2)
    Nxc: int, Nyc: int, Nxf: int, Nyf: int,
):
    i, j = wp.tid()
    il, ir, wil, wir = linear_weights(i, Nxc, Nxf)
    jl, jr, wjl, wjr = linear_weights(j, Nyc, Nyf)
    sie = Nyc + 2
    interp = wp.float64(0.0)
    for di in range(2):
        ii = il
        wi = wil
        if di == 1:
            ii = ir
            wi = wir
        for dj in range(2):
            jj = jl
            wj = wjl
            if dj == 1:
                jj = jr
                wj = wjr
            interp = interp + wi * wj * ec[(ii + 1) * sie + (jj + 1)]
    sip = Nyf + 2
    pidx = (i + 1) * sip + (j + 1)
    p[pidx] = p[pidx] + interp


# ── launch wrappers (mirror native ops.py signatures) ────────────────────────

def mg_residual_2d_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol):
    Nx, Ny = f.shape
    r = torch.empty_like(f)
    wp.launch(mg_residual_2d, dim=(Nx, Ny),
              inputs=[_f(p), _f(f), _f(cp0), _f(cm0), _f(cp1), _f(cm1), _f(r),
                      int(Nx), int(Ny), wp.float64(jcap_tol)],
              device=_wd(p))
    return r


def restrict_residual_2d_warp(r, rc):
    Nxf, Nyf = r.shape
    Nxc, Nyc = rc.shape
    wp.launch(restrict_residual_2d, dim=(Nxc, Nyc),
              inputs=[_f(r), _f(rc), int(Nxf), int(Nyf), int(Nxc), int(Nyc)],
              device=_wd(r))


def restrict_face_2d_warp(src, dst, face_dim):
    Nf0, Nf1 = src.shape
    Nc0, Nc1 = dst.shape
    wp.launch(restrict_face_2d, dim=(Nc0, Nc1),
              inputs=[_f(src), _f(dst), int(Nf0), int(Nf1), int(Nc0), int(Nc1),
                      int(face_dim)],
              device=_wd(src))


def prolongate_add_2d_warp(ec, p):
    Nxc, Nyc = ec.shape[0] - 2, ec.shape[1] - 2
    Nxf, Nyf = p.shape[0] - 2, p.shape[1] - 2
    wp.launch(prolongate_add_2d, dim=(Nxf, Nyf),
              inputs=[_f(ec), _f(p), int(Nxc), int(Nyc), int(Nxf), int(Nyf)],
              device=_wd(p))


# ── Warp-only geometric V-cycle (2-D) ────────────────────────────────────────

def _const_faces_2d(n, device):
    o = dict(dtype=torch.float64, device=device)
    ch = torch.ones(n + 1, n, **o); ch[0] = 0; ch[n] = 0
    cv = torch.ones(n, n + 1, **o); cv[:, 0] = 0; cv[:, n] = 0
    return ch, cv


class WarpVCycle2D:
    """2-D geometric multigrid V-cycle (Neumann Laplacian) built entirely from
    the ported Warp kernels — the 2-D analogue of `WarpVCycle`."""

    def __init__(self, N, device="cuda:0", nu1=2, nu2=2, min_N=4, jcap=1e-30):
        self.device = device
        self.nu1, self.nu2, self.min_N, self.jcap = nu1, nu2, min_N, jcap
        self.levels = []
        ch, cv = _const_faces_2d(N, device)
        n = N
        while True:
            self.levels.append(self._make_level(n, ch, cv))
            if n <= min_N or n % 2 != 0:
                break
            ch, cv = self._restrict_faces(ch, cv, n)
            n //= 2
        self._graph: Optional[wp.Graph] = None

    def _restrict_faces(self, ch, cv, n):
        nc = n // 2
        o = dict(dtype=torch.float64, device=self.device)
        ch_c = torch.empty(nc + 1, nc, **o)
        cv_c = torch.empty(nc, nc + 1, **o)
        restrict_face_2d_warp(ch, ch_c, 0)
        restrict_face_2d_warp(cv, cv_c, 1)
        return ch_c, cv_c

    def _make_level(self, n, ch, cv):
        d = self.device
        p = wp.zeros((n + 2) ** 2, dtype=wp.float64, device=d)
        f = wp.zeros(n ** 2, dtype=wp.float64, device=d)
        r = wp.zeros(n ** 2, dtype=wp.float64, device=d)
        ct = [ch[1:, :].contiguous(), ch[:-1, :].contiguous(),
              cv[:, 1:].contiguous(), cv[:, :-1].contiguous()]
        coefs = [_f(t) for t in ct]
        return dict(n=n, p=p, f=f, r=r, coefs=coefs, _ct=ct)

    def _rbgs(self, lv, sweeps):
        n = lv["n"]
        for _ in range(sweeps):
            for color in (0, 1):
                wp.launch(rbgs_halfsweep_2d_f64, dim=(n, n),
                          inputs=[lv["p"], lv["f"], *lv["coefs"],
                                  n, n, wp.float64(self.jcap), color],
                          device=self.device)

    def _vcycle(self, lvl):
        lv = self.levels[lvl]
        n = lv["n"]
        if lvl == len(self.levels) - 1:
            self._rbgs(lv, self.nu1 + self.nu2 + 8)
            return
        self._rbgs(lv, self.nu1)
        wp.launch(mg_residual_2d, dim=(n, n),
                  inputs=[lv["p"], lv["f"], *lv["coefs"], lv["r"],
                          n, n, wp.float64(self.jcap)],
                  device=self.device)
        cvl = self.levels[lvl + 1]
        nc = cvl["n"]
        wp.launch(restrict_residual_2d, dim=(nc, nc),
                  inputs=[lv["r"], cvl["f"], n, n, nc, nc],
                  device=self.device)
        cvl["p"].zero_()
        self._vcycle(lvl + 1)
        wp.launch(prolongate_add_2d, dim=(n, n),
                  inputs=[cvl["p"], lv["p"], nc, nc, n, n],
                  device=self.device)
        self._rbgs(lv, self.nu2)

    def set_rhs(self, f_torch):
        wp.copy(self.levels[0]["f"], _f(f_torch))
        self.levels[0]["p"].zero_()

    def cycle(self):
        self._vcycle(0)

    def capture(self):
        with wp.ScopedCapture(device=self.device) as cap:
            self._vcycle(0)
        self._graph = cap.graph

    def run_graph(self):
        wp.capture_launch(self._graph)

    def residual_norm(self):
        lv = self.levels[0]
        n = lv["n"]
        wp.launch(mg_residual_2d, dim=(n, n),
                  inputs=[lv["p"], lv["f"], *lv["coefs"], lv["r"],
                          n, n, wp.float64(self.jcap)],
                  device=self.device)
        return float(wp.to_torch(lv["r"]).norm().item())
