"""Warp single-source **multigrid transfer ops** + a Warp V-cycle (AP6).

Ports the four native multigrid building blocks that, together with the already
ported RBGS smoother (``warp_poisson.py``), make a *complete* geometric V-cycle
expressible in one ``@wp.kernel`` source:

  * ``jacobi_3d``           — weighted Jacobi smoother (``multigrid_smoothers.cu``)
  * ``mg_residual_3d``      — residual ``r = f - A·p`` (register J, no global)
  * ``restrict_residual_3d``— sum-of-8 children residual restriction
  * ``restrict_face_3d``    — WaterLily 0.5·sum face-coefficient restriction
  * ``prolongate_add_3d``   — trilinear (align_corners=False) prolong + add

All grid arrays are FLAT row-major with precomputed base + ±stride offsets
(HANDOFF lesson 2).  Faithful ports of the ``.cu`` discretization (read it
first): the residual/jacobi use the padded-`p` ghosts directly (caller owns BC),
matching ``mg_residual_3d_kernel`` / ``jacobi_3d_kernel``; the smoother folds
homogeneous Neumann into the stencil by index-clamp (same trick as the RBGS
port), which is bit-identical to the native explicit-ghost refresh on the
interior.  Prolongation reproduces the native double-precision source-coordinate
arithmetic of ``linear_weights``.

The SAME kernels run on Warp ``device="cpu"`` and ``"cuda:0"``.  Tested in
float64 for machine-precision parity vs the native ops.
"""
from __future__ import annotations

import warp as wp
import torch
from typing import Optional

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  float64 RBGS half-sweep (the warp_poisson smoother is float32; the V-cycle
#  below runs in float64 for machine-precision parity, so we mirror it here —
#  same clamp-folded homogeneous-Neumann red-black GS, double precision).
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def rbgs_halfsweep_3d_f64(
    p: wp.array(dtype=wp.float64),
    f: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    cp2: wp.array(dtype=wp.float64), cm2: wp.array(dtype=wp.float64),
    Nx: int, Ny: int, Nz: int,
    jcap_tol: wp.float64, color: int,
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
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc; pkp = pc; pkm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + sj]
    if j > 0:      pjm = p[b - sj]
    if k < Nz - 1: pkp = p[b + 1]
    if k > 0:      pkm = p[b - 1]
    s = (cp0[c] * pip + cm0[c] * pim + cp1[c] * pjp
         + cm1[c] * pjm + cp2[c] * pkp + cm2[c] * pkm)
    p[b] = (-f[c] + s) / J


# ─────────────────────────────────────────────────────────────────────────────
#  Weighted Jacobi smoother (ping-pong; homogeneous-Neumann folded by clamp)
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def jacobi_3d(
    p_out: wp.array(dtype=wp.float64),
    p_in: wp.array(dtype=wp.float64),
    f: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    cp2: wp.array(dtype=wp.float64), cm2: wp.array(dtype=wp.float64),
    Nx: int, Ny: int, Nz: int,
    jcap_tol: wp.float64, w: wp.float64,
):
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    pc = p_in[b]
    if J < jcap_tol and J > -jcap_tol:
        p_out[b] = pc          # degenerate cell — copy unchanged
        return
    pip = pc; pim = pc; pjp = pc; pjm = pc; pkp = pc; pkm = pc
    if i < Nx - 1: pip = p_in[b + si]
    if i > 0:      pim = p_in[b - si]
    if j < Ny - 1: pjp = p_in[b + sj]
    if j > 0:      pjm = p_in[b - sj]
    if k < Nz - 1: pkp = p_in[b + 1]
    if k > 0:      pkm = p_in[b - 1]
    s = (cp0[c] * pip + cm0[c] * pim + cp1[c] * pjp
         + cm1[c] * pjm + cp2[c] * pkp + cm2[c] * pkm)
    p_new = (-f[c] + s) / J
    p_out[b] = w * p_new + (wp.float64(1.0) - w) * pc


# ─────────────────────────────────────────────────────────────────────────────
#  Residual r = f - A·p   (A·p = sum - J·pc).  Reads the padded-p ghosts
#  directly (caller owns the BC) — faithful to mg_residual_3d_kernel.
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def mg_residual_3d(
    p: wp.array(dtype=wp.float64),
    f: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    cp2: wp.array(dtype=wp.float64), cm2: wp.array(dtype=wp.float64),
    r: wp.array(dtype=wp.float64),
    Nx: int, Ny: int, Nz: int,
    jcap_tol: wp.float64,
):
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    if J < jcap_tol and J > -jcap_tol:
        r[c] = wp.float64(0.0)
        return
    s = (cp0[c] * p[b + si] + cm0[c] * p[b - si]
         + cp1[c] * p[b + sj] + cm1[c] * p[b - sj]
         + cp2[c] * p[b + 1] + cm2[c] * p[b - 1])
    r[c] = f[c] - s + J * p[b]


# ─────────────────────────────────────────────────────────────────────────────
#  Restriction: sum-of-8 children residual, and 0.5·sum WaterLily face coeff.
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def restrict_residual_3d(
    r: wp.array(dtype=wp.float64),
    rc: wp.array(dtype=wp.float64),
    Nxf: int, Nyf: int, Nzf: int,
    Nxc: int, Nyc: int, Nzc: int,
):
    I, J, K = wp.tid()
    sjf = Nzf
    sif = Nyf * Nzf
    sjc = Nzc
    sic = Nyc * Nzc
    s = wp.float64(0.0)
    for di in range(2):
        ii = 2 * I + di
        if ii < Nxf:
            for dj in range(2):
                jj = 2 * J + dj
                if jj < Nyf:
                    for dk in range(2):
                        kk = 2 * K + dk
                        if kk < Nzf:
                            s = s + r[ii * sif + jj * sjf + kk]
    rc[I * sic + J * sjc + K] = s


@wp.kernel
def restrict_face_3d(
    src: wp.array(dtype=wp.float64),
    dst: wp.array(dtype=wp.float64),
    Nf0: int, Nf1: int, Nf2: int,
    Nc0: int, Nc1: int, Nc2: int,
    face_dim: int,
):
    I, J, K = wp.tid()
    sjf = Nf2
    sif = Nf1 * Nf2
    sjc = Nc2
    sic = Nc1 * Nc2
    ni = 2; nj = 2; nk = 2
    if face_dim == 0:
        ni = 1
    elif face_dim == 1:
        nj = 1
    else:
        nk = 1
    s = wp.float64(0.0)
    for di in range(ni):
        ii = 2 * I + di
        if ii < Nf0:
            for dj in range(nj):
                jj = 2 * J + dj
                if jj < Nf1:
                    for dk in range(nk):
                        kk = 2 * K + dk
                        if kk < Nf2:
                            s = s + src[ii * sif + jj * sjf + kk]
    dst[I * sic + J * sjc + K] = wp.float64(0.5) * s


# ─────────────────────────────────────────────────────────────────────────────
#  Prolongation + correction (trilinear, align_corners=False), fused add.
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def linear_weights(dst: int, Nc: int, Nf: int):
    """Cell-centred align_corners=False weights — double-precision coord math
    matching the native ``linear_weights`` helper exactly."""
    src = (wp.float64(dst) + wp.float64(0.5)) * (wp.float64(Nc) / wp.float64(Nf)) - wp.float64(0.5)
    sf = wp.floor(src)
    il = wp.int32(sf)
    ir = il + 1
    if il < 0:
        il = 0
    if ir < 0:
        ir = 0
    if il > Nc - 1:
        il = Nc - 1
    if ir > Nc - 1:
        ir = Nc - 1
    w = src - sf
    if w < wp.float64(0.0):
        w = wp.float64(0.0)
    if w > wp.float64(1.0):
        w = wp.float64(1.0)
    wl = wp.float64(0.0)
    wr = wp.float64(0.0)
    if src <= wp.float64(0.0):
        wl = wp.float64(1.0)
        wr = wp.float64(0.0)
    elif src >= wp.float64(Nc) - wp.float64(1.0):
        wl = wp.float64(0.0)
        wr = wp.float64(1.0)
    else:
        wl = wp.float64(1.0) - w
        wr = w
    return il, ir, wl, wr


@wp.kernel
def prolongate_add_3d(
    ec: wp.array(dtype=wp.float64),   # ghost-padded coarse (Nxc+2, Nyc+2, Nzc+2)
    p: wp.array(dtype=wp.float64),    # ghost-padded fine   (Nxf+2, Nyf+2, Nzf+2)
    Nxc: int, Nyc: int, Nzc: int,
    Nxf: int, Nyf: int, Nzf: int,
):
    i, j, k = wp.tid()
    il, ir, wil, wir = linear_weights(i, Nxc, Nxf)
    jl, jr, wjl, wjr = linear_weights(j, Nyc, Nyf)
    kl, kr, wkl, wkr = linear_weights(k, Nzc, Nzf)

    sje = Nzc + 2
    sie = (Nyc + 2) * (Nzc + 2)

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
            for dk in range(2):
                kk = kl
                wk = wkl
                if dk == 1:
                    kk = kr
                    wk = wkr
                interp = interp + wi * wj * wk * ec[
                    (ii + 1) * sie + (jj + 1) * sje + (kk + 1)]

    sjp = Nzf + 2
    sip = (Nyf + 2) * (Nzf + 2)
    pidx = (i + 1) * sip + (j + 1) * sjp + (k + 1)
    p[pidx] = p[pidx] + interp


# ─────────────────────────────────────────────────────────────────────────────
#  Python launch wrappers (mirror the native ops.py signatures)
# ─────────────────────────────────────────────────────────────────────────────

def _wd(t):
    return "cuda:0" if t.device.type == "cuda" else "cpu"


def _f(x):
    return wp.from_torch(x.contiguous().reshape(-1))


def jacobi_sweep_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2,
                         jcap_tol, w, nsmoothing):
    """Weighted-Jacobi smoother; mutates ``p`` (padded Nx+2,Ny+2,Nz+2) in place.
    Returns nothing (matches native).  Homogeneous Neumann folded via clamp →
    interior bit-identical to the native explicit-ghost path."""
    Nx, Ny, Nz = f.shape
    wd = _wd(p)
    pin = _f(p)
    ptmp = wp.zeros_like(pin)
    coefs = [_f(cp0), _f(cm0), _f(cp1), _f(cm1), _f(cp2), _f(cm2)]
    ff = _f(f)
    src, dst = pin, ptmp
    for _ in range(int(nsmoothing)):
        wp.launch(jacobi_3d, dim=(Nx, Ny, Nz),
                  inputs=[dst, src, ff, *coefs, int(Nx), int(Ny), int(Nz),
                          wp.float64(jcap_tol), wp.float64(w)],
                  device=wd)
        src, dst = dst, src
    if int(nsmoothing) % 2 == 1:
        wp.copy(pin, src)   # result landed in ptmp → copy back into p storage


def mg_residual_3d_warp(p, f, cp0, cm0, cp1, cm1, cp2, cm2, jcap_tol):
    """Returns r (Nx,Ny,Nz) torch float64."""
    Nx, Ny, Nz = f.shape
    wd = _wd(p)
    r = torch.empty_like(f)
    wp.launch(mg_residual_3d, dim=(Nx, Ny, Nz),
              inputs=[_f(p), _f(f), _f(cp0), _f(cm0), _f(cp1), _f(cm1),
                      _f(cp2), _f(cm2), _f(r), int(Nx), int(Ny), int(Nz),
                      wp.float64(jcap_tol)],
              device=wd)
    return r


def restrict_residual_3d_warp(r, rc):
    """Sum-of-8 children restriction; writes rc in place."""
    Nxf, Nyf, Nzf = r.shape
    Nxc, Nyc, Nzc = rc.shape
    wp.launch(restrict_residual_3d, dim=(Nxc, Nyc, Nzc),
              inputs=[_f(r), _f(rc), int(Nxf), int(Nyf), int(Nzf),
                      int(Nxc), int(Nyc), int(Nzc)],
              device=_wd(r))


def restrict_face_3d_warp(src, dst, face_dim):
    """WaterLily 0.5·sum face restriction; writes dst in place."""
    Nf0, Nf1, Nf2 = src.shape
    Nc0, Nc1, Nc2 = dst.shape
    wp.launch(restrict_face_3d, dim=(Nc0, Nc1, Nc2),
              inputs=[_f(src), _f(dst), int(Nf0), int(Nf1), int(Nf2),
                      int(Nc0), int(Nc1), int(Nc2), int(face_dim)],
              device=_wd(src))


def prolongate_add_3d_warp(ec, p):
    """Trilinear prolong of ec[interior] added into p[interior]; in place."""
    Nxc, Nyc, Nzc = ec.shape[0] - 2, ec.shape[1] - 2, ec.shape[2] - 2
    Nxf, Nyf, Nzf = p.shape[0] - 2, p.shape[1] - 2, p.shape[2] - 2
    wp.launch(prolongate_add_3d, dim=(Nxf, Nyf, Nzf),
              inputs=[_f(ec), _f(p), int(Nxc), int(Nyc), int(Nzc),
                      int(Nxf), int(Nyf), int(Nzf)],
              device=_wd(p))


# ─────────────────────────────────────────────────────────────────────────────
#  Geometric V-cycle (Warp-only), graph-capturable.
#
#  Demonstrates the ported transfer ops + the RBGS smoother compose into a
#  working V-cycle, built EXACTLY like the native ``poisson_mult._vcycle_rbgs_3d``:
#  the operator is the WaterLily 7-point Laplacian whose six stencil coefficients
#  (``cp0..cm2``) are slices of the staggered face-coefficient arrays
#  ``ch/cv/cw``; the coarse operator is obtained by restricting those faces with
#  ``restrict_face`` (the 0.5·sum WaterLily rule), the residual is restricted by
#  sum-of-8, and the coarse correction is trilinearly prolongated back.  Domain
#  boundary faces carry coefficient 0 → a proper (singular) Neumann Laplacian.
#  Persistent per-level Warp arrays so the whole cycle is CUDA-graph capturable.
# ─────────────────────────────────────────────────────────────────────────────

def _const_faces(n, device):
    """Fine face-coefficient arrays for the unit 7-point Laplacian: interior
    faces = 1, domain-boundary faces = 0 (no-flux Neumann)."""
    o = dict(dtype=torch.float64, device=device)
    ch = torch.ones(n + 1, n, n, **o); ch[0] = 0; ch[n] = 0
    cv = torch.ones(n, n + 1, n, **o); cv[:, 0] = 0; cv[:, n] = 0
    cw = torch.ones(n, n, n + 1, **o); cw[:, :, 0] = 0; cw[:, :, n] = 0
    return ch, cv, cw


class WarpVCycle:
    """Geometric multigrid V-cycle for the Neumann Laplacian, built entirely
    from the ported Warp kernels (smoother + the 4 transfer ops).  Validation
    vehicle for the transfer ops, mirroring ``poisson_mult._vcycle_rbgs_3d``."""

    def __init__(self, N, device="cuda:0", nu1=2, nu2=2, min_N=4, jcap=1e-30):
        self.device = device
        self.nu1, self.nu2, self.min_N = nu1, nu2, min_N
        self.jcap = jcap
        self.levels = []
        ch, cv, cw = _const_faces(N, device)
        n = N
        while True:
            self.levels.append(self._make_level(n, ch, cv, cw))
            if n <= min_N or n % 2 != 0:
                break
            ch, cv, cw = self._restrict_faces(ch, cv, cw, n)
            n //= 2
        self._graph: Optional[wp.Graph] = None

    def _restrict_faces(self, ch, cv, cw, n):
        nc = n // 2
        d = self.device
        o = dict(dtype=torch.float64, device=d)
        ch_c = torch.empty(nc + 1, nc, nc, **o)
        cv_c = torch.empty(nc, nc + 1, nc, **o)
        cw_c = torch.empty(nc, nc, nc + 1, **o)
        restrict_face_3d_warp(ch, ch_c, 0)
        restrict_face_3d_warp(cv, cv_c, 1)
        restrict_face_3d_warp(cw, cw_c, 2)
        return ch_c, cv_c, cw_c

    def _make_level(self, n, ch, cv, cw):
        d = self.device
        p = wp.zeros((n + 2) ** 3, dtype=wp.float64, device=d)
        f = wp.zeros(n ** 3, dtype=wp.float64, device=d)
        r = wp.zeros(n ** 3, dtype=wp.float64, device=d)
        # cp0..cm2 are face-coefficient slices (contiguous so wp.from_torch can
        # take a flat zero-copy view), exactly as native _vcycle_rbgs_3d.
        ct = [ch[1:, :, :].contiguous(), ch[:-1, :, :].contiguous(),
              cv[:, 1:, :].contiguous(), cv[:, :-1, :].contiguous(),
              cw[:, :, 1:].contiguous(), cw[:, :, :-1].contiguous()]
        coefs = [_f(t) for t in ct]
        return dict(n=n, p=p, f=f, r=r, coefs=coefs, _ct=ct)

    def _rbgs(self, lv, sweeps):
        n = lv["n"]
        for _ in range(sweeps):
            for color in (0, 1):
                wp.launch(rbgs_halfsweep_3d_f64, dim=(n, n, n),
                          inputs=[lv["p"], lv["f"], *lv["coefs"],
                                  n, n, n, wp.float64(self.jcap), color],
                          device=self.device)

    def _vcycle(self, lvl):
        lv = self.levels[lvl]
        n = lv["n"]
        if lvl == len(self.levels) - 1:
            self._rbgs(lv, self.nu1 + self.nu2 + 8)   # coarse "solve"
            return
        self._rbgs(lv, self.nu1)
        # residual on this level
        wp.launch(mg_residual_3d, dim=(n, n, n),
                  inputs=[lv["p"], lv["f"], *lv["coefs"], lv["r"],
                          n, n, n, wp.float64(self.jcap)],
                  device=self.device)
        cv = self.levels[lvl + 1]
        nc = cv["n"]
        # restrict residual → coarse f, zero coarse p
        wp.launch(restrict_residual_3d, dim=(nc, nc, nc),
                  inputs=[lv["r"], cv["f"], n, n, n, nc, nc, nc],
                  device=self.device)
        cv["p"].zero_()
        self._vcycle(lvl + 1)
        # prolongate coarse correction into fine p
        wp.launch(prolongate_add_3d, dim=(n, n, n),
                  inputs=[cv["p"], lv["p"], nc, nc, nc, n, n, n],
                  device=self.device)
        self._rbgs(lv, self.nu2)

    def set_rhs(self, f_torch):
        """f_torch: (N,N,N) zero-mean RHS."""
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
        wp.launch(mg_residual_3d, dim=(n, n, n),
                  inputs=[lv["p"], lv["f"], *lv["coefs"], lv["r"],
                          n, n, n, wp.float64(self.jcap)],
                  device=self.device)
        return float(wp.to_torch(lv["r"]).norm().item())

    def solution_interior(self):
        n = self.levels[0]["n"]
        pt = wp.to_torch(self.levels[0]["p"]).reshape(n + 2, n + 2, n + 2)
        return pt[1:-1, 1:-1, 1:-1]
