"""Variable-coefficient, CUDA-graph-captured Warp multigrid (3-D + 2-D).

Closes the Item-2 Poisson perf gap.  The Item-2 driver (`src_warp.poisson_mult`)
runs the multigrid OUTER loop in Python with a per-cycle ``.item()`` residual
sync — correct + native-independent but 5–20× slower than the native C++
``poisson_solve_*`` which fuses the whole solve into one sync-free host launch.

This module rebuilds the V-cycle as an **all-Warp** geometric multigrid (smoother
+ residual + restriction + prolongation all on ``@wp.kernel``s, no torch in the
cycle) so the entire fixed-cycle solve can be captured into ONE Warp CUDA graph
and replayed per step with a single host launch — the Warp analogue of the
native GU6 ``tol<0`` sync-free path.

Differences vs the demo ``warp_multigrid.WarpVCycle`` (square / f64 / constant
coefficient):
  * **variable coefficients** — the live BDIM ``ch/cv/cw`` are copied into the
    level-0 face buffers each step; coarse-level coefficients are recomputed
    in-graph by ``restrict_face`` (the body moves every step).
  * **anisotropic grids** — independent ``(Nx, Ny, Nz)`` per level.
  * **dtype-generic** — f32 + f64 (``wp.overload``).
  * **fixed cycle count** — no early-exit, no host sync → graph-capturable.

Parity vs native is residual-level (same converged residual), matching the
Item-2 contract.  Sign convention: ``mg_residual`` returns ``f + A·p`` (native
``mg_residual`` sign), self-consistent through the restriction → coarse-RHS path.
"""
from __future__ import annotations

from typing import Any, Optional

import warp as wp
import torch

wp.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Generic kernels (3-D)
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def rbgs_halfsweep_3d(
    p: wp.array(dtype=Any), f: wp.array(dtype=Any),
    cp0: wp.array(dtype=Any), cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any), cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any), cm2: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int, jcap_tol: Any, color: int,
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


@wp.kernel
def mg_residual_3d(
    p: wp.array(dtype=Any), f: wp.array(dtype=Any),
    cp0: wp.array(dtype=Any), cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any), cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any), cm2: wp.array(dtype=Any),
    r: wp.array(dtype=Any), Nx: int, Ny: int, Nz: int, jcap_tol: Any,
):
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    if J < jcap_tol and J > -jcap_tol:
        r[c] = type(jcap_tol)(0.0)
        return
    # Clamp ghosts (homogeneous-Neumann fold) IDENTICALLY to the smoother, so
    # the residual is consistent with the operator the smoother inverts even
    # when domain-boundary face coefficients are non-zero (the Warp smoother
    # never writes ghost cells).
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
    r[c] = f[c] - s + J * pc


@wp.kernel
def restrict_residual_3d(
    r: wp.array(dtype=Any), rc: wp.array(dtype=Any),
    Nxf: int, Nyf: int, Nzf: int, Nxc: int, Nyc: int, Nzc: int,
):
    I, J, K = wp.tid()
    sjf = Nzf; sif = Nyf * Nzf
    sjc = Nzc; sic = Nyc * Nzc
    s = type(r[0])(0.0)
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
    src: wp.array(dtype=Any), dst: wp.array(dtype=Any),
    Nf0: int, Nf1: int, Nf2: int, Nc0: int, Nc1: int, Nc2: int, face_dim: int,
):
    I, J, K = wp.tid()
    sjf = Nf2; sif = Nf1 * Nf2
    sjc = Nc2; sic = Nc1 * Nc2
    ni = 2; nj = 2; nk = 2
    if face_dim == 0:
        ni = 1
    elif face_dim == 1:
        nj = 1
    else:
        nk = 1
    s = type(src[0])(0.0)
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
    dst[I * sic + J * sjc + K] = type(src[0])(0.5) * s


@wp.func
def _lw(dst: int, Nc: int, Nf: int):
    """align_corners=False cell-centred weights (double-precision coord math)."""
    src = (wp.float64(dst) + wp.float64(0.5)) * (wp.float64(Nc) / wp.float64(Nf)) - wp.float64(0.5)
    sf = wp.floor(src)
    il = wp.int32(sf)
    ir = il + 1
    if il < 0: il = 0
    if ir < 0: ir = 0
    if il > Nc - 1: il = Nc - 1
    if ir > Nc - 1: ir = Nc - 1
    w = src - sf
    wl = wp.float64(0.0); wr = wp.float64(0.0)
    if src <= wp.float64(0.0):
        wl = wp.float64(1.0); wr = wp.float64(0.0)
    elif src >= wp.float64(Nc) - wp.float64(1.0):
        wl = wp.float64(0.0); wr = wp.float64(1.0)
    else:
        wl = wp.float64(1.0) - w; wr = w
    return il, ir, wl, wr


@wp.kernel
def prolongate_add_3d(
    ec: wp.array(dtype=Any), p: wp.array(dtype=Any),
    Nxc: int, Nyc: int, Nzc: int, Nxf: int, Nyf: int, Nzf: int,
):
    i, j, k = wp.tid()
    il, ir, wil, wir = _lw(i, Nxc, Nxf)
    jl, jr, wjl, wjr = _lw(j, Nyc, Nyf)
    kl, kr, wkl, wkr = _lw(k, Nzc, Nzf)
    sje = Nzc + 2
    sie = (Nyc + 2) * (Nzc + 2)
    interp = wp.float64(0.0)
    for di in range(2):
        ii = il; wi = wil
        if di == 1:
            ii = ir; wi = wir
        for dj in range(2):
            jj = jl; wj = wjl
            if dj == 1:
                jj = jr; wj = wjr
            for dk in range(2):
                kk = kl; wk = wkl
                if dk == 1:
                    kk = kr; wk = wkr
                interp = interp + wi * wj * wk * wp.float64(
                    ec[(ii + 1) * sie + (jj + 1) * sje + (kk + 1)])
    sjp = Nzf + 2
    sip = (Nyf + 2) * (Nzf + 2)
    pidx = (i + 1) * sip + (j + 1) * sjp + (k + 1)
    p[pidx] = p[pidx] + type(p[0])(interp)


@wp.kernel
def extract_pairs_3d(
    ch: wp.array(dtype=Any), cv: wp.array(dtype=Any), cw: wp.array(dtype=Any),
    cp0: wp.array(dtype=Any), cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any), cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any), cm2: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int,
):
    """Materialise the six face-pair coefficient arrays into contiguous
    (Nx,Ny,Nz) buffers from the face arrays ch(Nx+1,Ny,Nz)/cv(Nx,Ny+1,Nz)/
    cw(Nx,Ny,Nz+1).  Needed because y/z face slices are non-contiguous, so a
    persistent flat view can't alias them inside a captured graph."""
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    # ch: (Nx+1, Ny, Nz)
    sci = Ny * Nz
    cp0[c] = ch[(i + 1) * sci + j * Nz + k]
    cm0[c] = ch[i * sci + j * Nz + k]
    # cv: (Nx, Ny+1, Nz)
    svi = (Ny + 1) * Nz
    cp1[c] = cv[i * svi + (j + 1) * Nz + k]
    cm1[c] = cv[i * svi + j * Nz + k]
    # cw: (Nx, Ny, Nz+1)
    swi = Ny * (Nz + 1)
    swj = Nz + 1
    cp2[c] = cw[i * swi + j * swj + (k + 1)]
    cm2[c] = cw[i * swi + j * swj + k]


@wp.kernel
def jacobi_3d(
    p_out: wp.array(dtype=Any), p_in: wp.array(dtype=Any), f: wp.array(dtype=Any),
    cp0: wp.array(dtype=Any), cm0: wp.array(dtype=Any),
    cp1: wp.array(dtype=Any), cm1: wp.array(dtype=Any),
    cp2: wp.array(dtype=Any), cm2: wp.array(dtype=Any),
    Nx: int, Ny: int, Nz: int, jcap_tol: Any, w: Any,
):
    i, j, k = wp.tid()
    c = i * (Ny * Nz) + j * Nz + k
    si = (Ny + 2) * (Nz + 2)
    sj = Nz + 2
    b = (i + 1) * si + (j + 1) * sj + (k + 1)
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c] + cp2[c] + cm2[c]
    pc = p_in[b]
    if J < jcap_tol and J > -jcap_tol:
        p_out[b] = pc
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
    p_out[b] = w * p_new + (type(w)(1.0) - w) * pc


for _dt in (wp.float32, wp.float64):
    _A = wp.array(dtype=_dt)
    _C6 = {"cp0": _A, "cm0": _A, "cp1": _A, "cm1": _A, "cp2": _A, "cm2": _A}
    wp.overload(rbgs_halfsweep_3d, {"p": _A, "f": _A, **_C6, "jcap_tol": _dt})
    wp.overload(mg_residual_3d, {"p": _A, "f": _A, **_C6, "r": _A, "jcap_tol": _dt})
    wp.overload(jacobi_3d, {"p_out": _A, "p_in": _A, "f": _A, **_C6,
                            "jcap_tol": _dt, "w": _dt})
    wp.overload(restrict_residual_3d, {"r": _A, "rc": _A})
    wp.overload(restrict_face_3d, {"src": _A, "dst": _A})
    wp.overload(prolongate_add_3d, {"ec": _A, "p": _A})
    wp.overload(extract_pairs_3d, {
        "ch": _A, "cv": _A, "cw": _A, "cp0": _A, "cm0": _A, "cp1": _A,
        "cm1": _A, "cp2": _A, "cm2": _A})


# ─────────────────────────────────────────────────────────────────────────────
#  Variable-coefficient, graph-captured multigrid driver (3-D)
# ─────────────────────────────────────────────────────────────────────────────

def _wdev(t):
    return "cuda:0" if t.device.type == "cuda" else "cpu"


class WarpMG3D:
    """Geometric multigrid (Neumann) for variable BDIM coefficients, captured as
    one CUDA graph (``n_vcycles`` fixed cycles, sync-free).

    Per step: ``solve(f, ch, cv, cw, p0)`` copies the live RHS + face coefficients
    into persistent buffers, replays the captured graph (face restriction down all
    levels + ``n_vcycles`` V-cycles), and copies the solution back.  No host sync.
    """

    def __init__(self, Nx, Ny, Nz, device="cuda:0", dtype=torch.float64,
                 smoother="rbgs", nu1=2, nu2=2, n_vcycles=4, coarse_extra=8,
                 jcap_tol=1e-30, min_dim=4):
        self.device = device
        self.tdt = dtype
        self.wpf = wp.float64 if dtype == torch.float64 else wp.float32
        self.smoother = smoother
        self.nu1, self.nu2, self.n_vcycles = nu1, nu2, n_vcycles
        self.coarse_extra = coarse_extra
        self.jcap = jcap_tol
        self.min_dim = min_dim
        self._wdev = "cuda:0" if (isinstance(device, str) and "cuda" in device) \
            or getattr(device, "type", "") == "cuda" else "cpu"
        self._tdev = device
        self._build(Nx, Ny, Nz)
        self._graph = None

    # ── level hierarchy ──────────────────────────────────────────────────────
    def _build(self, Nx, Ny, Nz):
        o = dict(dtype=self.tdt, device=self._tdev)
        self.levels = []
        shape = (Nx, Ny, Nz)
        while True:
            nx, ny, nz = shape
            p = torch.zeros((nx + 2) * (ny + 2) * (nz + 2), **o)
            f = torch.zeros(nx * ny * nz, **o)
            r = torch.zeros(nx * ny * nz, **o)
            ch = torch.zeros((nx + 1, ny, nz), **o)
            cv = torch.zeros((nx, ny + 1, nz), **o)
            cw = torch.zeros((nx, ny, nz + 1), **o)
            # Contiguous (Nx,Ny,Nz) face-pair buffers (materialised in-graph by
            # extract_pairs_3d — y/z face slices are non-contiguous so a flat
            # view can't alias them).
            pair = [torch.zeros(nx * ny * nz, **o) for _ in range(6)]
            lv = dict(
                n=(nx, ny, nz), p=p, f=f, r=r, ch=ch, cv=cv, cw=cw,
                pw=wp.from_torch(p), fw=wp.from_torch(f), rw=wp.from_torch(r),
                chw=wp.from_torch(ch.reshape(-1)),
                cvw=wp.from_torch(cv.reshape(-1)),
                cww=wp.from_torch(cw.reshape(-1)),
                _pair_t=pair,
                cp0=wp.from_torch(pair[0]), cm0=wp.from_torch(pair[1]),
                cp1=wp.from_torch(pair[2]), cm1=wp.from_torch(pair[3]),
                cp2=wp.from_torch(pair[4]), cm2=wp.from_torch(pair[5]),
            )
            if self.smoother == "jacobi":
                lv["ptmp"] = wp.zeros_like(lv["pw"])
            self.levels.append(lv)
            if (nx % 2 == 0 and ny % 2 == 0 and nz % 2 == 0
                    and min(nx, ny, nz) // 2 >= self.min_dim):
                shape = (nx // 2, ny // 2, nz // 2)
            else:
                break

    def _coef(self, lv):
        return [lv["cp0"], lv["cm0"], lv["cp1"], lv["cm1"], lv["cp2"], lv["cm2"]]

    def _smooth(self, lv, n):
        nx, ny, nz = lv["n"]
        jt = self.wpf(self.jcap)
        if self.smoother == "jacobi":
            src, dst = lv["pw"], lv["ptmp"]
            for _ in range(n):
                wp.launch(jacobi_3d, dim=(nx, ny, nz),
                          inputs=[dst, src, lv["fw"], *self._coef(lv),
                                  nx, ny, nz, jt, self.wpf(0.7)], device=self._wdev)
                src, dst = dst, src
            if n % 2 == 1:
                wp.copy(lv["pw"], src)
        else:
            for _ in range(n):
                for color in (0, 1):
                    wp.launch(rbgs_halfsweep_3d, dim=(nx, ny, nz),
                              inputs=[lv["pw"], lv["fw"], *self._coef(lv),
                                      nx, ny, nz, jt, color], device=self._wdev)

    def _extract(self, lv):
        nx, ny, nz = lv["n"]
        wp.launch(extract_pairs_3d, dim=(nx, ny, nz),
                  inputs=[lv["chw"], lv["cvw"], lv["cww"],
                          lv["cp0"], lv["cm0"], lv["cp1"], lv["cm1"],
                          lv["cp2"], lv["cm2"], nx, ny, nz], device=self._wdev)

    def _restrict_faces(self):
        for l in range(len(self.levels) - 1):
            f, c = self.levels[l], self.levels[l + 1]
            nf = f["n"]; nc = c["n"]
            wp.launch(restrict_face_3d, dim=(nc[0] + 1, nc[1], nc[2]),
                      inputs=[f["chw"], c["chw"], nf[0] + 1, nf[1], nf[2],
                              nc[0] + 1, nc[1], nc[2], 0], device=self._wdev)
            wp.launch(restrict_face_3d, dim=(nc[0], nc[1] + 1, nc[2]),
                      inputs=[f["cvw"], c["cvw"], nf[0], nf[1] + 1, nf[2],
                              nc[0], nc[1] + 1, nc[2], 1], device=self._wdev)
            wp.launch(restrict_face_3d, dim=(nc[0], nc[1], nc[2] + 1),
                      inputs=[f["cww"], c["cww"], nf[0], nf[1], nf[2] + 1,
                              nc[0], nc[1], nc[2] + 1, 2], device=self._wdev)

    def _vcycle(self, lvl):
        lv = self.levels[lvl]
        nx, ny, nz = lv["n"]
        jt = self.wpf(self.jcap)
        if lvl == len(self.levels) - 1:
            self._smooth(lv, self.nu1 + self.nu2 + self.coarse_extra)
            return
        self._smooth(lv, self.nu1)
        wp.launch(mg_residual_3d, dim=(nx, ny, nz),
                  inputs=[lv["pw"], lv["fw"], *self._coef(lv), lv["rw"],
                          nx, ny, nz, jt], device=self._wdev)
        c = self.levels[lvl + 1]
        ncx, ncy, ncz = c["n"]
        wp.launch(restrict_residual_3d, dim=(ncx, ncy, ncz),
                  inputs=[lv["rw"], c["fw"], nx, ny, nz, ncx, ncy, ncz],
                  device=self._wdev)
        c["pw"].zero_()
        self._vcycle(lvl + 1)
        wp.launch(prolongate_add_3d, dim=(nx, ny, nz),
                  inputs=[c["pw"], lv["pw"], ncx, ncy, ncz, nx, ny, nz],
                  device=self._wdev)
        self._smooth(lv, self.nu2)

    def _cycle(self):
        # Coarsen the face arrays down the hierarchy, then materialise the six
        # contiguous face-pair buffers per level, then the fixed V-cycles.
        self._restrict_faces()
        for lv in self.levels:
            self._extract(lv)
        for _ in range(self.n_vcycles):
            self._vcycle(0)

    def capture(self):
        # warm-up (JIT all kernels) before capture.
        self._cycle()
        with wp.ScopedCapture(device=self._wdev) as cap:
            self._cycle()
        self._graph = cap.graph

    def solve(self, f, ch, cv, cw, p0=None):
        """f: (Nx,Ny,Nz) interior RHS; ch/cv/cw: live face coeffs; p0: optional
        padded warm-start.  Returns padded p (the solution)."""
        l0 = self.levels[0]
        l0["f"].copy_(f.reshape(-1))
        l0["ch"].copy_(ch); l0["cv"].copy_(cv); l0["cw"].copy_(cw)
        if p0 is not None:
            l0["p"].copy_(p0.reshape(-1))
        else:
            l0["p"].zero_()
        if self._graph is None:
            self.capture()
        else:
            wp.capture_launch(self._graph)
        nx, ny, nz = l0["n"]
        return l0["p"].view(nx + 2, ny + 2, nz + 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Variable-coefficient, graph-captured multigrid driver (2-D)
# ─────────────────────────────────────────────────────────────────────────────
from lilytorch.warp_poc.warp_multigrid_2d import (  # noqa: E402
    rbgs_halfsweep_2d_f64 as _rbgs_2d,
    mg_residual_2d as _mgres_2d,
    restrict_residual_2d as _rres_2d,
    restrict_face_2d as _rface_2d,
    prolongate_add_2d as _prol_2d,
)


@wp.kernel
def mg_residual_2d_clamped(
    p: wp.array(dtype=wp.float64), f: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    r: wp.array(dtype=wp.float64), Nx: int, Ny: int, jcap_tol: wp.float64,
):
    """2-D residual that CLAMPS ghost cells (homogeneous-Neumann fold) IDENTICALLY
    to the smoother — required for the graphed V-cycle, which never updates the
    ghost layer (the unclamped ``warp_multigrid_2d.mg_residual_2d`` reads the raw,
    stale ghosts → operator/residual mismatch at the boundary → divergence).
    2-D analogue of :func:`mg_residual_3d`."""
    i, j = wp.tid()
    c = i * Ny + j
    si = Ny + 2
    b = (i + 1) * si + (j + 1)
    J = cp0[c] + cm0[c] + cp1[c] + cm1[c]
    if J < jcap_tol and J > -jcap_tol:
        r[c] = wp.float64(0.0)
        return
    pc = p[b]
    pip = pc; pim = pc; pjp = pc; pjm = pc
    if i < Nx - 1: pip = p[b + si]
    if i > 0:      pim = p[b - si]
    if j < Ny - 1: pjp = p[b + 1]
    if j > 0:      pjm = p[b - 1]
    s = cp0[c] * pip + cm0[c] * pim + cp1[c] * pjp + cm1[c] * pjm
    r[c] = f[c] - s + J * pc


def mg_residual_2d_clamped_warp(p, f, cp0, cm0, cp1, cm1, jcap_tol):
    """Host wrapper for :func:`mg_residual_2d_clamped` (f64).  ``p`` padded
    (Nx+2,Ny+2); ``f``/coeffs interior (Nx,Ny).  Returns r (Nx,Ny)."""
    Nx, Ny = f.shape
    wdev = "cuda:0" if f.is_cuda else "cpu"
    r = torch.empty_like(f)
    wp.launch(mg_residual_2d_clamped, dim=(Nx, Ny),
              inputs=[wp.from_torch(p.reshape(-1)), wp.from_torch(f.reshape(-1)),
                      wp.from_torch(cp0.reshape(-1)), wp.from_torch(cm0.reshape(-1)),
                      wp.from_torch(cp1.reshape(-1)), wp.from_torch(cm1.reshape(-1)),
                      wp.from_torch(r.reshape(-1)), int(Nx), int(Ny),
                      wp.float64(jcap_tol)], device=wdev)
    return r


@wp.kernel
def extract_pairs_2d(
    ch: wp.array(dtype=wp.float64), cv: wp.array(dtype=wp.float64),
    cp0: wp.array(dtype=wp.float64), cm0: wp.array(dtype=wp.float64),
    cp1: wp.array(dtype=wp.float64), cm1: wp.array(dtype=wp.float64),
    Nx: int, Ny: int,
):
    """Materialise the four face-pair coefficient arrays into contiguous (Nx,Ny)
    buffers from ch(Nx+1,Ny)/cv(Nx,Ny+1).  The cv y-face slices are
    non-contiguous, so a persistent flat view can't alias them in a captured
    graph (2-D analogue of :func:`extract_pairs_3d`)."""
    i, j = wp.tid()
    c = i * Ny + j
    cp0[c] = ch[(i + 1) * Ny + j]
    cm0[c] = ch[i * Ny + j]
    svi = Ny + 1
    cp1[c] = cv[i * svi + (j + 1)]
    cm1[c] = cv[i * svi + j]


class WarpMG2D:
    """2-D analogue of :class:`WarpMG3D` — an all-Warp, variable-coefficient,
    anisotropic, CUDA-graph-captured geometric multigrid (Neumann), f64 (the 2-D
    eel target precision; the 2-D smoother/transfer kernels are f64).  Per step
    ``solve(f, ch, cv, p0)`` copies the live RHS + face coefficients into the
    persistent level-0 buffers and replays the captured fixed-cycle graph with one
    host launch (no per-cycle ``.item()`` sync)."""

    def __init__(self, Nx, Ny, device="cuda:0", dtype=torch.float64,
                 nu1=2, nu2=2, n_vcycles=4, coarse_extra=8,
                 jcap_tol=1e-30, min_dim=4):
        assert dtype == torch.float64, "WarpMG2D is f64 (2-D MG kernels are f64)"
        self.device = device
        self.tdt = dtype
        self.wpf = wp.float64
        self.nu1, self.nu2, self.n_vcycles = nu1, nu2, n_vcycles
        self.coarse_extra = coarse_extra
        self.jcap = jcap_tol
        self.min_dim = min_dim
        self._wdev = "cuda:0" if (isinstance(device, str) and "cuda" in device) \
            or getattr(device, "type", "") == "cuda" else "cpu"
        self._tdev = device
        self._build(Nx, Ny)
        self._graph = None

    def _build(self, Nx, Ny):
        o = dict(dtype=self.tdt, device=self._tdev)
        self.levels = []
        shape = (Nx, Ny)
        while True:
            nx, ny = shape
            p = torch.zeros((nx + 2) * (ny + 2), **o)
            f = torch.zeros(nx * ny, **o)
            r = torch.zeros(nx * ny, **o)
            ch = torch.zeros((nx + 1, ny), **o)
            cv = torch.zeros((nx, ny + 1), **o)
            pair = [torch.zeros(nx * ny, **o) for _ in range(4)]
            lv = dict(
                n=(nx, ny), p=p, f=f, r=r, ch=ch, cv=cv,
                pw=wp.from_torch(p), fw=wp.from_torch(f), rw=wp.from_torch(r),
                chw=wp.from_torch(ch.reshape(-1)),
                cvw=wp.from_torch(cv.reshape(-1)),
                _pair_t=pair,
                cp0=wp.from_torch(pair[0]), cm0=wp.from_torch(pair[1]),
                cp1=wp.from_torch(pair[2]), cm1=wp.from_torch(pair[3]),
            )
            self.levels.append(lv)
            if (nx % 2 == 0 and ny % 2 == 0
                    and min(nx, ny) // 2 >= self.min_dim):
                shape = (nx // 2, ny // 2)
            else:
                break

    def _coef(self, lv):
        return [lv["cp0"], lv["cm0"], lv["cp1"], lv["cm1"]]

    def _smooth(self, lv, n):
        nx, ny = lv["n"]
        jt = self.wpf(self.jcap)
        for _ in range(n):
            for color in (0, 1):
                wp.launch(_rbgs_2d, dim=(nx, ny),
                          inputs=[lv["pw"], lv["fw"], *self._coef(lv),
                                  nx, ny, jt, color], device=self._wdev)

    def _extract(self, lv):
        nx, ny = lv["n"]
        wp.launch(extract_pairs_2d, dim=(nx, ny),
                  inputs=[lv["chw"], lv["cvw"], lv["cp0"], lv["cm0"],
                          lv["cp1"], lv["cm1"], nx, ny], device=self._wdev)

    def _restrict_faces(self):
        for l in range(len(self.levels) - 1):
            f, c = self.levels[l], self.levels[l + 1]
            nf = f["n"]; nc = c["n"]
            wp.launch(_rface_2d, dim=(nc[0] + 1, nc[1]),
                      inputs=[f["chw"], c["chw"], nf[0] + 1, nf[1],
                              nc[0] + 1, nc[1], 0], device=self._wdev)
            wp.launch(_rface_2d, dim=(nc[0], nc[1] + 1),
                      inputs=[f["cvw"], c["cvw"], nf[0], nf[1] + 1,
                              nc[0], nc[1] + 1, 1], device=self._wdev)

    def _vcycle(self, lvl):
        lv = self.levels[lvl]
        nx, ny = lv["n"]
        jt = self.wpf(self.jcap)
        if lvl == len(self.levels) - 1:
            self._smooth(lv, self.nu1 + self.nu2 + self.coarse_extra)
            return
        self._smooth(lv, self.nu1)
        wp.launch(mg_residual_2d_clamped, dim=(nx, ny),
                  inputs=[lv["pw"], lv["fw"], *self._coef(lv), lv["rw"],
                          nx, ny, jt], device=self._wdev)
        c = self.levels[lvl + 1]
        ncx, ncy = c["n"]
        wp.launch(_rres_2d, dim=(ncx, ncy),
                  inputs=[lv["rw"], c["fw"], nx, ny, ncx, ncy],
                  device=self._wdev)
        c["pw"].zero_()
        self._vcycle(lvl + 1)
        wp.launch(_prol_2d, dim=(nx, ny),
                  inputs=[c["pw"], lv["pw"], ncx, ncy, nx, ny],
                  device=self._wdev)
        self._smooth(lv, self.nu2)

    def _cycle(self):
        self._restrict_faces()
        for lv in self.levels:
            self._extract(lv)
        for _ in range(self.n_vcycles):
            self._vcycle(0)

    def capture(self):
        self._cycle()  # warm-up / JIT
        with wp.ScopedCapture(device=self._wdev) as cap:
            self._cycle()
        self._graph = cap.graph

    def solve(self, f, ch, cv, p0=None):
        """f: (Nx,Ny) interior RHS; ch/cv: live face coeffs; p0: optional padded
        warm-start.  Returns padded p (Nx+2, Ny+2)."""
        l0 = self.levels[0]
        l0["f"].copy_(f.reshape(-1))
        l0["ch"].copy_(ch); l0["cv"].copy_(cv)
        if p0 is not None:
            l0["p"].copy_(p0.reshape(-1))
        else:
            l0["p"].zero_()
        if self._graph is None:
            self.capture()
        else:
            wp.capture_launch(self._graph)
        nx, ny = l0["n"]
        return l0["p"].view(nx + 2, ny + 2)
