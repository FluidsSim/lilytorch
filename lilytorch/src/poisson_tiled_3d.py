"""Warp 3-D fused TILED smoothers (Jacobi + RBGS) — shared-mem multi-sweep.

3-D analogue of `warp_poisson_tiled_2d.py`.  Per block: load the (TILE+2)^3
p-halo + the six face coeffs + f into shared ONCE, run N sweeps entirely in
shared via `tile_view` 6-neighbour shifts + cooperative `tile_map`, store once.
Confirms the shared-mem fusion that closes the smoother gap works in 3-D too.

TILE=8 → 10^3 halo; shared budget ~ (10^3 + 8*8^3)*4B ≈ 20 KB (p-halo + 6
coeffs + f), within the 48 KB default.  Grid N must be a multiple of TILE.
Jacobi ~parity with native; RBGS is the FUSED colour-select formulation (one
`tile_map` for the 6-term stencil sum split across two passes + one fused
divide+select, HANDOFF lesson 27) — ~1.15× tiled Jacobi at the realistic ns=2
(was ~1.33×), still ~3× faster than the (unfused) native 3-D RBGS.
"""
from __future__ import annotations

import warp as wp
import torch

wp.init()

TILE = 8
TILEH = 10            # TILE + 2 (compile-time literal)
BLOCK = 256

_JAC: dict = {}
_RB: dict = {}


@wp.func
def _gs8(c0: wp.float32, xp: wp.float32, c1: wp.float32, xm: wp.float32,
         c2: wp.float32, yp: wp.float32, c3: wp.float32, ym: wp.float32):
    # first 4 stencil terms (8 tile args — the tile_map max)
    return c0 * xp + c1 * xm + c2 * yp + c3 * ym


@wp.func
def _add2(c4: wp.float32, zp: wp.float32, c5: wp.float32, zm: wp.float32,
          s: wp.float32):
    # fold the remaining 2 (z) terms into the partial sum
    return c4 * zp + c5 * zm + s


@wp.func
def _sel_red(m: wp.float32, s: wp.float32, ff: wp.float32, Jt: wp.float32,
             old: wp.float32):
    # red cells (m==1) -> GS update (s-ff)/Jt ; black -> unchanged
    return m * ((s - ff) / Jt) + (1.0 - m) * old


@wp.func
def _sel_blk(m: wp.float32, s: wp.float32, ff: wp.float32, Jt: wp.float32,
             old: wp.float32):
    # black cells (m==0) -> GS update ; red -> unchanged
    return (1.0 - m) * ((s - ff) / Jt) + m * old


def _make_jacobi_3d(nsweep: int, w: float):
    key = (int(nsweep), float(w))
    if key in _JAC:
        return _JAC[key]
    NS = int(nsweep)
    WW = wp.constant(wp.float32(w))

    @wp.kernel(enable_backward=False)
    def jac3d(p: wp.array3d(dtype=wp.float32), f: wp.array3d(dtype=wp.float32),
              cp0: wp.array3d(dtype=wp.float32), cm0: wp.array3d(dtype=wp.float32),
              cp1: wp.array3d(dtype=wp.float32), cm1: wp.array3d(dtype=wp.float32),
              cp2: wp.array3d(dtype=wp.float32), cm2: wp.array3d(dtype=wp.float32),
              out: wp.array3d(dtype=wp.float32)):
        I, J, K = wp.tid()
        oi = I * TILE; oj = J * TILE; ok = K * TILE
        ps = wp.tile_load(p, shape=(TILEH, TILEH, TILEH), offset=(oi, oj, ok), storage="shared")
        c0 = wp.tile_load(cp0, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c1 = wp.tile_load(cm0, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c2 = wp.tile_load(cp1, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c3 = wp.tile_load(cm1, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c4 = wp.tile_load(cp2, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c5 = wp.tile_load(cm2, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        ff = wp.tile_load(f, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        Jt = c0 + c1 + c2 + c3 + c4 + c5
        for _s in range(NS):
            xp = wp.tile_view(ps, offset=(2, 1, 1), shape=(TILE, TILE, TILE))
            xm = wp.tile_view(ps, offset=(0, 1, 1), shape=(TILE, TILE, TILE))
            yp = wp.tile_view(ps, offset=(1, 2, 1), shape=(TILE, TILE, TILE))
            ym = wp.tile_view(ps, offset=(1, 0, 1), shape=(TILE, TILE, TILE))
            zp = wp.tile_view(ps, offset=(1, 1, 2), shape=(TILE, TILE, TILE))
            zm = wp.tile_view(ps, offset=(1, 1, 0), shape=(TILE, TILE, TILE))
            ce = wp.tile_view(ps, offset=(1, 1, 1), shape=(TILE, TILE, TILE))
            num = wp.tile_map(wp.mul, c0, xp)
            num = num + wp.tile_map(wp.mul, c1, xm)
            num = num + wp.tile_map(wp.mul, c2, yp)
            num = num + wp.tile_map(wp.mul, c3, ym)
            num = num + wp.tile_map(wp.mul, c4, zp)
            num = num + wp.tile_map(wp.mul, c5, zm)
            num = num - ff
            pn = wp.tile_map(wp.div, num, Jt)
            newi = pn * WW + ce * (wp.float32(1.0) - WW)
            wp.tile_assign(ps, newi, offset=(1, 1, 1))
        inner = wp.tile_view(ps, offset=(1, 1, 1), shape=(TILE, TILE, TILE))
        wp.tile_store(out, inner, offset=(oi + 1, oj + 1, ok + 1))

    _JAC[key] = jac3d
    return jac3d


def _make_rbgs_3d(nsweep: int):
    key = int(nsweep)
    if key in _RB:
        return _RB[key]
    NS = int(nsweep)

    @wp.kernel(enable_backward=False)
    def rb3d(p: wp.array3d(dtype=wp.float32), f: wp.array3d(dtype=wp.float32),
             cp0: wp.array3d(dtype=wp.float32), cm0: wp.array3d(dtype=wp.float32),
             cp1: wp.array3d(dtype=wp.float32), cm1: wp.array3d(dtype=wp.float32),
             cp2: wp.array3d(dtype=wp.float32), cm2: wp.array3d(dtype=wp.float32),
             red: wp.array3d(dtype=wp.float32),
             out: wp.array3d(dtype=wp.float32)):
        I, J, K = wp.tid()
        oi = I * TILE; oj = J * TILE; ok = K * TILE
        ps = wp.tile_load(p, shape=(TILEH, TILEH, TILEH), offset=(oi, oj, ok), storage="shared")
        c0 = wp.tile_load(cp0, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c1 = wp.tile_load(cm0, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c2 = wp.tile_load(cp1, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c3 = wp.tile_load(cm1, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c4 = wp.tile_load(cp2, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        c5 = wp.tile_load(cm2, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        ff = wp.tile_load(f, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        rm = wp.tile_load(red, shape=(TILE, TILE, TILE), offset=(oi, oj, ok), storage="shared")
        Jt = c0 + c1 + c2 + c3 + c4 + c5
        for _s in range(NS):
            for half in range(2):
                xp = wp.tile_view(ps, offset=(2, 1, 1), shape=(TILE, TILE, TILE))
                xm = wp.tile_view(ps, offset=(0, 1, 1), shape=(TILE, TILE, TILE))
                yp = wp.tile_view(ps, offset=(1, 2, 1), shape=(TILE, TILE, TILE))
                ym = wp.tile_view(ps, offset=(1, 0, 1), shape=(TILE, TILE, TILE))
                zp = wp.tile_view(ps, offset=(1, 1, 2), shape=(TILE, TILE, TILE))
                zm = wp.tile_view(ps, offset=(1, 1, 0), shape=(TILE, TILE, TILE))
                ce = wp.tile_view(ps, offset=(1, 1, 1), shape=(TILE, TILE, TILE))
                # fused: 2 tile_maps for the 6-term sum, 1 for divide+colour select
                s = wp.tile_map(_gs8, c0, xp, c1, xm, c2, yp, c3, ym)
                s = wp.tile_map(_add2, c4, zp, c5, zm, s)
                # half 0 = red active, half 1 = black active
                if half == 0:
                    newi = wp.tile_map(_sel_red, rm, s, ff, Jt, ce)
                else:
                    newi = wp.tile_map(_sel_blk, rm, s, ff, Jt, ce)
                wp.tile_assign(ps, newi, offset=(1, 1, 1))
        inner = wp.tile_view(ps, offset=(1, 1, 1), shape=(TILE, TILE, TILE))
        wp.tile_store(out, inner, offset=(oi + 1, oj + 1, ok + 1))

    _RB[key] = rb3d
    return rb3d


class WarpTiledSmoother3D:
    """Fused tiled Jacobi/RBGS 3-D smoother.  Grid N a multiple of TILE (8)."""

    def __init__(self, N: int, device: str = "cuda:0"):
        assert N % TILE == 0, f"N={N} must be a multiple of TILE={TILE}"
        self.N = N
        self.nt = N // TILE
        self.device = device
        idx = torch.arange(N, device=device)
        gi = idx.view(-1, 1, 1); gj = idx.view(1, -1, 1); gk = idx.view(1, 1, -1)
        self.red = ((gi + gj + gk) % 2 == 0).to(torch.float32)

    def jacobi(self, p_pad, f, coeffs, nsweep, w=1.0):
        out = p_pad.clone()
        k = _make_jacobi_3d(nsweep, w)
        wp.launch_tiled(k, dim=[self.nt, self.nt, self.nt],
                        inputs=[wp.from_torch(p_pad), wp.from_torch(f),
                                *[wp.from_torch(c.contiguous()) for c in coeffs],
                                wp.from_torch(out)],
                        block_dim=BLOCK, device=self.device)
        return out

    def rbgs(self, p_pad, f, coeffs, nsweep):
        out = p_pad.clone()
        k = _make_rbgs_3d(nsweep)
        wp.launch_tiled(k, dim=[self.nt, self.nt, self.nt],
                        inputs=[wp.from_torch(p_pad), wp.from_torch(f),
                                *[wp.from_torch(c.contiguous()) for c in coeffs],
                                wp.from_torch(self.red),
                                wp.from_torch(out)],
                        block_dim=BLOCK, device=self.device)
        return out
