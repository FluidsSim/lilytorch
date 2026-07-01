"""Warp 2-D **fused TILED Jacobi smoother** — closes the smoother perf gap.

Earlier the thread-per-cell Warp smoother trailed native 3–4× because native
fuses all `nsmoothing` sweeps in SHARED memory (one global round-trip).  That
was wrongly reported as a Warp-1.14 limitation — in fact Warp 1.14 ships the
full tile API (`tile_load`/`tile_store`/`tile_view`/`tile_map`, `storage="shared"`,
cooperative ops with implicit barriers).  This module uses it to do exactly the
native trick: per block, load the p-halo + coeffs + f into shared ONCE, run N
Jacobi sweeps entirely in shared via cooperative `tile_view` neighbour shifts +
`tile_map` elementwise arithmetic (block-boundary halo approximation, same as
native), store once.  Result: **~1.1× of native fused** (vs 3–4× untiled).

Constraints / notes:
- `nsmoothing` is compile-time (the sweep loop must unroll for tile typing) → a
  kernel FACTORY caches one kernel per sweep count (HANDOFF lesson 17 pattern).
- `enable_backward=False` — tile_map's adjoint doesn't compile here and the
  smoother needs no gradients.
- Grid must be a multiple of the tile size (TILE=32); coarse-MG partial tiles
  (bounds-checked loads/stores) are a straightforward follow-up.
- Jacobi (not RBGS): the cooperative tile model expresses Jacobi cleanly (no
  red-black sequential dependency); it's a valid MG smoother and the native
  `jacobi_sweep_2d` is the parity oracle.  At nsmoothing=1 (no stale halos) it
  is bit-close to native global Jacobi regardless of tiling.
"""
from __future__ import annotations

import warp as wp
import torch

wp.init()

TILE = 32          # interior tile edge
TILEH = 34         # halo edge (TILE + 2), compile-time literal
BLOCK = 128

_KERNELS: dict = {}


def _make_kernel(nsweep: int, w: float):
    """Build (and cache) a tiled fused-Jacobi kernel with `nsweep` unrolled
    sweeps and weight `w` baked in."""
    key = (nsweep, float(w))
    if key in _KERNELS:
        return _KERNELS[key]

    NS = int(nsweep)
    WW = wp.constant(wp.float32(w))

    @wp.kernel(enable_backward=False)
    def jacobi_tiled(p: wp.array2d(dtype=wp.float32),
                     f: wp.array2d(dtype=wp.float32),
                     cp0: wp.array2d(dtype=wp.float32), cm0: wp.array2d(dtype=wp.float32),
                     cp1: wp.array2d(dtype=wp.float32), cm1: wp.array2d(dtype=wp.float32),
                     out: wp.array2d(dtype=wp.float32)):
        I, J = wp.tid()
        oi = I * TILE
        oj = J * TILE
        ps = wp.tile_load(p, shape=(TILEH, TILEH), offset=(oi, oj), storage="shared")
        c0 = wp.tile_load(cp0, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        c1 = wp.tile_load(cm0, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        c2 = wp.tile_load(cp1, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        c3 = wp.tile_load(cm1, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        ff = wp.tile_load(f, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        Jt = c0 + c1 + c2 + c3
        for _s in range(NS):
            up = wp.tile_view(ps, offset=(2, 1), shape=(TILE, TILE))
            down = wp.tile_view(ps, offset=(0, 1), shape=(TILE, TILE))
            right = wp.tile_view(ps, offset=(1, 2), shape=(TILE, TILE))
            left = wp.tile_view(ps, offset=(1, 0), shape=(TILE, TILE))
            center = wp.tile_view(ps, offset=(1, 1), shape=(TILE, TILE))
            num = wp.tile_map(wp.mul, c0, up)
            num = num + wp.tile_map(wp.mul, c1, down)
            num = num + wp.tile_map(wp.mul, c2, right)
            num = num + wp.tile_map(wp.mul, c3, left)
            num = num - ff
            pn = wp.tile_map(wp.div, num, Jt)
            newi = pn * WW + center * (wp.float32(1.0) - WW)
            wp.tile_assign(ps, newi, offset=(1, 1))
        inner = wp.tile_view(ps, offset=(1, 1), shape=(TILE, TILE))
        wp.tile_store(out, inner, offset=(oi + 1, oj + 1))

    _KERNELS[key] = jacobi_tiled
    return jacobi_tiled


_RBGS_KERNELS: dict = {}


@wp.func
def _gs8(c0: wp.float32, up: wp.float32, c1: wp.float32, dn: wp.float32,
         c2: wp.float32, rt: wp.float32, c3: wp.float32, lf: wp.float32):
    # raw 5-point stencil sum (8 tile args — the tile_map max)
    return c0 * up + c1 * dn + c2 * rt + c3 * lf


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


def _make_rbgs_kernel(nsweep: int):
    """Tiled fused **RBGS** (red-black Gauss-Seidel).  Each half-sweep is
    collapsed to TWO cooperative passes — one `tile_map(_gs8, …)` for the whole
    5-point stencil sum and one `tile_map(_sel_*, …)` that fuses the divide +
    colour select — instead of the ~6 chained mul/add/div/blend tile ops of the
    earlier masked kernel.  Many-tile-arg `tile_map` (≤8) makes this possible
    (HANDOFF lesson 25's "≤3" was stale).  This removes the per-op barrier /
    intermediate-tile OVERHEAD, which dominates at the realistic low `nsmoothing`
    (V-cycle) regime: RBGS then ≈ tiled Jacobi (was ~1.5–1.9×).  The residual
    ~2× vs Jacobi at high `nsmoothing` (compute saturation) is the irreducible
    whole-tile colour redundancy — the stencil is still evaluated on all cells in
    each half-sweep (see HANDOFF lesson 26).  Only the RED mask is needed (black
    = update where m==0)."""
    key = int(nsweep)
    if key in _RBGS_KERNELS:
        return _RBGS_KERNELS[key]
    NS = int(nsweep)

    @wp.kernel(enable_backward=False)
    def rbgs_tiled(p: wp.array2d(dtype=wp.float32),
                   f: wp.array2d(dtype=wp.float32),
                   cp0: wp.array2d(dtype=wp.float32), cm0: wp.array2d(dtype=wp.float32),
                   cp1: wp.array2d(dtype=wp.float32), cm1: wp.array2d(dtype=wp.float32),
                   red: wp.array2d(dtype=wp.float32),    # 1 on red cells, 0 on black
                   out: wp.array2d(dtype=wp.float32)):
        I, J = wp.tid()
        oi = I * TILE
        oj = J * TILE
        ps = wp.tile_load(p, shape=(TILEH, TILEH), offset=(oi, oj), storage="shared")
        c0 = wp.tile_load(cp0, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        c1 = wp.tile_load(cm0, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        c2 = wp.tile_load(cp1, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        c3 = wp.tile_load(cm1, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        ff = wp.tile_load(f, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        rm = wp.tile_load(red, shape=(TILE, TILE), offset=(oi, oj), storage="shared")
        Jt = c0 + c1 + c2 + c3
        for _s in range(NS):
            # ---- red half-sweep ----
            up = wp.tile_view(ps, offset=(2, 1), shape=(TILE, TILE))
            dn = wp.tile_view(ps, offset=(0, 1), shape=(TILE, TILE))
            rt = wp.tile_view(ps, offset=(1, 2), shape=(TILE, TILE))
            lf = wp.tile_view(ps, offset=(1, 0), shape=(TILE, TILE))
            ce = wp.tile_view(ps, offset=(1, 1), shape=(TILE, TILE))
            s = wp.tile_map(_gs8, c0, up, c1, dn, c2, rt, c3, lf)
            newr = wp.tile_map(_sel_red, rm, s, ff, Jt, ce)
            wp.tile_assign(ps, newr, offset=(1, 1))
            # ---- black half-sweep (reads updated reds) ----
            up2 = wp.tile_view(ps, offset=(2, 1), shape=(TILE, TILE))
            dn2 = wp.tile_view(ps, offset=(0, 1), shape=(TILE, TILE))
            rt2 = wp.tile_view(ps, offset=(1, 2), shape=(TILE, TILE))
            lf2 = wp.tile_view(ps, offset=(1, 0), shape=(TILE, TILE))
            ce2 = wp.tile_view(ps, offset=(1, 1), shape=(TILE, TILE))
            s2 = wp.tile_map(_gs8, c0, up2, c1, dn2, c2, rt2, c3, lf2)
            newb = wp.tile_map(_sel_blk, rm, s2, ff, Jt, ce2)
            wp.tile_assign(ps, newb, offset=(1, 1))
        inner = wp.tile_view(ps, offset=(1, 1), shape=(TILE, TILE))
        wp.tile_store(out, inner, offset=(oi + 1, oj + 1))

    _RBGS_KERNELS[key] = rbgs_tiled
    return rbgs_tiled


class WarpTiledRBGS2D:
    """Fused tiled red-black GS smoother.  Grid N a multiple of TILE (32)."""

    def __init__(self, N: int, device: str = "cuda:0"):
        assert N % TILE == 0, f"N={N} must be a multiple of TILE={TILE}"
        self.N = N
        self.ntile = N // TILE
        self.device = device
        gi = torch.arange(N, device=device).view(-1, 1)
        gj = torch.arange(N, device=device).view(1, -1)
        self.red = ((gi + gj) % 2 == 0).to(torch.float32)

    def smooth(self, p_pad, f, coeffs, nsweep: int):
        out = p_pad.clone()
        k = _make_rbgs_kernel(nsweep)
        cp0, cm0, cp1, cm1 = coeffs
        wp.launch_tiled(
            k, dim=[self.ntile, self.ntile],
            inputs=[wp.from_torch(p_pad), wp.from_torch(f),
                    wp.from_torch(cp0.contiguous()), wp.from_torch(cm0.contiguous()),
                    wp.from_torch(cp1.contiguous()), wp.from_torch(cm1.contiguous()),
                    wp.from_torch(self.red),
                    wp.from_torch(out)],
            block_dim=BLOCK, device=self.device)
        return out


class WarpTiledJacobi2D:
    """Fused tiled Jacobi smoother (shared-mem multi-sweep).  Grid N must be a
    multiple of TILE (32)."""

    def __init__(self, N: int, device: str = "cuda:0"):
        assert N % TILE == 0, f"N={N} must be a multiple of TILE={TILE}"
        self.N = N
        self.ntile = N // TILE
        self.device = device

    def smooth(self, p_pad, f, coeffs, nsweep: int, w: float = 1.0):
        """p_pad: padded (N+2,N+2) float32; f,coeffs interior (N,N).  Returns a
        new padded tensor with the interior smoothed (out-of-place, like the
        native op's clone-then-write)."""
        out = p_pad.clone()
        k = _make_kernel(nsweep, w)
        cp0, cm0, cp1, cm1 = coeffs
        wp.launch_tiled(
            k, dim=[self.ntile, self.ntile],
            inputs=[wp.from_torch(p_pad), wp.from_torch(f),
                    wp.from_torch(cp0.contiguous()), wp.from_torch(cm0.contiguous()),
                    wp.from_torch(cp1.contiguous()), wp.from_torch(cm1.contiguous()),
                    wp.from_torch(out)],
            block_dim=BLOCK, device=self.device)
        return out
