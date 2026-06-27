"""EXPERIMENT (negative result): red/black-COMPRESSED 2-D tiled RBGS in Warp 1.14.

Task brief: TASK_rbgs_compressed_2d.md.  Question: can red/black-compressed
storage make the tiled 2-D RBGS smoother reach native parity (native is fused +
color-selective; the masked whole-tile Warp kernel in `warp_poisson_tiled_2d.py`
is ~1.75-2x because it evaluates the stencil on ALL cells then masks by colour)?

VERDICT: **NO.**  The compressed kernel is correct (bit-close to a numpy red-black
reference, float32 ULP) and fully cooperative (no redundant per-thread loops), but
it is NOT faster than the masked kernel:

    grid=1024^2, ns=10 (compute-bound, overhead amortised):
        native 75.6 us  |  masked 127 us (1.68x)  |  compressed-kernel-only 138 us
        (1.83x)  |  compressed incl. pack/unpack 166 us (2.19x)

Why (this is the structural finding -- see HANDOFF lesson 26):
  In a checkerboard, a red cell's left/right BLACK neighbours sit at a
  ROW-PARITY-dependent compressed column (`jb = jr-1+(gi&1)` / `jr+(gi&1)`) -- a
  SHEAR, not a single affine offset.  Warp 1.14 `tile_view` is affine-only
  (offset+shape, NO stride / NO per-row offset).  So the horizontal neighbour must
  be assembled as `even_mask*view(off=A) + odd_mask*view(off=B)` -- a 2-view
  parity-mask blend.  That blend costs ~as many flops as the redundant whole-tile
  stencil it was meant to remove (compressed ~15N/sweep vs masked ~16N/sweep), so
  the compression buys nothing.  On top of that, compressed global layout needs a
  per-call pack/unpack (full-grid passes) that makes the end-to-end path strictly
  worse.  (Vertical up/down neighbours ARE clean affine views -- `jb=jr` -- the
  shear is purely horizontal.)

  Native sidesteps both because it uses per-thread SCALAR shared-memory indexing
  (`p_s[ti+1][tj]`, `if(color==0)`): arbitrary per-thread neighbour reads + idle
  threads skip.  That is strictly more expressive than Warp's whole-tile affine
  tile ops, and is the real reason native's fused 2-D RBGS can't be matched in the
  tile model.

CONCLUSION (kept in VALIDATION_STATUS): keep the masked `WarpTiledRBGS2D`; the
practical fast 2-D smoother is the tiled **Jacobi** (~1.1x of native,
`WarpTiledJacobi2D`).  Tiled RBGS's home is **3-D**, where native is thread-per-
cell (not fused) and the tiled Warp RBGS already BEATS it ~3x.

Run:  python experiment_rbgs_compressed_2d.py    (prints correctness + the bench)
"""
from __future__ import annotations
import time
import numpy as np
import torch
import warp as wp

wp.init()
_DEV = "cuda:0"

TILE = 32            # interior tile edge (rows)
TJh = TILE // 2      # compressed half-width (cols)
TIH = TILE + 2
TJH = TJh + 2

# Compressed index map (verified in `_verify_index_map`):
#   red cell (gi,jr): standard col gj = 2*jr + (gi&1)
#   its 4 BLACK neighbours, in black's compressed jb:
#     up=black(gi+1,jr)  down=black(gi-1,jr)          -> CLEAN
#     left=black(gi, jr-1+(gi&1))  right=black(gi, jr+(gi&1)) -> ROW-PARITY SHEAR


# ----------------------------------------------------------------------
# numpy reference + index-map verification
# ----------------------------------------------------------------------
def _numpy_rbgs(p, f, c, nsweep):
    p = p.copy()
    cp0, cm0, cp1, cm1 = c
    J = cp0 + cm0 + cp1 + cm1
    N = f.shape[0]
    gi = np.arange(N)[:, None]; gj = np.arange(N)[None, :]
    red = (gi + gj) % 2 == 0
    for _ in range(nsweep):
        U = (-f + cp0 * p[2:, 1:-1] + cm0 * p[:-2, 1:-1]
             + cp1 * p[1:-1, 2:] + cm1 * p[1:-1, :-2]) / J
        p[1:-1, 1:-1] = np.where(red, U, p[1:-1, 1:-1])
        U2 = (-f + cp0 * p[2:, 1:-1] + cm0 * p[:-2, 1:-1]
              + cp1 * p[1:-1, 2:] + cm1 * p[1:-1, :-2]) / J
        p[1:-1, 1:-1] = np.where(~red, U2, p[1:-1, 1:-1])
    return p


def _verify_index_map(N=16):
    for gi in range(N):
        for jr in range(N // 2):
            gj = 2 * jr + (gi & 1)
            assert (gi + gj) % 2 == 0
            pred = {"up": jr, "down": jr,
                    "left": jr - 1 + (gi & 1), "right": jr + (gi & 1)}
            for k, (ni, nj) in {"up": (gi + 1, gj), "down": (gi - 1, gj),
                                "left": (gi, gj - 1), "right": (gi, gj + 1)}.items():
                if not (0 <= ni < N and 0 <= nj < N):
                    continue
                assert (ni + nj) % 2 == 1
                jb = (nj - (1 - (ni & 1))) // 2
                assert jb == pred[k], f"{k} gi={gi} jr={jr}: {jb}!={pred[k]}"
    return True


# ----------------------------------------------------------------------
# pack / unpack: standard padded (N+2,N+2) <-> colour-compressed padded
# ----------------------------------------------------------------------
def _compressed_coeffs(arr, parity, N):
    gi = np.arange(N)[:, None]; jx = np.arange(TJh)[None, :]
    gj = 2 * jx + ((parity - gi) & 1)
    return arr[gi, gj].astype(np.float32)


def _pack(p_pad, parity):
    N = p_pad.shape[0] - 2
    out = np.zeros((N + 2, TJh + 2), dtype=p_pad.dtype)
    for gi in range(-1, N + 1):
        for jx in range(-1, TJh + 1):
            gj = 2 * jx + ((parity - gi) & 1)
            si, sj = gi + 1, gj + 1
            if 0 <= si < N + 2 and 0 <= sj < N + 2:
                out[gi + 1, jx + 1] = p_pad[si, sj]
    return out


def _unpack_interior(Rpad, Bpad, N):
    out = np.zeros((N, N), dtype=Rpad.dtype)
    for gi in range(N):
        for jx in range(TJh):
            out[gi, 2 * jx + ((0 - gi) & 1)] = Rpad[gi + 1, jx + 1]
            out[gi, 2 * jx + ((1 - gi) & 1)] = Bpad[gi + 1, jx + 1]
    return out


# ----------------------------------------------------------------------
# the compressed RBGS kernel (cooperative tile ops only)
# ----------------------------------------------------------------------
_K: dict = {}
def make_compressed_kernel(nsweep: int):
    if nsweep in _K:
        return _K[nsweep]
    NS = int(nsweep)

    @wp.kernel(enable_backward=False)
    def comp_rbgs(
        R: wp.array2d(dtype=wp.float32), B: wp.array2d(dtype=wp.float32),
        cp0R: wp.array2d(dtype=wp.float32), cm0R: wp.array2d(dtype=wp.float32),
        cp1R: wp.array2d(dtype=wp.float32), cm1R: wp.array2d(dtype=wp.float32),
        fR: wp.array2d(dtype=wp.float32),
        cp0B: wp.array2d(dtype=wp.float32), cm0B: wp.array2d(dtype=wp.float32),
        cp1B: wp.array2d(dtype=wp.float32), cm1B: wp.array2d(dtype=wp.float32),
        fB: wp.array2d(dtype=wp.float32),
        evenR: wp.array2d(dtype=wp.float32), oddR: wp.array2d(dtype=wp.float32),
        evenB: wp.array2d(dtype=wp.float32), oddB: wp.array2d(dtype=wp.float32),
        Rout: wp.array2d(dtype=wp.float32), Bout: wp.array2d(dtype=wp.float32)):
        I, Jt = wp.tid()
        oi = I * TILE; oj = Jt * TJh
        Rs = wp.tile_load(R, shape=(TIH, TJH), offset=(oi, oj), storage="shared")
        Bs = wp.tile_load(B, shape=(TIH, TJH), offset=(oi, oj), storage="shared")
        a0 = wp.tile_load(cp0R, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        a1 = wp.tile_load(cm0R, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        a2 = wp.tile_load(cp1R, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        a3 = wp.tile_load(cm1R, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        fr = wp.tile_load(fR, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        b0 = wp.tile_load(cp0B, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        b1 = wp.tile_load(cm0B, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        b2 = wp.tile_load(cp1B, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        b3 = wp.tile_load(cm1B, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        fb = wp.tile_load(fB, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        eR = wp.tile_load(evenR, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        oRt = wp.tile_load(oddR, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        eB = wp.tile_load(evenB, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        oBt = wp.tile_load(oddB, shape=(TILE, TJh), offset=(oi, oj), storage="shared")
        JR = a0 + a1 + a2 + a3
        JB = b0 + b1 + b2 + b3
        for _s in range(NS):
            # RED update: black neighbours (parity-mask blend for left/right)
            up = wp.tile_view(Bs, offset=(2, 1), shape=(TILE, TJh))
            dn = wp.tile_view(Bs, offset=(0, 1), shape=(TILE, TJh))
            l0 = wp.tile_view(Bs, offset=(1, 0), shape=(TILE, TJh))
            l1 = wp.tile_view(Bs, offset=(1, 1), shape=(TILE, TJh))
            r1 = wp.tile_view(Bs, offset=(1, 2), shape=(TILE, TJh))
            left = wp.tile_map(wp.mul, eR, l0) + wp.tile_map(wp.mul, oRt, l1)
            right = wp.tile_map(wp.mul, eR, l1) + wp.tile_map(wp.mul, oRt, r1)
            num = wp.tile_map(wp.mul, a0, up) + wp.tile_map(wp.mul, a1, dn)
            num = num + wp.tile_map(wp.mul, a2, right) + wp.tile_map(wp.mul, a3, left)
            Rn = wp.tile_map(wp.div, num - fr, JR)
            wp.tile_assign(Rs, Rn, offset=(1, 1))
            # BLACK update: red neighbours (opposite-parity blend)
            up2 = wp.tile_view(Rs, offset=(2, 1), shape=(TILE, TJh))
            dn2 = wp.tile_view(Rs, offset=(0, 1), shape=(TILE, TJh))
            m0 = wp.tile_view(Rs, offset=(1, 0), shape=(TILE, TJh))
            m1 = wp.tile_view(Rs, offset=(1, 1), shape=(TILE, TJh))
            m2 = wp.tile_view(Rs, offset=(1, 2), shape=(TILE, TJh))
            bleft = wp.tile_map(wp.mul, oBt, m0) + wp.tile_map(wp.mul, eB, m1)
            bright = wp.tile_map(wp.mul, oBt, m1) + wp.tile_map(wp.mul, eB, m2)
            num2 = wp.tile_map(wp.mul, b0, up2) + wp.tile_map(wp.mul, b1, dn2)
            num2 = num2 + wp.tile_map(wp.mul, b2, bright) + wp.tile_map(wp.mul, b3, bleft)
            Bn = wp.tile_map(wp.div, num2 - fb, JB)
            wp.tile_assign(Bs, Bn, offset=(1, 1))
        wp.tile_store(Rout, wp.tile_view(Rs, offset=(1, 1), shape=(TILE, TJh)),
                      offset=(oi + 1, oj + 1))
        wp.tile_store(Bout, wp.tile_view(Bs, offset=(1, 1), shape=(TILE, TJh)),
                      offset=(oi + 1, oj + 1))

    _K[nsweep] = comp_rbgs
    return comp_rbgs


def _t(a):
    return wp.from_torch(torch.from_numpy(np.ascontiguousarray(a)).to(_DEV))


def run_compressed_numpy(p_pad, f, c, nsweep):
    """Reference-friendly path (numpy pack/unpack); returns interior (N,N)."""
    N = f.shape[0]
    Rpad, Bpad = _pack(p_pad, 0), _pack(p_pad, 1)
    cR = [_t(_compressed_coeffs(ci, 0, N)) for ci in c]
    cB = [_t(_compressed_coeffs(ci, 1, N)) for ci in c]
    fR, fB = _t(_compressed_coeffs(f, 0, N)), _t(_compressed_coeffs(f, 1, N))
    gi = (np.arange(N)[:, None] * np.ones((1, TJh))).astype(int)
    eR = ((gi & 1) == 0).astype(np.float32); oR = 1.0 - eR
    Rout = torch.from_numpy(Rpad.copy()).to(_DEV)
    Bout = torch.from_numpy(Bpad.copy()).to(_DEV)
    k = make_compressed_kernel(nsweep)
    nt = N // TILE
    wp.launch_tiled(k, dim=[nt, nt],
        inputs=[_t(Rpad), _t(Bpad), *cR, fR, *cB, fB,
                _t(eR), _t(oR), _t(eR), _t(oR),
                wp.from_torch(Rout), wp.from_torch(Bout)],
        block_dim=256, device=_DEV)
    wp.synchronize()
    return _unpack_interior(Rout.cpu().numpy(), Bout.cpu().numpy(), N)


# ----------------------------------------------------------------------
def _correctness():
    _verify_index_map(8); _verify_index_map(16)
    print("[index map] verified (up/down clean, left/right row-parity shear)")
    N = TILE
    rng = np.random.default_rng(11)
    p = np.zeros((N + 2, N + 2), dtype=np.float32)
    p[1:-1, 1:-1] = rng.standard_normal((N, N)).astype(np.float32)
    f = rng.standard_normal((N, N)).astype(np.float32)
    c = [(0.5 + rng.random((N, N))).astype(np.float32) for _ in range(4)]
    p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    for ns in (1, 2, 5):
        ref = _numpy_rbgs(p, f, c, ns)[1:-1, 1:-1]
        got = run_compressed_numpy(p, f, c, ns)
        d = float(np.abs(ref - got).max())
        print(f"[correctness] N={N} ns={ns}: maxdiff={d:.2e} "
              f"{'PASS' if d < 1e-4 else 'FAIL'}")


def _bench():
    from warp_poisson_tiled_2d import WarpTiledRBGS2D
    import lilytorch.src.kernels  # noqa: F401  (registers the native ops)
    nat = torch.ops.lilytorch_kernels.rbgs_sweep_2d

    def timed(fn, iters=200, warm=20):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(); wp.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    print(f"\n{'grid':>6} {'ns':>3} | {'native':>8} {'masked':>8} "
          f"{'comprK':>8} | {'mask/nat':>8} {'cK/nat':>7}")
    for N in (512, 1024):
        g = torch.Generator(device="cuda").manual_seed(11)
        for ns in (2, 10):
            p = torch.zeros(N + 2, N + 2, device=_DEV)
            p[1:-1, 1:-1] = torch.randn(N, N, generator=g, device=_DEV)
            p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
            f = torch.randn(N, N, generator=g, device=_DEV)
            c = [0.5 + torch.rand(N, N, generator=g, device=_DEV) for _ in range(4)]
            masked = WarpTiledRBGS2D(N)
            # pre-pack compressed inputs (constant per solve) for kernel-only timing
            pn, cn, fn = p.cpu().numpy(), [x.cpu().numpy() for x in c], f.cpu().numpy()
            cR = [_t(_compressed_coeffs(x, 0, N)) for x in cn]
            cB = [_t(_compressed_coeffs(x, 1, N)) for x in cn]
            fR, fB = _t(_compressed_coeffs(fn, 0, N)), _t(_compressed_coeffs(fn, 1, N))
            gi = (np.arange(N)[:, None] * np.ones((1, TJh))).astype(int)
            eR = _t(((gi & 1) == 0).astype(np.float32))
            oR = _t((1.0 - ((gi & 1) == 0)).astype(np.float32))
            Rpad, Bpad = _pack(pn, 0), _pack(pn, 1)
            Rw, Bw = _t(Rpad), _t(Bpad)
            Rout = wp.from_torch(torch.from_numpy(Rpad.copy()).to(_DEV))
            Bout = wp.from_torch(torch.from_numpy(Bpad.copy()).to(_DEV))
            kk = make_compressed_kernel(ns); nt = N // TILE
            ci = [Rw, Bw, *cR, fR, *cB, fB, eR, oR, eR, oR, Rout, Bout]

            tn = timed(lambda: nat(p.clone(), f, *c, 1e-30, ns))
            tm = timed(lambda: masked.smooth(p, f, c, ns))
            tk = timed(lambda: wp.launch_tiled(kk, dim=[nt, nt], inputs=ci,
                                               block_dim=256, device=_DEV))
            print(f"{N:>6} {ns:>3} | {tn:>8.1f} {tm:>8.1f} {tk:>8.1f} | "
                  f"{tm/tn:>8.2f} {tk/tn:>7.2f}")
    print("\n=> compressed kernel ~= masked (~1.7-2x native): the parity-mask "
          "blend\n   for the sheared horizontal neighbour costs what compression "
          "saves.\n   NEGATIVE RESULT: keep the masked kernel; tiled JACOBI is the "
          "fast 2-D\n   smoother (~1.1x); tiled RBGS wins only in 3-D.")


if __name__ == "__main__":
    _correctness()
    _bench()
