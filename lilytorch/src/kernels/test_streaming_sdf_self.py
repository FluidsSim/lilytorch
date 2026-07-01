"""Self-contained CPU parity test for ``streaming_sdf_min_3d_multi``.

The streaming SDF kernel doesn't yet have a CPU-only parity test.  A
companion test (``test_bdim_forces_self.py``) exercises the cached-cc
forces output, which depends on ``sparse_cc_flat`` written by the
streaming kernel — so that test is a strong indirect check of the
*sparse cc* output.  But it does NOT verify the union-min ``sdf_cc`` /
``sdf_u`` / ``sdf_v`` / ``sdf_w`` outputs nor the per-face body
velocities ``body_u`` / ``body_v`` / ``body_w``.

This test exercises the streaming kernel with a synthetic 3-body scene
(rotated sphere SDFs at random positions/velocities) and compares the
seven written outputs against a pure-PyTorch reference.  It is meant
to catch regressions when the kernel is refactored for performance.

Run with::

    python -m lilytorch.src.kernels.test_streaming_sdf_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers the namespace
from lilytorch.src.kernels import streaming_sdf_min_3d_multi


def _random_rotation(dtype, device, gen):
    q = torch.randn(4, generator=gen)
    q = (q / q.norm()).tolist()
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=dtype, device=device)


def _make_body(seed, Mx, My, Mz, *, dtype, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    half = 0.7
    bx = torch.linspace(-half, half, Mx, dtype=dtype, device=device)
    by = torch.linspace(-0.9*half, 0.9*half, My, dtype=dtype, device=device)
    bz = torch.linspace(-0.7*half, 0.7*half, Mz, dtype=dtype, device=device)
    Bx, By, Bz = torch.meshgrid(bx, by, bz, indexing="ij")
    F = (torch.sqrt(Bx**2 + By**2 + Bz**2) - 0.35).contiguous()
    R = _random_rotation(dtype, device, gen)
    R_T = R.T.contiguous().flatten().tolist()
    bp = ((torch.rand(3, generator=gen, dtype=dtype) - 0.5) * 0.4).tolist()
    cm = bp[:]
    lv = (torch.rand(3, generator=gen, dtype=dtype) - 0.5).tolist()
    av = (torch.rand(3, generator=gen, dtype=dtype) - 0.5).tolist()
    return dict(F=F, bx=bx, by=by, bz=bz, R_T=R_T, bp=bp, cm=cm, lv=lv, av=av,
                Mx=Mx, My=My, Mz=Mz)


def _axis_meta(bd):
    bx, by, bz = bd["bx"], bd["by"], bd["bz"]
    return [
        float(bx[0]), float(by[0]), float(bz[0]),
        float(bx[-1]), float(by[-1]), float(bz[-1]),
        1.0/float(bx[1]-bx[0]), 1.0/float(by[1]-by[0]), 1.0/float(bz[1]-bz[0]),
        1.0/float((bx[1]-bx[0])*(by[1]-by[0])*(bz[1]-bz[0])),
    ]


def _trilinear_sample_border_torch(F, bx, by, bz, x, y, z):
    """Pure-PyTorch reference: trilinear interp with border clamp.

    ``x/y/z`` may be tensors of arbitrary shape.  Returns a tensor of the
    same broadcast shape.
    """
    Mx, My, Mz = F.shape
    x = torch.clamp(x, bx[0].item(), bx[-1].item())
    y = torch.clamp(y, by[0].item(), by[-1].item())
    z = torch.clamp(z, bz[0].item(), bz[-1].item())
    inv_dx = 1.0 / (bx[1] - bx[0]).item()
    inv_dy = 1.0 / (by[1] - by[0]).item()
    inv_dz = 1.0 / (bz[1] - bz[0]).item()
    inv_vol = inv_dx * inv_dy * inv_dz
    ix = torch.clamp(((x - bx[0]) * inv_dx).floor().long(), 0, Mx-2)
    iy = torch.clamp(((y - by[0]) * inv_dy).floor().long(), 0, My-2)
    iz = torch.clamp(((z - bz[0]) * inv_dz).floor().long(), 0, Mz-2)
    ixp, iyp, izp = ix+1, iy+1, iz+1
    bxix  = bx[ix];  bxixp = bx[ixp]
    byiy  = by[iy];  byiyp = by[iyp]
    bziz  = bz[iz];  bzizp = bz[izp]
    wx0, wx1 = bxixp - x, x - bxix
    wy0, wy1 = byiyp - y, y - byiy
    wz0, wz1 = bzizp - z, z - bziz
    f000 = F[ix,  iy,  iz ]
    f001 = F[ix,  iy,  izp]
    f010 = F[ix,  iyp, iz ]
    f011 = F[ix,  iyp, izp]
    f100 = F[ixp, iy,  iz ]
    f101 = F[ixp, iy,  izp]
    f110 = F[ixp, iyp, iz ]
    f111 = F[ixp, iyp, izp]
    return (
        wx0*wy0*wz0*f000 + wx0*wy0*wz1*f001 +
        wx0*wy1*wz0*f010 + wx0*wy1*wz1*f011 +
        wx1*wy0*wz0*f100 + wx1*wy0*wz1*f101 +
        wx1*wy1*wz0*f110 + wx1*wy1*wz1*f111
    ) * inv_vol


def _triquadratic_sample_torch(F, bx, by, bz, x, y, z):
    """Pure-PyTorch reference: triquadratic Lagrange interp with the
    same lower-bracketing convention as the streaming kernel.

    Mirrors the C++/CUDA ``triquadratic_sample_uniform`` line-for-line:
    a 3x3x3 stencil ``[ix-1, ix, ix+1]^3`` with Lagrange weights, with
    fallback to trilinear in any cell whose lower stencil neighbour is
    out of range.
    """
    Mx, My, Mz = F.shape
    inv_dx = 1.0 / (bx[1] - bx[0]).item()
    inv_dy = 1.0 / (by[1] - by[0]).item()
    inv_dz = 1.0 / (bz[1] - bz[0]).item()
    bx0, by0, bz0 = bx[0].item(), by[0].item(), bz[0].item()

    tx = torch.clamp((x - bx0) * inv_dx, 0.0, Mx - 1.0)
    ty = torch.clamp((y - by0) * inv_dy, 0.0, My - 1.0)
    tz = torch.clamp((z - bz0) * inv_dz, 0.0, Mz - 1.0)
    ix = torch.clamp(tx.floor().long(), 0, Mx - 2)
    iy = torch.clamp(ty.floor().long(), 0, My - 2)
    iz = torch.clamp(tz.floor().long(), 0, Mz - 2)

    fx = tx - ix.to(tx.dtype)
    fy = ty - iy.to(ty.dtype)
    fz = tz - iz.to(tz.dtype)

    # Triquadratic-eligible mask: all axes have ix >= 1 and grid M >= 3.
    eligible = (ix >= 1) & (iy >= 1) & (iz >= 1) \
               & (Mx >= 3) & (My >= 3) & (Mz >= 3)

    # Trilinear value for the fallback / non-eligible cells.
    tri_val = _trilinear_sample_border_torch(F, bx, by, bz, x, y, z)

    if not eligible.any():
        return tri_val

    # Triquadratic value (computed everywhere; selected on eligibility).
    # Clamp ix,iy,iz to [1, M-2] so [ix-1, ix+1] is in-range; for
    # ineligible cells we'll overwrite with the trilinear value anyway.
    ixc = torch.clamp(ix, 1, max(Mx - 2, 1))
    iyc = torch.clamp(iy, 1, max(My - 2, 1))
    izc = torch.clamp(iz, 1, max(Mz - 2, 1))
    half = 0.5
    wxm = half * fx * (fx - 1.0)
    wx0 = 1.0 - fx * fx
    wxp = half * fx * (fx + 1.0)
    wym = half * fy * (fy - 1.0)
    wy0 = 1.0 - fy * fy
    wyp = half * fy * (fy + 1.0)
    wzm = half * fz * (fz - 1.0)
    wz0_w = 1.0 - fz * fz
    wzp = half * fz * (fz + 1.0)

    out = torch.zeros_like(tri_val)
    for dx_off, wx in enumerate((wxm, wx0, wxp)):
        ixs = ixc + (dx_off - 1)
        plane = torch.zeros_like(tri_val)
        for dy_off, wy in enumerate((wym, wy0, wyp)):
            iys = iyc + (dy_off - 1)
            row = (
                wzm * F[ixs, iys, izc - 1] +
                wz0_w * F[ixs, iys, izc] +
                wzp * F[ixs, iys, izc + 1]
            )
            plane = plane + wy * row
        out = out + wx * plane

    return torch.where(eligible, out, tri_val)


_REF_SAMPLER = {0: _trilinear_sample_border_torch,
                1: _triquadratic_sample_torch}


def _ref_streaming_sdf(bodies, aabbs, gx, gy, gz, h, *, dtype, device,
                       interp_method=0):
    """Pure-PyTorch reference for streaming_sdf_min_3d_multi outputs."""
    sampler = _REF_SAMPLER[interp_method]
    Nx, Ny, Nz = gx.numel(), gy.numel(), gz.numel()
    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_w  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    bW = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    sparse_cc_chunks = []
    half_h = 0.5 * h

    for bd, ab in zip(bodies, aabbs):
        i0, i1, j0, j1, k0, k1 = ab
        Ai, Aj, Ak = i1-i0, j1-j0, k1-k0
        ii = torch.arange(i0, i1, device=device)
        jj = torch.arange(j0, j1, device=device)
        kk = torch.arange(k0, k1, device=device)
        xc = gx[ii].view(Ai,1,1).expand(Ai,Aj,Ak)
        yc = gy[jj].view(1,Aj,1).expand(Ai,Aj,Ak)
        zc = gz[kk].view(1,1,Ak).expand(Ai,Aj,Ak)

        R_T = torch.tensor(bd["R_T"], dtype=dtype, device=device).view(3,3)
        bp = torch.tensor(bd["bp"], dtype=dtype, device=device)
        cm = torch.tensor(bd["cm"], dtype=dtype, device=device)
        lv = bd["lv"]; av = bd["av"]

        def world_to_body(xw, yw, zw):
            dx = xw - bp[0]; dy = yw - bp[1]; dz = zw - bp[2]
            bxq = R_T[0,0]*dx + R_T[0,1]*dy + R_T[0,2]*dz
            byq = R_T[1,0]*dx + R_T[1,1]*dy + R_T[1,2]*dz
            bzq = R_T[2,0]*dx + R_T[2,1]*dy + R_T[2,2]*dz
            return bxq, byq, bzq

        # cc
        bxq, byq, bzq = world_to_body(xc, yc, zc)
        s_cc = sampler(bd["F"], bd["bx"], bd["by"], bd["bz"],
                                              bxq, byq, bzq)
        sparse_cc_chunks.append(s_cc.flatten())
        sub_cc = sdf_cc[i0:i1, j0:j1, k0:k1]
        sdf_cc[i0:i1, j0:j1, k0:k1] = torch.minimum(sub_cc, s_cc)

        # u-face
        bxq_u, byq_u, bzq_u = world_to_body(xc - half_h, yc, zc)
        s_u = sampler(bd["F"], bd["bx"], bd["by"], bd["bz"],
                                             bxq_u, byq_u, bzq_u)
        sub_u = sdf_u[i0:i1, j0:j1, k0:k1]
        win = s_u < sub_u
        sdf_u[i0:i1, j0:j1, k0:k1] = torch.where(win, s_u, sub_u)
        bU_val = lv[0] + av[1]*(zc - cm[2]) - av[2]*(yc - cm[1])
        bU[i0:i1, j0:j1, k0:k1] = torch.where(win, bU_val, bU[i0:i1, j0:j1, k0:k1])

        # v-face
        bxq_v, byq_v, bzq_v = world_to_body(xc, yc - half_h, zc)
        s_v = sampler(bd["F"], bd["bx"], bd["by"], bd["bz"],
                                             bxq_v, byq_v, bzq_v)
        sub_v = sdf_v[i0:i1, j0:j1, k0:k1]
        win = s_v < sub_v
        sdf_v[i0:i1, j0:j1, k0:k1] = torch.where(win, s_v, sub_v)
        bV_val = lv[1] + av[2]*(xc - cm[0]) - av[0]*(zc - cm[2])
        bV[i0:i1, j0:j1, k0:k1] = torch.where(win, bV_val, bV[i0:i1, j0:j1, k0:k1])

        # w-face
        bxq_w, byq_w, bzq_w = world_to_body(xc, yc, zc - half_h)
        s_w = sampler(bd["F"], bd["bx"], bd["by"], bd["bz"],
                                             bxq_w, byq_w, bzq_w)
        sub_w = sdf_w[i0:i1, j0:j1, k0:k1]
        win = s_w < sub_w
        sdf_w[i0:i1, j0:j1, k0:k1] = torch.where(win, s_w, sub_w)
        bW_val = lv[2] + av[0]*(yc - cm[1]) - av[1]*(xc - cm[0])
        bW[i0:i1, j0:j1, k0:k1] = torch.where(win, bW_val, bW[i0:i1, j0:j1, k0:k1])

    sparse_cc_flat = torch.cat(sparse_cc_chunks)
    return sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW, sparse_cc_flat


def main():
    device = "cpu"
    dtype = torch.float64
    torch.manual_seed(0)

    Nx, Ny, Nz = 56, 48, 40
    h = 0.05
    gx = torch.arange(Nx, dtype=dtype) * h - 1.0
    gy = torch.arange(Ny, dtype=dtype) * h - 0.9
    gz = torch.arange(Nz, dtype=dtype) * h - 0.8

    B = 3
    bodies = [_make_body(seed=10+b, Mx=24, My=20, Mz=18, dtype=dtype, device=device)
              for b in range(B)]
    aabbs = [(2+b*4, 2+b*4+22, 3+b*3, 3+b*3+20, 2+b*2, 2+b*2+16) for b in range(B)]

    # ---- pack ----
    F_chunks  = [bd["F"].flatten()  for bd in bodies]
    bx_chunks = [bd["bx"]            for bd in bodies]
    by_chunks = [bd["by"]            for bd in bodies]
    bz_chunks = [bd["bz"]            for bd in bodies]
    F_off = [0]; bx_off = [0]; by_off = [0]; bz_off = [0]; cell_off = [0]
    shapes = []; metas = []; kin = []; lo = []; dim_ = []; max_vol = 0
    for bd, ab in zip(bodies, aabbs):
        F_off.append(F_off[-1]   + bd["F"].numel())
        bx_off.append(bx_off[-1] + bd["bx"].numel())
        by_off.append(by_off[-1] + bd["by"].numel())
        bz_off.append(bz_off[-1] + bd["bz"].numel())
        shapes.append([bd["Mx"], bd["My"], bd["Mz"]])
        metas.append(_axis_meta(bd))
        kin.append(bd["R_T"] + bd["bp"] + bd["cm"] + bd["lv"] + bd["av"])
        i0, i1, j0, j1, k0, k1 = ab
        lo.append([i0, j0, k0])
        dim_.append([i1-i0, j1-j0, k1-k0])
        vol = (i1-i0)*(j1-j0)*(k1-k0)
        cell_off.append(cell_off[-1] + vol)
        max_vol = max(max_vol, vol)

    F_flat  = torch.cat(F_chunks ).contiguous()
    bx_flat = torch.cat(bx_chunks).contiguous()
    by_flat = torch.cat(by_chunks).contiguous()
    bz_flat = torch.cat(bz_chunks).contiguous()
    F_off_t  = torch.tensor(F_off,  dtype=torch.int64)
    bx_off_t = torch.tensor(bx_off, dtype=torch.int64)
    by_off_t = torch.tensor(by_off, dtype=torch.int64)
    bz_off_t = torch.tensor(bz_off, dtype=torch.int64)
    cell_off_t = torch.tensor(cell_off, dtype=torch.int64)
    shapes_t = torch.tensor(shapes, dtype=torch.int64)
    metas_t  = torch.tensor(metas,  dtype=dtype)
    kin_t    = torch.tensor(kin,    dtype=dtype)
    lo_t     = torch.tensor(lo,     dtype=torch.int64)
    dim_t    = torch.tensor(dim_,   dtype=torch.int64)

    FAR = 1e4

    def _run_one(interp_method, label):
        sdf_cc = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        sdf_u  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        sdf_v  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        sdf_w  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        bU = torch.zeros((Nx, Ny, Nz), dtype=dtype)
        bV = torch.zeros((Nx, Ny, Nz), dtype=dtype)
        bW = torch.zeros((Nx, Ny, Nz), dtype=dtype)
        sparse_cc = torch.zeros(cell_off[-1], dtype=dtype)

        streaming_sdf_min_3d_multi(
            F_flat, F_off_t, bx_flat, bx_off_t, by_flat, by_off_t, bz_flat, bz_off_t,
            shapes_t, metas_t, kin_t, lo_t, dim_t, cell_off_t,
            gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW, sparse_cc,
            interp_method,
        )

        rsdf_cc, rsdf_u, rsdf_v, rsdf_w, rbU, rbV, rbW, rsparse = _ref_streaming_sdf(
            bodies, aabbs, gx, gy, gz, h, dtype=dtype, device=device,
            interp_method=interp_method,
        )

        # Sanity: outputs are non-degenerate (some cells should be inside the band).
        inside_cc = (sdf_cc.abs() < 1.0).sum().item()
        inside_u  = (sdf_u.abs()  < 1.0).sum().item()
        print(f"[{label}] cells inside band: cc={inside_cc}  u={inside_u}")
        assert inside_cc > 100 and inside_u > 100, "scene degenerate"

        def cmp(name, kernel, ref, tol):
            diff = (kernel - ref).abs().max().item()
            norm = ref.abs().max().item()
            rel = diff / max(norm, 1e-30)
            print(f"  {name:<14}  max |Δ|={diff:.3e}  max |ref|={norm:.3e}  rel={rel:.3e}")
            assert rel < tol, f"{name}: rel={rel:.3e} > tol={tol:.3e}"

        cmp("sdf_cc",         sdf_cc, rsdf_cc, 1e-12)
        cmp("sdf_u",          sdf_u,  rsdf_u,  1e-12)
        cmp("sdf_v",          sdf_v,  rsdf_v,  1e-12)
        cmp("sdf_w",          sdf_w,  rsdf_w,  1e-12)
        cmp("body_u",         bU,     rbU,     1e-12)
        cmp("body_v",         bV,     rbV,     1e-12)
        cmp("body_w",         bW,     rbW,     1e-12)
        cmp("sparse_cc_flat", sparse_cc, rsparse, 1e-12)
        return sdf_cc, sparse_cc

    cc_tri,    sparse_tri    = _run_one(0, "trilinear")
    cc_triqq,  sparse_triqq  = _run_one(1, "triquadratic")

    # The two methods must give different sparse_cc on at least some
    # cells (otherwise the triquadratic path is silently aliasing to the
    # trilinear path).
    diff_pts = (sparse_tri - sparse_triqq).abs()
    n_diff = (diff_pts > 1e-10).sum().item()
    max_diff = diff_pts.max().item()
    print(f"trilinear vs triquadratic: {n_diff} sparse cells differ "
          f"(max |Δ|={max_diff:.3e})")
    assert n_diff > 0, (
        "triquadratic path produced bit-identical output to trilinear -- "
        "it is probably not being dispatched."
    )
    print("OK: streaming_sdf_min_3d_multi matches the pure-PyTorch reference "
          "for both trilinear and triquadratic interpolation methods.")


if __name__ == "__main__":
    main()
