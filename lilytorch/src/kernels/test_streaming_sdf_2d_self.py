"""Self-contained parity test for ``streaming_sdf_min_2d_multi``
(and the single-body ``streaming_sdf_min_2d`` op).

Mirrors ``test_streaming_sdf_self.py`` (the 3-D test) with the z-axis
stripped: we exercise the 2-D streaming kernel with a synthetic
multi-body scene (rotated disc SDFs at random positions/velocities)
and compare the five written outputs (``sdf_cc``, ``sdf_u``, ``sdf_v``,
``body_u``, ``body_v``) plus the per-body cached ``sparse_cc_flat``
against a pure-PyTorch reference.

Both interpolation methods are checked:
  * ``interp_method=0`` -- bilinear;
  * ``interp_method=1`` -- biquadratic Lagrange (3x3 stencil with
    bilinear fallback in the boundary layer).

When ``torch.cuda.is_available()``, the CUDA implementation is run and
cross-checked against the CPU implementation as well (PR-2 gate).

Run with::

    python -m lilytorch.src.kernels.test_streaming_sdf_2d_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401  -- registers the namespace
from lilytorch.src.kernels import (
    streaming_sdf_min_2d,
    streaming_sdf_min_2d_multi,
)


def _random_rotation_2d(dtype, device, gen):
    """Random 2x2 rotation R = [[c, -s], [s, c]]."""
    theta = (torch.rand(1, generator=gen) * (2 * math.pi)).item()
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)


def _make_body_2d(seed, Mx, My, *, dtype, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    half = 0.7
    bx = torch.linspace(-half, half, Mx, dtype=dtype, device=device)
    by = torch.linspace(-0.9 * half, 0.9 * half, My, dtype=dtype, device=device)
    Bx, By = torch.meshgrid(bx, by, indexing="ij")
    # Disc-like SDF (radius 0.35).
    F = (torch.sqrt(Bx**2 + By**2) - 0.35).contiguous()
    R = _random_rotation_2d(dtype, device, gen)
    R_T = R.T.contiguous().flatten().tolist()  # 4 floats, row-major of R^T
    bp = ((torch.rand(2, generator=gen, dtype=dtype) - 0.5) * 0.4).tolist()
    cm = bp[:]
    lv = (torch.rand(2, generator=gen, dtype=dtype) - 0.5).tolist()
    omega = float((torch.rand(1, generator=gen, dtype=dtype) - 0.5).item())
    return dict(F=F, bx=bx, by=by, R_T=R_T, bp=bp, cm=cm, lv=lv, omega=omega,
                Mx=Mx, My=My)


def _axis_meta_2d(bd):
    """7-element meta row: bx0, by0, bxL, byL, inv_dx, inv_dy, inv_vol."""
    bx, by = bd["bx"], bd["by"]
    dx = float(bx[1] - bx[0])
    dy = float(by[1] - by[0])
    return [
        float(bx[0]), float(by[0]),
        float(bx[-1]), float(by[-1]),
        1.0 / dx, 1.0 / dy,
        1.0 / (dx * dy),
    ]


def _bilinear_sample_torch_2d(F, bx, by, x, y):
    """Pure-PyTorch reference: bilinear interp on a uniform body grid.

    Mirrors the C++/CUDA ``bilinear_sample_uniform_2d`` line-for-line:
    clamp ``t`` in body-grid coordinates, lower-bracket index, factored
    weights ``(1-frac, frac)`` per axis.
    """
    Mx, My = F.shape
    inv_dx = 1.0 / (bx[1] - bx[0]).item()
    inv_dy = 1.0 / (by[1] - by[0]).item()
    bx0 = bx[0].item()
    by0 = by[0].item()

    tx = torch.clamp((x - bx0) * inv_dx, 0.0, Mx - 1.0)
    ty = torch.clamp((y - by0) * inv_dy, 0.0, My - 1.0)
    ix = torch.clamp(tx.floor().long(), 0, Mx - 2)
    iy = torch.clamp(ty.floor().long(), 0, My - 2)
    fx = tx - ix.to(tx.dtype)
    fy = ty - iy.to(ty.dtype)
    wx0, wx1 = 1.0 - fx, fx
    wy0, wy1 = 1.0 - fy, fy

    f00 = F[ix,     iy    ]
    f01 = F[ix,     iy + 1]
    f10 = F[ix + 1, iy    ]
    f11 = F[ix + 1, iy + 1]
    return wx0 * (wy0 * f00 + wy1 * f01) + wx1 * (wy0 * f10 + wy1 * f11)


def _biquadratic_sample_torch_2d(F, bx, by, x, y):
    """Pure-PyTorch reference: biquadratic Lagrange interp (3x3 stencil).

    Mirrors the C++/CUDA ``biquadratic_sample_uniform_2d``.  Falls back
    to bilinear in cells whose lower stencil neighbour is out of range.
    """
    Mx, My = F.shape
    inv_dx = 1.0 / (bx[1] - bx[0]).item()
    inv_dy = 1.0 / (by[1] - by[0]).item()
    bx0 = bx[0].item()
    by0 = by[0].item()

    tx = torch.clamp((x - bx0) * inv_dx, 0.0, Mx - 1.0)
    ty = torch.clamp((y - by0) * inv_dy, 0.0, My - 1.0)
    ix = torch.clamp(tx.floor().long(), 0, Mx - 2)
    iy = torch.clamp(ty.floor().long(), 0, My - 2)
    fx = tx - ix.to(tx.dtype)
    fy = ty - iy.to(ty.dtype)

    eligible = (ix >= 1) & (iy >= 1) & (Mx >= 3) & (My >= 3)
    bil_val = _bilinear_sample_torch_2d(F, bx, by, x, y)
    if not eligible.any():
        return bil_val

    ixc = torch.clamp(ix, 1, max(Mx - 2, 1))
    iyc = torch.clamp(iy, 1, max(My - 2, 1))
    half = 0.5
    wxm = half * fx * (fx - 1.0)
    wx0 = 1.0 - fx * fx
    wxp = half * fx * (fx + 1.0)
    wym = half * fy * (fy - 1.0)
    wy0_w = 1.0 - fy * fy
    wyp = half * fy * (fy + 1.0)

    out = torch.zeros_like(bil_val)
    for dx_off, wx in enumerate((wxm, wx0, wxp)):
        ixs = ixc + (dx_off - 1)
        col = (
            wym * F[ixs, iyc - 1] +
            wy0_w * F[ixs, iyc] +
            wyp * F[ixs, iyc + 1]
        )
        out = out + wx * col

    return torch.where(eligible, out, bil_val)


_REF_SAMPLER_2D = {
    0: _bilinear_sample_torch_2d,
    1: _biquadratic_sample_torch_2d,
}


def _ref_streaming_sdf_2d(bodies, aabbs, gx, gy, h, *, dtype, device,
                          interp_method=0):
    """Pure-PyTorch reference for streaming_sdf_min_2d_multi outputs."""
    sampler = _REF_SAMPLER_2D[interp_method]
    Nx, Ny = gx.numel(), gy.numel()
    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    sparse_cc_chunks = []
    half_h = 0.5 * h

    for bd, ab in zip(bodies, aabbs):
        i0, i1, j0, j1 = ab
        Ai, Aj = i1 - i0, j1 - j0
        ii = torch.arange(i0, i1, device=device)
        jj = torch.arange(j0, j1, device=device)
        xc = gx[ii].view(Ai, 1).expand(Ai, Aj)
        yc = gy[jj].view(1, Aj).expand(Ai, Aj)

        # R_T is column-major flatten of R^T, i.e. row-major of R^T,
        # which equals row-major of R^T.  Storing as 2x2 reshapes
        # to [[r00, r01], [r10, r11]].
        R_T = torch.tensor(bd["R_T"], dtype=dtype, device=device).view(2, 2)
        bp = torch.tensor(bd["bp"], dtype=dtype, device=device)
        cm = torch.tensor(bd["cm"], dtype=dtype, device=device)
        lv = bd["lv"]
        omega = bd["omega"]

        def world_to_body(xw, yw):
            dxw = xw - bp[0]
            dyw = yw - bp[1]
            bxq = R_T[0, 0] * dxw + R_T[0, 1] * dyw
            byq = R_T[1, 0] * dxw + R_T[1, 1] * dyw
            return bxq, byq

        # cc
        bxq, byq = world_to_body(xc, yc)
        s_cc = sampler(bd["F"], bd["bx"], bd["by"], bxq, byq)
        sparse_cc_chunks.append(s_cc.flatten())
        sub_cc = sdf_cc[i0:i1, j0:j1]
        sdf_cc[i0:i1, j0:j1] = torch.minimum(sub_cc, s_cc)

        # u-face: world (xc - h/2, yc)
        bxq_u, byq_u = world_to_body(xc - half_h, yc)
        s_u = sampler(bd["F"], bd["bx"], bd["by"], bxq_u, byq_u)
        sub_u = sdf_u[i0:i1, j0:j1]
        win = s_u < sub_u
        sdf_u[i0:i1, j0:j1] = torch.where(win, s_u, sub_u)
        bU_val = lv[0] - omega * (yc - cm[1])
        bU[i0:i1, j0:j1] = torch.where(win, bU_val, bU[i0:i1, j0:j1])

        # v-face: world (xc, yc - h/2)
        bxq_v, byq_v = world_to_body(xc, yc - half_h)
        s_v = sampler(bd["F"], bd["bx"], bd["by"], bxq_v, byq_v)
        sub_v = sdf_v[i0:i1, j0:j1]
        win = s_v < sub_v
        sdf_v[i0:i1, j0:j1] = torch.where(win, s_v, sub_v)
        bV_val = lv[1] + omega * (xc - cm[0])
        bV[i0:i1, j0:j1] = torch.where(win, bV_val, bV[i0:i1, j0:j1])

    sparse_cc_flat = torch.cat(sparse_cc_chunks)
    return sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc_flat


def _pack_scene(bodies, aabbs, *, dtype, device):
    """Pack a list of bodies + their AABBs into the multi-body op layout."""
    F_chunks  = [bd["F"].flatten()  for bd in bodies]
    bx_chunks = [bd["bx"]            for bd in bodies]
    by_chunks = [bd["by"]            for bd in bodies]
    F_off = [0]; bx_off = [0]; by_off = [0]; cell_off = [0]
    shapes = []; metas = []; kin = []; lo = []; dim_ = []; max_vol = 0
    for bd, ab in zip(bodies, aabbs):
        F_off.append(F_off[-1]   + bd["F"].numel())
        bx_off.append(bx_off[-1] + bd["bx"].numel())
        by_off.append(by_off[-1] + bd["by"].numel())
        shapes.append([bd["Mx"], bd["My"]])
        metas.append(_axis_meta_2d(bd))
        kin.append(bd["R_T"] + bd["bp"] + bd["cm"] + bd["lv"] + [bd["omega"]])
        i0, i1, j0, j1 = ab
        lo.append([i0, j0])
        dim_.append([i1 - i0, j1 - j0])
        vol = (i1 - i0) * (j1 - j0)
        cell_off.append(cell_off[-1] + vol)
        max_vol = max(max_vol, vol)

    F_flat  = torch.cat(F_chunks ).contiguous().to(device)
    bx_flat = torch.cat(bx_chunks).contiguous().to(device)
    by_flat = torch.cat(by_chunks).contiguous().to(device)
    return dict(
        F_flat=F_flat, bx_flat=bx_flat, by_flat=by_flat,
        F_off=torch.tensor(F_off, dtype=torch.int64, device=device),
        bx_off=torch.tensor(bx_off, dtype=torch.int64, device=device),
        by_off=torch.tensor(by_off, dtype=torch.int64, device=device),
        cell_off=torch.tensor(cell_off, dtype=torch.int64, device=device),
        shapes=torch.tensor(shapes, dtype=torch.int64, device=device),
        metas=torch.tensor(metas, dtype=dtype, device=device),
        kin=torch.tensor(kin, dtype=dtype, device=device),
        lo=torch.tensor(lo, dtype=torch.int64, device=device),
        dim_=torch.tensor(dim_, dtype=torch.int64, device=device),
        max_vol=max_vol,
        cell_off_total=cell_off[-1],
    )


def _run_multi_kernel(packed, gx, gy, h, *, Nx, Ny, dtype, device,
                      interp_method):
    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    sparse_cc = torch.zeros(packed["cell_off_total"], dtype=dtype, device=device)

    streaming_sdf_min_2d_multi(
        packed["F_flat"], packed["F_off"],
        packed["bx_flat"], packed["bx_off"],
        packed["by_flat"], packed["by_off"],
        packed["shapes"], packed["metas"], packed["kin"],
        packed["lo"], packed["dim_"], packed["cell_off"],
        gx, gy, h, packed["max_vol"],
        sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc,
        interp_method,
    )
    return sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc


def _run_single_kernel(bd, ab, gx, gy, h, *, Nx, Ny, dtype, device,
                       interp_method):
    """Drive ``streaming_sdf_min_2d`` (single-body op) on a single body
    so we cover that op too -- it shares the same update-cell helper but
    a different kernel launcher."""
    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    i0, i1, j0, j1 = ab
    sparse_cc = torch.zeros((i1 - i0) * (j1 - j0), dtype=dtype, device=device)

    meta = _axis_meta_2d(bd)
    streaming_sdf_min_2d(
        bd["F"].to(device), bd["bx"].to(device), bd["by"].to(device),
        meta[0], meta[1], meta[2], meta[3],
        meta[4], meta[5], meta[6],
        bd["R_T"], bd["bp"], bd["cm"], bd["lv"], bd["omega"],
        gx, gy, h,
        i0, i1, j0, j1,
        sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc,
        interp_method,
    )
    return sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc


def _cmp(name, kernel, ref, tol):
    diff = (kernel.cpu() - ref.cpu()).abs().max().item()
    norm = ref.cpu().abs().max().item()
    rel = diff / max(norm, 1e-30)
    print(f"  {name:<14}  max |Δ|={diff:.3e}  max |ref|={norm:.3e}  rel={rel:.3e}")
    assert rel < tol, f"{name}: rel={rel:.3e} > tol={tol:.3e}"


def main():
    dtype = torch.float64
    torch.manual_seed(0)

    Nx, Ny = 56, 48
    h = 0.05
    gx_cpu = torch.arange(Nx, dtype=dtype) * h - 1.0
    gy_cpu = torch.arange(Ny, dtype=dtype) * h - 0.9

    B = 3
    bodies = [_make_body_2d(seed=10 + b, Mx=24, My=20, dtype=dtype, device="cpu")
              for b in range(B)]
    aabbs = [(2 + b * 4, 2 + b * 4 + 22, 3 + b * 3, 3 + b * 3 + 20) for b in range(B)]

    # ---- CPU pass: parity vs PyTorch reference (multi-body) ----------
    print("== CPU multi-body parity vs PyTorch reference ==")
    packed_cpu = _pack_scene(bodies, aabbs, dtype=dtype, device="cpu")
    cpu_outs = {}
    for interp_method, label in [(0, "bilinear"), (1, "biquadratic")]:
        sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc = _run_multi_kernel(
            packed_cpu, gx_cpu, gy_cpu, h,
            Nx=Nx, Ny=Ny, dtype=dtype, device="cpu",
            interp_method=interp_method,
        )
        rcc, ru, rv, rbU, rbV, rsparse = _ref_streaming_sdf_2d(
            bodies, aabbs, gx_cpu, gy_cpu, h, dtype=dtype, device="cpu",
            interp_method=interp_method,
        )
        inside_cc = (sdf_cc.abs() < 1.0).sum().item()
        inside_u  = (sdf_u.abs()  < 1.0).sum().item()
        print(f"[{label}] cells inside band: cc={inside_cc}  u={inside_u}")
        assert inside_cc > 50 and inside_u > 50, "scene degenerate"
        _cmp("sdf_cc", sdf_cc, rcc, 1e-12)
        _cmp("sdf_u",  sdf_u,  ru,  1e-12)
        _cmp("sdf_v",  sdf_v,  rv,  1e-12)
        _cmp("body_u", bU,     rbU, 1e-12)
        _cmp("body_v", bV,     rbV, 1e-12)
        _cmp("sparse_cc_flat", sparse_cc, rsparse, 1e-12)
        cpu_outs[interp_method] = (sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc)

    # The two methods must produce different sparse_cc on at least some
    # cells (otherwise the biquadratic path is silently aliasing to the
    # bilinear path).
    diff_pts = (cpu_outs[0][5] - cpu_outs[1][5]).abs()
    n_diff = (diff_pts > 1e-10).sum().item()
    max_diff = diff_pts.max().item()
    print(f"bilinear vs biquadratic: {n_diff} sparse cells differ "
          f"(max |Δ|={max_diff:.3e})")
    assert n_diff > 0, (
        "biquadratic path produced bit-identical output to bilinear -- "
        "it is probably not being dispatched."
    )

    # ---- CPU single-body op covers ``streaming_sdf_min_2d`` ----------
    print("== CPU single-body op vs reference (body 0 only) ==")
    for interp_method, label in [(0, "bilinear"), (1, "biquadratic")]:
        sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc = _run_single_kernel(
            bodies[0], aabbs[0], gx_cpu, gy_cpu, h,
            Nx=Nx, Ny=Ny, dtype=dtype, device="cpu",
            interp_method=interp_method,
        )
        rcc, ru, rv, rbU, rbV, rsparse = _ref_streaming_sdf_2d(
            [bodies[0]], [aabbs[0]], gx_cpu, gy_cpu, h,
            dtype=dtype, device="cpu", interp_method=interp_method,
        )
        print(f"[{label}] single-body checks")
        _cmp("sdf_cc", sdf_cc, rcc, 1e-12)
        _cmp("sdf_u",  sdf_u,  ru,  1e-12)
        _cmp("sdf_v",  sdf_v,  rv,  1e-12)
        _cmp("body_u", bU,     rbU, 1e-12)
        _cmp("body_v", bV,     rbV, 1e-12)
        _cmp("sparse_cc",       sparse_cc, rsparse, 1e-12)

    # ---- CUDA cross-check (PR-2 gate) --------------------------------
    if torch.cuda.is_available():
        print("== CUDA multi-body cross-check vs CPU ==")
        device = "cuda"
        gx_cu = gx_cpu.to(device)
        gy_cu = gy_cpu.to(device)
        bodies_cu = []
        for bd in bodies:
            bd2 = dict(bd)
            bd2["F"]  = bd["F"].to(device)
            bd2["bx"] = bd["bx"].to(device)
            bd2["by"] = bd["by"].to(device)
            bodies_cu.append(bd2)
        packed_cu = _pack_scene(bodies_cu, aabbs, dtype=dtype, device=device)
        for interp_method, label in [(0, "bilinear"), (1, "biquadratic")]:
            sdf_cc, sdf_u, sdf_v, bU, bV, sparse_cc = _run_multi_kernel(
                packed_cu, gx_cu, gy_cu, h,
                Nx=Nx, Ny=Ny, dtype=dtype, device=device,
                interp_method=interp_method,
            )
            torch.cuda.synchronize()
            print(f"[{label}] CUDA-vs-CPU")
            ref = cpu_outs[interp_method]
            _cmp("sdf_cc", sdf_cc, ref[0], 1e-12)
            _cmp("sdf_u",  sdf_u,  ref[1], 1e-12)
            _cmp("sdf_v",  sdf_v,  ref[2], 1e-12)
            _cmp("body_u", bU,     ref[3], 1e-12)
            _cmp("body_v", bV,     ref[4], 1e-12)
            _cmp("sparse_cc_flat", sparse_cc, ref[5], 1e-12)
    else:
        print("CUDA not available -- skipping CUDA cross-check.")

    print("OK: streaming_sdf_min_2d_multi (and single-body) match the "
          "pure-PyTorch reference for both bilinear and biquadratic "
          "interpolation methods.")


if __name__ == "__main__":
    main()
