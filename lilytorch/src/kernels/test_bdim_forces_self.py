"""Self-contained correctness test for the refactored `bdim_forces_3d_multi`.

Implements the high-priority TODO item that asks to avoid recomputing the
per-body cell-centred SDF inside the forces kernel. The lilytorch op now
reads the cached cc-SDF from `sparse_cc_flat` (populated by
`streaming_sdf_min_3d_multi`) instead of re-sampling the body SDF grid.

This test:

  1. Builds a small synthetic multi-body setup (random rotated AABB sphere
     SDFs, random kinematics, random stress / pressure-force fields).
  2. Runs `streaming_sdf_min_3d_multi` to obtain the union-min SDF and
     `sparse_cc_flat`.
  3. Runs the new `bdim_forces_3d_multi` (cache-based).
  4. Computes a reference (B, 12) force / torque tensor in pure PyTorch
     using `sparse_cc_flat` and the same δ-kernel.
  5. Asserts that the two match within tight numerical tolerance.

Runs on CPU (works even without CUDA). Self-contained: no FARMS or solver
dependencies. Run with:

    python -m lilytorch.src.kernels.test_bdim_forces_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers the lilytorch_kernels namespace
from lilytorch.src.kernels import (
    streaming_sdf_min_3d_multi,
    bdim_forces_3d_multi,
)


def _random_rotation(dtype, device):
    q = torch.randn(4, dtype=dtype, device=device)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=dtype, device=device)


def _make_body(seed, Mx, My, Mz, *, dtype, device):
    torch.manual_seed(seed)
    half = 0.7
    bx = torch.linspace(-half, half, Mx, dtype=dtype, device=device)
    by = torch.linspace(-0.9*half, 0.9*half, My, dtype=dtype, device=device)
    bz = torch.linspace(-0.7*half, 0.7*half, Mz, dtype=dtype, device=device)
    Bx, By, Bz = torch.meshgrid(bx, by, bz, indexing="ij")
    F = (torch.sqrt(Bx**2 + By**2 + Bz**2) - 0.35).contiguous()
    R = _random_rotation(dtype, device)
    R_T = R.T.contiguous().flatten().tolist()
    bp = ((torch.rand(3, dtype=dtype, device=device) - 0.5) * 0.4).tolist()
    cm = bp[:]                                              # COM = body pos
    lv = ((torch.rand(3, dtype=dtype, device=device) - 0.5)).tolist()
    av = ((torch.rand(3, dtype=dtype, device=device) - 0.5)).tolist()
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


def _reference_forces(
    sparse_cc_flat, cell_offsets, kin, aabb_lo, aabb_dim,
    gx, gy, gz, u_i0, u_j0, u_k0, Sj, Sk,
    xs, ys, zs, px, py, pz,
    eps_body, eps_solver, h3,
):
    """Pure-PyTorch reference matching the kernel's math, no atomics."""
    B = aabb_dim.shape[0]
    out = torch.zeros((B, 12), dtype=torch.float64, device=sparse_cc_flat.device)
    pi_v = math.pi
    inv_2eps = 0.5 / eps_body

    cell_off = cell_offsets.tolist()
    lo = aabb_lo.tolist()
    dim_ = aabb_dim.tolist()

    for b in range(B):
        Ai, Aj, Ak = dim_[b]
        i0, j0, k0 = lo[b]
        if Ai * Aj * Ak <= 0:
            continue
        ii = torch.arange(i0, i0 + Ai, device=gx.device)
        jj = torch.arange(j0, j0 + Aj, device=gx.device)
        kk = torch.arange(k0, k0 + Ak, device=gx.device)

        xc = gx[ii].view(Ai, 1, 1).expand(Ai, Aj, Ak)
        yc = gy[jj].view(1, Aj, 1).expand(Ai, Aj, Ak)
        zc = gz[kk].view(1, 1, Ak).expand(Ai, Aj, Ak)

        sdf = sparse_cc_flat[cell_off[b] : cell_off[b] + Ai*Aj*Ak].view(Ai, Aj, Ak)

        d_visc = sdf - eps_solver
        delta_visc = torch.where(
            (d_visc > -eps_body) & (d_visc < eps_body),
            (1.0 + torch.cos(pi_v * d_visc / eps_body)) * inv_2eps,
            torch.zeros_like(sdf),
        )
        delta_pres = torch.where(
            (sdf > -eps_body) & (sdf < eps_body),
            (1.0 + torch.cos(pi_v * sdf / eps_body)) * inv_2eps,
            torch.zeros_like(sdf),
        )

        sub_i = ii - u_i0
        sub_j = jj - u_j0
        sub_k = kk - u_k0
        # Stress / pforce sub-block
        xs_b = xs[sub_i.view(-1, 1, 1), sub_j.view(1, -1, 1), sub_k.view(1, 1, -1)]
        ys_b = ys[sub_i.view(-1, 1, 1), sub_j.view(1, -1, 1), sub_k.view(1, 1, -1)]
        zs_b = zs[sub_i.view(-1, 1, 1), sub_j.view(1, -1, 1), sub_k.view(1, 1, -1)]
        px_b = px[sub_i.view(-1, 1, 1), sub_j.view(1, -1, 1), sub_k.view(1, 1, -1)]
        py_b = py[sub_i.view(-1, 1, 1), sub_j.view(1, -1, 1), sub_k.view(1, 1, -1)]
        pz_b = pz[sub_i.view(-1, 1, 1), sub_j.view(1, -1, 1), sub_k.view(1, 1, -1)]

        cm_x = float(kin[b, 12])
        cm_y = float(kin[b, 13])
        cm_z = float(kin[b, 14])
        arm_x = (xc - cm_x).double()
        arm_y = (yc - cm_y).double()
        arm_z = (zc - cm_z).double()

        fv_x = (xs_b * delta_visc).double()
        fv_y = (ys_b * delta_visc).double()
        fv_z = (zs_b * delta_visc).double()
        fp_x = (px_b * delta_pres).double()
        fp_y = (py_b * delta_pres).double()
        fp_z = (pz_b * delta_pres).double()

        out[b, 0]  = fv_x.sum() * h3
        out[b, 1]  = fv_y.sum() * h3
        out[b, 2]  = fv_z.sum() * h3
        out[b, 3]  = (arm_y * fv_z - arm_z * fv_y).sum() * h3
        out[b, 4]  = (arm_z * fv_x - arm_x * fv_z).sum() * h3
        out[b, 5]  = (arm_x * fv_y - arm_y * fv_x).sum() * h3
        out[b, 6]  = fp_x.sum() * h3
        out[b, 7]  = fp_y.sum() * h3
        out[b, 8]  = fp_z.sum() * h3
        out[b, 9]  = (arm_y * fp_z - arm_z * fp_y).sum() * h3
        out[b, 10] = (arm_z * fp_x - arm_x * fp_z).sum() * h3
        out[b, 11] = (arm_x * fp_y - arm_y * fp_x).sum() * h3
    return out


def main():
    device = "cpu"
    dtype = torch.float64        # high-precision parity with float64 reference
    torch.manual_seed(0)

    # ----- fluid grid -----
    Nx, Ny, Nz = 64, 56, 48
    h = 0.05
    gx = torch.arange(Nx, dtype=dtype, device=device) * h - 1.0
    gy = torch.arange(Ny, dtype=dtype, device=device) * h - 0.9
    gz = torch.arange(Nz, dtype=dtype, device=device) * h - 0.8

    # ----- bodies -----
    B = 4
    bodies = [_make_body(seed=10+b, Mx=24, My=20, Mz=18, dtype=dtype, device=device)
              for b in range(B)]
    aabbs = [(2+b*4, 2+b*4+22, 3+b*3, 3+b*3+20, 2+b*2, 2+b*2+16) for b in range(B)]

    # ----- pack arrays -----
    F_chunks  = [bd["F"].flatten() for bd in bodies]
    bx_chunks = [bd["bx"] for bd in bodies]
    by_chunks = [bd["by"] for bd in bodies]
    bz_chunks = [bd["bz"] for bd in bodies]

    F_off  = [0]; bx_off = [0]; by_off = [0]; bz_off = [0]; cell_off = [0]
    shapes = []; metas  = []; kin    = []; lo = []; dim = []; max_vol = 0
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
        dim.append([i1-i0, j1-j0, k1-k0])
        vol = (i1-i0)*(j1-j0)*(k1-k0)
        cell_off.append(cell_off[-1] + vol)
        max_vol = max(max_vol, vol)

    F_flat  = torch.cat(F_chunks).contiguous()
    bx_flat = torch.cat(bx_chunks).contiguous()
    by_flat = torch.cat(by_chunks).contiguous()
    bz_flat = torch.cat(bz_chunks).contiguous()

    F_off_t  = torch.tensor(F_off,  dtype=torch.int64, device=device)
    bx_off_t = torch.tensor(bx_off, dtype=torch.int64, device=device)
    by_off_t = torch.tensor(by_off, dtype=torch.int64, device=device)
    bz_off_t = torch.tensor(bz_off, dtype=torch.int64, device=device)
    cell_off_t = torch.tensor(cell_off, dtype=torch.int64, device=device)
    shapes_t = torch.tensor(shapes, dtype=torch.int64, device=device)
    metas_t  = torch.tensor(metas,  dtype=dtype, device=device)
    kin_t    = torch.tensor(kin,    dtype=dtype, device=device)
    lo_t     = torch.tensor(lo,     dtype=torch.int64, device=device)
    dim_t    = torch.tensor(dim,    dtype=torch.int64, device=device)

    # Output buffers for streaming kernel (start at +inf for running min)
    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    sdf_w  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    bW = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)
    sparse_cc_flat = torch.zeros(cell_off[-1], dtype=dtype, device=device)

    # ----- step 1: streaming SDF (this populates sparse_cc_flat) -----
    streaming_sdf_min_3d_multi(
        F_flat, F_off_t,
        bx_flat, bx_off_t,
        by_flat, by_off_t,
        bz_flat, bz_off_t,
        shapes_t, metas_t, kin_t,
        lo_t, dim_t, cell_off_t,
        gx, gy, gz, h, max_vol,
        sdf_cc, sdf_u, sdf_v, sdf_w,
        bU, bV, bW,
        sparse_cc_flat,
    )
    # sanity: sparse_cc_flat must contain finite values inside each AABB
    assert torch.isfinite(sparse_cc_flat).all(), "sparse_cc_flat contains non-finite values"
    assert sparse_cc_flat.abs().max().item() < FAR, "sparse_cc_flat appears not to be written"

    # ----- step 2: forces — random stress / pforce -----
    Si, Sj, Sk = Nx, Ny, Nz
    u_i0 = u_j0 = u_k0 = 0
    xs = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    ys = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    zs = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    px = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    py = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    pz = torch.randn(Si, Sj, Sk, dtype=dtype, device=device)
    eps_body   = 1.5 * h
    eps_solver = 2.0 * h
    h3 = h * h * h

    out_kernel = torch.zeros((B, 12), dtype=torch.float64, device=device)
    bdim_forces_3d_multi(
        sparse_cc_flat, cell_off_t,
        kin_t,
        lo_t, dim_t,
        gx, gy, gz,
        u_i0, u_j0, u_k0, Sj, Sk,
        xs, ys, zs, px, py, pz,
        eps_body, eps_solver, h3,
        max_vol,
        out_kernel,
    )

    # ----- step 3: pure-Python reference -----
    out_ref = _reference_forces(
        sparse_cc_flat, cell_off_t, kin_t, lo_t, dim_t,
        gx, gy, gz, u_i0, u_j0, u_k0, Sj, Sk,
        xs, ys, zs, px, py, pz,
        eps_body, eps_solver, h3,
    )

    diff = (out_kernel - out_ref).abs().max().item()
    norm = out_ref.abs().max().item()
    rel = diff / max(norm, 1e-30)
    print(f"max |Δ|        = {diff:.3e}")
    print(f"max |ref|      = {norm:.3e}")
    print(f"max relative   = {rel:.3e}")
    # On CPU/double, kernel and reference should match to ~1e-12 relative.
    assert rel < 1e-10, (
        f"bdim_forces_3d_multi diverges from reference: "
        f"max abs {diff:.3e}, max rel {rel:.3e}"
    )
    # Sanity: forces should be non-trivial (the bodies' AABBs intersect the
    # δ-band) — a zero output would be a silent failure.
    assert norm > 1e-6, "reference output is ~0; test setup is degenerate"
    print("OK: bdim_forces_3d_multi matches the cached-cc-SDF reference.")

    # ----- step 4: edge case for the early-skip optimisation ---------
    # Replace the cached SDF with values that lie entirely outside both
    # delta-bands.  All 12 force/torque accumulators must come out zero,
    # *and* the kernel must not crash (no out-of-range stress reads, no
    # NaN propagation).  Hits the fast-path branch where every cell is
    # skipped before stress / pforce loads.
    far = float(eps_body * 100.0 + eps_solver * 100.0)
    sparse_far = torch.full_like(sparse_cc_flat, far)
    out_far = torch.zeros((B, 12), dtype=torch.float64, device=device)
    bdim_forces_3d_multi(
        sparse_far, cell_off_t,
        kin_t,
        lo_t, dim_t,
        gx, gy, gz,
        u_i0, u_j0, u_k0, Sj, Sk,
        xs, ys, zs, px, py, pz,
        eps_body, eps_solver, h3,
        max_vol,
        out_far,
    )
    assert torch.all(out_far == 0).item(), (
        f"early-skip path leaked non-zero outputs: max |out| = "
        f"{out_far.abs().max().item():.3e}"
    )
    print("OK: all-out-of-band SDF produces exactly zero forces (early-skip).")


if __name__ == "__main__":
    main()
