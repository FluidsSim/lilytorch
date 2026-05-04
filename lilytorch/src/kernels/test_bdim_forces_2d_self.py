"""Self-contained correctness test for ``bdim_forces_2d_multi``.

2-D analogue of ``test_bdim_forces_self.py``: builds a small synthetic
multi-body scene (rotated disc SDFs, random kinematics, random stress /
pressure-force fields), runs ``streaming_sdf_min_2d_multi`` to populate
``sparse_cc_flat``, then runs ``bdim_forces_2d_multi`` and compares the
6-channel (B, 6) result against a pure-PyTorch reference using the same
δ-kernel.

Layout of ``out`` per body:
    [fv_x, fv_y, t_v, fp_x, fp_y, t_p, 0, 0]
where t_v = arm_x*fv_y - arm_y*fv_x and t_p = arm_x*fp_y - arm_y*fp_x.
The trailing two slots are reserved by the kernel and must be zero.

Runs on CPU.  When CUDA is available, the same problem is run on CUDA
and cross-checked against the CPU output.

Run with::

    python -m lilytorch.src.kernels.test_bdim_forces_2d_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers the lilytorch_kernels namespace
from lilytorch.src.kernels import (
    streaming_sdf_min_2d_multi,
    bdim_forces_2d_multi,
)


def _random_rotation_2d(dtype, device, gen):
    theta = (torch.rand(1, generator=gen) * (2 * math.pi)).item()
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)


def _make_body(seed, Mx, My, *, dtype, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    half = 0.7
    bx = torch.linspace(-half, half, Mx, dtype=dtype, device=device)
    by = torch.linspace(-0.9 * half, 0.9 * half, My, dtype=dtype, device=device)
    Bx, By = torch.meshgrid(bx, by, indexing="ij")
    F = (torch.sqrt(Bx**2 + By**2) - 0.35).contiguous()
    R = _random_rotation_2d(dtype, device, gen)
    R_T = R.T.contiguous().flatten().tolist()
    bp = ((torch.rand(2, generator=gen, dtype=dtype) - 0.5) * 0.4).tolist()
    cm = bp[:]
    lv = (torch.rand(2, generator=gen, dtype=dtype) - 0.5).tolist()
    omega = float((torch.rand(1, generator=gen, dtype=dtype) - 0.5).item())
    return dict(F=F, bx=bx, by=by, R_T=R_T, bp=bp, cm=cm, lv=lv, omega=omega,
                Mx=Mx, My=My)


def _axis_meta(bd):
    bx, by = bd["bx"], bd["by"]
    dx = float(bx[1] - bx[0])
    dy = float(by[1] - by[0])
    return [
        float(bx[0]), float(by[0]),
        float(bx[-1]), float(by[-1]),
        1.0 / dx, 1.0 / dy,
        1.0 / (dx * dy),
    ]


def _reference_forces_2d(
    sparse_cc_flat, cell_offsets, kin, aabb_lo, aabb_dim,
    gx, gy, u_i0, u_j0, Sj,
    xs, ys, px, py,
    eps_body, eps_solver, h2,
):
    """Pure-PyTorch reference matching the kernel's math, no atomics."""
    B = aabb_dim.shape[0]
    out = torch.zeros((B, 6), dtype=torch.float64, device=sparse_cc_flat.device)
    pi_v = math.pi
    inv_2eps = 0.5 / eps_body

    cell_off = cell_offsets.tolist()
    lo = aabb_lo.tolist()
    dim_ = aabb_dim.tolist()

    for b in range(B):
        Ai, Aj = dim_[b]
        i0, j0 = lo[b]
        if Ai * Aj <= 0:
            continue
        ii = torch.arange(i0, i0 + Ai, device=gx.device)
        jj = torch.arange(j0, j0 + Aj, device=gx.device)

        xc = gx[ii].view(Ai, 1).expand(Ai, Aj)
        yc = gy[jj].view(1, Aj).expand(Ai, Aj)

        sdf = sparse_cc_flat[cell_off[b] : cell_off[b] + Ai * Aj].view(Ai, Aj)

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
        xs_b = xs[sub_i.view(-1, 1), sub_j.view(1, -1)]
        ys_b = ys[sub_i.view(-1, 1), sub_j.view(1, -1)]
        px_b = px[sub_i.view(-1, 1), sub_j.view(1, -1)]
        py_b = py[sub_i.view(-1, 1), sub_j.view(1, -1)]

        # 2-D kin row: cm_xy at offsets 6..7.
        cm_x = float(kin[b, 6])
        cm_y = float(kin[b, 7])
        arm_x = (xc - cm_x).double()
        arm_y = (yc - cm_y).double()

        fv_x = (xs_b * delta_visc).double()
        fv_y = (ys_b * delta_visc).double()
        fp_x = (px_b * delta_pres).double()
        fp_y = (py_b * delta_pres).double()

        out[b, 0] = fv_x.sum() * h2
        out[b, 1] = fv_y.sum() * h2
        out[b, 2] = (arm_x * fv_y - arm_y * fv_x).sum() * h2
        out[b, 3] = fp_x.sum() * h2
        out[b, 4] = fp_y.sum() * h2
        out[b, 5] = (arm_x * fp_y - arm_y * fp_x).sum() * h2
        # out[b, 6], out[b, 7] reserved
    return out


def _build_scene(*, dtype, device):
    torch.manual_seed(0)
    Nx, Ny = 64, 56
    h = 0.05
    gx = torch.arange(Nx, dtype=dtype, device=device) * h - 1.0
    gy = torch.arange(Ny, dtype=dtype, device=device) * h - 0.9

    B = 4
    bodies = [_make_body(seed=10 + b, Mx=24, My=20, dtype=dtype, device=device)
              for b in range(B)]
    aabbs = [(2 + b * 4, 2 + b * 4 + 22, 3 + b * 3, 3 + b * 3 + 20)
             for b in range(B)]

    F_chunks  = [bd["F"].flatten() for bd in bodies]
    bx_chunks = [bd["bx"]          for bd in bodies]
    by_chunks = [bd["by"]          for bd in bodies]

    F_off = [0]; bx_off = [0]; by_off = [0]; cell_off = [0]
    shapes = []; metas = []; kin = []; lo = []; dim_ = []; max_vol = 0
    for bd, ab in zip(bodies, aabbs):
        F_off.append(F_off[-1]   + bd["F"].numel())
        bx_off.append(bx_off[-1] + bd["bx"].numel())
        by_off.append(by_off[-1] + bd["by"].numel())
        shapes.append([bd["Mx"], bd["My"]])
        metas.append(_axis_meta(bd))
        kin.append(bd["R_T"] + bd["bp"] + bd["cm"] + bd["lv"] + [bd["omega"]])
        i0, i1, j0, j1 = ab
        lo.append([i0, j0])
        dim_.append([i1 - i0, j1 - j0])
        vol = (i1 - i0) * (j1 - j0)
        cell_off.append(cell_off[-1] + vol)
        max_vol = max(max_vol, vol)

    F_flat  = torch.cat(F_chunks).contiguous()
    bx_flat = torch.cat(bx_chunks).contiguous()
    by_flat = torch.cat(by_chunks).contiguous()

    F_off_t  = torch.tensor(F_off,  dtype=torch.int64, device=device)
    bx_off_t = torch.tensor(bx_off, dtype=torch.int64, device=device)
    by_off_t = torch.tensor(by_off, dtype=torch.int64, device=device)
    cell_off_t = torch.tensor(cell_off, dtype=torch.int64, device=device)
    shapes_t = torch.tensor(shapes, dtype=torch.int64, device=device)
    metas_t  = torch.tensor(metas,  dtype=dtype, device=device)
    kin_t    = torch.tensor(kin,    dtype=dtype, device=device)
    lo_t     = torch.tensor(lo,     dtype=torch.int64, device=device)
    dim_t    = torch.tensor(dim_,   dtype=torch.int64, device=device)

    FAR = 1e4
    sdf_cc = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_u  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    sdf_v  = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    bU = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    bV = torch.zeros((Nx, Ny), dtype=dtype, device=device)
    sparse_cc_flat = torch.zeros(cell_off[-1], dtype=dtype, device=device)

    streaming_sdf_min_2d_multi(
        F_flat, F_off_t,
        bx_flat, bx_off_t,
        by_flat, by_off_t,
        shapes_t, metas_t, kin_t,
        lo_t, dim_t, cell_off_t,
        gx, gy, h, max_vol,
        sdf_cc, sdf_u, sdf_v, bU, bV,
        sparse_cc_flat,
    )
    assert torch.isfinite(sparse_cc_flat).all(), "sparse_cc_flat non-finite"
    assert sparse_cc_flat.abs().max().item() < FAR, "sparse_cc_flat empty"

    return dict(
        Nx=Nx, Ny=Ny, h=h, B=B,
        gx=gx, gy=gy,
        kin_t=kin_t, lo_t=lo_t, dim_t=dim_t, cell_off_t=cell_off_t,
        sparse_cc_flat=sparse_cc_flat,
        max_vol=max_vol, bodies=bodies, aabbs=aabbs,
    )


def _run_forces(scene, stress, *, dtype, device):
    Nx, Ny, h, B = scene["Nx"], scene["Ny"], scene["h"], scene["B"]
    Si, Sj = Nx, Ny
    u_i0 = u_j0 = 0
    xs, ys, px, py = (t.to(device=device, dtype=dtype) for t in stress)

    eps_body   = 1.5 * h
    eps_solver = 2.0 * h
    h2 = h * h

    out_kernel = torch.zeros((B, 6), dtype=torch.float64, device=device)
    bdim_forces_2d_multi(
        scene["sparse_cc_flat"], scene["cell_off_t"],
        scene["kin_t"],
        scene["lo_t"], scene["dim_t"],
        scene["gx"], scene["gy"],
        u_i0, u_j0, Sj,
        xs, ys, px, py,
        eps_body, eps_solver, h2,
        scene["max_vol"],
        1,  # delta_order
        out_kernel,
    )

    out_ref = _reference_forces_2d(
        scene["sparse_cc_flat"], scene["cell_off_t"], scene["kin_t"],
        scene["lo_t"], scene["dim_t"],
        scene["gx"], scene["gy"], u_i0, u_j0, Sj,
        xs, ys, px, py,
        eps_body, eps_solver, h2,
    )
    return out_kernel, out_ref


def main():
    dtype = torch.float64

    # ---- CPU parity ---------------------------------------------------
    print("== CPU parity vs PyTorch reference ==")
    torch.manual_seed(0)
    scene_cpu = _build_scene(dtype=dtype, device="cpu")

    # Build stress fields once on CPU; reuse for CUDA cross-check.
    Nx, Ny = scene_cpu["Nx"], scene_cpu["Ny"]
    torch.manual_seed(123)
    stress = (
        torch.randn(Nx, Ny, dtype=dtype),
        torch.randn(Nx, Ny, dtype=dtype),
        torch.randn(Nx, Ny, dtype=dtype),
        torch.randn(Nx, Ny, dtype=dtype),
    )

    out_kernel_cpu, out_ref = _run_forces(scene_cpu, stress, dtype=dtype, device="cpu")

    diff = (out_kernel_cpu - out_ref).abs().max().item()
    norm = out_ref.abs().max().item()
    rel = diff / max(norm, 1e-30)
    print(f"  max |Δ|        = {diff:.3e}")
    print(f"  max |ref|      = {norm:.3e}")
    print(f"  rel error      = {rel:.3e}")
    assert rel < 1e-12, f"CPU parity failed: rel={rel:.3e}"

    # Reserved trailing channels must be zero.
    reserved = out_kernel_cpu[:, 6:].abs().max().item()
    print(f"  reserved cols  = {reserved:.3e}  (must be 0)")
    assert reserved == 0.0, "trailing reserved channels must be zero"

    # Sanity: at least one body has non-zero force.
    nonzero = (out_kernel_cpu[:, :6].abs() > 0).any().item()
    assert nonzero, "all force/torque entries are zero — scene degenerate"

    # ---- CUDA cross-check (optional) ----------------------------------
    if torch.cuda.is_available():
        print("== CUDA-vs-CPU cross-check ==")
        # Rebuild the scene on CUDA with the SAME RNG seed so per-body
        # rotations / kinematics / SDFs match the CPU scene exactly.
        torch.manual_seed(0)
        scene_cuda = _build_scene(dtype=dtype, device="cuda")
        out_kernel_cuda, out_ref_cuda = _run_forces(
            scene_cuda, stress, dtype=dtype, device="cuda")
        diff = (out_kernel_cuda - out_ref_cuda).abs().max().item()
        norm = out_ref_cuda.abs().max().item()
        rel = diff / max(norm, 1e-30)
        print(f"  CUDA vs ref:  max |Δ|={diff:.3e}  rel={rel:.3e}")
        assert rel < 1e-12, f"CUDA parity failed: rel={rel:.3e}"
        cpu_to_cuda_diff = (out_kernel_cpu.cpu() - out_kernel_cuda.cpu()).abs().max().item()
        cpu_to_cuda_rel = cpu_to_cuda_diff / max(norm, 1e-30)
        print(f"  CPU vs CUDA: max |Δ|={cpu_to_cuda_diff:.3e}  rel={cpu_to_cuda_rel:.3e}")
        assert cpu_to_cuda_rel < 1e-10, (
            f"CPU and CUDA outputs disagree (rel={cpu_to_cuda_rel:.3e})")
    else:
        print("CUDA not available, skipping CUDA cross-check.")

    print("test_bdim_forces_2d_self: PASSED")


if __name__ == "__main__":
    main()
