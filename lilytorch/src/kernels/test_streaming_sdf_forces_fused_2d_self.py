"""Self-contained CPU parity test for ``streaming_sdf_forces_fused_2d_multi``.

2-D analogue of ``test_streaming_sdf_forces_fused_self.py``.  The op
fuses Phase C (per-body SDF/face union update) with a compact Phase D
force pass that uses CURRENT union-SDF normals and CURRENT per-body
delta support.  This test extends the existing 2-D
``streaming_sdf_min`` parity test (``test_streaming_sdf_2d_self.py``)
with checks for the additional fused-only outputs:

    * ``winning_rho_cc``  — running-min cc-SDF body density;
    * ``out`` (B, 6) float64 — per-body accumulator
      ``[fv_x, fv_y, t_v, fp_x, fp_y, t_p]``.

Run with::

    python -m lilytorch.src.kernels.test_streaming_sdf_forces_fused_2d_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers the namespace
from lilytorch.src.kernels import (
    streaming_sdf_forces_fused_2d_multi,
    bdim_forces_2d_multi,
)

from lilytorch.src.kernels.test_streaming_sdf_2d_self import (
    _make_body_2d, _axis_meta_2d,
    _bilinear_sample_torch_2d, _ref_streaming_sdf_2d,
)
from lilytorch.src.kernels.test_bdim_forces_2d_self import _reference_forces_2d


def _ref_winning_rho_and_forces_2d(
        bodies, aabbs, rho_bodies, gx, gy, h,
        rho_fluid,
        *, dtype, device):
    """Pure-PyTorch reference for ``winning_rho_cc``.

    The force reference is built separately from the current union SDF
    normals plus the current per-body sparse CC-SDF slabs so the test
    matches the stable 2-D `kernels` force path.
    """
    Nx, Ny = gx.numel(), gy.numel()
    FAR = 1e4
    sdf_cc_ref = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    winning_rho_ref = torch.full((Nx, Ny), float(rho_fluid),
                                 dtype=dtype, device=device)

    for b, (bd, ab) in enumerate(zip(bodies, aabbs)):
        i0, i1, j0, j1 = ab
        Ai, Aj = i1 - i0, j1 - j0
        rho_b = float(rho_bodies[b])

        ii = torch.arange(i0, i1, device=device)
        jj = torch.arange(j0, j1, device=device)
        xc = gx[ii].view(Ai, 1).expand(Ai, Aj)
        yc = gy[jj].view(1, Aj).expand(Ai, Aj)

        R_T = torch.tensor(bd["R_T"], dtype=dtype, device=device).view(2, 2)
        bp  = torch.tensor(bd["bp"], dtype=dtype, device=device)
        cm  = torch.tensor(bd["cm"], dtype=dtype, device=device)

        dxw = xc - bp[0]; dyw = yc - bp[1]
        bxq = R_T[0,0]*dxw + R_T[0,1]*dyw
        byq = R_T[1,0]*dxw + R_T[1,1]*dyw
        s_cc_body = _bilinear_sample_torch_2d(
            bd["F"], bd["bx"], bd["by"], bxq, byq)

        # Phase C: running-min cc-SDF + winning rho.
        sub_cc = sdf_cc_ref[i0:i1, j0:j1]
        win = s_cc_body < sub_cc
        sdf_cc_ref[i0:i1, j0:j1] = torch.where(win, s_cc_body, sub_cc)
        sub_rho = winning_rho_ref[i0:i1, j0:j1]
        winning_rho_ref[i0:i1, j0:j1] = torch.where(
            win, torch.full_like(sub_rho, rho_b), sub_rho)
    return winning_rho_ref


def _current_union_normals_2d(sdf_cc, h):
    gx_cc, gy_cc = torch.gradient(sdf_cc, spacing=[h, h], edge_order=2)
    norm = torch.sqrt(gx_cc**2 + gy_cc**2)
    inv_norm = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))
    return gx_cc * inv_norm, gy_cc * inv_norm


def main():
    device = "cpu"
    dtype = torch.float64
    torch.manual_seed(0)

    Nx, Ny = 56, 48
    h = 0.05
    gx = torch.arange(Nx, dtype=dtype) * h - 1.0
    gy = torch.arange(Ny, dtype=dtype) * h - 0.9

    B = 3
    bodies = [_make_body_2d(seed=10+b, Mx=24, My=20, dtype=dtype, device=device)
              for b in range(B)]
    # Overlapping AABBs so multiple bodies compete on shared cells.
    aabbs = [(2+b*4, 2+b*4+22, 3+b*3, 3+b*3+20) for b in range(B)]

    rho_bodies = torch.tensor([1.05, 1.10, 1.20], dtype=dtype, device=device)
    rho_fluid = 1.0

    F_chunks = [bd["F"].flatten() for bd in bodies]
    F_off = [0]; shapes = []; metas = []; kin = []; lo = []; dim_ = []; max_vol = 0
    for bd, ab in zip(bodies, aabbs):
        F_off.append(F_off[-1] + bd["F"].numel())
        shapes.append([bd["Mx"], bd["My"]])
        metas.append(_axis_meta_2d(bd))
        kin.append(bd["R_T"] + bd["bp"] + bd["cm"] + bd["lv"] + [bd["omega"]])
        i0, i1, j0, j1 = ab
        lo.append([i0, j0])
        dim_.append([i1-i0, j1-j0])
        max_vol = max(max_vol, (i1-i0)*(j1-j0))

    F_flat   = torch.cat(F_chunks).contiguous()
    F_off_t  = torch.tensor(F_off, dtype=torch.int64)
    shapes_t = torch.tensor(shapes, dtype=torch.int64)
    metas_t  = torch.tensor(metas, dtype=dtype)
    kin_t    = torch.tensor(kin, dtype=dtype)
    lo_t     = torch.tensor(lo, dtype=torch.int64)
    dim_t    = torch.tensor(dim_, dtype=torch.int64)

    gen = torch.Generator(device="cpu").manual_seed(123)
    u_prev = torch.randn(Nx, Ny, dtype=dtype, generator=gen) * 0.4
    v_prev = torch.randn(Nx, Ny, dtype=dtype, generator=gen) * 0.4
    p_prev = torch.randn(Nx, Ny, dtype=dtype, generator=gen) * 0.5
    nx_cc  = torch.randn(Nx, Ny, dtype=dtype, generator=gen) * 0.6
    ny_cc  = torch.randn(Nx, Ny, dtype=dtype, generator=gen) * 0.6

    eps_body = 2.5 * h
    eps_solver = 0.0
    h2 = h ** 2
    delta_order = 1
    interp_method = 0  # bilinear

    FAR = 1e4

    def _alloc_state():
        sdf_cc = torch.full((Nx, Ny), FAR, dtype=dtype)
        sdf_u  = torch.full((Nx, Ny), FAR, dtype=dtype)
        sdf_v  = torch.full((Nx, Ny), FAR, dtype=dtype)
        bU = torch.zeros((Nx, Ny), dtype=dtype)
        bV = torch.zeros((Nx, Ny), dtype=dtype)
        winning_rho = torch.full((Nx, Ny), float(rho_fluid), dtype=dtype)
        out = torch.zeros((B, 6), dtype=torch.float64)
        return sdf_cc, sdf_u, sdf_v, bU, bV, winning_rho, out

    def _run_and_check(label, nu_rho_field):
        sdf_cc, sdf_u, sdf_v, bU, bV, winning_rho, out = _alloc_state()

        streaming_sdf_forces_fused_2d_multi(
            F_flat, F_off_t, shapes_t, metas_t, kin_t, lo_t, dim_t,
            gx, gy, h, max_vol,
            sdf_cc, sdf_u, sdf_v, bU, bV,
            interp_method, rho_bodies, winning_rho,
            u_prev, v_prev, p_prev, nx_cc, ny_cc, nu_rho_field,
            eps_body, eps_solver, h2, delta_order, out,
        )

        rsdf_cc, rsdf_u, rsdf_v, rbU, rbV, rsparse = _ref_streaming_sdf_2d(
            bodies, aabbs, gx, gy, h, dtype=dtype, device=device,
            interp_method=interp_method,
        )
        rwinning_rho = _ref_winning_rho_and_forces_2d(
            bodies, aabbs, rho_bodies, gx, gy, h,
            rho_fluid,
            dtype=dtype, device=device,
        )

        in_band = (sdf_cc.abs() < eps_body).sum().item()
        print(f"[{label}] cells inside cosine-delta support: {in_band}")
        assert in_band > 50, "scene degenerate"
        assert out.abs().max().item() > 1e-12, "all-zero forces — test trivial"

        def cmp(name, kernel, ref, tol):
            diff = (kernel - ref).abs().max().item()
            norm = ref.abs().max().item()
            rel = diff / max(norm, 1e-30)
            print(f"  {name:<20}  max |Δ|={diff:.3e}  max |ref|={norm:.3e}  rel={rel:.3e}")
            assert rel < tol, f"{name}: rel={rel:.3e} > tol={tol:.3e}"

        cmp("sdf_cc",          sdf_cc,      rsdf_cc,       1e-12)
        cmp("sdf_u",           sdf_u,       rsdf_u,        1e-12)
        cmp("sdf_v",           sdf_v,       rsdf_v,        1e-12)
        cmp("body_u",          bU,          rbU,           1e-12)
        cmp("body_v",          bV,          rbV,           1e-12)
        cmp("winning_rho_cc",  winning_rho, rwinning_rho,  1e-15)

        # Also compare the fused Phase D result against the separate
        # `kernels` force path fed with the same current union normals
        # and per-body sparse CC-SDFs.
        u_cc = 0.5 * (u_prev + u_prev.roll(-1, dims=0))
        u_cc[-1, :] = u_prev[-1, :]
        v_cc = 0.5 * (v_prev + v_prev.roll(-1, dims=1))
        v_cc[:, -1] = v_prev[:, -1]

        dudx = torch.empty_like(u_prev)
        dudx[:-1, :] = (u_prev[1:, :] - u_prev[:-1, :]) / h
        dudx[-1, :] = dudx[-2, :]
        dvdy = torch.empty_like(v_prev)
        dvdy[:, :-1] = (v_prev[:, 1:] - v_prev[:, :-1]) / h
        dvdy[:, -1] = dvdy[:, -2]

        dudy = torch.empty_like(u_cc)
        dudy[:, 1:-1] = (u_cc[:, 2:] - u_cc[:, :-2]) * 0.5 / h
        dudy[:, 0] = (-3.0 * u_cc[:, 0] + 4.0 * u_cc[:, 1] - u_cc[:, 2]) * 0.5 / h
        dudy[:, -1] = (3.0 * u_cc[:, -1] - 4.0 * u_cc[:, -2] + u_cc[:, -3]) * 0.5 / h

        dvdx = torch.empty_like(v_cc)
        dvdx[1:-1, :] = (v_cc[2:, :] - v_cc[:-2, :]) * 0.5 / h
        dvdx[0, :] = (-3.0 * v_cc[0, :] + 4.0 * v_cc[1, :] - v_cc[2, :]) * 0.5 / h
        dvdx[-1, :] = (3.0 * v_cc[-1, :] - 4.0 * v_cc[-2, :] + v_cc[-3, :]) * 0.5 / h

        nx_cur, ny_cur = _current_union_normals_2d(rsdf_cc, h)
        xstress = nu_rho_field * (2 * dudx * nx_cur + (dudy + dvdx) * ny_cur)
        ystress = nu_rho_field * ((dvdx + dudy) * nx_cur + 2 * dvdy * ny_cur)
        pforce_x = -p_prev * nx_cur
        pforce_y = -p_prev * ny_cur

        cell_off = [0]
        for ab in aabbs:
            i0, i1, j0, j1 = ab
            cell_off.append(cell_off[-1] + (i1 - i0) * (j1 - j0))
        cell_off_t = torch.tensor(cell_off, dtype=torch.int64, device=device)
        out_sep = torch.zeros((B, 6), dtype=torch.float64, device=device)
        bdim_forces_2d_multi(
            rsparse, cell_off_t, kin_t, lo_t, dim_t, gx, gy,
            0, 0, Ny,
            xstress, ystress, pforce_x, pforce_y,
            eps_body, eps_solver, h2, max_vol, delta_order, out_sep,
        )
        out_sep_ref = _reference_forces_2d(
            rsparse, cell_off_t, kin_t, lo_t, dim_t, gx, gy,
            0, 0, Ny,
            xstress, ystress, pforce_x, pforce_y,
            eps_body, eps_solver, h2,
        )
        cmp("out (forces)",    out[:, :6],     out_sep_ref[:, :6], 1e-10)
        cmp("separate out",     out_sep[:, :6], out_sep_ref[:, :6], 1e-10)
        cmp("fused vs separate", out[:, :6], out_sep[:, :6], 1e-10)

    _run_and_check("nu_rho=const",  torch.tensor([0.5], dtype=dtype))
    nu_rho_var = (0.3 + 0.4 * torch.rand(Nx, Ny, dtype=dtype, generator=gen))
    _run_and_check("nu_rho=field",  nu_rho_var)

    print("OK: streaming_sdf_forces_fused_2d_multi matches the pure-PyTorch "
          "reference (sdf/face/winning_rho_cc/out) for both constant and "
          "variable nu_rho_field.")


if __name__ == "__main__":
    main()
