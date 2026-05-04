"""Self-contained CPU parity test for ``streaming_sdf_forces_fused_2d_multi``.

2-D analogue of ``test_streaming_sdf_forces_fused_self.py``.  The op
fuses Phase C (per-body SDF/face union update) with Phase D (lagged
force/torque integration).  This test extends the existing 2-D
``streaming_sdf_min`` parity test (``test_streaming_sdf_2d_self.py``)
with checks for the additional fused-only outputs:

    * ``winning_rho_cc``  — running-min cc-SDF body density;
    * ``out`` (B, 6) float64 — per-body 6-channel accumulator
      ``[fv_x, fv_y, t_v, fp_x, fp_y, t_p, 0, 0]``.

Run with::

    python -m lilytorch.src.kernels.test_streaming_sdf_forces_fused_2d_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers the namespace
from lilytorch.src.kernels import streaming_sdf_forces_fused_2d_multi

from lilytorch.src.kernels.test_streaming_sdf_2d_self import (
    _make_body_2d, _axis_meta_2d,
    _bilinear_sample_torch_2d, _ref_streaming_sdf_2d,
)


def _ref_winning_rho_and_forces_2d(
        bodies, aabbs, rho_bodies, gx, gy, h,
        u_prev, v_prev, p_prev, nx_cc, ny_cc,
        nu_rho_field, eps_body, eps_solver, rho_fluid,
        *, dtype, device):
    """Pure-PyTorch reference for ``winning_rho_cc`` and 6-channel ``out``.

    Mirrors the 2-D fused kernel's Phase C cc-SDF tracking and Phase D
    force integration with ``delta_order=1`` (no Towers correction).
    """
    Nx, Ny = gx.numel(), gy.numel()
    FAR = 1e4
    sdf_cc_ref = torch.full((Nx, Ny), FAR, dtype=dtype, device=device)
    winning_rho_ref = torch.full((Nx, Ny), float(rho_fluid),
                                 dtype=dtype, device=device)

    h_grid = float(h)
    inv_h = 1.0 / h_grid
    eps_b = float(eps_body)
    eps_s = float(eps_solver)
    pi_over_eb = math.pi / eps_b
    inv_2eps = 0.5 / eps_b

    out_ref = torch.zeros((len(bodies), 6), dtype=torch.float64, device=device)
    nu_rho_is_scalar = (nu_rho_field.numel() == 1)

    def cc_avg(u, axis):
        ip = u.roll(-1, dims=axis)
        result = 0.5 * (u + ip)
        last_slc = [slice(None)] * u.dim()
        last_slc[axis] = slice(-1, None)
        result[tuple(last_slc)] = u[tuple(last_slc)]
        return result

    def fwd_diff(u, axis):
        ip = u.roll(-1, dims=axis)
        last_slc = [slice(None)] * u.dim()
        last_slc[axis] = slice(-1, None)
        ip[tuple(last_slc)] = u[tuple(last_slc)]
        return (ip - u) * inv_h

    def central_diff(u_field, axis):
        ip = u_field.roll(-1, dims=axis)
        im = u_field.roll(+1, dims=axis)
        last_slc = [slice(None)] * u_field.dim()
        first_slc = [slice(None)] * u_field.dim()
        last_slc[axis] = slice(-1, None)
        first_slc[axis] = slice(0, 1)
        ip[tuple(last_slc)] = u_field[tuple(last_slc)]
        im[tuple(first_slc)] = u_field[tuple(first_slc)]
        return (ip - im) * 0.5 * inv_h

    u_cc = cc_avg(u_prev, 0)
    v_cc = cc_avg(v_prev, 1)
    dudx = fwd_diff(u_prev, 0)
    dvdy = fwd_diff(v_prev, 1)
    dudy = central_diff(u_cc, 1)
    dvdx = central_diff(v_cc, 0)

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

        # Phase D: cosine smoothed delta forces.
        band_lo = min(eps_s - eps_b, -eps_b)
        band_hi = max(eps_s + eps_b,  eps_b)
        in_band = (s_cc_body > band_lo) & (s_cc_body < band_hi)

        d_visc = s_cc_body - eps_s
        delta_visc = torch.where(
            (d_visc > -eps_b) & (d_visc < eps_b),
            (1.0 + torch.cos(pi_over_eb * d_visc)) * inv_2eps,
            torch.zeros_like(d_visc),
        )
        delta_pres = torch.where(
            (s_cc_body > -eps_b) & (s_cc_body < eps_b),
            (1.0 + torch.cos(pi_over_eb * s_cc_body)) * inv_2eps,
            torch.zeros_like(s_cc_body),
        )

        nx_b = nx_cc[i0:i1, j0:j1]
        ny_b = ny_cc[i0:i1, j0:j1]

        if nu_rho_is_scalar:
            nu_rho_b = nu_rho_field.item()
        else:
            nu_rho_b = nu_rho_field[i0:i1, j0:j1]

        dudx_b = dudx[i0:i1, j0:j1]
        dvdy_b = dvdy[i0:i1, j0:j1]
        dudy_b = dudy[i0:i1, j0:j1]
        dvdx_b = dvdx[i0:i1, j0:j1]

        xs_v = nu_rho_b * (2*dudx_b*nx_b + (dudy_b+dvdx_b)*ny_b)
        ys_v = nu_rho_b * ((dvdx_b+dudy_b)*nx_b + 2*dvdy_b*ny_b)

        p_c = p_prev[i0:i1, j0:j1]
        pxv = -p_c * nx_b
        pyv = -p_c * ny_b

        zero = torch.zeros_like(xs_v)
        fv_x = (xs_v * delta_visc).where(in_band, zero)
        fv_y = (ys_v * delta_visc).where(in_band, zero)
        fp_x = (pxv  * delta_pres).where(in_band, zero)
        fp_y = (pyv  * delta_pres).where(in_band, zero)

        arm_x = (xc - cm[0]).to(torch.float64)
        arm_y = (yc - cm[1]).to(torch.float64)
        fv_x_d, fv_y_d = fv_x.to(torch.float64), fv_y.to(torch.float64)
        fp_x_d, fp_y_d = fp_x.to(torch.float64), fp_y.to(torch.float64)

        h2 = h_grid ** 2
        out_ref[b, 0] += fv_x_d.sum().item() * h2
        out_ref[b, 1] += fv_y_d.sum().item() * h2
        out_ref[b, 2] += (arm_x * fv_y_d - arm_y * fv_x_d).sum().item() * h2
        out_ref[b, 3] += fp_x_d.sum().item() * h2
        out_ref[b, 4] += fp_y_d.sum().item() * h2
        out_ref[b, 5] += (arm_x * fp_y_d - arm_y * fp_x_d).sum().item() * h2
        # cols 6, 7 reserved (kernel writes nothing)

    return winning_rho_ref, out_ref


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

        rsdf_cc, rsdf_u, rsdf_v, rbU, rbV, _ = _ref_streaming_sdf_2d(
            bodies, aabbs, gx, gy, h, dtype=dtype, device=device,
            interp_method=interp_method,
        )
        rwinning_rho, rout = _ref_winning_rho_and_forces_2d(
            bodies, aabbs, rho_bodies, gx, gy, h,
            u_prev, v_prev, p_prev, nx_cc, ny_cc,
            nu_rho_field, eps_body, eps_solver, rho_fluid,
            dtype=dtype, device=device,
        )

        in_band = (sdf_cc.abs() < eps_body).sum().item()
        print(f"[{label}] cells inside cosine-delta support: {in_band}")
        assert in_band > 50, "scene degenerate"
        assert out.abs().max().item() > 1e-12, "all-zero forces — test trivial"

        # Reserved force columns (6, 7) must be exactly zero.
        reserved = out[:, 6:8].abs().max().item()
        assert reserved == 0.0, f"reserved force cols nonzero: {reserved}"

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
        cmp("out (forces)",    out[:, :6],  rout[:, :6],   1e-10)

    _run_and_check("nu_rho=const",  torch.tensor([0.5], dtype=dtype))
    nu_rho_var = (0.3 + 0.4 * torch.rand(Nx, Ny, dtype=dtype, generator=gen))
    _run_and_check("nu_rho=field",  nu_rho_var)

    print("OK: streaming_sdf_forces_fused_2d_multi matches the pure-PyTorch "
          "reference (sdf/face/winning_rho_cc/out) for both constant and "
          "variable nu_rho_field.")


if __name__ == "__main__":
    main()
