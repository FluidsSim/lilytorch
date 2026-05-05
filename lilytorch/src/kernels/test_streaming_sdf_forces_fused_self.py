"""Self-contained CPU parity test for ``streaming_sdf_forces_fused_3d_multi``.

This op fuses Phase C (per-body SDF / face-velocity union update) with
Phase D (lagged force/torque integration) in a single kernel pass per
body.  ``test_streaming_sdf_self.py`` covers the SDF/face outputs of
``streaming_sdf_min_3d_multi``; this test extends that coverage to the
fused variant, which additionally writes:

    * ``winning_rho_cc`` (B, *)   — per-cell density of the body that
      currently owns the union-min ``sdf_cc`` (or the initial fluid
      density on cells no body claims);
    * ``out`` (B, 12) float64    — per-body 12-channel accumulator of
      [F_visc, T_visc, F_pres, T_pres] integrated against the cosine
      smoothed delta of ``s_cc``.

The reference is a pure-PyTorch transcription of Phase D of the C++
kernel (clamped MAC velocity gradients, CC-interpolated cross
derivatives, cosine smoothed delta with ``delta_order=1``).  We
deliberately exercise multiple bodies whose AABBs overlap to stress the
running-min ``sdf_cc`` / ``winning_rho_cc`` race-free CPU path.

Run with::

    python -m lilytorch.src.kernels.test_streaming_sdf_forces_fused_self
"""
from __future__ import annotations

import math
import torch

import lilytorch.src.kernels  # noqa: F401 — registers the namespace
from lilytorch.src.kernels import streaming_sdf_forces_fused_3d_multi

# Reuse helpers from the streaming_sdf_min parity test.
from lilytorch.src.kernels.test_streaming_sdf_self import (
    _make_body, _axis_meta, _ref_streaming_sdf,
)


def _ref_winning_rho_and_forces(bodies, aabbs, rho_bodies, gx, gy, gz, h,
                                u_prev, v_prev, w_prev, p_prev,
                                nx_cc, ny_cc, nz_cc,
                                nu_rho_field, eps_body, eps_solver,
                                rho_fluid, *, dtype, device):
    """Pure-PyTorch reference for ``winning_rho_cc`` and 12-channel ``out``.

    Mirrors Phase C cc-SDF tracking and Phase D force integration
    (``delta_order=1``) of the fused kernel.
    """
    Nx, Ny, Nz = gx.numel(), gy.numel(), gz.numel()
    FAR = 1e4
    sdf_cc_ref = torch.full((Nx, Ny, Nz), FAR, dtype=dtype, device=device)
    winning_rho_ref = torch.full((Nx, Ny, Nz), float(rho_fluid),
                                 dtype=dtype, device=device)

    h_grid = float(h)
    inv_h = 1.0 / h_grid
    eps_b = float(eps_body)
    eps_s = float(eps_solver)
    pi_over_eb = math.pi / eps_b
    inv_2eps = 0.5 / eps_b

    out_ref = torch.zeros((len(bodies), 12), dtype=torch.float64, device=device)
    nu_rho_is_scalar = (nu_rho_field.numel() == 1)

    # CC-interpolated faces with boundary-clamped neighbour indices.
    # u_cc[i,j,k] = 0.5*(u[i,j,k] + u[i+1,j,k])  with clamp at i=Nx-1.
    def cc_avg(u, axis):
        ip = u.roll(-1, dims=axis)
        # Edge cell: (u[Nx-1] + u[Nx]) / 2 should reduce to (u[Nx-1] + u[Nx-1])/2;
        # since ``ip1 = Nx-1`` clamp in the kernel, the averaged stencil at
        # the last cell is just u[Nx-1]. Replicate with a slice fix.
        result = 0.5 * (u + ip)
        # Last index along axis: replace with u itself (clamp ip1 -> i).
        last_slc = [slice(None)] * u.dim()
        last_slc[axis] = slice(-1, None)
        result[tuple(last_slc)] = u[tuple(last_slc)]
        return result

    u_cc = cc_avg(u_prev, 0)  # average along i
    v_cc = cc_avg(v_prev, 1)  # average along j
    w_cc = cc_avg(w_prev, 2)  # average along k

    # Forward MAC differences with clamped ip1.
    def fwd_diff(u, axis):
        ip = u.roll(-1, dims=axis)
        last_slc = [slice(None)] * u.dim()
        last_slc[axis] = slice(-1, None)
        ip[tuple(last_slc)] = u[tuple(last_slc)]
        return (ip - u) * inv_h

    dudx = fwd_diff(u_prev, 0)
    dvdy = fwd_diff(v_prev, 1)
    dwdz = fwd_diff(w_prev, 2)

    # Central differences with edge clamp (im1 -> 0, ip1 -> Nx-1).
    def central_diff(u_cc_field, axis):
        ip = u_cc_field.roll(-1, dims=axis)
        im = u_cc_field.roll(+1, dims=axis)
        last_slc = [slice(None)] * u_cc_field.dim()
        first_slc = [slice(None)] * u_cc_field.dim()
        last_slc[axis] = slice(-1, None)
        first_slc[axis] = slice(0, 1)
        ip[tuple(last_slc)] = u_cc_field[tuple(last_slc)]
        im[tuple(first_slc)] = u_cc_field[tuple(first_slc)]
        return (ip - im) * 0.5 * inv_h

    dudy = central_diff(u_cc, 1)
    dudz = central_diff(u_cc, 2)
    dvdx = central_diff(v_cc, 0)
    dvdz = central_diff(v_cc, 2)
    dwdx = central_diff(w_cc, 0)
    dwdy = central_diff(w_cc, 1)

    # Sampler matches the existing trilinear-with-border helper.
    from lilytorch.src.kernels.test_streaming_sdf_self import (
        _trilinear_sample_border_torch as sampler,
    )

    for b, (bd, ab) in enumerate(zip(bodies, aabbs)):
        i0, i1, j0, j1, k0, k1 = ab
        Ai, Aj, Ak = i1-i0, j1-j0, k1-k0
        rho_b = float(rho_bodies[b])

        ii = torch.arange(i0, i1, device=device)
        jj = torch.arange(j0, j1, device=device)
        kk = torch.arange(k0, k1, device=device)
        xc = gx[ii].view(Ai,1,1).expand(Ai,Aj,Ak)
        yc = gy[jj].view(1,Aj,1).expand(Ai,Aj,Ak)
        zc = gz[kk].view(1,1,Ak).expand(Ai,Aj,Ak)

        R_T = torch.tensor(bd["R_T"], dtype=dtype, device=device).view(3,3)
        bp  = torch.tensor(bd["bp"], dtype=dtype, device=device)
        cm  = torch.tensor(bd["cm"], dtype=dtype, device=device)

        dx = xc - bp[0]; dy = yc - bp[1]; dz = zc - bp[2]
        bxq = R_T[0,0]*dx + R_T[0,1]*dy + R_T[0,2]*dz
        byq = R_T[1,0]*dx + R_T[1,1]*dy + R_T[1,2]*dz
        bzq = R_T[2,0]*dx + R_T[2,1]*dy + R_T[2,2]*dz

        s_cc_body = sampler(bd["F"], bd["bx"], bd["by"], bd["bz"],
                            bxq, byq, bzq)

        # ---- Phase C union-min cc-SDF + winning rho ----
        sub_cc = sdf_cc_ref[i0:i1, j0:j1, k0:k1]
        win = s_cc_body < sub_cc
        sdf_cc_ref[i0:i1, j0:j1, k0:k1] = torch.where(win, s_cc_body, sub_cc)
        sub_rho = winning_rho_ref[i0:i1, j0:j1, k0:k1]
        winning_rho_ref[i0:i1, j0:j1, k0:k1] = torch.where(
            win, torch.full_like(sub_rho, rho_b), sub_rho)

        # ---- Phase D inline force integration (delta_order=1) ----
        band_lo = min(eps_s - eps_b, -eps_b)
        band_hi = max(eps_s + eps_b,  eps_b)
        in_band = (s_cc_body > band_lo) & (s_cc_body < band_hi)

        # cosine smoothed deltas
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

        nx_b = nx_cc[i0:i1, j0:j1, k0:k1]
        ny_b = ny_cc[i0:i1, j0:j1, k0:k1]
        nz_b = nz_cc[i0:i1, j0:j1, k0:k1]

        if nu_rho_is_scalar:
            nu_rho_b = nu_rho_field.item()
        else:
            nu_rho_b = nu_rho_field[i0:i1, j0:j1, k0:k1]

        dudx_b = dudx[i0:i1, j0:j1, k0:k1]
        dvdy_b = dvdy[i0:i1, j0:j1, k0:k1]
        dwdz_b = dwdz[i0:i1, j0:j1, k0:k1]
        dudy_b = dudy[i0:i1, j0:j1, k0:k1]
        dudz_b = dudz[i0:i1, j0:j1, k0:k1]
        dvdx_b = dvdx[i0:i1, j0:j1, k0:k1]
        dvdz_b = dvdz[i0:i1, j0:j1, k0:k1]
        dwdx_b = dwdx[i0:i1, j0:j1, k0:k1]
        dwdy_b = dwdy[i0:i1, j0:j1, k0:k1]

        xs_v = nu_rho_b * (2*dudx_b*nx_b + (dudy_b+dvdx_b)*ny_b + (dudz_b+dwdx_b)*nz_b)
        ys_v = nu_rho_b * ((dvdx_b+dudy_b)*nx_b + 2*dvdy_b*ny_b + (dvdz_b+dwdy_b)*nz_b)
        zs_v = nu_rho_b * ((dwdx_b+dudz_b)*nx_b + (dwdy_b+dvdz_b)*ny_b + 2*dwdz_b*nz_b)

        p_c = p_prev[i0:i1, j0:j1, k0:k1]
        pxv = -p_c * nx_b
        pyv = -p_c * ny_b
        pzv = -p_c * nz_b

        fv_x = (xs_v * delta_visc).where(in_band, torch.zeros_like(xs_v))
        fv_y = (ys_v * delta_visc).where(in_band, torch.zeros_like(ys_v))
        fv_z = (zs_v * delta_visc).where(in_band, torch.zeros_like(zs_v))
        fp_x = (pxv  * delta_pres).where(in_band, torch.zeros_like(pxv))
        fp_y = (pyv  * delta_pres).where(in_band, torch.zeros_like(pyv))
        fp_z = (pzv  * delta_pres).where(in_band, torch.zeros_like(pzv))

        arm_x = (xc - cm[0]).to(torch.float64)
        arm_y = (yc - cm[1]).to(torch.float64)
        arm_z = (zc - cm[2]).to(torch.float64)
        fv_x_d, fv_y_d, fv_z_d = fv_x.to(torch.float64), fv_y.to(torch.float64), fv_z.to(torch.float64)
        fp_x_d, fp_y_d, fp_z_d = fp_x.to(torch.float64), fp_y.to(torch.float64), fp_z.to(torch.float64)

        h3 = h_grid ** 3
        out_ref[b, 0]  += fv_x_d.sum().item() * h3
        out_ref[b, 1]  += fv_y_d.sum().item() * h3
        out_ref[b, 2]  += fv_z_d.sum().item() * h3
        out_ref[b, 3]  += (arm_y * fv_z_d - arm_z * fv_y_d).sum().item() * h3
        out_ref[b, 4]  += (arm_z * fv_x_d - arm_x * fv_z_d).sum().item() * h3
        out_ref[b, 5]  += (arm_x * fv_y_d - arm_y * fv_x_d).sum().item() * h3
        out_ref[b, 6]  += fp_x_d.sum().item() * h3
        out_ref[b, 7]  += fp_y_d.sum().item() * h3
        out_ref[b, 8]  += fp_z_d.sum().item() * h3
        out_ref[b, 9]  += (arm_y * fp_z_d - arm_z * fp_y_d).sum().item() * h3
        out_ref[b, 10] += (arm_z * fp_x_d - arm_x * fp_z_d).sum().item() * h3
        out_ref[b, 11] += (arm_x * fp_y_d - arm_y * fp_x_d).sum().item() * h3

    return winning_rho_ref, out_ref


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
    # Overlapping AABBs to exercise the running-min winning_rho_cc path.
    aabbs = [(2+b*4, 2+b*4+22, 3+b*3, 3+b*3+20, 2+b*2, 2+b*2+16) for b in range(B)]

    rho_bodies = torch.tensor([1.05, 1.10, 1.20], dtype=dtype, device=device)
    rho_fluid = 1.0

    # ---- pack op inputs ----
    F_chunks = [bd["F"].flatten() for bd in bodies]
    F_off = [0]
    shapes = []; metas = []; kin = []; lo = []; dim_ = []; max_vol = 0
    for bd, ab in zip(bodies, aabbs):
        F_off.append(F_off[-1] + bd["F"].numel())
        shapes.append([bd["Mx"], bd["My"], bd["Mz"]])
        metas.append(_axis_meta(bd))
        kin.append(bd["R_T"] + bd["bp"] + bd["cm"] + bd["lv"] + bd["av"])
        i0, i1, j0, j1, k0, k1 = ab
        lo.append([i0, j0, k0])
        dim_.append([i1-i0, j1-j0, k1-k0])
        max_vol = max(max_vol, (i1-i0)*(j1-j0)*(k1-k0))

    F_flat = torch.cat(F_chunks).contiguous()
    F_off_t = torch.tensor(F_off, dtype=torch.int64)
    shapes_t = torch.tensor(shapes, dtype=torch.int64)
    metas_t  = torch.tensor(metas,  dtype=dtype)
    kin_t    = torch.tensor(kin,    dtype=dtype)
    lo_t     = torch.tensor(lo,     dtype=torch.int64)
    dim_t    = torch.tensor(dim_,   dtype=torch.int64)

    # Smooth random fluid fields (full grid).
    gen = torch.Generator(device="cpu").manual_seed(123)
    u_prev = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.4
    v_prev = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.4
    w_prev = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.4
    p_prev = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.5
    nx_cc  = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.6
    ny_cc  = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.6
    nz_cc  = torch.randn(Nx, Ny, Nz, dtype=dtype, generator=gen) * 0.6

    # Constant ν·ρ first; then variable.
    eps_body = 2.5 * h
    eps_solver = 0.0
    h3 = h ** 3
    delta_order = 1
    interp_method = 0  # trilinear

    FAR = 1e4

    def _alloc_state():
        sdf_cc = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        sdf_u  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        sdf_v  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        sdf_w  = torch.full((Nx, Ny, Nz), FAR, dtype=dtype)
        bU = torch.zeros((Nx, Ny, Nz), dtype=dtype)
        bV = torch.zeros((Nx, Ny, Nz), dtype=dtype)
        bW = torch.zeros((Nx, Ny, Nz), dtype=dtype)
        winning_rho = torch.full((Nx, Ny, Nz), float(rho_fluid), dtype=dtype)
        out = torch.zeros((B, 12), dtype=torch.float64)
        return sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW, winning_rho, out

    def _run_and_check(label, nu_rho_field):
        sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW, winning_rho, out = _alloc_state()

        streaming_sdf_forces_fused_3d_multi(
            F_flat, F_off_t, shapes_t, metas_t, kin_t, lo_t, dim_t,
            gx, gy, gz, h, max_vol,
            sdf_cc, sdf_u, sdf_v, sdf_w, bU, bV, bW,
            interp_method, rho_bodies, winning_rho,
            u_prev, v_prev, w_prev, p_prev,
            nx_cc, ny_cc, nz_cc, nu_rho_field,
            eps_body, eps_solver, h3, delta_order, out,
        )

        rsdf_cc, rsdf_u, rsdf_v, rsdf_w, rbU, rbV, rbW, _ = _ref_streaming_sdf(
            bodies, aabbs, gx, gy, gz, h, dtype=dtype, device=device,
            interp_method=interp_method,
        )
        rwinning_rho, rout = _ref_winning_rho_and_forces(
            bodies, aabbs, rho_bodies, gx, gy, gz, h,
            u_prev, v_prev, w_prev, p_prev, nx_cc, ny_cc, nz_cc,
            nu_rho_field, eps_body, eps_solver, rho_fluid,
            dtype=dtype, device=device,
        )

        # Sanity: forces are non-degenerate.
        in_band_cells = ((sdf_cc.abs() < eps_body)).sum().item()
        print(f"[{label}] cells inside cosine-delta support: {in_band_cells}")
        assert in_band_cells > 100, "scene is degenerate"
        assert out.abs().max().item() > 1e-12, "all-zero forces — test is trivial"

        def cmp(name, kernel, ref, tol):
            diff = (kernel - ref).abs().max().item()
            norm = ref.abs().max().item()
            rel = diff / max(norm, 1e-30)
            print(f"  {name:<20}  max |Δ|={diff:.3e}  max |ref|={norm:.3e}  rel={rel:.3e}")
            assert rel < tol, f"{name}: rel={rel:.3e} > tol={tol:.3e}"

        cmp("sdf_cc",          sdf_cc,      rsdf_cc,       1e-12)
        cmp("sdf_u",           sdf_u,       rsdf_u,        1e-12)
        cmp("sdf_v",           sdf_v,       rsdf_v,        1e-12)
        cmp("sdf_w",           sdf_w,       rsdf_w,        1e-12)
        cmp("body_u",          bU,          rbU,           1e-12)
        cmp("body_v",          bV,          rbV,           1e-12)
        cmp("body_w",          bW,          rbW,           1e-12)
        cmp("winning_rho_cc",  winning_rho, rwinning_rho,  1e-15)
        cmp("out (forces)",    out,         rout,          1e-10)

    # constant ν·ρ
    _run_and_check("nu_rho=const",  torch.tensor([0.5], dtype=dtype))
    # variable ν·ρ (full-grid)
    nu_rho_var = (0.3 + 0.4 * torch.rand(Nx, Ny, Nz, dtype=dtype, generator=gen))
    _run_and_check("nu_rho=field",  nu_rho_var)

    print("OK: streaming_sdf_forces_fused_3d_multi matches the pure-PyTorch "
          "reference (sdf/face/winning_rho_cc/out) for both constant and "
          "variable nu_rho_field.")


if __name__ == "__main__":
    main()
