"""CPU self-test for ``lagrangian_forces_2d`` / ``lagrangian_forces_3d``.

Cross-checks the fused native kernels against a pure-PyTorch reference
that mirrors ``forces.forces_lagrangian_{2d,3d}`` (per-body Python
loop using ``RegularGridInterpolator``).
"""
import math
import sys

import torch

from lilytorch.src.kernels import lagrangian_forces_2d, lagrangian_forces_3d
from lilytorch.src.kernels import RegularGridInterpolator


def _ref_lagrangian_2d(eps_xx, eps_xy, eps_yy, p, nu_rho,
                       cnt_flat, cnt_offsets, com_pos, axes):
    """Reference per-body loop in pure PyTorch (matches forces_lagrangian_2d)."""
    B = com_pos.shape[0]
    out = torch.zeros(B, 6, dtype=torch.float64)
    interp = RegularGridInterpolator(axes, p, method="linear")

    def _sample(F, qx, qy):
        interp.F = F.contiguous()
        return interp(qx, qy)

    cnt_x = cnt_flat[0]; cnt_y = cnt_flat[1]
    for b in range(B):
        i0, i1 = int(cnt_offsets[b]), int(cnt_offsets[b + 1])
        qx = cnt_x[i0:i1]; qy = cnt_y[i0:i1]
        if qx.numel() <= 1: continue
        tx = (torch.roll(qx, -1, 0) - torch.roll(qx, 1, 0)) * 0.5
        ty = (torch.roll(qy, -1, 0) - torch.roll(qy, 1, 0)) * 0.5
        L = torch.sqrt(tx * tx + ty * ty).clamp_min(1e-30)
        tx /= L; ty /= L
        nx = ty; ny = -tx
        e_xx = _sample(eps_xx, qx, qy)
        e_xy = _sample(eps_xy, qx, qy)
        e_yy = _sample(eps_yy, qx, qy)
        if nu_rho.numel() == 1:
            nrm = nu_rho.item()
        else:
            nrm = _sample(nu_rho, qx, qy)
        tvx = nrm * (e_xx * nx + e_xy * ny)
        tvy = nrm * (e_xy * nx + e_yy * ny)
        p_m = _sample(p, qx, qy)
        tpx = -p_m * nx; tpy = -p_m * ny

        dx = torch.roll(qx, -1, 0) - qx
        dy = torch.roll(qy, -1, 0) - qy
        ds_seg = torch.sqrt(dx * dx + dy * dy)
        def _li(f):
            return 0.5 * ((f + torch.roll(f, -1, 0)) * ds_seg).sum()
        fv_x = _li(tvx); fv_y = _li(tvy)
        fp_x = _li(tpx); fp_y = _li(tpy)
        com = com_pos[b]
        rx = qx - com[0]; ry = qy - com[1]
        tv_t = _li(rx * tvy - ry * tvx)
        tp_t = _li(rx * tpy - ry * tpx)
        out[b, 0] = fv_x.to(torch.float64)
        out[b, 1] = fv_y.to(torch.float64)
        out[b, 2] = tv_t.to(torch.float64)
        out[b, 3] = fp_x.to(torch.float64)
        out[b, 4] = fp_y.to(torch.float64)
        out[b, 5] = tp_t.to(torch.float64)
    return out


def _ref_lagrangian_3d(eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
                       p, nu_rho,
                       tri_centroid, tri_normal, tri_area,
                       tri_offsets, com_pos, axes):
    B = com_pos.shape[0]
    out = torch.zeros(B, 12, dtype=torch.float64)
    interp = RegularGridInterpolator(axes, p, method="linear")

    def _sample(F, qx, qy, qz):
        interp.F = F.contiguous()
        return interp(qx, qy, qz)

    for b in range(B):
        t0, t1 = int(tri_offsets[b]), int(tri_offsets[b + 1])
        if t1 <= t0: continue
        qx = tri_centroid[0, t0:t1]
        qy = tri_centroid[1, t0:t1]
        qz = tri_centroid[2, t0:t1]
        nx = tri_normal[0, t0:t1]
        ny = tri_normal[1, t0:t1]
        nz = tri_normal[2, t0:t1]
        A  = tri_area[t0:t1]

        e_xx = _sample(eps_xx, qx, qy, qz)
        e_yy = _sample(eps_yy, qx, qy, qz)
        e_zz = _sample(eps_zz, qx, qy, qz)
        e_xy = _sample(eps_xy, qx, qy, qz)
        e_xz = _sample(eps_xz, qx, qy, qz)
        e_yz = _sample(eps_yz, qx, qy, qz)
        if nu_rho.numel() == 1:
            nrm = nu_rho.item()
        else:
            nrm = _sample(nu_rho, qx, qy, qz)
        tvx = nrm * (e_xx * nx + e_xy * ny + e_xz * nz)
        tvy = nrm * (e_xy * nx + e_yy * ny + e_yz * nz)
        tvz = nrm * (e_xz * nx + e_yz * ny + e_zz * nz)
        p_m = _sample(p, qx, qy, qz)
        tpx = -p_m * nx; tpy = -p_m * ny; tpz = -p_m * nz
        com = com_pos[b]
        rx = qx - com[0]; ry = qy - com[1]; rz = qz - com[2]

        out[b, 0] = (tvx * A).sum().to(torch.float64)
        out[b, 1] = (tvy * A).sum().to(torch.float64)
        out[b, 2] = (tvz * A).sum().to(torch.float64)
        out[b, 3] = ((ry * tvz - rz * tvy) * A).sum().to(torch.float64)
        out[b, 4] = ((rz * tvx - rx * tvz) * A).sum().to(torch.float64)
        out[b, 5] = ((rx * tvy - ry * tvx) * A).sum().to(torch.float64)
        out[b, 6] = (tpx * A).sum().to(torch.float64)
        out[b, 7] = (tpy * A).sum().to(torch.float64)
        out[b, 8] = (tpz * A).sum().to(torch.float64)
        out[b, 9] = ((ry * tpz - rz * tpy) * A).sum().to(torch.float64)
        out[b, 10] = ((rz * tpx - rx * tpz) * A).sum().to(torch.float64)
        out[b, 11] = ((rx * tpy - ry * tpx) * A).sum().to(torch.float64)
    return out


def _build_circle(cx, cy, R, M):
    th = torch.linspace(0, 2 * math.pi, M + 1, dtype=torch.float64)[:-1]
    return torch.stack([cx + R * torch.cos(th), cy + R * torch.sin(th)], dim=0)


def test_2d():
    torch.manual_seed(0)
    Mx, My = 64, 48
    h = 0.05
    x = torch.arange(Mx, dtype=torch.float64) * h
    y = torch.arange(My, dtype=torch.float64) * h
    # Random smooth fields
    eps_xx = torch.randn(Mx, My, dtype=torch.float64)
    eps_xy = torch.randn(Mx, My, dtype=torch.float64)
    eps_yy = torch.randn(Mx, My, dtype=torch.float64)
    p      = torch.randn(Mx, My, dtype=torch.float64)
    nu_rho = torch.tensor([0.7], dtype=torch.float64)

    bodies = [
        _build_circle(1.5, 0.9, 0.3, 64),
        _build_circle(2.1, 1.4, 0.25, 48),
        _build_circle(0.8, 1.7, 0.2, 32),
    ]
    cnt_flat = torch.cat(bodies, dim=1)
    cnt_offsets = torch.tensor(
        [0, bodies[0].shape[1],
         bodies[0].shape[1] + bodies[1].shape[1],
         cnt_flat.shape[1]], dtype=torch.int64)
    com_pos = torch.tensor(
        [[1.5, 0.9], [2.1, 1.4], [0.8, 1.7]], dtype=torch.float64)

    ref = _ref_lagrangian_2d(
        eps_xx, eps_xy, eps_yy, p, nu_rho,
        cnt_flat, cnt_offsets, com_pos, (x, y))

    out = lagrangian_forces_2d(
        eps_xx, eps_xy, eps_yy, p, nu_rho,
        cnt_flat, cnt_offsets, com_pos,
        float(x[0]), float(y[0]),
        1.0 / h, 1.0 / h,
        Mx, My, method="linear")

    err = (out - ref).abs().max().item()
    print(f"[2d/scalar nu_rho] max abs error: {err:.3e}")
    assert err < 1e-12, f"2-D fused kernel mismatch: {err}"

    # Repeat with full-grid nu_rho
    nu_rho_grid = torch.randn(Mx, My, dtype=torch.float64).abs() + 0.1
    ref2 = _ref_lagrangian_2d(
        eps_xx, eps_xy, eps_yy, p, nu_rho_grid,
        cnt_flat, cnt_offsets, com_pos, (x, y))
    out2 = lagrangian_forces_2d(
        eps_xx, eps_xy, eps_yy, p, nu_rho_grid,
        cnt_flat, cnt_offsets, com_pos,
        float(x[0]), float(y[0]),
        1.0 / h, 1.0 / h,
        Mx, My, method="linear")
    err2 = (out2 - ref2).abs().max().item()
    print(f"[2d/field  nu_rho] max abs error: {err2:.3e}")
    assert err2 < 1e-12

    # Biquadratic
    out3 = lagrangian_forces_2d(
        eps_xx, eps_xy, eps_yy, p, nu_rho,
        cnt_flat, cnt_offsets, com_pos,
        float(x[0]), float(y[0]),
        1.0 / h, 1.0 / h,
        Mx, My, method="quadratic")
    # Different reference because RegularGridInterpolator linear vs kernel quadratic
    # — just sanity-check shape & finiteness.
    assert out3.shape == (3, 6) and torch.isfinite(out3).all()
    print(f"[2d/quadratic] sample out[0]: {out3[0].tolist()}")


def _build_sphere_tris(cx, cy, cz, R, lon, lat):
    # Simple uv-sphere triangulation; returns (centroids, normals, areas)
    lons = torch.linspace(0, 2 * math.pi, lon + 1, dtype=torch.float64)[:-1]
    lats = torch.linspace(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, lat,
                          dtype=torch.float64)
    verts = []
    for la in lats:
        for lo in lons:
            verts.append((cx + R * math.cos(la) * math.cos(lo),
                          cy + R * math.cos(la) * math.sin(lo),
                          cz + R * math.sin(la)))
    verts = torch.tensor(verts, dtype=torch.float64)  # (lat*lon, 3)
    tris = []
    for j in range(lat - 1):
        for i in range(lon):
            ip = (i + 1) % lon
            v0 = j * lon + i
            v1 = j * lon + ip
            v2 = (j + 1) * lon + i
            v3 = (j + 1) * lon + ip
            tris.append((v0, v2, v3))
            tris.append((v0, v3, v1))
    tris = torch.tensor(tris, dtype=torch.long)
    v = verts[tris]   # (T, 3, 3)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    cross = torch.cross(e1, e2, dim=1)
    area = 0.5 * torch.norm(cross, dim=1)
    n = cross / (2 * area.unsqueeze(1)).clamp_min(1e-30)
    centroid = v.mean(dim=1)
    # Outward orientation: sphere centroid is "outside" along its (centroid-com)
    radial = centroid - torch.tensor([cx, cy, cz], dtype=torch.float64)
    sign = torch.sign((n * radial).sum(dim=1)).clamp_min(-1).clamp_max(1)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    n = n * sign.unsqueeze(1)
    return centroid.T.contiguous(), n.T.contiguous(), area.contiguous()


def test_3d():
    torch.manual_seed(1)
    Mx, My, Mz = 24, 22, 20
    h = 0.1
    x = torch.arange(Mx, dtype=torch.float64) * h
    y = torch.arange(My, dtype=torch.float64) * h
    z = torch.arange(Mz, dtype=torch.float64) * h
    exx = torch.randn(Mx, My, Mz, dtype=torch.float64)
    eyy = torch.randn(Mx, My, Mz, dtype=torch.float64)
    ezz = torch.randn(Mx, My, Mz, dtype=torch.float64)
    exy = torch.randn(Mx, My, Mz, dtype=torch.float64)
    exz = torch.randn(Mx, My, Mz, dtype=torch.float64)
    eyz = torch.randn(Mx, My, Mz, dtype=torch.float64)
    p   = torch.randn(Mx, My, Mz, dtype=torch.float64)
    nu_rho = torch.tensor([0.4], dtype=torch.float64)

    s1 = _build_sphere_tris(1.0, 0.9, 0.8, 0.3, 12, 7)
    s2 = _build_sphere_tris(1.5, 1.1, 1.0, 0.25, 10, 6)
    centroid = torch.cat([s1[0], s2[0]], dim=1)
    normal   = torch.cat([s1[1], s2[1]], dim=1)
    area     = torch.cat([s1[2], s2[2]], dim=0)
    offsets  = torch.tensor(
        [0, s1[0].shape[1], s1[0].shape[1] + s2[0].shape[1]],
        dtype=torch.int64)
    com_pos  = torch.tensor(
        [[1.0, 0.9, 0.8], [1.5, 1.1, 1.0]], dtype=torch.float64)

    ref = _ref_lagrangian_3d(
        exx, eyy, ezz, exy, exz, eyz, p, nu_rho,
        centroid, normal, area, offsets, com_pos, (x, y, z))
    out = lagrangian_forces_3d(
        exx, eyy, ezz, exy, exz, eyz, p, nu_rho,
        centroid, normal, area, offsets, com_pos,
        float(x[0]), float(y[0]), float(z[0]),
        1.0 / h, 1.0 / h, 1.0 / h,
        Mx, My, Mz, method="linear")
    err = (out - ref).abs().max().item()
    print(f"[3d/scalar nu_rho] max abs error: {err:.3e}")
    assert err < 1e-12, f"3-D fused kernel mismatch: {err}"

    nu_rho_grid = torch.randn(Mx, My, Mz, dtype=torch.float64).abs() + 0.1
    ref2 = _ref_lagrangian_3d(
        exx, eyy, ezz, exy, exz, eyz, p, nu_rho_grid,
        centroid, normal, area, offsets, com_pos, (x, y, z))
    out2 = lagrangian_forces_3d(
        exx, eyy, ezz, exy, exz, eyz, p, nu_rho_grid,
        centroid, normal, area, offsets, com_pos,
        float(x[0]), float(y[0]), float(z[0]),
        1.0 / h, 1.0 / h, 1.0 / h,
        Mx, My, Mz, method="linear")
    err2 = (out2 - ref2).abs().max().item()
    print(f"[3d/field  nu_rho] max abs error: {err2:.3e}")
    assert err2 < 1e-12


if __name__ == "__main__":
    test_2d()
    test_3d()
    print("OK")
