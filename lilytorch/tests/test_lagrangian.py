"""Warp Lagrangian surface-force single-source checks: Warp CPU == Warp GPU.

Exercises ``lagrangian_forces_{2d,3d}_warp`` on synthetic scenes (circles in
2-D, uv-sphere triangulations in 3-D): scalar + full-field nu_rho, linear +
quadratic sampling, a nonzero ``sample_offset``, and f32 + f64.

Every problem is built ONCE on CPU and ``.to(cuda)``-ed — torch's per-device
generators yield different sequences for the same seed (HANDOFF lesson 5), so
building twice would compare two different problems.

Run:  pytest lilytorch/src/kernels/test_lagrangian.py -v
      python -m lilytorch.src.test_lagrangian
"""
from __future__ import annotations

import math

import pytest
import torch

from lilytorch.src.lagrangian import (
    lagrangian_forces_2d_warp,
    lagrangian_forces_3d_warp,
)
from lilytorch.src.native import (
    lagrangian_forces_2d as lagrangian_forces_2d_native,
    lagrangian_forces_3d as lagrangian_forces_3d_native,
)

SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
# atomicAdd accumulation order differs between devices → float64 reduction
# noise; same tolerance the retired native self-test used (1e-12), relaxed
# slightly for the GPU's much wider concurrent atomic fan-in.
ATOL = 1e-9
# float32: per-element products are computed in f32 before the double
# atomicAdd, so the drift is single-precision rounding / FMA contraction
# differences between the CPU and CUDA codegen — ~1e-5 relative here.
RTOL_F32 = 2e-4
ATOL_F32 = 1e-4


def _err(a, b, dtype):
    """Max abs error, plus a pass/fail for the dtype's tolerance."""
    e = (a - b).abs().max().item()
    if dtype == torch.float32:
        scale = b.abs().max().item()
        return e, e <= ATOL_F32 + RTOL_F32 * scale
    return e, e < ATOL


# ─── scene builders ──────────────────────────────────────────────────────────

def _build_circle(cx, cy, R, M):
    th = torch.linspace(0, 2 * math.pi, M + 1, dtype=torch.float64)[:-1]
    return torch.stack([cx + R * torch.cos(th), cy + R * torch.sin(th)], dim=0)


def _build_sphere_tris(cx, cy, cz, R, lon, lat):
    lons = torch.linspace(0, 2 * math.pi, lon + 1, dtype=torch.float64)[:-1]
    lats = torch.linspace(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, lat,
                          dtype=torch.float64)
    verts = []
    for la in lats:
        for lo in lons:
            verts.append((cx + R * math.cos(la) * math.cos(lo),
                          cy + R * math.cos(la) * math.sin(lo),
                          cz + R * math.sin(la)))
    verts = torch.tensor(verts, dtype=torch.float64)
    tris = []
    for j in range(lat - 1):
        for i in range(lon):
            ip = (i + 1) % lon
            v0 = j * lon + i; v1 = j * lon + ip
            v2 = (j + 1) * lon + i; v3 = (j + 1) * lon + ip
            tris.append((v0, v2, v3)); tris.append((v0, v3, v1))
    tris = torch.tensor(tris, dtype=torch.long)
    v = verts[tris]
    e1 = v[:, 1] - v[:, 0]; e2 = v[:, 2] - v[:, 0]
    cross = torch.cross(e1, e2, dim=1)
    area = 0.5 * torch.norm(cross, dim=1)
    n = cross / (2 * area.unsqueeze(1)).clamp_min(1e-30)
    centroid = v.mean(dim=1)
    radial = centroid - torch.tensor([cx, cy, cz], dtype=torch.float64)
    sign = torch.sign((n * radial).sum(dim=1)).clamp_min(-1).clamp_max(1)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    n = n * sign.unsqueeze(1)
    return centroid.T.contiguous(), n.T.contiguous(), area.contiguous()


def _scene_2d():
    torch.manual_seed(0)
    Mx, My, h = 64, 48, 0.05
    fields = {k: torch.randn(Mx, My, dtype=torch.float64)
              for k in ("exx", "exy", "eyy", "p")}
    bodies = [_build_circle(1.5, 0.9, 0.3, 64),
              _build_circle(2.1, 1.4, 0.25, 48),
              _build_circle(0.8, 1.7, 0.2, 32)]
    cnt_flat = torch.cat(bodies, dim=1)
    sizes = [b.shape[1] for b in bodies]
    cnt_offsets = torch.tensor(
        [0, sizes[0], sizes[0] + sizes[1], sum(sizes)], dtype=torch.int64)
    com_pos = torch.tensor([[1.5, 0.9], [2.1, 1.4], [0.8, 1.7]],
                           dtype=torch.float64)
    return Mx, My, h, fields, cnt_flat, cnt_offsets, com_pos


def _scene_3d():
    torch.manual_seed(1)
    Mx, My, Mz, h = 24, 22, 20, 0.1
    fields = {k: torch.randn(Mx, My, Mz, dtype=torch.float64)
              for k in ("exx", "eyy", "ezz", "exy", "exz", "eyz", "p")}
    s1 = _build_sphere_tris(1.0, 0.9, 0.8, 0.3, 12, 7)
    s2 = _build_sphere_tris(1.5, 1.1, 1.0, 0.25, 10, 6)
    centroid = torch.cat([s1[0], s2[0]], dim=1)
    normal = torch.cat([s1[1], s2[1]], dim=1)
    area = torch.cat([s1[2], s2[2]], dim=0)
    offs = torch.tensor([0, s1[0].shape[1], s1[0].shape[1] + s2[0].shape[1]],
                        dtype=torch.int64)
    com = torch.tensor([[1.0, 0.9, 0.8], [1.5, 1.1, 1.0]], dtype=torch.float64)
    return Mx, My, Mz, h, fields, centroid, normal, area, offs, com


# ─── runners (args built on CPU; `dev` moves the tensor inputs) ──────────────

def _args_2d(scalar_nrho, dtype):
    Mx, My, h, F, cnt, offs, com = _scene_2d()
    if scalar_nrho:
        nrho = torch.tensor([0.7], dtype=torch.float64)
    else:
        nrho = torch.randn(Mx, My, dtype=torch.float64).abs() + 0.1
    F = {k: v.to(dtype) for k, v in F.items()}
    nrho = nrho.to(dtype); cnt = cnt.to(dtype); com = com.to(dtype)
    tensors = (F["exx"], F["exy"], F["eyy"], F["p"], nrho, cnt, offs, com)
    geom = (0.0, 0.0, 1.0 / h, 1.0 / h, Mx, My)
    return tensors, geom


def _args_3d(scalar_nrho, dtype):
    Mx, My, Mz, h, F, c, n_, a, offs, com = _scene_3d()
    if scalar_nrho:
        nrho = torch.tensor([0.4], dtype=torch.float64)
    else:
        nrho = torch.randn(Mx, My, Mz, dtype=torch.float64).abs() + 0.1
    F = {k: v.to(dtype) for k, v in F.items()}
    nrho = nrho.to(dtype); c = c.to(dtype); n_ = n_.to(dtype)
    a = a.to(dtype); com = com.to(dtype)
    tensors = (F["exx"], F["eyy"], F["ezz"], F["exy"], F["exz"], F["eyz"],
               F["p"], nrho, c, n_, a, offs, com)
    geom = (0.0, 0.0, 0.0, 1.0 / h, 1.0 / h, 1.0 / h, Mx, My, Mz)
    return tensors, geom


def _cpu_vs_gpu(fn, tensors, geom, method, offset):
    wc = fn(*tensors, *geom, method=method, sample_offset=offset)
    gpu = tuple(t.cuda() for t in tensors)
    wg = fn(*gpu, *geom, method=method, sample_offset=offset)
    return wc.cpu(), wg.cpu()


# ─── tests ───────────────────────────────────────────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
@pytest.mark.parametrize("offset", [0.0, 0.08])
def test_2d_cpu_eq_gpu(method, scalar_nrho, offset):
    tensors, geom = _args_2d(scalar_nrho, torch.float64)
    wc, wg = _cpu_vs_gpu(lagrangian_forces_2d_warp, tensors, geom, method, offset)
    err, ok = _err(wc, wg, torch.float64)
    assert ok, f"2D {method} scalar={scalar_nrho} off={offset}: {err:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_3d_cpu_eq_gpu(method, scalar_nrho, offset):
    tensors, geom = _args_3d(scalar_nrho, torch.float64)
    wc, wg = _cpu_vs_gpu(lagrangian_forces_3d_warp, tensors, geom, method, offset)
    err, ok = _err(wc, wg, torch.float64)
    assert ok, f"3D {method} scalar={scalar_nrho} off={offset}: {err:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_2d_cpu_eq_gpu_f32(method, scalar_nrho):
    tensors, geom = _args_2d(scalar_nrho, torch.float32)
    wc, wg = _cpu_vs_gpu(lagrangian_forces_2d_warp, tensors, geom, method, 0.08)
    err, ok = _err(wc, wg, torch.float32)
    assert ok, f"2D f32 {method} scalar={scalar_nrho}: {err:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_3d_cpu_eq_gpu_f32(method, scalar_nrho):
    tensors, geom = _args_3d(scalar_nrho, torch.float32)
    wc, wg = _cpu_vs_gpu(lagrangian_forces_3d_warp, tensors, geom, method, 0.2)
    err, ok = _err(wc, wg, torch.float32)
    assert ok, f"3D f32 {method} scalar={scalar_nrho}: {err:.3e}"


# ─── cuda_native_port Phase 0.2 parity gate: native == Warp oracle ───────────

_DEVS = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", _DEVS)
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
@pytest.mark.parametrize("offset", [0.0, 0.08])
def test_2d_native_eq_warp(dev, method, scalar_nrho, offset):
    tensors, geom = _args_2d(scalar_nrho, torch.float64)
    tensors = tuple(t.to(dev) for t in tensors)
    w = lagrangian_forces_2d_warp(*tensors, *geom, method=method, sample_offset=offset)
    n = lagrangian_forces_2d_native(*tensors, *geom, method=method, sample_offset=offset)
    err, ok = _err(w.cpu(), n.cpu(), torch.float64)
    assert ok, f"2D native vs warp {dev} {method} scalar={scalar_nrho} off={offset}: {err:.3e}"


@pytest.mark.parametrize("dev", _DEVS)
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_3d_native_eq_warp(dev, method, scalar_nrho, offset):
    tensors, geom = _args_3d(scalar_nrho, torch.float64)
    tensors = tuple(t.to(dev) for t in tensors)
    w = lagrangian_forces_3d_warp(*tensors, *geom, method=method, sample_offset=offset)
    n = lagrangian_forces_3d_native(*tensors, *geom, method=method, sample_offset=offset)
    err, ok = _err(w.cpu(), n.cpu(), torch.float64)
    assert ok, f"3D native vs warp {dev} {method} scalar={scalar_nrho} off={offset}: {err:.3e}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("  no CUDA — nothing to compare")
    else:
        for method in ("linear", "quadratic"):
            for sc in (True, False):
                t2, g2 = _args_2d(sc, torch.float64)
                wc, wg = _cpu_vs_gpu(lagrangian_forces_2d_warp, t2, g2, method, 0.08)
                e2 = (wc - wg).abs().max().item()
                t3, g3 = _args_3d(sc, torch.float64)
                wc, wg = _cpu_vs_gpu(lagrangian_forces_3d_warp, t3, g3, method, 0.2)
                e3 = (wc - wg).abs().max().item()
                tag = f"{method:9s} scalar={sc!s:5s}"
                print(f"  {tag}  2D {e2:.2e}  3D {e3:.2e}  "
                      f"{'PASS' if max(e2, e3) < ATOL else 'FAIL'}")
