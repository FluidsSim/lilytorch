"""Parity tests: Warp Lagrangian surface-force kernels vs native CUDA/CPP.

Validates ``lagrangian_forces_{2d,3d}_warp`` against the native ops (the parity
oracle) on the same synthetic scenes as the native self-test
(``src/kernels/test_lagrangian_forces_self.py``): circles in 2-D, uv-sphere
triangulations in 3-D, scalar + full-field nu_rho, linear + quadratic sampling,
and a nonzero ``sample_offset``.  Also checks Warp CPU == Warp GPU.

Run:  pytest lilytorch/warp_poc/test_lagrangian.py -v
      python -m lilytorch.src.kernels.test_lagrangian
"""
from __future__ import annotations

import math

import pytest
import torch

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        lagrangian_forces_2d as native_2d,
        lagrangian_forces_3d as native_3d,
    )
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.lagrangian import (
    lagrangian_forces_2d_warp,
    lagrangian_forces_3d_warp,
)

SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
# atomicAdd accumulation order differs from the deterministic native sum →
# float64 reduction noise; same tolerance the native self-test uses (1e-12),
# relaxed slightly for the GPU's much wider concurrent atomic fan-in.
ATOL = 1e-9
# float32: per-element products are computed in f32 (matching native's
# scalar_t=float dispatch) before the double atomicAdd, so the only drift is
# single-precision rounding / FMA contraction differences between the Warp
# codegen and the native nvcc codegen — ~1e-5 relative on these scenes.
RTOL_F32 = 2e-4
ATOL_F32 = 1e-4


def _err(w, n, dtype):
    """Max abs error, plus a relative scale for the f32 tolerance check."""
    a = (w - n).abs().max().item()
    if dtype == torch.float32:
        scale = n.abs().max().item()
        return a, a <= ATOL_F32 + RTOL_F32 * scale
    return a, a < ATOL


# ─── scene builders (mirror src/kernels/test_lagrangian_forces_self.py) ──────

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


def _scene_2d(dev):
    torch.manual_seed(0)
    Mx, My, h = 64, 48, 0.05
    fields = {k: torch.randn(Mx, My, dtype=torch.float64, device=dev)
              for k in ("exx", "exy", "eyy", "p")}
    bodies = [_build_circle(1.5, 0.9, 0.3, 64),
              _build_circle(2.1, 1.4, 0.25, 48),
              _build_circle(0.8, 1.7, 0.2, 32)]
    cnt_flat = torch.cat(bodies, dim=1).to(dev)
    sizes = [b.shape[1] for b in bodies]
    cnt_offsets = torch.tensor(
        [0, sizes[0], sizes[0] + sizes[1], sum(sizes)],
        dtype=torch.int64, device=dev)
    com_pos = torch.tensor([[1.5, 0.9], [2.1, 1.4], [0.8, 1.7]],
                           dtype=torch.float64, device=dev)
    return Mx, My, h, fields, cnt_flat, cnt_offsets, com_pos


def _scene_3d(dev):
    torch.manual_seed(1)
    Mx, My, Mz, h = 24, 22, 20, 0.1
    fields = {k: torch.randn(Mx, My, Mz, dtype=torch.float64, device=dev)
              for k in ("exx", "eyy", "ezz", "exy", "exz", "eyz", "p")}
    s1 = _build_sphere_tris(1.0, 0.9, 0.8, 0.3, 12, 7)
    s2 = _build_sphere_tris(1.5, 1.1, 1.0, 0.25, 10, 6)
    centroid = torch.cat([s1[0], s2[0]], dim=1).to(dev)
    normal = torch.cat([s1[1], s2[1]], dim=1).to(dev)
    area = torch.cat([s1[2], s2[2]], dim=0).to(dev)
    offs = torch.tensor([0, s1[0].shape[1], s1[0].shape[1] + s2[0].shape[1]],
                        dtype=torch.int64, device=dev)
    com = torch.tensor([[1.0, 0.9, 0.8], [1.5, 1.1, 1.0]],
                       dtype=torch.float64, device=dev)
    return Mx, My, Mz, h, fields, centroid, normal, area, offs, com


# ─── runners ──────────────────────────────────────────────────────────────

def _run_2d(dev, method, scalar_nrho, offset, dtype=torch.float64):
    Mx, My, h, F, cnt, offs, com = _scene_2d(dev)
    if scalar_nrho:
        nrho = torch.tensor([0.7], dtype=torch.float64, device=dev)
    else:
        nrho = torch.randn(Mx, My, dtype=torch.float64, device=dev).abs() + 0.1
    # Cast the field/geometry inputs to the working dtype (native dispatches on
    # ``p.scalar_type()``, so f32 inputs exercise the f32 kernel both sides).
    F = {k: v.to(dtype) for k, v in F.items()}
    nrho = nrho.to(dtype); cnt = cnt.to(dtype); com = com.to(dtype)
    args = (F["exx"], F["exy"], F["eyy"], F["p"], nrho,
            cnt, offs, com, 0.0, 0.0, 1.0 / h, 1.0 / h, Mx, My)
    w = lagrangian_forces_2d_warp(*args, method=method, sample_offset=offset)
    n = native_2d(*args, method=method, sample_offset=offset)
    return w.cpu(), n.cpu()


def _run_3d(dev, method, scalar_nrho, offset, dtype=torch.float64):
    Mx, My, Mz, h, F, c, n_, a, offs, com = _scene_3d(dev)
    if scalar_nrho:
        nrho = torch.tensor([0.4], dtype=torch.float64, device=dev)
    else:
        nrho = torch.randn(Mx, My, Mz, dtype=torch.float64, device=dev).abs() + 0.1
    F = {k: v.to(dtype) for k, v in F.items()}
    nrho = nrho.to(dtype); c = c.to(dtype); n_ = n_.to(dtype)
    a = a.to(dtype); com = com.to(dtype)
    args = (F["exx"], F["eyy"], F["ezz"], F["exy"], F["exz"], F["eyz"],
            F["p"], nrho, c, n_, a, offs, com,
            0.0, 0.0, 0.0, 1.0 / h, 1.0 / h, 1.0 / h, Mx, My, Mz)
    w = lagrangian_forces_3d_warp(*args, method=method, sample_offset=offset)
    nat = native_3d(*args, method=method, sample_offset=offset)
    return w.cpu(), nat.cpu()


# ─── tests ───────────────────────────────────────────────────────────────

@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
@pytest.mark.parametrize("offset", [0.0, 0.08])
def test_2d_cpu_parity(method, scalar_nrho, offset):
    w, n = _run_2d("cpu", method, scalar_nrho, offset)
    err = (w - n).abs().max().item()
    assert err < ATOL, f"2D cpu {method} scalar={scalar_nrho} off={offset}: {err:.3e}"


@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_3d_cpu_parity(method, scalar_nrho, offset):
    w, n = _run_3d("cpu", method, scalar_nrho, offset)
    err = (w - n).abs().max().item()
    assert err < ATOL, f"3D cpu {method} scalar={scalar_nrho} off={offset}: {err:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_2d_gpu_parity(method):
    w, n = _run_2d("cuda:0", method, False, 0.08)
    err = (w - n).abs().max().item()
    assert err < ATOL, f"2D gpu {method}: {err:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_3d_gpu_parity(method):
    w, n = _run_3d("cuda:0", method, False, 0.2)
    err = (w - n).abs().max().item()
    assert err < ATOL, f"3D gpu {method}: {err:.3e}"


# ─── float32 parity (dtype-generic kernel; native dispatches f32) ────────────

@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_2d_cpu_parity_f32(method, scalar_nrho):
    w, n = _run_2d("cpu", method, scalar_nrho, 0.08, dtype=torch.float32)
    err, ok = _err(w, n, torch.float32)
    assert ok, f"2D cpu f32 {method} scalar={scalar_nrho}: {err:.3e}"


@SKIP_NO_NATIVE
@pytest.mark.parametrize("method", ["linear", "quadratic"])
@pytest.mark.parametrize("scalar_nrho", [True, False])
def test_3d_cpu_parity_f32(method, scalar_nrho):
    w, n = _run_3d("cpu", method, scalar_nrho, 0.2, dtype=torch.float32)
    err, ok = _err(w, n, torch.float32)
    assert ok, f"3D cpu f32 {method} scalar={scalar_nrho}: {err:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_2d_gpu_parity_f32(method):
    w, n = _run_2d("cuda:0", method, False, 0.08, dtype=torch.float32)
    err, ok = _err(w, n, torch.float32)
    assert ok, f"2D gpu f32 {method}: {err:.3e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("method", ["linear", "quadratic"])
def test_3d_gpu_parity_f32(method):
    w, n = _run_3d("cuda:0", method, False, 0.2, dtype=torch.float32)
    err, ok = _err(w, n, torch.float32)
    assert ok, f"3D gpu f32 {method}: {err:.3e}"


@SKIP_NO_CUDA
def test_cpu_eq_gpu():
    """Single Warp source: CPU and GPU agree to float64 reduction noise.

    Build the problem ONCE on CPU and ``.to(cuda)`` it — torch's per-device
    generators yield different sequences for the same seed (HANDOFF lesson 5),
    so building twice would compare two different problems.
    """
    Mx, My, Mz, h, F, c, n_, a, offs, com = _scene_3d("cpu")
    nrho = torch.randn(Mx, My, Mz, dtype=torch.float64).abs() + 0.1
    base = (F["exx"], F["eyy"], F["ezz"], F["exy"], F["exz"], F["eyz"],
            F["p"], nrho, c, n_, a, offs, com)
    geom = (0.0, 0.0, 0.0, 1.0 / h, 1.0 / h, 1.0 / h, Mx, My, Mz)

    wc = lagrangian_forces_3d_warp(*base, *geom, method="linear", sample_offset=0.2)
    gpu = tuple(t.cuda() for t in base)
    wg = lagrangian_forces_3d_warp(*gpu, *geom, method="linear", sample_offset=0.2)
    err = (wc.cpu() - wg.cpu()).abs().max().item()
    assert err < ATOL, f"warp cpu vs gpu: {err:.3e}"


if __name__ == "__main__":
    devs = (["cuda:0"] if torch.cuda.is_available() else []) + ["cpu"]
    for dev in devs:
        for method in ("linear", "quadratic"):
            for sc in (True, False):
                w, n = _run_2d(dev, method, sc, 0.08)
                e2 = (w - n).abs().max().item()
                w, n = _run_3d(dev, method, sc, 0.2)
                e3 = (w - n).abs().max().item()
                tag = f"{dev:7s} {method:9s} scalar={sc!s:5s}"
                print(f"  {tag}  2D {e2:.2e}  3D {e3:.2e}  "
                      f"{'PASS' if max(e2, e3) < ATOL else 'FAIL'}")
