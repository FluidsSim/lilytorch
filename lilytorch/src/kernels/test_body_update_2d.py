"""Parity: 2-D Warp streaming SDF vs native body_update_2d.

Covers fanned + sequential designs, blend-eps off/on, interp_method 0/1, on
CPU and GPU.  Mirrors `test_parity.py` (3-D) with the z axis stripped.
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import body_update_2d
    _NATIVE = True
except Exception:
    _NATIVE = False

from lilytorch.src.kernels.scene_2d import make_synthetic_scene_2d
from lilytorch.src.kernels.streaming_sdf_2d import WarpStreamingSDF2D

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SKIP_NO_NATIVE = pytest.mark.skipif(not _NATIVE, reason="native _C.so unavailable")
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _run_native(sc, interp):
    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    sc["sdf_cc"].fill_(1e4); sc["sdf_u"].fill_(1e4); sc["sdf_v"].fill_(1e4)
    sc["body_u"].zero_(); sc["body_v"].zero_()
    sc["num_u"].zero_(); sc["num_v"].zero_(); sc["den_u"].zero_(); sc["den_v"].zero_()
    di0, dj0, dAi, dAj = sc["dirty_bounds"]
    body_update_2d(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
        sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
        float(sc["h"]), int(sc["max_vol"]),
        sc["sdf_cc"].view(Ngx, Ngy), sc["sdf_u"].view(Ngx, Ngy), sc["sdf_v"].view(Ngx, Ngy),
        sc["body_u"].view(Ngx, Ngy), sc["body_v"].view(Ngx, Ngy),
        sc["key_cc"], sc["key_u"], sc["key_v"],
        interp, di0, dj0, dAi, dAj,
        sc["num_u"], sc["num_v"], sc["den_u"], sc["den_v"], float(sc["blend_eps"]),
    )
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()
    return {k: sc[k].clone() for k in
            ("sdf_cc", "sdf_u", "sdf_v", "body_u", "body_v")}


def _make_warp_out(sc, device):
    N = sc["Ngx"] * sc["Ngy"]
    out = {}
    for k in ("sdf_cc", "sdf_u", "sdf_v"):
        out[k] = wp.full(N, 1e4, dtype=wp.float32, device=device)
    for k in ("body_u", "body_v"):
        out[k] = wp.zeros(N, dtype=wp.float32, device=device)
    return out


def _run_warp(sc, device, interp, mode):
    w = WarpStreamingSDF2D(sc["Ngx"], sc["Ngy"], device=device)
    w.setup(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]),
            interp_method=interp, blend_eps=float(sc["blend_eps"]))
    w.update_kinematics(sc["kin"], sc["aabb_lo"], sc["aabb_dim"])
    out = _make_warp_out(sc, device)
    args = (out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["body_u"], out["body_v"])
    if mode == "fan-graph" and device.startswith("cuda"):
        w.run_fanned_eager(*args)              # warmup/JIT
        for k in out: out[k].fill_(1e4) if "sdf" in k else out[k].zero_()
        w.capture_graph_fanned(*args)
        for k in out: out[k].fill_(1e4) if "sdf" in k else out[k].zero_()
        w.run_graph_fanned()
    elif mode == "seq":
        w.run_eager(*args)
    else:
        w.run_fanned_eager(*args)
    wp.synchronize()
    return {k: wp.to_torch(out[k]).clone() for k in out}


def _compare(native, warp, label, sdf_rtol=5e-4):
    for key in ("sdf_cc", "sdf_u", "sdf_v"):
        n = native[key]; w = warp[key]
        mask = n < 1e3
        if not mask.any():
            continue
        rel = (n[mask] - w[mask]).abs() / n[mask].abs().clamp_min(1e-8)
        assert rel.max().item() < sdf_rtol, f"[{label}] {key} rel {rel.max():.2e}"
    for key in ("body_u", "body_v"):
        n = native[key]; w = warp[key]
        mask = n.abs() > 1e-10
        if not mask.any():
            continue
        rel = (n[mask] - w[mask]).abs() / n[mask].abs().clamp_min(1e-10)
        assert rel.max().item() < 1e-4, f"[{label}] {key} rel {rel.max():.2e}"


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3, 5])
@pytest.mark.parametrize("mode", ["fan-eager", "fan-graph", "seq"])
def test_kernelA_2d(B, mode):
    sc = make_synthetic_scene_2d(96, 64, B, device=DEV)
    native = _run_native(sc, 0)
    warp = _run_warp(sc, DEV, 0, mode)
    _compare(native, warp, f"B={B} {mode}")


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_kernelA_2d_quadratic():
    """interp_method=1 (biquadratic)."""
    sc = make_synthetic_scene_2d(96, 64, 3, device=DEV)
    native = _run_native(sc, 1)
    warp = _run_warp(sc, DEV, 1, "fan-eager")
    _compare(native, warp, "quadratic")


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
def test_kernelA_2d_blend():
    """blend_eps>0 softmin body-velocity path, overlapping AABBs."""
    sc = make_synthetic_scene_2d(96, 64, 3, device=DEV, blend=True, overlap=True)
    native = _run_native(sc, 0)
    warp = _run_warp(sc, DEV, 0, "fan-eager")
    # SDF unaffected by blend; body velocity is the blended field.
    _compare(native, warp, "blend", sdf_rtol=5e-4)


@SKIP_NO_NATIVE
@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3])
def test_kernelA_2d_float64(B):
    """float64 specialisation (the f64-solver bridge dtype) vs native f64.

    Build the scene in float64 and force F_flat/body_meta/kin to f64 too so the
    native op runs a consistent-dtype f64 path (the live bridge normalises the
    same way inside ``WarpStreamingSDF2D.setup``)."""
    sc = make_synthetic_scene_2d(96, 64, B, device=DEV, dtype=torch.float64)
    for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
        sc[k] = sc[k].double()
    native = _run_native(sc, 0)

    Ngx, Ngy = sc["Ngx"], sc["Ngy"]
    w = WarpStreamingSDF2D(Ngx, Ngy, device=DEV, dtype=wp.float64)
    w.setup(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
            sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]),
            interp_method=0, blend_eps=0.0)
    w.update_kinematics(sc["kin"], sc["aabb_lo"], sc["aabb_dim"])
    N = Ngx * Ngy
    out = {k: wp.full(N, 1e4, dtype=wp.float64, device=DEV)
           for k in ("sdf_cc", "sdf_u", "sdf_v")}
    out["body_u"] = wp.zeros(N, dtype=wp.float64, device=DEV)
    out["body_v"] = wp.zeros(N, dtype=wp.float64, device=DEV)
    w.run_fanned_eager(out["sdf_cc"], out["sdf_u"], out["sdf_v"],
                       out["body_u"], out["body_v"])
    wp.synchronize()
    warp = {k: wp.to_torch(out[k]).clone() for k in out}
    # The native op interpolates the body SDF in float32 internally even for
    # f64 output, while this Warp kernel is true-f64 (strictly more accurate),
    # so they agree to ~float32-epsilon (~1e-7), not bit-exactly.
    _compare(native, warp, f"f64 B={B}", sdf_rtol=1e-6)


@SKIP_NO_NATIVE
def test_kernelA_2d_cpu_matches_gpu():
    """Single Warp source: CPU == GPU (fanned)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for the GPU half")
    sc_g = make_synthetic_scene_2d(64, 48, 3, device="cuda:0")
    sc_c = make_synthetic_scene_2d(64, 48, 3, device="cpu")
    g = _run_warp(sc_g, "cuda:0", 0, "fan-eager")
    c = _run_warp(sc_c, "cpu", 0, "fan-eager")
    for k in g:
        gg = g[k].cpu(); cc = c[k]
        mask = gg.abs() < 1e3
        d = (gg[mask] - cc[mask]).abs().max().item()
        assert d < 1e-5, f"CPU vs GPU {k}: {d:.2e}"
