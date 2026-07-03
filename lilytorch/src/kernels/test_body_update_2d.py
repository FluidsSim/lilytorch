"""2-D Warp streaming SDF self-consistency tests.

Covers fanned + sequential + graphed designs, blend-eps off/on, interp_method
0/1, on CPU and GPU (single Warp source).  Mirrors `test_parity.py` (3-D) with
the z axis stripped.
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.scene_2d import make_synthetic_scene_2d
from lilytorch.src.streaming_sdf_2d import WarpStreamingSDF2D

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


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


def _compare(ref, got, label, sdf_rtol=5e-4):
    for key in ("sdf_cc", "sdf_u", "sdf_v"):
        n = ref[key].cpu(); w = got[key].cpu()
        mask = n < 1e3
        if not mask.any():
            continue
        rel = (n[mask] - w[mask]).abs() / n[mask].abs().clamp_min(1e-8)
        assert rel.max().item() < sdf_rtol, f"[{label}] {key} rel {rel.max():.2e}"
    for key in ("body_u", "body_v"):
        n = ref[key].cpu(); w = got[key].cpu()
        mask = n.abs() > 1e-10
        if not mask.any():
            continue
        rel = (n[mask] - w[mask]).abs() / n[mask].abs().clamp_min(1e-10)
        assert rel.max().item() < 1e-4, f"[{label}] {key} rel {rel.max():.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3, 5])
@pytest.mark.parametrize("mode", ["fan-graph", "seq"])
def test_kernelA_2d_modes_agree(B, mode):
    """Graphed and sequential designs match the fanned-eager reference."""
    sc = make_synthetic_scene_2d(96, 64, B, device=DEV)
    ref = _run_warp(sc, DEV, 0, "fan-eager")
    got = _run_warp(sc, DEV, 0, mode)
    _compare(ref, got, f"B={B} {mode}")


@SKIP_NO_CUDA
@pytest.mark.parametrize("interp", [0, 1], ids=["linear", "quadratic"])
def test_kernelA_2d_cpu_matches_gpu(interp):
    """Single Warp source: CPU == GPU (fanned)."""
    sc_g = make_synthetic_scene_2d(64, 48, 3, device="cuda:0")
    sc_c = make_synthetic_scene_2d(64, 48, 3, device="cpu")
    g = _run_warp(sc_g, "cuda:0", interp, "fan-eager")
    c = _run_warp(sc_c, "cpu", interp, "fan-eager")
    for k in g:
        gg = g[k].cpu(); cc = c[k]
        mask = gg.abs() < 1e3
        d = (gg[mask] - cc[mask]).abs().max().item()
        assert d < 1e-5, f"CPU vs GPU {k}: {d:.2e}"


@SKIP_NO_CUDA
def test_kernelA_2d_blend_cpu_matches_gpu():
    """blend_eps>0 softmin body-velocity path, overlapping AABBs."""
    sc_g = make_synthetic_scene_2d(96, 64, 3, device="cuda:0", blend=True, overlap=True)
    sc_c = make_synthetic_scene_2d(96, 64, 3, device="cpu", blend=True, overlap=True)
    g = _run_warp(sc_g, "cuda:0", 0, "fan-eager")
    c = _run_warp(sc_c, "cpu", 0, "fan-eager")
    for k in g:
        gg = g[k].cpu(); cc = c[k]
        mask = gg.abs() < 1e3
        d = (gg[mask] - cc[mask]).abs().max().item()
        assert d < 1e-5, f"CPU vs GPU (blend) {k}: {d:.2e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3])
def test_kernelA_2d_float64_cpu_matches_gpu(B):
    """float64 specialisation (the f64-solver bridge dtype): CPU == GPU."""
    def run(dev):
        sc = make_synthetic_scene_2d(96, 64, B, device=dev, dtype=torch.float64)
        for k in ("F_flat", "body_meta", "kin", "gx", "gy"):
            sc[k] = sc[k].double()
        Ngx, Ngy = sc["Ngx"], sc["Ngy"]
        w = WarpStreamingSDF2D(Ngx, Ngy, device=dev, dtype=wp.float64)
        w.setup(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
                sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]),
                interp_method=0, blend_eps=0.0)
        w.update_kinematics(sc["kin"], sc["aabb_lo"], sc["aabb_dim"])
        N = Ngx * Ngy
        out = {k: wp.full(N, 1e4, dtype=wp.float64, device=dev)
               for k in ("sdf_cc", "sdf_u", "sdf_v")}
        out["body_u"] = wp.zeros(N, dtype=wp.float64, device=dev)
        out["body_v"] = wp.zeros(N, dtype=wp.float64, device=dev)
        w.run_fanned_eager(out["sdf_cc"], out["sdf_u"], out["sdf_v"],
                           out["body_u"], out["body_v"])
        wp.synchronize()
        return {k: wp.to_torch(out[k]).clone() for k in out}

    g, c = run("cuda:0"), run("cpu")
    for k in g:
        gg = g[k].cpu(); cc = c[k]
        mask = gg.abs() < 1e3
        d = (gg[mask] - cc[mask]).abs().max().item()
        assert d < 1e-10, f"f64 CPU vs GPU {k}: {d:.2e}"
