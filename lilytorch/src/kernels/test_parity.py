"""Self-consistency tests: the 3-D Warp streaming SDF across execution designs.

Checks that every execution design of the Warp 3-pass implementation
(sequential eager, CUDA graph, fanned eager, fanned graph) produces the same
outputs, that the single Warp source matches between CPU and GPU, and that the
dtype-generic float64 specialisation is device-independent.

Run with:
    python -m lilytorch.src.kernels.test_parity
    pytest lilytorch/src/kernels/test_parity.py -v
"""
from __future__ import annotations

import pytest
import torch
import warp as wp

from lilytorch.src.kernels.bench_viability import (
    make_synthetic_scene,
    setup_warp_runner,
    _reset_warp_outputs,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_and_collect_warp(sc: dict, mode: str = "eager", device: str = DEVICE) -> dict:
    """Run Warp kernel and return output tensors.

    mode ∈ {"eager", "graph"}            → sequential per-body design
    mode ∈ {"fan-eager", "fan-graph"}    → fanned all-body design (const in B)
    """
    wsdf, wp_out = setup_warp_runner(sc, device)
    out_args = (
        wp_out["sdf_cc"], wp_out["sdf_u"], wp_out["sdf_v"], wp_out["sdf_w"],
        wp_out["body_u"], wp_out["body_v"], wp_out["body_w"],
    )
    _reset_warp_outputs(wp_out)

    if mode == "graph":
        wsdf.capture_graph(*out_args)
        _reset_warp_outputs(wp_out)
        wsdf.run_graph()
    elif mode == "fan-eager":
        wsdf.run_fanned_eager(*out_args)
    elif mode == "fan-graph":
        wsdf.run_fanned_eager(*out_args)        # warmup/JIT
        wsdf.capture_graph_fanned(*out_args)
        _reset_warp_outputs(wp_out)
        wsdf.run_graph_fanned()
    else:
        wsdf.run_eager(*out_args)

    wp.synchronize()
    return {k: wp.to_torch(wp_out[k]).clone() for k in wp_out}


def _compare(ref: dict, got: dict, label: str, sdf_rtol: float = 5e-4):
    """Assert SDF and body velocity match within tolerance.

    Tie-breaking note: when two bodies have overlapping AABBs and nearly-equal
    SDF at a cell the winner differs between the sequential and fanned-atomic
    designs.  The absolute error is always < 2 ULP (~1e-8) but the relative
    error can reach ~2e-4 near the zero-crossing band.  We use 5e-4 relative
    tolerance, which is physically tight (< 1/2000 of a grid cell).
    """
    for key in ("sdf_cc", "sdf_u", "sdf_v", "sdf_w"):
        n = ref[key].cpu()
        w = got[key].cpu()
        # Only compare cells that were touched (< FAR)
        mask = n < 1e3
        if not mask.any():
            continue
        diff = (n[mask] - w[mask]).abs()
        rel  = diff / (n[mask].abs().clamp_min(1e-8))
        max_rel = rel.max().item()
        max_abs = diff.max().item()
        assert max_rel < sdf_rtol, (
            f"[{label}] {key}: max rel err {max_rel:.3e} > {sdf_rtol:.0e} "
            f"(max abs {max_abs:.3e})"
        )

    for key in ("body_u", "body_v", "body_w"):
        n = ref[key].cpu()
        w = got[key].cpu()
        mask = n.abs() > 1e-10  # non-zero body velocity
        if not mask.any():
            continue
        diff = (n[mask] - w[mask]).abs()
        rel  = diff / (n[mask].abs().clamp_min(1e-10))
        max_rel = rel.max().item()
        assert max_rel < 1e-4, (
            f"[{label}] {key}: max rel err {max_rel:.3e} > 1e-4"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3, 9])
@pytest.mark.parametrize("mode", ["graph", "fan-eager", "fan-graph"])
def test_warp_modes_agree(B, mode):
    """Every execution design matches the sequential-eager reference."""
    sc = make_synthetic_scene(64, 32, 32, B, device=DEVICE)
    ref = _run_and_collect_warp(sc, mode="eager")
    got = _run_and_collect_warp(sc, mode=mode)
    _compare(ref, got, label=f"{mode} B={B}")


@SKIP_NO_CUDA
def test_warp_larger_grid():
    """Sequential vs fanned-graph on a 128×64×32 grid, 9 bodies."""
    sc = make_synthetic_scene(128, 64, 32, 9, device=DEVICE)
    ref = _run_and_collect_warp(sc, mode="eager")
    _compare(ref, _run_and_collect_warp(sc, mode="fan-graph"),
             label="128×64×32 B=9 fan")


@SKIP_NO_CUDA
def test_warp_graph_eager_identical():
    """Warp eager and Warp graph produce identical results (same computation)."""
    sc = make_synthetic_scene(64, 32, 32, 3, device=DEVICE)
    eager_out = _run_and_collect_warp(sc, mode="eager")
    graph_out = _run_and_collect_warp(sc, mode="graph")
    for key in eager_out:
        n = eager_out[key]; w = graph_out[key]
        mask = n.abs() > 0
        if not mask.any():
            continue
        max_diff = (n[mask] - w[mask]).abs().max().item()
        assert max_diff == 0.0, (
            f"{key}: eager vs graph differ by {max_diff:.3e} (should be 0)"
        )


@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3, 9])
def test_warp_cpu_matches_gpu(B):
    """Single Warp source: CPU == GPU (fanned eager)."""
    sc_g = make_synthetic_scene(64, 32, 32, B, device="cuda:0")
    sc_c = make_synthetic_scene(64, 32, 32, B, device="cpu")
    g = _run_and_collect_warp(sc_g, mode="fan-eager", device="cuda:0")
    c = _run_and_collect_warp(sc_c, mode="fan-eager", device="cpu")
    for key in g:
        gg = g[key].cpu(); cc = c[key]
        mask = gg.abs() < 1e3
        d = (gg[mask] - cc[mask]).abs().max().item()
        assert d < 1e-5, f"B={B} CPU vs GPU {key}: {d:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
#  float64 (dtype-generic 3-D body_update) — used by the f64 solver bridge
# ─────────────────────────────────────────────────────────────────────────────

def _run_warp_f64(sc: dict, device: str) -> dict:
    """Run the dtype-generic WarpStreamingSDF at float64 on the f64 scene `sc`,
    wrapping the scene's torch output buffers zero-copy.  Returns torch tensors."""
    from lilytorch.src.kernels.streaming_sdf import WarpStreamingSDF
    Ngx, Ngy, Ngz = sc["Ngx"], sc["Ngy"], sc["Ngz"]
    sc["sdf_cc"].fill_(1e4); sc["sdf_u"].fill_(1e4)
    sc["sdf_v"].fill_(1e4);  sc["sdf_w"].fill_(1e4)
    sc["body_u"].zero_();    sc["body_v"].zero_();  sc["body_w"].zero_()

    wsdf = WarpStreamingSDF(Ngx, Ngy, Ngz, device=device, dtype=wp.float64)
    wsdf.setup(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
               sc["gx"], sc["gy"], sc["gz"], float(sc["h"]), int(sc["max_vol"]))
    wsdf.update_kinematics(sc["kin"], sc["aabb_lo"], sc["aabb_dim"])

    def f(t):
        return wp.from_torch(t.reshape(-1))

    wsdf.run_fanned_eager(f(sc["sdf_cc"]), f(sc["sdf_u"]), f(sc["sdf_v"]),
                          f(sc["sdf_w"]), f(sc["body_u"]), f(sc["body_v"]),
                          f(sc["body_w"]))
    wp.synchronize()
    return {k: sc[k].clone() for k in
            ("sdf_cc", "sdf_u", "sdf_v", "sdf_w", "body_u", "body_v", "body_w")}


@SKIP_NO_CUDA
@pytest.mark.parametrize("B", [1, 3])
def test_warp_fanned_f64_cpu_matches_gpu(B):
    """Dtype-generic body_update at float64: CPU == GPU."""
    def run(device):
        sc = make_synthetic_scene(64, 32, 32, B, device=device, dtype=torch.float64)
        for k in ("F_flat", "body_meta", "kin", "gx", "gy", "gz"):
            sc[k] = sc[k].double()
        return _run_warp_f64(sc, device)

    g, c = run("cuda:0"), run("cpu")
    for key in g:
        gg = g[key].cpu(); cc = c[key]
        mask = gg.abs() < 1e3
        d = (gg[mask] - cc[mask]).abs().max().item()
        assert d < 1e-10, f"f64 B={B} CPU vs GPU {key}: {d:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke (no pytest) — run with:  python -m lilytorch.src.kernels.test_parity
# ─────────────────────────────────────────────────────────────────────────────

def _smoke(B: int, label: str):
    sc = make_synthetic_scene(64, 32, 32, B, device=DEVICE)
    warp_out = _run_and_collect_warp(sc, mode="eager")
    warp_graph_out = _run_and_collect_warp(sc, mode="graph")
    for key in warp_out:
        diff = (warp_out[key] - warp_graph_out[key]).abs().max().item()
        if diff > 0:
            print(f"  [{label}] eager/graph differ on {key}: {diff:.3e}")
        else:
            print(f"  [{label}] eager vs graph {key}: bit-identical")


if __name__ == "__main__":
    print("\nWarp streaming-SDF self-consistency smoke test")
    print("=" * 50)
    for B in [1, 3, 9]:
        _smoke(B, f"B={B}")
    print()
