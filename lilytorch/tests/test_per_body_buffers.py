"""Phase-2 per-body-buffer parity gates (cuda_native_port plan items 2.1–2.4).

The streaming SDF runs on a SINGLE regime (the union-AABB packed-key ``_multi``
and the ``_direct`` paths were DELETED in items 2.4 and CL2 respectively):

  * **Regime B** — ``streaming_sdf_stag_{2,3}d_resolve`` — per-body PRIVATE buffers (raw
    fp64, no packed key; single writer per slot) + a one-thread-per-cell
    resolve that picks the PER-STAGGER minimum over the covering bodies
    (mirrors the retired ``_multi``'s independent atomicMin per stagger).
    ``blend_eps > 0`` adds the softmin velocity blend Σwᵢvᵢ/Σwᵢ,
    wᵢ = sigmoid(-sᵢ/eps), accumulated in registers (deterministic).

Durable gate structure (post-2.4 + CL2 oracle strategy — the ``_multi`` and
``_direct`` oracles are gone; correctness rests on):

  1. **GPU == CPU twin** on the production facade dispatch, ALL layouts —
     the race detector: a racing GPU path produces nondeterministic output
     that cannot match the serial CPU twin.
  2. Blend invariants: blend must not touch the SDF; on single-cover cells
     the blend is a velocity no-op (Σwv/Σw = v); on overlap scenes it must
     actually blend; blended GPU == blended CPU.
  3. The coupled-sim gates (2-D + 3-D salamander, see the plan 2.4 log)
     validated resolve == _multi BYTE-IDENTICAL (fp32 SDF) on real
     multi-link geometry before the deletion.

Scene note: body centres carry a sub-cell jitter and per-body velocities.
Without the jitter the inter-body seams land exactly between cell centres
and the per-stagger winner never differs from the cell-centre winner — the
degenerate geometry that hid the resolve cc-winner-for-all-staggers bug.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lilytorch.src import native
from lilytorch.src import facade

DEV = "cuda:0"
SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
FAR = 1e4

# fp64: house native-parity gate.  fp32 velocity bound; SDF cross-kernel
# checks are byte-identity (same-device) or FMA-noise bounded (GPU vs CPU).
ATOL_F64 = 1e-9
ATOL_F32 = 2e-7
_CPU_ATOL_F32 = 5e-6  # GPU-vs-CPU fp32 SDF FMA-ordering noise (3-D triquadratic)

_BLEND_EPS_CELLS = 2.0  # matches gen_configs_swim_3d's body_velocity_blend_eps_cells


# ── controlled synthetic scenes ─────────────────────────────────────────────
# Each body is a disc/sphere SDF table, placed with R = identity and its centre
# near the world-centre of its AABB (sub-cell jitter, see module docstring).
# Layout controls the AABB arrangement, hence the regime.

def _disc_table_2d(M: int, r0: float):
    ax = np.linspace(-(M - 1) * 0.005, (M - 1) * 0.005, M).astype(np.float64)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    return (np.sqrt(X ** 2 + Y ** 2) - r0).astype(np.float32), ax


def _sphere_table_3d(M: int, r0: float):
    ax = np.linspace(-(M - 1) * 0.005, (M - 1) * 0.005, M).astype(np.float64)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return (np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - r0).astype(np.float32), ax


_AABBS_2D = {
    "single":    [(36, 24, 24, 24)],
    "separated": [(8, 24, 24, 24), (64, 24, 24, 24)],           # disjoint
    "multilink": [(20, 24, 24, 24), (34, 24, 24, 24), (48, 24, 24, 24)],  # overlap
}
_AABBS_3D = {
    "single":    [(13, 9, 9, 14, 14, 14)],
    "separated": [(2, 9, 9, 14, 14, 14), (24, 9, 9, 14, 14, 14)],  # disjoint
    "multilink": [(6, 9, 9, 14, 14, 14), (14, 9, 9, 14, 14, 14),
                  (22, 9, 9, 14, 14, 14)],                          # overlap
}
DISJOINT = ("single", "separated")
ALL_LAYOUTS = ("single", "separated", "multilink")


def make_scene_2d(layout, dtype, Ngx=96, Ngy=72, M=32, r0=0.10, h=0.02):
    disc, ax = _disc_table_2d(M, r0)
    inv = 1.0 / (ax[1] - ax[0])
    meta_row = [ax[0], ax[0], ax[-1], ax[-1], inv, inv, inv * inv]
    aabbs = _AABBS_2D[layout]
    B = len(aabbs)
    kin, alo, adim = [], [], []
    for b, (i0, j0, Ai, Aj) in enumerate(aabbs):
        # Sub-cell jitter + per-body velocities — see the module docstring.
        cx = (i0 + Ai / 2 + 0.23 * ((b % 3) - 1)) * h
        cy = (j0 + Aj / 2 + 0.31 * ((b % 2) - 0.5)) * h
        kin.append([1, 0, 0, 1, cx, cy, cx, cy,
                    0.1 * (1 + b), -0.05 * (1 + 0.5 * b), 0.3 * (1 - 0.3 * b)])
        alo.append([i0, j0]); adim.append([Ai, Aj])

    def T(a, dt):
        return torch.tensor(np.array(a), dtype=dt, device=DEV)

    return dict(
        dim=2, Ngx=Ngx, Ngy=Ngy, B=B, h=h, aabbs=aabbs,
        F_flat=T(np.tile(disc.ravel(), B), dtype),
        F_offsets=T(np.arange(B + 1) * (M * M), torch.int64),
        body_shapes=T([[M, M]] * B, torch.int64),
        body_meta=T([meta_row] * B, dtype),
        kin=T(kin, dtype), aabb_lo=T(alo, torch.int64), aabb_dim=T(adim, torch.int64),
        gx=T(np.arange(Ngx) * h, dtype), gy=T(np.arange(Ngy) * h, dtype),
        max_vol=int(max(Ai * Aj for _, _, Ai, Aj in aabbs)),
    )


def make_scene_3d(layout, dtype, Ngx=40, Ngy=32, Ngz=32, M=16, r0=0.06, h=0.02):
    sph, ax = _sphere_table_3d(M, r0)
    inv = 1.0 / (ax[1] - ax[0])
    meta_row = [ax[0], ax[0], ax[0], ax[-1], ax[-1], ax[-1], inv, inv, inv, inv ** 3]
    aabbs = _AABBS_3D[layout]
    B = len(aabbs)
    kin, alo, adim = [], [], []
    for b, (i0, j0, k0, Ai, Aj, Ak) in enumerate(aabbs):
        # Sub-cell jitter + per-body velocities — see the module docstring.
        cx = (i0 + Ai / 2 + 0.23 * ((b % 3) - 1)) * h
        cy = (j0 + Aj / 2 + 0.31 * ((b % 2) - 0.5)) * h
        cz = (k0 + Ak / 2 + 0.17 * ((b % 3) - 1)) * h
        kin.append([1, 0, 0, 0, 1, 0, 0, 0, 1,          # R
                    cx, cy, cz, cx, cy, cz,             # bp, cm
                    0.1 * (1 + b), -0.05 * (1 + 0.5 * b), 0.02 * (1 + b),
                    0.3 * (1 - 0.3 * b), -0.1 * (1 + 0.4 * b), 0.05 * (1 + b)])
        alo.append([i0, j0, k0]); adim.append([Ai, Aj, Ak])

    def T(a, dt):
        return torch.tensor(np.array(a), dtype=dt, device=DEV)

    return dict(
        dim=3, Ngx=Ngx, Ngy=Ngy, Ngz=Ngz, B=B, h=h, aabbs=aabbs,
        F_flat=T(np.tile(sph.ravel(), B), dtype),
        F_offsets=T(np.arange(B + 1) * (M ** 3), torch.int64),
        body_shapes=T([[M, M, M]] * B, torch.int64),
        body_meta=T([meta_row] * B, dtype),
        kin=T(kin, dtype), aabb_lo=T(alo, torch.int64), aabb_dim=T(adim, torch.int64),
        gx=T(np.arange(Ngx) * h, dtype), gy=T(np.arange(Ngy) * h, dtype),
        gz=T(np.arange(Ngz) * h, dtype),
        max_vol=int(max(Ai * Aj * Ak for _, _, _, Ai, Aj, Ak in aabbs)),
    )


def make_scene(dim, layout, dtype):
    return make_scene_2d(layout, dtype) if dim == 2 else make_scene_3d(layout, dtype)


def _dirty(sc):
    """Union AABB = the dirty rect passed to the streaming kernels."""
    alo = sc["aabb_lo"].cpu().numpy(); adim = sc["aabb_dim"].cpu().numpy()
    lo = alo.min(0); hi = (alo + adim).max(0)
    return lo.tolist(), (hi - lo).tolist()


def _coverage(sc):
    """Per-cell count of AABBs covering it (host, from the layout)."""
    if sc["dim"] == 2:
        cov = np.zeros((sc["Ngx"], sc["Ngy"]), dtype=int)
        for (i0, j0, Ai, Aj) in sc["aabbs"]:
            cov[i0:i0 + Ai, j0:j0 + Aj] += 1
    else:
        cov = np.zeros((sc["Ngx"], sc["Ngy"], sc["Ngz"]), dtype=int)
        for (i0, j0, k0, Ai, Aj, Ak) in sc["aabbs"]:
            cov[i0:i0 + Ai, j0:j0 + Aj, k0:k0 + Ak] += 1
    return cov


# ── runners ──────────────────────────────────────────────────────────────────

def _out_buffers(sc, dtype, dev=DEV):
    shp = (sc["Ngx"], sc["Ngy"]) if sc["dim"] == 2 else (sc["Ngx"], sc["Ngy"], sc["Ngz"])
    comps = ("cc", "u", "v") if sc["dim"] == 2 else ("cc", "u", "v", "w")
    vels = ("u", "v") if sc["dim"] == 2 else ("u", "v", "w")
    out = {f"sdf_{c}": torch.full(shp, FAR, dtype=dtype, device=dev) for c in comps}
    out.update({f"body_{c}": torch.zeros(shp, dtype=dtype, device=dev) for c in vels})
    return out


def run_facade(sc, dtype, dev=DEV, blend_eps=0.0):
    """Production dispatch: Regime A (direct) for disjoint, Regime B (resolve)
    for overlap.  `dev` lets the GPU-vs-CPU twin gate drive the same path on CPU."""
    out = _out_buffers(sc, dtype, dev=dev)
    lo, dim = _dirty(sc)
    if sc["dim"] == 2:
        facade.body_update_2d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["body_u"], out["body_v"],
            0, lo[0], lo[1], dim[0], dim[1],
            blend_eps=float(blend_eps))
    else:
        facade.body_update_3d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["sdf_w"],
            out["body_u"], out["body_v"], out["body_w"],
            0, lo[0], lo[1], lo[2], dim[0], dim[1], dim[2],
            blend_eps=float(blend_eps))
    return out


def run_resolve(sc, dtype, dev=DEV, blend_eps=0.0):
    """Op-level Regime-B resolve path, forced on ANY layout (incl. disjoint —
    the resolve is regime-agnostic; only its performance motivates the direct
    kernel for disjoint sets)."""
    out = _out_buffers(sc, dtype, dev=dev)
    lo, dim = _dirty(sc)
    device = out["sdf_cc"].device
    priv_offsets, pb = facade._regime_b_priv(
        sc["body_shapes"].size(0), sc["max_vol"], dtype, device, sc["dim"])
    if sc["dim"] == 2:
        native.streaming_sdf_stag_2d_resolve(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["body_u"], out["body_v"],
            0, lo[0], lo[1], dim[0], dim[1],
            priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4],
            blend_eps=float(blend_eps))
    else:
        native.streaming_sdf_stag_3d_resolve(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["sdf_w"],
            out["body_u"], out["body_v"], out["body_w"],
            0, lo[0], lo[1], lo[2], dim[0], dim[1], dim[2],
            priv_offsets, pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6],
            blend_eps=float(blend_eps))
    return out


def _to_cpu_scene(sc):
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in sc.items()}


# ── layer 1: scene guards ────────────────────────────────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_scene_liveness(dim, layout):
    sc = make_scene(dim, layout, torch.float32)
    out = run_facade(sc, torch.float32)
    live = (out["sdf_cc"] < 1e3).sum().item()
    assert live > 0, f"{dim}D {layout}: no live cells (vacuous scene)"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_scene_regime(dim, layout):
    """separated/single must be disjoint; multilink must overlap."""
    cov = _coverage(make_scene(dim, layout, torch.float32))
    n_overlap = int((cov >= 2).sum())
    if layout in DISJOINT:
        assert n_overlap == 0, f"{dim}D {layout}: expected disjoint AABBs"
    else:
        assert n_overlap > 0, f"{dim}D {layout}: multilink must have overlap cells"


# ── layer 2: durable GPU-vs-CPU twin gates (race detectors) ──────────────────

def _assert_gpu_eq_cpu(out_gpu, out_cpu, dtype, label):
    for k in out_gpu:
        md = (out_gpu[k].cpu() - out_cpu[k]).abs().max().item()
        if dtype == torch.float32:
            atol = _CPU_ATOL_F32 if k.startswith("sdf") else ATOL_F32
        else:
            atol = ATOL_F64
        assert md <= atol, f"[{label}] {k} maxdiff {md:.3e} > {atol:.1e}"


# ── layer 3: softmin velocity-blend gates ────────────────────────────────────
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_regimeB_gpu_eq_cpu(dim, layout, dtype):
    """Race detector: the production facade dispatch must give GPU == CPU.

    A passing GPU==CPU twin is what proves there is no data race: a racing GPU
    path produces a nondeterministic result that will NOT match the serial CPU
    twin.  Covers all layouts, including ``multilink`` (overlap)."""
    sc = make_scene(dim, layout, dtype)
    out_gpu = run_facade(sc, dtype, dev=DEV)
    out_cpu = run_facade(_to_cpu_scene(sc), dtype, dev="cpu")
    _assert_gpu_eq_cpu(out_gpu, out_cpu, dtype, f"regimeB_gpu_eq_cpu/{dim}D/{layout}/{dtype}")


# ── layer 4: softmin velocity-blend gates ────────────────────────────────────
# blend_eps > 0 replaces the winner-take-all body velocity with the softmin
# blend Σwᵢvᵢ/Σwᵢ, wᵢ = sigmoid(-sᵢ/eps) per stagger, accumulated in registers
# over the covering bodies (ascending b — deterministic, no atomics).

@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ["multilink"])
def test_blend_gpu_eq_cpu(dim, layout, dtype):
    """Race/parity detector for the blended resolve: GPU == CPU twin."""
    sc = make_scene(dim, layout, dtype)
    blend_eps = _BLEND_EPS_CELLS * sc["h"]
    out_gpu = run_facade(sc, dtype, dev=DEV, blend_eps=blend_eps)
    out_cpu = run_facade(_to_cpu_scene(sc), dtype, dev="cpu", blend_eps=blend_eps)
    _assert_gpu_eq_cpu(out_gpu, out_cpu, dtype, f"blend_gpu_eq_cpu/{dim}D/{layout}/{dtype}")


@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", DISJOINT)
def test_blend_single_cover_noop(dim, layout, dtype):
    """On disjoint scenes every covered cell has exactly one covering body, so
    the blend must be a velocity no-op (Σwv/Σw = v up to one rounding) and the
    SDF must be untouched.  This is also why the facade may keep routing
    disjoint scenes to the (blend-less) direct kernel with blend_eps > 0."""
    sc = make_scene(dim, layout, dtype)
    hard = run_resolve(sc, dtype)
    soft = run_resolve(sc, dtype, blend_eps=_BLEND_EPS_CELLS * sc["h"])
    for k in hard:
        md = (hard[k] - soft[k]).abs().max().item()
        if k.startswith("sdf"):
            assert md == 0.0, f"[blend_noop/{dim}D/{layout}] {k} SDF changed by blend"
        else:
            atol = ATOL_F32 if dtype == torch.float32 else ATOL_F64
            assert md <= atol, f"[blend_noop/{dim}D/{layout}] {k} maxdiff {md:.3e}"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dim", [2, 3])
def test_blend_actually_blends(dim):
    """Guard against a silently-ignored blend_eps: on an overlap scene the
    blended velocity field must differ from the winner-take-all one in the
    seam band (bodies have distinct velocities there)."""
    dtype = torch.float64
    sc = make_scene(dim, "multilink", dtype)
    hard = run_facade(sc, dtype)
    soft = run_facade(sc, dtype, blend_eps=_BLEND_EPS_CELLS * sc["h"])
    diff = max((hard[k] - soft[k]).abs().max().item()
               for k in hard if k.startswith("body"))
    assert diff > 1e-6, "blend_eps>0 produced the identical velocity field"
    sdf_diff = max((hard[k] - soft[k]).abs().max().item()
                   for k in hard if k.startswith("sdf"))
    assert sdf_diff == 0.0, "blend must not change the SDF"
