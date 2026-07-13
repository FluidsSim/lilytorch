"""Phase-2 per-body-buffer parity gates (cuda_native_port plan item 10 / 2.1).

These are the *tests-first* gate for `milestones/per_body_key_buffers.md`.  The
per-body-buffer work (items 2.2 / 2.3) replaces the union-AABB streaming-SDF
pipeline with two regimes and must reproduce the union-AABB output:

  * **Regime A** — disjoint bodies (incl. B=1): a direct-write kernel that
    writes the winning SDF/velocity straight into the global tensors, no key
    buffers, no init/decode pass.
  * **Regime B** — overlapping links (multi-link robots): per-body PRIVATE
    buffers (raw fp64, no packed key; single writer per slot) + a one-thread-
    per-cell resolve that reads the covering bodies' buffers and picks the
    deterministic true min.  No atomics, no race — see the resolve kernel.

Reference oracle = the current union-AABB path `streaming_sdf_stag_{2,3}d_multi`.
NOTE (2026-07-11 decision): the new per-body method is the SOLE path and `_multi`
is being removed. `_multi` here is a *transitional* oracle; item 2.4 migrates
these gates to the new CPU twin (new-GPU vs new-CPU) once the old path is deleted.

Parity contract (derived from ``csrc/cuda/packed_key.cuh``)
----------------------------------------------------------
The multi path routes each winning SDF through a 64-bit packed key that keeps
the top 48 sortable bits and **drops the low 16 mantissa bits** to carry the
body id.  Consequences for "byte-identical vs the union-AABB path":

  * **SDF, fp32:** an fp32-origin value has those low 16 bits exactly zero, so
    the key round-trips it bit-exactly.  A direct-write path that computes the
    SDF with the *same sampler* therefore matches multi **byte-identically**
    (`torch.equal`).  This is the strong, load-bearing gate: it forces 2.2/2.3
    to reuse the identical `sdf_sample_dispatch_*` math rather than re-derive it.
  * **SDF, fp64:** the key loses ~2^-36 (~1.5e-11) relative.  A direct-written
    cell keeps the raw (more accurate) fp64 value, so it differs from multi's
    quantised value at that floor — **not** byte-identical; gated at ``ATOL_F64``.
    (Regime B stores raw fp64 in per-body buffers — no key — so it too is more
    accurate than multi and differs from it at the ~1.5e-11 quantisation floor;
    same ATOL_F64 tolerance as Regime A.  A >~1e-8 divergence means a lost
    winner, i.e. a race — NOT a reason to relax this gate.)
  * **Body velocity:** recomputed from the winning body id (never quantised),
    so it does not carry the key floor; gated tightly (fp32 ``ATOL_F32``, fp64
    ``ATOL_F64``).

Test layers
-----------
  1. ``test_scene_*`` — guard the synthetic scenes: liveness, and that
     "separated" is genuinely disjoint (Regime A) while "multilink" has real
     overlap cells (Regime B).  Runs today.
  2. ``test_facade_matches_multi_*`` — the production invariant.  Compares the
     facade entry point ``_native_body_update_{2,3}d`` (which 2.2/2.3 will teach
     to dispatch regimes) against the pinned multi reference.  Passes today
     (facade → multi) and REMAINS the gate as the internals change.
  3. ``test_direct_write_matches_multi_*`` — the op-level Regime-A gate.
     **SKIPs** until ``native.streaming_sdf_stag_{2,3}d_direct`` exists (item
     2.2 restores + generalises the reverted B=1 kernel); flips SKIP→PASS then.
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

# fp32 SDF is byte-identical; fp64 sits at the packed-key quantisation floor
# (~1.5e-11, see packed_key.cuh).  ATOL_F64 is the house native-parity gate,
# comfortably above the floor; ATOL_F32 bounds the (un-quantised) velocity.
ATOL_F64 = 1e-9
ATOL_F32 = 2e-7


def _need(op_name: str):
    if not hasattr(torch.ops.lilytorch_kernels, op_name):
        pytest.skip(f"item 2.2: native op {op_name} not implemented yet")


# ── controlled synthetic scenes ─────────────────────────────────────────────
# Each body is a disc/sphere SDF table, placed with R = identity and its centre
# at the world-centre of its AABB (so every body writes live cells inside its
# own AABB).  Layout controls the AABB arrangement, hence the regime.

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
    for (i0, j0, Ai, Aj) in aabbs:
        cx, cy = (i0 + Ai / 2) * h, (j0 + Aj / 2) * h
        kin.append([1, 0, 0, 1, cx, cy, cx, cy, 0.1, -0.05, 0.3])  # R,bp,cm,lv,om
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
    for (i0, j0, k0, Ai, Aj, Ak) in aabbs:
        cx, cy, cz = (i0 + Ai / 2) * h, (j0 + Aj / 2) * h, (k0 + Ak / 2) * h
        kin.append([1, 0, 0, 0, 1, 0, 0, 0, 1,          # R
                    cx, cy, cz, cx, cy, cz,             # bp, cm
                    0.1, -0.05, 0.02, 0.3, -0.1, 0.05])  # lv, av
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
    """Union AABB = the dirty rect passed to the multi kernel."""
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


# ── runners: reference (multi), production (facade), op-level (direct) ───────

def _out_buffers(sc, dtype):
    shp = (sc["Ngx"], sc["Ngy"]) if sc["dim"] == 2 else (sc["Ngx"], sc["Ngy"], sc["Ngz"])
    comps = ("cc", "u", "v") if sc["dim"] == 2 else ("cc", "u", "v", "w")
    vels = ("u", "v") if sc["dim"] == 2 else ("u", "v", "w")
    out = {f"sdf_{c}": torch.full(shp, FAR, dtype=dtype, device=DEV) for c in comps}
    out.update({f"body_{c}": torch.zeros(shp, dtype=dtype, device=DEV) for c in vels})
    return out


def _out_buffers_cpu(sc, dtype):
    shp = (sc["Ngx"], sc["Ngy"]) if sc["dim"] == 2 else (sc["Ngx"], sc["Ngy"], sc["Ngz"])
    comps = ("cc", "u", "v") if sc["dim"] == 2 else ("cc", "u", "v", "w")
    vels = ("u", "v") if sc["dim"] == 2 else ("u", "v", "w")
    out = {f"sdf_{c}": torch.full(shp, FAR, dtype=dtype, device="cpu") for c in comps}
    out.update({f"body_{c}": torch.zeros(shp, dtype=dtype, device="cpu") for c in vels})
    return out


def run_multi(sc, dtype):
    out = _out_buffers(sc, dtype)
    N = int(np.prod([sc["Ngx"], sc["Ngy"]] + ([sc["Ngz"]] if sc["dim"] == 3 else [])))
    lo, dim = _dirty(sc)
    keys = [torch.empty(N, dtype=torch.int64, device=DEV) for _ in out if _.startswith("sdf")]
    z = lambda: torch.zeros(1, dtype=dtype, device=DEV)
    if sc["dim"] == 2:
        native.streaming_sdf_stag_2d_multi(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["body_u"], out["body_v"],
            keys[0], keys[1], keys[2], 0, lo[0], lo[1], dim[0], dim[1],
            z(), z(), z(), z(), 0.0)
    else:
        native.streaming_sdf_stag_3d_multi(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["sdf_w"],
            out["body_u"], out["body_v"], out["body_w"],
            keys[0], keys[1], keys[2], keys[3], 0, lo[0], lo[1], lo[2], dim[0], dim[1], dim[2],
            z(), z(), z(), z(), z(), z(), 0.0)
    return out


def run_facade(sc, dtype, dev=DEV):
    """Production dispatch: Regime A (direct) for disjoint, Regime B (resolve)
    for overlap.  `dev` lets the GPU-vs-CPU twin gate drive the same path on CPU."""
    out = _out_buffers(sc, dtype) if dev != "cpu" else _out_buffers_cpu(sc, dtype)
    N = int(np.prod([sc["Ngx"], sc["Ngy"]] + ([sc["Ngz"]] if sc["dim"] == 3 else [])))
    lo, dim = _dirty(sc)
    keys = [torch.empty(N, dtype=torch.int64, device=dev) for _ in out if _.startswith("sdf")]
    z = lambda: torch.zeros(1, dtype=dtype, device=dev)
    if sc["dim"] == 2:
        facade._native_body_update_2d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["body_u"], out["body_v"],
            0, lo[0], lo[1], dim[0], dim[1],
            z(), z(), z(), z(), 0.0, keys[0], keys[1], keys[2])
    else:
        facade._native_body_update_3d(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["sdf_w"],
            out["body_u"], out["body_v"], out["body_w"],
            0, lo[0], lo[1], lo[2], dim[0], dim[1], dim[2],
            z(), z(), z(), z(), z(), z(), 0.0, keys[0], keys[1], keys[2], keys[3])
    return out


def run_direct(sc, dtype):
    """Op-level Regime-A direct-write path (item 2.2).  Requires the native op."""
    out = _out_buffers(sc, dtype)
    lo, dim = _dirty(sc)
    if sc["dim"] == 2:
        native.streaming_sdf_stag_2d_direct(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["body_u"], out["body_v"],
            0, lo[0], lo[1], dim[0], dim[1])
    else:
        native.streaming_sdf_stag_3d_direct(
            sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"], sc["kin"],
            sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"], sc["gz"], sc["h"], sc["max_vol"],
            out["sdf_cc"], out["sdf_u"], out["sdf_v"], out["sdf_w"],
            out["body_u"], out["body_v"], out["body_w"],
            0, lo[0], lo[1], lo[2], dim[0], dim[1], dim[2])
    return out


# ── the parity contract ─────────────────────────────────────────────────────

def assert_matches_multi(ref, got, dtype, label):
    """Unified contract vs the union-AABB reference (the deterministic true min).

    SDF fields:  fp32 → byte-identical (torch.equal); fp64 → ≤ ATOL_F64.
    Velocity:    ≤ ATOL_F32 (fp32) / ATOL_F64 (fp64).

    This holds for BOTH regimes: `_multi` (atomicMin), the disjoint direct-write
    kernel, and — once correct — the Regime-B resolve all compute the same true
    minimum.  A path that diverges here at the ~0.1 level is losing the winner
    (e.g. the direct kernel *raced* on overlapping bodies — that is a bug in the
    path, NOT a reason to relax this gate; the atomic/serial min is the truth).
    """
    for k, v in ref.items():
        g = got[k]
        if k.startswith("sdf") and dtype == torch.float32:
            assert torch.equal(v, g), (
                f"[{label}] {k} not byte-identical in fp32 "
                f"(maxdiff={(v - g).abs().max().item():.3e}); the direct/resolve "
                f"path must reuse the multi sampler bit-for-bit")
        else:
            atol = ATOL_F32 if (dtype == torch.float32 and k.startswith("body")) else ATOL_F64
            md = (v - g).abs().max().item()
            assert md <= atol, f"[{label}] {k} maxdiff {md:.3e} > {atol:.1e}"


# ── layer 1: scene guards ────────────────────────────────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_scene_liveness(dim, layout):
    sc = make_scene(dim, layout, torch.float32)
    out = run_multi(sc, torch.float32)
    live = (out["sdf_cc"] < 1e3).sum().item()
    assert live > 0, f"{dim}D {layout}: no live cells (vacuous scene)"


@SKIP_NO_CUDA
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_scene_regime(dim, layout):
    """separated/single must be disjoint (Regime A); multilink must overlap (B)."""
    cov = _coverage(make_scene(dim, layout, torch.float32))
    n_overlap = int((cov >= 2).sum())
    if layout in DISJOINT:
        assert n_overlap == 0, f"{dim}D {layout}: expected disjoint AABBs"
    else:
        assert n_overlap > 0, f"{dim}D {layout}: multilink must have overlap cells"


# ── layer 2: production invariant (runs now, gate through 2.2/2.3) ───────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_facade_matches_multi(dim, layout, dtype):
    sc = make_scene(dim, layout, dtype)
    ref = run_multi(sc, dtype)
    got = run_facade(sc, dtype)
    assert_matches_multi(ref, got, dtype, f"facade/{dim}D/{layout}/{dtype}")


# ── layer 3: op-level Regime-A gate (SKIP until item 2.2 lands) ──────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", DISJOINT)
def test_direct_write_matches_multi(dim, layout, dtype):
    _need(f"streaming_sdf_stag_{dim}d_direct")
    sc = make_scene(dim, layout, dtype)
    ref = run_multi(sc, dtype)
    got = run_direct(sc, dtype)
    assert_matches_multi(ref, got, dtype, f"direct/{dim}D/{layout}/{dtype}")


# ── layer 4: durable GPU-vs-CPU twin gate (item 2.2) ─────────────────────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", DISJOINT)
def test_direct_gpu_eq_cpu(dim, layout, dtype):
    """Regime-A direct kernel: GPU output must match CPU twin."""
    _need(f"streaming_sdf_stag_{dim}d_direct")
    sc_gpu = make_scene(dim, layout, dtype)
    # Run on GPU
    out_gpu = _out_buffers(sc_gpu, dtype)
    lo, dirty_dim = _dirty(sc_gpu)
    if dim == 2:
        native.streaming_sdf_stag_2d_direct(
            sc_gpu["F_flat"], sc_gpu["F_offsets"], sc_gpu["body_shapes"],
            sc_gpu["body_meta"], sc_gpu["kin"],
            sc_gpu["aabb_lo"], sc_gpu["aabb_dim"],
            sc_gpu["gx"], sc_gpu["gy"], sc_gpu["h"], sc_gpu["max_vol"],
            out_gpu["sdf_cc"], out_gpu["sdf_u"], out_gpu["sdf_v"],
            out_gpu["body_u"], out_gpu["body_v"],
            0, lo[0], lo[1], dirty_dim[0], dirty_dim[1])
    else:
        native.streaming_sdf_stag_3d_direct(
            sc_gpu["F_flat"], sc_gpu["F_offsets"], sc_gpu["body_shapes"],
            sc_gpu["body_meta"], sc_gpu["kin"],
            sc_gpu["aabb_lo"], sc_gpu["aabb_dim"],
            sc_gpu["gx"], sc_gpu["gy"], sc_gpu["gz"], sc_gpu["h"], sc_gpu["max_vol"],
            out_gpu["sdf_cc"], out_gpu["sdf_u"], out_gpu["sdf_v"], out_gpu["sdf_w"],
            out_gpu["body_u"], out_gpu["body_v"], out_gpu["body_w"],
            0, lo[0], lo[1], lo[2], dirty_dim[0], dirty_dim[1], dirty_dim[2])

    # Run same scene on CPU
    sc_cpu = {}
    for k, v in sc_gpu.items():
        if isinstance(v, torch.Tensor):
            sc_cpu[k] = v.cpu()
        else:
            sc_cpu[k] = v
    out_cpu = _out_buffers_cpu(sc_cpu, dtype)
    if dim == 2:
        native.streaming_sdf_stag_2d_direct(
            sc_cpu["F_flat"], sc_cpu["F_offsets"], sc_cpu["body_shapes"],
            sc_cpu["body_meta"], sc_cpu["kin"],
            sc_cpu["aabb_lo"], sc_cpu["aabb_dim"],
            sc_cpu["gx"], sc_cpu["gy"], sc_cpu["h"], sc_cpu["max_vol"],
            out_cpu["sdf_cc"], out_cpu["sdf_u"], out_cpu["sdf_v"],
            out_cpu["body_u"], out_cpu["body_v"],
            0, lo[0], lo[1], dirty_dim[0], dirty_dim[1])
    else:
        native.streaming_sdf_stag_3d_direct(
            sc_cpu["F_flat"], sc_cpu["F_offsets"], sc_cpu["body_shapes"],
            sc_cpu["body_meta"], sc_cpu["kin"],
            sc_cpu["aabb_lo"], sc_cpu["aabb_dim"],
            sc_cpu["gx"], sc_cpu["gy"], sc_cpu["gz"], sc_cpu["h"], sc_cpu["max_vol"],
            out_cpu["sdf_cc"], out_cpu["sdf_u"], out_cpu["sdf_v"], out_cpu["sdf_w"],
            out_cpu["body_u"], out_cpu["body_v"], out_cpu["body_w"],
            0, lo[0], lo[1], lo[2], dirty_dim[0], dirty_dim[1], dirty_dim[2])

    # Compare GPU vs CPU (both direct).
    # fp32 SDF: GPU vs CPU may differ by ~1e-6 due to different FMA ordering
    # in the sampler accumulators (CUDA vs at::parallel_for).
    _DIRECT_CPU_ATOL_F32 = 1e-6
    for k in out_gpu:
        g = out_gpu[k].cpu()
        c = out_cpu[k]
        if dtype == torch.float32 and k.startswith("sdf"):
            md = (g - c).abs().max().item()
            assert md <= _DIRECT_CPU_ATOL_F32, (
                f"[gpu_eq_cpu/{dim}D/{layout}/{dtype}] {k} maxdiff {md:.3e} > {_DIRECT_CPU_ATOL_F32:.1e}")
        elif dtype == torch.float32 and k.startswith("body"):
            md = (g - c).abs().max().item()
            assert md <= ATOL_F32, (
                f"[gpu_eq_cpu/{dim}D/{layout}/{dtype}] {k} maxdiff {md:.3e} > {ATOL_F32:.1e}")
        else:
            # fp64: expect tight match
            md = (g - c).abs().max().item()
            assert md <= ATOL_F64, (
                f"[gpu_eq_cpu/{dim}D/{layout}/{dtype}] {k} maxdiff {md:.3e} > {ATOL_F64:.1e}")


# ── layer 5: Regime-B durable parity gate (GPU vs CPU twin, item 2.3) ────────

@SKIP_NO_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("layout", ALL_LAYOUTS)  # includes multilink (Regime B)
def test_regimeB_gpu_eq_cpu(dim, layout, dtype):
    """Race detector: the production facade dispatch (Regime A direct for
    disjoint, Regime B resolve for overlap) must give GPU == CPU.

    A passing GPU==CPU twin is what proves there is no data race: a racing GPU
    path (e.g. the direct kernel run on overlapping bodies) produces a
    nondeterministic result that will NOT match the serial CPU twin.  Covers all
    layouts, including `multilink` (overlap → Regime-B resolve).  Do NOT special-
    case multilink to the direct kernel — that is the race that this gate exists
    to catch."""
    sc_gpu = make_scene(dim, layout, dtype)
    out_gpu = run_facade(sc_gpu, dtype, dev=DEV)
    sc_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
              for k, v in sc_gpu.items()}
    out_cpu = run_facade(sc_cpu, dtype, dev="cpu")

    # fp32 SDF: GPU vs CPU differ ~1e-6 (2-D bilinear) up to ~5e-6 (3-D
    # triquadratic) from FMA ordering; velocity/fp64 are tight.
    _CPU_ATOL_F32 = 5e-6
    for k in out_gpu:
        md = (out_gpu[k].cpu() - out_cpu[k]).abs().max().item()
        if dtype == torch.float32 and k.startswith("sdf"):
            atol = _CPU_ATOL_F32
        elif dtype == torch.float32 and k.startswith("body"):
            atol = ATOL_F32
        else:
            atol = ATOL_F64
        assert md <= atol, (
            f"[regimeB_gpu_eq_cpu/{dim}D/{layout}/{dtype}] {k} maxdiff {md:.3e}"
            f" > {atol:.1e}")
