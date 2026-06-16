"""Parity: BDIMhandler._update_streaming_multi == legacy _update_{2,3}d_streaming_multi.

Self-test for Step 6 of the 2D/3D solver unification (kernel path).  The unified
``_update_streaming_multi`` (now dispatched for the kernel path) must produce the
exact same packed per-step tensors the CUDA kernels consume as the former per-dim
``_update_2d_streaming_multi`` / ``_update_3d_streaming_multi``.  The legacy
methods are retained solely as the oracle.

CPU-only, no FARMS/MuJoCo/CUDA. Builds a synthetic composite whose bodies carry
``_stream_meta`` regular-grid SDF descriptors, runs both paths on identical fresh
state for two steps (the 2nd exercises the prev-union dirty-AABB branch), and
asserts every produced tensor / dict entry matches bit-for-bit.

Run with:
    python -m lilytorch.integration.test_update_streaming_parity
"""
import math
import types
import numpy as np
import torch

from lilytorch.integration.BDIMhandler import BDIMhandler

torch.manual_seed(0)
DT = torch.float64
DEV = 'cpu'


# ----------------------------------------------------------------------
# Synthetic bodies with _stream_meta
# ----------------------------------------------------------------------
class Body:
    pass


def make_stream_meta(D, lo, hi, M, seed):
    """Build a regular-grid SDF descriptor matching _init_interp's layout."""
    g = torch.Generator().manual_seed(seed)
    anames = ('x', 'y', 'z')[:D]
    m = {}
    shape = []
    axes = []
    for ax, a in enumerate(anames):
        n = M[ax]
        coord = torch.linspace(lo[ax], hi[ax], n, dtype=DT)
        axes.append(coord)
        m[f'b{a}'] = coord
        m[f'b{a}0'] = float(coord[0].item())
        m[f'b{a}_last'] = float(coord[-1].item())
        m[f'inv_d{a}'] = float((n - 1) / (hi[ax] - lo[ax]))
        shape.append(n)
    inv_vol = 1.0
    for a in anames:
        inv_vol *= m[f'inv_d{a}']
    m['inv_vol'] = inv_vol
    m['F'] = torch.rand(shape, generator=g, dtype=DT)
    return m


def make_body(D, lo, hi, M, seed, local_pose=None, with_local_aabb=False,
              lagr=False):
    b = Body()
    b.h = 1.0 / 64.0
    b._stream_meta = make_stream_meta(D, lo, hi, M, seed)
    if local_pose is not None:
        b.local_pose = local_pose
    if D == 2 and with_local_aabb:
        b._stream_meta['local_aabb_lo'] = torch.tensor(
            [lo[0] + 0.01, lo[1] + 0.01], dtype=DT)
        b._stream_meta['local_aabb_hi'] = torch.tensor(
            [hi[0] - 0.01, hi[1] - 0.01], dtype=DT)
    if lagr:
        if D == 2:
            th = torch.linspace(0, 2 * math.pi, 40, dtype=DT)
            r = 0.5 * min(hi[0] - lo[0], hi[1] - lo[1]) * 0.6
            b.cnt = torch.stack([r * torch.cos(th), r * torch.sin(th)])
        else:
            u = torch.linspace(0, math.pi, 5, dtype=DT)
            v = torch.linspace(0, 2 * math.pi, 7, dtype=DT)
            U, V = torch.meshgrid(u, v, indexing='ij')
            r = 0.4 * min(hi[ax] - lo[ax] for ax in range(3))
            cen = torch.stack([(r * torch.sin(U) * torch.cos(V)).flatten(),
                               (r * torch.sin(U) * torch.sin(V)).flatten(),
                               (r * torch.cos(U)).flatten()])
            b.tri_centroid_local = cen
            b.tri_normal_local = cen / cen.norm(dim=0, keepdim=True).clamp_min(1e-9)
            b.tri_centroid_world = cen + torch.tensor([[0.05], [0.0], [-0.02]], dtype=DT)
    return b


class Comp:
    pass


def make_comp(D, n, bodies, extent):
    c = Comp()
    h = extent / (n - 1)
    c.h = h
    anames = ('x', 'y', 'z')[:D]
    for ax, a in enumerate(anames):
        coord = torch.linspace(0.0, extent, n, dtype=DT)
        setattr(c, a, coord)
        setattr(c, f'g{a}_1d', coord.contiguous())
    gs = (n,) * D
    c.sdf_val = (torch.arange(np.prod(gs), dtype=DT).reshape(gs) % 7.0)
    B = len(bodies)
    c.bodies = bodies
    c.nbodies = B
    c.body_ids = [(0, i) for i in range(B)]
    c.com_pos = torch.zeros((B, D), dtype=DT)
    c._body_aabbs = [None] * B
    c._sdf_sparse = [None] * B
    return c, gs


def make_handler(D, comp, gs, kin, force_method):
    h = BDIMhandler.__new__(BDIMhandler)
    h.ndim = D
    h.device = DEV
    h.dtype = DT
    h.dtype_np = np.float64
    h._sim_axes = list(range(D))
    h.fluid_solver = types.SimpleNamespace(
        composite_body=comp, grid_shape=gs, _use_kernels=True)
    h.force_method = force_method
    h.gather_data = types.MethodType(lambda self, it: kin, h)
    return h


def snapshot(comp, D):
    s = {
        'sdf_val': comp.sdf_val.clone(),
        'com_pos': comp.com_pos.clone(),
        'body_aabbs': tuple(comp._body_aabbs),
        'union_aabb': tuple(comp._combined_union_aabb),
    }
    ks = comp._kernel_step
    for k, v in ks.items():
        s[f'ks_{k}'] = v.clone() if torch.is_tensor(v) else v
    sm = getattr(comp, '_kernel_static_3d', None) or comp._kernel_static_2d
    for k in ('F_flat', 'F_offsets', 'body_shapes', 'body_meta'):
        s[f'sm_{k}'] = sm[k].clone()
    for i, b in enumerate(comp.bodies):
        if getattr(b, 'cnt_update', None) is not None:
            s[f'cnt_update_{i}'] = b.cnt_update.clone()
        if getattr(b, 'tri_centroid_world', None) is not None:
            s[f'tcw_{i}'] = b.tri_centroid_world.clone()
        if getattr(b, 'tri_normal_world', None) is not None:
            s[f'tnw_{i}'] = b.tri_normal_world.clone()
    return s


def compare(a, b, label):
    keys = set(a) | set(b)
    worst = 0.0
    for k in sorted(keys):
        if k not in a or k not in b:
            print(f"  [{label}] {k} present in only one"); worst = np.inf; continue
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            d = (va - vb).abs().max().item() if va.numel() else 0.0
        elif isinstance(va, (tuple, list)):
            d = 0.0 if va == vb else np.inf
        else:
            d = abs(float(va) - float(vb))
        worst = max(worst, d)
        if d != 0:
            print(f"  [{label}] {k:18s} Δ={d}")
    print(f"  [{'OK ' if worst == 0 else 'FAIL'}] {label:34s} worst={worst:.3e}")
    return worst == 0


def rotmat2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def kin_2d(B, step):
    com = np.zeros((B, 2)); urdf = np.zeros((B, 2)); R = np.zeros((B, 2, 2))
    lin = np.zeros((B, 2)); ang = np.zeros((B,))
    for i in range(B):
        urdf[i] = [0.9 + 0.4 * i + 0.05 * step, 0.7 - 0.2 * i - 0.03 * step]
        com[i] = urdf[i] + [0.01, -0.01]
        R[i] = rotmat2(0.3 + 0.4 * i + 0.1 * step)
        lin[i] = [0.2 - 0.1 * i, -0.15 + 0.05 * i]
        ang[i] = 0.7 - 0.2 * i
    return ([com], [urdf], [R], [lin], [ang])


def kin_3d(B, step):
    from scipy.spatial.transform import Rotation
    com = np.zeros((B, 3)); urdf = np.zeros((B, 3)); R = np.zeros((B, 3, 3))
    lin = np.zeros((B, 3)); ang = np.zeros((B, 3))
    for i in range(B):
        urdf[i] = [0.8 + 0.5 * i + 0.05 * step, 0.6 - 0.2 * i, 0.4 + 0.1 * i + 0.02 * step]
        com[i] = urdf[i] + [0.01, -0.01, 0.005]
        R[i] = Rotation.from_euler('xyz', [0.2 + 0.1 * i + 0.05 * step, 0.3, -0.25 + 0.2 * i]).as_matrix()
        lin[i] = [0.2 - 0.1 * i, -0.12, 0.08 + 0.03 * i]
        ang[i] = [0.5, -0.3 + 0.1 * i, 0.4]
    return ([com], [urdf], [R], [lin], [ang])


def build_bodies(D, lagr):
    if D == 2:
        return [
            make_body(2, [-0.3, -0.3], [0.3, 0.3], [48, 48], 1,
                      with_local_aabb=True, lagr=lagr),
            make_body(2, [-0.25, -0.2], [0.25, 0.2], [40, 36], 2,
                      with_local_aabb=False, lagr=lagr),
        ]
    return [
        make_body(3, [-0.3, -0.3, -0.2], [0.3, 0.3, 0.2], [40, 40, 28], 1,
                  local_pose=[0.02, -0.01, 0.005, 0.1, 0.2, -0.1], lagr=lagr),
        make_body(3, [-0.25, -0.2, -0.15], [0.25, 0.2, 0.15], [34, 30, 24], 2,
                  local_pose=None, lagr=lagr),
    ]


def run_case(D, force_method):
    n = 64
    extent = 3.0
    comp_n, gs = make_comp(D, n, build_bodies(D, force_method == "lagrangian"), extent)
    comp_o, _ = make_comp(D, n, build_bodies(D, force_method == "lagrangian"), extent)
    h_n = make_handler(D, comp_n, gs, None, force_method)
    h_o = make_handler(D, comp_o, gs, None, force_method)
    legacy = h_o._update_3d_streaming_multi if D == 3 else h_o._update_2d_streaming_multi

    ok = True
    kinf = kin_3d if D == 3 else kin_2d
    for step in range(2):
        kin = kinf(2, step)
        h_n.gather_data = types.MethodType(lambda self, it, k=kin: k, h_n)
        h_o.gather_data = types.MethodType(lambda self, it, k=kin: k, h_o)
        h_n._update_streaming_multi(0.0, step)
        legacy(0.0, step)
        ok &= compare(snapshot(comp_n, D), snapshot(comp_o, D),
                      f"{D}D fm={force_method} step={step}")
    return ok


def main():
    ok = True
    for D in (2, 3):
        for fm in ('eulerian', 'lagrangian'):
            ok &= run_case(D, fm)
    print()
    print("STREAMING PARITY: PASSED" if ok else "STREAMING PARITY: FAILED")
    return ok


def test_update_streaming_parity():
    assert main(), "unified _update_streaming_multi diverges from legacy"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
