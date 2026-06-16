"""Parity harness: BDIMhandler._update_python == legacy _update_2d/_update_3d.

Self-test for Step 6 of the 2D/3D solver unification: the unified
``_update_python`` (now dispatched for the Python path) must reproduce the
former per-dim ``_update_2d`` / ``_update_3d`` exactly.  The legacy methods
are retained solely as the oracle this test compares against.

CPU-only, no FARMS/MuJoCo. Builds a synthetic composite + a handler via
__new__ (bypassing __init__), stubs gather_data, runs the unified path and
the legacy per-dim path on identical fresh state, and asserts every output
field matches bit-for-bit (max|Δ| == 0 in fp64) across the Eulerian,
velocity-blend, Lagrangian, and contour-mask variants in 2-D and 3-D.

Run with:
    python -m lilytorch.integration.test_update_python_parity
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
# Synthetic bodies
# ----------------------------------------------------------------------
class Circle:
    def __init__(self, radius, with_aabb, eps=0.05):
        self.radius = float(radius)
        self.h = 1.0 / 64.0
        self.eps = eps
        thetas = torch.linspace(0, 2 * math.pi, 48, dtype=DT)
        self.cnt = torch.stack([radius * torch.cos(thetas),
                                radius * torch.sin(thetas)])
        self.cnt_update = self.cnt.clone()
        if with_aabb:
            m = 4.0 * eps + 4.0 * self.h
            lo = self.cnt.min(dim=1).values - m
            hi = self.cnt.max(dim=1).values + m
            self.local_aabb = torch.stack([lo, hi], dim=0)

    def sdf(self, X, Y):
        return torch.sqrt(X**2 + Y**2) - self.radius


class Sphere:
    def __init__(self, radius, with_aabb, eps=0.05):
        self.radius = float(radius)
        self.h = 1.0 / 64.0
        self.eps = eps
        # surface triangulation markers (just sample points on the sphere)
        u = torch.linspace(0, math.pi, 6, dtype=DT)
        v = torch.linspace(0, 2 * math.pi, 8, dtype=DT)
        U, V = torch.meshgrid(u, v, indexing='ij')
        x = radius * torch.sin(U) * torch.cos(V)
        y = radius * torch.sin(U) * torch.sin(V)
        z = radius * torch.cos(U)
        cen = torch.stack([x.flatten(), y.flatten(), z.flatten()])
        self.tri_centroid_local = cen
        nrm = cen / cen.norm(dim=0, keepdim=True).clamp_min(1e-9)
        self.tri_normal_local = nrm
        # init world = local shifted by a fake bbox-centre offset
        self.tri_centroid_world = cen + torch.tensor([[0.1], [0.0], [-0.05]],
                                                     dtype=DT)
        self.local_pose = None
        if with_aabb:
            m = 4.0 * eps + 4.0 * self.h
            lo = cen.min(dim=1).values - m
            hi = cen.max(dim=1).values + m
            self.local_aabb = torch.stack([lo, hi], dim=0)

    def sdf(self, X, Y, Z):
        return torch.sqrt(X**2 + Y**2 + Z**2) - self.radius


# ----------------------------------------------------------------------
# Synthetic composite
# ----------------------------------------------------------------------
class Comp:
    pass


def make_comp(ndim, n, bodies):
    c = Comp()
    h = 1.0 / (n - 1)
    c.h = h
    ax = torch.linspace(-0.5, 0.5, n, dtype=DT)
    c.x = ax; c.y = ax
    if ndim == 3:
        c.z = ax
    gs = (n,) * ndim
    if ndim == 2:
        c.X, c.Y = torch.meshgrid(ax, ax, indexing='ij')
        # staggered: shift by +-h/2 (consistent, arbitrary)
        xs = ax - h / 2
        c.Xu_stag, c.Yu_stag = torch.meshgrid(xs, ax, indexing='ij')
        c.Xv_stag, c.Yv_stag = torch.meshgrid(ax, xs, indexing='ij')
        c.sdf_val = torch.full(gs, 1e4, dtype=DT)
        c.sdf_val_u = torch.full(gs, 1e4, dtype=DT)
        c.sdf_val_v = torch.full(gs, 1e4, dtype=DT)
        c.body_u = torch.zeros(gs, dtype=DT)
        c.body_v = torch.zeros(gs, dtype=DT)
    else:
        c.X, c.Y, c.Z_grid = torch.meshgrid(ax, ax, ax, indexing='ij')
        xs = ax - h / 2
        c.Xu_stag, c.Yu_stag, c.Zu_stag = torch.meshgrid(xs, ax, ax, indexing='ij')
        c.Xv_stag, c.Yv_stag, c.Zv_stag = torch.meshgrid(ax, xs, ax, indexing='ij')
        c.Xw_stag, c.Yw_stag, c.Zw_stag = torch.meshgrid(ax, ax, xs, indexing='ij')
        c.sdf_val = torch.full(gs, 1e4, dtype=DT)
        c.sdf_val_u = torch.full(gs, 1e4, dtype=DT)
        c.sdf_val_v = torch.full(gs, 1e4, dtype=DT)
        c.sdf_val_w = torch.full(gs, 1e4, dtype=DT)
        c.body_u = torch.zeros(gs, dtype=DT)
        c.body_v = torch.zeros(gs, dtype=DT)
        c.body_w = torch.zeros(gs, dtype=DT)
    B = len(bodies)
    c.bodies = bodies
    c.nbodies = B
    c.body_ids = [(0, i) for i in range(B)]
    c.com_pos = torch.zeros((B, ndim), dtype=DT)
    c._body_aabbs = [None] * B
    c._sdf_sparse = [None] * B
    return c, gs


def make_handler(ndim, comp, gs, kin, blend_cells, force_method,
                 contour_mask=False, prev_idx=None, next_idx=None):
    h = BDIMhandler.__new__(BDIMhandler)
    h.ndim = ndim
    h.device = DEV
    h.dtype = DT
    h.dtype_np = np.float64
    h._sim_axes = list(range(ndim))
    h.fluid_solver = types.SimpleNamespace(composite_body=comp, grid_shape=gs)
    h._blend_eps_cells = blend_cells
    h._blend_den = None
    h._blend_eps = None
    h.force_method = force_method
    h.contour_mask = contour_mask
    h._prev_body_index = prev_idx
    h._next_body_index = next_idx
    h.gather_data = types.MethodType(lambda self, it: kin, h)
    return h


def reset_fields(comp, ndim):
    comp.sdf_val.fill_(1e4)
    comp.sdf_val_u.fill_(1e4); comp.sdf_val_v.fill_(1e4)
    comp.body_u.zero_(); comp.body_v.zero_()
    if ndim == 3:
        comp.sdf_val_w.fill_(1e4); comp.body_w.zero_()
    comp.com_pos.zero_()


def snapshot(comp, ndim):
    s = {
        'sdf_val': comp.sdf_val.clone(),
        'sdf_val_u': comp.sdf_val_u.clone(),
        'sdf_val_v': comp.sdf_val_v.clone(),
        'body_u': comp.body_u.clone(),
        'body_v': comp.body_v.clone(),
        'com_pos': comp.com_pos.clone(),
    }
    if ndim == 3:
        s['sdf_val_w'] = comp.sdf_val_w.clone()
        s['body_w'] = comp.body_w.clone()
    for i, b in enumerate(comp.bodies):
        if getattr(b, 'cnt_update', None) is not None:
            s[f'cnt_update_{i}'] = b.cnt_update.clone()
        if getattr(b, 'r_com', None) is not None:
            s[f'r_com_{i}'] = b.r_com.clone()
        if getattr(b, 'tri_centroid_world', None) is not None:
            s[f'tcw_{i}'] = b.tri_centroid_world.clone()
        if getattr(b, 'tri_normal_world', None) is not None:
            s[f'tnw_{i}'] = b.tri_normal_world.clone()
        if getattr(b, 'mask', None) is not None:
            s[f'mask_{i}'] = b.mask.clone()
    return s


def compare(a, b, label):
    keys = set(a) | set(b)
    worst = 0.0
    for k in keys:
        if k not in a or k not in b:
            raise AssertionError(f"[{label}] key {k} present in only one snapshot")
        va, vb = a[k], b[k]
        if va.dtype == torch.bool:
            d = (va ^ vb).sum().item()
        else:
            d = (va - vb).abs().max().item()
        worst = max(worst, d)
        if d != 0:
            print(f"  [{label}] {k:16s} max|Δ| = {d:.3e}")
    status = "OK " if worst == 0 else "FAIL"
    print(f"  [{status}] {label:28s} worst max|Δ| = {worst:.3e}")
    return worst == 0


def rotmat2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def build_kin_2d(B):
    com = np.zeros((B, 2)); urdf = np.zeros((B, 2)); R = np.zeros((B, 2, 2))
    lin = np.zeros((B, 2)); ang = np.zeros((B,))
    for i in range(B):
        urdf[i] = [(-0.12 + 0.2 * i), 0.05 - 0.03 * i]
        com[i] = urdf[i] + [0.01, -0.01]
        R[i] = rotmat2(0.3 + 0.4 * i)
        lin[i] = [0.2 - 0.1 * i, -0.15 + 0.05 * i]
        ang[i] = 0.7 - 0.2 * i
    return ([com], [urdf], [R], [lin], [ang])


def build_kin_3d(B):
    from scipy.spatial.transform import Rotation
    com = np.zeros((B, 3)); urdf = np.zeros((B, 3)); R = np.zeros((B, 3, 3))
    lin = np.zeros((B, 3)); ang = np.zeros((B, 3))
    for i in range(B):
        urdf[i] = [(-0.1 + 0.18 * i), 0.04 - 0.02 * i, 0.02 + 0.01 * i]
        com[i] = urdf[i] + [0.01, -0.01, 0.005]
        R[i] = Rotation.from_euler('xyz', [0.2 + 0.1 * i, 0.3, -0.25 + 0.2 * i]).as_matrix()
        lin[i] = [0.2 - 0.1 * i, -0.12, 0.08 + 0.03 * i]
        ang[i] = [0.5, -0.3 + 0.1 * i, 0.4]
    return ([com], [urdf], [R], [lin], [ang])


def run_case(ndim, blend_cells, force_method, contour_mask=False):
    n = 40 if ndim == 3 else 96
    if ndim == 2:
        bodies_new = [Circle(0.07, with_aabb=True), Circle(0.06, with_aabb=False)]
        bodies_old = [Circle(0.07, with_aabb=True), Circle(0.06, with_aabb=False)]
        kin = build_kin_2d(2)
    else:
        bodies_new = [Sphere(0.10, with_aabb=True), Sphere(0.08, with_aabb=False)]
        bodies_old = [Sphere(0.10, with_aabb=True), Sphere(0.08, with_aabb=False)]
        kin = build_kin_3d(2)
    prev_idx = next_idx = None
    if contour_mask:
        prev_idx = [None, 0]
        next_idx = [1, None]

    # ---- new unified ----
    comp_n, gs = make_comp(ndim, n, bodies_new)
    h_n = make_handler(ndim, comp_n, gs, kin, blend_cells, force_method,
                       contour_mask, prev_idx, next_idx)
    h_n._update_python(0.0, 0)
    snap_new = snapshot(comp_n, ndim)

    # ---- legacy ----
    comp_o, gs = make_comp(ndim, n, bodies_old)
    h_o = make_handler(ndim, comp_o, gs, kin, blend_cells, force_method,
                       contour_mask, prev_idx, next_idx)
    legacy = h_o._update_2d if ndim == 2 else h_o._update_3d
    legacy(0.0, 0)
    snap_old = snapshot(comp_o, ndim)

    label = f"{ndim}D blend={blend_cells} fm={force_method} cmask={contour_mask}"
    return compare(snap_new, snap_old, label)


def main():
    ok = True
    cases = [
        (2, None, 'eulerian', False),
        (2, 2.0, 'eulerian', False),
        (2, None, 'lagrangian', False),
        (2, None, 'eulerian', True),
        (2, 2.0, 'lagrangian', False),
        (3, None, 'eulerian', False),
        (3, 2.0, 'eulerian', False),
        (3, None, 'lagrangian', False),
        (3, 2.0, 'lagrangian', False),
    ]
    for c in cases:
        ok &= run_case(*c)
    print()
    print("PARITY: PASSED" if ok else "PARITY: FAILED")
    return ok


def test_update_python_parity():
    """pytest entry point."""
    assert main(), "unified _update_python diverges from legacy _update_2d/_update_3d"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
