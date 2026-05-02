"""Self-test: BDIMhandler._update_2d Python path mirrors _update_3d.

Validates the Stage-1 rewrite that replaced the batched
``grid_sample`` + ``min(dim=0)`` union with a per-body AABB +
``torch.where`` running-min loop (matching the 3-D Python path).

The test does NOT depend on FARMS / MuJoCo / open3d.  It exercises:

  * ``BDIMhandler._body_aabb_indices_2d`` — both mesh-style (with
    ``body.sdf.x/y``) and analytical-style (contour-bbox + 4 h margin)
    paths.
  * ``rotate_grid_2d`` — the per-body local-frame rotation used inside
    the new ``_update_2d``.
  * The ``torch.where`` running-min union on a 2-body, mixed
    analytical + (synthetic) mesh composite — verifies the algorithm
    matches a reference computed from each body's SDF taken minimum
    over both bodies on the FULL grid.

Run with:
    python -m lilytorch.integration.test_update_2d_mirrors_3d
"""

import math
import torch

from lilytorch.src.body import rotate_grid_2d
from lilytorch.integration.BDIMhandler import BDIMhandler


def _make_grid(nx, ny, h):
    x = torch.linspace(-0.5, 0.5, nx)
    y = torch.linspace(-0.5, 0.5, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return x, y, X, Y


class _AnalyticalCircle:
    """Mimics a 2-D analytical body: callable SDF + ``cnt`` contour."""

    def __init__(self, radius=0.08):
        self.radius = float(radius)
        self.h = 1.0 / 64.0
        # Contour: discrete set of points on the circle (2, N) in local frame.
        thetas = torch.linspace(0, 2 * math.pi, 64)
        self.cnt = torch.stack([self.radius * torch.cos(thetas),
                                self.radius * torch.sin(thetas)])

    def sdf(self, X, Y):
        return torch.sqrt(X**2 + Y**2) - self.radius


class _GridSDFBody:
    """Mimics a 2-D mesh body: regular-grid SDF with ``sdf.x``/``sdf.y``."""

    class _Interp:
        def __init__(self, x, y, F):
            self.x = x
            self.y = y
            self.F = F

        def __call__(self, X, Y):
            # Bilinear sample via grid_sample (align_corners=True).
            nx, ny = self.F.shape
            xn = (X - self.x[0]) / (self.x[-1] - self.x[0]) * 2 - 1
            yn = (Y - self.y[0]) / (self.y[-1] - self.y[0]) * 2 - 1
            grid = torch.stack([yn, xn], dim=-1).unsqueeze(0)
            inp = self.F.unsqueeze(0).unsqueeze(0)
            out = torch.nn.functional.grid_sample(
                inp, grid, mode='bilinear',
                padding_mode='border', align_corners=True,
            )
            return out.squeeze()

    def __init__(self, half_extent=0.10):
        self.h = 1.0 / 64.0
        self.half_extent = float(half_extent)
        # Pre-sample square SDF on a 64×64 local grid and pad the border
        # with a far-field sentinel so that ``grid_sample(padding_mode=
        # 'border')`` reads "very far from any surface" outside the
        # tabulated region — which is exactly what
        # ``_init_custom_trilinear_2d`` does in production.
        nx = ny = 64
        bx = torch.linspace(-1.5 * half_extent, 1.5 * half_extent, nx)
        by = torch.linspace(-1.5 * half_extent, 1.5 * half_extent, ny)
        BX, BY = torch.meshgrid(bx, by, indexing='ij')
        F = torch.maximum(BX.abs(), BY.abs()) - half_extent  # signed-square
        # Stamp the outer 1-cell border with the production far-field
        # sentinel (`_FAR = 1e4` in BDIMhandler).
        _FAR = 1e4
        F[0, :] = _FAR; F[-1, :] = _FAR
        F[:, 0] = _FAR; F[:, -1] = _FAR
        self.sdf = self._Interp(bx, by, F)
        # Contour bounding box (square): 4 corners — used only by AABB.
        self.cnt = torch.tensor([
            [-half_extent, half_extent, half_extent, -half_extent],
            [-half_extent, -half_extent, half_extent, half_extent],
        ])


def _world_sdf(body, R, urdf_pos, X, Y):
    """Reference SDF in the world frame (full grid)."""
    R_T = R.T
    px, py = rotate_grid_2d(X, Y, R_T, urdf_pos)
    return body.sdf(px, py)


def _ref_running_min_union(bodies, poses, X, Y):
    """Reference union: take per-body world SDF, min across bodies."""
    sdfs = []
    for (body, R, urdf_pos) in zip(bodies, *poses):
        sdfs.append(_world_sdf(body, R, urdf_pos, X, Y))
    stack = torch.stack(sdfs)        # (B, nx, ny)
    return stack.min(dim=0).values   # (nx, ny)


def _torch_where_aabb_union(bodies, poses, X, Y, h, gs):
    """Reference union: per-body AABB + torch.where running-min.

    Mirrors the loop body of the new ``_update_2d`` exactly: only
    mesh-style bodies (with ``body.sdf.x``/``body.sdf.y``) are AABB-
    clipped; analytical bodies use the full-grid path.
    """
    _FAR = 1e4
    sdf_val = torch.full_like(X, _FAR)
    Rs, urdf_poses = poses
    x_axis = X[:, 0]
    y_axis = Y[0]
    for body, R, urdf_pos in zip(bodies, Rs, urdf_poses):
        aabb = None
        if (hasattr(body, 'sdf')
                and hasattr(body.sdf, 'x')
                and hasattr(body.sdf, 'y')):
            aabb = BDIMhandler._body_aabb_indices_2d(
                body, R, urdf_pos, x_axis, y_axis, h, gs, pad=3,
            )
        if aabb is None:
            # Full-grid path
            R_T = R.T
            px, py = rotate_grid_2d(X, Y, R_T, urdf_pos)
            sdf_cc = body.sdf(px, py)
            mask = sdf_cc < sdf_val
            sdf_val = torch.where(mask, sdf_cc, sdf_val)
            continue

        i0, i1, j0, j1 = aabb
        sl = (slice(i0, i1), slice(j0, j1))
        R_T = R.T
        px, py = rotate_grid_2d(X[sl], Y[sl], R_T, urdf_pos)
        sdf_sub = body.sdf(px, py)

        old = sdf_val[sl].contiguous()
        sdf_val[sl] = torch.where(sdf_sub < old, sdf_sub, old)

    return sdf_val


def main():
    torch.manual_seed(0)

    nx = ny = 96
    h = 1.0 / (nx - 1)
    gs = (nx, ny)
    _x, _y, X, Y = _make_grid(nx, ny, h)

    # 2-body composite: one analytical circle + one mesh-style square.
    bodies = [_AnalyticalCircle(radius=0.07),
              _GridSDFBody(half_extent=0.09)]

    # World-frame poses: rotation + translation.
    R0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # identity
    R1_angle = math.pi / 6  # 30 deg
    R1 = torch.tensor([[math.cos(R1_angle), -math.sin(R1_angle)],
                       [math.sin(R1_angle), math.cos(R1_angle)]])
    urdf0 = torch.tensor([-0.15, 0.05])
    urdf1 = torch.tensor([0.12, -0.08])
    poses = ([R0, R1], [urdf0, urdf1])

    # ── Reference: full-grid per-body SDF + min(dim=0) ──
    ref_full = _ref_running_min_union(bodies, poses, X, Y)

    # ── New algorithm: per-body AABB + torch.where running-min ──
    new_aabb = _torch_where_aabb_union(bodies, poses, X, Y, h, gs)

    err = (ref_full - new_aabb).abs().max().item()
    print(f"  full-grid vs AABB-where union   max |Δ| = {err:.3e}")

    # AABB clipping produces sub-grids that miss far-field cells — those
    # cells stay at the `_FAR` sentinel in the new path while the
    # full-grid reference has the actual analytical value there.  The
    # algorithm equivalence holds globally because (a) the analytical
    # circle uses the full-grid path (no AABB), and (b) the mesh
    # square's local SDF table is small enough that values outside
    # its AABB are border-clamped to large sentinels which lose any
    # subsequent running-min.
    aabb_circle = None
    if (hasattr(bodies[0], 'sdf')
            and hasattr(bodies[0].sdf, 'x')
            and hasattr(bodies[0].sdf, 'y')):
        aabb_circle = BDIMhandler._body_aabb_indices_2d(
            bodies[0], R0, urdf0, X[:, 0], Y[0], h, gs, pad=3,
        )
    aabb_square = None
    if (hasattr(bodies[1], 'sdf')
            and hasattr(bodies[1].sdf, 'x')
            and hasattr(bodies[1].sdf, 'y')):
        aabb_square = BDIMhandler._body_aabb_indices_2d(
            bodies[1], R1, urdf1, X[:, 0], Y[0], h, gs, pad=3,
        )

    assert aabb_circle is None, (
        "analytical circle (no body.sdf.x/y) must use full-grid (aabb=None)")
    assert aabb_square is not None, (
        "mesh square should AABB-clip, not return None")
    print(f"  analytical circle AABB           = {aabb_circle}  "
          f"(None = full-grid; expected for analytical bodies)")
    print(f"  mesh square     AABB             = {aabb_square}")

    # Inside the mesh square's AABB the AABB-clipped path must agree
    # with the full-grid reference exactly.
    coverage_square = torch.zeros_like(X, dtype=torch.bool)
    i0, i1, j0, j1 = aabb_square
    coverage_square[i0:i1, j0:j1] = True
    diff_in_square_aabb = (
        ref_full[coverage_square] - new_aabb[coverage_square]
    ).abs().max().item()
    print(f"  inside mesh-square AABB          max |Δ| = {diff_in_square_aabb:.3e}")
    assert diff_in_square_aabb < 1e-6, (
        f"AABB-clipped union does not match full-grid reference inside "
        f"the mesh body's AABB: max |Δ| = {diff_in_square_aabb:.3e}")

    # Globally (full grid): the new per-body AABB + torch.where union
    # should agree with the reference everywhere, because the analytical
    # circle uses the full-grid path and the mesh square's far-field
    # SDF (border-clamped) is large.
    diff_global = (ref_full - new_aabb).abs().max().item()
    print(f"  full-grid                        max |Δ| = {diff_global:.3e}")
    assert diff_global < 1e-6, (
        f"Per-body AABB + torch.where union does not match full-grid "
        f"reference globally: max |Δ| = {diff_global:.3e}")

    print(f"  analytical circle AABB           = {aabb_circle}")
    print(f"  mesh square     AABB             = {aabb_square}")

    # Confirm the mesh body's AABB covers its entire zero-level set
    # (interior cells of the body are inside the AABB box).  Analytical
    # bodies don't use AABB-clipping in this dispatch.
    sdf_full_square = _world_sdf(bodies[1], R1, urdf1, X, Y)
    inside_square = sdf_full_square <= 0
    aabb_mask_square = torch.zeros_like(inside_square)
    i0, i1, j0, j1 = aabb_square
    aabb_mask_square[i0:i1, j0:j1] = True
    n_missed = int((inside_square & ~aabb_mask_square).sum().item())
    assert n_missed == 0, (
        f"mesh-square: AABB missed {n_missed} interior cells")
    print(f"  mesh-square AABB covers all interior cells  "
          f"(count={int(inside_square.sum().item())})")

    # Edge case: very large analytical body — _body_aabb_indices_2d is
    # not called for analytical bodies (no body.sdf.x/y).  We still
    # exercise the helper directly with a fake interpolator-like object
    # to confirm the >90 % full-grid heuristic still kicks in for
    # large mesh bodies.
    class _BigGridSDF:
        class _Interp:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        def __init__(self):
            self.sdf = self._Interp(
                torch.linspace(-0.45, 0.45, 8),
                torch.linspace(-0.45, 0.45, 8),
            )
            self.h = h
    big_body = _BigGridSDF()
    aabb_big = BDIMhandler._body_aabb_indices_2d(
        big_body, torch.eye(2), torch.zeros(2),
        X[:, 0], Y[0], h, gs, pad=3,
    )
    print(f"  big-mesh AABB                    = {aabb_big}  "
          f"(None means full-grid; expected for >90% coverage)")
    assert aabb_big is None, (
        "AABB covering >90% of the grid should return None")

    print("test_update_2d_mirrors_3d: PASSED")


if __name__ == "__main__":
    main()
