
import logging
import math  # used for evaluating math operations for sdfs
import os

import numpy as np
import torch
from lilytorch.src.kernels import RegularGridInterpolator, RegularGridInterpolatorAutomatic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-imported heavy dependencies — only loaded when first used.
# This avoids ~2-3 s of unnecessary startup cost for code paths that never
# call SDF construction, plotting, or mesh operations.
# ---------------------------------------------------------------------------

def _import_model_sdf():
    """Lazy import for farms_core.io.sdf.ModelSDF (only needed for SDF mesh bodies)."""
    from farms_core.io.sdf import ModelSDF
    return ModelSDF

def _import_open3d():
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    return o3d

def _import_cv2():
    import cv2
    return cv2

def _import_skfmm():
    import skfmm
    return skfmm

def _import_measure():
    from skimage import measure
    return measure

def _import_cubic_spline():
    from scipy.interpolate import CubicSpline
    return CubicSpline

def _import_matplotlib():
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    return matplotlib, plt, cm


# ---------------------------------------------------------------------------
# Staggered-grid cache – avoids duplicate meshgrid allocations when several
# Body instances share the same (x, y, z) coordinate vectors.
# ---------------------------------------------------------------------------
class _StaggeredGrids:
    """Pre-computed cell-centre and MAC staggered meshgrids.

    Instances are created and cached by :func:`_get_staggered_grids` so that
    every Body / BodyAnalytical / etc. that lives on the same computational
    domain reuses the **same** tensors (zero extra memory).
    """

    def __init__(self, x, y, z=None):
        h = float(x[1] - x[0])
        ndim = 2 if z is None else 3
        nx, ny = len(x), len(y)

        # ---- cell-centre meshgrid ------------------------------------
        if ndim == 2:
            self.X, self.Y = torch.meshgrid(x, y, indexing="ij")
            self.Z_grid = None
            self.grid_shape = (nx, ny)
        else:
            nz = len(z)
            self.X, self.Y, self.Z_grid = torch.meshgrid(x, y, z, indexing="ij")
            self.grid_shape = (nx, ny, nz)

        # ---- staggered 1-D coordinates --------------------------------
        self.x_stag = x - h / 2
        self.y_stag = y - h / 2

        # ---- staggered meshgrids -------------------------------------
        if ndim == 2:
            self.Xu_stag, self.Yu_stag = torch.meshgrid(self.x_stag, y, indexing="ij")
            self.Xv_stag, self.Yv_stag = torch.meshgrid(x, self.y_stag, indexing="ij")
            # 3-D placeholders
            self.z_stag = None
            self.Zu_stag = self.Zv_stag = None
            self.Xw_stag = self.Yw_stag = self.Zw_stag = None
        else:
            self.z_stag = z - h / 2
            self.Xu_stag, self.Yu_stag, self.Zu_stag = torch.meshgrid(self.x_stag, y, z, indexing="ij")
            self.Xv_stag, self.Yv_stag, self.Zv_stag = torch.meshgrid(x, self.y_stag, z, indexing="ij")
            self.Xw_stag, self.Yw_stag, self.Zw_stag = torch.meshgrid(x, y, self.z_stag, indexing="ij")


# =====================================================================
# Rotation helpers for meshgrid-based SDF evaluation
# =====================================================================
# These avoid flattening + stacking + matmul, operating directly on the
# (nx, ny) or (nx, ny, nz) meshgrid tensors with scalar broadcasting.

def rotate_grid_2d(X, Y, R_T, origin):
    """Rotate 2-D meshgrids into a body's local frame.

    Parameters
    ----------
    X, Y : Tensor  (nx, ny) – meshgrid coordinates
    R_T  : Tensor  (2, 2)   – **transposed** rotation matrix  (R.T)
    origin : Tensor  (2,)   – body-frame origin (URDF position)

    Returns
    -------
    px, py : Tensor  (nx, ny) – coordinates in the body-local frame
    """
    dx = X - origin[0]
    dy = Y - origin[1]
    px = R_T[0, 0] * dx + R_T[0, 1] * dy
    py = R_T[1, 0] * dx + R_T[1, 1] * dy
    return px, py


def rotate_grid_3d(X, Y, Z, R_T, origin):
    """Rotate 3-D meshgrids into a body's local frame.

    Parameters
    ----------
    X, Y, Z : Tensor  (nx, ny, nz) – meshgrid coordinates
    R_T     : Tensor  (3, 3)       – **transposed** rotation matrix  (R.T)
    origin  : Tensor  (3,)         – body-frame origin (URDF position)

    Returns
    -------
    px, py, pz : Tensor  (nx, ny, nz) – coordinates in the body-local frame
    """
    dx = X - origin[0]
    dy = Y - origin[1]
    dz = Z - origin[2]
    px = R_T[0, 0] * dx + R_T[0, 1] * dy + R_T[0, 2] * dz
    py = R_T[1, 0] * dx + R_T[1, 1] * dy + R_T[1, 2] * dz
    pz = R_T[2, 0] * dx + R_T[2, 1] * dy + R_T[2, 2] * dz
    return px, py, pz


# Compiled variant – fuses the 9 element-wise ops into ~1 kernel.
try:
    _rotate_grid_3d_compiled = torch.compile(rotate_grid_3d, mode="reduce-overhead")
except Exception:
    _rotate_grid_3d_compiled = rotate_grid_3d



def _mu_normals_batched(sdf_stack, h, eps):
    """Batched mu0/mu1 and unit normals for N SDF grids (2-D or 3-D).

    Parameters
    ----------
    sdf_stack : (N, *spatial) tensor — N grids stacked along dim 0.
                ``spatial`` is ``(Nx, Ny)`` for 2-D or ``(Nx, Ny, Nz)`` for 3-D.
    h, eps    : grid spacing and BDIM half-width.

    Returns
    -------
    mu0, mu1  : (N, *spatial) Heaviside and delta weightings.
    normals   : (nx, ny) for 2-D or (nx, ny, nz) for 3-D, each (N, *spatial).
    """
    ndim = sdf_stack.ndim - 1          # number of spatial dims
    dims = list(range(1, 1 + ndim))
    spacing = [h] * ndim

    deps = sdf_stack / eps
    s = torch.sin(torch.pi * deps)
    c = torch.cos(torch.pi * deps)
    mu0 = torch.where(
        sdf_stack <= -eps, torch.zeros_like(sdf_stack),
        torch.where(sdf_stack >= eps, torch.ones_like(sdf_stack),
                    0.5 * (1 + deps + s / torch.pi)))
    mu1 = torch.where(
        torch.abs(sdf_stack) >= eps, torch.zeros_like(sdf_stack),
        eps * (0.25 - (0.5 * deps) ** 2
               - (s * deps + (1 + c) / torch.pi) / (2 * torch.pi)))

    grads = torch.gradient(sdf_stack, spacing=spacing, dim=dims, edge_order=2)
    norm = torch.sqrt(sum(g ** 2 for g in grads))
    inv_norm = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))
    return (mu0, mu1) + tuple(g * inv_norm for g in grads)



# Module-level cache:  (data_ptr_x, data_ptr_y, data_ptr_z) -> _StaggeredGrids
_grid_cache: dict[tuple, _StaggeredGrids] = {}


def _get_staggered_grids(x, y, z=None) -> _StaggeredGrids:
    """Return (possibly cached) staggered grids for the given coordinate vectors."""
    key = (x.data_ptr(), y.data_ptr(), z.data_ptr() if z is not None else None)
    if key not in _grid_cache:
        _grid_cache[key] = _StaggeredGrids(x, y, z)
    return _grid_cache[key]

"""
Analytical SDFs
"""
def circle(x,y,xt=0,yt=60,r=25):
    return torch.sqrt((x-xt)**2+(y-yt)**2)-r


def sphere(x, y, z, xt=0, yt=0, zt=0, r=25):
    return torch.sqrt((x - xt)**2 + (y - yt)**2 + (z - zt)**2) - r


def sdUnevenCapsule(Y, X, r1, r2, h, side="L"):
    if side=="L":
        X = -X
    Yabs=torch.abs(Y)
    b=(r1-r2)/h
    a=torch.sqrt(1.0-b*b)
    k=-b*Yabs+a*X
    return torch.where(
        k<0.0, torch.sqrt(Yabs**2+X**2)-r1,
        torch.where(
            k>a*h, torch.sqrt(Yabs**2+(X-h)**2)-r2,
            a*Yabs+b*X-r1
        )
    )


def segment(X,Y,A,B,r1,r2):
    pa_x=X-A[0]
    pa_y=Y-A[1]
    ba=B-A
    h=torch.clamp((pa_x*ba[0]+pa_y*ba[1])/torch.dot(ba,ba),0.0,1.0)
    return torch.sqrt(
        (pa_x-h*ba[0])**2+(pa_y-h*ba[1])**2
    )-(r1+h*(r2-r1))


def capsule_3d(x, y, z, r1, r2, h, side="L"):
    # SDF/MuJoCo capsules are aligned with the local z-axis and centred on the
    # geom frame; the cylindrical section runs from -h/2 to +h/2.
    radial = torch.sqrt(x**2 + y**2)
    axis = z + 0.5 * h
    return sdUnevenCapsule(radial, axis, r1, r2, h, side="R")

def box(x,y,xb=20,yb=20):
    qx=torch.abs(x)-xb
    qy=torch.abs(y)-yb
    return torch.sqrt(
        torch.maximum(qx,torch.zeros_like(x))**2 +
        torch.maximum(qy,torch.zeros_like(y))**2
    )+torch.minimum(torch.maximum(qx,qy),torch.zeros_like(x))


def box_3d(x, y, z, xb=20, yb=20, zb=20):
    qx = torch.abs(x) - xb
    qy = torch.abs(y) - yb
    qz = torch.abs(z) - zb
    return (
        torch.sqrt(
            torch.maximum(qx, torch.zeros_like(x))**2 +
            torch.maximum(qy, torch.zeros_like(y))**2 +
            torch.maximum(qz, torch.zeros_like(z))**2
        )
        + torch.minimum(
            torch.maximum(torch.maximum(qx, qy), qz),
            torch.zeros_like(x),
        )
    )

def resample_contour_exact_spacing(x, y, spacing, closed=True):
    """
    Resample contour with exactly constant spacing
    """
    if closed:
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

    # Convert to points array
    points = np.column_stack([x, y])
    resampled_points = [points[0]]

    # Walk along contour with exact spacing
    current_position = points[0].copy()
    segment_idx = 0
    distance_along_segment = 0.0

    while segment_idx < len(points) - 1:
        # Current segment
        segment_start = points[segment_idx]
        segment_end = points[segment_idx + 1]
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)

        # Distance remaining in current segment
        remaining_in_segment = segment_length - distance_along_segment

        if remaining_in_segment >= spacing:
            # Place next point within current segment
            if segment_length > 1e-12:  # Avoid division by zero
                direction = segment_vector / segment_length
                current_position = segment_start + (distance_along_segment + spacing) * direction
                resampled_points.append(current_position.copy())
                distance_along_segment += spacing
            else:
                # Degenerate segment, skip
                segment_idx += 1
                distance_along_segment = 0.0
        else:
            # Move to next segment
            segment_idx += 1
            distance_along_segment = spacing - remaining_in_segment

    resampled_points = np.array(resampled_points)

    # Compute arc-length coordinates
    if len(resampled_points) > 1:
        diffs = np.diff(resampled_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        s_uniform = np.concatenate(([0], np.cumsum(distances)))
    else:
        s_uniform = np.array([0])

    return resampled_points[:, 0], resampled_points[:, 1], s_uniform

def resample_contour(x, y, spacing, closed=True):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        s_uniform = np.arange(0, s[-1], spacing)
        x_new = np.interp(s_uniform, s, x)
        y_new = np.interp(s_uniform, s, y)
        x_new = np.r_[x_new, x_new[0]]
        y_new = np.r_[y_new, y_new[0]]

        return x_new, y_new, s_uniform

def compute_inertias_2d(sdf_fun, inside_mask, x, y, x_g, y_g, density=1000.0):
    """
    Compute the inertial properties of a 2D shape defined by an SDF over a grid
    sdf_fun: function that takes (N,M,2) array of points and returns (N,M) array of sdf values
    inside_mask: (N,M) boolean array where True indicates points inside the shape
    x: (N,) array of x coordinates of the grid
    y: (M,) array of y coordinates of the grid
    x_g, y_g: coordinates of the centroid
    density: material density
    """
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    xx, yy = np.meshgrid(x, y)

    dA = dx * dy
    mass = density * np.sum(inside_mask) * dA


    # Raw moments about the origin
    I_x = np.sum((yy[inside_mask]**2) * dA) # around x-axis (horizontal), i.e. y^2 dA
    I_y = np.sum((xx[inside_mask]**2) * dA) # around y-axis (vertical), i.e. x^2 dA
    I_xy = np.sum((xx[inside_mask]*yy[inside_mask]) * dA)

    # Shift to centroid using parallel axis theorem
    I_x_centroid = I_x - mass * y_g**2
    I_y_centroid = I_y - mass * x_g**2
    I_xy_centroid = I_xy - mass * x_g * y_g

    return mass, I_x_centroid, I_y_centroid, I_xy_centroid


def body_from_yaml(device, x, y, body_pars, eps=0.05, custom_update=None, starting_time=0, z=None, **kwargs):

    if custom_update is not None:
        update_map = custom_update

    body_type = body_pars["type"]

    if body_type == "composite_analytical":
        sdf_funs = body_pars["sdf"]
        plotting=body_pars["plotting"]
        update_maps = body_pars["update_maps"]
        return CompositeBodyAnalytical(
            device, x, y,
            [eval(sdf_fun) for sdf_fun in sdf_funs],
            [
                (
                    eval(update_map["rotation"]),
                    tuple(eval(s) for s in update_map["translation"])
                ) for update_map in update_maps
            ],
            z=z,
            eps=eps,
            plotting=plotting,
        )

    elif body_type == "composite_mesh":
        sdf_name = body_pars["sdf_name"]
        sdf_folder = body_pars["sdf_folder"]
        nsamples, msamples, ksamples = None, None, None
        if "n_samples" in body_pars and body_pars["n_samples"] is not None:
            _ns = eval(body_pars["n_samples"])
            nsamples, msamples = _ns[0], _ns[1]
            if len(_ns) >= 3:
                ksamples = _ns[2]
        compute_interp = body_pars["compute_interp"]
        plotting= body_pars["plotting"]
        plotting_meshes = body_pars["plotting_meshes"]
        return CompositeBodyMesh(
            device, x, y,
            sdf_folder, sdf_name,
            custom_update,
            eps             = eps,
            compute_interp  = compute_interp,
            nsamples        = nsamples,
            msamples        = msamples,
            ksamples        = ksamples,
            plotting        = plotting,
            plotting_meshes = plotting_meshes,
            suit            = body_pars["suit"],
            convexify       = body_pars["convexify"],
            scale           = body_pars["scale"],
            **kwargs
        )

    elif body_type == "multi_animat":

        nsamples, msamples, ksamples = None, None, None
        if "n_samples" in body_pars and body_pars["n_samples"] is not None:
            _ns = body_pars["n_samples"]
            nsamples, msamples = _ns[0], _ns[1]
            if len(_ns) >= 3:
                ksamples = _ns[2]

        return MultiAnimatBodies(
            device, x, y,
            experiment_options = body_pars["experiment_options"],
            z                  = z,
            eps                = eps,
            compute_interp     = body_pars["compute_interp"],
            nsamples           = nsamples,
            msamples           = msamples,
            ksamples           = ksamples,
            plotting           = body_pars["plotting"],
            plotting_meshes    = body_pars["plotting_meshes"],
            suit               = body_pars["suit"],
            convexify          = body_pars["convexify"],
            scale              = body_pars["scale"],
            save_folder        = body_pars["save_folder"],
            use_kernels        = kwargs.pop("use_kernels", False),
            **kwargs
        )


    elif body_type == "fish_analytical":
        control_pars = body_pars["control"]
        return BodyFishAnalytical(
            device, x, y,
            eps=eps,
            L=control_pars["L"], A=control_pars["A"], f=control_pars["f"],
            wavefrequency=control_pars["wavefrequency"],
            c1=control_pars["c1"], c2=control_pars["c2"], c3=control_pars["c3"],
            xshift=control_pars["xshift"], yshift=control_pars["yshift"],
            sb=control_pars["sb"], wh=control_pars["wh"], st=control_pars["st"], wt=control_pars["wt"], thk=control_pars["thk"],
        )

    elif body_type == "fish_experimental":
        control_pars = body_pars["control"]
        return BodyFishExperimental(
            device, x, y,
            eps             = eps,
            body_length     = control_pars["body_length"],
            folder_name     = control_pars["folder_name"],
            file_name       = control_pars["file_name"],
            save_data       = control_pars["save_data"],
            plot_data       = control_pars["plot_data"],
            target_fish     = control_pars["target_fish"],
            start_recording = control_pars["start_recording"],
            end_recording   = control_pars["end_recording"],
            timestep        = control_pars["timestep"],
            total_duration  = control_pars["total_duration"],
            freq_scaling    = control_pars["freq_scaling"],
            filter_freqs    = control_pars["filter_freqs"],
            xshift          = control_pars["xshift"],
            yshift          = control_pars["yshift"],
            initial_time    = starting_time,
        )

    elif body_type == "composite_segment_body":
        sdf_name = body_pars["sdf_name"]
        sdf_folder = body_pars["sdf_folder"]
        return CompositeSegmentBody(
                    device, x, y,
                    sdf_folder, sdf_name,
                    eps=eps,
                )

class mesh2sdf():
    """
    It is assumed that all vector inputs are numpy arrays
    """
    def __init__(self, mesh_file, convexify=True, scale=1):
        o3d = _import_open3d()
        self.mesh_file = mesh_file
        self._mesh = o3d.io.read_triangle_mesh(self.mesh_file)
        self.update_mesh(convexify=convexify, scale=scale)

    def update_mesh(self, convexify, scale):
        o3d = _import_open3d()
        self._mesh = self._mesh.scale(scale, (0,0,0)) #self._mesh.get_center())
        if convexify:
            self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh.compute_convex_hull()[0])
        else:
            self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh)

        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._ = self._raycasting_scene.add_triangles(self._mesht)
        self._mesh.compute_triangle_normals()
        self._face_normals = np.asarray(self._mesh.triangle_normals)
        self._sign_nsamples = 11

    def __call__(self, points_in_object_frame: np.array):
        signed_distance = self._raycasting_scene.compute_signed_distance(
            points_in_object_frame,
            nsamples=self._sign_nsamples,
        ).numpy()

        closest = self._raycasting_scene.compute_closest_points(points_in_object_frame)
        closest_points = closest['points']
        face_ids = closest['primitive_ids']
        pts = closest_points.numpy()
        # negative SDF gradient outside the object and positive SDF gradient inside the object
        gradient = pts - points_in_object_frame

        distance = np.abs(signed_distance)
        # normalize gradients
        has_direction = distance > 0
        gradient[has_direction] = gradient[has_direction] / distance[has_direction, None]

        is_inside = signed_distance < 0
        # fix gradient direction to point away from surface outside
        gradient[~is_inside] = gradient[~is_inside] * -1

        # for any points very close to the surface, it is better to use the surface normal as the gradient
        # this is because the closest point on the surface may be noisy when close by
        # e.g. if you are actually on the surface, the closest surface point is itself so you get no gradient info
        on_surface = distance < 1e-3
        surface_normals = self._face_normals[face_ids.numpy()[on_surface]]
        gradient[on_surface] = surface_normals

        return signed_distance, gradient

    def bounding_box(self, padding=0., padding_ratio=0):
        aabb = self._mesh.get_axis_aligned_bounding_box()
        world_min = aabb.get_min_bound()
        world_max = aabb.get_max_bound()
        # already scaled, but we add a little padding
        ranges = np.array(list(zip(world_min, world_max)))
        extents = ranges[:, 1] - ranges[:, 0]
        ranges[:, 0] -= padding + padding_ratio * extents
        ranges[:, 1] += padding + padding_ratio * extents
        return ranges

    def visualize(self, wireframe=True):
        o3d = _import_open3d()

        print("Visualizing the mesh file: {}".format(self.mesh_file))

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        if wireframe:
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(self._mesh)
            viewer.add_geometry(line_set)
        else:
            viewer.add_geometry(self._mesh)
        opt = viewer.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()


class COMPOSITEmesh2sdf():

    def __init__(self, sdf_name, sdf_folder):
        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """
        self.sdf = _import_model_sdf().read(sdf_folder+sdf_name)[0]
        self.sdfs = []
        for link in self.sdf.links:
            mesh_name = link["visuals"][0]["geometry"]["uri"]
            sdf = mesh2sdf(sdf_folder+mesh_name)
            # initial translation according to the initial poses in the world reference frame (assumes no initial rotation)
            # sdf.translate_3d(link.pose[:3])
            self.sdfs.append(sdf)

    def transform_3d(self, quat_list=[], center_list=[], pos_list=[]):
        """Apply quaternion rotations and translations to each link mesh.

        .. note::
           Not yet implemented — the underlying ``mesh2sdf`` class does not
           expose a ``transform_3d`` method.  Add the required mesh
           transformation logic to ``mesh2sdf`` first.
        """
        raise NotImplementedError(
            "COMPOSITEmesh2sdf.transform_3d requires mesh2sdf.transform_3d "
            "which has not been implemented yet."
        )


    def __call__(self, points_in_object_frame: np.array):

        sdfv = []
        sdfg = []
        for i, sdf in enumerate(self.sdfs):
            v, g = sdf(points_in_object_frame)
            sdfv.append(v)
            sdfg.append(g)
        return sdfv, sdfg


    def visualize(self):
        o3d = _import_open3d()

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        for sdf in self.sdfs:
            viewer.add_geometry(o3d.geometry.LineSet.create_from_triangle_mesh(sdf._mesh))
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()

class Body:

    def __init__(self, device, x, y, z=None, eps=0.05):
        """Base class for immersed bodies on a MAC staggered grid.

        Works in 2-D (z is None) or 3-D (z is a 1-D tensor).

        Staggered meshgrids (``X``, ``Xu_stag``, etc.) are *not* stored here.
        Each composite / standalone body class calls ``_setup_grids()`` after
        ``super().__init__()`` to bind the shared ``_StaggeredGrids`` object
        (and its meshgrid attributes) to *that* body only.  Child bodies
        inside a ``CompositeBodyAnalytical`` / ``MultiAnimatBodies`` do not
        call ``_setup_grids()``, so they carry no redundant grid references.
        """
        self.device = device
        self.dtype  = x.dtype
        self.h      = float(x[1] - x[0])
        self.eps    = eps

        # ---- dimensionality -------------------------------------------
        self.x = x
        self.y = y
        self.z = z
        self.nx = len(x)
        self.ny = len(y)
        self.ndim = 2 if z is None else 3

        if z is not None:
            self.nz = len(z)

        # grid_shape is always available (no meshgrid allocation needed)
        if z is None:
            self.grid_shape = (self.nx, self.ny)
        else:
            self.grid_shape = (self.nx, self.ny, self.nz)

        self.rad_conv   = (torch.pi / 180)

    def _setup_grids(self):
        """Bind full staggered meshgrids to this body instance.

        Called eagerly by composite / standalone body classes (e.g.
        ``CompositeBodyAnalytical``, ``BodyFishAnalytical``,
        ``CompositeBodyMesh``) after ``super().__init__()``, and by
        ``MultiAnimatBodies`` when ``use_kernels=False`` (python mode).
        In kernel mode ``MultiAnimatBodies`` skips this call entirely so
        no staggered-grid tensors are allocated.  Child ``BodyAnalytical``
        instances inside a
        composite do *not* call this.
        """
        g = _get_staggered_grids(self.x, self.y, self.z)
        self._grids  = g
        self.X       = g.X
        self.Y       = g.Y
        if self.ndim == 3:
            self.Z_grid = g.Z_grid
        self.x_stag  = g.x_stag
        self.y_stag  = g.y_stag
        self.Xu_stag = g.Xu_stag
        self.Yu_stag = g.Yu_stag
        self.Xv_stag = g.Xv_stag
        self.Yv_stag = g.Yv_stag
        if self.ndim == 3:
            self.z_stag  = g.z_stag
            self.Zu_stag = g.Zu_stag
            self.Zv_stag = g.Zv_stag
            self.Xw_stag = g.Xw_stag
            self.Yw_stag = g.Yw_stag
            self.Zw_stag = g.Zw_stag

    def compute_normals(self, sdf_val):
        """Compute unit normals from an SDF field (2-D or 3-D).

        Returns
        -------
        2-D: (nx, ny)
        3-D: (nx, ny, nz)
        """
        ndim = sdf_val.ndim
        spacing = [self.h] * ndim

        grads = torch.gradient(sdf_val, spacing=spacing, edge_order=2)
        norm = torch.sqrt(sum(g ** 2 for g in grads))
        inv_norm = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))

        normals = tuple(g * inv_norm for g in grads)
        return normals

    def compute_normals_3d_batched(self, sdf_vals_4):
        """Compute unit normals for 4 stacked SDF grids in one pass.

        Parameters
        ----------
        sdf_vals_4 : (4, Nx, Ny, Nz) tensor — the p/u/v/w SDF fields stacked
                     along dimension 0.

        Returns
        -------
        (nx, ny, nz) : each (4, Nx, Ny, Nz) — batched unit normals.
        """
        h = self.h
        gx, gy, gz = torch.gradient(sdf_vals_4, spacing=[h, h, h],
                                     dim=[1, 2, 3], edge_order=2)
        norm = torch.sqrt(gx**2 + gy**2 + gz**2)
        inv_norm = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))
        nx = gx * inv_norm
        ny = gy * inv_norm
        nz = gz * inv_norm
        return (nx, ny, nz)

    def mu_funcs_batched(self, d):
        """Heaviside mu_0 and mu_1 — works on any shape (including batched).

        Narrow-band optimised: sin/cos are only evaluated where |d| < eps,
        which is typically < 5 % of the grid, giving a large speedup.

        Parameters
        ----------
        d : tensor of any shape (e.g. (3, Nx, Ny) or (4, Nx, Ny, Nz)).

        Returns
        -------
        (mu_0, mu_1) : tensors with the same shape as d.
        """
        eps = self.eps
        # Pre-fill: 0 inside body (d<0), 1 in fluid (d>=0);
        # band values will be overwritten below.
        mu_0 = (d >= 0).to(d.dtype)
        mu_1 = torch.zeros_like(d)

        band = (d > -eps) & (d < eps)
        d_b  = d[band]
        deps = d_b / eps
        s = torch.sin(torch.pi * deps)
        c = torch.cos(torch.pi * deps)
        mu_0[band] = 0.5 * (1 + deps + s / torch.pi)
        mu_1[band] = eps * (0.25 - (0.5 * deps)**2
                            - (s * deps + (1 + c) / torch.pi) / (2 * torch.pi))
        return (mu_0, mu_1)

    def phi(self,d):
        # return 0.5+0.5*torch.cos(torch.pi*d.clamp(-1,1))
        return torch.where(
            torch.abs(d)<self.eps,
            ( 1 + torch.cos(torch.pi*d/self.eps) )/( 2*self.eps ),
            torch.zeros_like(d)
        )


    def mu_funcs(self, d):
        """Narrow-band optimised: sin/cos only where |d| < eps."""
        eps = self.eps
        mu_0 = (d >= 0).to(d.dtype)
        mu_1 = torch.zeros_like(d)

        band = (d > -eps) & (d < eps)
        d_b  = d[band]
        deps = d_b / eps
        s = torch.sin(torch.pi * deps)
        c = torch.cos(torch.pi * deps)
        mu_0[band] = 0.5 * (1 + deps + s / torch.pi)
        mu_1[band] = eps * (0.25 - (0.5 * deps)**2
                            - (s * deps + (1 + c) / torch.pi) / (2 * torch.pi))
        return (mu_0, mu_1)


class BodyAnalytical(Body):

    def __init__(self, device, x, y, sdf, update_maps, z=None, eps=0.05, plotting=False, pre_update=True, local_aabb=None):
        super().__init__(device, x, y, z=z, eps=eps)
        self.sdf = sdf
        self.update_theta = update_maps[0]
        self.update_translation = update_maps[1]
        self.plotting = plotting
        self.body = self
        self.pre_update = pre_update
        # Optional body-local AABB ``[[lo_x, lo_y[, lo_z]], [hi_x, hi_y[, hi_z]]]``
        # used by :class:`BDIMhandler` to crop per-body SDF evaluation.
        # In both 2-D and 3-D this is auto-derived from the analytical
        # zero-level set during ``_initialize_2d`` / ``_initialize_3d``
        # (2-D: ``measure.find_contours``; 3-D: ``measure.marching_cubes``)
        # with a safety margin large enough that the analytical SDF
        # outside the AABB is provably ≥ band radius, so cells outside
        # don't affect the running-min union of bodies.  An explicit
        # ``local_aabb=torch.tensor([[xmin,ymin,zmin],[xmax,ymax,zmax]])``
        # passed in the constructor wins (used when the user knows a
        # tighter / looser bound than the auto-derived one, or when
        # the local grid does not capture a zero-level set).
        self.local_aabb = local_aabb
        self.initialize()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialize(self):
        """Compute initial contour (2-D only) and SDF."""

        if self.ndim == 2:
            self._initialize_2d()
        else:
            self._initialize_3d()

    def _initialize_2d(self):
        """2-D initialisation: find contour, resample, set up arrays."""
        measure = _import_measure()
        xmid = (self.x.min() + self.x.max()) / 2
        ymid = (self.y.min() + self.y.max()) / 2
        xcnt = self.x - xmid
        ycnt = self.y - ymid

        X, Y = torch.meshgrid(xcnt, ycnt, indexing="ij")
        sdf_cnt = self.sdf(X, Y)

        sdf_np = sdf_cnt.cpu().numpy()
        xnp = xcnt.cpu().numpy()
        ynp = ycnt.cpu().numpy()

        cnt = np.array(measure.find_contours(sdf_np - self.h, 0)[0]).T
        cnt[0] = xnp[0] + cnt[0] * (xnp[1] - xnp[0])
        cnt[1] = ynp[0] + cnt[1] * (ynp[1] - ynp[0])

        ds = self.h
        x, y, s_uniform = resample_contour(cnt[0], cnt[1], spacing=ds, closed=True)
        del cnt
        cnt = np.array([x, y])

        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        curv_coord = np.concatenate(([0], np.cumsum(ds)))

        self.curv_coord = torch.from_numpy(curv_coord).type(self.dtype).to(self.device)
        self.cnt = torch.from_numpy(cnt).type(self.dtype).to(self.device)
        self.cnt_update = self.cnt.clone().detach()
        self.ds = self.curv_coord[1] - self.curv_coord[0]

        # ──────────────────────────────────────────────────────────────
        # Local-frame AABB for analytical bodies (2-D)
        # ──────────────────────────────────────────────────────────────
        # ``self.cnt`` traces the (offset) zero-level set of the local
        # SDF.  Expand its extent by ``band_margin`` so that any point
        # outside the AABB is guaranteed to lie outside the BDIM band
        # of width ``2*eps`` (Lipschitz-1 SDF ⇒ |sdf| ≥ band_margin
        # outside the box).  Outside the band the body contributes
        # only ``mu=1`` (pure fluid), so cells outside the AABB can be
        # safely skipped during the per-body running-min union — they
        # remain at ``_FAR`` (or whatever closer body wrote there),
        # which is equivalent to "this analytical body doesn't matter
        # here" for downstream BDIM/forces stages.
        # An explicit ``local_aabb`` provided in the constructor wins.
        if self.local_aabb is None:
            band_margin = float(self.eps) + 4.0 * float(self.h)
            cnt_lo = self.cnt.min(dim=1).values - band_margin
            cnt_hi = self.cnt.max(dim=1).values + band_margin
            self.local_aabb = torch.stack([cnt_lo, cnt_hi], dim=0)

        if self.plotting:
            _, plt, cm = _import_matplotlib()
            plt.imshow(
                sdf_np.T,
                extent=(
                    torch.min(self.x.cpu()), torch.max(self.x.cpu()),
                    torch.min(self.y.cpu()), torch.max(self.y.cpu())
                ),
                origin="lower", cmap="Greys",
            )
            plt.colorbar()
            cmap = cm.get_cmap('RdBu')
            plt.plot(self.cnt_update[0].cpu(), self.cnt_update[1].cpu())
            plt.show()

        self.cnt_u = torch.zeros_like(self.cnt_update[0])
        self.cnt_v = torch.zeros_like(self.cnt_update[1])
        self.cnt_f_u = torch.zeros_like(self.cnt_update[0])
        self.cnt_f_v = torch.zeros_like(self.cnt_update[1])
        self.cnt_int_f_u = torch.zeros_like(self.cnt_update[0])
        self.cnt_int_f_v = torch.zeros_like(self.cnt_update[1])
        self.mask = torch.arange(len(self.curv_coord), device=self.device)
        self.com_pos = torch.zeros(2, device=self.device, dtype=self.dtype)

        if self.pre_update:
            self.update(torch.tensor(0.0), 0, update_cnt=False)

    def _initialize_3d(self):
        """3-D initialisation: no contour; just set up placeholder arrays."""
        # 3-D bodies don't have 1-D contour representations.
        # We set minimal stubs so that solver code doesn't crash
        # when checking for these attributes.
        self.cnt = torch.zeros((3, 1), device=self.device, dtype=self.dtype)
        self.cnt_update = self.cnt.clone().detach()
        self.curv_coord = torch.tensor([0, 1], device=self.device, dtype=self.dtype)
        self.ds = self.curv_coord[1] - self.curv_coord[0]
        self.cnt_u = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_v = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_w = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_f_u = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_f_v = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_f_w = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_int_f_u = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_int_f_v = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_int_f_w = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.mask = torch.arange(1, device=self.device)
        self.com_pos = torch.zeros(3, device=self.device, dtype=self.dtype)

        # ──────────────────────────────────────────────────────────────
        # Local-frame AABB for analytical bodies (3-D)
        # ──────────────────────────────────────────────────────────────
        # Mirror of the 2-D path: sample the analytical SDF on a
        # body-local centred grid built from ``self.x/y/z``, run
        # marching cubes at level ``self.h`` to extract the (offset)
        # zero-level set, and use the min/max of the surface vertices
        # plus ``band_margin = eps + 4*h`` as the AABB.  Lipschitz-1
        # SDF ⇒ |sdf| ≥ band_margin outside that box, so cells outside
        # contribute only ``mu=1`` (pure fluid) and can be safely
        # skipped during the per-body running-min union.
        #
        # Skipped when:
        #   * the user already passed ``local_aabb`` to the ctor;
        #   * the local grid does not contain a zero-level set
        #     (e.g. body wholly outside ``self.x/y/z``); marching
        #     cubes raises ``RuntimeError``/``ValueError`` and we
        #     fall back to the full-grid path.
        if self.local_aabb is None:
            try:
                measure = _import_measure()
                xmid = (self.x.min() + self.x.max()) / 2
                ymid = (self.y.min() + self.y.max()) / 2
                zmid = (self.z.min() + self.z.max()) / 2
                xcnt = self.x - xmid
                ycnt = self.y - ymid
                zcnt = self.z - zmid

                X, Y, Z = torch.meshgrid(xcnt, ycnt, zcnt, indexing="ij")
                sdf_cnt = self.sdf(X, Y, Z)

                sdf_np = sdf_cnt.cpu().numpy()
                xnp = xcnt.cpu().numpy()
                ynp = ycnt.cpu().numpy()
                znp = zcnt.cpu().numpy()

                verts, _faces, _normals, _vals = measure.marching_cubes(
                    sdf_np, level=float(self.h)
                )
                # ``verts`` are in voxel-index space; convert each
                # column to physical body-local coordinates.
                vx = xnp[0] + verts[:, 0] * (xnp[1] - xnp[0])
                vy = ynp[0] + verts[:, 1] * (ynp[1] - ynp[0])
                vz = znp[0] + verts[:, 2] * (znp[1] - znp[0])

                band_margin = float(self.eps) + 4.0 * float(self.h)
                cnt_lo = torch.tensor(
                    [float(vx.min()) - band_margin,
                     float(vy.min()) - band_margin,
                     float(vz.min()) - band_margin],
                    device=self.device, dtype=self.dtype,
                )
                cnt_hi = torch.tensor(
                    [float(vx.max()) + band_margin,
                     float(vy.max()) + band_margin,
                     float(vz.max()) + band_margin],
                    device=self.device, dtype=self.dtype,
                )
                self.local_aabb = torch.stack([cnt_lo, cnt_hi], dim=0)
            except (RuntimeError, ValueError, ImportError):
                # No zero-level set in the local grid (or skimage not
                # importable); leave ``local_aabb`` as ``None`` so the
                # BDIMhandler falls through to the full-grid path.
                self.local_aabb = None

        if self.pre_update:
            self.update(torch.tensor(0.0), 0, update_cnt=False)

    # ------------------------------------------------------------------
    # Roto-translation
    # ------------------------------------------------------------------
    def rototranslate_points(self, t):
        """Build rotation matrix and translation vector.

        Returns
        -------
        transl : Tensor  (2,) or (3,) – centre-of-mass position
        rot    : Tensor  (2,2) or (3,3) – rotation matrix

        2-D: scalar theta  → 2×2 rotation
        3-D: update_theta returns (θx, θy, θz) Euler angles (deg)
             → 3×3 rotation Rz·Ry·Rx
        """
        if self.ndim == 2:
            transl = torch.tensor([
                self.update_translation[0](t),
                self.update_translation[1](t),
            ], device=self.device, dtype=self.dtype)

            _theta_raw = self.update_theta(t)
            theta = self.rad_conv * (
                _theta_raw.clone().detach().to(device=self.device, dtype=self.dtype)
                if isinstance(_theta_raw, torch.Tensor)
                else torch.tensor(_theta_raw, device=self.device, dtype=self.dtype)
            )
            self.com_pos = transl

            s, c = torch.sin(theta), torch.cos(theta)
            rot = torch.stack([torch.stack([c, -s]),
                               torch.stack([s, c])])
            return (transl, rot)

        else:  # 3-D
            transl = torch.tensor([
                self.update_translation[0](t),
                self.update_translation[1](t),
                self.update_translation[2](t),
            ], device=self.device, dtype=self.dtype)

            angles_raw = self.update_theta(t)
            # Accept scalar (rotate about z only) or 3-tuple Euler (x,y,z)
            is_scalar = (isinstance(angles_raw, (int, float))
                         or (isinstance(angles_raw, torch.Tensor) and angles_raw.dim() == 0))
            if is_scalar:
                angles_raw = (0.0, 0.0, angles_raw)
            ax, ay, az = [
                self.rad_conv * (a.clone().detach().to(device=self.device, dtype=self.dtype)
                                 if isinstance(a, torch.Tensor)
                                 else torch.tensor(a, device=self.device, dtype=self.dtype))
                for a in angles_raw
            ]
            self.com_pos = transl

            # Rx
            sx, cx = torch.sin(ax), torch.cos(ax)
            Rx = torch.stack([
                torch.stack([torch.ones_like(ax), torch.zeros_like(ax), torch.zeros_like(ax)]),
                torch.stack([torch.zeros_like(ax), cx, -sx]),
                torch.stack([torch.zeros_like(ax), sx, cx]),
            ])
            # Ry
            sy, cy = torch.sin(ay), torch.cos(ay)
            Ry = torch.stack([
                torch.stack([cy, torch.zeros_like(ay), sy]),
                torch.stack([torch.zeros_like(ay), torch.ones_like(ay), torch.zeros_like(ay)]),
                torch.stack([-sy, torch.zeros_like(ay), cy]),
            ])
            # Rz
            sz, cz = torch.sin(az), torch.cos(az)
            Rz = torch.stack([
                torch.stack([cz, -sz, torch.zeros_like(az)]),
                torch.stack([sz, cz, torch.zeros_like(az)]),
                torch.stack([torch.zeros_like(az), torch.zeros_like(az), torch.ones_like(az)]),
            ])
            rot = Rz @ Ry @ Rx

            return (transl, rot)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, t, iteration, dt=1, update_cnt=True, grids=None):
        """Update SDF and body-velocity fields on all staggered grids.

        Parameters
        ----------
        grids : _StaggeredGrids
            Staggered meshgrid container.  When called from
            ``CompositeBodyAnalytical.update()`` this is the composite's
            ``_grids``; when called standalone (e.g. from tests) pass the
            body's own ``_StaggeredGrids`` or use ``_get_staggered_grids``.
        """
        (transl, rot) = self.rototranslate_points(t)
        R_T = rot.T

        # --- linear / angular velocities via autograd ------------------
        t_var = t.clone().detach().requires_grad_(True)

        def _safe_grad(val, t_var):
            """autograd.grad that handles constants (float / int / non-graph tensors)."""
            if not isinstance(val, torch.Tensor) or not val.requires_grad:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            g = torch.autograd.grad(val, t_var, create_graph=False, allow_unused=True)[0]
            if g is None:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return g

        if self.ndim == 2:
            vx = self.update_translation[0](t_var)
            vy = self.update_translation[1](t_var)
            w = self.update_theta(t_var) * self.rad_conv

            lin_vel_x = _safe_grad(vx, t_var)
            lin_vel_y = _safe_grad(vy, t_var)
            ang_vel = _safe_grad(w, t_var)

            # SDF at cell-centres (meshgrid broadcasting)
            px, py = rotate_grid_2d(grids.X, grids.Y, R_T, transl)
            self.sdf_val = self.sdf(px, py)

            # SDF at u-faces
            px, py = rotate_grid_2d(grids.Xu_stag, grids.Yu_stag, R_T, transl)
            self.sdf_u = self.sdf(px, py)

            # SDF at v-faces
            px, py = rotate_grid_2d(grids.Xv_stag, grids.Yv_stag, R_T, transl)
            self.sdf_v = self.sdf(px, py)

            # body velocities (staggered)
            # v = v_lin + ω × r  (2-D:  ω×r = (-ω*ry, ω*rx))
            ry_u = grids.Yu_stag - transl[1]
            self.body_u = lin_vel_x - ang_vel * ry_u
            rx_v = grids.Xv_stag - transl[0]
            self.body_v = lin_vel_y + ang_vel * rx_v

            # Aliases so standalone BodyAnalytical works directly with solver
            self.sdf_val_u = self.sdf_u
            self.sdf_val_v = self.sdf_v

            if update_cnt:
                self.cnt_update = rot @ self.cnt
                self.cnt_update[0] += self.com_pos[0]
                self.cnt_update[1] += self.com_pos[1]
                self.cnt_u = lin_vel_x - ang_vel * (self.cnt_update[1] - self.com_pos[1])
                self.cnt_v = lin_vel_y + ang_vel * (self.cnt_update[0] - self.com_pos[0])

        else:  # 3-D
            vx = self.update_translation[0](t_var)
            vy = self.update_translation[1](t_var)
            vz = self.update_translation[2](t_var)

            angles_raw = self.update_theta(t_var)
            is_scalar = (isinstance(angles_raw, (int, float))
                         or (isinstance(angles_raw, torch.Tensor) and angles_raw.dim() == 0))
            if is_scalar:
                angles_raw = (torch.tensor(0.0, requires_grad=True),
                              torch.tensor(0.0, requires_grad=True),
                              angles_raw)
            wx = angles_raw[0] * self.rad_conv
            wy = angles_raw[1] * self.rad_conv
            wz = angles_raw[2] * self.rad_conv

            lin_vel_x = _safe_grad(vx, t_var)
            lin_vel_y = _safe_grad(vy, t_var)
            lin_vel_z = _safe_grad(vz, t_var)
            ang_vel_x = _safe_grad(wx, t_var)
            ang_vel_y = _safe_grad(wy, t_var)
            ang_vel_z = _safe_grad(wz, t_var)

            # SDF evaluation (meshgrid broadcasting, no flatten)
            def _eval_sdf(X, Y, Z):
                px, py, pz = rotate_grid_3d(X, Y, Z, R_T, transl)
                return self.sdf(px, py, pz)

            self.sdf_val = _eval_sdf(grids.X, grids.Y, grids.Z_grid)
            self.sdf_u = _eval_sdf(grids.Xu_stag, grids.Yu_stag, grids.Zu_stag)
            self.sdf_v = _eval_sdf(grids.Xv_stag, grids.Yv_stag, grids.Zv_stag)
            self.sdf_w = _eval_sdf(grids.Xw_stag, grids.Yw_stag, grids.Zw_stag)

            # body velocities: v = v_lin + ω × r
            # ω × r = (ωy*rz - ωz*ry, ωz*rx - ωx*rz, ωx*ry - ωy*rx)
            def _body_vel_component(Xg, Yg, Zg):
                rx = Xg - transl[0]
                ry = Yg - transl[1]
                rz = Zg - transl[2]
                bu = lin_vel_x + ang_vel_y * rz - ang_vel_z * ry
                bv = lin_vel_y + ang_vel_z * rx - ang_vel_x * rz
                bw = lin_vel_z + ang_vel_x * ry - ang_vel_y * rx
                return bu, bv, bw

            self.body_u, _, _ = _body_vel_component(
                grids.Xu_stag, grids.Yu_stag, grids.Zu_stag)
            _, self.body_v, _ = _body_vel_component(
                grids.Xv_stag, grids.Yv_stag, grids.Zv_stag)
            _, _, self.body_w = _body_vel_component(
                grids.Xw_stag, grids.Yw_stag, grids.Zw_stag)

            # Aliases so standalone BodyAnalytical works directly with solver
            self.sdf_val_u = self.sdf_u
            self.sdf_val_v = self.sdf_v
            self.sdf_val_w = self.sdf_w




class CompositeBodyAnalytical(Body):

    def __init__(self, device, x, y, sdf_funs, update_maps, z=None, plotting=False, **kwargs):
        """Composite body: union of several BodyAnalytical objects."""
        super().__init__(device, x, y, z=z, **kwargs)
        self._setup_grids()

        self.nbodies = len(sdf_funs)
        assert self.nbodies == len(update_maps), "Number of sdf functions and update maps must be the same"

        # Child BodyAnalytical instances are created with pre_update=False so
        # they don't call update() (which requires grids) during __init__.
        # CompositeBodyAnalytical.initialize() drives the first update and
        # passes self._grids to each child.
        self.bodies = [
            BodyAnalytical(
                device, x, y,
                sdf_funs[i],
                update_maps[i],
                z=z,
                plotting=plotting,
                pre_update=False,
                **kwargs
            ) for i in range(self.nbodies)
        ]

        self.mu_funcs = self.bodies[0].mu_funcs
        self.com_pos = torch.zeros((self.nbodies, self.ndim), device=device)
        self.initialize()

    def initialize(self):
        self.update(torch.tensor(0.0, device=self.device, dtype=self.dtype), 0)

    def update(self, t, iteration, dt=1):
        # Streaming union: process bodies one at a time to avoid
        # allocating (nbodies, *grid_shape) stacks.
        for i, body in enumerate(self.bodies):
            body.update(t, iteration, dt=dt, grids=self._grids)
            if i == 0:
                self.sdf_val   = body.sdf_val
                self.sdf_val_u = body.sdf_u
                self.body_u    = body.body_u
                self.sdf_val_v = body.sdf_v
                self.body_v    = body.body_v
                if self.ndim == 3:
                    self.sdf_val_w = body.sdf_w
                    self.body_w    = body.body_w
            else:
                mask = body.sdf_val < self.sdf_val
                self.sdf_val = torch.where(mask, body.sdf_val, self.sdf_val)

                mask_u = body.sdf_u < self.sdf_val_u
                self.sdf_val_u = torch.where(mask_u, body.sdf_u, self.sdf_val_u)
                self.body_u    = torch.where(mask_u, body.body_u, self.body_u)

                mask_v = body.sdf_v < self.sdf_val_v
                self.sdf_val_v = torch.where(mask_v, body.sdf_v, self.sdf_val_v)
                self.body_v    = torch.where(mask_v, body.body_v, self.body_v)

                if self.ndim == 3:
                    mask_w = body.sdf_w < self.sdf_val_w
                    self.sdf_val_w = torch.where(mask_w, body.sdf_w, self.sdf_val_w)
                    self.body_w    = torch.where(mask_w, body.body_w, self.body_w)


class BodyFishAnalytical(Body):

    def __init__(
        self,
        device,
        x,
        y,
        L             = 0.015,
        A             = 0.015,
        f             = 3,
        wavefrequency = 0.95,
        c1            = +0.05,
        c2            = -0.13,
        c3            = +0.28,
        xshift        = 0.0,
        yshift        = 0.0,
        eps           = 0.05,
        sb            = 0.07,
        wh            = 0.07,
        st            = 0.95,
        wt            = 0.01,
        thk           = False,

    ):
        super().__init__(device, x, y, eps=eps)
        """

        """
        self._setup_grids()
        self.L=L
        self.A=A
        self.f=f
        self.wavefrequency=wavefrequency
        self.XC=self.X-xshift
        self.YC=self.Y-yshift

        self.wh=wh*L
        self.sb=sb*L
        self.st=st*L
        self.wt=wt*L

        # OLD ENVELOPE (Di Santo et al. 2021 - All Fishes)
        self.c1=c1
        self.c2=c2
        self.c3=c3

        # NEW ENVELOPE (Di Santo et al. 2021 - Danio Rerio)
        self.p0 = 44.5 / 110.0
        self.p1 = self.p0 + 22.0 / 110.0
        self.p2 = 1.0

        self.a0 = 0.04
        self.a1 = 0.16
        self.a2 = 0.24

        self.s1 = (self.a1 - self.a0) / (self.p1 - self.p0)
        self.s2 = (self.a2 - self.a1) / (self.p2 - self.p1)

        self.bodies = [self]
        if thk:
            self.thk = lambda s: thk
        else:
            self.thk = self.thk_nonconst

        # Staggered shifted coordinates
        self.XC_u = self.Xu_stag - xshift
        self.YC_u = self.Yu_stag - yshift
        self.XC_v = self.Xv_stag - xshift
        self.YC_v = self.Yv_stag - yshift

        # Old positions on cell-centre grid
        self.oldpos_u = torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype)
        self.oldpos_v = torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype)

        # Old positions on staggered grids (for staggered body-velocity FD)
        self.oldpos_u_ustag = torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype)
        self.oldpos_v_vstag = torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype)

        self.initialize()

    def envelope(self, s):
        """
        Amplitude envelope — width tapers toward the tail.
        Uses the old polynomial envelope (c1 + c2*s + c3*s^2).
        """
        return self.c1+self.c2*s+self.c3*s**2

    def thk_nonconst(self,s):
        """
        fish width
        """
        return torch.where(
            s<self.sb,
            torch.sqrt(2*self.wh*s-s**2),
            torch.where(
                s<self.st,
                self.wh-(self.wh-self.wt)*(((s-self.sb)/(self.st-self.sb))**2),
                self.wt*(self.L-s)/(self.L-self.st)
            )
        )

    def sdf_fun(self, x,y):
        s = x.clamp(0,self.L)
        sdf = torch.sqrt((x-s)**2+y**2)
        return sdf-self.thk(s)

    def _deform_y(self, XC, YC, t):
        """Compute deformed y-coordinates on a given (XC, YC) grid."""
        s = XC.clamp(0, self.L)
        return YC + self.A * self.envelope(s / self.L) * torch.sin(
            2 * torch.pi * (self.wavefrequency * s / self.L - self.f * t)
        )

    def update(self, t, iteration, dt=1):
        """Update SDF and body-velocity fields on cell-centre and staggered grids."""

        # --- Cell-centre grid ---
        new_x = self.XC
        new_y = self._deform_y(self.XC, self.YC, t)

        self.oldpos_u = new_x
        self.oldpos_v = new_y
        self.sdf_val = self.sdf_fun(new_x, new_y)

        # --- U-staggered grid ---
        new_x_u = self.XC_u
        new_y_u = self._deform_y(self.XC_u, self.YC_u, t)
        self.sdf_u = self.sdf_fun(new_x_u, new_y_u)
        self.sdf_val_u = self.sdf_u  # alias for solver compatibility
        self.body_u = -(new_x_u - self.oldpos_u_ustag) / dt
        self.oldpos_u_ustag = new_x_u

        # --- V-staggered grid ---
        new_y_v = self._deform_y(self.XC_v, self.YC_v, t)
        self.sdf_v = self.sdf_fun(self.XC_v, new_y_v)
        self.sdf_val_v = self.sdf_v  # alias for solver compatibility
        self.body_v = -(new_y_v - self.oldpos_v_vstag) / dt
        self.oldpos_v_vstag = new_y_v

    def initialize(self):
        """Initialize SDF properties at time 0."""
        self.cnt        = torch.zeros((2, 1), device=self.device, dtype=self.dtype)
        self.cnt_update = self.cnt.clone().detach()
        self.curv_coord = torch.tensor([0, 1], device=self.device, dtype=self.dtype)
        self.com_pos    = torch.tensor([[0, 0]], device=self.device, dtype=self.dtype)
        self.update(0, 0)

        # Zero-out initial body velocities (the first update computed
        # spurious velocities from the zero-initialised old positions).
        self.body_u = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.body_v = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

class BodyFishExperimental(Body):

    def __init__(
        self,
        device,
        x,
        y,
        body_length,
        folder_name,
        file_name,
        save_data,
        plot_data,
        target_fish,
        start_recording,
        end_recording,
        timestep,
        total_duration,
        freq_scaling,
        filter_freqs,
        xshift       = -0.0,
        yshift       = 0.0,
        eps          = 0.05,
        initial_time = 0.0,
    ):
        super().__init__(device, x, y, eps=eps)
        self._setup_grids()

        self.L               = body_length
        self.folder_name     = folder_name
        self.file_name       = file_name
        self.save_data       = save_data
        self.plot_data       = plot_data

        self.target_fish     = target_fish
        self.start_recording = start_recording
        self.end_recording   = end_recording
        self.timestep        = timestep
        self.total_duration  = total_duration
        self.freq_scaling    = freq_scaling
        self.filter_freqs    = filter_freqs
        self.initial_time    = initial_time

        self.XC              = self.X - xshift
        self.YC              = self.Y - yshift

        # Staggered shifted coordinates
        self.XC_u = self.Xu_stag - xshift
        self.YC_u = self.Yu_stag - yshift
        self.XC_v = self.Xv_stag - xshift
        self.YC_v = self.Yv_stag - yshift

        # TYTELL-LIKE
        self.sb              = 0.07 * body_length
        self.st              = 0.95 * body_length
        self.wh              = 0.07 * body_length
        self.wt              = 0.01 * body_length

        # LIU-LIKE
        self.s1 = 0.54
        self.s2 = 0.72
        self.s3 = 0.83
        self.s4 = 0.85
        self.w1 = 0.16
        self.w2 = 0.004

        # Get the signal
        from lilytorch.src.scripts.zebrafish_files.load_data import get_experimental_signal
        self.points_coords_df = get_experimental_signal(
            folder_name     = self.folder_name,
            file_name       = self.file_name,
            target_fish     = self.target_fish,
            start_recording = self.start_recording,
            end_recording   = self.end_recording,
            timestep        = self.timestep,
            total_duration  = self.total_duration,
            freq_scaling    = self.freq_scaling,
            save_data       = self.save_data,
            plot_data       = self.plot_data,
        )

        self.times   = self.points_coords_df['time'].values
        self.n_steps = len(self.times)

        # Coordinates
        self.points_x    = self.points_coords_df.filter(regex='x_').values * self.L
        self.points_y    = self.points_coords_df.filter(regex='y_').values * self.L
        self.points_x[:] = np.mean(self.points_x, axis=0)

        self.bodies = [self]

        # Old positions for finite-difference body velocity
        self.oldpos_v        = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.oldpos_v_vstag  = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.oldpos_u_ustag  = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        self.initialize()

    def thk_liu(self, s):
        """
        fish width
        """

        s1, s2, s3, s4, w1, w2 = self.s1, self.s2, self.s3, self.s4, self.w1, self.w2
        s_star                 = s / ( s4 * self.L )

        c0, c1, c2, c3, c4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
        x0, x1, x2, x3, x4 = s_star**0.5, s_star, s_star**2, s_star**3, s_star**4

        return torch.where(
            s < s3 * self.L,
            5 * (s4 * w1) * ( c0*x0 + c1*x1 + c2*x2 + c3*x3 + c4*x4 ),
            w2
        )

    def thk_gazzola(self,s):
        """
        fish width
        """
        return torch.where(
            s<self.sb,
            torch.sqrt(2*self.wh*s-s**2),
            torch.where(
                s<self.st,
                self.wh-(self.wh-self.wt)*(((s-self.sb)/(self.st-self.sb))**2),
                self.wt*(self.L-s)/(self.L-self.st)
            )
        )

    def thk(self,s):
        return self.thk_gazzola(s)

    def sdf_fun(self, x,y):
        s = x.clamp(0,self.L)
        sdf = torch.sqrt((x-s)**2+y**2)
        return sdf-self.thk(s)


    def _interp_y_at_time(self, t):
        """Build a lateral-displacement interpolator for time *t*."""
        t0 = self.times[self.times <= t][-1]
        t1 = self.times[self.times > t][0]
        t0_ind = (self.times == t0)
        t1_ind = (self.times == t1)

        x0, x1 = self.points_x[t0_ind], self.points_x[t1_ind]
        y0, y1 = self.points_y[t0_ind], self.points_y[t1_ind]

        x_coords_t = (x0 + (x1 - x0) * (t - t0) / (t1 - t0)).flatten()
        y_coords_t = (y0 + (y1 - y0) * (t - t0) / (t1 - t0)).flatten()

        s_coords_t = x_coords_t / x_coords_t[-1]
        return lambda s: np.interp(s, s_coords_t, y_coords_t)

    def _deform_y(self, XC, YC, interp_y):
        """Compute deformed y-coordinates on a given grid using *interp_y*."""
        s = XC.clamp(0, self.L)
        return YC + torch.tensor(
            interp_y(s.cpu().numpy() / self.L),
            dtype=self.dtype,
            device=self.device,
        )

    def update(self, t, iteration, dt=1):
        """Update SDF and body-velocity fields on cell-centre and staggered grids."""
        interp_y = self._interp_y_at_time(t)

        # --- Cell-centre grid ---
        new_x = self.XC
        new_y = self._deform_y(self.XC, self.YC, interp_y)
        self.oldpos_v = new_y
        self.sdf_val = self.sdf_fun(new_x, new_y)

        # --- U-staggered grid ---
        new_x_u = self.XC_u
        new_y_u = self._deform_y(self.XC_u, self.YC_u, interp_y)
        self.sdf_u = self.sdf_fun(new_x_u, new_y_u)
        self.sdf_val_u = self.sdf_u
        self.body_u = -(new_x_u - self.oldpos_u_ustag) / dt
        self.oldpos_u_ustag = new_x_u

        # --- V-staggered grid ---
        new_y_v = self._deform_y(self.XC_v, self.YC_v, interp_y)
        self.sdf_v = self.sdf_fun(self.XC_v, new_y_v)
        self.sdf_val_v = self.sdf_v
        self.body_v = -(new_y_v - self.oldpos_v_vstag) / dt
        self.oldpos_v_vstag = new_y_v

    def initialize(self):
        """Initialize SDF properties at initial time."""
        self.cnt        = torch.zeros((2, 1), device=self.device, dtype=self.dtype)
        self.cnt_update = self.cnt.clone().detach()
        self.curv_coord = torch.tensor([0, 1], device=self.device, dtype=self.dtype)
        self.com_pos    = torch.tensor([[0, 0]], device=self.device, dtype=self.dtype)
        self.update(self.initial_time, 0)

        # Zero-out initial body velocities
        self.body_u = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.body_v = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

    def save_signal(self, folder_name):
        ''' Save the signal to a csv file '''
        self.points_coords_df.to_csv(
            os.path.join(folder_name, 'kinematics_signals.csv'),
            index = False,
        )

class BodyMesh(Body):
    """Immersed body whose SDF is derived from a triangle-mesh file.

    If *nsamples* / *msamples* are ``None`` (the default) the SDF sampling
    resolution is chosen automatically so that the spacing is half the
    simulation grid spacing *h*, ensuring the interpolated SDF is well-resolved.
    """
    def __init__(self, device, x, y, mesh_file, update_maps, z=None, eps=0.05,
                 compute_interp=True, nsamples=None, msamples=None, ksamples=None,
                 suit=0, plotting_meshes=False, zpos=0, **kwargs):
        super().__init__(device, x, y, z=z, eps=eps)
        self.mesh_file           = mesh_file
        self.compute_interp      = compute_interp
        self.save_folder         = kwargs.pop("save_folder", "")
        self.update_theta        = update_maps[0]
        self.update_translation  = update_maps[1]
        self.suit                = suit
        self.plotting            = plotting_meshes
        self.apply_closing_morph = kwargs.pop("apply_closing_morph", False)
        self.m2s                 = mesh2sdf(
            mesh_file,
            convexify=kwargs.pop("convexify", True),
            scale=kwargs.pop("scale", 1)
            )

        # ---- auto-compute nsamples / msamples / ksamples --------------
        # Target SDF spacing = h/2 so the interpolated field is well-resolved
        # on the simulation grid.  The sampling domain is sized per-axis:
        # each axis covers the bounding-box span plus padding on each side.
        # The Heaviside band extends eps from the surface and normals need
        # ~2 extra cells, so pad ≈ eps + 2*h suffices; the interpolator
        # uses fill_value="nearest" for anything beyond.
        bb = self.m2s.bounding_box()
        target_spacing = self.h / 2.0
        self.pad = float(self.eps + 2 * self.h)

        if nsamples is None:
            span_x = (bb[0, 1] - bb[0, 0]) + 2 * self.pad
            nsamples = max(64, int(np.ceil(span_x / target_spacing)))
        if msamples is None:
            span_y = (bb[1, 1] - bb[1, 0]) + 2 * self.pad
            msamples = max(64, int(np.ceil(span_y / target_spacing)))
        if ksamples is None and self.ndim == 3:
            span_z = (bb[2, 1] - bb[2, 0]) + 2 * self.pad
            ksamples = max(64, int(np.ceil(span_z / target_spacing)))
        self.nsamples = nsamples
        self.msamples = msamples
        self.ksamples = ksamples

        self.compute_sdfs(zpos)
        del self.m2s
        self.initialize()
        self.bodies = [self]


    def resample_closed_contour(self, points, spacing, keep_duplicate_endpoint=True):
        """
        Resample a closed contour for (approximately) uniform spacing.
        - points: (M,2) numpy array of (x,y). Can be closed (first==last) or open; treated as closed.
        - spacing: desired spacing between resampled points (float > 0).
        - keep_duplicate_endpoint: if True, return N+1 points with last == first (explicit closure).
                                if False, return N points (no duplicate at end).
        Returns:
        - new_pts: (N+1,2) or (N,2) array of resampled points.
        - actual_spacing: total_length / N  (the spacing actually used)
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("points must be an (M,2) array-like")

        if spacing <= 0:
            raise ValueError("spacing must be positive")

        # If not already closed, append first point for segment math
        if not np.allclose(pts[0], pts[-1]):
            pts_closed = np.vstack([pts, pts[0]])
        else:
            pts_closed = pts.copy()

        # segment vectors and lengths
        segs = pts_closed[1:] - pts_closed[:-1]
        seg_lens = np.hypot(segs[:,0], segs[:,1])
        total_length = seg_lens.sum()
        if total_length == 0:
            raise ValueError("zero-length contour")

        # choose number of equal intervals such that spacing ~ requested spacing
        N = max(3, int(round(total_length / spacing)))
        actual_spacing = total_length / N

        # cumulative distances along the closed polyline (start at 0, last = total_length)
        s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        # x,y coordinates corresponding to s
        x = pts_closed[:,0]
        y = pts_closed[:,1]

        # target sample locations: include the final total_length so last interpolates to first point
        target_s = np.linspace(0.0, total_length, N+1)

        # np.interp requires strictly increasing x; s is non-decreasing
        xi = np.interp(target_s, s, x)
        yi = np.interp(target_s, s, y)
        new_pts = np.vstack([xi, yi]).T

        if not keep_duplicate_endpoint:
            return new_pts[:-1], actual_spacing  # return N points
        return new_pts, actual_spacing      # return N+1 points where last==first


    def compute_sdfs(self, zpos=0):
        """Compute the SDF from the mesh and build an interpolation function.

        Works in 2-D (takes a slice at *zpos*) or 3-D (full volume query).
        The output arrays are saved to *self.save_folder* so that
        ``initialize()`` can reload them without re-computing.
        """
        self.bb = self.m2s.bounding_box()
        if not self.compute_interp:
            return

        if self.ndim == 2:
            self._compute_sdfs_2d(zpos, self.pad)
        else:
            self._compute_sdfs_3d(self.pad)

    # ---- 2-D SDF computation ------------------------------------------
    def _compute_sdfs_2d(self, zpos, pad):
        cv2 = _import_cv2()
        skfmm = _import_skfmm()
        measure = _import_measure()
        cx_bb = (self.bb[0, 1] + self.bb[0, 0]) / 2
        cy_bb = (self.bb[1, 1] + self.bb[1, 0]) / 2
        half_x = (self.bb[0, 1] - self.bb[0, 0]) / 2 + pad
        half_y = (self.bb[1, 1] - self.bb[1, 0]) / 2 + pad
        xnp = np.linspace(cx_bb - half_x, cx_bb + half_x, self.nsamples)
        ynp = np.linspace(cy_bb - half_y, cy_bb + half_y, self.msamples)

        X, Y = np.meshgrid(xnp, ynp, indexing="ij")
        xflat = X.flatten()
        yflat = Y.flatten()
        zflat = zpos * np.ones_like(xflat)
        query_pts = np.stack([xflat, yflat, zflat], axis=1).astype(np.float32)

        sdf_val_o3d, _ = self.m2s(query_pts)
        inside_mask = sdf_val_o3d.reshape(X.shape) < 0
        labels = measure.label(inside_mask, connectivity=1)
        component_ids, component_sizes = np.unique(labels[labels > 0], return_counts=True)
        tiny_components = component_ids[component_sizes < 4]
        if len(tiny_components) > 0:
            inside_mask[np.isin(labels, tiny_components)] = False

        binary_2d = np.zeros((self.nsamples, self.msamples))
        binary_2d[inside_mask] = 1

        if self.plotting:
            self.m2s.visualize()

        if self.apply_closing_morph:
            gray = (255 * binary_2d).astype('uint8')
            im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            im = cv2.dilate(im, element, iterations=1)
            im = cv2.erode(im, element, iterations=3)
            im = im[:, :, 0]
        else:
            im = binary_2d

        if self.plotting:
            display_scale = 0.5
            display_size = (int(im.shape[1] * display_scale), int(im.shape[0] * display_scale))
            im_resized = cv2.resize(im.astype(np.float32), display_size)
            cv2.imshow("window_name", im_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        binary_2d = np.where(im == 0, 1, -1)  # inside mask

        dx, dy = xnp[1] - xnp[0], ynp[1] - ynp[0]
        print(f"Computing the sdf for {self.mesh_file}, with space steps ({dx},{dy})")
        sdf_val = skfmm.distance(binary_2d, dx=[dx, dy]) - self.suit

        # ---- contour computation (2-D only) ----------------------------
        cnt = np.array(measure.find_contours(sdf_val-self.h, 0)[0]).T
        cnt[0] = xnp[0] + cnt[0] * (xnp[1] - xnp[0])
        cnt[1] = ynp[0] + cnt[1] * (ynp[1] - ynp[0])

        def signed_area(contour):
            x, y = contour[0, :], contour[1, :]
            return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

        def ensure_clockwise(contour):
            if signed_area(contour) > 0:
                contour = contour[:, ::-1]
            return contour

        cnt = ensure_clockwise(cnt)

        # ensure starting point is at the middle of the bounding box
        start_point = np.array([self.bb[0, 0], 0])
        dists = np.sqrt((cnt[0] - start_point[0]) ** 2 + (cnt[1] - start_point[1]) ** 2)
        valid_indices = np.where(cnt[1] > 0)[0]
        idx = valid_indices[np.argmin(dists[valid_indices])] if len(valid_indices) > 0 else np.argmin(dists)
        cnt = np.concatenate((cnt[:, idx + 1:], cnt[:, :idx]), axis=1)

        ds = self.h
        x, y, _ = resample_contour(cnt[0], cnt[1], spacing=ds, closed=True)
        del cnt
        cnt = np.array([x, y])

        dx_cnt = np.diff(x)
        dy_cnt = np.diff(y)
        ds_cnt = np.sqrt(dx_cnt ** 2 + dy_cnt ** 2)
        curv_coord = np.concatenate(([0], np.cumsum(ds_cnt)))
        sign_vec = np.where(cnt[1] >= cnt[1][0], 1, -1)

        if self.plotting:
            _, plt, cm = _import_matplotlib()
            plt.figure()
            plt.contourf(X, Y, sdf_val)
            plt.plot(cnt[0], cnt[1], 'r', linewidth=2)
            plt.colorbar()
            plt.show()

            cmap = cm.get_cmap('RdBu')
            n_points = cnt.shape[1]
            colors = cmap(np.linspace(0, 1, n_points))
            plt.scatter(cnt[0], cnt[1], c=colors, cmap=cmap, s=10)
            plt.show()

        print(f"Computing the interpolation functions for {self.mesh_file}")

        os.makedirs(self.save_folder, exist_ok=True)
        mesh_tag = self.mesh_file.split('/')[-1].split('.')[0]
        np.save(os.path.join(self.save_folder, f"xnp_{mesh_tag}.npy"), xnp)
        np.save(os.path.join(self.save_folder, f"ynp_{mesh_tag}.npy"), ynp)
        np.save(os.path.join(self.save_folder, f"sdf_val_{mesh_tag}.npy"), sdf_val)
        np.save(os.path.join(self.save_folder, f"cnt_{mesh_tag}.npy"), cnt)
        np.save(os.path.join(self.save_folder, f"curv_coord_{mesh_tag}.npy"), curv_coord)
        np.save(os.path.join(self.save_folder, f"sign_vec_{mesh_tag}.npy"), sign_vec)

    # ---- 3-D SDF computation ------------------------------------------
    def _compute_sdfs_3d(self, pad):
        """Build a 3-D SDF field from the mesh using open3d + skfmm."""
        skfmm = _import_skfmm()
        centres = [(self.bb[i, 1] + self.bb[i, 0]) / 2 for i in range(3)]
        halves = [(self.bb[i, 1] - self.bb[i, 0]) / 2 + float(pad) for i in range(3)]
        xnp = np.linspace(centres[0] - halves[0], centres[0] + halves[0], self.nsamples)
        ynp = np.linspace(centres[1] - halves[1], centres[1] + halves[1], self.msamples)
        znp = np.linspace(centres[2] - halves[2], centres[2] + halves[2], self.ksamples)

        X, Y, Z = np.meshgrid(xnp, ynp, znp, indexing="ij")
        query_pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)

        print(f"Computing 3-D SDF for {self.mesh_file} ({self.nsamples}×{self.msamples}×{self.ksamples}) ...")
        sdf_val_o3d, _ = self.m2s(query_pts)
        if self.plotting:
            self.m2s.visualize()

        binary_3d = np.where(sdf_val_o3d.reshape(X.shape) < 0, -1, 1)

        dx, dy, dz = xnp[1] - xnp[0], ynp[1] - ynp[0], znp[1] - znp[0]
        print(f"  skfmm distance with spacing ({dx:.6f},{dy:.6f},{dz:.6f})")
        sdf_val = skfmm.distance(binary_3d, dx=[dx, dy, dz]) - self.suit

        print(f"  Saving 3-D interpolation data for {self.mesh_file}")
        os.makedirs(self.save_folder, exist_ok=True)
        mesh_tag = self.mesh_file.split('/')[-1].split('.')[0]
        np.save(os.path.join(self.save_folder, f"xnp_{mesh_tag}.npy"), xnp)
        np.save(os.path.join(self.save_folder, f"ynp_{mesh_tag}.npy"), ynp)
        np.save(os.path.join(self.save_folder, f"znp_{mesh_tag}.npy"), znp)
        np.save(os.path.join(self.save_folder, f"sdf_val_{mesh_tag}.npy"), sdf_val)

        if self.plotting:
            self._plot_sdf_3d(xnp, ynp, znp, sdf_val, centres)

    def _plot_sdf_3d(self, xnp, ynp, znp, sdf_val, centres):
        """Visualise a 3-D SDF: three orthogonal slices + isosurface."""
        _, plt, _ = _import_matplotlib()
        measure = _import_measure()
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ix_mid = np.argmin(np.abs(xnp - centres[0]))
        iy_mid = np.argmin(np.abs(ynp - centres[1]))
        iz_mid = np.argmin(np.abs(znp - centres[2]))

        # --- 1. Three orthogonal slice plots ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # XY slice (z = centre)
        ax = axes[0]
        Xxy, Yxy = np.meshgrid(xnp, ynp, indexing="ij")
        cf = ax.contourf(Xxy, Yxy, sdf_val[:, :, iz_mid], levels=30, cmap="RdBu_r")
        ax.contour(Xxy, Yxy, sdf_val[:, :, iz_mid], levels=[0], colors="k", linewidths=2)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"XY slice (z={znp[iz_mid]:.4f})")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax)

        # XZ slice (y = centre)
        ax = axes[1]
        Xxz, Zxz = np.meshgrid(xnp, znp, indexing="ij")
        cf = ax.contourf(Xxz, Zxz, sdf_val[:, iy_mid, :], levels=30, cmap="RdBu_r")
        ax.contour(Xxz, Zxz, sdf_val[:, iy_mid, :], levels=[0], colors="k", linewidths=2)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(f"XZ slice (y={ynp[iy_mid]:.4f})")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax)

        # YZ slice (x = centre)
        ax = axes[2]
        Yyz, Zyz = np.meshgrid(ynp, znp, indexing="ij")
        cf = ax.contourf(Yyz, Zyz, sdf_val[ix_mid, :, :], levels=30, cmap="RdBu_r")
        ax.contour(Yyz, Zyz, sdf_val[ix_mid, :, :], levels=[0], colors="k", linewidths=2)
        ax.set_xlabel("y"); ax.set_ylabel("z")
        ax.set_title(f"YZ slice (x={xnp[ix_mid]:.4f})")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax)

        fig.suptitle(f"3-D SDF slices: {self.mesh_file.split('/')[-1]}", fontsize=13)
        fig.tight_layout()
        plt.show()

        # --- 2. Isosurface of the zero level-set ---
        try:
            verts, faces, _, _ = measure.marching_cubes(sdf_val, level=0)
            # Convert voxel indices to physical coordinates
            verts_phys = np.column_stack([
                xnp[0] + verts[:, 0] * (xnp[1] - xnp[0]),
                ynp[0] + verts[:, 1] * (ynp[1] - ynp[0]),
                znp[0] + verts[:, 2] * (znp[1] - znp[0]),
            ])

            fig3d = plt.figure(figsize=(8, 8))
            ax3d = fig3d.add_subplot(111, projection="3d")
            mesh_coll = Poly3DCollection(
                verts_phys[faces], alpha=0.6, edgecolor="k",
                linewidth=0.1, facecolor="steelblue",
            )
            ax3d.add_collection3d(mesh_coll)
            ax3d.set_xlim(xnp[0], xnp[-1])
            ax3d.set_ylim(ynp[0], ynp[-1])
            ax3d.set_zlim(znp[0], znp[-1])
            ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
            ax3d.set_title(f"SDF=0 isosurface: {self.mesh_file.split('/')[-1]}")
            plt.show()
        except (RuntimeError, ValueError) as e:
            logger.warning("Could not extract isosurface: %s", e)

    def initialize(self):
        """Load pre-computed SDF data and build the interpolation function."""
        mesh_tag = self.mesh_file.split('/')[-1].split('.')[0]

        xnp = np.load(os.path.join(self.save_folder, f"xnp_{mesh_tag}.npy"))
        ynp = np.load(os.path.join(self.save_folder, f"ynp_{mesh_tag}.npy"))
        sdf_val = np.load(os.path.join(self.save_folder, f"sdf_val_{mesh_tag}.npy"))

        if self.ndim == 2:
            self._initialize_2d_mesh(xnp, ynp, sdf_val, mesh_tag)
        else:
            self._initialize_3d_mesh(xnp, ynp, sdf_val, mesh_tag)

    def _initialize_2d_mesh(self, xnp, ynp, sdf_val, mesh_tag):
        cnt = np.load(os.path.join(self.save_folder, f"cnt_{mesh_tag}.npy"))
        curv_coord = np.load(os.path.join(self.save_folder, f"curv_coord_{mesh_tag}.npy"))
        sign_vec = np.load(os.path.join(self.save_folder, f"sign_vec_{mesh_tag}.npy"))

        self.sdf = RegularGridInterpolatorAutomatic(
            (
                torch.from_numpy(xnp).type(self.dtype).to(self.device),
                torch.from_numpy(ynp).type(self.dtype).to(self.device)
            ),
            torch.from_numpy(sdf_val).type(self.dtype).to(self.device),
            fill_value="nearest",
            method="quadratic"
        )

        self.curv_coord = torch.from_numpy(curv_coord).type(self.dtype).to(self.device)
        self.cnt        = torch.from_numpy(cnt).type(self.dtype).to(self.device)
        self.cnt_update = self.cnt.clone().detach()
        self.cnt_u = torch.zeros_like(self.cnt_update[0])
        self.cnt_v = torch.zeros_like(self.cnt_update[1])
        self.cnt_f_u = torch.zeros_like(self.cnt_update[0])
        self.cnt_f_v = torch.zeros_like(self.cnt_update[1])
        self.cnt_int_f_u = torch.zeros_like(self.cnt_update[0])
        self.cnt_int_f_v = torch.zeros_like(self.cnt_update[1])
        self.r_com = torch.zeros_like(self.cnt_update)
        self.ds = self.curv_coord[1] - self.curv_coord[0]
        self.mask = torch.arange(len(self.curv_coord), device=self.device)
        self.sign_vec = torch.from_numpy(sign_vec).type(self.dtype).to(self.device)

    def _initialize_3d_mesh(self, xnp, ynp, sdf_val, mesh_tag):
        znp = np.load(os.path.join(self.save_folder, f"znp_{mesh_tag}.npy"))

        self.sdf = RegularGridInterpolatorAutomatic(
            (
                torch.from_numpy(xnp).type(self.dtype).to(self.device),
                torch.from_numpy(ynp).type(self.dtype).to(self.device),
                torch.from_numpy(znp).type(self.dtype).to(self.device),
            ),
            torch.from_numpy(sdf_val).type(self.dtype).to(self.device),
            fill_value="nearest",
        )

        # 3-D bodies don't have 1-D contour representations – set stubs
        self.cnt = torch.zeros((3, 1), device=self.device, dtype=self.dtype)
        self.cnt_update = self.cnt.clone().detach()
        self.curv_coord = torch.tensor([0, 1], device=self.device, dtype=self.dtype)
        self.ds = self.curv_coord[1] - self.curv_coord[0]
        self.cnt_u = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_v = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.cnt_w = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.mask = torch.arange(1, device=self.device)
        self.sign_vec = torch.ones(1, device=self.device, dtype=self.dtype)

    def update(self, t, iteration, dt=1):
        pass

    def visualize(self):
        self.m2s.visualize()

class CompositeBodyMesh(Body):

    def __init__(self, device, x, y, sdf_folder, sdf_name, custom_update, eps=0.05,
                 compute_interp=True, nsamples=None, msamples=None, ksamples=None,
                 plotting=False, plotting_meshes=False, suit=0.0, **kwargs):
        """Composite body built from a multi-link SDF model file."""
        super().__init__(device, x, y, eps=eps)
        self._setup_grids()

        self.sdf_folder      = sdf_folder
        self.sdf             = _import_model_sdf().read(sdf_folder+sdf_name)[0]
        self.bodies          = []
        self.suit            = suit
        self.plotting        = plotting
        self.plotting_meshes = plotting_meshes
        for link_i, link in enumerate(self.sdf.links):
            mesh_name = link["visuals"][0]["geometry"]["uri"]
            mesh_gpath = sdf_folder+mesh_name
            initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
            update_funcs = (
                lambda t: 180,
                [
                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                ]
            )
            body = BodyMesh(
                device, x, y,
                mesh_gpath,
                update_funcs,
                eps=eps,
                compute_interp=compute_interp,
                nsamples=nsamples, msamples=msamples, ksamples=ksamples,
                suit=suit,
                plotting_meshes=plotting_meshes,
                **kwargs
            )
            body.id = link_i
            self.bodies.append(body)

        self.nbodies = len(self.bodies)
        self.custom_update = custom_update
        self.compute_interp = compute_interp

        self.mu_funcs        = self.bodies[0].mu_funcs
        self.compute_normals = self.bodies[0].compute_normals
        gs = self.grid_shape
        is_3d = len(gs) == 3

        # Per-body SDF stacks – 2-D only.
        # 3-D paths always use comp._sdf_sparse (per-body sparse sub-blocks),
        # so skip the dense (B, Nx, Ny, Nz) allocations entirely for 3-D.
        # For streaming 2-D, BDIMhandler.__init__ deletes sdf_vals after init
        # since _update_2d_streaming_multi / streaming_sdf_forces_post_2d never read it.
        # if not is_3d:
        #     self.sdf_vals   = torch.zeros((self.nbodies, *gs), device=device)
        #     self.sdf_vals_u = torch.zeros((self.nbodies, *gs), device=device)
        #     self.sdf_vals_v = torch.zeros((self.nbodies, *gs), device=device)
        #     self.u_vals     = torch.zeros((self.nbodies, *gs), device=device)
        #     self.v_vals     = torch.zeros((self.nbodies, *gs), device=device)
        #     self.sdf_val_u  = torch.zeros_like(self.X)
        #     self.sdf_val_v  = torch.zeros_like(self.X)

        self.com_pos   = torch.zeros((self.nbodies, self.ndim), device=device)

        # Composite-level body-velocity output fields (written by BDIMhandler).
        self.body_u = torch.zeros(gs, device=device, dtype=self.dtype)
        self.body_v = torch.zeros(gs, device=device, dtype=self.dtype)
        if is_3d:
            self.body_w = torch.zeros(gs, device=device, dtype=self.dtype)

        if not self.custom_update:
            self.initialize()


    def initialize(self):
        self.update(torch.tensor(0.0, device=self.device, dtype=self.dtype), 0)

    def update(self, t, iteration, dt=1):
        (angles, translations) = self.custom_update(t)
        sdf_properties = []
        for body_i, body in enumerate(self.bodies):
            sdf_properties.append(
                body.update_body(
                    body.sdf_interp,
                    angles[body_i],
                    (
                        translations[body_i,0],
                        translations[body_i,1]
                    ),
                    dt=dt
                )
            )
        self.sdf_val = torch.min(torch.stack([prop[0] for idx, prop in enumerate(sdf_properties)]),axis=0)[0]


    def visualize(self):
        o3d = _import_open3d()
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        for body in self.bodies:
            viewer.add_geometry(body.m2s._mesh)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()


    # Function to create a Gaussian kernel
    def gaussian_kernel(self, size: int, sigma: float):
        """Creates a 2D Gaussian kernel."""
        x_coord = torch.arange(size, dtype=self.dtype, device=self.device)
        x_grid = x_coord.repeat(size).view(size, size)
        y_grid = x_grid.t()

        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (size - 1) * 0.5
        variance = sigma * sigma

        two_pi_var = torch.tensor(2.0 * 3.141592653589793 * variance,
                                  dtype=self.dtype, device=self.device)
        two_var = torch.tensor(2.0 * variance,
                               dtype=self.dtype, device=self.device)
        gaussian_kernel = two_pi_var.reciprocal() * \
                        torch.exp(
                            -torch.sum((xy_grid - mean) ** 2., dim=-1) *
                            two_var.reciprocal()
                        )

        gaussian_kernel = gaussian_kernel * gaussian_kernel.to(torch.float64).sum().to(self.dtype).reciprocal()
        return gaussian_kernel




class MultiAnimatBodies(Body):

    def __init__(self, device, x, y, experiment_options, z=None, eps=0.05, compute_interp=True,
                 nsamples=None, msamples=None, ksamples=None, plotting=False, plotting_meshes=False,
                 suit=0.0, use_kernels=False, **kwargs):
        """Union of bodies from one or more MuJoCo/SDF model files.

        Mesh-based bodies that share the same mesh file (and scale) are
        automatically deduplicated: the expensive open3d → skfmm → interpolation
        pipeline runs only once per unique mesh, and the resulting BodyMesh
        is reused (with its own pose) for every duplicate.
        """
        super().__init__(device, x, y, z=z, eps=eps)
        if not use_kernels:
            self._setup_grids()

        self.suit = suit
        self.plotting        = plotting
        self.plotting_meshes = plotting_meshes

        self.body_ids = []
        self.bodies = []

        # ---- mesh SDF deduplication cache ----------------------------
        # key: (mesh_gpath, scale)  →  BodyMesh instance (used as template)
        _mesh_body_cache: dict[tuple, BodyMesh] = {}

        for animat_i, animat in enumerate(experiment_options.animats):
            sdf        = _import_model_sdf().read(animat.sdf)[0]
            sdf_folder = os.path.dirname(animat.sdf)
            morphology_links = getattr(getattr(animat, "morphology", None), "links", None)

            for link_i, link in enumerate(sdf.links):
                # ---- extract MuJoCo / SDF visual colour (RGBA) ----
                _link_rgba = None
                if hasattr(link, "visuals") and link.visuals:
                    _vis = link.visuals[0]
                    if hasattr(_vis, "color") and _vis.color is not None:
                        _link_rgba = list(_vis.color)  # [R, G, B, A]

                morphology_link = None
                if morphology_links is not None and link_i < len(morphology_links):
                    morphology_link = morphology_links[link_i]

                link_fluid_interaction = True
                if morphology_link is not None:
                    link_fluid_interaction = getattr(
                        morphology_link,
                        "fluid_interaction",
                        link_fluid_interaction,
                    )

                collisions = link["collisions"]
                if not collisions:
                    if link_fluid_interaction:
                        raise ValueError(
                            f"Link '{link['name']}' in '{animat.sdf}' has no collision geometry "
                            "but morphology.fluid_interaction=True. "
                            "Add collision geometry or disable fluid interaction for that link."
                        )
                    print(f"  Skipping non-fluid link without collisions: {link['name']}")
                    continue

                initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
                link_extras = {}
                if morphology_link is not None:
                    link_extras = dict(getattr(morphology_link, "extras", {}) or {})

                for collision in collisions:
                    collision_pose = np.array(
                        collision["pose"] if "pose" in collision else np.zeros(6),
                        dtype=x.cpu().numpy().dtype,
                    )
                    geometry = collision["geometry"]
                    if "uri" in geometry:
                        mesh_name = geometry["uri"]
                        mesh_gpath = os.path.normpath(sdf_folder + "/" + mesh_name)
                        update_funcs = (
                            lambda t: 180,
                            [
                                lambda t, initial_pose=initial_pose: -initial_pose[0],
                                lambda t, initial_pose=initial_pose: -initial_pose[1],
                            ]
                        )

                        scale = 1
                        local_kwargs = dict(kwargs)
                        if "scale" in geometry:
                            scale_vec = geometry["scale"]
                            assert scale_vec[0] == scale_vec[1] == scale_vec[2], "Non-uniform scaling not supported."
                            scale = scale_vec[0]
                            local_kwargs["scale"] = scale
                            local_kwargs["zpos"] = link.pose[2]

                        cache_key = (mesh_gpath, scale)
                        if cache_key in _mesh_body_cache:
                            # Reuse the already-computed SDF data
                            template = _mesh_body_cache[cache_key]
                            body = BodyMesh(
                                device, x, y,
                                mesh_gpath,
                                update_funcs,
                                z=self.z,
                                eps=eps,
                                compute_interp=False,  # skip heavy computation
                                nsamples=template.nsamples,
                                msamples=template.msamples,
                                ksamples=template.ksamples,
                                suit=suit,
                                plotting_meshes=False,
                                **local_kwargs
                            )
                            # Copy the pre-computed SDF interpolation data
                            body.sdf       = template.sdf
                            body.bb        = template.bb
                            body.cnt       = template.cnt.clone()
                            body.cnt_update = template.cnt_update.clone()
                            body.curv_coord = template.curv_coord
                            body.sign_vec   = template.sign_vec
                            body.ds         = template.ds
                            body.mask       = template.mask
                            print(f"  Reusing cached SDF for {mesh_gpath} (scale={scale})")
                        else:
                            body = BodyMesh(
                                device, x, y,
                                mesh_gpath,
                                update_funcs,
                                z=self.z,
                                eps=eps,
                                compute_interp=compute_interp,
                                nsamples=nsamples, msamples=msamples, ksamples=ksamples,
                                suit=suit,
                                plotting_meshes=plotting_meshes,
                                **local_kwargs
                            )
                            _mesh_body_cache[cache_key] = body

                    elif "radius" in geometry and "length" in geometry:
                        radius = torch.tensor(geometry["radius"], dtype=x.dtype, device=x.device)
                        length = torch.tensor(geometry["length"], dtype=x.dtype, device=x.device)
                        if self.ndim == 2:
                            if "L" in link["name"]:
                                side = "L"
                            elif "R" in link["name"]:
                                side = "R"
                            else:
                                raise ValueError("Capsule link name must contain 'L' or 'R' to define the side.")

                        if self.ndim == 3:
                            sdf_fun = (
                                lambda x, y, z, radius=radius, length=length:
                                capsule_3d(x, y, z, radius, radius, length)
                            )
                            update_maps = (
                                lambda t: (0.0, 0.0, 0.0),
                                [
                                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                                    lambda t, initial_pose=initial_pose: -initial_pose[2],
                                ],
                            )
                        else:
                            sdf_fun = (
                                lambda x, y, radius=radius, length=length, side=side:
                                sdUnevenCapsule(x, y, radius, radius, length, side=side)
                            )
                            update_maps = (
                                lambda t: 0,
                                [
                                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                                ],
                            )
                        # Analytical local_aabb in body-centred coordinates.
                        # The SDF callables (capsule_3d / sdUnevenCapsule) are
                        # defined at the local origin (0,0[,0]), so the AABB
                        # is derived purely from the geometric parameters plus
                        # the BDIM band margin (eps + 4*h).
                        _h_grid = float(x[1].item() - x[0].item())
                        _bm = float(eps) + 4.0 * _h_grid
                        _r = float(radius.item())
                        _l = float(length.item())
                        if self.ndim == 3:
                            # capsule_3d: axis along z, cylindrical section
                            # from z=-l/2 to z=+l/2, hemispherical caps of
                            # radius r at each end.
                            _local_aabb = torch.tensor(
                                [[-_r - _bm, -_r - _bm, -(0.5 * _l + _r) - _bm],
                                 [ _r + _bm,  _r + _bm,  (0.5 * _l + _r) + _bm]],
                                dtype=x.dtype, device=x.device,
                            )
                        else:
                            # 2-D sdUnevenCapsule(x, y, r, r, l, side):
                            #   side="L": pill runs along -y, from y=0 to y=-l
                            #   side="R": pill runs along +y, from y=0 to y=+l
                            if side == "L":
                                _local_aabb = torch.tensor(
                                    [[-_r - _bm, -(_l + _r) - _bm],
                                     [ _r + _bm,          _r + _bm]],
                                    dtype=x.dtype, device=x.device,
                                )
                            else:  # side == "R"
                                _local_aabb = torch.tensor(
                                    [[-_r - _bm,        -_r - _bm],
                                     [ _r + _bm, (_l + _r) + _bm]],
                                    dtype=x.dtype, device=x.device,
                                )
                        body = BodyAnalytical(
                            device, x, y, sdf_fun, update_maps, z=self.z,
                            eps=eps, plotting=False, pre_update=False,
                            local_aabb=_local_aabb,
                        )
                        radius_cpu = radius.detach().cpu()
                        length_cpu = length.detach().cpu()
                        body.bb = [
                            [-radius_cpu, radius_cpu],
                            [-radius_cpu, radius_cpu],
                            [-(0.5 * length_cpu + radius_cpu), 0.5 * length_cpu + radius_cpu],
                        ]

                    elif "radius" in geometry and "length" not in geometry:
                        radius = torch.tensor(geometry["radius"], dtype=x.dtype, device=x.device)
                        if self.ndim == 3:
                            sdf_fun = (
                                lambda x, y, z, radius=radius:
                                sphere(x, y, z, xt=0, yt=0, zt=0, r=radius)
                            )
                            update_maps = (
                                lambda t: (0.0, 0.0, 0.0),
                                [
                                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                                    lambda t, initial_pose=initial_pose: -initial_pose[2],
                                ],
                            )
                        else:
                            sdf_fun = (
                                lambda x, y, radius=radius:
                                circle(x, y, xt=0, yt=0, r=radius)
                            )
                            update_maps = (
                                lambda t: 0,
                                [
                                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                                ],
                            )
                        # Analytical local_aabb in body-centred coordinates.
                        # sphere / circle SDF is sqrt(x^2+y^2[+z^2]) - r,
                        # centred at the body origin — AABB is just [-r-bm, r+bm]
                        # per axis.
                        _h_grid = float(x[1].item() - x[0].item())
                        _bm = float(eps) + 4.0 * _h_grid
                        _r = float(radius.item())
                        if self.ndim == 3:
                            _local_aabb = torch.tensor(
                                [[-_r - _bm, -_r - _bm, -_r - _bm],
                                 [ _r + _bm,  _r + _bm,  _r + _bm]],
                                dtype=x.dtype, device=x.device,
                            )
                        else:
                            _local_aabb = torch.tensor(
                                [[-_r - _bm, -_r - _bm],
                                 [ _r + _bm,  _r + _bm]],
                                dtype=x.dtype, device=x.device,
                            )
                        body = BodyAnalytical(
                            device, x, y, sdf_fun, update_maps, z=self.z,
                            eps=eps, plotting=False, pre_update=False,
                            local_aabb=_local_aabb,
                        )
                        radius_cpu = radius.detach().cpu()
                        body.bb = [
                            [-radius_cpu, radius_cpu],
                            [-radius_cpu, radius_cpu],
                            [-radius_cpu, radius_cpu],
                        ]

                    elif "size" in geometry:
                        size = torch.tensor(geometry["size"], dtype=x.dtype, device=x.device)
                        half_size = 0.5 * size
                        if self.ndim == 3:
                            sdf_fun = (
                                lambda x, y, z, half_size=half_size: box_3d(
                                    x, y, z,
                                    xb=half_size[0],
                                    yb=half_size[1],
                                    zb=half_size[2],
                                )
                            )
                            update_maps = (
                                lambda t: (0.0, 0.0, 0.0),
                                [
                                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                                    lambda t, initial_pose=initial_pose: -initial_pose[2],
                                ],
                            )
                        else:
                            sdf_fun = (
                                lambda x, y, half_size=half_size: box(
                                    x, y,
                                    xb=half_size[0],
                                    yb=half_size[1],
                                )
                            )
                            update_maps = (
                                lambda t: 0,
                                [
                                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                                ],
                            )
                        # Analytical local_aabb in body-centred coordinates.
                        # box / box_3d SDF is centred at origin with half-extents
                        # (xb, yb[, zb]) = half_size — AABB is [-hs-bm, hs+bm].
                        _h_grid = float(x[1].item() - x[0].item())
                        _bm = float(eps) + 4.0 * _h_grid
                        _hs = half_size.detach().cpu().tolist()
                        if self.ndim == 3:
                            _local_aabb = torch.tensor(
                                [[-_hs[0] - _bm, -_hs[1] - _bm, -_hs[2] - _bm],
                                 [ _hs[0] + _bm,  _hs[1] + _bm,  _hs[2] + _bm]],
                                dtype=x.dtype, device=x.device,
                            )
                        else:
                            _local_aabb = torch.tensor(
                                [[-_hs[0] - _bm, -_hs[1] - _bm],
                                 [ _hs[0] + _bm,  _hs[1] + _bm]],
                                dtype=x.dtype, device=x.device,
                            )
                        body = BodyAnalytical(
                            device, x, y, sdf_fun, update_maps, z=self.z,
                            eps=eps, plotting=False, pre_update=False,
                            local_aabb=_local_aabb,
                        )
                        body.bb = [
                            [-half_size[0].cpu(), half_size[0].cpu()],
                            [-half_size[1].cpu(), half_size[1].cpu()],
                            [-half_size[2].cpu(), half_size[2].cpu()],
                        ]

                    else:
                        raise ValueError("Unsupported geometry type in SDF.")

                    body.mujoco_rgba = _link_rgba
                    body.local_pose = collision_pose
                    body.name = link["name"]
                    body.collision_name = collision["name"] if "name" in collision else None
                    body.link_extras = link_extras
                    self.bodies.append(body)
                    self.body_ids.append([animat_i, link_i])

        self.nbodies = len(self.bodies)
        gs = self.grid_shape
        # Output fields — filled by streaming union in update() or
        # BDIMhandler3D.update().  No (nbodies, *gs) stacks needed.
        # Initialised to the SDF sentinel _FAR=1e4 (i.e. "outside body
        # everywhere") rather than zeros.  This lets BDIMhandler restrict
        # the first-step "dirty AABB" (the region reset to _FAR + recomputed
        # by Kernel A) to only the bodies' current footprint instead of the
        # whole grid — which on a 512³ run shrinks the int64 key buffers
        # from 4×1 GiB = 4 GiB to a few MB.  Cells the bodies never visit
        # keep their _FAR value, which is the correct "outside body" answer
        # for the BDIM stencil.
        self.sdf_val   = torch.full(gs, 1e4, device=device, dtype=self.dtype)
        # In kernel mode the staggered face-SDF and rigid-body face-velocity
        # tensors are per-step temporaries owned by FluidSolver.fluid_step —
        # they live only between Kernel A (streaming SDF) and Kernel B
        # (fused BDIM2 + var-dens) of the same step.  Skip the persistent
        # full-grid allocations here to save 6 * Ngrid * sizeof(float)
        # of permanent GPU storage per composite body.
        if not use_kernels:
            self.sdf_val_u = torch.zeros(gs, device=device, dtype=self.dtype)
            self.sdf_val_v = torch.zeros(gs, device=device, dtype=self.dtype)
            self.body_u    = torch.zeros(gs, device=device, dtype=self.dtype)
            self.body_v    = torch.zeros(gs, device=device, dtype=self.dtype)
            if self.ndim == 3:
                self.sdf_val_w = torch.zeros(gs, device=device, dtype=self.dtype)
                self.body_w    = torch.zeros(gs, device=device, dtype=self.dtype)
        self.com_pos   = torch.zeros((self.nbodies, self.ndim), device=device)

        # Null out any SDF tensor that was stored directly on a child body
        # (interpolators / callables are kept; raw Tensor SDFs are freed).
        for body in self.bodies:
            body.sdf = body.sdf if not isinstance(body.sdf, torch.Tensor) else None


