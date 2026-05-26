"""
FlowIsoViewer – render a fluid field as a quasi-isosurface inside the
MuJoCo viewer and the recorded video.

Unlike ``FlowViewer`` (which paints grid cells above |field| > threshold
as spheres), this extension runs **marching cubes** on the 3D field to
get a true triangulated isosurface, then approximates each vertex with a
small *oriented flat ellipsoid* — a "tile" aligned with the local
surface normal.  The resulting cloud of tiles visually approximates a
continuous isosurface while using only MuJoCo's built-in primitive
geoms, so it composes correctly with the MuJoCo-textured body meshes
(the textures on the animat remain fully visible).

Tiles are injected into both:
- the interactive viewer's ``user_scn`` (visible in the MuJoCo GUI), and
- the ``CameraRecording`` extension's offscreen renderer (visible in
  the saved MP4 video).

Usage
-----
Add to the simulation extensions list, **after** the FluidExtension
entry::

    {
        "loader": "lilytorch.integration.flow_iso_viewer.FlowIsoViewer",
        "config": {
            "field":          "omega_z",   # which field to display
            "max_tiles":      8000,        # budget for visual geoms
            "iso_fraction":   0.15,        # iso = fraction * peak |field|
            "iso_value":      null,        # absolute iso threshold; overrides iso_fraction
            "smooth_sigma":   2.5,         # Gaussian smoothing (grid-cells)
            "crop_boundary":  3,           # cells to crop from domain faces
            "tile_size":      0.005,       # in-plane half-extent of each tile
            "tile_thickness": 0.001,       # half-extent along normal (thin)
            "tile_alpha":     0.7,         # tile opacity in [0, 1]
            "exclude_body":   true,        # drop tiles whose nearest cell is inside an SDF body
            "update_every":   null,        # null → same cadence as solver.save_every
        }
    }

Notes
-----
- Requires ``scikit-image`` (marching cubes).
- 3D solver only.  In 2D the extension silently disables itself.
- Tile orientation is built from the marching-cubes vertex normal, with
  the local Z axis aligned with the surface normal and X/Y filling the
  tangent plane.
"""

import numpy as np
import mujoco

from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics


# ── Field extractors ─────────────────────────────────────────────────
# Each receives (fluid_solver, u, v, p, w) and returns a numpy array.

def _vort_components(fs, u, v, w):
    return fs.vorticity_components(u, v, w)

def _field_omega_x(fs, u, v, p, w):
    return _vort_components(fs, u, v, w)["omega_x"].cpu().numpy()

def _field_omega_y(fs, u, v, p, w):
    return _vort_components(fs, u, v, w)["omega_y"].cpu().numpy()

def _field_omega_z(fs, u, v, p, w):
    return _vort_components(fs, u, v, w)["omega_z"].cpu().numpy()

def _field_omega_mag(fs, u, v, p, w):
    return _vort_components(fs, u, v, w)["omega_mag"].cpu().numpy()

def _field_vel_mag(fs, u, v, p, w):
    return (u**2 + v**2 + w**2).sqrt().cpu().numpy()

def _field_pressure(fs, u, v, p, w):
    return p.cpu().numpy()


FIELD_MAP = {
    "omega_x":   _field_omega_x,
    "omega_y":   _field_omega_y,
    "omega_z":   _field_omega_z,
    "omega_mag": _field_omega_mag,
    "vel_mag":   _field_vel_mag,
    "pressure":  _field_pressure,
}


def _bases_from_normals(normals: np.ndarray) -> np.ndarray:
    """Build (M, 3, 3) rotation matrices whose third column is each input
    normal and whose first/second columns span the tangent plane.

    ``normals`` is assumed to be unit length; non-finite or zero normals
    are replaced by world-Z so the resulting matrix is still valid.
    """
    n = np.asarray(normals, dtype=np.float64)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("normals must be (M, 3)")

    # Sanitize: replace zero / non-finite with world-Z.
    mag = np.linalg.norm(n, axis=1)
    bad = (~np.isfinite(mag)) | (mag < 1e-12)
    n = np.where(bad[:, None], np.array([[0.0, 0.0, 1.0]]), n)
    mag = np.where(bad, 1.0, mag)
    n = n / mag[:, None]

    # Pick a helper axis not parallel to n (per row).
    use_z = (np.abs(n[:, 2]) < 0.9)
    helper = np.where(
        use_z[:, None],
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
    )
    t1 = np.cross(helper, n)
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
    t1 = t1 / np.maximum(t1_norm, 1e-12)
    t2 = np.cross(n, t1)

    # Columns: [t1, t2, n] → local X, Y, Z.
    bases = np.stack([t1, t2, n], axis=2)  # (M, 3, 3)
    return bases


class FlowIsoViewer(TaskExtension):
    """Render a flow field as marching-cubes oriented tiles in the MuJoCo viewer."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        field: str = "omega_z",
        max_tiles: int = 8000,
        iso_fraction: float = 0.15,
        iso_value: float | None = None,
        smooth_sigma: float = 2.5,
        crop_boundary: int = 3,
        tile_size: float = 0.005,
        tile_thickness: float = 0.001,
        tile_alpha: float = 0.7,
        exclude_body: bool = True,
        update_every: int | None = None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.field_name = field
        self.max_tiles = int(max_tiles)
        self.iso_fraction = float(iso_fraction)
        self.iso_value = float(iso_value) if iso_value is not None else None
        self.smooth_sigma = float(smooth_sigma)
        self.crop_boundary = int(crop_boundary)
        self.tile_size = float(tile_size)
        self.tile_thickness = float(tile_thickness)
        self.tile_alpha = float(np.clip(tile_alpha, 0.0, 1.0))
        self.exclude_body = bool(exclude_body)
        self.update_every = update_every

        self._viewer = None
        self._fluid_ext = None
        self._geom_start: int | None = None
        self._n_active: int = 0
        self._field_fn = FIELD_MAP.get(field)
        self._iteration = 0
        self._initialized = False
        self._active_positions: np.ndarray | None = None     # (max_tiles, 3)
        self._active_mats: np.ndarray | None = None          # (max_tiles, 9)
        self._active_colors: np.ndarray | None = None        # (max_tiles, 4)
        self._cam_renderer_patched = False
        self._tile_size_arr = np.array(
            [self.tile_size, self.tile_size, self.tile_thickness],
            dtype=np.float64,
        )
        self._warned_2d = False
        self._warned_skimage = False
        # Lazy import of marching_cubes — defer until we actually run.
        self._marching_cubes = None

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            field=config.get("field", "omega_z"),
            max_tiles=config.get("max_tiles", 8000),
            iso_fraction=config.get("iso_fraction", 0.15),
            iso_value=config.get("iso_value", None),
            smooth_sigma=config.get("smooth_sigma", 2.5),
            crop_boundary=config.get("crop_boundary", 3),
            tile_size=config.get("tile_size", 0.005),
            tile_thickness=config.get("tile_thickness", 0.001),
            tile_alpha=config.get("tile_alpha", 0.7),
            exclude_body=config.get("exclude_body", True),
            update_every=config.get("update_every", None),
        )

    # ── lifecycle hooks ──────────────────────────────────────────────

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._viewer = task.viewer

        if self._initialized:
            return

        # Find sibling FluidExtension
        from lilytorch.integration.extensions import FluidExtension
        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break
        if self._fluid_ext is None:
            print("[FlowIsoViewer] FluidExtension not found – disabled.")
            return

        if self._field_fn is None:
            print(f"[FlowIsoViewer] Unknown field '{self.field_name}', "
                  f"choose from: {list(FIELD_MAP)}. Disabled.")
            return

        try:
            from skimage.measure import marching_cubes
            self._marching_cubes = marching_cubes
        except ImportError:
            print("[FlowIsoViewer] scikit-image not installed – disabled.")
            return

        self._active_positions = np.zeros((self.max_tiles, 3), dtype=np.float64)
        self._active_mats = np.zeros((self.max_tiles, 9), dtype=np.float64)
        self._active_colors = np.zeros((self.max_tiles, 4), dtype=np.float32)

        # Pre-allocate ellipsoid slots in user_scn (interactive viewer)
        if self._viewer is not None:
            scn = self._viewer.user_scn
            self._geom_start = scn.ngeom
            _eye3 = np.eye(3, dtype=np.float64).ravel()
            _transparent = np.array([0, 0, 0, 0], dtype=np.float32)
            _zero = np.zeros(3, dtype=np.float64)
            reserved = 0
            for _ in range(self.max_tiles):
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    self._tile_size_arr, _zero, _eye3, _transparent,
                )
                scn.ngeom += 1
                reserved += 1
            if reserved < self.max_tiles:
                print(f"[FlowIsoViewer] WARNING: user_scn.maxgeom={scn.maxgeom} "
                      f"reached – only {reserved}/{self.max_tiles} tile slots "
                      f"reserved. Reduce max_tiles or increase maxgeom.")
                self.max_tiles = reserved
            print(f"[FlowIsoViewer] Reserved {self.max_tiles} tile slots "
                  f"(geom {self._geom_start}..{scn.ngeom - 1}).")
        else:
            print("[FlowIsoViewer] No viewer (headless) – "
                  "tiles will only appear in recorded video.")

        self._patch_camera_renderer(task)
        self._initialized = True

    def _patch_camera_renderer(self, task: ExperimentTask):
        """Monkey-patch ``CameraRecording``'s offscreen renderer so that
        the iso-tiles are included in every recorded video frame.

        ``CameraRecording`` uses ``mujoco.Renderer`` which builds its own
        ``MjvScene`` via ``update_scene()`` — that scene does *not*
        contain the ``viewer.user_scn`` geoms.  We wrap the renderer's
        ``render()`` method to inject our tile geoms into
        ``renderer.scene`` just before the actual render call.
        """
        if self._cam_renderer_patched:
            return

        cam_ext = None
        for ext in task.extensions:
            if type(ext).__name__ == 'CameraRecording':
                cam_ext = ext
                break
        if cam_ext is None:
            return

        renderer = getattr(cam_ext, 'renderer', None)
        if renderer is None:
            return

        self_ref = self
        original_render = renderer.render

        def _render_with_tiles(out=None):
            n = self_ref._n_active
            if n > 0:
                scn = renderer.scene
                for i in range(n):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        self_ref._tile_size_arr,
                        self_ref._active_positions[i],
                        self_ref._active_mats[i],
                        self_ref._active_colors[i],
                    )
                    scn.ngeom += 1
            return original_render(out=out)

        renderer.render = _render_with_tiles
        self._cam_renderer_patched = True
        print("[FlowIsoViewer] Patched CameraRecording renderer for video output.")

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        if self._fluid_ext is None or self._field_fn is None or self._marching_cubes is None:
            self._iteration += 1
            return

        # Deferred patch in case CameraRecording was not yet initialised.
        if not self._cam_renderer_patched:
            self._patch_camera_renderer(task)

        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        if handler is None:
            return
        fs = getattr(handler, "fluid_solver", None)
        if fs is None:
            return

        every = self.update_every or getattr(fs, "save_every", 200)
        iteration = getattr(handler, "iteration", self._iteration)
        self._iteration = iteration
        if iteration % every != 0:
            return

        u, v, w, p = fs.u0, fs.v0, getattr(fs, "w0", None), fs.p0
        if w is None:
            if not self._warned_2d:
                print("[FlowIsoViewer] WARNING: 2D solver detected – "
                      "FlowIsoViewer requires 3D. Skipping.")
                self._warned_2d = True
            return

        try:
            tiles = self._extract_surface_tiles(fs, u, v, p, w)
        except Exception as e:
            print(f"[FlowIsoViewer] surface extraction error: {e}")
            tiles = None

        if tiles is None:
            self._hide_all()
            return

        positions, mats, colors = tiles
        n = positions.shape[0]
        if n == 0:
            self._hide_all()
            return

        self._active_positions[:n] = positions
        self._active_mats[:n] = mats
        self._active_colors[:n] = colors
        self._n_active = n

        if self._viewer is not None:
            self._refresh_viewer_geoms(n)

    # ── core extraction ─────────────────────────────────────────────

    def _extract_surface_tiles(self, fs, u, v, p, w):
        """Run marching cubes on the chosen field, return per-tile
        (positions, rotation_mats_flat, colors).

        Returns ``None`` if no surface is found or skimage is unavailable.
        """
        field = self._field_fn(fs, u, v, p, w)
        x = fs.x.cpu().numpy()
        y = fs.y.cpu().numpy()
        z = fs.z.cpu().numpy()

        c = self.crop_boundary
        if c > 0:
            field = field[c:-c, c:-c, c:-c]
            x, y, z = x[c:-c], y[c:-c], z[c:-c]

        if self.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            field = gaussian_filter(field, sigma=self.smooth_sigma)

        sdf = None
        if self.exclude_body and hasattr(fs, "composite_body") and \
                hasattr(fs.composite_body, "sdf_val"):
            sdf = fs.composite_body.sdf_val.cpu().numpy()
            if c > 0:
                sdf = sdf[c:-c, c:-c, c:-c]

        # Threshold determination
        if sdf is not None:
            abs_f = np.abs(field.ravel()[sdf.ravel() > 0])
        else:
            abs_f = np.abs(field.ravel())
        peak = float(abs_f.max()) if abs_f.size > 0 else 0.0

        if self.iso_value is not None and self.iso_value > 0:
            threshold = float(self.iso_value)
        elif peak < 1e-12:
            return None
        else:
            threshold = self.iso_fraction * peak

        fmin, fmax = float(field.min()), float(field.max())
        is_bipolar = (fmin < -1e-12) and (fmax > 1e-12)

        # Grid spacing (uniform assumed by the solver)
        dx = (x[-1] - x[0]) / max(1, len(x) - 1) if len(x) > 1 else 1.0
        dy = (y[-1] - y[0]) / max(1, len(y) - 1) if len(y) > 1 else 1.0
        dz = (z[-1] - z[0]) / max(1, len(z) - 1) if len(z) > 1 else 1.0
        origin = np.array([float(x[0]), float(y[0]), float(z[0])])

        def _iso(level: float):
            try:
                verts, _faces, normals, _ = self._marching_cubes(
                    field, level=level, spacing=(dx, dy, dz),
                )
            except (ValueError, RuntimeError):
                return None
            if verts is None or len(verts) == 0:
                return None
            verts = verts + origin
            # marching_cubes normals point in the direction of *increasing*
            # field; for the negative isosurface we want outward-pointing
            # (away from the more-negative interior), which is the same
            # direction returned by skimage — both isosurfaces give
            # consistent normals.
            normals = -normals  # flip to point "outward" of the +iso lobe
            # (For our purposes, exact sign of the normal does not matter:
            # the tile is symmetric under z → -z.  Keeping a deterministic
            # sign helps with subtle shading.)
            if sdf is not None:
                ix = np.clip(np.rint((verts[:, 0] - origin[0]) / dx).astype(np.int64),
                             0, len(x) - 1)
                iy = np.clip(np.rint((verts[:, 1] - origin[1]) / dy).astype(np.int64),
                             0, len(y) - 1)
                iz = np.clip(np.rint((verts[:, 2] - origin[2]) / dz).astype(np.int64),
                             0, len(z) - 1)
                m = sdf[ix, iy, iz] > 0
                verts = verts[m]
                normals = normals[m]
            if len(verts) == 0:
                return None
            return verts, normals

        if is_bipolar:
            rgba_pos = np.array([0.85, 0.15, 0.15, self.tile_alpha], dtype=np.float32)
            rgba_neg = np.array([0.15, 0.15, 0.85, self.tile_alpha], dtype=np.float32)
        else:
            rgba_pos = np.array([0.88, 0.38, 0.19, self.tile_alpha], dtype=np.float32)
            rgba_neg = rgba_pos  # unused

        chunks_pos: list[np.ndarray] = []
        chunks_nrm: list[np.ndarray] = []
        chunks_col: list[np.ndarray] = []

        res_pos = _iso(threshold)
        if res_pos is not None:
            vp, npp = res_pos
            chunks_pos.append(vp)
            chunks_nrm.append(npp)
            chunks_col.append(np.broadcast_to(rgba_pos, (vp.shape[0], 4)).copy())

        if is_bipolar:
            res_neg = _iso(-threshold)
            if res_neg is not None:
                vn, nn = res_neg
                chunks_pos.append(vn)
                chunks_nrm.append(nn)
                chunks_col.append(np.broadcast_to(rgba_neg, (vn.shape[0], 4)).copy())

        if not chunks_pos:
            return None

        positions = np.concatenate(chunks_pos, axis=0)
        normals = np.concatenate(chunks_nrm, axis=0)
        colors = np.concatenate(chunks_col, axis=0).astype(np.float32)

        n = positions.shape[0]
        if n > self.max_tiles:
            idx = np.random.choice(n, self.max_tiles, replace=False)
            idx.sort()
            positions = positions[idx]
            normals = normals[idx]
            colors = colors[idx]

        bases = _bases_from_normals(normals)  # (M, 3, 3) cols = [t1, t2, n]
        mats_flat = bases.reshape(bases.shape[0], 9)  # row-major as MuJoCo wants

        return positions, mats_flat, colors

    # ── viewer geom maintenance ─────────────────────────────────────

    def _refresh_viewer_geoms(self, n_active: int):
        scn = self._viewer.user_scn

        # If user_scn was reset externally (ngeom < our start), re-anchor.
        if self._geom_start is None or scn.ngeom < self._geom_start:
            self._geom_start = scn.ngeom

        count = 0
        for i in range(n_active):
            target_idx = self._geom_start + i
            if target_idx >= scn.maxgeom:
                break
            if target_idx >= scn.ngeom:
                scn.ngeom = target_idx + 1

            g = scn.geoms[target_idx]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                self._tile_size_arr,
                self._active_positions[i],
                self._active_mats[i],
                self._active_colors[i],
            )
            g.category = mujoco.mjtCatBit.mjCAT_ALL
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            count += 1

        # Hide any previously-used slots beyond the new active range.
        max_used = self._geom_start + self._n_active
        curr_used = self._geom_start + count
        if max_used > curr_used and max_used <= scn.ngeom:
            for i in range(curr_used, max_used):
                scn.geoms[i].rgba[3] = 0

        self._n_active = count

    def _hide_all(self):
        if self._viewer is not None and self._geom_start is not None:
            scn = self._viewer.user_scn
            for i in range(self._n_active):
                idx = self._geom_start + i
                if idx < scn.ngeom:
                    scn.geoms[idx].rgba[3] = 0
        self._n_active = 0
