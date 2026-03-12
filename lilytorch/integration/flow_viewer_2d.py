"""
FlowViewer2D – render a 2-D fluid field (e.g. curl, pressure) as a
colour-mapped grid of thin, semi-transparent boxes directly inside the
MuJoCo viewer **and** in the recorded video.

This is the 2-D counterpart of ``FlowViewer`` (which uses isosurface
spheres in 3-D).  The 2-D field is down-sampled to a configurable
visual grid (``nx_vis × ny_vis``) and rendered as flat coloured tiles
covering the fluid domain.  The plane is placed at a configurable
z-offset so it sits just behind the animat when viewed from above.

The colour map uses a diverging red/blue scheme (like ``RdBu_r``):
positive values → red, negative → blue, zero → white/transparent.
The tiles are semi-transparent so the animat body remains visible.

Usage
-----
Add to the simulation extensions list in your gen_configs file, **after**
the FluidExtension entry::

    {
        "loader": "lilytorch.integration.flow_viewer_2d.FlowViewer2D",
        "config": {
            "field"          : "curl",        # "curl", "pressure", "divergence"
            "nx_vis"         : 80,            # visual grid cols  (x direction)
            "ny_vis"         : 40,            # visual grid rows  (y direction)
            "alpha"          : 0.55,          # tile opacity  (0 = invisible, 1 = opaque)
            "z_offset"       : 0.005,         # z-position of the plane (above pool floor, below animat)
            "vmin"           : null,          # colour-map min (null → auto)
            "vmax"           : null,          # colour-map max (null → auto)
            "smooth_sigma"   : 1.5,           # Gaussian pre-smoothing (grid cells)
            "crop_boundary"  : 2,             # cells cropped from domain edges
            "update_every"   : null,          # null → same as solver.save_every
        }
    }

Works with ``headless: false`` (viewer + video) and ``headless: true``
(video only, if CameraRecording is present).
"""

from __future__ import annotations

import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter

from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics


# ── 2-D field extractors ─────────────────────────────────────────────
# Each receives (fluid_solver, u, v, p) and returns a 2-D numpy array.

def _field_curl(fs, u, v, p):
    return fs.vorticity(u, v).cpu().numpy()

def _field_pressure(fs, u, v, p):
    return p.cpu().numpy()

def _field_divergence(fs, u, v, p):
    return fs.divergence(u, v).cpu().numpy()

def _field_vel_mag(fs, u, v, p):
    return (u**2 + v**2).sqrt().cpu().numpy()


FIELD_MAP_2D = {
    "curl":       _field_curl,
    "vorticity":  _field_curl,
    "pressure":   _field_pressure,
    "divergence": _field_divergence,
    "vel_mag":    _field_vel_mag,
}


# ── Diverging colour-map  (RdBu_r style) ────────────────────────────
# Maps a normalised value in [-1, 1] → RGBA.
# -1 → blue, 0 → white, +1 → red.  Values outside are clamped.

def _rdbu_r(t: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised diverging colour map.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Normalised values in [-1, 1].
    alpha : float
        Base opacity for the tiles.

    Returns
    -------
    rgba : np.ndarray, shape (N, 4), dtype float32
    """
    t = np.clip(t, -1.0, 1.0)
    rgba = np.empty((len(t), 4), dtype=np.float32)

    # Positive (red)
    pos = t > 0
    rgba[pos, 0] = 1.0
    rgba[pos, 1] = 1.0 - t[pos]
    rgba[pos, 2] = 1.0 - t[pos]
    # Negative (blue)
    neg = t < 0
    rgba[neg, 0] = 1.0 + t[neg]
    rgba[neg, 1] = 1.0 + t[neg]
    rgba[neg, 2] = 1.0
    # Zero
    zero = ~pos & ~neg
    rgba[zero, :3] = 1.0

    # Fade near-zero values toward transparent so that the background
    # is visible where the field is quiescent.
    # alpha = base_alpha * lerp(min_alpha_frac, 1.0, |t|^0.4)
    abs_t = np.abs(t)
    min_alpha_frac = 0.30          # floor: always at least 30 % of base alpha
    fading = min_alpha_frac + (1.0 - min_alpha_frac) * np.power(abs_t, 0.4)
    rgba[:, 3] = alpha * fading

    return rgba


class FlowViewer2D(TaskExtension):
    """Render a 2-D flow field as semi-transparent coloured tiles in MuJoCo."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        field: str = "curl",
        nx_vis: int = 80,
        ny_vis: int = 40,
        alpha: float = 0.55,
        z_offset: float = 0.005,
        vmin: float | None = None,
        vmax: float | None = None,
        smooth_sigma: float = 1.5,
        crop_boundary: int = 2,
        update_every: int | None = None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.field_name = field
        self.nx_vis = nx_vis
        self.ny_vis = ny_vis
        self.alpha = alpha
        self.z_offset = z_offset
        self.user_vmin = vmin
        self.user_vmax = vmax
        self.smooth_sigma = smooth_sigma
        self.crop_boundary = crop_boundary
        self.update_every = update_every

        self._viewer = None
        self._fluid_ext = None
        self._geom_start = None
        self._n_tiles = 0                # nx_vis * ny_vis
        self._field_fn = FIELD_MAP_2D.get(field)
        self._iteration = 0
        self._initialized = False
        self._cam_renderer_patched = False

        # Pre-computed tile geometry (filled at init)
        self._tile_cx: np.ndarray | None = None   # (n_tiles,) x-centres
        self._tile_cy: np.ndarray | None = None   # (n_tiles,) y-centres
        self._tile_half_dx: float = 0.0
        self._tile_half_dy: float = 0.0

        # Shared buffer for CameraRecording renderer patching
        self._active_colors: np.ndarray | None = None   # (n_tiles, 4) float32

    # ── class method for FARMS extension loader ──────────────────────

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            field=config.get("field", "curl"),
            nx_vis=config.get("nx_vis", 80),
            ny_vis=config.get("ny_vis", 40),
            alpha=config.get("alpha", 0.55),
            z_offset=config.get("z_offset", 0.005),
            vmin=config.get("vmin", None),
            vmax=config.get("vmax", None),
            smooth_sigma=config.get("smooth_sigma", 1.5),
            crop_boundary=config.get("crop_boundary", 2),
            update_every=config.get("update_every", None),
        )

    # ── lifecycle hooks ──────────────────────────────────────────────

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._viewer = task.viewer
        print(f"[FlowViewer2D] initialize_episode called. "
              f"viewer={'present' if self._viewer else 'None'}, "
              f"already_init={self._initialized}")

        if self._initialized:
            return

        # Find the FluidExtension among sibling extensions
        from lilytorch.integration.extensions import FluidExtension
        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break
        if self._fluid_ext is None:
            print("[FlowViewer2D] FluidExtension not found – disabled.")
            return

        if self._field_fn is None:
            print(f"[FlowViewer2D] Unknown field '{self.field_name}', "
                  f"choose from: {list(FIELD_MAP_2D)}. Disabled.")
            return

        # Compute tile geometry from the fluid solver grid
        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        if handler is not None:
            fs = getattr(handler, "fluid_solver", None)
        else:
            fs = None

        print(f"[FlowViewer2D] handler={'present' if handler else 'None'}, "
              f"solver={'present' if fs else 'None'}")

        if fs is not None:
            self._setup_tile_geometry(fs)
        else:
            # Will try again lazily on first before_step
            print("[FlowViewer2D] Fluid solver not ready yet – "
                  "tile geometry will be set up lazily.")

        self._initialized = True

    def _setup_tile_geometry(self, fs):
        """Compute tile positions and sizes from the fluid solver's domain."""
        xmin, xmax = float(fs.xmin), float(fs.xmax)
        ymin, ymax = float(fs.ymin), float(fs.ymax)

        print(f"[FlowViewer2D] Setting up tile geometry:")
        print(f"  Solver domain: x=[{xmin:.4f}, {xmax:.4f}], y=[{ymin:.4f}, {ymax:.4f}]")
        print(f"  Visual grid: {self.nx_vis} x {self.ny_vis} = {self.nx_vis * self.ny_vis} tiles")
        print(f"  z_offset={self.z_offset}, alpha={self.alpha}")

        # Cell centres for the visual grid
        dx_vis = (xmax - xmin) / self.nx_vis
        dy_vis = (ymax - ymin) / self.ny_vis
        self._tile_half_dx = dx_vis / 2.0
        self._tile_half_dy = dy_vis / 2.0

        print(f"  Tile size: {dx_vis:.5f} x {dy_vis:.5f} (half: {self._tile_half_dx:.5f} x {self._tile_half_dy:.5f})")

        cx = np.linspace(xmin + dx_vis / 2, xmax - dx_vis / 2, self.nx_vis)
        cy = np.linspace(ymin + dy_vis / 2, ymax - dy_vis / 2, self.ny_vis)
        CX, CY = np.meshgrid(cx, cy, indexing="ij")  # (nx_vis, ny_vis)
        self._tile_cx = CX.ravel()
        self._tile_cy = CY.ravel()
        self._n_tiles = self.nx_vis * self.ny_vis

        print(f"  Tile x range: [{self._tile_cx.min():.4f}, {self._tile_cx.max():.4f}]")
        print(f"  Tile y range: [{self._tile_cy.min():.4f}, {self._tile_cy.max():.4f}]")

        # Allocate shared colour buffer
        self._active_colors = np.zeros((self._n_tiles, 4), dtype=np.float32)

        # Pre-allocate tile (box) geoms in the viewer's user_scn
        if self._viewer is not None:
            scn = self._viewer.user_scn
            self._geom_start = scn.ngeom

            print(f"  Viewer user_scn: ngeom={scn.ngeom}, maxgeom={scn.maxgeom}")

            _eye3 = np.eye(3, dtype=np.float64).ravel()
            # Box half-size: dx/2, dy/2, and a thin but visible z-thickness
            _tile_z_half = 5e-4
            _sz = np.array(
                [self._tile_half_dx, self._tile_half_dy, _tile_z_half],
                dtype=np.float64,
            )
            # Start all tiles with a visible neutral colour so they are
            # immediately visible even before the first field update.
            _init_rgba = np.array([0.7, 0.7, 0.85, 0.25], dtype=np.float32)
            for k in range(self._n_tiles):
                if scn.ngeom >= scn.maxgeom:
                    print(f"[FlowViewer2D] WARNING: maxgeom ({scn.maxgeom}) "
                          f"reached after {k} tiles. Reduce nx_vis/ny_vis.")
                    self._n_tiles = k
                    break
                _pos = np.array(
                    [self._tile_cx[k], self._tile_cy[k], self.z_offset],
                    dtype=np.float64,
                )
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    _sz, _pos, _eye3, _init_rgba,
                )
                scn.ngeom += 1

            print(f"[FlowViewer2D] Reserved {self._n_tiles} tile slots "
                  f"(geom {self._geom_start}..{self._geom_start + self._n_tiles - 1}), "
                  f"z_offset={self.z_offset}, domain=[{xmin:.3f},{xmax:.3f}]x[{ymin:.3f},{ymax:.3f}].")
            print(f"  user_scn.ngeom now={scn.ngeom}")

            # ── DEBUG MARKER: bright red box at domain centre ─────────
            # Visible from any camera angle.  Remove once positioning is
            # confirmed.
            if scn.ngeom < scn.maxgeom:
                mid_x = (xmin + xmax) / 2.0
                mid_y = (ymin + ymax) / 2.0
                marker_sz = np.array([0.05, 0.05, 0.01], dtype=np.float64)
                marker_pos = np.array([mid_x, mid_y, self.z_offset + 0.005],
                                       dtype=np.float64)
                marker_rgba = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_BOX,
                    marker_sz, marker_pos, _eye3, marker_rgba,
                )
                scn.ngeom += 1
                print(f"  DEBUG: red marker at ({mid_x:.3f}, {mid_y:.3f}, {self.z_offset + 0.005:.4f})")
        else:
            print("[FlowViewer2D] No viewer (headless) – "
                  "tiles will only appear in recorded video.")

        # Patch CameraRecording so tiles appear in video too
        # (deferred; may not exist yet)

    def _patch_camera_renderer(self, task: ExperimentTask):
        """Monkey-patch CameraRecording's offscreen renderer so that
        FlowViewer2D tiles are included in every recorded video frame."""
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

        flow_viewer = self
        original_render = renderer.render
        _eye3 = np.eye(3, dtype=np.float64).ravel()

        def _render_with_tiles(out=None):
            """Inject FlowViewer2D tiles into the offscreen scene."""
            n = flow_viewer._n_tiles
            if n > 0 and flow_viewer._tile_cx is not None:
                scn = renderer.scene
                _sz = np.array(
                    [flow_viewer._tile_half_dx,
                     flow_viewer._tile_half_dy,
                     5e-4],
                    dtype=np.float64,
                )
                for i in range(n):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    _pos = np.array(
                        [flow_viewer._tile_cx[i],
                         flow_viewer._tile_cy[i],
                         flow_viewer.z_offset],
                        dtype=np.float64,
                    )
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_BOX,
                        _sz, _pos, _eye3,
                        flow_viewer._active_colors[i],
                    )
                    scn.ngeom += 1
            return original_render(out=out)

        renderer.render = _render_with_tiles
        self._cam_renderer_patched = True
        print("[FlowViewer2D] Patched CameraRecording renderer for video output.")

    # ── per-step update ──────────────────────────────────────────────

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        if self._fluid_ext is None or self._field_fn is None:
            self._iteration += 1
            return

        if not self._cam_renderer_patched:
            self._patch_camera_renderer(task)

        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        if handler is None:
            return

        fs = getattr(handler, "fluid_solver", None)
        if fs is None:
            return

        # Lazy tile-geometry setup (if fluid solver wasn't ready at init)
        if self._tile_cx is None:
            self._setup_tile_geometry(fs)
            if self._tile_cx is None:
                return

        # Only update at configured interval
        every = self.update_every or getattr(fs, "save_every", 200)
        iteration = getattr(handler, "iteration", self._iteration)
        self._iteration = iteration
        if iteration % every != 0:
            return

        # Get current 2-D flow state
        u, v, p = fs.u0, fs.v0, fs.p0
        w = getattr(fs, "w0", None)
        if w is not None:
            # This is a 3-D solver — FlowViewer2D is for 2-D only
            if not getattr(self, "_warned_3d", False):
                print("[FlowViewer2D] WARNING: 3D solver detected – "
                      "FlowViewer2D is for 2D simulations. "
                      "Use FlowViewer for 3D. Skipping.")
                self._warned_3d = True
            return

        try:
            field = self._field_fn(fs, u, v, p)
        except Exception as e:
            print(f"[FlowViewer2D] field computation error: {e}")
            return

        # Crop boundary cells
        c = self.crop_boundary
        if c > 0 and min(field.shape) > 2 * c:
            field = field[c:-c, c:-c]

        # Gaussian smoothing
        if self.smooth_sigma > 0:
            field = gaussian_filter(field, sigma=self.smooth_sigma)

        # SDF masking (set body-interior to 0 for cleaner visuals)
        if hasattr(fs, "composite_body") and hasattr(fs.composite_body, "sdf_val"):
            sdf = fs.composite_body.sdf_val.cpu().numpy()
            if c > 0 and min(sdf.shape) > 2 * c:
                sdf = sdf[c:-c, c:-c]
            # Inside the body: sdf < 0 → set field to 0
            if sdf.shape == field.shape:
                field = np.where(sdf < 0, 0.0, field)

        # Determine colour range
        vmin = self.user_vmin
        vmax = self.user_vmax
        if vmin is None or vmax is None:
            abs_f = np.abs(field)
            limit = float(np.percentile(abs_f, 99))
            if limit < 1e-12:
                limit = float(abs_f.max()) or 1.0
            if vmin is None:
                vmin = -limit
            if vmax is None:
                vmax = limit

        # Down-sample the field to visual grid (nx_vis × ny_vis)
        field_vis = self._downsample(field, self.nx_vis, self.ny_vis)

        # Normalise to [-1, 1]
        scale = max(abs(vmin), abs(vmax))
        if scale < 1e-15:
            scale = 1.0
        t = field_vis.ravel() / scale  # (n_tiles,)

        # Compute RGBA colours
        rgba = _rdbu_r(t, self.alpha)
        self._active_colors[:self._n_tiles] = rgba[:self._n_tiles]

        # Update viewer tile geoms
        if self._viewer is not None and self._geom_start is not None:
            scn = self._viewer.user_scn

            # Guard against scene resets
            if scn.ngeom < self._geom_start:
                self._geom_start = scn.ngeom

            _eye3 = np.eye(3, dtype=np.float64).ravel()
            _sz = np.array(
                [self._tile_half_dx, self._tile_half_dy, 5e-4],
                dtype=np.float64,
            )

            for k in range(self._n_tiles):
                idx = self._geom_start + k
                if idx >= scn.maxgeom:
                    break
                if idx >= scn.ngeom:
                    scn.ngeom = idx + 1

                g = scn.geoms[idx]
                _pos = np.array(
                    [self._tile_cx[k], self._tile_cy[k], self.z_offset],
                    dtype=np.float64,
                )
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    _sz, _pos, _eye3,
                    self._active_colors[k],
                )
                g.category = mujoco.mjtCatBit.mjCAT_ALL
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _downsample(field: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """Downsample a 2-D array to (nx, ny) by block-averaging.

        Uses reshape-based averaging when possible, otherwise falls back
        to ``scipy.ndimage.zoom``.
        """
        from scipy.ndimage import zoom

        h, w = field.shape
        if h == nx and w == ny:
            return field.copy()
        return zoom(field, (nx / h, ny / w), order=1)
