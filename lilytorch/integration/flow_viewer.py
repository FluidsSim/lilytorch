"""
FlowViewer – render a fluid field (e.g. omega_z) as coloured spheres
directly inside the MuJoCo viewer **and** in the recorded video.

Spheres are injected into both:
- the interactive viewer's ``user_scn`` (visible in the MuJoCo GUI), and
- the ``CameraRecording`` extension's offscreen renderer (visible in the
  saved MP4 video).

Usage
-----
Add to the simulation extensions list in your gen_configs file, **after**
the FluidExtension entry::

    {
        "loader": "lilytorch.integration.flow_viewer.FlowViewer",
        "config": {
            "field"        : "omega_z",   # which field to display
            "max_spheres"  : 4000,        # budget for visual geoms
            "iso_fraction" : 0.15,        # iso = fraction * peak |field|
            "smooth_sigma" : 2.5,         # Gaussian smoothing (grid-cells)
            "crop_boundary": 3,           # cells to crop from domain faces
            "sphere_size"  : 0.004,       # radius of each sphere
            "update_every" : null,        # null → same as solver.save_every
        }
    }

Works with ``headless: false`` (viewer + video) and ``headless: true``
(video only via CameraRecording — no interactive viewer needed).
"""

import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter

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


class FlowViewer(TaskExtension):
    """Render a flow field as coloured spheres in the MuJoCo viewer."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        field: str = "omega_z",
        max_spheres: int = 4000,
        iso_fraction: float = 0.15,
        smooth_sigma: float = 2.5,
        crop_boundary: int = 3,
        sphere_size: float = 0.02,
        update_every: int | None = None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.field_name = field
        self.max_spheres = max_spheres
        self.iso_fraction = iso_fraction
        self.smooth_sigma = smooth_sigma
        self.crop_boundary = crop_boundary
        self.sphere_size = sphere_size
        self.update_every = update_every

        self._viewer = None
        self._fluid_ext = None        # reference to FluidExtension
        self._geom_start = None       # index in user_scn.geoms where our slots begin
        self._n_active = 0            # how many spheres are currently visible
        self._field_fn = FIELD_MAP.get(field)
        self._iteration = 0
        self._initialized = False
        self._active_positions = None  # (max_spheres, 3) float64
        self._active_colors = None     # (max_spheres, 4) float32
        self._cam_renderer_patched = False

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            field=config.get("field", "omega_z"),
            max_spheres=config.get("max_spheres", 4000),
            iso_fraction=config.get("iso_fraction", 0.15),
            smooth_sigma=config.get("smooth_sigma", 2.5),
            crop_boundary=config.get("crop_boundary", 3),
            sphere_size=config.get("sphere_size", 0.004),
            update_every=config.get("update_every", None),
        )

    # ── lifecycle hooks ──────────────────────────────────────────────

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._viewer = task.viewer

        # Skip re-allocation on episode re-initialization
        if self._initialized:
            return

        # Find the FluidExtension among sibling extensions
        from lilytorch.integration.extensions import FluidExtension
        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break
        if self._fluid_ext is None:
            print("[FlowViewer] FluidExtension not found – disabled.")
            return

        if self._field_fn is None:
            print(f"[FlowViewer] Unknown field '{self.field_name}', "
                  f"choose from: {list(FIELD_MAP)}. Disabled.")
            return

        # Allocate storage for active sphere data
        # (shared between the interactive viewer and the offscreen renderer)
        self._active_positions = np.zeros((self.max_spheres, 3), dtype=np.float64)
        self._active_colors = np.zeros((self.max_spheres, 4), dtype=np.float32)

        # Pre-allocate sphere slots in user_scn (interactive viewer)
        if self._viewer is not None:
            scn = self._viewer.user_scn
            self._geom_start = scn.ngeom
            _eye3 = np.eye(3, dtype=np.float64).ravel()
            _transparent = np.array([0, 0, 0, 0], dtype=np.float32)
            _zero = np.zeros(3, dtype=np.float64)
            _sz = np.array([self.sphere_size, 0, 0], dtype=np.float64)
            for _ in range(self.max_spheres):
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    _sz, _zero, _eye3, _transparent,
                )
                scn.ngeom += 1
            print(f"[FlowViewer] Reserved {self.max_spheres} sphere slots "
                  f"(geom {self._geom_start}..{scn.ngeom - 1}).")
        else:
            print("[FlowViewer] No viewer (headless) – "
                  "spheres will only appear in recorded video.")

        # Patch CameraRecording's offscreen renderer so spheres
        # also appear in the recorded video
        self._patch_camera_renderer(task)

        self._initialized = True

    def _patch_camera_renderer(self, task: ExperimentTask):
        """Monkey-patch CameraRecording's offscreen renderer so that
        FlowViewer spheres are included in every recorded video frame.

        CameraRecording uses ``mujoco.Renderer`` which builds its own
        ``MjvScene`` via ``update_scene()`` — that scene does *not*
        contain the ``viewer.user_scn`` geoms.  We wrap the renderer's
        ``render()`` method to inject our sphere geoms into
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

        flow_viewer = self
        original_render = renderer.render
        _eye3 = np.eye(3, dtype=np.float64).ravel()
        _sz = np.array([self.sphere_size, 0, 0], dtype=np.float64)

        def _render_with_spheres(out=None):
            """Inject FlowViewer spheres into the offscreen scene,
            then call the real render."""
            n = flow_viewer._n_active
            if n > 0:
                scn = renderer.scene
                for i in range(n):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        _sz,
                        flow_viewer._active_positions[i],
                        _eye3,
                        flow_viewer._active_colors[i],
                    )
                    scn.ngeom += 1
            return original_render(out=out)

        renderer.render = _render_with_spheres
        self._cam_renderer_patched = True
        print("[FlowViewer] Patched CameraRecording renderer for video output.")

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        if self._fluid_ext is None or self._field_fn is None:
            self._iteration += 1
            return

        # Deferred patch: CameraRecording's renderer may not exist yet
        # during initialize_episode (depends on extension order).
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

        # Get current flow state
        u, v, w, p = fs.u0, fs.v0, getattr(fs, "w0", None), fs.p0
        if w is None:
            # 2D solver – FlowViewer only works in 3D
            if not getattr(self, "_warned_2d", False):
                print("[FlowViewer] WARNING: 2D solver detected – FlowViewer requires 3D. Skipping.")
                self._warned_2d = True
            return

        try:
            field = self._field_fn(fs, u, v, p, w)
        except Exception as e:
            print(f"[FlowViewer] field computation error: {e}")
            return

        # Coordinates
        x = fs.x.cpu().numpy()
        y = fs.y.cpu().numpy()
        z = fs.z.cpu().numpy()

        # Crop boundary
        c = self.crop_boundary
        if c > 0:
            field = field[c:-c, c:-c, c:-c]
            x, y, z = x[c:-c], y[c:-c], z[c:-c]

        # Smooth
        if self.smooth_sigma > 0:
            field = gaussian_filter(field, sigma=self.smooth_sigma)

        # SDF mask (exclude body interior)
        sdf = None
        if hasattr(fs, "composite_body") and hasattr(fs.composite_body, "sdf_val"):
            sdf = fs.composite_body.sdf_val.cpu().numpy()
            if c > 0:
                sdf = sdf[c:-c, c:-c, c:-c]

        # Determine threshold
        if sdf is not None:
            exterior = sdf.ravel() > 0
            abs_f = np.abs(field.ravel()[exterior])
        else:
            abs_f = np.abs(field.ravel())
        peak = float(abs_f.max()) if abs_f.size > 0 else 0.0
        threshold = self.iso_fraction * peak
        if threshold < 1e-12:
            self._hide_all()
            return

        # Classify bipolar vs non-negative
        fmin, fmax = float(field.min()), float(field.max())
        is_bipolar = (fmin < -1e-12) and (fmax > 1e-12)

        # Build coordinate grids
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Find near-isosurface points
        if is_bipolar:
            mask_pos = field > threshold
            mask_neg = field < -threshold
            if sdf is not None:
                outside = sdf > 0
                mask_pos &= outside
                mask_neg &= outside
            mask_all = mask_pos | mask_neg
        else:
            mask_all = field > threshold
            mask_pos = mask_all
            mask_neg = np.zeros_like(mask_all)
            if sdf is not None:
                mask_all &= (sdf > 0)
                mask_pos = mask_all

        # Collect points
        pts_x = X[mask_all]
        pts_y = Y[mask_all]
        pts_z = Z[mask_all]
        is_pos = mask_pos[mask_all]  # True for positive, False for negative
        n_pts = len(pts_x)

        if n_pts == 0:
            self._hide_all()
            return

        # Sub-sample if we exceed budget
        if n_pts > self.max_spheres:
            idx = np.random.choice(n_pts, self.max_spheres, replace=False)
            idx.sort()
            pts_x = pts_x[idx]
            pts_y = pts_y[idx]
            pts_z = pts_z[idx]
            is_pos = is_pos[idx]
            n_pts = self.max_spheres

        # Colours: red for positive, blue for negative, orange for non-negative
        if is_bipolar:
            rgba_pos = np.array([0.85, 0.15, 0.15, 0.6], dtype=np.float32)
            rgba_neg = np.array([0.15, 0.15, 0.85, 0.6], dtype=np.float32)
        else:
            rgba_pos = np.array([0.88, 0.38, 0.19, 0.6], dtype=np.float32)
            rgba_neg = rgba_pos  # won't be used

        # Store sphere data (used by both viewer and offscreen renderer)
        for i in range(n_pts):
            self._active_positions[i] = [pts_x[i], pts_y[i], pts_z[i]]
            self._active_colors[i] = rgba_pos if is_pos[i] else rgba_neg

        # Always update _n_active so the offscreen renderer picks up
        # the spheres even in headless mode (no interactive viewer).
        self._n_active = n_pts

        # Update viewer sphere geoms (interactive GUI)
        if self._viewer is not None:
            scn = self._viewer.user_scn

            # Robustness: if scn.ngeom was reset to 0 by outside code (e.g. task reset),
            # our _geom_start is invalid. We must re-allocate.
            if self._geom_start is None or scn.ngeom < self._geom_start:
                 self._geom_start = scn.ngeom

            # Re-initialize geoms fully (safer than just updating .pos)
            _eye3 = np.eye(3, dtype=np.float64).ravel()
            _sz = np.array([self.sphere_size, 0, 0], dtype=np.float64)

            count = 0
            for i in range(n_pts):
                target_idx = self._geom_start + i
                if target_idx >= scn.maxgeom:
                    break

                # Check if we need to expand ngeom
                if target_idx >= scn.ngeom:
                    scn.ngeom = target_idx + 1

                g = scn.geoms[target_idx]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    _sz,
                    self._active_positions[i],
                    _eye3,
                    self._active_colors[i]
                )
                # Force category to Decor (0) as mjVIS flags don't hide it
                # or mjCAT_ALL
                g.category = mujoco.mjtCatBit.mjCAT_ALL
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1

                count += 1

            # Hide unused slots if they were previously used
            # (Note: if ngeom was reset, these are naturally hidden)
            max_used = self._geom_start + self._n_active
            curr_used = self._geom_start + count
            if max_used > curr_used and max_used <= scn.ngeom:
                 for i in range(curr_used, max_used):
                     scn.geoms[i].rgba[3] = 0

            self._n_active = count
        # In headless mode _n_active was already set above.

    def _hide_all(self):
        """Make all reserved spheres transparent."""
        if self._viewer is not None and self._geom_start is not None:
            scn = self._viewer.user_scn
            for i in range(self._n_active):
                scn.geoms[self._geom_start + i].rgba[3] = 0
        self._n_active = 0
