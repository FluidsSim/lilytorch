"""
ParticleViewer – Lagrangian dye-particle advection rendered as coloured
spheres directly inside the MuJoCo viewer **and** in the recorded video.

Particles are seeded from the trailing edge of the body (SDF zero-level-set)
every ``seed_interval`` solver steps, advected with the live velocity field
using RK2 (Heun's method), and optionally dispersed with a stochastic
turbulent diffusivity model.

Spheres are injected into both:
- the interactive viewer's ``user_scn`` (visible in the MuJoCo GUI), and
- the ``CameraRecording`` extension's offscreen renderer (visible in the
  saved MP4 video).

Usage
-----
Add to the simulation extensions list in your gen_configs file, **after**
the FluidExtension entry::

    {
        "loader": "lilytorch.integration.particle_viewer.ParticleViewer",
        "config": {
            "max_particles"      : 8000,         # budget for visual geoms
            "seed_n_particles"   : 24,           # seeds per injection
            "seed_interval"      : 1,            # inject every N solver steps
            "turb_diffusivity"   : 0.0,          # stochastic dispersion [m²/s]
            "sphere_size"        : 0.003,        # radius of each sphere
            "particle_color"     : [1.0, 0.0, 0.65, 0.85],  # RGBA
            "trail_length"       : 0,            # max batches (0 = unlimited)
            "update_every"       : null,         # null → every solver step
            "n_z_layers"         : 3,            # z replicas for 3-D tail seeding
            "floor_color"        : null,         # RGBA or "#hex" for arena geoms
            "body_color"         : null,         # RGBA or "#hex" for animat geoms
        }
    }

Works with ``headless: false`` (viewer + video) and ``headless: true``
(video only via CameraRecording — no interactive viewer needed).
"""

from __future__ import annotations

import numpy as np
import torch
import mujoco

from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics


# =====================================================================
#  SDF-based seeding helpers  (extracted from plot_particles_robot.py)
# =====================================================================

def _sdf_boundary_xy(sdf: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Return (bx, by) of SDF zero-level-set boundary cells (2-D projection)."""
    from scipy import ndimage
    if sdf.ndim == 3:
        sdf_2d = sdf[1:-1, 1:-1, :].min(axis=2)
    elif sdf.ndim == 2:
        sdf_2d = sdf[1:-1, 1:-1]
    else:
        return np.array([]), np.array([])
    xp, yp = x[1:-1], y[1:-1]
    inside = sdf_2d <= 0
    boundary = ndimage.binary_dilation(inside, iterations=1) & ~inside
    if not np.any(boundary):
        return np.array([]), np.array([])
    bi, bj = np.where(boundary)
    return xp[bi], yp[bj]


def _body_z_extent(sdf, z):
    """Return (zmin, zmax) of the body from SDF, or (None, None)."""
    if sdf is None or sdf.ndim != 3 or z is None:
        return None, None
    body_mask = (sdf[1:-1, 1:-1, :] <= 0).any(axis=(0, 1))
    if body_mask.any():
        idx = np.where(body_mask)[0]
        return float(z[idx[0]]), float(z[idx[-1]])
    return None, None


def _replicate_at_z_levels(pts_xy: np.ndarray, sdf, z, n_z_layers: int):
    """Expand (2, N) → (3, N*n_z_layers) across the body z-extent."""
    if z is None:
        return pts_xy
    zlo, zhi = _body_z_extent(sdf, z)
    if zlo is None:
        zmid = 0.5 * (float(z[1]) + float(z[-2]))
        zlo, zhi = zmid - 0.01, zmid + 0.01
    dz = zhi - zlo
    if n_z_layers == 1:
        z_levels = np.array([0.5 * (zlo + zhi)])
    else:
        z_levels = np.linspace(zlo + 0.05 * dz, zhi - 0.05 * dz, n_z_layers)
    n = pts_xy.shape[1]
    layers = []
    for zl in z_levels:
        layer = np.vstack([pts_xy, np.full((1, n), zl)])
        layers.append(layer)
    return np.concatenate(layers, axis=1)


def seed_tail_from_sdf(sdf: np.ndarray, x: np.ndarray, y: np.ndarray,
                       n_particles: int = 24, z: np.ndarray | None = None,
                       n_z_layers: int = 3,
                       ) -> np.ndarray:
    """Seed *n_particles* near the tail tip — identical to
    ``_extract_body_tail_from_sdf`` in plot_particles_robot.py.

    1. Extract all SDF zero-level-set boundary cells.
    2. Pick the 8 boundary cells with the **smallest x** (trailing edge).
    3. Densify by interpolating along that cluster to ``n_particles``.
    4. Optionally replicate across z-layers for 3-D.

    Returns (ndim, N) numpy array.
    """
    bx, by = _sdf_boundary_xy(sdf, x, y)
    if len(bx) == 0:
        ndim = 3 if (z is not None and sdf.ndim == 3) else 2
        return np.zeros((ndim, 0))

    n_anchor = min(8, len(bx))
    sel = np.argsort(bx)[:n_anchor]
    ax_, ay_ = bx[sel], by[sel]

    # Densify: interpolate along the anchor cluster to n_particles
    if n_particles > n_anchor:
        t_anc = np.linspace(0, 1, n_anchor)
        t_dense = np.linspace(0, 1, n_particles)
        ax_ = np.interp(t_dense, t_anc, ax_)
        ay_ = np.interp(t_dense, t_anc, ay_)

    pts_xy = np.stack([ax_, ay_], axis=0)
    if sdf.ndim == 3 and z is not None:
        return _replicate_at_z_levels(pts_xy, sdf, z, n_z_layers)
    return pts_xy


# =====================================================================
#  Advection helpers
# =====================================================================

@torch.no_grad()
def _advect_particles(particles: torch.Tensor,
                      interps: list,
                      dt: float,
                      turb_diff: float = 0.0) -> torch.Tensor:
    """RK2 (Heun's method) advection + optional stochastic turbulent diffusion.

    Heun's method evaluates the velocity at the current position (k1)
    and at the Forward-Euler predicted position (k2), then averages
    the two.  This is second-order accurate and dramatically reduces
    the trajectory noise compared to plain Forward-Euler.

    Parameters
    ----------
    particles : (ndim, N)  tensor of particle positions.
    interps   : list[RegularGridInterpolator]  one per axis.
    dt        : timestep.
    turb_diff : effective turbulent diffusivity D_eff [m²/s].
    """
    if particles.shape[1] == 0:
        return particles

    ndim = particles.shape[0]

    # k1 = velocity at current position
    k1 = torch.stack([interp(*particles) for interp in interps])  # (ndim, N)

    # Predicted position (Forward-Euler)
    p_pred = particles + k1 * dt

    # k2 = velocity at predicted position
    k2 = torch.stack([interp(*p_pred) for interp in interps])  # (ndim, N)

    # Heun update: x_{n+1} = x_n + dt/2 * (k1 + k2)
    particles.add_(k1 + k2, alpha=0.5 * dt)

    if turb_diff > 0.0:
        sigma = (2.0 * turb_diff * abs(dt)) ** 0.5
        particles.add_(torch.randn_like(particles) * sigma)
    return particles


@torch.no_grad()
def _cull_out_of_domain(particles: torch.Tensor,
                        bounds: list[tuple[float, float]]) -> torch.Tensor:
    """Remove particles outside the domain bounds."""
    if particles.shape[1] == 0:
        return particles
    mask = torch.ones(particles.shape[1], dtype=torch.bool,
                      device=particles.device)
    for i, (lo, hi) in enumerate(bounds):
        mask &= (particles[i] >= lo) & (particles[i] <= hi)
    return particles[:, mask]


@torch.no_grad()
def _cull_inside_body(particles: torch.Tensor,
                      sdf_val: torch.Tensor,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      margin: float = 0.0) -> torch.Tensor:
    """Remove particles inside the body (sdf <= margin)."""
    if particles.shape[1] == 0:
        return particles
    sdf_2d = sdf_val
    if sdf_val.ndim == 3:
        sdf_2d = sdf_val.min(dim=2).values
    ix = torch.searchsorted(x, particles[0].contiguous()).clamp(0, x.shape[0] - 1)
    iy = torch.searchsorted(y, particles[1].contiguous()).clamp(0, y.shape[0] - 1)
    return particles[:, sdf_2d[ix, iy] > margin]


# =====================================================================
#  ParticleViewer  TaskExtension
# =====================================================================

class ParticleViewer(TaskExtension):
    """Lagrangian dye-particle advection rendered as MuJoCo spheres.

    Supports both 2-D and 3-D solvers. Particles are seeded from the
    body SDF, advected with the live velocity field, and rendered as
    coloured spheres in the MuJoCo viewer and in the CameraRecording.
    """

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        max_particles: int = 8000,
        seed_n_particles: int = 24,
        seed_interval: int = 1,
        turb_diffusivity: float = 0.0,
        sphere_size: float = 0.003,
        particle_color: list[float] | None = None,
        trail_length: int = 0,
        update_every: int | None = None,
        n_z_layers: int = 3,
        body_cull_margin: float | None = None,
        cull_every: int = 10,
        floor_color: list[float] | str | None = None,
        body_color: list[float] | str | None = None,
        no_light: bool = False,
        light_color: list[float] | str | None = None,
        emissive_particles: bool = False,
    ):
        super().__init__()
        self.experiment_options = experiment_options

        # Config
        self.max_particles = max_particles
        self.seed_n_particles = seed_n_particles
        self.seed_interval = max(1, seed_interval)
        self.turb_diffusivity = turb_diffusivity
        self.sphere_size = sphere_size
        self.particle_rgba = np.array(
            particle_color or [1.0, 0.0, 0.65, 0.85], dtype=np.float32
        )
        self.trail_length = trail_length
        self.update_every = update_every
        self.n_z_layers = n_z_layers
        self.body_cull_margin = body_cull_margin
        self.cull_every = max(1, cull_every)
        self.floor_color = self._parse_color(floor_color)
        self.body_color = self._parse_color(body_color)
        self.no_light = no_light
        self.light_color = self._parse_color(light_color)
        self.emissive_particles = emissive_particles

        # Internal state
        self._viewer = None
        self._fluid_ext = None
        self._geom_start = None
        self._n_active = 0          # GUI geom count (may be clamped)
        self._n_active_video = 0   # full count for offscreen renderer
        self._iteration = 0
        self._initialized = False
        self._cam_ext = None             # reference to CameraRecording
        self._patched_renderer_id = None # id() of the renderer we patched
        self._patched_fn = None          # ref to our replacement function
        self._max_particles_video = 0    # unclamped limit for video
        self._grid_h = None  # grid spacing, set on first step
        self._tail_seeds_local = None  # (ndim, N) body-local seeds
        self._tail_link_id = None      # (animat_id, link_id)

        # Buffers (allocated in initialize_episode)
        self._active_positions = None   # (max_particles, 3)
        self._active_colors = None      # (max_particles, 4)
        self._particles = None          # (ndim, N) torch tensor
        self._interps = None            # list of RegularGridInterpolator
        self._ndim = None
        self._bounds = None             # list of (lo, hi) per axis
        self._device = None
        self._dtype = None

    @staticmethod
    def _parse_color(c) -> np.ndarray | None:
        """Accept RGBA list, hex string, or None."""
        if c is None:
            return None
        if isinstance(c, str):
            c = c.lstrip("#")
            if len(c) == 6:
                c += "ff"  # default alpha = 1
            return np.array(
                [int(c[i:i+2], 16) / 255.0 for i in range(0, 8, 2)],
                dtype=np.float32,
            )
        return np.array(c, dtype=np.float32)

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            max_particles=config.get("max_particles", 8000),
            seed_n_particles=config.get("seed_n_particles", 24),
            seed_interval=config.get("seed_interval", 1),
            turb_diffusivity=config.get("turb_diffusivity", 0.0),
            sphere_size=config.get("sphere_size", 0.003),
            particle_color=config.get("particle_color", None),
            trail_length=config.get("trail_length", 0),
            update_every=config.get("update_every", None),
            n_z_layers=config.get("n_z_layers", 3),
            body_cull_margin=config.get("body_cull_margin", None),
            cull_every=config.get("cull_every", 10),
            floor_color=config.get("floor_color", None),
            body_color=config.get("body_color", None),
            no_light=config.get("no_light", False),
            light_color=config.get("light_color", None),
            emissive_particles=config.get("emissive_particles", False),
        )

    # ── lifecycle hooks ──────────────────────────────────────────────

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._viewer = task.viewer

        if self._initialized:
            return

        # Find the FluidExtension among sibling extensions
        from lilytorch.integration.extensions import FluidExtension
        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break
        if self._fluid_ext is None:
            print("[ParticleViewer] FluidExtension not found – disabled.")
            return

        # Find CameraRecording (for later patching in before_step)
        for ext in task.extensions:
            if type(ext).__name__ == 'CameraRecording':
                self._cam_ext = ext
                break

        # Save the un-clamped max_particles for video rendering.
        # The offscreen mujoco.Renderer has maxgeom=10000 by default,
        # so the video can display many more geoms than the GUI user_scn.
        self._max_particles_video = self.max_particles

        # Allocate storage for active sphere data -- uses ORIGINAL
        # max_particles (before GUI clamping) so video can use full buffer.
        self._active_positions = np.zeros(
            (self.max_particles, 3), dtype=np.float64)
        self._active_colors = np.tile(
            self.particle_rgba, (self.max_particles, 1)).astype(np.float32)

        # Pre-allocate sphere slots in user_scn (interactive viewer)
        if self._viewer is not None:
            scn = self._viewer.user_scn
            self._geom_start = scn.ngeom
            # Clamp to available scene capacity
            max_geom = scn.maxgeom if hasattr(scn, 'maxgeom') else len(scn.geoms)
            avail = max_geom - scn.ngeom
            if self.max_particles > avail:
                print(f"[ParticleViewer] Clamping GUI max_particles from "
                      f"{self.max_particles} to {avail} (user_scn capacity). "
                      f"Video keeps {self._max_particles_video}.")
                self.max_particles = avail
            _eye3 = np.eye(3, dtype=np.float64).ravel()
            _transparent = np.array([0, 0, 0, 0], dtype=np.float32)
            _zero = np.zeros(3, dtype=np.float64)
            _sz = np.array([self.sphere_size, 0, 0], dtype=np.float64)
            for _ in range(self.max_particles):
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    _sz, _zero, _eye3, _transparent,
                )
                scn.ngeom += 1
            print(f"[ParticleViewer] Reserved {self.max_particles} sphere slots "
                  f"(geom {self._geom_start}..{scn.ngeom - 1}).")
        else:
            print("[ParticleViewer] No viewer (headless) – "
                  "spheres will only appear in recorded video.")

        # Override MuJoCo geom colours (floor / animat body)
        self._apply_geom_colors(task, physics)

        self._initialized = True

    def _ensure_video_patch(self):
        """Replace CameraRecording.before_step with a version that injects
        particle geoms between ``update_scene`` and ``render``.

        We replace the *instance* method on the CameraRecording object
        (a pure-Python class), avoiding monkey-patching C-extension types
        like ``mujoco.Renderer``.  Called on every ``before_step`` so
        we automatically patch once the renderer exists.  Also re-applies
        if something resets the instance attribute.
        """
        if self._cam_ext is None:
            return
        # Wait until CameraRecording has actually created its renderer
        if getattr(self._cam_ext, 'renderer', None) is None:
            return

        cam = self._cam_ext

        # Check if patch is still alive (re-apply if hijacked)
        if (self._patched_renderer_id is not None
                and getattr(cam, 'before_step', None) is self._patched_fn):
            return  # already patched and still in place

        if (self._patched_renderer_id is not None
                and getattr(cam, 'before_step', None) is not self._patched_fn):
            print("[PV-video] WARNING: CameraRecording.before_step was "
                  "reset – re-patching!")

        # ── Recreate the renderer with enough maxgeom for particles ──
        # The default maxgeom=10000 is too small: with ~400 model geoms,
        # only 9600 particle geoms can fit.  We need room for
        # max_particles (up to 100k after GUI clamp) + model geoms.
        needed_geom = self.max_particles + 1000  # headroom for model geoms
        old_maxgeom = cam.renderer.scene.maxgeom
        if old_maxgeom < needed_geom:
            model_ptr = cam.renderer._model
            cam.renderer.close()
            cam.renderer = mujoco.Renderer(
                model_ptr,
                width=cam.width,
                height=cam.height,
                max_geom=needed_geom,
            )
            print(f"[ParticleViewer] Recreated offscreen renderer: "
                  f"maxgeom {old_maxgeom} → {needed_geom}")

        pv = self
        _eye3 = np.eye(3, dtype=np.float64).ravel()
        _sz = np.array([pv.sphere_size, 0, 0], dtype=np.float64)
        _frame = [0]

        def _inject_particles_into_scene(renderer):
            """Push particle geoms into the renderer's scene."""
            n = pv._n_active_video
            if n <= 0:
                return 0
            scn = renderer.scene
            base = scn.ngeom
            limit = min(n, scn.maxgeom - base)
            if limit <= 0:
                return 0
            for i in range(limit):
                g = scn.geoms[base + i]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    _sz,
                    pv._active_positions[i],
                    _eye3,
                    pv._active_colors[i],
                )
                if pv.emissive_particles:
                    g.emission = 1.0
            scn.ngeom = base + limit
            return limit

        def _cam_before_step_with_particles(task, action, physics):
            """Drop-in replacement for CameraRecording.before_step that
            injects particle geoms between update_scene and render."""
            import traceback as _tb
            del action
            if task.iteration % (cam.skips + 1) != 0:
                return
            try:
                if cam.viewer == 'dm_control':
                    cam.data[cam.sample, :, :, :] = physics.render(
                        width=cam.width,
                        height=cam.height,
                        camera_id=cam.camera,
                    )
                else:
                    now = physics.time() / task.units.seconds
                    timediff = now - cam.last_capture
                    cam.last_capture = now
                    cam.camera.azimuth += cam.angular_velocity * timediff
                    cam.camera.lookat[:] = np.array(cam.offset)
                    if cam.links is not None:
                        cam.camera.lookat[:] += np.array(
                            cam.links.global_com_position(
                                iteration=task.iteration - 1,
                            ),
                        )
                    cam.camera.distance = cam.distance
                    cam.camera.elevation = cam.elevation
                    if cam.renderer is not None:
                        cam.renderer.update_scene(
                            physics.data.ptr,
                            camera=cam.camera,
                            scene_option=cam.render_options,
                        )
                        # ── inject particles AFTER scene update ──
                        injected = _inject_particles_into_scene(
                            cam.renderer)
                        cam.renderer.render(
                            out=cam.data[cam.sample, :, :, :])

                        if _frame[0] < 5 or _frame[0] % 200 == 0:
                            print(
                                f"[PV-video] f={_frame[0]} "
                                f"n_active={pv._n_active_video} "
                                f"injected={injected} "
                                f"maxgeom="
                                f"{cam.renderer.scene.maxgeom}"
                            )
                        _frame[0] += 1

                if cam.out is not None:
                    cam.out.write(
                        cam.data[cam.sample, :, :, :][:, :, [2, 1, 0]])
                cam.sample += 1
            except Exception as exc:
                print(f"[PV-video] EXCEPTION at frame={_frame[0]} "
                      f"iter={task.iteration}: {exc}")
                _tb.print_exc()
                # Fall back to original class method so video isn't broken
                type(cam).before_step(cam, task, None, physics)

        # Replace the instance method and store reference for liveness check
        cam.before_step = _cam_before_step_with_particles
        self._patched_fn = _cam_before_step_with_particles
        self._patched_renderer_id = True
        print(f"[ParticleViewer] Replaced CameraRecording.before_step "
              f"(renderer maxgeom={cam.renderer.scene.maxgeom})")

    # ── MuJoCo colour overrides ──────────────────────────────────────

    def _apply_geom_colors(self, task: ExperimentTask, physics: Physics):
        """Override MuJoCo model geom RGBA for the floor plane and animat body.

        ``floor_color`` is applied **only** to the floor plane geom and its
        checker material/texture — other arena geoms (walls, water) are
        left untouched.

        ``body_color`` is applied to every geom that belongs to an animat.
        """
        if self.floor_color is None and self.body_color is None:
            return

        model = physics.model

        # Build set of body-indices that belong to animats
        body_names = physics.named.model.body_pos.axes.row.names
        n_animats = len(task.experiment_options.animats)
        animat_body_ids: set[int] = set()
        for ai in range(n_animats):
            prefix = f"a{ai}_"
            for bi, name in enumerate(body_names):
                if name.startswith(prefix):
                    animat_body_ids.add(bi)

        # ── Floor colour: target "tile_*" geoms by name ──────────────
        if self.floor_color is not None:
            _mj_model = model.ptr if hasattr(model, 'ptr') else model

            # The pool SDF names checker-floor tiles as "tile_{i}_{j}".
            # After dm_control compilation they get namespace-prefixed
            # (e.g. "pool/tile_0_0"), so we match on the substring "tile".
            n_floor = 0
            for gi in range(model.ngeom):
                gn = mujoco.mj_id2name(
                    _mj_model, mujoco.mjtObj.mjOBJ_GEOM, gi)
                if gn is not None and "tile" in gn.lower():
                    model.geom_matid[gi] = -1    # detach material
                    model.geom_rgba[gi] = self.floor_color
                    n_floor += 1

            print(f"[ParticleViewer] Coloured {n_floor} tile geoms "
                  f"with floor colour.")

        # ── Body colour SECOND: re-paint animat geoms after floor nuke ──
        if self.body_color is not None:
            for gi in range(model.ngeom):
                body_id = int(model.geom_bodyid[gi])
                if body_id in animat_body_ids:
                    model.geom_matid[gi] = -1    # detach material
                    model.geom_rgba[gi] = self.body_color
        # ── Disable lights ──────────────────────────────────────────
        if self.no_light:
            _mj = model.ptr if hasattr(model, 'ptr') else model
            # Disable the scene light ("light_animat" attached to animat)
            for li in range(_mj.nlight):
                _mj.light_active[li] = 0
            # Disable MuJoCo default headlight
            _mj.vis.headlight.active = 0
            _mj.vis.headlight.ambient[:] = 0
            _mj.vis.headlight.diffuse[:] = 0
            _mj.vis.headlight.specular[:] = 0
            print(f"[ParticleViewer] Disabled {_mj.nlight} scene light(s) "
                  f"+ headlight.")

        # ── Coloured light (e.g. blue excitation lamp) ───────────────
        if self.light_color is not None:
            _mj = model.ptr if hasattr(model, 'ptr') else model
            rgb = self.light_color[:3]  # ignore alpha

            # Recolour every scene light
            for li in range(_mj.nlight):
                _mj.light_active[li] = 1
                _mj.light_diffuse[li]  = rgb
                _mj.light_specular[li] = rgb
                # Dim ambient so unlit areas stay dark
                _mj.light_ambient[li]  = [c * 0.15 for c in rgb]

            # Recolour the MuJoCo default headlight as well
            _mj.vis.headlight.active   = 1
            _mj.vis.headlight.diffuse[:]  = rgb
            _mj.vis.headlight.specular[:] = [c * 0.6 for c in rgb]
            _mj.vis.headlight.ambient[:]  = [c * 0.10 for c in rgb]

            print(f"[ParticleViewer] Set {_mj.nlight} scene light(s) "
                  f"+ headlight to colour {rgb}.")

        # ── Make particle geom material emissive (for fluorescent dye) ──
        if self.emissive_particles:
            _mj = model.ptr if hasattr(model, 'ptr') else model
            # Create a bright emissive material for particles that glows
            # regardless of scene lighting — mimics fluorescent emission.
            # We add a material at runtime so particles are unaffected by
            # the coloured light.
            mat_id = mujoco.mj_name2id(
                _mj, mujoco.mjtObj.mjOBJ_MATERIAL, "particle_emissive")
            if mat_id >= 0:
                _mj.mat_emission[mat_id] = 0.95
                _mj.mat_rgba[mat_id] = self.particle_rgba
                self._emissive_mat_id = mat_id
                print(f"[ParticleViewer] Using pre-defined emissive material "
                      f"(id={mat_id}).")
            else:
                self._emissive_mat_id = None
                print(f"[ParticleViewer] No 'particle_emissive' material found; "
                      f"particles use bright RGBA only.")

        print(f"[ParticleViewer] Applied colour overrides "
              f"(floor={self.floor_color is not None}, "
              f"body={self.body_color is not None}, "
              f"light={self.light_color is not None}).")

    # ── Seeding ──────────────────────────────────────────────────────

    def _get_tail_xy(self, task, physics, fs):
        """Return (prev_xy, tail_xy): world positions of the penultimate
        and last links of the first animat.

        The vector ``tail_xy - prev_xy`` gives the local tail *direction*.
        This lets us project boundary cells along that direction to find
        the distal tip reliably even when the body curves.

        Returns (prev_xy, tail_xy) or (None, None).
        """
        try:
            comp = fs.composite_body
            if len(comp.body_ids) < 2:
                return None, None
            # penultimate link and last link
            a_prev, l_prev = comp.body_ids[-2]
            a_tail, l_tail = comp.body_ids[-1]
            idx_prev = int(task.maps[a_prev]["sensors"]["data2xfrc"][l_prev])
            idx_tail = int(task.maps[a_tail]["sensors"]["data2xfrc"][l_tail])
            prev_pos = physics.data.xpos[idx_prev]
            tail_pos = physics.data.xpos[idx_tail]
            prev_xy = (float(prev_pos[0]), float(prev_pos[1]))
            tail_xy = (float(tail_pos[0]), float(tail_pos[1]))
            return prev_xy, tail_xy
        except Exception:
            return None, None

    def _get_sdf_numpy(self, fs):
        """Get the composite body SDF as a numpy array."""
        if not hasattr(fs, 'composite_body'):
            return None
        sdf_val = getattr(fs.composite_body, 'sdf_val', None)
        if sdf_val is None:
            return None
        return sdf_val.cpu().numpy()

    def _cache_tail_seeds_body_local(self, fs, handler, iteration):
        """At the first valid iteration, find the tail-tip seed points
        from the **composite (union) SDF** using the min-x boundary
        cells, then transform them into the body-local frame of the
        last link.  Subsequent seeding roto-translates these fixed
        body-local seeds to the current world frame each step.
        """
        from scipy.spatial.transform import Rotation as ScipyRotation

        # 1) Get composite SDF boundary (all links merged)
        sdf = self._get_sdf_numpy(fs)
        if sdf is None:
            print("[ParticleViewer] Cache deferred: SDF is None")
            return
        x = fs.x.cpu().numpy()
        y = fs.y.cpu().numpy()
        z = fs.z.cpu().numpy() if self._ndim == 3 else None

        pts = seed_tail_from_sdf(
            sdf, x, y,
            n_particles=self.seed_n_particles,
            z=z, n_z_layers=self.n_z_layers,
        )
        if pts.shape[1] == 0:
            print("[ParticleViewer] Cache deferred: no seed points from SDF")
            return

        # 2) Get last link's pose from FARMS sensors
        comp = fs.composite_body
        if len(comp.body_ids) < 1:
            return
        a_tail, l_tail = comp.body_ids[-1]

        sen = handler.data[a_tail].sensors.links
        all_pos = np.array(sen.urdf_positions()[iteration, :],
                           dtype=np.float64)
        all_quat = np.array(sen.urdf_orientations()[iteration, :],
                            dtype=np.float64)
        tail_pos = all_pos[l_tail]
        quat = all_quat[l_tail]
        qnorm = np.linalg.norm(quat)

        if qnorm < 1e-12:
            print(f"[ParticleViewer] Cache deferred: zero quat at "
                  f"sensor iter={iteration}")
            return
        R_np = ScipyRotation.from_quat(quat).as_matrix()

        # 3) Transform world seeds → body-local frame of last link
        #    local = R^T @ (world - urdf_pos)
        if self._ndim == 3:
            local_pts = R_np.T @ (pts - tail_pos[:3, None])
        else:
            R2 = R_np[:2, :2]
            local_pts = R2.T @ (pts - tail_pos[:2, None])

        self._tail_seeds_local = local_pts
        self._tail_link_id = (a_tail, l_tail)
        print(f"[ParticleViewer] ✓ Cached {local_pts.shape[1]} tail-tip "
              f"seeds (min-x from union SDF at iter={iteration}) in "
              f"body-local frame of link ({a_tail}, {l_tail}).")

    def _seed_particles(self, fs, handler, iteration) -> np.ndarray | None:
        """Transform the cached body-local tail-tip seeds to the current
        world frame using the last link's roto-translation.

        Returns (ndim, N) numpy array, or None if the cache is not yet
        ready (first 1–2 steps while quaternion is zero).
        """
        from scipy.spatial.transform import Rotation as ScipyRotation

        if self._tail_seeds_local is None:
            return None   # cache not ready yet — skip seeding this step

        animat_id, link_id = self._tail_link_id
        sen = handler.data[animat_id].sensors.links
        # BDIMhandler pattern: fetch all links, then index
        all_pos = np.array(sen.urdf_positions()[iteration, :],
                           dtype=np.float64)
        all_quat = np.array(sen.urdf_orientations()[iteration, :],
                            dtype=np.float64)
        urdf_pos_np = all_pos[link_id]
        quat = all_quat[link_id]
        qnorm = np.linalg.norm(quat)
        if qnorm < 1e-12:
            return None   # quaternion still zero — skip
        R_np = ScipyRotation.from_quat(quat).as_matrix()  # (3, 3)

        if self._ndim == 3:
            world_pts = R_np @ self._tail_seeds_local + urdf_pos_np[:3, None]
        else:
            R2 = R_np[:2, :2]
            world_pts = R2 @ self._tail_seeds_local + urdf_pos_np[:2, None]

        return world_pts

    # ── Initialise interpolators from solver ──────────────────────────

    def _init_from_solver(self, fs):
        """Initialise interpolators and particle buffer from the fluid solver."""
        self._device = fs.device
        self._dtype = fs.dtype
        self._ndim = 3 if fs.z is not None else 2

        from pytorch_interpolation import (
            RegularGridInterpolator,
            RegularGridInterpolator3D,
        )

        if self._ndim == 3:
            coords = (fs.x, fs.y, fs.z)
            shape = fs.grid_shape
            Interp = RegularGridInterpolator3D
        else:
            coords = (fs.x, fs.y)
            shape = (len(fs.x), len(fs.y))
            Interp = RegularGridInterpolator

        self._interps = [
            Interp(
                tuple(coords),
                torch.zeros(shape, device=self._device, dtype=self._dtype).contiguous(),
            )
            for _ in range(self._ndim)
        ]

        # Domain bounds (interior, excluding ghost cells)
        self._bounds = []
        for ax in [fs.x, fs.y, fs.z][:self._ndim]:
            self._bounds.append((float(ax[1]), float(ax[-2])))

        # Empty particle buffer
        self._particles = torch.zeros(
            (self._ndim, 0), device=self._device, dtype=self._dtype
        )

    # ── Main step ─────────────────────────────────────────────────────

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        if self._fluid_ext is None:
            self._iteration += 1
            return

        # Ensure CameraRecording's renderer is patched
        # (deferred: renderer may not exist during initialize_episode,
        #  and a new one may appear on episode restart)
        self._ensure_video_patch()

        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        if handler is None:
            return

        fs = getattr(handler, "fluid_solver", None)
        if fs is None:
            return

        iteration = getattr(handler, "iteration", self._iteration)
        # BDIMhandler.step() increments self.iteration at the end,
        # so by the time we run, handler.iteration = N+1
        # but MuJoCo sensors are only populated up to index N.
        # Use N (= iteration - 1) for sensor reads.
        sensor_iteration = max(iteration - 1, 0)
        self._iteration = iteration

        # Lazy initialisation on first step (solver must be ready)
        if self._interps is None:
            self._init_from_solver(fs)
            self._grid_h = float(fs.h)
            # Default body_cull_margin: -3h (buffer around body surface)
            if self.body_cull_margin is None:
                self.body_cull_margin = -3.0 * self._grid_h
            print(f"[ParticleViewer] Initialised: {self._ndim}D, "
                  f"max_particles={self.max_particles}, "
                  f"body_cull_margin={self.body_cull_margin:.4f}, "
                  f"cull_every={self.cull_every}")

        # Update interval
        every = self.update_every or 1
        if iteration % every != 0:
            return

        dt = float(fs.dt)

        # ── Update interpolator fields with current velocities ────
        vel_names = ["u0", "v0", "w0"][:self._ndim]
        for i, name in enumerate(vel_names):
            field = getattr(fs, name, None)
            if field is not None:
                self._interps[i].F = field.contiguous()

        # ── Cache tail seeds on first seeding opportunity ─────
        if self._tail_seeds_local is None:
            self._cache_tail_seeds_body_local(fs, handler, sensor_iteration)

        # ── Seed new particles ────────────────────────────────────
        if iteration % self.seed_interval == 0:
            new_pts = self._seed_particles(fs, handler, sensor_iteration)
            if new_pts is not None:
                new_t = torch.as_tensor(
                    new_pts, device=self._device, dtype=self._dtype
                )
                self._particles = torch.cat(
                    [self._particles, new_t], dim=1
                )

        # ── Trim trail ────────────────────────────────────────────
        if self.trail_length > 0:
            batch_size = self.seed_n_particles * (
                self.n_z_layers if self._ndim == 3 else 1
            )
            max_pts = self.trail_length * batch_size
            if self._particles.shape[1] > max_pts:
                self._particles = self._particles[:, -max_pts:]
        else:
            # No explicit trail → cap buffer to avoid unbounded growth
            buf_cap = self.max_particles * 50
            if self._particles.shape[1] > buf_cap:
                self._particles = self._particles[:, -buf_cap:]

        # ── Advect ────────────────────────────────────────────────
        if self._particles.shape[1] > 0:
            self._particles = _advect_particles(
                self._particles, self._interps, dt,
                turb_diff=self.turb_diffusivity,
            )

            # Cull (only every cull_every steps to avoid over-culling
            # near the moving body surface)
            if iteration % self.cull_every == 0:
                self._particles = _cull_out_of_domain(
                    self._particles, self._bounds
                )

                # Cull deep inside body (margin < 0 gives a buffer zone)
                sdf_val = getattr(fs.composite_body, 'sdf_val', None)
                if sdf_val is not None and self._particles.shape[1] > 0:
                    self._particles = _cull_inside_body(
                        self._particles, sdf_val, fs.x, fs.y,
                        margin=self.body_cull_margin,
                    )

        # ── Update sphere visuals ─────────────────────────────────
        if iteration % 200 == 0:
            print(f"[ParticleViewer] step={iteration} "
                  f"particles={self._particles.shape[1]} "
                  f"display={min(self._particles.shape[1], self.max_particles)}")
        self._update_spheres()

    def _update_spheres(self):
        """Copy particle positions into the shared position/colour buffers
        and update the MuJoCo viewer geoms.

        All particles are kept in ``self._particles`` for advection.
        If more particles exist than ``max_particles`` sphere slots,
        we uniformly sub-sample across the full buffer so that both
        old wake particles and fresh seeds are represented.
        """
        n_pts = self._particles.shape[1]
        if n_pts == 0:
            self._hide_all()
            return

        n_show = min(n_pts, self.max_particles)

        # Sub-sample with fixed stride when more particles than display slots.
        # A stride-based approach is more visually stable than linspace because
        # it always picks the same particles (0, stride, 2*stride, …) and only
        # appends new ones — no frame-to-frame jitter.
        if n_pts > self.max_particles:
            stride = max(1, n_pts // self.max_particles)
            pts = self._particles[:, ::stride].cpu().numpy()  # (ndim, ~n_show)
            if pts.shape[1] > self.max_particles:
                pts = pts[:, :self.max_particles]
            n_show = pts.shape[1]
        else:
            pts = self._particles.cpu().numpy()  # (ndim, n_show)
        if self._ndim == 2:
            # Place 2-D particles slightly above z=0 so they're visible
            self._active_positions[:n_show, 0] = pts[0]
            self._active_positions[:n_show, 1] = pts[1]
            self._active_positions[:n_show, 2] = 0.005
        else:
            self._active_positions[:n_show, :] = pts.T

        # Colours are pre-filled in initialize_episode; just update count
        self._n_active = n_show
        # Video count: unclamped by GUI scene capacity
        self._n_active_video = min(n_show, self._max_particles_video)

        # Update interactive viewer geoms
        if self._viewer is not None and self._geom_start is not None:
            scn = self._viewer.user_scn

            # Robustness: if ngeom was reset by outside code
            if scn.ngeom < self._geom_start:
                self._geom_start = scn.ngeom

            _eye3 = np.eye(3, dtype=np.float64).ravel()
            _sz = np.array([self.sphere_size, 0, 0], dtype=np.float64)

            count = 0
            for i in range(n_show):
                target_idx = self._geom_start + i
                if target_idx >= scn.maxgeom:
                    break

                if target_idx >= scn.ngeom:
                    scn.ngeom = target_idx + 1

                g = scn.geoms[target_idx]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    _sz,
                    self._active_positions[i],
                    _eye3,
                    self._active_colors[i],
                )
                if self.emissive_particles:
                    g.emission = 1.0
                g.category = mujoco.mjtCatBit.mjCAT_ALL
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                count += 1

            # Hide previously-used but now-unused slots
            prev_max = self._geom_start + self._n_active
            curr_max = self._geom_start + count
            if prev_max > curr_max and prev_max <= scn.ngeom:
                for i in range(curr_max, prev_max):
                    scn.geoms[i].rgba[3] = 0

            self._n_active = count

    def _hide_all(self):
        """Make all reserved spheres transparent."""
        if self._viewer is not None and self._geom_start is not None:
            scn = self._viewer.user_scn
            for i in range(self._n_active):
                idx = self._geom_start + i
                if idx < scn.ngeom:
                    scn.geoms[idx].rgba[3] = 0
        self._n_active = 0
        self._n_active_video = 0
