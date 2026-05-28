"""
AccelerationViewer – draw the linear (and, optionally, angular) acceleration
of each immersed body as an arrow inside the MuJoCo viewer **and** in the
recorded video.

One arrow is drawn per FARMS link/body, anchored at the body centre of mass
(``data.xipos``) and pointing along the body's world-frame linear
acceleration.  When ``show_angular`` is enabled, a second arrow per body is
drawn along the angular-acceleration axis (right-hand rule), length
proportional to |α|.

Accelerations come from ``mujoco.mj_objectAcceleration`` (classical spatial
acceleration of the point on the body coincident with the body CoM, world
frame).  In ``before_step`` this reflects the *previous* timestep's
``qacc`` — i.e. the acceleration that produced the current state — which is
the most physically meaningful quantity to overlay on the current frame.

Appearance knobs:

- ``acc_scale``    – linear-arrow **length** per unit |a| (metres per m/s²).
- ``acc_width``    – the **circular (shaft) radius** of the linear arrow (m).
- ``show_angular`` – also draw an angular-acceleration arrow per body
                     (default ``False``).
- ``ang_scale``    – angular-arrow length per unit |α| (m per rad/s²);
                     defaults to ``acc_scale``.
- ``ang_width``    – shaft radius of the angular arrow (m); defaults to
                     ``acc_width``.

Arrows are injected into both:
- the interactive viewer's ``user_scn`` (visible in the MuJoCo GUI), and
- the ``CameraRecording`` extension's offscreen renderer (visible in the
  saved MP4 video).

Usage
-----
Add to the simulation extensions list in your gen_configs file (order does
not matter – this extension reads MuJoCo state directly and has no
dependency on the FluidExtension)::

    extensions.append({
        "loader": "lilytorch.integration.acceleration_viewer.AccelerationViewer",
        "config": {
            "acc_scale"   : 0.01,   # metres of arrow per (m/s²)
            "acc_width"   : 0.003,  # shaft (circular) radius in metres
            "color"       : "#FF00AA",
            "max_length"  : 0.3,    # clamp arrow length (m); null = no clamp
            "min_acc"     : 0.0,    # hide arrows below this |a| (m/s²)
            "show_angular": True,   # also draw angular-acceleration arrows
            "ang_scale"   : 0.005,  # m per rad/s² (default: acc_scale)
            "ang_width"   : 0.002,  # shaft radius (default: acc_width)
            "ang_color"   : "#AA00FF",
            "min_ang"     : 0.0,    # hide α arrows below this (rad/s²)
            "update_every": null,   # null -> same cadence as solver.save_every
        },
    })

Works with ``headless: false`` (viewer + video) and ``headless: true``
(video only via CameraRecording — no interactive viewer needed).
"""

import numpy as np
import mujoco

from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics


_ARROW = int(mujoco.mjtGeom.mjGEOM_ARROW)
_OBJ_BODY = int(mujoco.mjtObj.mjOBJ_BODY)


def _parse_color(color, default=(1.0, 0.0, 0.67, 0.9)):
    """Return an RGBA float32 array from a hex string or a 3/4-length list."""
    if color is None:
        return np.array(default, dtype=np.float32)
    if isinstance(color, str):
        h = color.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        a = int(h[6:8], 16) / 255.0 if len(h) >= 8 else default[3]
        return np.array([*rgb, a], dtype=np.float32)
    arr = np.asarray(color, dtype=np.float32)
    if arr.size == 3:
        arr = np.array([*arr, default[3]], dtype=np.float32)
    return arr


class AccelerationViewer(TaskExtension):
    """Render the linear (and optionally angular) acceleration of each body."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        acc_scale: float = 0.01,
        acc_width: float = 0.003,
        color="#FF00AA",
        max_length: float | None = None,
        min_acc: float = 0.0,
        show_angular: bool = False,
        ang_scale: float | None = None,
        ang_width: float | None = None,
        ang_color="#AA00FF",
        min_ang: float = 0.0,
        update_every: int | None = None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.acc_scale = float(acc_scale)
        self.acc_width = float(acc_width)
        self.color = _parse_color(color)
        self.max_length = None if max_length is None else float(max_length)
        self.min_acc = float(min_acc)

        self.show_angular = bool(show_angular)
        self.ang_scale = (self.acc_scale if ang_scale is None
                          else float(ang_scale))
        self.ang_width = (self.acc_width if ang_width is None
                          else float(ang_width))
        self.ang_color = _parse_color(ang_color,
                                      default=(0.67, 0.0, 1.0, 0.9))
        self.min_ang = float(min_ang)

        self.update_every = update_every

        self._viewer = None
        self._fluid_ext = None          # optional; used only for save_every cadence
        self._geom_start = None         # first slot we own in user_scn
        self._n_bodies = 0              # number of bodies
        self._max_arrows = 0            # geom budget (bodies × arrows_per_body)
        self._arrows_per_body = 2 if self.show_angular else 1
        self._n_active = 0              # arrows currently visible
        self._iteration = 0
        self._slots_reserved = False
        self._patched_renderers = set()   # id() of already-wrapped renderers

        # Per-body MuJoCo IDs and arrow data (interactive + offscreen).
        self._body_mj_ids = None        # (n_bodies,) int — physics.data.xipos row
        self._sa = np.zeros(6, dtype=np.float64)   # scratch 6D spatial acceleration
        self._from = None               # (max_arrows, 3) float64 — arrow base
        self._to = None                 # (max_arrows, 3) float64 — arrow tip
        self._rgba = None               # (max_arrows, 4) float32 — per arrow
        self._width = None              # (max_arrows,)  float64 — shaft radius

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            acc_scale=config.get("acc_scale", 0.01),
            acc_width=config.get("acc_width", 0.003),
            color=config.get("color", "#FF00AA"),
            max_length=config.get("max_length", None),
            min_acc=config.get("min_acc", 0.0),
            show_angular=config.get("show_angular", False),
            ang_scale=config.get("ang_scale", None),
            ang_width=config.get("ang_width", None),
            ang_color=config.get("ang_color", "#AA00FF"),
            min_ang=config.get("min_ang", 0.0),
            update_every=config.get("update_every", None),
        )

    # ── lifecycle hooks ──────────────────────────────────────────────

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._viewer = task.viewer

        # FluidExtension is optional – we only consult it to inherit the
        # solver's ``save_every`` cadence when ``update_every`` is null.
        try:
            from lilytorch.integration.extensions import FluidExtension
            for ext in task.extensions:
                if isinstance(ext, FluidExtension):
                    self._fluid_ext = ext
                    break
        except Exception:  # noqa: BLE001
            self._fluid_ext = None

    def _resolve_body_ids(self, task: ExperimentTask, physics: Physics):
        """Build the list of MuJoCo body indices to visualise.

        Prefers the FluidExtension/CompositeBody body list (so we render
        exactly the immersed links the solver knows about); falls back to
        every body in animat 0 when no FluidExtension is present.
        """
        ids = []

        handler = None
        if self._fluid_ext is not None:
            handler = getattr(self._fluid_ext, "BDIMhandler", None)
        fs = getattr(handler, "fluid_solver", None) if handler is not None else None
        comp = getattr(fs, "composite_body", None) if fs is not None else None

        if comp is not None:
            for body_i in range(len(comp.bodies)):
                animat_id, link_id = comp.body_ids[body_i]
                ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]
                ids.append(int(ind))
            return ids

        # Fallback: every body under animat 0's data2xfrc map.
        try:
            ids = [int(i) for i in task.maps[0]["sensors"]["data2xfrc"]]
        except Exception:  # noqa: BLE001
            # Last resort: every body except worldbody.
            ids = list(range(1, physics.model.nbody))
        return ids

    def _reserve_slots(self, n_bodies: int):
        """Allocate arrow storage and pre-create user_scn slots."""
        self._n_bodies = n_bodies
        self._max_arrows = n_bodies * self._arrows_per_body
        self._from = np.zeros((self._max_arrows, 3), dtype=np.float64)
        self._to = np.zeros((self._max_arrows, 3), dtype=np.float64)
        self._rgba = np.zeros((self._max_arrows, 4), dtype=np.float32)
        self._width = np.full(self._max_arrows, self.acc_width, dtype=np.float64)

        if self._viewer is not None:
            scn = self._viewer.user_scn
            self._geom_start = scn.ngeom
            _transparent = np.zeros(4, dtype=np.float32)
            _zero = np.zeros(3, dtype=np.float64)
            _eye3 = np.eye(3, dtype=np.float64).ravel()
            reserved = 0
            for _ in range(self._max_arrows):
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    _ARROW, _zero, _zero, _eye3, _transparent,
                )
                scn.ngeom += 1
                reserved += 1
            if reserved < self._max_arrows:
                print(f"[AccelerationViewer] WARNING: user_scn.maxgeom={scn.maxgeom} "
                      f"reached – only {reserved}/{self._max_arrows} arrow "
                      f"slots reserved.")
                self._max_arrows = reserved
            print(f"[AccelerationViewer] Reserved {self._max_arrows} arrow slots "
                  f"(geom {self._geom_start}..{scn.ngeom - 1}); "
                  f"{self._arrows_per_body} per body.")
        else:
            print("[AccelerationViewer] No viewer (headless) – "
                  "arrows will only appear in recorded video.")

        self._slots_reserved = True

    def _patch_camera_renderers(self, task: ExperimentTask):
        """Inject arrows into every CameraRecording offscreen renderer so they
        appear in the saved videos / PNG frames (same trick as ForceViewer).
        """
        try:
            from farms_mujoco.sensors.camera import CameraRecording
        except Exception:  # noqa: BLE001
            return

        for ext in task.extensions:
            if not isinstance(ext, CameraRecording):
                continue
            renderer = getattr(ext, "renderer", None)
            if renderer is None or id(renderer) in self._patched_renderers:
                continue
            self._patch_one_renderer(renderer)
            self._patched_renderers.add(id(renderer))
            print(f"[AccelerationViewer] Patched {type(ext).__name__} renderer "
                  f"({getattr(getattr(ext, 'video', None), 'path', '?')}) "
                  f"for video output.")

    def _patch_one_renderer(self, renderer):
        """Wrap a single ``mujoco.Renderer.render`` to append our arrow geoms."""
        viewer_self = self
        original_render = renderer.render
        _eye3 = np.eye(3, dtype=np.float64).ravel()
        _zero = np.zeros(3, dtype=np.float64)

        def _render_with_arrows(out=None):
            n = viewer_self._n_active
            if n > 0:
                scn = renderer.scene
                for i in range(n):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        g, _ARROW, _zero, viewer_self._from[i], _eye3,
                        viewer_self._rgba[i],
                    )
                    mujoco.mjv_connector(
                        g, _ARROW, float(viewer_self._width[i]),
                        viewer_self._from[i], viewer_self._to[i],
                    )
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                    scn.ngeom += 1
            return original_render(out=out)

        renderer.render = _render_with_arrows

    # ── per-step update ──────────────────────────────────────────────

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        # Deferred CameraRecording patch (renderers are created in each
        # recorder's initialize_episode, so they exist by the first step).
        self._patch_camera_renderers(task)

        if self._body_mj_ids is None:
            self._body_mj_ids = self._resolve_body_ids(task, physics)
        if not self._slots_reserved:
            self._reserve_slots(len(self._body_mj_ids))

        # Cadence: explicit update_every wins; otherwise inherit the solver's
        # save_every (or fall back to 1 if there is no FluidExtension).
        if self.update_every is not None:
            every = self.update_every
        else:
            handler = (getattr(self._fluid_ext, "BDIMhandler", None)
                       if self._fluid_ext is not None else None)
            fs = getattr(handler, "fluid_solver", None) if handler else None
            every = getattr(fs, "save_every", 1) if fs is not None else 1

        self._iteration += 1
        if every > 1 and self._iteration % every != 0:
            return

        # Read per-body classical spatial acceleration from MuJoCo (world
        # frame, at the body CoM). Reflects the previous step's qacc.
        model_ptr = physics.model.ptr
        data_ptr = physics.data.ptr
        xipos = physics.data.xipos
        sa = self._sa

        n_active = 0
        for body_i in range(min(self._n_bodies, len(self._body_mj_ids))):
            ind = self._body_mj_ids[body_i]
            mujoco.mj_objectAcceleration(
                model_ptr, data_ptr, _OBJ_BODY, ind, sa, 0,  # 0 = world frame
            )
            ang = sa[:3]
            lin = sa[3:]
            base = np.asarray(xipos[ind], dtype=np.float64).copy()

            n_active = self._push_arrow(
                n_active, base, lin,
                self.acc_scale, self.min_acc, self.color, self.acc_width,
            )
            if self.show_angular:
                n_active = self._push_arrow(
                    n_active, base, ang,
                    self.ang_scale, self.min_ang,
                    self.ang_color, self.ang_width,
                )

        self._update_viewer_arrows(n_active)
        self._n_active = n_active

    def _push_arrow(self, n_active, base, vec, scale, min_mag, rgba, width):
        """Append one arrow (base → base + scaled vec) to the buffers.

        Returns the updated active-arrow count. Arrows below ``min_mag`` (or
        exceeding the reserved geom budget) are skipped.
        """
        if n_active >= self._max_arrows:
            return n_active
        mag = float(np.linalg.norm(vec))
        if mag <= min_mag or mag < 1e-12:
            return n_active
        length = scale * mag
        if self.max_length is not None:
            length = min(length, self.max_length)
        self._from[n_active] = base
        self._to[n_active] = base + (np.asarray(vec, dtype=np.float64) / mag) * length
        self._rgba[n_active] = rgba
        self._width[n_active] = width
        return n_active + 1

    def _update_viewer_arrows(self, n_active: int):
        if self._viewer is None or self._geom_start is None:
            return
        scn = self._viewer.user_scn

        # Re-anchor if user_scn was reset by outside code.
        if scn.ngeom < self._geom_start:
            self._geom_start = scn.ngeom

        _eye3 = np.eye(3, dtype=np.float64).ravel()
        for i in range(n_active):
            target = self._geom_start + i
            if target >= scn.maxgeom:
                break
            if target >= scn.ngeom:
                scn.ngeom = target + 1
            g = scn.geoms[target]
            mujoco.mjv_initGeom(g, _ARROW, np.zeros(3), self._from[i],
                                _eye3, self._rgba[i])
            mujoco.mjv_connector(g, _ARROW, float(self._width[i]),
                                 self._from[i], self._to[i])
            g.category = mujoco.mjtCatBit.mjCAT_ALL
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1

        # Hide previously-used slots that are now idle.
        for i in range(n_active, self._n_active):
            target = self._geom_start + i
            if target < scn.ngeom:
                scn.geoms[target].rgba[3] = 0
