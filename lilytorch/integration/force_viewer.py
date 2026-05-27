"""
ForceViewer – draw the hydrodynamic force (and, optionally, torque) acting on
each immersed body as an arrow inside the MuJoCo viewer **and** in the
recorded video.

One arrow is drawn per FARMS link/body, anchored at the body centre of mass
and pointing along the total fluid force (viscous + pressure).  When
``show_torque`` is enabled, a second arrow per body is drawn along the net
hydrodynamic torque axis (right-hand rule), length proportional to |torque|.

Appearance knobs:

- ``force_scale``  – force-arrow **length** per unit force (metres per Newton).
- ``force_width``  – the **circular (shaft) radius** of the force arrow (m).
- ``show_torque``  – also draw a torque arrow per body (default ``False``).
- ``torque_scale`` – torque-arrow length per unit torque (m per N·m); defaults
                     to ``force_scale``.
- ``torque_width`` – shaft radius of the torque arrow (m); defaults to
                     ``force_width``.

Arrows are injected into both:
- the interactive viewer's ``user_scn`` (visible in the MuJoCo GUI), and
- the ``CameraRecording`` extension's offscreen renderer (visible in the
  saved MP4 video).

Usage
-----
Add to the simulation extensions list in your gen_configs file, **after**
the FluidExtension entry (so ``xfrc_applied`` / the cached force tensors are
already fresh when this extension reads them)::

    extensions.append({
        "loader": "lilytorch.integration.force_viewer.ForceViewer",
        "config": {
            "force_scale" : 0.05,    # metres of arrow per Newton
            "force_width" : 0.003,   # shaft (circular) radius in metres
            "color"       : "#FF4500",
            "force_source": "hydro", # "hydro" (viscous+pressure, no buoyancy)
                                     # or "applied" (raw xfrc, incl. buoyancy)
            "max_length"  : 0.3,     # clamp arrow length (m); null = no clamp
            "min_force"   : 0.0,     # hide arrows below this magnitude (N)
            "show_torque" : True,    # also draw torque arrows
            "torque_scale": 0.5,     # m per N·m (default: force_scale)
            "torque_width": 0.002,   # shaft radius (default: force_width)
            "torque_color": "#00B0FF",
            "min_torque"  : 0.0,     # hide torque arrows below this (N·m)
            "update_every": null,    # null -> same cadence as solver.save_every
        },
    })

Works with ``headless: false`` (viewer + video) and ``headless: true``
(video only via CameraRecording — no interactive viewer needed).
"""

import numpy as np
import mujoco
import torch

from farms_core.simulation.extensions import TaskExtension
from farms_core.experiment.options import ExperimentOptions
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics


_ARROW = int(mujoco.mjtGeom.mjGEOM_ARROW)


def _parse_color(color, default=(0.95, 0.45, 0.1, 0.9)):
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


class ForceViewer(TaskExtension):
    """Render the fluid force (and optionally torque) on each body as arrows."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        force_scale: float = 0.05,
        force_width: float = 0.003,
        color="#FF4500",
        force_source: str = "hydro",
        max_length: float | None = None,
        min_force: float = 0.0,
        show_torque: bool = False,
        torque_scale: float | None = None,
        torque_width: float | None = None,
        torque_color="#00B0FF",
        min_torque: float = 0.0,
        update_every: int | None = None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.force_scale = float(force_scale)
        self.force_width = float(force_width)
        self.color = _parse_color(color)
        self.force_source = str(force_source).lower()
        self.max_length = None if max_length is None else float(max_length)
        self.min_force = float(min_force)

        self.show_torque = bool(show_torque)
        self.torque_scale = (self.force_scale if torque_scale is None
                             else float(torque_scale))
        self.torque_width = (self.force_width if torque_width is None
                             else float(torque_width))
        self.torque_color = _parse_color(torque_color,
                                         default=(0.0, 0.69, 1.0, 0.9))
        self.min_torque = float(min_torque)

        self.update_every = update_every

        self._viewer = None
        self._fluid_ext = None          # reference to FluidExtension
        self._geom_start = None         # first slot we own in user_scn
        self._n_bodies = 0              # number of bodies
        self._max_arrows = 0            # geom budget (bodies × arrows_per_body)
        self._arrows_per_body = 2 if self.show_torque else 1
        self._n_active = 0              # arrows currently visible
        self._iteration = 0
        self._slots_reserved = False
        self._cam_renderer_patched = False
        self._warned_hydro = False

        # Shared arrow data (interactive viewer + offscreen renderer).
        self._from = None               # (max_arrows, 3) float64 — arrow base
        self._to = None                 # (max_arrows, 3) float64 — arrow tip
        self._rgba = None               # (max_arrows, 4) float32 — per arrow
        self._width = None              # (max_arrows,)  float64 — shaft radius

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            force_scale=config.get("force_scale", 0.05),
            force_width=config.get("force_width", 0.003),
            color=config.get("color", "#FF4500"),
            force_source=config.get("force_source", "hydro"),
            max_length=config.get("max_length", None),
            min_force=config.get("min_force", 0.0),
            show_torque=config.get("show_torque", False),
            torque_scale=config.get("torque_scale", None),
            torque_width=config.get("torque_width", None),
            torque_color=config.get("torque_color", "#00B0FF"),
            min_torque=config.get("min_torque", 0.0),
            update_every=config.get("update_every", None),
        )

    # ── lifecycle hooks ──────────────────────────────────────────────

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        self._viewer = task.viewer

        # Find the FluidExtension among sibling extensions.  The handler
        # (and hence the body count) is not guaranteed to exist yet here,
        # so geom-slot reservation is deferred to the first ``before_step``.
        from lilytorch.integration.extensions import FluidExtension
        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break
        if self._fluid_ext is None:
            print("[ForceViewer] FluidExtension not found – disabled.")

    def _handler(self):
        if self._fluid_ext is None:
            return None
        return getattr(self._fluid_ext, "BDIMhandler", None)

    def _reserve_slots(self, n_bodies: int):
        """Allocate arrow storage and pre-create user_scn slots.

        Up to ``arrows_per_body`` arrows are reserved per body (1 for force,
        +1 for torque when ``show_torque``).
        """
        self._n_bodies = n_bodies
        self._max_arrows = n_bodies * self._arrows_per_body
        self._from = np.zeros((self._max_arrows, 3), dtype=np.float64)
        self._to = np.zeros((self._max_arrows, 3), dtype=np.float64)
        self._rgba = np.zeros((self._max_arrows, 4), dtype=np.float32)
        self._width = np.full(self._max_arrows, self.force_width, dtype=np.float64)

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
                print(f"[ForceViewer] WARNING: user_scn.maxgeom={scn.maxgeom} "
                      f"reached – only {reserved}/{self._max_arrows} arrow "
                      f"slots reserved.")
                self._max_arrows = reserved
            print(f"[ForceViewer] Reserved {self._max_arrows} arrow slots "
                  f"(geom {self._geom_start}..{scn.ngeom - 1}); "
                  f"{self._arrows_per_body} per body.")
        else:
            print("[ForceViewer] No viewer (headless) – "
                  "arrows will only appear in recorded video.")

        self._slots_reserved = True

    def _patch_camera_renderer(self, task: ExperimentTask):
        """Inject arrows into CameraRecording's offscreen renderer so they
        appear in the saved video (same trick as FlowViewer)."""
        if self._cam_renderer_patched:
            return

        cam_ext = None
        for ext in task.extensions:
            if type(ext).__name__ == "CameraRecording":
                cam_ext = ext
                break
        if cam_ext is None:
            return
        renderer = getattr(cam_ext, "renderer", None)
        if renderer is None:
            return

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
                    scn.ngeom += 1
            return original_render(out=out)

        renderer.render = _render_with_arrows
        self._cam_renderer_patched = True
        print("[ForceViewer] Patched CameraRecording renderer for video output.")

    # ── per-step update ──────────────────────────────────────────────

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        if self._fluid_ext is None:
            self._iteration += 1
            return

        # Deferred CameraRecording patch (renderer may not exist at init).
        if not self._cam_renderer_patched:
            self._patch_camera_renderer(task)

        handler = self._handler()
        if handler is None:
            return
        fs = getattr(handler, "fluid_solver", None)
        if fs is None:
            return

        comp = getattr(fs, "composite_body", None)
        if comp is None:
            return

        if not self._slots_reserved:
            self._reserve_slots(len(comp.bodies))

        every = self.update_every or getattr(fs, "save_every", 200)
        iteration = getattr(handler, "iteration", self._iteration)
        self._iteration = iteration
        if iteration % every != 0:
            return

        # World-frame linear force (and angular torque) per body.
        lin, ang = self._gather_wrenches(
            task, physics, handler, fs, comp, self.show_torque,
        )
        if lin is None:
            return

        n_active = 0
        for body_i in range(min(self._n_bodies, len(comp.bodies))):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]
            base = np.asarray(physics.data.xipos[ind], dtype=np.float64).copy()

            # Force arrow.
            n_active = self._push_arrow(
                n_active, base, lin[body_i],
                self.force_scale, self.min_force, self.color, self.force_width,
            )
            # Torque arrow (optional).
            if self.show_torque and ang is not None:
                n_active = self._push_arrow(
                    n_active, base, ang[body_i],
                    self.torque_scale, self.min_torque,
                    self.torque_color, self.torque_width,
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

    def _gather_wrenches(self, task, physics, handler, fs, comp, want_torque):
        """Return ``(lin, ang)`` world-frame arrays, each ``(n_bodies, 3)``.

        ``ang`` is ``None`` when ``want_torque`` is False. Returns
        ``(None, None)`` on failure.
        """
        n = len(comp.bodies)

        if self.force_source == "applied":
            lin = np.zeros((n, 3), dtype=np.float64)
            ang = np.zeros((n, 3), dtype=np.float64) if want_torque else None
            for body_i in range(n):
                (animat_id, link_id) = comp.body_ids[body_i]
                ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]
                lin[body_i] = physics.data.xfrc_applied[ind, 0:3]
                if want_torque:
                    ang[body_i] = physics.data.xfrc_applied[ind, 3:6]
            return lin, ang

        # "hydro": reconstruct viscous + pressure in world frame, exactly as
        # BDIMhandler._apply_forces does (minus buoyancy), so the arrows show
        # the pure hydrodynamic load.
        try:
            D = handler.ndim
            Nt = len(handler._ang_xfrc_idx)
            attrs = (handler._lin_visc_attrs + handler._ang_visc_attrs
                     + handler._lin_pres_attrs + handler._ang_pres_attrs)
            forces_gpu = torch.stack([getattr(fs, a) for a in attrs])
            fc = (handler.force_scaling * forces_gpu).detach().cpu().numpy()
            units_N = float(task.units.newtons)

            lin_total = fc[:D] + fc[D + Nt:2 * D + Nt]      # (D, B) fluid frame
            B = lin_total.shape[1]
            lin = np.zeros((B, 3), dtype=np.float64)
            for d, axis in enumerate(handler._lin_xfrc_idx):
                lin[:, axis] = lin_total[d] * units_N

            ang = None
            if want_torque:
                ang_total = fc[D:D + Nt] + fc[2 * D + Nt:]  # (Nt, B)
                ang = np.zeros((B, 3), dtype=np.float64)
                for d, axis in enumerate(handler._ang_xfrc_idx):
                    ang[:, axis] = ang_total[d] * units_N
            return lin, ang
        except Exception as exc:  # noqa: BLE001
            if not self._warned_hydro:
                print(f"[ForceViewer] 'hydro' wrench read failed ({exc}); "
                      f"falling back to applied xfrc.")
                self._warned_hydro = True
            self.force_source = "applied"
            return self._gather_wrenches(
                task, physics, handler, fs, comp, want_torque,
            )

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
