"""Restore animat body colours in the MuJoCo viewer.

FARMS exports both visual and collision geoms. When both stay visible,
the collision geoms can wash out the intended SDF colours in the viewer.
This extension keeps native visual colours where available, hides
duplicate collision geoms for links that already have visuals, and
recolours collision-only links from the SDF material colour or a stable
fallback palette.
"""

from __future__ import annotations

import numpy as np

from dm_control.mjcf.physics import Physics

from farms_core.experiment.options import ExperimentOptions
from farms_core.io.sdf import ModelSDF
from farms_core.simulation.extensions import TaskExtension
from farms_mujoco.simulation.mjcf import get_prefix
from farms_mujoco.simulation.task import ExperimentTask


_FALLBACK_BODY_PALETTE = (
    np.array([0.20, 0.53, 0.31, 1.0], dtype=np.float32),
    np.array([0.16, 0.44, 0.62, 1.0], dtype=np.float32),
    np.array([0.61, 0.34, 0.27, 1.0], dtype=np.float32),
    np.array([0.55, 0.45, 0.18, 1.0], dtype=np.float32),
)


def _as_rgba(color, alpha=None):
    if color is None:
        return None
    rgba = np.asarray(color, dtype=np.float32).reshape(-1)
    if rgba.size == 3:
        rgba = np.concatenate((rgba, np.array([1.0], dtype=np.float32)))
    if rgba.size < 4:
        return None
    rgba = rgba[:4].copy()
    if alpha is not None:
        rgba[3] = float(alpha)
    return rgba


class NativeBodyColors(TaskExtension):
    """Ensure animat bodies keep native-looking colours in MuJoCo."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        alpha: float | None = None,
        hide_collision_geoms_with_visuals: bool = True,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.alpha = alpha
        self.hide_collision_geoms_with_visuals = bool(
            hide_collision_geoms_with_visuals,
        )

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            alpha=config.get("alpha", None),
            hide_collision_geoms_with_visuals=config.get(
                "hide_collision_geoms_with_visuals",
                True,
            ),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        model = physics.model
        body_names = tuple(physics.named.model.body_pos.axes.row.names)
        body_name_to_id = {name: idx for idx, name in enumerate(body_names)}
        visual_rgba_by_body = self._collect_visual_rgba_by_body(model)
        body_rgba = self._collect_body_rgba(body_name_to_id, visual_rgba_by_body)
        if not body_rgba:
            return

        bodies_with_visuals = set(visual_rgba_by_body)
        hidden_collisions = 0
        recolored_geoms = 0

        for geom_i in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_i])
            rgba = body_rgba.get(body_id)
            if rgba is None:
                continue

            geom_group = int(model.geom_group[geom_i])
            if (
                self.hide_collision_geoms_with_visuals
                and geom_group == 2
                and body_id in bodies_with_visuals
            ):
                model.geom_matid[geom_i] = -1
                model.geom_rgba[geom_i] = np.array(
                    [0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                )
                hidden_collisions += 1
                continue

            if geom_group == 2 or body_id not in bodies_with_visuals:
                model.geom_matid[geom_i] = -1
                model.geom_rgba[geom_i] = rgba
                recolored_geoms += 1

        print(
            "[NativeBodyColors] Applied viewer colours "
            f"to {len(body_rgba)} bodies, recoloured {recolored_geoms} geoms, "
            f"hid {hidden_collisions} duplicate collision geoms.",
        )

    def _collect_visual_rgba_by_body(self, model):
        visual_rgba_by_body = {}
        for geom_i in range(model.ngeom):
            if int(model.geom_group[geom_i]) != 1:
                continue

            body_id = int(model.geom_bodyid[geom_i])
            rgba = None
            mat_id = int(model.geom_matid[geom_i])
            if 0 <= mat_id < model.nmat:
                rgba = _as_rgba(model.mat_rgba[mat_id], self.alpha)
            if rgba is None or float(rgba[3]) <= 0.0:
                rgba = _as_rgba(model.geom_rgba[geom_i], self.alpha)
            if rgba is None or float(rgba[3]) <= 0.0:
                continue
            visual_rgba_by_body.setdefault(body_id, rgba)
        return visual_rgba_by_body

    def _collect_body_rgba(self, body_name_to_id, visual_rgba_by_body):
        body_rgba = {}
        for animat_i, animat in enumerate(self.experiment_options.animats):
            sdf = ModelSDF.read(animat.sdf)[0]
            prefix = get_prefix(animat_i)
            for link_i, link in enumerate(sdf.links):
                body_id = body_name_to_id.get(f"{prefix}{link.name}")
                if body_id is None:
                    continue

                rgba = None
                for visual in getattr(link, "visuals", ()):
                    rgba = _as_rgba(getattr(visual, "color", None), self.alpha)
                    if rgba is not None:
                        break

                if rgba is None:
                    rgba = visual_rgba_by_body.get(body_id)
                if rgba is None:
                    rgba = _FALLBACK_BODY_PALETTE[
                        (animat_i + link_i) % len(_FALLBACK_BODY_PALETTE)
                    ].copy()
                    if self.alpha is not None:
                        rgba[3] = float(self.alpha)

                body_rgba[body_id] = rgba

        return body_rgba