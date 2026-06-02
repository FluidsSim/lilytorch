"""Apply a fixed user-defined RGBA colour to all animat geoms in the MuJoCo viewer."""

from __future__ import annotations

import numpy as np

from dm_control.mjcf.physics import Physics

from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.extensions import TaskExtension
from farms_mujoco.simulation.task import ExperimentTask


def _parse_color(c) -> np.ndarray | None:
    """Accept ``[r, g, b]``/``[r, g, b, a]`` list or ``"#rrggbb[aa]"`` hex string."""
    if c is None:
        return None
    if isinstance(c, str):
        c = c.lstrip("#")
        if len(c) == 6:
            c += "ff"
        return np.array(
            [int(c[i:i + 2], 16) / 255.0 for i in range(0, 8, 2)],
            dtype=np.float32,
        )
    arr = np.asarray(c, dtype=np.float32).reshape(-1)
    if arr.size == 3:
        arr = np.concatenate((arr, np.array([1.0], dtype=np.float32)))
    return arr[:4].copy()


class BodyColorOverride(TaskExtension):
    """Paint every animat geom with a single fixed colour at episode start.

    Parameters
    ----------
    color:
        Target colour as ``[r, g, b]``, ``[r, g, b, a]`` (floats in 0–1),
        or a CSS hex string (``"#rrggbb"`` / ``"#rrggbbaa"``).
    group:
        If ``None`` (default) all geom groups are painted.
        Pass ``1`` to paint only visual geoms, ``2`` for collision geoms only.
    """

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        color: list[float] | str,
        group: int | None = None,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.rgba = _parse_color(color)
        self.group = group

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            color=config["color"],
            group=config.get("group", None),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        model = physics.model
        body_names = physics.named.model.body_pos.axes.row.names
        n_animats = len(self.experiment_options.animats)

        animat_body_ids: set[int] = set()
        for ai in range(n_animats):
            prefix = f"a{ai}_"
            for bi, name in enumerate(body_names):
                if name.startswith(prefix):
                    animat_body_ids.add(bi)

        painted = 0
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) not in animat_body_ids:
                continue
            if self.group is not None and int(model.geom_group[gi]) != self.group:
                continue
            model.geom_matid[gi] = -1
            model.geom_rgba[gi] = self.rgba
            painted += 1

        print(f"[BodyColorOverride] Painted {painted} geoms with {self.rgba}.")
