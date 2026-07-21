from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lilytorch.FARMS_V2.farms_core.farms_core.sensors.sensor_convention import sc
from lilytorch.util.paths import lilytorch_repo_root, save_path


_HEAD_VISUAL_MESH_PATH = (
    Path(lilytorch_repo_root)
    / "examples"
    / "sdfs"
    / "1guilla"
    / "meshes"
    / "link0.obj"
)
_HEAD_TIP_WINDOW_FRACTION = 5.0e-3


def resolve_saved_simulation_path(simulation_path: str, run_dir: str) -> str:
    direct_path = Path(simulation_path)
    if direct_path.exists():
        return str(direct_path)

    run_timestamp = Path(run_dir).name
    matches = sorted(Path(save_path).glob(f"**/{run_timestamp}/output/simulation.hdf5"))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(
            "Unable to locate simulation.hdf5 for run timestamp "
            f"{run_timestamp} under {save_path}"
        )
    raise FileNotFoundError(
        "Multiple simulation.hdf5 files matched run timestamp "
        f"{run_timestamp}: {matches}"
    )


def _compute_head_tip_proxy(link_array: np.ndarray, head_idx: int) -> np.ndarray:
    link_positions = link_array[:, :, sc.link_com_position_x:sc.link_com_position_z + 1]
    head_pos = link_positions[:, head_idx, :]
    if link_positions.shape[1] < 2:
        return head_pos

    if head_idx == 0:
        neighbor_idx = 1
    elif head_idx == link_positions.shape[1] - 1:
        neighbor_idx = head_idx - 1
    else:
        neighbor_idx = head_idx + 1

    neighbor_pos = link_positions[:, neighbor_idx, :]
    return head_pos + 0.5 * (head_pos - neighbor_pos)


@lru_cache(maxsize=1)
def load_1guilla_head_tip_local_point() -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with _HEAD_VISUAL_MESH_PATH.open("r", encoding="utf-8") as mesh_file:
        for line in mesh_file:
            if not line.startswith("v "):
                continue
            _, x_coord, y_coord, z_coord, *_ = line.split()
            vertices.append((float(x_coord), float(y_coord), float(z_coord)))

    if not vertices:
        raise ValueError(f"No vertices found in head mesh: {_HEAD_VISUAL_MESH_PATH}")

    verts = np.asarray(vertices, dtype=float)
    x_coords = verts[:, 0]
    x_min = float(np.min(x_coords))
    x_span = float(np.max(x_coords) - x_min)
    x_window = max(1.0e-6, _HEAD_TIP_WINDOW_FRACTION * x_span)

    # In the 1guilla model, link0's local +x points tailward, so the head tip
    # is the centroid of the front cap near x_min, not the single extreme vertex.
    front_cap = verts[x_coords <= x_min + x_window]
    return np.mean(front_cap, axis=0)


def extract_1guilla_head_tip_trajectory(
    link_array: np.ndarray,
    link_names: list[str],
    head_link_name: str = "link0",
) -> tuple[np.ndarray, str]:
    if not link_names:
        raise ValueError("No link names found in simulation.hdf5")

    if head_link_name in link_names:
        head_idx = link_names.index(head_link_name)
    else:
        head_idx = 0

    fallback_tip = _compute_head_tip_proxy(link_array, head_idx)
    if link_array.shape[-1] <= sc.link_urdf_orientation_w:
        return fallback_tip, f"{link_names[head_idx]} tip proxy"

    try:
        tip_local = load_1guilla_head_tip_local_point()
    except (FileNotFoundError, ValueError):
        return fallback_tip, f"{link_names[head_idx]} tip proxy"

    urdf_positions = link_array[:, head_idx, sc.link_urdf_position_x:sc.link_urdf_position_z + 1]
    urdf_quats = link_array[:, head_idx, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w + 1]

    tip_positions = np.full_like(urdf_positions, np.nan, dtype=float)
    quat_norm = np.linalg.norm(urdf_quats, axis=1)
    valid_pose = np.all(np.isfinite(urdf_positions), axis=1)
    valid_pose &= np.all(np.isfinite(urdf_quats), axis=1)
    valid_pose &= quat_norm > 1.0e-12

    if np.any(valid_pose):
        rotations = Rotation.from_quat(urdf_quats[valid_pose] / quat_norm[valid_pose, None]).as_matrix()
        tip_positions[valid_pose] = urdf_positions[valid_pose] + np.einsum(
            "nij,j->ni",
            rotations,
            tip_local,
        )

    if not np.all(valid_pose):
        tip_positions[~valid_pose] = fallback_tip[~valid_pose]
        label = f"{link_names[head_idx]} rigid tip (proxy fallback)"
    else:
        label = f"{link_names[head_idx]} rigid tip"

    return tip_positions, label