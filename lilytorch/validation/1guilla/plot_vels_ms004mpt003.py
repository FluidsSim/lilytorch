import os

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from lilytorch.util.oneguilla_head_tip import (
    extract_1guilla_head_tip_trajectory,
    resolve_saved_simulation_path,
)
from lilytorch.util.paths import save_path

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '20'


RUN_DIR = os.path.join(save_path, "2026-06-23T10:32:14.843214")
# RUN_DIR = os.path.join(save_path, "1guilla_surface/2026-06-18T14:35:00.741808")

# RUN_DIR = os.path.join(save_path, "1guilla_self_propelled/2026-06-02T16:48:32.413301")
SIMULATION_PATH = os.path.join(RUN_DIR, "output", "simulation.hdf5")
TRACK_CSV_PATH = "/data/andreaferrario/1guilla_experiments/swim/videos/ms004mpt003_track.csv"
HEAD_LINK_NAME = "link0"
IT_MAX = 20000
LOWPASS_CUTOFF_HZ = 4.0
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUTPUT_NAME = "vels_ms004mpt003.svg"
PLOT_TMAX = 17.5
FORWARD_COLOR = "tab:orange"
LATERAL_COLOR = "tab:green"
SIM_MARKER = ""
EXP_MARKER = ""


def _lowpass(arr: np.ndarray, fps: float, cutoff_hz: float = 3.0) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth low-pass filter.

    This matches the filtering strategy used in track_robot.py: NaN gaps are
    interpolated before filtering and restored afterwards.
    """
    valid = ~np.isnan(arr)
    if valid.sum() < 10:
        return arr

    nyquist = 0.5 * fps
    wn = cutoff_hz / nyquist
    if wn >= 1.0:
        return arr

    b, a = butter(4, wn, btype="low")
    idx = np.arange(len(arr))
    arr_filled = np.interp(idx, idx[valid], arr[valid])
    filtered = filtfilt(b, a, arr_filled)
    filtered[~valid] = np.nan
    return filtered


def _flatten_to_best_fit_plane(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.all(np.isfinite(points_3d), axis=1)
    if valid.sum() < 3:
        raise ValueError("Need at least three valid trajectory samples to fit a plane.")

    centroid = points_3d[valid].mean(axis=0)
    centered = points_3d[valid] - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T

    planar_points = (points_3d - centroid) @ basis
    first_valid = np.flatnonzero(valid)[0]
    last_valid = np.flatnonzero(valid)[-1]
    net_displacement = planar_points[last_valid] - planar_points[first_valid]
    if net_displacement[0] < 0.0:
        basis[:, 0] *= -1.0
        planar_points[:, 0] *= -1.0

    planar_points -= planar_points[first_valid]
    return planar_points, basis, centroid


def _compute_projected_speed(times: np.ndarray, x_pos: np.ndarray, y_pos: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(x_pos) & np.isfinite(y_pos)
    if valid.sum() < 3:
        raise ValueError("Need at least three valid planar COM samples for the quadratic fit.")

    px_coeffs = np.polyfit(times[valid], x_pos[valid], 2)
    py_coeffs = np.polyfit(times[valid], y_pos[valid], 2)

    x_fit = np.polyval(px_coeffs, times)
    y_fit = np.polyval(py_coeffs, times)

    dt = np.gradient(times)
    vx = np.gradient(x_pos) / dt
    vy = np.gradient(y_pos) / dt
    speed_2d = np.hypot(vx, vy)

    tx = 2.0 * px_coeffs[0] * times + px_coeffs[1]
    ty = 2.0 * py_coeffs[0] * times + py_coeffs[1]
    tangent_norm = np.hypot(tx, ty)
    tangent_norm = np.where(tangent_norm > 0.0, tangent_norm, 1.0)
    tx_unit = tx / tangent_norm
    ty_unit = ty / tangent_norm
    nx_unit = -ty_unit
    ny_unit = tx_unit

    speed_fwd = vx * tx_unit + vy * ty_unit
    speed_lat = vx * nx_unit + vy * ny_unit

    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "vx": vx,
        "vy": vy,
        "speed_2d": speed_2d,
        "speed_fwd": speed_fwd,
        "speed_lat": speed_lat,
    }


def _load_experiment_track(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Experimental track CSV not found: {csv_path}")

    track_df = pd.read_csv(csv_path)
    required_columns = {"time_s", "speed_fwd_mps", "speed_lat_mps"}
    missing_columns = required_columns.difference(track_df.columns)
    if missing_columns:
        raise ValueError(
            "Experimental track CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    return track_df


def _derive_experiment_log_path(track_csv_path: str) -> str:
    swim_dir = os.path.dirname(os.path.dirname(track_csv_path))
    track_name = os.path.basename(track_csv_path)
    if not track_name.endswith("_track.csv"):
        raise ValueError(f"Unexpected track CSV name: {track_name}")
    stem = track_name[:-10]
    return os.path.join(swim_dir, "log", f"{stem}log.csv")


def _load_position_control_metadata(track_csv_path: str) -> dict[str, float]:
    log_csv_path = _derive_experiment_log_path(track_csv_path)
    if not os.path.exists(log_csv_path):
        raise FileNotFoundError(f"Experiment log CSV not found: {log_csv_path}")

    meta_df = pd.read_csv(log_csv_path, nrows=1)
    return {
        "positionAmplitude_deg": float(meta_df["positionAmplitude[deg]"].iloc[0]),
        "Lambda": float(meta_df["Lambda"].iloc[0]),
        "Frequency_Hz": float(meta_df["Frequency[Hz]"].iloc[0]),
    }


def _find_first_lateral_peak_time(
    times: np.ndarray,
    lateral_speed: np.ndarray,
    freq_hz: float,
) -> float:
    valid = np.isfinite(times) & np.isfinite(lateral_speed)
    t = np.asarray(times[valid], dtype=float)
    y = np.asarray(lateral_speed[valid], dtype=float)
    if t.size < 3:
        raise ValueError("Need at least three samples to detect a lateral-speed peak.")

    max_abs = float(np.max(np.abs(y)))
    if max_abs <= 0.0:
        return float(t[0])

    dt = float(np.median(np.diff(t)))
    min_distance = max(1, int(round((0.5 / max(freq_hz, 1e-12)) / dt)))
    peaks, _ = find_peaks(
        y,
        height=0.25 * max_abs,
        prominence=0.15 * max_abs,
        distance=min_distance,
    )
    if peaks.size == 0:
        return float(t[int(np.argmax(y))])
    return float(t[peaks[0]])


def _compute_markevery(mask: np.ndarray, target_markers: int = 18) -> int:
    n_points = int(np.count_nonzero(mask))
    if n_points <= 1:
        return 1
    return max(1, n_points // target_markers)


def _decode_link_names(raw_names: np.ndarray) -> list[str]:
    return [name.decode() if isinstance(name, bytes) else str(name) for name in raw_names]


def _extract_sim_head_trajectory(
    link_array: np.ndarray,
    link_names: list[str],
) -> tuple[np.ndarray, str]:
    return extract_1guilla_head_tip_trajectory(
        link_array,
        link_names,
        head_link_name=HEAD_LINK_NAME,
    )


def main() -> None:
    simulation_path = resolve_saved_simulation_path(SIMULATION_PATH, RUN_DIR)

    exp_track = _load_experiment_track(TRACK_CSV_PATH)
    control_meta = _load_position_control_metadata(TRACK_CSV_PATH)

    with h5py.File(simulation_path, "r") as h5_file:
        link_group = h5_file["FARMSLISTanimats"]["0"]["sensors"]["links"]
        link_array = np.asarray(link_group["array"], dtype=float)
        link_names = _decode_link_names(link_group["names"][()])
        times = np.asarray(h5_file["times"], dtype=float)

    it_stop = min(IT_MAX, link_array.shape[0], times.shape[0])
    link_array = link_array[:it_stop]
    times = times[:it_stop]

    sim_head_positions_3d, sim_head_label = _extract_sim_head_trajectory(
        link_array,
        link_names,
    )
    planar_positions, _, _ = _flatten_to_best_fit_plane(sim_head_positions_3d)

    fps = 1.0 / np.median(np.diff(times))
    x_planar = planar_positions[:, 0]
    y_planar = planar_positions[:, 1]
    x_planar_sm = _lowpass(x_planar, fps=fps, cutoff_hz=LOWPASS_CUTOFF_HZ)
    y_planar_sm = _lowpass(y_planar, fps=fps, cutoff_hz=LOWPASS_CUTOFF_HZ)

    kinematics = _compute_projected_speed(times, x_planar_sm, y_planar_sm)

    sim_forward = np.asarray(kinematics["speed_fwd"], dtype=float)
    sim_lateral = np.asarray(kinematics["speed_lat"], dtype=float)
    exp_time = exp_track["time_s"].to_numpy(dtype=float)
    exp_forward = exp_track["speed_fwd_mps"].to_numpy(dtype=float)
    exp_lateral = exp_track["speed_lat_mps"].to_numpy(dtype=float)

    # sim_peak_time = _find_first_lateral_peak_time(
    #     times,
    #     sim_lateral,
    #     control_meta["Frequency_Hz"],
    # )
    # exp_peak_time = _find_first_lateral_peak_time(
    #     exp_time,
    #     exp_lateral,
    #     control_meta["Frequency_Hz"],
    # )

    sim_peak_time = 1.4
    exp_peak_time = 0
    sim_time_aligned = times + (exp_peak_time - sim_peak_time)

    sim_mask = (
        np.isfinite(sim_time_aligned)
        & np.isfinite(sim_forward)
        & np.isfinite(sim_lateral)
        & (sim_time_aligned >= 0.0)
        & (sim_time_aligned <= PLOT_TMAX)
    )
    exp_mask = (
        np.isfinite(exp_time)
        & np.isfinite(exp_forward)
        & np.isfinite(exp_lateral)
        & (exp_time >= 0.0)
        & (exp_time <= PLOT_TMAX)
    )

    all_speeds = np.concatenate([
        sim_forward[sim_mask],
        sim_lateral[sim_mask],
        exp_forward[exp_mask],
        exp_lateral[exp_mask],
    ])
    max_abs_speed = np.max(np.abs(all_speeds)) if all_speeds.size else 1.0
    y_margin = 0.08 * max_abs_speed
    y_limits = (-max_abs_speed - y_margin, max_abs_speed + y_margin)

    sim_markevery = _compute_markevery(sim_mask)
    exp_markevery = _compute_markevery(exp_mask)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))


    ax.plot(
        sim_time_aligned[sim_mask],
        sim_forward[sim_mask],
        color=FORWARD_COLOR,
        lw=2.0,
        marker=SIM_MARKER,
        markersize=4.0,
        markevery=sim_markevery,
        label=f"Forward speed — sim ",
    )
    ax.plot(
        exp_time[exp_mask],
        exp_forward[exp_mask],
        color=FORWARD_COLOR,
        lw=1.8,
        ls="--",
        marker=EXP_MARKER,
        markersize=4.2,
        markevery=exp_markevery,
        label="Forward speed — exp",
    )
    ax.plot(
        sim_time_aligned[sim_mask],
        sim_lateral[sim_mask],
        color=LATERAL_COLOR,
        lw=2.0,
        marker=SIM_MARKER,
        markersize=4.0,
        markevery=sim_markevery,
        label=f"Lateral speed — sim",
    )
    ax.plot(
        exp_time[exp_mask],
        exp_lateral[exp_mask],
        color=LATERAL_COLOR,
        lw=1.8,
        ls="--",
        marker=EXP_MARKER,
        markersize=4.2,
        markevery=exp_markevery,
        label="Lateral speed — exp",
    )
    ax.axhline(0.0, color="0.3", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_xlim(0.0, PLOT_TMAX)
    ax.set_ylim(y_limits)
    ax.grid(True)
    ax.legend(fontsize="small", ncol=2)

    fig.suptitle(
        f"Position control: f={control_meta['Frequency_Hz']:g} Hz, "
        f"A={control_meta['positionAmplitude_deg']:g} deg, "
        f"Lambda={control_meta['Lambda']:g}",
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_NAME), format="svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
