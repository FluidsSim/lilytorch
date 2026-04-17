from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pandas as pd

from lilytorch.util.paths import save_path


CASE_SCRIPTS = [
    "plot_vels_ms001mpt001.py",
    "plot_vels_ms004mpt003.py",
    "plot_vels_msms007mpt001.py",
]

ALIGNMENT_OVERRIDES = {
    "plot_vels_ms004mpt003.py": (1.4, 0.0),
    "plot_vels_msms007mpt001.py": (0.0, 3.2),
}

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_CSV_PATH = OUTPUT_DIR / "steady_state_forward_speed_comparison.csv"
OUTPUT_MD_PATH = OUTPUT_DIR / "steady_state_forward_speed_comparison.md"


@dataclass(frozen=True)
class SteadyStateSummary:
    case: str
    frequency_hz: float
    amplitude_deg: float
    lambda_value: float
    period_s: float
    window_start_s: float
    window_end_s: float
    exp_forward_mps: float
    sim_forward_mps: float
    sim_minus_exp_mps: float
    exp_samples: int
    sim_samples: int


def _load_case_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import case module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_simulation_path(module: ModuleType) -> Path:
    direct_path = Path(module.SIMULATION_PATH)
    if direct_path.exists():
        return direct_path

    run_timestamp = Path(module.RUN_DIR).name
    matches = sorted(Path(save_path).glob(f"**/{run_timestamp}/output/simulation.hdf5"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "Unable to locate simulation.hdf5 for run timestamp "
            f"{run_timestamp} under {save_path}"
        )
    raise FileNotFoundError(
        "Multiple simulation.hdf5 files matched run timestamp "
        f"{run_timestamp}: {matches}"
    )


def _load_simulation_kinematics(module: ModuleType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulation_path = _resolve_simulation_path(module)

    with h5py.File(simulation_path, "r") as h5_file:
        link_group = h5_file["FARMSLISTanimats"]["0"]["sensors"]["links"]
        link_array = np.asarray(link_group["array"], dtype=float)
        link_names = module._decode_link_names(link_group["names"][()])
        times = np.asarray(h5_file["times"], dtype=float)

    it_stop = min(module.IT_MAX, link_array.shape[0], times.shape[0])
    link_array = link_array[:it_stop]
    times = times[:it_stop]

    sim_head_positions_3d, _ = module._extract_sim_head_trajectory(link_array, link_names)
    planar_positions, _, _ = module._flatten_to_best_fit_plane(sim_head_positions_3d)

    fps = 1.0 / np.median(np.diff(times))
    x_planar_sm = module._lowpass(
        planar_positions[:, 0],
        fps=fps,
        cutoff_hz=module.LOWPASS_CUTOFF_HZ,
    )
    y_planar_sm = module._lowpass(
        planar_positions[:, 1],
        fps=fps,
        cutoff_hz=module.LOWPASS_CUTOFF_HZ,
    )
    kinematics = module._compute_projected_speed(times, x_planar_sm, y_planar_sm)

    sim_forward = np.asarray(kinematics["speed_fwd"], dtype=float)
    sim_lateral = np.asarray(kinematics["speed_lat"], dtype=float)
    return times, sim_forward, sim_lateral


def _compute_alignment_times(
    module: ModuleType,
    script_name: str,
    sim_times: np.ndarray,
    sim_lateral: np.ndarray,
    exp_times: np.ndarray,
    exp_lateral: np.ndarray,
    frequency_hz: float,
) -> tuple[float, float]:
    override = ALIGNMENT_OVERRIDES.get(script_name)
    if override is not None:
        return override

    sim_peak_time = module._find_first_lateral_peak_time(
        sim_times,
        sim_lateral,
        frequency_hz,
    )
    exp_peak_time = module._find_first_lateral_peak_time(
        exp_times,
        exp_lateral,
        frequency_hz,
    )
    return sim_peak_time, exp_peak_time


def _compute_summary(script_path: Path) -> SteadyStateSummary:
    module = _load_case_module(script_path)

    exp_track = module._load_experiment_track(module.TRACK_CSV_PATH)
    control_meta = module._load_position_control_metadata(module.TRACK_CSV_PATH)
    sim_times, sim_forward, sim_lateral = _load_simulation_kinematics(module)

    exp_times = exp_track["time_s"].to_numpy(dtype=float)
    exp_forward = exp_track["speed_fwd_mps"].to_numpy(dtype=float)
    exp_lateral = exp_track["speed_lat_mps"].to_numpy(dtype=float)

    sim_peak_time, exp_peak_time = _compute_alignment_times(
        module,
        script_path.name,
        sim_times,
        sim_lateral,
        exp_times,
        exp_lateral,
        control_meta["Frequency_Hz"],
    )
    sim_times_aligned = sim_times + (exp_peak_time - sim_peak_time)

    period_s = 1.0 / control_meta["Frequency_Hz"]
    window_end_s = float(module.PLOT_TMAX)
    window_start_s = window_end_s - 2.0 * period_s

    sim_mask = (
        np.isfinite(sim_times_aligned)
        & np.isfinite(sim_forward)
        & (sim_times_aligned >= window_start_s)
        & (sim_times_aligned <= window_end_s)
    )
    exp_mask = (
        np.isfinite(exp_times)
        & np.isfinite(exp_forward)
        & (exp_times >= window_start_s)
        & (exp_times <= window_end_s)
    )

    if not np.any(sim_mask):
        raise ValueError(f"No simulation samples in steady-state window for {script_path.name}")
    if not np.any(exp_mask):
        raise ValueError(f"No experiment samples in steady-state window for {script_path.name}")

    case_name = Path(module.TRACK_CSV_PATH).name.removesuffix("_track.csv")
    exp_forward_mps = float(np.nanmean(exp_forward[exp_mask]))
    sim_forward_mps = float(np.nanmean(sim_forward[sim_mask]))

    return SteadyStateSummary(
        case=case_name,
        frequency_hz=float(control_meta["Frequency_Hz"]),
        amplitude_deg=float(control_meta["positionAmplitude_deg"]),
        lambda_value=float(control_meta["Lambda"]),
        period_s=period_s,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        exp_forward_mps=exp_forward_mps,
        sim_forward_mps=sim_forward_mps,
        sim_minus_exp_mps=sim_forward_mps - exp_forward_mps,
        exp_samples=int(exp_mask.sum()),
        sim_samples=int(sim_mask.sum()),
    )


def _build_dataframe(summaries: list[SteadyStateSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [summary.case for summary in summaries],
            "frequency_hz": [summary.frequency_hz for summary in summaries],
            "amplitude_deg": [summary.amplitude_deg for summary in summaries],
            "lambda_value": [summary.lambda_value for summary in summaries],
            "period_s": [summary.period_s for summary in summaries],
            "window_start_s": [summary.window_start_s for summary in summaries],
            "window_end_s": [summary.window_end_s for summary in summaries],
            "exp_forward_mps": [summary.exp_forward_mps for summary in summaries],
            "sim_forward_mps": [summary.sim_forward_mps for summary in summaries],
            "sim_minus_exp_mps": [summary.sim_minus_exp_mps for summary in summaries],
            "exp_samples": [summary.exp_samples for summary in summaries],
            "sim_samples": [summary.sim_samples for summary in summaries],
        }
    )


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _build_markdown_table(df: pd.DataFrame) -> str:
    headers = [
        "Case",
        "f [Hz]",
        "A [deg]",
        "Lambda",
        "Period [s]",
        "Steady-state window [s]",
        "Exp avg fwd [m/s]",
        "Sim avg fwd [m/s]",
        "Sim - Exp [m/s]",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case,
                    _format_float(row.frequency_hz, digits=2),
                    _format_float(row.amplitude_deg, digits=1),
                    _format_float(row.lambda_value, digits=1),
                    _format_float(row.period_s, digits=2),
                    f"[{row.window_start_s:.2f}, {row.window_end_s:.2f}]",
                    _format_float(row.exp_forward_mps),
                    _format_float(row.sim_forward_mps),
                    _format_float(row.sim_minus_exp_mps),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    summaries = [_compute_summary(base_dir / script_name) for script_name in CASE_SCRIPTS]
    df = _build_dataframe(summaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    markdown_lines = [
        "# Steady-State Average Forward Speed Comparison",
        "",
        "Steady state is computed over the last two oscillation periods of the plotted window, using [xlim[1] - 2 * period, xlim[1]].",
        "",
        _build_markdown_table(df),
        "",
    ]
    OUTPUT_MD_PATH.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(df.to_string(index=False))
    print()
    print(f"Wrote {OUTPUT_CSV_PATH}")
    print(f"Wrote {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()