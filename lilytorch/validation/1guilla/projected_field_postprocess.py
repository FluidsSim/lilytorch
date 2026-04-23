#!/usr/bin/env python3
"""Render PIV-style projected fields from recorded HDF5 snapshots.

Usage
-----
    python projected_field_postprocess.py <run_dir>
    python projected_field_postprocess.py <save_path> --output-tag test_preview
    python projected_field_postprocess.py <run_dir> --camera-axis -z --zlim -0.04 0.04

The script reads ``fields.h5`` / ``fields.hdf5`` and synthesizes a 2-D
projected field from the recorded 3-D velocity volume. By default it:

1. projects the in-plane velocity through the selected depth slab, and
2. computes the corresponding 2-D curl, producing a PIV-style top view.

The generated PNG frames are saved into a new sub-folder under the run
directory and are then encoded into MP4 or GIF using ``video_postprocess``.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_python_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from lilytorch.src.video_postprocess import make_videos
except ImportError:
    _src_root = Path(__file__).resolve().parents[2] / "src"
    make_videos = _load_python_module(
        "lilytorch_video_postprocess",
        _src_root / "video_postprocess.py",
    ).make_videos


def _import_plotting_module():
    try:
        from lilytorch.src import plotting
    except ImportError:
        _src_root = Path(__file__).resolve().parents[2] / "src"
        plotting = _load_python_module(
            "lilytorch_plotting",
            _src_root / "plotting.py",
        )
    return plotting


def _latest_run(save_path: str | Path) -> Path:
    p = Path(save_path)
    if not p.exists():
        sys.exit(f"Path does not exist: {p}")
    if not p.is_dir():
        sys.exit(f"Not a directory: {p}")
    runs = sorted(
        [d for d in p.iterdir() if d.is_dir() and re.match(r"\d{4}-", d.name)],
        key=lambda d: d.name,
    )
    if not runs:
        sys.exit(f"No timestamped run directories found under {p}")
    return runs[-1]


def _resolve_fields_file(run_dir: Path) -> Path | None:
    for name in ("fields.h5", "fields.hdf5"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _load_grids_h5(fields_file: Path) -> dict[str, np.ndarray]:
    with h5py.File(fields_file, "r") as h5_file:
        if "grids" not in h5_file:
            sys.exit(f"Missing 'grids' group in {fields_file}")
        return {name: h5_file[f"grids/{name}"][:] for name in h5_file["grids"]}


def _available_iterations_h5(fields_file: Path) -> list[int]:
    with h5py.File(fields_file, "r") as h5_file:
        if "fields" not in h5_file:
            sys.exit(f"Missing 'fields' group in {fields_file}")
        iterations = sorted(int(name) for name in h5_file["fields"] if name.isdigit())
    if not iterations:
        sys.exit(f"No recorded field snapshots found in {fields_file}")
    return iterations


def _sanitize_output_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return cleaned or None


def _parse_camera_axis(axis_spec: str) -> tuple[int, int, str]:
    spec = axis_spec.strip().lower()
    axis_sign = 1
    if spec.startswith("+"):
        spec = spec[1:]
    elif spec.startswith("-"):
        axis_sign = -1
        spec = spec[1:]
    if spec not in ("x", "y", "z"):
        sys.exit(
            "--camera-axis must be one of +x, -x, +y, -y, +z, -z, or x/y/z",
        )
    return {"x": 0, "y": 1, "z": 2}[spec], axis_sign, spec


def _core_coords(coord: np.ndarray) -> np.ndarray:
    return coord[1:-1] if coord.size > 2 else coord


def _core_slice_for_limits(
    coord: np.ndarray,
    limits: tuple[float, float] | list[float] | None,
    axis_name: str,
) -> slice:
    if coord.size == 0:
        sys.exit(f"No interior samples available along {axis_name}")
    if limits is None:
        return slice(0, coord.size)
    lo, hi = sorted((float(limits[0]), float(limits[1])))
    idx = np.where((coord >= lo) & (coord <= hi))[0]
    if idx.size == 0:
        sys.exit(f"{axis_name} limits [{lo}, {hi}] do not intersect the recorded grid")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _full_dataset_slice(core_slice: slice) -> slice:
    return slice(core_slice.start + 1, core_slice.stop + 1)


def _coord_values(coord: np.ndarray, coord_slice: slice) -> np.ndarray:
    values = coord[coord_slice]
    if values.size == 0:
        sys.exit("Selected camera window produced an empty coordinate range")
    return values


def _extent_from_coords(coord_a: np.ndarray, coord_b: np.ndarray) -> tuple[float, float, float, float]:
    ha = 0.5 * float(coord_a[1] - coord_a[0]) if coord_a.size > 1 else 0.0
    hb = 0.5 * float(coord_b[1] - coord_b[0]) if coord_b.size > 1 else 0.0
    return (
        float(coord_a[0]) - ha,
        float(coord_a[-1]) + ha,
        float(coord_b[0]) - hb,
        float(coord_b[-1]) + hb,
    )


def _block_reduce_1d(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(values, dtype=np.float64)
    chunks = []
    for start in range(0, len(values), factor):
        stop = min(start + factor, len(values))
        chunks.append(np.mean(values[start:stop], dtype=np.float64))
    return np.asarray(chunks, dtype=np.float64)


def _block_reduce_2d(field: np.ndarray, factor: int, reducer) -> np.ndarray:
    if factor <= 1:
        return np.asarray(field, dtype=np.float64)
    rows = []
    for istart in range(0, field.shape[0], factor):
        row = []
        istop = min(istart + factor, field.shape[0])
        for jstart in range(0, field.shape[1], factor):
            jstop = min(jstart + factor, field.shape[1])
            row.append(reducer(field[istart:istop, jstart:jstop]))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _gaussian_filter_2d(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0:
        return np.asarray(field, dtype=np.float64)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("WARNING: scipy not available, skipping Gaussian smoothing")
        return np.asarray(field, dtype=np.float64)
    return gaussian_filter(np.asarray(field, dtype=np.float64), sigma=sigma_cells, mode="nearest")


def _apply_piv_measurement_model(
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    sdf_2d: np.ndarray | None,
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    *,
    coarsen: int,
    velocity_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Mimic PIV interrogation-window averaging before computing curl."""
    vel_a_out = np.asarray(vel_a, dtype=np.float64)
    vel_b_out = np.asarray(vel_b, dtype=np.float64)
    sdf_out = None if sdf_2d is None else np.asarray(sdf_2d, dtype=np.float64)
    coord_a_out = np.asarray(coord_a, dtype=np.float64)
    coord_b_out = np.asarray(coord_b, dtype=np.float64)

    if coarsen > 1:
        vel_a_out = _block_reduce_2d(vel_a_out, coarsen, lambda chunk: np.mean(chunk, dtype=np.float64))
        vel_b_out = _block_reduce_2d(vel_b_out, coarsen, lambda chunk: np.mean(chunk, dtype=np.float64))
        coord_a_out = _block_reduce_1d(coord_a_out, coarsen)
        coord_b_out = _block_reduce_1d(coord_b_out, coarsen)
        if sdf_out is not None:
            sdf_out = _block_reduce_2d(sdf_out, coarsen, np.min)

    if velocity_sigma > 0:
        vel_a_out = _gaussian_filter_2d(vel_a_out, velocity_sigma)
        vel_b_out = _gaussian_filter_2d(vel_b_out, velocity_sigma)

    return vel_a_out, vel_b_out, sdf_out, coord_a_out, coord_b_out


def _weighted_projection(
    data: np.ndarray,
    valid_mask: np.ndarray,
    depth_axis: int,
    mode: str,
    depth_coords: np.ndarray,
    focus_depth: float | None,
    depth_sigma: float | None,
    selector: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "sum":
        return np.sum(np.where(valid_mask, data, 0.0), axis=depth_axis)

    if mode == "max_abs":
        if selector is None:
            sys.exit("Internal error: max_abs projection requires a selector field")
        masked = np.where(valid_mask, selector, -np.inf)
        best_idx = np.argmax(masked, axis=depth_axis)
        all_invalid = ~np.isfinite(np.max(masked, axis=depth_axis))
        expanded = np.expand_dims(best_idx, axis=depth_axis)
        projected = np.take_along_axis(data, expanded, axis=depth_axis).squeeze(axis=depth_axis)
        projected[all_invalid] = 0.0
        return projected

    weights = valid_mask.astype(np.float64)
    if mode == "gaussian":
        if depth_coords.size == 0:
            sys.exit("Cannot build Gaussian depth weights on an empty depth axis")
        if focus_depth is None:
            focus_depth = 0.5 * float(depth_coords[0] + depth_coords[-1])
        if depth_sigma is None:
            span = abs(float(depth_coords[-1] - depth_coords[0]))
            if depth_coords.size > 1:
                span = max(span, float(abs(depth_coords[1] - depth_coords[0])))
            depth_sigma = max(span / 6.0, 1.0e-12)
        elif depth_sigma <= 0:
            sys.exit("--depth-sigma must be positive")
        profile = np.exp(-0.5 * ((depth_coords - focus_depth) / depth_sigma) ** 2)
        reshape = [1] * data.ndim
        reshape[depth_axis] = depth_coords.size
        weights = weights * profile.reshape(reshape)

    numerator = np.sum(data * weights, axis=depth_axis)
    denominator = np.sum(weights, axis=depth_axis)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def _compute_projected_curl(
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    sign: int,
) -> np.ndarray:
    curl = np.zeros_like(vel_a, dtype=np.float64)
    if vel_a.shape[0] < 2 or vel_a.shape[1] < 2:
        return curl

    dvelb_da = np.zeros_like(vel_a, dtype=np.float64)
    dvela_db = np.zeros_like(vel_a, dtype=np.float64)

    if coord_a.size > 2:
        da = (coord_a[2:] - coord_a[:-2]).reshape(-1, 1)
        dvelb_da[1:-1, :] = (vel_b[2:, :] - vel_b[:-2, :]) / da
    if coord_b.size > 2:
        db = (coord_b[2:] - coord_b[:-2]).reshape(1, -1)
        dvela_db[:, 1:-1] = (vel_a[:, 2:] - vel_a[:, :-2]) / db

    curl[:, :] = sign * (dvelb_da - dvela_db)
    return curl


def _apply_camera_transforms(
    field_2d: np.ndarray,
    sdf_2d: np.ndarray | None,
    roll_deg: float,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if abs(roll_deg) > 1.0e-12:
        try:
            from scipy.ndimage import rotate

            field_2d = rotate(
                field_2d,
                angle=roll_deg,
                reshape=False,
                order=1,
                mode="nearest",
            )
            if sdf_2d is not None:
                sdf_2d = rotate(
                    sdf_2d,
                    angle=roll_deg,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )
        except ImportError:
            print("WARNING: scipy not available, ignoring --camera-roll")

    if flip_horizontal:
        field_2d = np.flip(field_2d, axis=0)
        if sdf_2d is not None:
            sdf_2d = np.flip(sdf_2d, axis=0)
    if flip_vertical:
        field_2d = np.flip(field_2d, axis=1)
        if sdf_2d is not None:
            sdf_2d = np.flip(sdf_2d, axis=1)
    return field_2d, sdf_2d


def _normalize_visual_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate plotting arguments and recover from the most common typo."""
    try:
        numeric_cmap = float(args.projected_cmap)
    except (TypeError, ValueError):
        numeric_cmap = None

    if numeric_cmap is not None:
        if args.projected_vmax is None:
            print(
                "WARNING: numeric value passed to --projected-cmap; "
                "treating it as --projected-vmax and keeping the default colormap RdBu_r",
            )
            args.projected_vmax = numeric_cmap
            args.projected_cmap = "RdBu_r"
        else:
            sys.exit(
                "Received a numeric value for --projected-cmap. "
                "Use a colormap name such as RdBu_r, or pass the upper limit with --projected-vmax.",
            )

    try:
        import matplotlib.pyplot as plt

        plt.get_cmap(args.projected_cmap)
    except Exception:
        sys.exit(
            f"Invalid --projected-cmap '{args.projected_cmap}'. "
            "Use a valid Matplotlib colormap such as RdBu_r, or if you meant a color bound use --projected-vmax.",
        )

    if (
        args.projected_vmin is not None
        and args.projected_vmax is not None
        and args.projected_vmin >= args.projected_vmax
    ):
        sys.exit("--projected-vmin must be smaller than --projected-vmax")

    return args


def render_projected_field_frames(run_dir: Path, args: argparse.Namespace) -> str:
    if args.projected_field != "piv_curl":
        sys.exit(f"Unsupported projected field '{args.projected_field}'")

    fields_file = _resolve_fields_file(run_dir)
    if fields_file is None:
        sys.exit(f"No fields.h5 or fields.hdf5 found under {run_dir}")

    plotting = _import_plotting_module()
    iterations = _available_iterations_h5(fields_file)
    grids = _load_grids_h5(fields_file)
    if not {"x", "y", "z"}.issubset(grids):
        sys.exit(f"Projected fields require a 3-D run with x/y/z grids in {fields_file}")

    axis_index, axis_sign, axis_name = _parse_camera_axis(args.camera_axis)
    plane_axes = [idx for idx in range(3) if idx != axis_index]
    plane_names = ["x", "y", "z"]
    plane_labels = (
        f"{plane_names[plane_axes[0]]} [m]",
        f"{plane_names[plane_axes[1]]} [m]",
    )
    curl_sign = {0: 1, 1: -1, 2: 1}[axis_index] * axis_sign

    core = {name: _core_coords(values) for name, values in grids.items()}
    core_slices = {
        "x": _core_slice_for_limits(core["x"], args.xlim, "x"),
        "y": _core_slice_for_limits(core["y"], args.ylim, "y"),
        "z": _core_slice_for_limits(core["z"], args.zlim, "z"),
    }
    full_slices = tuple(_full_dataset_slice(core_slices[name]) for name in ("x", "y", "z"))
    coord_values = {name: _coord_values(core[name], core_slices[name]) for name in ("x", "y", "z")}
    coord_a = coord_values[plane_names[plane_axes[0]]]
    coord_b = coord_values[plane_names[plane_axes[1]]]
    depth_coords = coord_values[axis_name]

    if coord_a.size < 2 or coord_b.size < 2:
        sys.exit("Projected curl needs at least two samples along both image-plane axes")

    output_tag = _sanitize_output_tag(args.output_tag)
    output_name = args.projected_field if output_tag is None else f"{args.projected_field}_{output_tag}"

    print(
        "Rendering projected HDF5 field "
        f"'{output_name}' from {fields_file.name} "
        f"(axis={args.camera_axis}, projection={args.projection_mode}, frames={len(iterations)})",
    )

    with h5py.File(fields_file, "r") as h5_file:
        for iteration in iterations:
            grp = h5_file[f"fields/{iteration:06d}"]
            missing = [name for name in ("u", "v", "w") if name not in grp]
            if missing:
                sys.exit(f"Snapshot {iteration:06d} is missing {missing} in {fields_file}")

            u = grp["u"][full_slices]
            v = grp["v"][full_slices]
            w = grp["w"][full_slices]
            sdf = grp["sdf"][full_slices] if "sdf" in grp else None

            components = (u, v, w)
            vel_a_3d = components[plane_axes[0]]
            vel_b_3d = components[plane_axes[1]]

            if sdf is not None:
                valid_mask = sdf > 0.0
            else:
                valid_mask = np.ones_like(vel_a_3d, dtype=bool)

            selector = np.sqrt(vel_a_3d**2 + vel_b_3d**2)
            vel_a_2d = _weighted_projection(
                vel_a_3d,
                valid_mask,
                axis_index,
                args.projection_mode,
                depth_coords,
                args.focus_depth,
                args.depth_sigma,
                selector=selector,
            )
            vel_b_2d = _weighted_projection(
                vel_b_3d,
                valid_mask,
                axis_index,
                args.projection_mode,
                depth_coords,
                args.focus_depth,
                args.depth_sigma,
                selector=selector,
            )

            sdf_2d = np.min(sdf, axis=axis_index) if sdf is not None else None
            vel_a_2d, vel_b_2d, sdf_2d, coord_a_plot, coord_b_plot = _apply_piv_measurement_model(
                vel_a_2d,
                vel_b_2d,
                sdf_2d,
                coord_a,
                coord_b,
                coarsen=args.coarsen,
                velocity_sigma=args.velocity_sigma,
            )
            curl_2d = _compute_projected_curl(
                vel_a_2d,
                vel_b_2d,
                coord_a_plot,
                coord_b_plot,
                curl_sign,
            )
            curl_2d = 3.9*curl_2d
            if args.curl_sigma > 0:
                curl_2d = _gaussian_filter_2d(curl_2d, args.curl_sigma)
            if sdf_2d is not None:
                curl_2d = np.where(sdf_2d > 0.0, curl_2d, 0.0)
            curl_2d = 3.935713591 * curl_2d
            extent = _extent_from_coords(coord_a_plot, coord_b_plot)
            curl_2d, sdf_2d = _apply_camera_transforms(
                curl_2d,
                sdf_2d,
                args.camera_roll,
                args.flip_horizontal,
                args.flip_vertical,
            )

            plotting.plot_field_2d(
                curl_2d,
                extent,
                output_name,
                iteration,
                str(run_dir),
                vmin=args.projected_vmin,
                vmax=args.projected_vmax,
                sdf_2d=sdf_2d,
                cmap=args.projected_cmap,
                axis_labels=plane_labels,
            )

    return output_name


def main():
    parser = argparse.ArgumentParser(
        description="Render projected PIV-style fields from recorded 3-D HDF5 data.",
    )
    parser.add_argument(
        "path",
        help="Timestamped run directory, or parent save_path (picks latest run).",
    )
    parser.add_argument(
        "--projected-field",
        choices=["piv_curl"],
        default="piv_curl",
        help=(
            "Synthetic field to render. 'piv_curl' projects in-plane velocity through the "
            "selected depth range and then computes the corresponding 2-D curl."
        ),
    )
    parser.add_argument(
        "--camera-axis",
        default="-z",
        help=(
            "Orthographic viewing axis. Use +x/-x/+y/-y/+z/-z; default -z matches a "
            "top-down camera looking downward."
        ),
    )
    parser.add_argument(
        "--projection-mode",
        choices=["mean", "sum", "max_abs", "gaussian"],
        default="gaussian",
        help="How to collapse in-plane velocity along the camera axis.",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-range [m] of the rendered camera window.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-range [m] of the rendered camera window.",
    )
    parser.add_argument(
        "--zlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("ZMIN", "ZMAX"),
        help="Optional z-range [m] used as the projected depth slab.",
    )
    parser.add_argument(
        "--focus-depth",
        type=float,
        default=None,
        help="Focus depth [m] for Gaussian projection weighting (defaults to slab midpoint).",
    )
    parser.add_argument(
        "--depth-sigma",
        type=float,
        default=None,
        help="Gaussian depth-weighting sigma [m] for --projection-mode gaussian.",
    )
    parser.add_argument(
        "--coarsen",
        type=int,
        default=0,
        help=(
            "Block-average the projected in-plane velocity over NxN cells before curl. "
            "Values around 2 to 4 better mimic PIV interrogation windows."
        ),
    )


    parser.add_argument(
        "--velocity-sigma",
        type=float,
        default=0,
        help=(
            "Gaussian smoothing applied to the projected in-plane velocity before curl, "
            "in grid-cell units. Increase this if the synthetic movie shows too many tiny vortices."
        ),
    )
    parser.add_argument(
        "--curl-sigma",
        type=float,
        default=0,
        help="Optional Gaussian smoothing applied to the final curl field, in grid-cell units.",
    )


    parser.add_argument(
        "--camera-roll",
        type=float,
        default=0.0,
        help="In-plane image rotation in degrees applied after projection.",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Mirror the projected frame along the horizontal image axis.",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Mirror the projected frame along the vertical image axis.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix appended to the generated frame-folder and video name.",
    )
    parser.add_argument(
        "--projected-cmap",
        default="RdBu_r",
        help="Matplotlib colormap for the rendered field.",
    )
    parser.add_argument(
        "--projected-vmin",
        type=float,
        default=None,
        help="Fixed lower color limit for the rendered field.",
    )
    parser.add_argument(
        "--projected-vmax",
        type=float,
        default=None,
        help="Fixed upper color limit for the rendered field.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video frame rate (default: computed from dt and save_every).",
    )
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=1.0,
        help="Real-time multiplier for the generated video (default 1.0).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the simulation-time text overlay on the generated video.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality passed to video_postprocess (default 18).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output format for the generated video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Render PNG frames only; skip video encoding.",
    )
    args = parser.parse_args()
    args = _normalize_visual_args(args)

    run_dir = Path(args.path)
    if _resolve_fields_file(run_dir) is None and not (run_dir / "parameters.yaml").exists():
        run_dir = _latest_run(run_dir)
        print(f"Auto-selected latest run: {run_dir}")

    output_name = render_projected_field_frames(run_dir, args)

    if not args.no_video:
        make_videos(
            run_dir,
            fields=[output_name],
            fps=args.fps,
            slow_factor=args.slow_factor,
            time_overlay=not args.no_overlay,
            crf=args.crf,
            fmt=args.fmt,
        )


if __name__ == "__main__":
    main()