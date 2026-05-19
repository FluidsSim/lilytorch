#!/usr/bin/env python3
"""Shared utilities for the dimension-switched cost-analysis benchmark."""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DTYPE = "float32"
# Standalone multigrid is the recommended GPU solver.
# MGCG is avoided because its CG inner loop requires 3 CUDA pipeline stalls
# per iteration (dot products for alpha/beta step lengths), making it
# inherently slower on GPU than standalone multigrid regardless of grid size.
DEFAULT_POISSON_METHOD = "multigrid"
DEFAULT_TIMESTEP = 1.0e-4
DEFAULT_SPAWN_X = -0.65

DX_REF = 2.4 / 1024
MIN_LX_FISH = 1.0
BODY_CENTER_OFFSET = 0.325

MODE_SPECS = {
    "python": {
        "label": "python mode (reference path)",
        "short_label": "python",
        "mode": "python",
        "linestyle": "--",
        "marker": "s",
        "color": "#37474f",
    },
    "kernel": {
        "label": "kernel mode (optimised path)",
        "short_label": "kernel",
        "mode": "kernel",
        "linestyle": "-",
        "marker": "*",
        "color": "#f9a825",
    },
}

MODE_ALIASES = {
    "nboff": "python",
    "nbforces_opt": "kernel",
}

EXPLICIT_PREFIXES = (
    "1b",
    "2 ",
    "3a  ",
    "3b",
    "3c ",
    "3d",
    "3e",
    "3f",
    "4 ",
    "6 ",
)

CATEGORY_PREFIXES = OrderedDict([
    ("Body update\n(SDF eval)", ["1b"]),
    ("mu + normals", ["2 "]),
    ("Convection\n& diffusion", ["3a  "]),
    ("BDIM\nmeta-equation", ["3b"]),
    ("Projection\n(pressure)", ["3c "]),
    ("set_BCs", ["3d"]),
    ("var-density\ncoeffs", ["3e"]),
    ("release BDIM\nfields", ["3f"]),
    ("Forces\ncomputation", ["4 "]),
    ("FARMS\n(apply_forces)", ["6 "]),
])

OTHER_LABEL = "Other\n(residual)"

CAT_COLOURS = {
    "Body update\n(SDF eval)": "#26a69a",
    "mu + normals": "#66bb6a",
    "Convection\n& diffusion": "#42a5f5",
    "BDIM\nmeta-equation": "#ab47bc",
    "Projection\n(pressure)": "#ef5350",
    "set_BCs": "#fbc02d",
    "var-density\ncoeffs": "#7cb342",
    "release BDIM\nfields": "#bcaaa4",
    "Forces\ncomputation": "#ffa726",
    "FARMS\n(apply_forces)": "#5c6bc0",
    OTHER_LABEL: "#90a4ae",
}


@dataclass(frozen=True)
class DimensionSpec:
    dim: int
    short_tag: str
    label: str
    benchmark_label: str
    config_module: str
    grid_fields: tuple[str, ...]
    default_grid: tuple[int, ...]
    presets: dict[str, list[tuple[int, ...]]]
    state_names: tuple[str, ...]
    update_leaf_label: str
    vardens_leaf_label: str
    forces_method_name: str
    forces_timer_label: str
    terminate_on_handler: bool


DIMENSION_SPECS = {
    2: DimensionSpec(
        dim=2,
        short_tag="dim2",
        label="2-D",
        benchmark_label="Pinned 1guilla",
        config_module="lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_2d",
        grid_fields=("Nx", "Ny"),
        default_grid=(512, 128),
        presets={
            "small": [(128, 32), (256, 64), (512, 128)],
            "medium": [(256, 64), (512, 128), (1024, 256)],
            "large": [(256, 64), (512, 128), (1024, 256), (2048, 512)],
            "full": [
                (128, 32),
                (256, 64),
                (512, 128),
                (1024, 256),
                (2048, 512),
                (4096, 1024),
                (4096, 2048),
                (8192, 2048),
                (8192, 4096),
                ],
            "production": [(256, 64), (512, 128), (1024, 256), (2048, 512)],
        },
        state_names=("u0", "v0", "p0"),
        update_leaf_label="1b   SDF eval (per-body x 3 grids)",
        vardens_leaf_label="3e   var-density coeffs [ch cv ch_cc]",
        forces_method_name="forces_method2",
        forces_timer_label="4  forces_method2",
        terminate_on_handler=False,
    ),
    3: DimensionSpec(
        dim=3,
        short_tag="dim3",
        label="3-D",
        benchmark_label="Pinned 1guilla",
        config_module="lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_3d",
        grid_fields=("Nx", "Ny", "Nz"),
        default_grid=(256, 64, 64),
        presets={
            "small": [(128, 32, 32), (256, 64, 64), (256, 128, 64)],
            "medium": [(256, 64, 64), (256, 128, 64), (512, 128, 128)],
            "large": [(256, 64, 64), (256, 128, 64), (256, 128, 128), (512, 128, 128)],
            "full": [(128, 32, 32), (256, 64, 64), (256, 128, 64), (256, 128, 128), (512, 128, 128), (1024, 256, 128)],
            "production": [(256, 64, 64), (256, 128, 64), (512, 128, 128)],
        },
        state_names=("u0", "v0", "w0", "p0"),
        update_leaf_label="1b   SDF eval (per-body x 4 grids)",
        vardens_leaf_label="3e   var-density coeffs [ch cv cw ch_cc]",
        forces_method_name="forces_method2_3d",
        forces_timer_label="4  forces_method2_3d",
        terminate_on_handler=True,
    ),
}


def get_dimension_spec(dim: int) -> DimensionSpec:
    if dim not in DIMENSION_SPECS:
        raise ValueError(f"Unsupported dimension '{dim}'")
    return DIMENSION_SPECS[dim]


def normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    mode_key = mode.strip().lower()
    return MODE_ALIASES.get(mode_key, mode_key)


def parse_modes(raw: str) -> list[str]:
    modes = []
    for item in raw.split(","):
        mode = normalize_mode(item)
        if mode not in MODE_SPECS:
            raise ValueError(f"Unknown solver mode '{item}'")
        modes.append(mode)
    if not modes:
        raise ValueError("No solver modes supplied")
    return modes


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def parse_grid_list(raw: str, dim: int) -> list[tuple[int, ...]]:
    grids = []
    for item in raw.split(","):
        parts = tuple(int(part) for part in item.strip().split(":"))
        if len(parts) != dim:
            raise ValueError(
                f"Invalid grid '{item}' (expected {dim} components separated by ':')"
            )
        grids.append(parts)
    validate_grids(grids, dim)
    return grids


def validate_grids(grids: list[tuple[int, ...]], dim: int) -> None:
    if not grids:
        raise ValueError("No grids supplied")
    for grid in grids:
        if len(grid) != dim:
            raise ValueError(f"Grid {grid} does not match dimension {dim}")
        if not all(_is_power_of_two(value) for value in grid):
            raise ValueError(f"Grid {grid} is invalid: all axes must be powers of two")


def resolve_grid_tuple(args, spec: DimensionSpec) -> tuple[int, ...]:
    values = []
    for index, field in enumerate(spec.grid_fields):
        value = getattr(args, field)
        if value is None:
            value = spec.default_grid[index]
            setattr(args, field, value)
        values.append(int(value))
    grid = tuple(values)
    validate_grids([grid], spec.dim)
    return grid


def grid_tag(grid: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in grid)


def grid_label(grid: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in grid)


def grid_arg(grids: list[tuple[int, ...]]) -> str:
    return ",".join(":".join(str(value) for value in grid) for grid in grids)


def grid_cells(grid: tuple[int, ...]) -> int:
    total = 1
    for value in grid:
        total *= value
    return total


def default_results_dir(script_dir: str, spec: DimensionSpec) -> str:
    return os.path.join(script_dir, "figures", spec.short_tag)


def default_pipeline_dir(script_dir: str, spec: DimensionSpec, poisson_method: str | None = None) -> str:
    tag = spec.short_tag
    if poisson_method:
        tag = f"{tag}_{poisson_method}"
    return os.path.join(script_dir, "figures", f"scaling_conditions_{tag}")


def resolve_solver_mode(cli_args):
    legacy_kernel_mode = (
        getattr(cli_args, "use_kernels", False)
        or getattr(cli_args, "streaming_sdf_2d", False)
        or getattr(cli_args, "force_narrow_batch", False)
        or getattr(cli_args, "force_shared_union", False)
        or getattr(cli_args, "mu_normals_union", False)
        or getattr(cli_args, "bdim_union", False)
        or getattr(cli_args, "streaming_sdf_3d", False)
        or getattr(cli_args, "streaming_forces_3d", False)
    )

    explicit_mode = normalize_mode(getattr(cli_args, "mode", None))
    if explicit_mode == "python":
        if legacy_kernel_mode:
            raise ValueError("--mode python conflicts with kernel-enabling aliases")
        return "python"
    if explicit_mode == "kernel":
        if getattr(cli_args, "no_kernels", False):
            raise ValueError("--mode kernel conflicts with --no_kernels")
        return "kernel"

    if getattr(cli_args, "no_kernels", False) and legacy_kernel_mode:
        raise ValueError("--no_kernels conflicts with kernel-enabling aliases")
    if getattr(cli_args, "no_kernels", False):
        return "python"
    if legacy_kernel_mode:
        return "kernel"
    return None


def matched_columns(columns, prefixes) -> list[str]:
    return [
        column for column in columns
        if column != "TOTAL step" and any(column.startswith(prefix) for prefix in prefixes)
    ]


def timer_colour(label: str) -> str:
    for category, prefixes in CATEGORY_PREFIXES.items():
        if any(label.startswith(prefix) for prefix in prefixes):
            return CAT_COLOURS[category]
    return CAT_COLOURS[OTHER_LABEL]


def _grid_pattern(dim: int) -> re.Pattern[str]:
    if dim == 2:
        return re.compile(r"cost_breakdown_(\d+)x(\d+)\.csv$")
    if dim == 3:
        return re.compile(r"cost_breakdown_(\d+)x(\d+)x(\d+)\.csv$")
    raise ValueError(f"Unsupported dimension '{dim}'")


def load_cost_record(csv_path: str, perstep_path: str | None = None) -> dict | None:
    total_ms = None
    explicit_ms = None
    category_ms = {}
    category_pct = {}

    if perstep_path and os.path.exists(perstep_path):
        df_ps = pd.read_csv(perstep_path)
        if "used" in df_ps.columns:
            df_ps = df_ps[df_ps["used"] != "discarded"]
        if len(df_ps) > 0 and "TOTAL step" in df_ps.columns:
            total_series = df_ps["TOTAL step"].fillna(0.0).to_numpy(dtype=float)
            total_ms = float(np.median(total_series))
            explicit_series = np.zeros_like(total_series)
            total_sum = float(total_series.sum())
            for category, prefixes in CATEGORY_PREFIXES.items():
                columns = matched_columns(df_ps.columns, prefixes)
                if not columns:
                    continue
                series = df_ps[columns].fillna(0.0).sum(axis=1).to_numpy(dtype=float)
                explicit_series += series
                if series.sum() > 0:
                    category_ms[category] = float(np.median(series))
                    category_pct[category] = 100.0 * float(series.sum()) / total_sum if total_sum > 0 else 0.0
            explicit_ms = float(np.median(explicit_series)) if len(explicit_series) else 0.0
            residual_series = np.clip(total_series - explicit_series, 0.0, None)
            if residual_series.sum() > 0:
                category_ms[OTHER_LABEL] = float(np.median(residual_series))
                category_pct[OTHER_LABEL] = (
                    100.0 * float(residual_series.sum()) / total_sum if total_sum > 0 else 0.0
                )

    if total_ms is None:
        with open(csv_path) as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("component", "").strip() == "TOTAL step":
                if row.get("median_ms"):
                    total_ms = float(row["median_ms"])
                elif row.get("mean_ms"):
                    total_ms = float(row["mean_ms"])
                break
        if total_ms is None:
            return None

        explicit_ms = 0.0
        for category, prefixes in CATEGORY_PREFIXES.items():
            subtotal = 0.0
            for row in rows:
                label = row.get("component", "")
                if any(label.startswith(prefix) for prefix in prefixes):
                    value = row.get("median_ms") or row.get("mean_ms")
                    subtotal += float(value) if value else 0.0
            if subtotal > 0:
                category_ms[category] = subtotal
                explicit_ms += subtotal
        residual_ms = max(total_ms - explicit_ms, 0.0)
        if residual_ms > 0:
            category_ms[OTHER_LABEL] = residual_ms
    else:
        residual_ms = max(total_ms - (explicit_ms or 0.0), 0.0)

    return {
        "total": total_ms,
        "explicit": explicit_ms,
        "residual": residual_ms,
        "category_ms": category_ms,
        "category_pct": category_pct,
    }


def discover_cost_records(data_dir: str, dim: int) -> dict[tuple[int, ...], dict]:
    pattern = _grid_pattern(dim)
    records = {}
    for csv_path in glob.glob(os.path.join(data_dir, "cost_breakdown_*.csv")):
        match = pattern.search(os.path.basename(csv_path))
        if not match:
            continue
        grid = tuple(int(match.group(index)) for index in range(1, dim + 1))
        perstep_path = csv_path.replace("cost_breakdown_", "cost_perstep_")
        record = load_cost_record(csv_path, perstep_path)
        if record is None:
            continue
        records[grid] = record
    return records


def _save_figure(fig, output_path: str) -> None:
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".pdf", ".png"))
    plt.close(fig)


def plot_multigrid_summary(out_dir: str, spec: DimensionSpec, mode_tag: str, records: dict[tuple[int, ...], dict]) -> bool:
    if not records:
        return False

    grids = sorted(records, key=grid_cells)
    n_grids = len(grids)
    category_order = [
        category for category in list(CATEGORY_PREFIXES.keys()) + [OTHER_LABEL]
        if any(records[grid]["category_ms"].get(category, 0.0) > 0 for grid in grids)
    ]

    x = np.arange(n_grids, dtype=float)
    bar_width = 0.55

    # ── Figure 1: stacked bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n_grids + 2), 4.3))
    bottom = np.zeros(n_grids, dtype=float)
    for category in category_order:
        values = np.array([records[grid]["category_ms"].get(category, 0.0) for grid in grids], dtype=float)
        if np.allclose(values, 0.0):
            continue
        ax.bar(
            x, values, bar_width,
            bottom=bottom,
            color=CAT_COLOURS.get(category, CAT_COLOURS[OTHER_LABEL]),
            edgecolor="white",
            linewidth=0.5,
            label=category.replace("\n", " "),
        )
        bottom += values

    for i in range(n_grids):
        ax.text(x[i], bottom[i] + 0.02 * bottom.max(),
                f"{bottom[i]:.1f} ms", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([grid_label(grid) for grid in grids], rotation=20, ha="right")
    ax.set_ylabel("Median time per step (ms)")
    ax.set_title(f"Cost breakdown - {spec.benchmark_label} ({spec.label}, {mode_tag})")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, bottom.max() * 1.15)
    _save_figure(fig, os.path.join(out_dir, "cost_scaling_stacked.pdf"))
    _save_figure(fig, os.path.join(out_dir, f"cost_breakdown_{mode_tag}.pdf"))

    # ── Figure 2: per-category log-log scaling ─────────────────────────
    cells = np.array([grid_cells(grid) for grid in grids], dtype=float)
    totals = np.array([records[grid]["total"] for grid in grids], dtype=float)
    fig2, ax2 = plt.subplots(figsize=(6.8, 4.6))
    _markers = ["o", "s", "D", "^", "v", "p", "h", "X", "*", "P", "<", ">"]
    for i, category in enumerate(category_order):
        values = np.array([records[grid]["category_ms"].get(category, 0.0) for grid in grids], dtype=float)
        if np.allclose(values, 0.0) or not np.any(values > 0):
            continue
        ax2.loglog(
            cells, values,
            marker=_markers[i % len(_markers)], markersize=5,
            label=category.replace("\n", " "), linewidth=1.4,
            color=CAT_COLOURS.get(category, CAT_COLOURS[OTHER_LABEL]),
        )

    if len(cells) >= 2:
        x_ref = np.array([cells[0], cells[-1]], dtype=float)
        y0_proj = records[grids[0]]["category_ms"].get("Projection\n(pressure)", 0.0)
        y0 = 0.3 * y0_proj if y0_proj > 0 else totals[0]
        scale_lin = y0 * x_ref / x_ref[0]
        scale_nln = y0 * (x_ref / x_ref[0]) * np.log2(x_ref) / max(np.log2(x_ref[0]), 1.0)
        ax2.loglog(x_ref, scale_lin, "k--", alpha=0.35, linewidth=1.0, label=r"$\mathcal{O}(N)$")
        ax2.loglog(x_ref, scale_nln, "k:",  alpha=0.35, linewidth=1.0, label=r"$\mathcal{O}(N\log N)$")

        if np.all(totals > 0):
            A = np.column_stack([np.ones_like(cells), cells])
            coef, *_ = np.linalg.lstsq(A, totals, rcond=None)
            a_fit, b_fit = float(coef[0]), float(coef[1])
            if a_fit > 0 and b_fit > 0:
                x_dense = np.geomspace(cells[0], cells[-1], 64)
                ax2.loglog(
                    x_dense, a_fit + b_fit * x_dense,
                    color="#37474f", linestyle=(0, (4, 2)), linewidth=1.2, alpha=0.7,
                    label=(fr"$a + b\,N$  ($a={a_fit:.2f}$ ms, "
                           fr"$b={b_fit * 1e6:.2f}$ ns/cell)"),
                )

    axis_suffix = "N_x N_y" if spec.dim == 2 else "N_x N_y N_z"
    ax2.set_xlabel(f"Total cells  ${axis_suffix}$")
    ax2.set_ylabel("Median time per step (ms)")
    ax2.set_title(f"Computational scaling - {spec.benchmark_label} ({spec.label}, {mode_tag})")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7.5, ncol=2)
    _save_figure(fig2, os.path.join(out_dir, "cost_scaling_loglog.pdf"))
    _save_figure(fig2, os.path.join(out_dir, f"cost_scaling_loglog_{mode_tag}.pdf"))

    # ── Figure 3: percentage stacked bar ──────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(max(4.5, 1.2 * n_grids + 2), 4.0))
    bottoms3 = np.zeros(n_grids, dtype=float)
    cat_totals = np.zeros(n_grids, dtype=float)
    for category in category_order:
        cat_totals += np.array([records[grid]["category_ms"].get(category, 0.0) for grid in grids], dtype=float)

    for category in category_order:
        values = np.array([records[grid]["category_ms"].get(category, 0.0) for grid in grids], dtype=float)
        if np.allclose(values, 0.0):
            continue
        pcts = 100.0 * values / np.maximum(cat_totals, 1e-12)
        ax3.bar(x, pcts, bar_width, bottom=bottoms3,
                color=CAT_COLOURS.get(category, CAT_COLOURS[OTHER_LABEL]),
                edgecolor="white", linewidth=0.5,
                label=category.replace("\n", " "))
        for j in range(n_grids):
            if pcts[j] >= 8:
                ax3.text(x[j], bottoms3[j] + pcts[j] / 2,
                         f"{pcts[j]:.0f}%", ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
        bottoms3 += pcts

    ax3.set_xticks(x)
    ax3.set_xticklabels([grid_label(grid) for grid in grids], rotation=20, ha="right")
    ax3.set_ylabel("Fraction of step time (%)")
    ax3.set_title(f"Relative cost distribution - {spec.benchmark_label} ({spec.label}, {mode_tag})")
    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=7.5, ncol=1)
    _save_figure(fig3, os.path.join(out_dir, "cost_scaling_pct.pdf"))
    _save_figure(fig3, os.path.join(out_dir, f"cost_scaling_pct_{mode_tag}.pdf"))

    return True


def _plot_residual_vs_total(root_dir: str, spec: DimensionSpec, condition_ids: list[str], condition_breakdowns: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    plotted = False
    for condition in condition_ids:
        breakdown = condition_breakdowns.get(condition)
        if not breakdown:
            continue
        grids = sorted(breakdown, key=grid_cells)
        cells = np.array([grid_cells(grid) for grid in grids], dtype=float)
        totals = np.array([breakdown[grid]["total"] for grid in grids], dtype=float)
        residuals = np.array([
            breakdown[grid]["residual"] if breakdown[grid]["residual"] is not None else np.nan
            for grid in grids
        ], dtype=float)
        style = MODE_SPECS[condition]
        ax.loglog(
            cells,
            totals,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=6,
            color=style["color"],
            label=f"{style['short_label']} total",
        )
        finite = np.isfinite(residuals)
        if np.any(finite):
            ax.loglog(
                cells[finite],
                residuals[finite],
                linestyle=":",
                marker=style["marker"],
                linewidth=1.2,
                markersize=5,
                color=style["color"],
                alpha=0.9,
                label=f"{style['short_label']} residual",
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axis_suffix = "N_x N_y" if spec.dim == 2 else "N_x N_y N_z"
    ax.set_xlabel(f"Total cells  ${axis_suffix}$")
    ax.set_ylabel("Median time per step (ms)")
    ax.set_title(f"Total vs residual cost - {spec.benchmark_label} ({spec.label})")
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.92)
    _save_figure(fig, os.path.join(root_dir, "cost_residual_vs_total_conditions.pdf"))


def plot_mode_comparison(root_dir: str, spec: DimensionSpec, condition_ids: list[str]) -> bool:
    condition_breakdowns = {}
    condition_records = {}
    for condition in condition_ids:
        cond_dir = os.path.join(root_dir, condition)
        records = discover_cost_records(cond_dir, spec.dim)
        if not records:
            continue
        condition_breakdowns[condition] = records
        condition_records[condition] = {grid: record["total"] for grid, record in records.items()}

    if not condition_records:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    all_points = []
    for condition in condition_ids:
        records = condition_records.get(condition)
        if not records:
            continue
        grids = sorted(records, key=grid_cells)
        cells = np.array([grid_cells(grid) for grid in grids], dtype=float)
        totals = np.array([records[grid] for grid in grids], dtype=float)
        style = MODE_SPECS[condition]
        ax.loglog(
            cells,
            totals,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.8,
            color=style["color"],
            markeredgewidth=1.2,
            label=style["label"],
        )
        all_points.extend(zip(cells.tolist(), totals.tolist()))

    if len(all_points) >= 2:
        all_points.sort()
        x_ref = np.array([all_points[0][0], all_points[-1][0]], dtype=float)
        y_ref = all_points[0][1] * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e", linestyle=":", linewidth=1.0, alpha=0.7, label="O(N) reference")

    axis_suffix = "N_x N_y" if spec.dim == 2 else "N_x N_y N_z"
    ax.set_xlabel(f"Total cells  ${axis_suffix}$")
    ax.set_ylabel("Median time per step (ms)")
    ax.set_title(f"Cost scaling across solver modes ({spec.label}, {spec.benchmark_label})")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    _save_figure(fig, os.path.join(root_dir, "cost_scaling_loglog_conditions.pdf"))

    _plot_residual_vs_total(root_dir, spec, condition_ids, condition_breakdowns)

    if "python" not in condition_records:
        return True

    base_records = condition_records["python"]
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.3))
    plotted = False
    for condition in condition_ids:
        if condition == "python":
            continue
        records = condition_records.get(condition)
        if not records:
            continue
        shared = sorted(set(base_records) & set(records), key=grid_cells)
        if not shared:
            continue
        cells = np.array([grid_cells(grid) for grid in shared], dtype=float)
        speedup = np.array([base_records[grid] / records[grid] for grid in shared], dtype=float)
        style = MODE_SPECS[condition]
        ax2.semilogx(
            cells,
            speedup,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=6,
            color=style["color"],
            label=f"python / {condition}",
        )
        plotted = True

    if plotted:
        axis_suffix = "N_x N_y" if spec.dim == 2 else "N_x N_y N_z"
        ax2.set_xlabel(f"Total cells  ${axis_suffix}$")
        ax2.set_ylabel("Speed-up vs python")
        ax2.set_title(f"Mode speed-up ({spec.label}, {spec.benchmark_label})")
        ax2.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=1.0)
        ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        _save_figure(fig2, os.path.join(root_dir, "cost_scaling_speedup_conditions.pdf"))
    else:
        plt.close(fig2)

    return True