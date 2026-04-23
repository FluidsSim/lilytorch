#!/usr/bin/env python3
"""Generate matched Heun/Euler Coquerelle experiment configs.

This script creates one simulation config and one experiment config per case,
keeping the same physical end time across different timesteps.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_SIM = ROOT / "simulation_config.yaml"
BASE_EXP = ROOT / "experiment_config.yaml"

OUT_ROOT = Path("/data/andreaferrario/ns_data/coquerelle_heun_euler_study")
T_END = 0.65  # seconds

CASES = [
    {"name": "heun_dt_0p00005", "heun": True, "dt": 5.0e-5},
    {"name": "heun_dt_0p000025", "heun": True, "dt": 2.5e-5},
    {"name": "euler_dt_0p00005", "heun": False, "dt": 5.0e-5},
    {"name": "euler_dt_0p000025", "heun": False, "dt": 2.5e-5},
    {"name": "euler_dt_0p0000125", "heun": False, "dt": 1.25e-5},
]


def nt_from_dt(dt: float) -> int:
    """Return runtime iterations to match T_END at the given dt."""
    return int(round(T_END / dt)) + 1


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_extension(sim_cfg: dict, loader: str) -> dict:
    for ext in sim_cfg.get("extensions", []):
        if ext.get("loader") == loader:
            return ext
    raise KeyError(f"Extension loader '{loader}' not found")


def build_case(sim_base: dict, exp_base: dict, case: dict) -> tuple[Path, Path]:
    name = case["name"]
    dt = float(case["dt"])
    heun = bool(case["heun"])
    nt = nt_from_dt(dt)

    sim_cfg = deepcopy(sim_base)
    exp_cfg = deepcopy(exp_base)

    sim_cfg["runtime"]["n_iterations"] = nt
    sim_cfg["runtime"]["buffer_size"] = nt
    sim_cfg["physics"]["timestep"] = dt

    fluid_ext = get_extension(sim_cfg, "lilytorch.integration.extensions.FluidExtension")
    bdim_yaml = fluid_ext["config"]["bdim_yaml"]
    solver = bdim_yaml["solver"]

    # Keep solver values aligned with runtime even though FluidExtension
    # enforces physics/runtime values at launch.
    solver["dt"] = dt
    solver["nt"] = nt
    solver["heun"] = heun

    case_out = OUT_ROOT / name
    case_out.mkdir(parents=True, exist_ok=True)

    bdim_yaml.setdefault("output", {})
    bdim_yaml["output"]["save_path"] = ""
    bdim_yaml["output"]["existing_folder"] = str(case_out)

    logger_ext = get_extension(sim_cfg, "farms_core.simulation.extensions.ExperimentLogger")
    logger_ext["config"]["log_path"] = str(case_out / "output")

    sim_name = f"simulation_config_{name}.yaml"
    exp_name = f"experiment_config_{name}.yaml"
    sim_path = ROOT / sim_name
    exp_path = ROOT / exp_name

    exp_cfg["simulation"] = sim_name

    dump_yaml(sim_path, sim_cfg)
    dump_yaml(exp_path, exp_cfg)

    return sim_path, exp_path


def main() -> None:
    sim_base = load_yaml(BASE_SIM)
    exp_base = load_yaml(BASE_EXP)

    print("Generating matched Heun/Euler study configs...")
    generated = []
    for case in CASES:
        sim_path, exp_path = build_case(sim_base, exp_base, case)
        generated.append((case, sim_path, exp_path))

    print("Done. Generated files:")
    for case, sim_path, exp_path in generated:
        nt = nt_from_dt(case["dt"])
        print(
            f"  - {case['name']}: dt={case['dt']:.8f}, nt={nt}, heun={case['heun']}\n"
            f"    sim: {sim_path.name}\n"
            f"    exp: {exp_path.name}"
        )

    print("\nRun a case with:")
    print("  sh run_heun_euler_case.sh <case_name>")
    print("\nAvailable case names:")
    for case in CASES:
        print(f"  - {case['name']}")


if __name__ == "__main__":
    main()
