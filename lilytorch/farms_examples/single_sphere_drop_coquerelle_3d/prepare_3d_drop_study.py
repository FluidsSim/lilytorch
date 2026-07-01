#!/usr/bin/env python3
"""Generate 3-D Coquerelle-Cottet dropped-sphere study configs.

Creates one simulation config and one experiment config per (diameter, nu)
pair, with isolated output directories under the study root.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_SIM = ROOT / "simulation_config.yaml"
BASE_EXP = ROOT / "experiment_config.yaml"

OUT_ROOT = Path("/data/andreaferrario/ns_data/coquerelle_cottet_3d_drop")
DT = 1.0e-4
T_END = 3.0

CASES = [
    {"diameter": 0.2, "nu": 0.02},
    {"diameter": 0.2, "nu": 0.05},
    {"diameter": 0.2, "nu": 0.10},
    {"diameter": 0.3, "nu": 0.02},
    {"diameter": 0.3, "nu": 0.05},
    {"diameter": 0.3, "nu": 0.10},
]


def nt_from_dt(dt: float) -> int:
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


def fmt_float(value: float) -> str:
    return str(value).replace(".", "p")


def case_name(diameter: float, nu: float) -> str:
    return f"d{fmt_float(diameter)}_nu{fmt_float(nu)}"


def sphere_animat_name(diameter: float) -> str:
    return f"sphere_D{fmt_float(diameter)}.yaml"


def build_case(sim_base: dict, exp_base: dict, diameter: float, nu: float) -> tuple[str, Path, Path]:
    name = case_name(diameter, nu)
    nt = nt_from_dt(DT)

    sim_cfg = deepcopy(sim_base)
    exp_cfg = deepcopy(exp_base)

    sim_cfg["runtime"]["n_iterations"] = nt
    sim_cfg["runtime"]["buffer_size"] = nt
    sim_cfg["physics"]["timestep"] = DT

    fluid_ext = get_extension(sim_cfg, "lilytorch.integration.extensions.FluidExtension")
    bdim_yaml = fluid_ext["config"]["bdim_yaml"]
    solver = bdim_yaml["solver"]
    solver["dt"] = DT
    solver["nt"] = nt
    solver["nu"] = float(nu)

    case_out = OUT_ROOT / name
    case_out.mkdir(parents=True, exist_ok=True)

    bdim_yaml.setdefault("output", {})
    bdim_yaml["output"]["save_path"] = ""
    bdim_yaml["output"]["existing_folder"] = str(case_out)

    logger_ext = get_extension(sim_cfg, "farms_core.simulation.extensions.ExperimentLogger")
    logger_ext["config"]["log_path"] = str(case_out / "output")

    sim_name = f"simulation_config_{name}.yaml"
    exp_name = f"experiment_config_{name}.yaml"

    exp_cfg["simulation"] = sim_name
    exp_cfg["animats"] = [sphere_animat_name(diameter)]

    sim_path = ROOT / sim_name
    exp_path = ROOT / exp_name
    dump_yaml(sim_path, sim_cfg)
    dump_yaml(exp_path, exp_cfg)

    return name, sim_path, exp_path


def main() -> None:
    sim_base = load_yaml(BASE_SIM)
    exp_base = load_yaml(BASE_EXP)

    print("Generating 3-D dropped-sphere study configs...")
    generated = []
    for case in CASES:
        generated.append(build_case(sim_base, exp_base, case["diameter"], case["nu"]))

    print("Done. Generated files:")
    for name, sim_path, exp_path in generated:
        print(f"  - {name}\n    sim: {sim_path.name}\n    exp: {exp_path.name}")

    print("\nRun one case with:")
    print("  sh run_case.sh <case_name>")
    print("\nRun all cases with:")
    print("  sh run_all_cases.sh")


if __name__ == "__main__":
    main()
