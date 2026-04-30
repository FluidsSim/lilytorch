#!/usr/bin/env python3
"""
Body-fraction cost analysis: narrow-band vs full-grid.

Sweeps grid resolution N at TWO FIXED domain sizes so that the linear
body coverage ``L_body / L_domain`` stays constant within each sweep
but differs between the two sweeps by ~ 2× (~ 70 % vs ~ 30 % along the
longitudinal axis, with matched transverse fractions because the 4:1:1
grid ratio carries (Lx, Ly, Lz) together).

For every (domain × N) combination the script runs
``run_cost_analysis.py`` TWICE — once with every narrow-band /
streaming flag on (the production ``nbforces_opt`` flag set), once
with every narrow-band flag off — so the per-category cost curves can
be compared directly.

Motivation
----------
Narrow-band (union-AABB) kernels are expected to be a large win when
the swimmer occupies a small fraction of the fluid domain (because the
union sub-block then covers a small fraction of cells) but to approach
or even fall behind the full-grid baseline as the body fills the
domain and the dynamic-shape recompile / crop bookkeeping starts to
dominate.  The default cost analysis scales the domain with N so the
body fraction shrinks with N — it cannot answer this question.  This
runner fixes the domain so the fraction is held constant as N grows.

Default sweep
-------------
* ``small`` domain — ``Lx = 1.2 m``, ``Ly = Lz = 0.30 m`` (4:1:1).
  Body length 0.81 m fills ~ 68 % of x; undulation sweep fills ~ 67 %
  of y/z.  ⇒ "body comprises ~ 70 % of the domain".
* ``large`` domain — ``Lx = 2.7 m``, ``Ly = Lz = 0.675 m`` (4:1:1).
  Body fills ~ 30 % of x; undulation fills ~ 30 % of y/z.
  ⇒ "body comprises ~ 30 % of the domain".

Each domain is swept over **5 N points** (4:1:1 cubic-ish triplets);
combined with the default {nboff, nbcrop, nbon} mode toggle this gives
**30 runs** total (5 N × 2 domains × 3 NB modes).  Expected outcome: in
the small domain NB-on ≈ NB-off (cropping does not save much because
the body already covers ~ 70 % of the cells); in the large domain
NB-on ≪ NB-off because the body union covers only ~ 1/30 of the cells.

Usage
-----
    # Default shared-even sweep
    python run_body_fraction_analysis.py

    # Broader shared sweep (same grids for both domains)
    python run_body_fraction_analysis.py --grid_policy shared_broad

    # Power-of-two sweep (per-domain preset grids)
    python run_body_fraction_analysis.py --grid_policy power2

    # Single domain (e.g. only the tight one)
    python run_body_fraction_analysis.py --domains small

    # Custom grids (applied to every domain — must share Ny/Nx, Nz/Nx)
    python run_body_fraction_analysis.py --grids 128:32:32,256:64:64

Output
------
CSVs land in ``figures/body_fraction/`` with the tag
``{Nx}x{Ny}x{Nz}_{domain}_{nb}`` where ``nb`` is one of the selected
narrow-band modes.  The
companion ``plot_body_fraction.py`` reads those CSVs and produces
log-log cost curves per domain plus a narrow-band vs full-grid
speed-up summary.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_cost_analysis.py")
PLOT_SCRIPT = os.path.join(SCRIPT_DIR, "plot_body_fraction.py")

# ─────────────────────────────────────────────────────────────────────
# Domain presets
# ─────────────────────────────────────────────────────────────────────
# Body geometry (free-swimming 1guilla): length ≈ 0.81 m along x, thin
# transverse section (~0.05-0.10 m sweep during undulation, so total
# y/z extent ≈ 0.20 m once both sides are accounted for).
#
# ``Lx_fixed`` is the x-extent of the fluid domain.  Ly and Lz are
# derived from the Ny/Nx and Nz/Nx grid ratios so dx stays isotropic.
# All grid triplets below keep the 4:1:1 ratio, so
#     Ly = Lx_fixed * Ny/Nx = Lx_fixed / 4
#     Lz = Lx_fixed * Nz/Nx = Lx_fixed / 4
#
# Body coverage (linear, along the longitudinal axis):u
#
# "small"  : Lx = 1.2 m   (body fills ~ 68 % of x; undulation fills
#                          ~ 67 % of y/z)  →  Ly = Lz = 0.30 m
#                          → "body comprises ~ 70 % of the domain"
#
# "large"  : Lx = 2.7 m   (body fills ~ 30 % of x; undulation fills
#                          ~ 30 % of y/z)  →  Ly = Lz = 0.675 m
#                          → "body comprises ~ 30 % of the domain"
#
# Per-domain grid lists
# ---------------------
# The BDIM ↔ MuJoCo coupling is only stable for a bounded range of dx,
# roughly  dx ∈ [0.003, 0.013] m.  Grids whose dx = Lx / Nx falls
# outside that band fail for physics reasons with BOTH narrow-band ON
# and OFF identically, so they give no extra information about the
# narrow-band optimisation.  Each domain therefore carries 5 N points
# spaced inside the stable band.  Note: not all sizes are strict
# powers of 2 — the multigrid Poisson smoother handles arbitrary even
# extents (it stops coarsening once any axis falls to ≤ 8 cells), so
# this only costs at deep V-cycle levels and does not affect the cost
# breakdown that is being measured.
#
# Expected outcome:
#   * "small" domain — body ~ 70 %, very few cells outside the body-
#     union AABB.  Narrow-band trims little work: NB-on ≈ NB-off, with
#     NB-on possibly slower at small N due to dynamic-shape bookkeeping.
#   * "large" domain — body ~ 30 %, the body-union AABB covers ≲ 1/30
#     of the cells.  Narrow-band kernels skip the bulk of the grid →
#     large NB speed-up that grows with N.
# Both domains use the SAME 5 N points so the log-log plots overlay
# cell-for-cell.  The Nx range is constrained to the intersection of
# each domain's stable dx band:
#     small (Lx = 1.2 m)  : dx > 3 mm  →  Nx ≤ 400
#     large (Lx = 2.7 m)  : dx < 13 mm →  Nx ≥ 208
# Five values divisible by 4 (so Ny = Nz = Nx/4 is integer) inside the
# intersection [216, 384]:  Nx ∈ {224, 256, 288, 320, 352}.
SHARED_GRIDS = [
    (224,  56,  56),   #   702 k cells
    (256,  64,  64),   #  1.05 M cells
    (288,  72,  72),   #  1.49 M cells
    (320,  80,  80),   #  2.05 M cells
    (352,  88,  88),   #  2.73 M cells
]

# Broader common-band sweep for fixed-domain comparisons.  These are the
# same grids for both domains and span the validated shared dx range
# from Nx = 208 (large-domain coarse edge) to Nx = 400 (small-domain
# fine edge), reaching 4.0 M cells at the upper end.
SHARED_BROAD_GRIDS = [
    (208,  52,  52),   #   562 k cells
    (224,  56,  56),   #   702 k cells
    (240,  60,  60),   #   864 k cells
    (256,  64,  64),   #  1.05 M cells
    (272,  68,  68),   #  1.26 M cells
    (288,  72,  72),   #  1.49 M cells
    (304,  76,  76),   #  1.76 M cells
    (320,  80,  80),   #  2.05 M cells
    (336,  84,  84),   #  2.37 M cells
    (352,  88,  88),   #  2.73 M cells
    (368,  92,  92),   #  3.11 M cells
    (384,  96,  96),   #  3.54 M cells
    (400, 100, 100),   #  4.00 M cells
]

POWER2_GRIDS = {
    "small": [
        (128,  32,  32),   # dx = 9.38 mm
        (256,  64,  64),   # dx = 4.69 mm
    ],
    "large": [
        (256,  64,  64),   # dx = 10.55 mm
        (512, 128, 128),   # dx = 5.27 mm
    ],
}

DOMAIN_PRESETS = {
    "small": {
        "Lx_fixed": 1.2,
        "label":    "small (body ≈ 70 %)",
        "note":     "Ly = Lz = Lx/4 = 0.30 m  |  body length / Lx ≈ 68 %",
        # dx = 1.2 / Nx ∈ {5.36, 4.69, 4.17, 3.75, 3.41} mm  – all in band.
        "grids": list(SHARED_GRIDS),
        "power2_grids": list(POWER2_GRIDS["small"]),
    },
    "large": {
        "Lx_fixed": 2.7,
        "label":    "large (body ≈ 30 %)",
        "note":     "Ly = Lz = Lx/4 = 0.675 m  |  body length / Lx ≈ 30 %",
        # dx = 2.7 / Nx ∈ {12.05, 10.55, 9.38, 8.44, 7.67} mm  – all in band.
        "grids": list(SHARED_GRIDS),
        "power2_grids": list(POWER2_GRIDS["large"]),
    },
}

# ``--grids`` override (if supplied) is applied to every domain and
# still validated to keep the same Ny/Nx, Nz/Nx ratio across the sweep.

# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Body-fraction cost analysis (narrow-band vs full-grid)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""\
Grid presets:
    shared  → same grids for both domains:
                        224x56x56, 256x64x64, 288x72x72, 320x80x80, 352x88x88
    shared_broad  → broader same-grid sweep for both domains:
                        208x52x52, 224x56x56, ..., 384x96x96, 400x100x100
    power2  → power-of-two grids per domain:
                        small: 128x32x32, 256x64x64
                        large: 256x64x64, 512x128x128

All preset grids enforce the 4:1:1 ratio (Ly = Lz = Lx / 4) so dx is
isotropic and (Lx, Ly, Lz) stays fixed within each domain sweep.
""",
)
parser.add_argument(
    "--domains", type=str, default="small,large",
    help="Comma-separated domain presets (default: small,large)")
parser.add_argument(
    "--grids", type=str, default=None,
    help="Comma-separated Nx:Ny:Nz triplets used for EVERY domain. "
         "When omitted, each domain uses its own stable grid list "
         "(see DOMAIN_PRESETS). Ratio Nx:Ny:Nz MUST be 4:1:1 so dx "
         "stays isotropic.")
parser.add_argument(
    "--grid_policy", type=str, default="shared",
    choices=["shared", "shared_broad", "power2"],
    help="Preset grid family used when --grids is omitted: "
         "'shared' uses the same even grids for both domains; "
         "'shared_broad' uses a denser common-band sweep for both domains; "
         "'power2' uses per-domain power-of-two grids only.")
parser.add_argument(
    "--n_steps", type=int, default=50,
    help="Measured steps per run (default: 50)")
parser.add_argument(
    "--precompile", type=int, default=30,
    help="Pre-compilation steps per run (default: 30)")
parser.add_argument(
    "--discard_first", type=int, default=3,
    help="Discard first N timed steps (default: 3)")
parser.add_argument(
    "--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument(
    "--out_dir", type=str, default=None,
    help="Output directory (default: figures/body_fraction/ here)")
parser.add_argument(
    "--dry-run", action="store_true",
    help="Print commands without executing")
parser.add_argument(
    "--skip-plots", action="store_true",
    help="Skip final plot generation")
parser.add_argument(
    "--continue-on-error", action="store_true",
    help="Continue to next run if one fails")
parser.add_argument(
    "--nb_modes", type=str, default="nboff,nbcrop,nbon",
    help="Which narrow-band modes to measure. Each mode is a different "
         "subset of solver flags:\n"
         "  nboff  – every flag off (true baseline)\n"
         "  nbcrop – only the 3 AABB-cropping flags on "
         "(force_shared_union, mu_normals_union, bdim_union)\n"
         "  nbon   – every narrow-band / streaming flag on "
         "(matches the production ``nbforces_opt`` set)\n"
         "Default measures all three of nboff, nbcrop, nbon so the plot "
         "decomposes the combined win into 'cropping alone' vs "
         "'cropping + streaming/batching added on top' (cropping is "
         "body-fraction dependent; streaming/batching is a uniform "
         "launch-count reduction).")
args = parser.parse_args()

# ── Parse domains ────────────────────────────────────────────────────
domains = [d.strip() for d in args.domains.split(",") if d.strip()]
for d in domains:
    if d not in DOMAIN_PRESETS:
        print(f"ERROR: Unknown domain preset '{d}'. "
              f"Valid: {list(DOMAIN_PRESETS)}")
        sys.exit(1)

# ── Parse grids ──────────────────────────────────────────────────────
# If --grids is supplied the same list is used for every domain;
# otherwise each domain uses its own per-preset stable grid list.
def _parse_grid_triplets(s):
    out = []
    for triplet in s.split(","):
        parts = triplet.strip().split(":")
        if len(parts) != 3:
            print(f"ERROR: Invalid grid triplet '{triplet}'")
            sys.exit(1)
        out.append(tuple(int(p) for p in parts))
    return out


def _validate_grids(grid_list, ctx):
    for nx, ny, nz in grid_list:
        if nx * ny == 0 or nx * nz == 0:
            print(f"ERROR ({ctx}): zero dimension in grid ({nx},{ny},{nz})")
            sys.exit(1)
    ratios = {(ny / nx, nz / nx) for nx, ny, nz in grid_list}
    if len(ratios) != 1:
        print(f"ERROR ({ctx}): all grids must share the same Ny/Nx and "
              "Nz/Nx ratios so (Lx, Ly, Lz) stays constant.")
        print(f"  got grids: {grid_list}")
        sys.exit(1)


if args.grids is not None:
    override = _parse_grid_triplets(args.grids)
    _validate_grids(override, "--grids")
    override.sort(key=lambda g: g[0] * g[1] * g[2])
    grids_per_domain = {d: list(override) for d in domains}
else:
    grids_per_domain = {}
    for d in domains:
        if args.grid_policy == "power2":
            g = list(DOMAIN_PRESETS[d].get("power2_grids", []))
        elif args.grid_policy == "shared_broad":
            g = list(SHARED_BROAD_GRIDS)
        else:
            g = list(DOMAIN_PRESETS[d].get("grids", []))
        if not g:
            print(f"ERROR: domain '{d}' has no default grid list; "
                  "supply --grids explicitly.")
            sys.exit(1)
        _validate_grids(g, f"domain '{d}'")
        g.sort(key=lambda t: t[0] * t[1] * t[2])
        grids_per_domain[d] = g

# ── Parse nb_modes ───────────────────────────────────────────────────
# Each mode maps to a list of flags appended to the run_cost_analysis.py
# command.  Modes are chosen to factor the combined narrow-band win into
# the two functional groups whose effects scale very differently:
#   * cropping  — body-fraction dependent (wins on sparse domains)
#   * streaming/batching  — body-fraction independent (uniform launch
#                           reduction, plus fused-CUDA SDF/forces).
# ``nbon`` therefore mirrors the production ``nbforces_opt`` set used by
# ``run_scaling_conditions_pipeline.py``.
NB_MODE_FLAGS = {
    "nboff":  [],                              # true baseline
    "nbcrop": ["--force_shared_union",         # AABB cropping only
               "--mu_normals_union",
               "--bdim_union"],
    "nbon":   ["--force_shared_union",         # production set (nbforces_opt)
               "--mu_normals_union",
               "--bdim_union",
               "--force_narrow_batch",
               "--streaming_sdf_3d",
               "--streaming_forces_3d"],
}

nb_modes = [m.strip() for m in args.nb_modes.split(",") if m.strip()]
for m in nb_modes:
    if m not in NB_MODE_FLAGS:
        print(f"ERROR: unknown nb_mode '{m}'. "
              f"Use one of {list(NB_MODE_FLAGS)}.")
        sys.exit(1)

# ── Output directory ─────────────────────────────────────────────────
if args.out_dir is None:
    args.out_dir = os.path.join(SCRIPT_DIR, "figures", "body_fraction")
args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════
total_runs = sum(len(grids_per_domain[d]) for d in domains) * len(nb_modes)
print("\n" + "=" * 72)
print("  Body-Fraction Cost Analysis — Narrow-Band vs Full-Grid")
print("=" * 72)
print(f"  Grid policy  : {args.grid_policy}")
print(f"  Domains      : {', '.join(domains)}")
for d in domains:
    p = DOMAIN_PRESETS[d]
    gstr = ", ".join(f"{nx}×{ny}×{nz}" for nx, ny, nz in grids_per_domain[d])
    dxs  = ", ".join(f"{p['Lx_fixed'] / nx * 1000:.2f}"
                     for nx, _, _ in grids_per_domain[d])
    print(f"    {d:<6s}: Lx={p['Lx_fixed']:.2f} m  ({p['note']})")
    print(f"             grids = {gstr}   dx = {dxs} mm")
print(f"  NB modes     : {', '.join(nb_modes)}")
print(f"  Steps/run    : {args.n_steps} measured + {args.precompile} precompile")
print(f"  Total runs   : {total_runs}")
print(f"  Device       : {args.device.upper()}")
print(f"  Output       : {args.out_dir}")
print(f"  Timestamp    : {datetime.now().isoformat()}")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
# Run every (domain, grid, nb_mode)
# ═══════════════════════════════════════════════════════════════════════
results = []        # list of dicts: domain, grid, nb, rc, elapsed, csv
failed  = []
python_exe = sys.executable
wall_start = time.time()

run_idx = 0
for domain in domains:
    p = DOMAIN_PRESETS[domain]
    for (nx, ny, nz) in grids_per_domain[domain]:
        for nb in nb_modes:
            run_idx += 1
            suffix = f"_{domain}_{nb}"
            tag = f"{nx}x{ny}x{nz}{suffix}"
            n_cells = nx * ny * nz
            header = (f"\n{'─' * 72}\n"
                      f"  [{run_idx}/{total_runs}]  domain={domain}  "
                      f"grid={nx}×{ny}×{nz}  ({n_cells:,} cells)  "
                      f"nb={nb}\n{'─' * 72}")
            print(header)

            cmd = [
                python_exe, SINGLE_RUN_SCRIPT,
                "--Nx", str(nx),
                "--Ny", str(ny),
                "--Nz", str(nz),
                "--n_steps", str(args.n_steps),
                "--precompile", str(args.precompile),
                "--discard_first", str(args.discard_first),
                "--save_every", "9999",
                "--device", args.device,
                "--out_dir", args.out_dir,
                "--Lx_fixed", str(p["Lx_fixed"]),
                "--tag_suffix", suffix,
            ]
            cmd.extend(NB_MODE_FLAGS[nb])

            print(f"  CMD: {' '.join(cmd)}")

            if args.dry_run:
                print("  [DRY RUN] Skipping execution.")
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, text=True)
            elapsed = time.time() - t0

            csv_path = os.path.join(
                args.out_dir, f"cost_breakdown_{tag}.csv")
            rec = {
                "domain":  domain,
                "grid":    (nx, ny, nz),
                "nb":      nb,
                "rc":      proc.returncode,
                "elapsed": elapsed,
                "csv":     csv_path if os.path.isfile(csv_path) else "",
            }
            results.append(rec)

            status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
            print(f"  → {status}  ({elapsed:.1f} s)")

            if proc.returncode != 0:
                failed.append(rec)
                if not args.continue_on_error:
                    print("\n  Aborting sweep (use --continue-on-error to "
                          "run remaining combinations).")
                    break
        else:
            continue
        break
    else:
        continue
    break


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
total_wall = time.time() - wall_start
print("\n" + "=" * 72)
print("  BODY-FRACTION SWEEP SUMMARY")
print("=" * 72)
print(f"{'Domain':<8} {'Grid':<16} {'NB':<6} {'Cells':>12}  {'Status':<10} "
      f"{'Wall':>8}  CSV")
print("-" * 72)
for r in results:
    nx, ny, nz = r["grid"]
    grid_str = f"{nx}×{ny}×{nz}"
    cells = nx * ny * nz
    status = "OK" if r["rc"] == 0 else f"FAIL({r['rc']})"
    csv_str = "yes" if r["csv"] else "no"
    print(f"{r['domain']:<8} {grid_str:<16} {r['nb']:<6} {cells:>12,}  "
          f"{status:<10} {r['elapsed']:>6.1f} s  {csv_str}")
print("=" * 72)
print(f"  Total wall-time: {total_wall:.1f} s ({total_wall / 60:.1f} min)")
if failed:
    print(f"  WARNING: {len(failed)} run(s) failed")


# ═══════════════════════════════════════════════════════════════════════
# Combined plots
# ═══════════════════════════════════════════════════════════════════════
if args.skip_plots or args.dry_run:
    print("\n  Skipping plots.")
    sys.exit(0 if not failed else 1)

if not os.path.isfile(PLOT_SCRIPT):
    print(f"\n  WARNING: plot script not found: {PLOT_SCRIPT}")
    sys.exit(0 if not failed else 1)

print(f"\n{'─' * 72}")
print("  Generating body-fraction plots…")
print(f"{'─' * 72}")

rc_plot = subprocess.run(
    [python_exe, PLOT_SCRIPT, "--data_dir", args.out_dir],
    text=True,
).returncode

print("\n" + "╔" + "═" * 70 + "╗")
print("║  Body-fraction cost analysis complete." + " " * 31 + "║")
print("║" + " " * 70 + "║")
print(f"║  CSVs + figures: {args.out_dir:<51s} ║")
print("╚" + "═" * 70 + "╝")

sys.exit(rc_plot if rc_plot != 0 else (1 if failed else 0))
