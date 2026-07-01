"""Iso-speed gait comparison for the position-controlled zebrafish.

Tests the hypothesis: *to reach a given swim speed, is it cheaper (lower cost of
transport) to use tail-dominated kinematics at a higher tail-beat frequency than
full-body kinematics at a lower frequency?*

For each target speed and each body shape it **solves** for the tail-beat
frequency that makes the fish swim at that speed (speed is ~linear in frequency
for a fixed shape, so a calibrated v(f) guess + at most one rescaling iteration
hits the target to within tolerance), runs the coupled CFD+MuJoCo sim, and reads
back two purely fluid-side cost-of-transport metrics from ``diagnostics.h5``:

    CoT_diss  = E_diss              / (m g d)   -- irreversible viscous loss only
    CoT_fluid = (E_diss + dE_k)     / (m g d)   -- + kinetic energy left in the wake

(No mechanical/actuator power: for a position-controlled body the realised joint
torque is a tracking/constraint torque, not a physical actuator power, so the
joint-based CoT is meaningless.  See verify_energy_balance.py.)

The two shapes are then compared **at matched speed** (no interpolation): for each
target speed there is one tail point and one full point at the same speed, so the
vertical gap in CoT *is* the test.

Caveat: the runs blow up at t~0.4-0.65 s, so this is a *quasi-steady* comparison
in a fixed pre-blow-up window, identical for every run (fair, if not converged).

Usage
-----
    python run_iso_speed.py                 # full sweep, default speeds
    python run_iso_speed.py --speeds 9      # single target speed
    python run_iso_speed.py --analyze-only  # re-analyse existing runs, no sims
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import traceback

import h5py
import numpy as np

from farms_core.sensors.sensor_convention import sc
from lilytorch.util.metrics import compute_speed_PCA
from lilytorch.util.paths import save_path as NS_DATA_ROOT

from controller_parameters import (
    SLOW_SWIMMING_CONTROLLER_PARAMETERS as _SLOW,
    FAST_SWIMMING_CONTROLLER_PARAMETERS as _FAST,
)
from gen_configs_pd_3d_slow_fast import SimConfig as ZFConfig

F_SLOW = _SLOW["frequency"]   # ~11.4 Hz, natural freq of the slow (tail) gait
F_FAST = _FAST["frequency"]   # ~20.0 Hz, natural freq of the fast (full-body) gait
S0 = 0.00025                  # base kinematics_sampling (gaits natural at S0)
G = 9.81
STACK_ROOT = "zebrafish_iso_speed"

# (control mode / xlsx, natural frequency) per shape family.
SHAPES = {
    "tail": ("slow", F_SLOW),   # slow-shape xlsx = tail-dominated kinematics
    "full": ("fast", F_FAST),   # fast-shape xlsx = full-body undulation
}

# Calibrated v(f) = m*f + b per shape, from the freq-cross runs (BL/s vs Hz).
# Only the initial guess; the closed loop corrects any error.
VF_MODEL = {"tail": (0.757, -3.609), "full": (0.998, -4.000)}

# Quasi-steady measurement window [s] (fixed for every run -> fair comparison).
WIN = (0.20, 0.30)
# Each sim is run for a bit beyond the window so the window is covered.
T_RUN = 0.38


def _freq_for_speed(shape: str, v_target: float) -> float:
    m, b = VF_MODEL[shape]
    return (v_target - b) / m


class IsoConfig(ZFConfig):
    """ZF position-control config pinned to one (shape, frequency); no video."""

    def __init__(self, shape: str, freq: float, tag: str):
        super().__init__()
        mode, f_nat = SHAPES[shape]
        cp = self.animats_pars[0]["control_pars"]
        cp["mode"] = mode
        cp["kinematics_sampling"] = S0 * f_nat / freq

        # Persist diagnostics (KE + dissipation) but no field frames / forces.
        self.save = True
        self.save_frames = False
        self.save_every = 10_000_000
        self.diagnostics_every = 2
        self.save_drags = False
        self.headless = True

        self.stack_folder = f"{STACK_ROOT}/{tag}"

        # Smaller timestep at higher frequency (dt ~ 1/f beyond F_SLOW) so the
        # faster gaits stay stable through the measurement window.
        dt = 0.0005 * min(1.0, F_SLOW / freq)
        self.timestep = dt
        self.bdim_dt = dt
        self.n_iterations = int(np.ceil(T_RUN / dt)) + 1
        self.bdim_nt = self.n_iterations + 1

    def extra_simulation_extensions(self, output_folder):
        # Batch sweep: no viewer, no camera recordings.
        return []


def _latest_run_dir(tag: str):
    base = os.path.join(NS_DATA_ROOT, STACK_ROOT, tag)
    subs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    for d in reversed(subs):
        if os.path.exists(os.path.join(d, "output", "simulation.hdf5")):
            return d
    return None


def _cumtrapz0(y, x):
    if len(y) < 2:
        return np.zeros_like(y, dtype=float)
    incr = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(incr)])


def _read_rho(run_dir, default=1000.0):
    import yaml
    path = os.path.join(run_dir, "parameters.yaml")
    if os.path.exists(path):
        try:
            with open(path) as f:
                pars = yaml.unsafe_load(f)
            return float(pars["solver"]["rho"])
        except Exception:
            pass
    return default


def measure(run_dir, bl, win=WIN):
    """Return dict with achieved speed [BL/s] and CoT_diss / CoT_fluid in *win*.

    Both CoTs use cumulative energy over distance, E(t)/(m g d(t)), averaged over
    the window.  The window is truncated below any blow-up time so the explosion
    never enters the integral.  ``ok`` flags whether the window is usable.
    """
    rho = _read_rho(run_dir)
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        link = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        masses = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["masses"])
        dt = float(np.array(f["timestep"]))
        times = (np.array(f["times"])[: link.shape[0]] if "times" in f
                 else dt * np.arange(link.shape[0]))
    vel = link[:, :, sc.link_com_velocity_lin_x: sc.link_com_velocity_lin_z + 1]
    pos = link[:, :, sc.link_com_position_x: sc.link_com_position_z + 1]
    v_fwd, _ = compute_speed_PCA(pos, vel)
    v_fwd = np.asarray(v_fwd)
    m_total = float(masses.sum())
    com_xy = (masses[None, :, None] * pos[:, :, :2]).sum(axis=1) / m_total

    diag = os.path.join(run_dir, "diagnostics.h5")
    with h5py.File(diag, "r") as f:
        ke_raw = np.array(f["kinetic_energy"])
        diss_raw = np.array(f["dissipation_rate"])

    ke_valid = np.isfinite(ke_raw)
    ke_idx = np.where(ke_valid)[0]
    ke_t = ke_idx * dt
    ke_J = rho * ke_raw[ke_valid]
    diss_valid = np.isfinite(diss_raw)
    diss_idx = np.where(diss_valid)[0]
    diss_t = diss_idx * dt
    diss_W = rho * diss_raw[diss_valid]

    # Blow-up detection: first KE sample exceeding 10x the early-window median.
    base = np.median(ke_J[ke_t <= win[0]]) if (ke_t <= win[0]).any() else ke_J[0]
    blow = ke_t[ke_J > 10.0 * max(base, 1e-30)]
    t_blow = float(blow[0]) if blow.size else np.inf
    w1 = min(win[1], t_blow - 2 * dt)
    w0 = win[0]
    ok = w1 - w0 > 0.02   # need a usable averaging window

    # Cumulative fluid energies on the diagnostic grid, then onto sim times.
    e_diss = _cumtrapz0(diss_W, diss_t)
    e_diss_sim = np.interp(times, diss_t, e_diss, left=0.0, right=e_diss[-1])
    dek_sim = np.interp(times, ke_t, ke_J - ke_J[0], left=0.0, right=(ke_J - ke_J[0])[-1])
    e_fluid_sim = e_diss_sim + dek_sim

    d_xy = np.sqrt((com_xy[:, 0] - com_xy[0, 0]) ** 2 +
                   (com_xy[:, 1] - com_xy[0, 1]) ** 2)
    far = d_xy > 1e-4
    n = min(len(times), len(v_fwd))
    sel = (times[:n] >= w0) & (times[:n] <= w1) & far[:n]

    if not sel.any():
        return dict(ok=False, speed=float("nan"), cot_diss=float("nan"),
                    cot_fluid=float("nan"), t_blow=t_blow, w1=w1, run_dir=run_dir)

    speed = float(np.mean(v_fwd[:n][sel])) / bl
    cot_diss = float(np.mean((e_diss_sim[:n] / (m_total * G * d_xy[:n]))[sel]))
    cot_fluid = float(np.mean((e_fluid_sim[:n] / (m_total * G * d_xy[:n]))[sel]))
    return dict(ok=ok, speed=speed, cot_diss=cot_diss, cot_fluid=cot_fluid,
                t_blow=t_blow, w1=w1, m_fish=m_total, run_dir=run_dir)


def solve_iso_speed(shape, v_target, bl, tol=0.05, max_iter=3):
    """Run *shape* at the frequency that hits *v_target* BL/s; return measurement."""
    f = _freq_for_speed(shape, v_target)
    best = None
    for it in range(max_iter):
        tag = f"v{v_target:g}_{shape}_it{it}"
        print(f"\n--- iso-speed: target {v_target} BL/s | {shape} | "
              f"f={f:.2f} Hz | attempt {it} ---")
        try:
            IsoConfig(shape, f, tag).run()
        except Exception as exc:
            traceback.print_exc()
            print(f"[iso] run crashed ({exc}); using partial data.")
        run_dir = _latest_run_dir(tag)
        if run_dir is None:
            print(f"[iso] no output for {tag}; aborting this cell.")
            return None
        meas = measure(run_dir, bl)
        meas.update(shape=shape, v_target=v_target, freq=f, attempt=it)
        print(f"    -> achieved {meas['speed']:.2f} BL/s  "
              f"CoT_diss={meas['cot_diss']:.3f}  CoT_fluid={meas['cot_fluid']:.3f}  "
              f"(blow-up t={meas['t_blow']:.3f}s, ok={meas['ok']})")
        best = meas
        if not np.isfinite(meas["speed"]) or meas["speed"] <= 0:
            print("    speed unusable; halving dt is not implemented — keeping result.")
            break
        if abs(meas["speed"] - v_target) / v_target <= tol:
            print("    within tolerance.")
            break
        f *= v_target / meas["speed"]   # speed ~ linear in f
    return best


def run_one_freq(shape, freq, bl, analyze_only=False):
    """Run *shape* at fixed *freq* [Hz]; return measurement dict (no speed loop)."""
    tag = f"sweep_{shape}_f{freq:g}"
    if not analyze_only:
        print(f"\n--- freq-sweep: {shape} @ {freq:.2f} Hz ---")
        try:
            IsoConfig(shape, freq, tag).run()
        except Exception as exc:
            traceback.print_exc()
            print(f"[sweep] run crashed ({exc}); using partial data.")
    run_dir = _latest_run_dir(tag)
    if run_dir is None:
        print(f"[sweep] no output for {tag}.")
        return None
    meas = measure(run_dir, bl)
    meas.update(shape=shape, v_target=float("nan"), freq=freq, attempt=0)
    print(f"    {shape} @ {freq:.2f} Hz -> {meas['speed']:.2f} BL/s  "
          f"CoT_diss={meas['cot_diss']:.3f}  CoT_fluid={meas['cot_fluid']:.3f}  "
          f"(ok={meas['ok']})")
    return meas


def sweep_main(args):
    """Frequency-sweep mode: run each shape over a grid of frequencies."""
    rows = []
    for shape, freqs in (("tail", args.tail_freqs), ("full", args.full_freqs)):
        for f in freqs:
            meas = run_one_freq(shape, f, args.bl, analyze_only=args.analyze_only)
            if meas is not None:
                rows.append(meas)
    if not rows:
        print("No results.")
        return
    _write_csv(rows, args.out)

    print(f"\n{'shape':>5} {'f[Hz]':>7} {'v[BL/s]':>8} {'CoT_diss':>9} {'CoT_fluid':>10} {'ok':>4}")
    for r in sorted(rows, key=lambda r: (r["shape"], r["freq"])):
        print(f"{r['shape']:>5} {r['freq']:7.2f} {r['speed']:8.2f} "
              f"{r['cot_diss']:9.3f} {r['cot_fluid']:10.3f} {str(r['ok']):>4}")

    _overlap_report(rows)
    _plot(rows, args.out, by="speed")


def _overlap_report(rows):
    """In the overlapping speed band, interpolate both shapes and report dCoT."""
    def curve(shape, m):
        pts = sorted((r["speed"], r[m]) for r in rows
                     if r["shape"] == shape and np.isfinite(r["speed"]))
        if len(pts) < 2:
            return None
        v, c = zip(*pts)
        return np.array(v), np.array(c)
    print("\nEnergy difference in the OVERLAPPING speed band "
          "(interpolated to matched speed):")
    for m in ("cot_diss", "cot_fluid"):
        ct = curve("tail", m); cf = curve("full", m)
        if ct is None or cf is None:
            continue
        vt, cct = ct; vf, ccf = cf
        lo = max(vt.min(), vf.min()); hi = min(vt.max(), vf.max())
        if hi <= lo:
            print(f"  {m}: no speed overlap between shapes.")
            continue
        print(f"  {m}  (overlap {lo:.2f}-{hi:.2f} BL/s):")
        for v in np.linspace(lo, hi, 6):
            t = float(np.interp(v, vt, cct)); fu = float(np.interp(v, vf, ccf))
            win = "TAIL" if t < fu else "FULL"
            print(f"    v={v:5.2f}: tail {t:.3f}  full {fu:.3f}  "
                  f"-> {win} cheaper by {abs(fu-t)/min(t,fu)*100:4.1f}%")


def _write_csv(rows, out_prefix):
    csv_path = out_prefix + "_metrics.csv"
    keys = ["v_target", "shape", "freq", "speed", "cot_diss", "cot_fluid",
            "t_blow", "ok", "run_dir"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved table: {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speeds", type=float, nargs="+", default=[7.0, 9.0, 11.0],
                    help="target forward speeds [BL/s] (iso-speed mode).")
    ap.add_argument("--sweep", action="store_true",
                    help="frequency-sweep mode: run each shape over a freq grid "
                         "and compare energy in the overlapping speed band.")
    ap.add_argument("--tail-freqs", type=float, nargs="+",
                    default=[11, 13, 15, 17, 19, 21],
                    help="tail-shape frequencies [Hz] for --sweep.")
    ap.add_argument("--full-freqs", type=float, nargs="+",
                    default=[8, 10, 12, 14, 16],
                    help="full-shape frequencies [Hz] for --sweep.")
    ap.add_argument("--bl", type=float, default=0.017, help="body length [m].")
    ap.add_argument("--tol", type=float, default=0.05, help="speed tolerance (frac).")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip sims; re-analyse existing runs.")
    ap.add_argument("--out", default=os.path.join(NS_DATA_ROOT, STACK_ROOT, "iso_speed"))
    args = ap.parse_args()

    if args.sweep:
        args.out = os.path.join(NS_DATA_ROOT, STACK_ROOT, "freq_sweep")
        sweep_main(args)
        return

    rows = []
    for v in args.speeds:
        for shape in ("tail", "full"):
            if args.analyze_only:
                # pick the highest-attempt existing run for this cell
                tags = sorted(glob.glob(os.path.join(
                    NS_DATA_ROOT, STACK_ROOT, f"v{v:g}_{shape}_it*")))
                run_dir = None
                for t in reversed(tags):
                    run_dir = _latest_run_dir(os.path.basename(t))
                    if run_dir:
                        break
                if not run_dir:
                    print(f"[skip] no run for v{v:g}_{shape}")
                    continue
                meas = measure(run_dir, args.bl)
                meas.update(shape=shape, v_target=v, freq=float("nan"), attempt=-1)
            else:
                meas = solve_iso_speed(shape, v, args.bl, tol=args.tol)
                if meas is None:
                    continue
            rows.append(meas)

    if not rows:
        print("No results.")
        return

    csv_path = args.out + "_metrics.csv"
    keys = ["v_target", "shape", "freq", "speed", "cot_diss", "cot_fluid",
            "t_blow", "ok", "run_dir"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved table: {csv_path}")

    # Pretty table + matched-speed verdict.
    print(f"\n{'v*':>5} {'shape':>5} {'f[Hz]':>7} {'v[BL/s]':>8} "
          f"{'CoT_diss':>9} {'CoT_fluid':>10} {'ok':>4}")
    for r in rows:
        print(f"{r['v_target']:5.1f} {r['shape']:>5} {r.get('freq',float('nan')):7.2f} "
              f"{r['speed']:8.2f} {r['cot_diss']:9.3f} {r['cot_fluid']:10.3f} "
              f"{str(r['ok']):>4}")

    print("\nMatched-speed verdict (tail vs full at each target speed):")
    by_v = {}
    for r in rows:
        by_v.setdefault(r["v_target"], {})[r["shape"]] = r
    for v in sorted(by_v):
        c = by_v[v]
        if "tail" in c and "full" in c:
            for m in ("cot_diss", "cot_fluid"):
                t, fu = c["tail"][m], c["full"][m]
                win = "TAIL cheaper" if t < fu else "FULL cheaper"
                print(f"  v*={v:4.1f}  {m:10}: tail {t:.3f} (f={c['tail']['freq']:.1f}Hz, "
                      f"v={c['tail']['speed']:.2f})  vs  full {fu:.3f} "
                      f"(f={c['full']['freq']:.1f}Hz, v={c['full']['speed']:.2f})  "
                      f"-> {win} by {abs(fu-t)/min(t,fu)*100:.1f}%")

    _plot(rows, args.out)


def _plot(rows, out_prefix, by="speed"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, ttl in [(axes[0], "cot_diss", "CoT_diss (viscous loss)"),
                       (axes[1], "cot_fluid", "CoT_fluid (loss + wake KE)")]:
        for shape, col, mk in [("tail", "C0", "o"), ("full", "C3", "s")]:
            sel = [(r["speed"], r[m], r["freq"]) for r in rows
                   if r["shape"] == shape and np.isfinite(r["speed"])]
            sel.sort()
            if sel:
                xs, ys, fs = zip(*sel)
                ax.plot(xs, ys, mk + "-", color=col, label=f"{shape}-shape", ms=9)
                # annotate each point with its frequency
                for x, y, fr in zip(xs, ys, fs):
                    ax.annotate(f"{fr:.0f}Hz", (x, y), textcoords="offset points",
                                xytext=(0, 7), fontsize=6, color=col, ha="center")
        ax.set_xlabel("achieved speed [BL/s]")
        ax.set_ylabel(m)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Frequency sweep: tail vs full-body CoT vs speed "
                 f"(freq labels in Hz; window {WIN[0]}-{WIN[1]} s)")
    fig.tight_layout()
    png = out_prefix + "_cot_vs_speed.png"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print(f"\nSaved plot: {png}")


if __name__ == "__main__":
    main()
