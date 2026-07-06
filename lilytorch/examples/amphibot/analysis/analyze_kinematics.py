#!/usr/bin/env python3
"""
Complete kinematic parameter extraction from amphibot experimental data.
Loads tracking + controller data, reconstructs LED positions, computes joint angles,
compares with setpoints, and extracts frequency/amplitude/phase/wavelength.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, fftfreq
from collections import Counter
import os

# ──────────────────────────────
# Configuration
# ──────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'experimental_data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

NUM_LEDS = 13
MIN_DETECTIONS = 10  # minimum LEDs per frame to avoid wrong Hungarian assignments
MIN_SEGMENT = 20     # minimum angle segment length (frames)
TRIM = 2             # trim boundary frames from segments
MAX_JUMP = 20        # degrees, max jump within a segment

# ──────────────────────────────
# Data loading
# ──────────────────────────────
def load_controller_data(filepath):
    """Load controller data: time, 6 setpoints, 6 positions."""
    df = pd.read_csv(filepath, header=None)
    cols = ['time'] + [f'setpoint_{i}' for i in range(6)] + [f'pos_{i}' for i in range(6)]
    df.columns = cols[:df.shape[1]]
    return df

def load_track_data(filepath):
    """Load camera tracking data: variable columns per frame."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                frame = int(parts[1])
                count = int(parts[2])
                row = {'time': t, 'frame': frame}
                for i in range(count):
                    idx_start = 3 + i*3
                    if idx_start + 2 < len(parts):
                        try:
                            id_val = int(parts[idx_start])
                            x = float(parts[idx_start+1])
                            y = float(parts[idx_start+2])
                            row[f'x_{id_val}'] = x
                            row[f'y_{id_val}'] = y
                        except ValueError:
                            continue
                data.append(row)
            except ValueError:
                continue
    return pd.DataFrame(data)

# ──────────────────────────────
# Track reconstruction
# ──────────────────────────────
def extract_points_per_frame(df):
    x_cols = [c for c in df.columns if c.startswith('x_')]
    frames = []
    for _, row in df.iterrows():
        pts = []
        for xc in x_cols:
            id_str = xc.split('_')[1]
            yc = f'y_{id_str}'
            if yc in df.columns:
                xv, yv = row[xc], row[yc]
                if pd.notna(xv) and pd.notna(yv):
                    pts.append([xv, yv])
        frames.append(np.array(pts) if pts else np.empty((0, 2)))
    return frames

def deduplicate_cameras(points, merge_dist=0.025):
    if len(points) <= 1:
        return points
    pts = np.array(points)
    merged, used = [], set()
    for i in range(len(pts)):
        if i in used:
            continue
        cluster = [pts[i]]
        for j in range(i + 1, len(pts)):
            if j in used:
                continue
            if np.linalg.norm(pts[i] - pts[j]) < merge_dist:
                cluster.append(pts[j])
                used.add(j)
        used.add(i)
        merged.append(np.mean(cluster, axis=0))
    return np.array(merged)

def chain_order_points(points):
    if len(points) <= 1:
        return points
    pts = np.array(points)
    start = np.argmin(pts[:, 0])
    ordered = [start]
    remaining = set(range(len(pts))) - {start}
    while remaining:
        curr = ordered[-1]
        dists = {j: np.linalg.norm(pts[curr] - pts[j]) for j in remaining}
        nearest = min(dists, key=dists.get)
        ordered.append(nearest)
        remaining.remove(nearest)
    return pts[ordered]

def reconstruct_tracks(df, num_leds=NUM_LEDS, min_detections=MIN_DETECTIONS):
    """Full reconstruction pipeline — anchor at LATEST good frame and trace backward."""
    raw_frames = extract_points_per_frame(df)
    times = df['time'].values
    frame_nums = df['frame'].values
    n_frames = len(raw_frames)

    # Deduplicate
    processed_frames = []
    for pts in raw_frames:
        if len(pts) == 0:
            processed_frames.append(np.empty((0, 2)))
        else:
            processed_frames.append(deduplicate_cameras(pts))

    counts = [len(p) for p in processed_frames]
    nonempty = [c for c in counts if c > 0]
    print(f"  Detections: {len(nonempty)}/{n_frames} frames, "
          f"min={min(nonempty)}, max={max(nonempty)}, median={int(np.median(nonempty))}")

    # *** Blank frames with too few detections ***
    n_blanked = sum(1 for c in counts if 0 < c < min_detections)
    print(f"  Blanking {n_blanked} frames with <{min_detections} LEDs")
    for i in range(n_frames):
        if 0 < counts[i] < min_detections:
            processed_frames[i] = np.empty((0, 2))
            counts[i] = 0

    # Reference frame: pick from the LAST good frames (where tracking is cleanest)
    nonempty_indices = [i for i, c in enumerate(counts) if c > 0]
    if not nonempty_indices:
        return None

    # Prefer exact num_leds detection, as late as possible
    exact = [i for i, c in enumerate(counts) if c == num_leds]
    if exact:
        # Pick the latest one in the last 25% of the recording
        last_quarter = [i for i in exact if i >= n_frames * 0.75]
        if last_quarter:
            ref_idx = last_quarter[len(last_quarter) // 2]
        else:
            ref_idx = exact[-1]  # latest exact match
    else:
        close = [i for i, c in enumerate(counts) if c >= num_leds]
        if close:
            ref_idx = close[-1]
        else:
            max_c = max(counts)
            candidates = [i for i, c in enumerate(counts) if c == max_c]
            ref_idx = candidates[-1]

    ref_points = chain_order_points(processed_frames[ref_idx])
    if len(ref_points) > num_leds:
        ref_points = ref_points[:num_leds]
    n_tracks = len(ref_points)
    print(f"  Reference: frame {ref_idx} (of {n_frames}), {n_tracks} tracks")

    # Hungarian matching — backward from reference first, then forward
    track_history = np.full((n_frames, n_tracks, 2), np.nan)
    track_history[ref_idx] = ref_points

    base_thresh = 0.05
    max_speed = 1.0

    def do_pass(start, end, step):
        last_known = ref_points.copy()
        last_time = np.full(n_tracks, times[ref_idx])
        for i in range(start, end, step):
            obs = processed_frames[i]
            if len(obs) == 0:
                continue
            dt = np.abs(times[i] - last_time)
            thresh = base_thresh + max_speed * dt
            cost = cdist(last_known, obs)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] < thresh[r]:
                    track_history[i, r] = obs[c]
                    last_known[r] = obs[c]
                    last_time[r] = times[i]

    # Backward pass first (main direction — traces from good late data to noisier early data)
    if ref_idx > 0:
        do_pass(ref_idx - 1, -1, -1)
    # Forward pass (short — only frames after reference, if any)
    if ref_idx < n_frames - 1:
        do_pass(ref_idx + 1, n_frames, +1)

    # Build DataFrame
    data_dict = {'time': times, 'frame': frame_nums}
    for t in range(n_tracks):
        data_dict[f'led_{t}_x'] = track_history[:, t, 0]
        data_dict[f'led_{t}_y'] = track_history[:, t, 1]
    result_df = pd.DataFrame(data_dict)

    # Coverage stats
    for t in range(n_tracks):
        valid = result_df[f'led_{t}_x'].notna().sum()
        pct = 100 * valid / n_frames
        print(f"  LED {t}: {valid}/{n_frames} ({pct:.0f}%)")

    # Limited interpolation
    result_df = result_df.interpolate(method='linear', limit=3, limit_direction='both', axis=0)
    return result_df

# ──────────────────────────────
# Angle computation
# ──────────────────────────────
def find_contiguous_segments(mask):
    arr = np.asarray(mask, dtype=int)
    changes = np.diff(arr)
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if arr[0]:
        starts = np.concatenate([[0], starts])
    if arr[-1]:
        ends = np.concatenate([ends, [len(arr)]])
    return list(zip(starts, ends))

def clean_segment(segment, max_jump=MAX_JUMP):
    seg = segment.copy()
    for i in range(1, len(seg)):
        if abs(seg[i] - seg[i-1]) > max_jump:
            seg[i] = np.nan
    return seg

def compute_joint_angles(rdf, n_tracks, min_segment=MIN_SEGMENT, trim=TRIM):
    angles = {'time': rdf['time'].values}
    for i in range(1, n_tracks - 1):
        v1x = rdf[f'led_{i}_x'].values - rdf[f'led_{i-1}_x'].values
        v1y = rdf[f'led_{i}_y'].values - rdf[f'led_{i-1}_y'].values
        v2x = rdf[f'led_{i+1}_x'].values - rdf[f'led_{i}_x'].values
        v2y = rdf[f'led_{i+1}_y'].values - rdf[f'led_{i}_y'].values

        ang1 = np.arctan2(v1y, v1x)
        ang2 = np.arctan2(v2y, v2x)
        diff = ang2 - ang1
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        raw_deg = np.degrees(diff)

        not_nan = ~np.isnan(raw_deg)
        segments = find_contiguous_segments(not_nan)
        result = np.full_like(raw_deg, np.nan)

        for seg_start, seg_end in segments:
            if seg_end - seg_start < min_segment:
                continue
            s = seg_start + trim
            e = seg_end - trim
            if e - s < 5:
                s, e = seg_start, seg_end
                if e - s < 5:
                    continue
            segment = raw_deg[s:e].copy()
            segment = clean_segment(segment, MAX_JUMP)

            seg_mask = ~np.isnan(segment)
            sub_segs = find_contiguous_segments(seg_mask)
            for ss, se in sub_segs:
                sub = segment[ss:se]
                if len(sub) < 5:
                    continue
                if np.std(sub) > 15 and len(sub) < 50:
                    continue
                k = min(5, len(sub) if len(sub) % 2 == 1 else len(sub) - 1)
                if k >= 3:
                    sub = medfilt(sub, kernel_size=k)
                sg_win = min(11, len(sub) if len(sub) % 2 == 1 else len(sub) - 1)
                if sg_win >= 5:
                    sub = savgol_filter(sub, sg_win, min(3, sg_win - 1))
                result[s + ss : s + se] = sub

        angles[f'angle_{i}'] = result
    return pd.DataFrame(angles)

# ──────────────────────────────
# Parameter extraction (FFT-based + sine fitting)
# ──────────────────────────────
def sine_func(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C

def extract_params_fft(t, y, min_freq=0.3, max_freq=5.0):
    """Extract frequency, amplitude, and phase from FFT of periodic signal.
    Works for any periodic waveform (sine, triangle, clipped, etc.)."""
    valid = ~np.isnan(y)
    if valid.sum() < 20:
        return None
    t_v, y_v = t[valid], y[valid]

    # Resample to uniform grid for FFT
    dt_med = np.median(np.diff(t_v))
    t_uniform = np.arange(t_v[0], t_v[-1], dt_med)
    y_uniform = np.interp(t_uniform, t_v, y_v)

    n = len(y_uniform)
    if n < 20:
        return None

    y_centered = y_uniform - np.mean(y_uniform)
    yf = fft(y_centered)
    xf = fftfreq(n, dt_med)

    # Find fundamental frequency in range
    pos_mask = (xf > min_freq) & (xf < max_freq)
    if not pos_mask.any():
        return None

    magnitudes = np.abs(yf[pos_mask])
    freqs = xf[pos_mask]
    fund_idx = np.argmax(magnitudes)

    freq = freqs[fund_idx]
    phase = np.angle(yf[pos_mask][fund_idx])
    amplitude = (np.max(y_v) - np.min(y_v)) / 2  # peak-to-peak / 2
    offset = np.mean(y_v)

    return {
        'frequency': freq,
        'amplitude': amplitude,
        'phase': phase,
        'offset': offset,
        'fft_amplitude': 2.0 / n * magnitudes[fund_idx],  # FFT fundamental amplitude
    }

def fit_sine(t, y, f_guess=1.0):
    """Fit y = A*sin(2*pi*f*t + phi) + C. Returns dict or None."""
    try:
        amp_guess = (np.nanmax(y) - np.nanmin(y)) / 2
        offset_guess = np.nanmean(y)
        popt, pcov = curve_fit(
            sine_func, t, y,
            p0=[amp_guess, f_guess, 0, offset_guess],
            bounds=([0, 0.1, -np.pi, -90], [90, 5.0, np.pi, 90]),
            maxfev=10000
        )
        residual = np.sqrt(np.mean((y - sine_func(t, *popt))**2))
        return {
            'amplitude': popt[0], 'frequency': popt[1],
            'phase': popt[2], 'offset': popt[3],
            'rmse': residual
        }
    except Exception:
        return None

# ──────────────────────────────
# Main analysis
# ──────────────────────────────
def main():
    # Load controller data
    print("=" * 60)
    print("Loading controller data...")
    dfs = {}
    for i in range(1, 4):
        fp = os.path.join(DATA_DIR, f'data{i}.csv')
        if os.path.exists(fp):
            dfs[f'data{i}.csv'] = load_controller_data(fp)
            print(f"  data{i}.csv: {len(dfs[f'data{i}.csv'])} rows")

    # Load tracking data
    print("\nLoading tracking data...")
    track_dfs = {}
    for i in range(1, 4):
        fp = os.path.join(DATA_DIR, f'track{i}.csv')
        if os.path.exists(fp):
            track_dfs[f'track{i}.csv'] = load_track_data(fp)
            print(f"  track{i}.csv: {len(track_dfs[f'track{i}.csv'])} frames")

# ── Extract setpoint parameters (FFT-based, steady-state only) ──
    print("\n" + "=" * 60)
    print("Extracting parameters from controller setpoints...")
    setpoint_params = {}
    for data_name, df_data in dfs.items():
        t = df_data['time'].values
        t_rel = t - t[0]

        # Use only latter half (steady-state, after ramp-up)
        ss_mask = t_rel > t_rel[-1] * 0.5
        t_ss = t_rel[ss_mask]

        params = {}
        for j in range(6):
            col = f'setpoint_{j}'
            y_ss = df_data[col].values[ss_mask]
            if np.std(y_ss) < 0.5:
                continue
            result = extract_params_fft(t_ss, y_ss)
            if result is not None:
                params[j] = result

        setpoint_params[data_name] = params
        if params:
            amps = [p['amplitude'] for p in params.values()]
            freqs = [p['frequency'] for p in params.values()]
            print(f"  {data_name}: {len(params)} active joints, "
                  f"avg freq={np.mean(freqs):.3f} Hz, avg amp={np.mean(amps):.1f} deg")

            # Phase differences (wavelength estimation)
            sorted_joints = sorted(params.keys())
            if len(sorted_joints) > 1:
                dphis = []
                for k in range(1, len(sorted_joints)):
                    dp = params[sorted_joints[k]]['phase'] - params[sorted_joints[k-1]]['phase']
                    dp = (dp + np.pi) % (2 * np.pi) - np.pi
                    dphis.append(dp)
                avg_dphi = np.mean(dphis)
                if abs(avg_dphi) > 0.01:
                    wavelength_joints = 2 * np.pi / abs(avg_dphi)
                    print(f"           avg phase diff={np.degrees(avg_dphi):.1f} deg/joint, "
                          f"wavelength~{wavelength_joints:.1f} joints")

    # ── Reconstruct tracks ──
    print("\n" + "=" * 60)
    print("Reconstructing LED positions from tracking data...")
    reconstructed_dfs = {}
    for name, df in track_dfs.items():
        print(f"\n  {name}:")
        reconstructed_dfs[name] = reconstruct_tracks(df)

    # ── Compute joint angles ──
    print("\n" + "=" * 60)
    print("Computing joint angles...")
    all_angles = {}
    for name, rdf in reconstructed_dfs.items():
        if rdf is None:
            continue
        x_cols = [c for c in rdf.columns if c.startswith('led_') and c.endswith('_x')]
        n_tracks = len(x_cols)
        adf = compute_joint_angles(rdf, n_tracks)
        all_angles[name] = adf

        angle_cols = [c for c in adf.columns if c.startswith('angle_')]
        # Count usable data per angle
        for col in angle_cols:
            valid = adf[col].notna().sum()
            pct = 100 * valid / len(adf)
            if pct > 5:
                print(f"  {name} {col}: {valid} frames ({pct:.0f}%)")

    # ── Plot angles ──
    print("\nPlotting angles...")
    for name, adf in all_angles.items():
        angle_cols = [c for c in adf.columns if c.startswith('angle_')]

        # Individual subplots
        fig, axes = plt.subplots(len(angle_cols), 1, figsize=(14, 2.5 * len(angle_cols)), sharex=True)
        if len(angle_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(angle_cols):
            axes[idx].plot(adf['time'], adf[col], '-', linewidth=0.8)
            axes[idx].set_ylabel(f'{col} (deg)')
            axes[idx].grid(True)
            axes[idx].set_ylim(-40, 40)
        axes[0].set_title(f'Joint Angles - {name}')
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'angles_{name.replace(".csv","")}.png'), dpi=150)
        plt.close()

        # Overlaid
        plt.figure(figsize=(14, 6))
        for col in angle_cols:
            plt.plot(adf['time'], adf[col], '-', linewidth=0.8, label=col)
        plt.title(f'All Joint Angles Overlaid - {name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        plt.grid(True)
        plt.ylim(-40, 40)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'angles_overlaid_{name.replace(".csv","")}.png'), dpi=150)
        plt.close()

# ── Extract camera angle parameters (FFT + sine fit) ──
    print("\n" + "=" * 60)
    print("Extracting parameters from camera-derived angles...")

    track_data_map = {
        'track1.csv': 'data1.csv',
        'track2.csv': 'data2.csv',
        'track3.csv': 'data3.csv',
    }

    camera_params = {}

    for track_name, adf in all_angles.items():
        data_name = track_data_map.get(track_name)
        sp = setpoint_params.get(data_name, {}) if data_name else {}

        t_cam = adf['time'].values
        t_rel = t_cam - t_cam[0]  # relative time
        angle_cols = [c for c in adf.columns if c.startswith('angle_')]

        cam_fit_results = {}

        # Plot camera angles with fits
        n_plot = min(6, len(angle_cols))
        fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]

        for idx in range(n_plot):
            angle_col = angle_cols[idx]
            angle_idx = int(angle_col.split('_')[1])
            y_cam = adf[angle_col].values

            ax = axes[idx]
            valid = ~np.isnan(y_cam)

            if valid.sum() > 10:
                ax.plot(t_rel[valid], y_cam[valid], '.', markersize=2, alpha=0.5,
                       color='blue', label=f'Camera {angle_col}')

            # Also plot setpoint waveform (using FFT params, relative time)
            setpoint_idx = min(angle_idx - 1, 5)
            if setpoint_idx in sp:
                p = sp[setpoint_idx]
                t_sine = np.linspace(0, t_rel[-1], 1000)
                y_sine = p['amplitude'] * np.sin(2*np.pi*p['frequency']*t_sine + p['phase']) + p['offset']
                ax.plot(t_sine, y_sine, '-', linewidth=1, color='red', alpha=0.5,
                       label=f'Setpoint {setpoint_idx} (A={p["amplitude"]:.1f}, f={p["frequency"]:.2f}Hz)')

            # Extract params from camera data using FFT on longest valid segment
            if valid.sum() > 50:
                segs = find_contiguous_segments(valid)
                longest = max(segs, key=lambda s: s[1]-s[0])
                ls, le = longest

                if le - ls > 30:
                    t_fit = t_rel[ls:le]
                    y_fit = y_cam[ls:le]
                    fit_valid = ~np.isnan(y_fit)

                    if fit_valid.sum() > 30:
                        t_f = t_fit[fit_valid]
                        y_f = y_fit[fit_valid]

                        # FFT-based extraction
                        fft_result = extract_params_fft(t_f, y_f)
                        # Also try sine fit using FFT frequency as seed
                        f_guess = fft_result['frequency'] if fft_result else 1.0
                        sine_result = fit_sine(t_f - t_f[0], y_f, f_guess=f_guess)

                        # Prefer sine fit if RMSE is good, else use FFT
                        if sine_result is not None and sine_result['rmse'] < 0.3 * np.std(y_f):
                            best = sine_result
                            method = 'sine'
                        elif fft_result is not None:
                            best = fft_result
                            method = 'fft'
                        elif sine_result is not None:
                            best = sine_result
                            method = 'sine'
                        else:
                            best = None
                            method = None

                        if best is not None:
                            cam_fit_results[angle_col] = {
                                'amplitude': best['amplitude'],
                                'frequency': best['frequency'],
                                'phase': best['phase'],
                                'offset': best['offset'],
                                'n_points': int(fit_valid.sum()),
                                'segment_duration': float(t_f[-1] - t_f[0]),
                                'method': method,
                            }
                            # Plot the fit
                            t_plot = t_f - t_f[0]
                            y_fitted = sine_func(t_plot, best['amplitude'], best['frequency'],
                                                 best['phase'], best['offset'])
                            ax.plot(t_f, y_fitted, '-', linewidth=1.5, color='green',
                                   label=f'{method} (A={best["amplitude"]:.1f}, f={best["frequency"]:.2f}Hz)')

            ax.set_ylabel(f'{angle_col} (deg)')
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True)
            ax.set_ylim(-40, 40)

        axes[-1].set_xlabel('Relative Time (s)')
        plt.suptitle(f'Camera Angles + Fits - {track_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'comparison_{track_name.replace(".csv","")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        camera_params[track_name] = cam_fit_results
        n_fitted = len(cam_fit_results)
        if n_fitted > 0:
            avg_f = np.mean([p['frequency'] for p in cam_fit_results.values()])
            avg_a = np.mean([p['amplitude'] for p in cam_fit_results.values()])
            methods = [p['method'] for p in cam_fit_results.values()]
            print(f"  {track_name}: {n_fitted} angles fitted, avg freq={avg_f:.3f} Hz, "
                  f"avg amp={avg_a:.1f} deg, methods={dict(Counter(methods))}")
        else:
            print(f"  {track_name}: no angles could be fitted")

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("KINEMATIC PARAMETER SUMMARY")
    print("=" * 60)

    print("\n--- Controller Setpoint Parameters ---")
    print(f"{'Dataset':<12} {'Joint':<8} {'Amplitude (deg)':<17} {'Frequency (Hz)':<16} "
          f"{'Phase (deg)':<14} {'Offset (deg)':<14}")
    print("-" * 80)
    for data_name, params in setpoint_params.items():
        for j in sorted(params.keys()):
            p = params[j]
            print(f"{data_name:<12} {j:<8} {p['amplitude']:>14.2f}   {p['frequency']:>13.3f}   "
                  f"{np.degrees(p['phase']):>11.1f}   {p['offset']:>11.2f}")

    print("\n--- Camera-Derived Angle Parameters ---")
    print(f"{'Track':<12} {'Angle':<10} {'Amplitude (deg)':<17} {'Frequency (Hz)':<16} "
          f"{'Phase (deg)':<14} {'Offset (deg)':<14} {'N pts':<8} {'Duration (s)':<12}")
    print("-" * 100)
    for track_name, params in camera_params.items():
        for angle_col in sorted(params.keys()):
            p = params[angle_col]
            print(f"{track_name:<12} {angle_col:<10} {p['amplitude']:>14.2f}   {p['frequency']:>13.3f}   "
                  f"{np.degrees(p['phase']):>11.1f}   {p['offset']:>11.2f}   {p['n_points']:>5}   "
                  f"{p['segment_duration']:>9.1f}")

    # ── Wavelength estimation from camera data ──
    print("\n--- Wavelength Estimation ---")
    for track_name, params in camera_params.items():
        angle_names = sorted(params.keys())
        if len(angle_names) < 2:
            continue
        dphis = []
        for k in range(1, len(angle_names)):
            dp = params[angle_names[k]]['phase'] - params[angle_names[k-1]]['phase']
            dp = (dp + np.pi) % (2 * np.pi) - np.pi
            dphis.append(dp)
        if dphis:
            avg_dphi = np.mean(dphis)
            print(f"  {track_name}: avg phase diff = {np.degrees(avg_dphi):.1f} deg/joint")
            if abs(avg_dphi) > 0.01:
                wl = 2 * np.pi / abs(avg_dphi)
                print(f"  {track_name}: estimated wavelength = {wl:.1f} body segments")

    print(f"\nFigures saved to {FIG_DIR}/")
    print("Done!")

if __name__ == '__main__':
    main()
