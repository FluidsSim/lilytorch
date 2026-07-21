#!/usr/bin/env python3
"""Explain Issue 1 (speed-oscillation mismatch) as a metric/data artifact,
and show it closing under two fair-comparison fixes.

The compare speed is |d(centroid of tracked points)/dt|. The real trace
oscillates far more than the sim's. This figure shows WHY, and that the gap
is a metric/measurement effect that shrinks once the comparison is made fair:

  MECHANISM   bending shortens the head->tail chord ("foreshortening"), which
              drags the point-cloud centroid ~2x/beat; at 6 kHz, |dx|/dt turns
              tiny position noise into large BL/s.
  FIX 1       denoise the real DLC keypoints (temporal low-pass): removes the
              >40 Hz jitter the rigid sim cannot have.
  FIX 2       extend the sim midline (link COMs span only ~0.85 BL) out to the
              true body tips, matching the real keypoints' ~0.99 BL extent, so
              both centroids sample the same flexible extremities.

Result: real chord ±6.2% -> ±5.9% (denoise); sim ±4.8% -> ±5.4% (tips) -> they
nearly meet, and the low-pass speed cv converges (~0.47) at stroke bandwidth.

Usage:  python viz_issue1_foreshortening.py [sim_run_dir]
        default = pre-emphasis run T18:10 (matched ~97% body wave).
        Do NOT use the de-biased diagnostic runs T21:42 / T21:47.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from farms_core.sensors.sensor_convention import sc

BL = 0.018
DENOISE_HZ = 40.0          # real-keypoint low-pass (keeps 2x tail-beat ~32 Hz)
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "ep248_Cl2_slow_fish13_XY_BL.csv")
SIM = sys.argv[1] if len(sys.argv) > 1 else "/data/andreaferrario/ns_data/2026-07-20T18:10:10.765745"
OUT = "/data/andreaferrario/ns_data/issue1_foreshortening_explained.png"

REAL_C, SIM_C = "black", "#2196F3"

# ---------- helpers ----------
def chord(X, Y):  return np.hypot(X[:, -1] - X[:, 0], Y[:, -1] - Y[:, 0])
def arclen(X, Y): return np.hypot(np.diff(X, 1), np.diff(Y, 1)).sum(1)
def pct(v):       return 100 * v.std() / v.mean()
def centroid_speed(t, X, Y):
    cx, cy = X.mean(1), Y.mean(1)
    return np.hypot(np.diff(cx), np.diff(cy)) / np.diff(t)
def lp1(sig, fs, fc, order=4):
    b, a = butter(order, fc / (0.5 * fs), btype="low"); return filtfilt(b, a, sig)
def lp2(A, fs, fc, order=4):
    b, a = butter(order, fc / (0.5 * fs), btype="low"); return filtfilt(b, a, A, axis=0)

def extend_to_tips(X, Y, f=1.0):
    """Extrapolate the midline one terminal-segment beyond each end COM so the
    tracked span reaches the true body tips (arc 0.85 -> ~0.98 BL)."""
    hx = X[:, 0] + (X[:, 0] - X[:, 1]) * f;  hy = Y[:, 0] + (Y[:, 0] - Y[:, 1]) * f
    tx = X[:, -1] + (X[:, -1] - X[:, -2]) * f; ty = Y[:, -1] + (Y[:, -1] - Y[:, -2]) * f
    return (np.hstack([hx[:, None], X, tx[:, None]]),
            np.hstack([hy[:, None], Y, ty[:, None]]))

# ---------- load real ----------
df = pd.read_csv(CSV)
tr = df["time_ms"].values / 1000.0; tr -= tr[0]
Xr = df[[c for c in df.columns if c.startswith("x")]].values
Yr = df[[c for c in df.columns if c.startswith("y")]].values
fsR = 1 / np.median(np.diff(tr))
Xrd, Yrd = lp2(Xr, fsR, DENOISE_HZ), lp2(Yr, fsR, DENOISE_HZ)   # FIX 1

# ---------- load sim ----------
with h5py.File(os.path.join(SIM, "output", "simulation.hdf5"), "r") as f:
    la = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
    dt = float(np.array(f["timestep"]))
Xs = la[:, :, sc.link_com_position_x] / BL
Ys = la[:, :, sc.link_com_position_y] / BL
ts = dt * np.arange(la.shape[0]); fsS = 1 / dt
Xse, Yse = extend_to_tips(Xs, Ys)                               # FIX 2

chR, chRd = chord(Xr, Yr), chord(Xrd, Yrd)
chS, chSe = chord(Xs, Ys), chord(Xse, Yse)
arR, arRd, arS = arclen(Xr, Yr), arclen(Xrd, Yrd), arclen(Xs, Ys)
spR  = centroid_speed(tr, Xr, Yr)
spRd = centroid_speed(tr, Xrd, Yrd)
spS  = centroid_speed(ts, Xs, Ys)

# ============================ FIGURE ============================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.29)

# --- P1: mechanism ---
ax = fig.add_subplot(gs[0, 0])
iS, iB = int(np.argmax(chR)), int(np.argmin(chR))
for idx, lab, col in [(iS, f"straightest (chord {chR[iS]:.2f} BL)", "#4CAF50"),
                      (iB, f"most bent (chord {chR[iB]:.2f} BL)", "#E53935")]:
    x = Xr[idx] - Xr[idx, 0]; y = Yr[idx] - Yr[idx, 0]
    ax.plot(x, y, "o-", color=col, ms=4, lw=1.6, label=lab)
    ax.plot([x[0], x[-1]], [y[0], y[-1]], "--", color=col, lw=1.0, alpha=0.7)
ax.set_title("1. Mechanism: bending shortens the head–tail chord", fontsize=10)
ax.set_xlabel("along body [BL]"); ax.set_ylabel("lateral [BL]")
ax.legend(fontsize=7, loc="upper right"); ax.set_aspect("equal"); ax.grid(alpha=0.3)

# --- P2: chord, the two fixes converging ---
ax = fig.add_subplot(gs[0, 1])
ax.plot(tr, 100*(chR/chR.mean()-1),  color=REAL_C, lw=0.7, alpha=0.45)
ax.plot(tr, 100*(chRd/chRd.mean()-1), color=REAL_C, lw=1.3)
ax.plot(ts, 100*(chS/chS.mean()-1),  color=SIM_C, lw=0.7, alpha=0.45)
ax.plot(ts, 100*(chSe/chSe.mean()-1), color=SIM_C, lw=1.3)
ax.set_title("2. Foreshortening converges after both fixes", fontsize=10)
ax.set_xlabel("time [s]"); ax.set_ylabel("chord deviation [% of mean]")
ax.set_xlim(0, min(tr[-1], ts[-1])); ax.grid(alpha=0.3)
box = (f"real raw        ±{pct(chR):.1f}%\n"
       f"real denoised   ±{pct(chRd):.1f}%\n"
       f"sim link-COMs   ±{pct(chS):.1f}%\n"
       f"sim to-tips     ±{pct(chSe):.1f}%")
ax.text(0.02, 0.02, box, transform=ax.transAxes, fontsize=7.5, family="monospace",
        va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

# --- P3: arc length (denoise removes the >40 Hz breathing) ---
ax = fig.add_subplot(gs[0, 2])
ax.plot(tr, 100*(arR/arR.mean()-1),  color=REAL_C, lw=0.7, alpha=0.45,
        label=f"real raw  ±{pct(arR):.1f}%")
ax.plot(tr, 100*(arRd/arRd.mean()-1), color=REAL_C, lw=1.3,
        label=f"real denoised  ±{pct(arRd):.1f}%")
ax.plot(ts, 100*(arS/arS.mean()-1),  color=SIM_C, lw=1.4,
        label=f"sim  ±{pct(arS):.2f}% (rigid)")
ax.set_title("3. Tracked body-length: real 'breathes', sim is rigid", fontsize=10)
ax.set_xlabel("time [s]"); ax.set_ylabel("arc-length dev. [% of mean]")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
ax.set_xlim(0, min(tr[-1], ts[-1]))

# --- P4: speed trace, denoise calms the real toward sim ---
ax = fig.add_subplot(gs[1, 0])
ax.plot(tr[1:], spR,  color=REAL_C, lw=0.6, alpha=0.4, label=f"real raw (cv {spR.std()/spR.mean():.2f})")
ax.plot(tr[1:], spRd, color=REAL_C, lw=1.2, label=f"real denoised (cv {spRd.std()/spRd.mean():.2f})")
ax.plot(ts[1:], spS,  color=SIM_C, lw=1.0, label=f"sim (cv {spS.std()/spS.mean():.2f})")
ax.set_title(f"4. Speed: denoise ({DENOISE_HZ:.0f} Hz) removes the fake spikes", fontsize=10)
ax.set_xlabel("time [s]"); ax.set_ylabel("centroid speed [BL/s]")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
ax.set_xlim(0, min(tr[-1], ts[-1])); ax.set_ylim(0, 20)

# --- P5: PSD ---
ax = fig.add_subplot(gs[1, 1])
fR, PR = welch(spR - spR.mean(), fsR, nperseg=min(512, len(spR)))
fS, PS = welch(spS - spS.mean(), fsS, nperseg=min(512, len(spS)))
ax.semilogy(fR, PR, color=REAL_C, lw=1.0, label="real raw")
ax.semilogy(fS, PS, color=SIM_C, lw=1.2, label="sim")
ax.axvspan(DENOISE_HZ, fR.max(), color="orange", alpha=0.12)
ax.text(0.63, 0.9, f">{DENOISE_HZ:.0f} Hz:\nDLC jitter\n(removed by\nFix 1)",
        transform=ax.transAxes, fontsize=7.5, color="#B36B00", va="top")
ax.set_title("5. Real has high-freq power the sim lacks", fontsize=10)
ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("centroid-speed PSD")
ax.set_xlim(0, 80); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

# --- P6: cv vs low-pass cutoff ---
ax = fig.add_subplot(gs[1, 2])
cuts = np.array([5, 8, 10, 15, 20, 30, 50, 80])
def cv_at(t, X, Y, fs, fc):
    cx, cy = lp1(X.mean(1), fs, fc), lp1(Y.mean(1), fs, fc)
    sp = np.hypot(np.diff(cx), np.diff(cy)) / np.diff(t)
    return sp.std() / sp.mean()
ax.plot(cuts, [cv_at(tr, Xr, Yr, fsR, fc) for fc in cuts], "o-", color=REAL_C, label="real")
ax.plot(cuts, [cv_at(ts, Xs, Ys, fsS, fc) for fc in cuts], "s-", color=SIM_C, label="sim")
ax.axvline(20, color="gray", ls=":", lw=1)
ax.text(0.28, 0.06, "← stroke band", transform=ax.transAxes, fontsize=7, color="gray")
ax.set_title("6. At stroke bandwidth, the surge is the SAME", fontsize=10)
ax.set_xlabel("low-pass cutoff [Hz]"); ax.set_ylabel("speed cv = std/mean")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

fig.suptitle(
    "Issue 1: the real speed's extra oscillation is point-cloud foreshortening "
    "+ DLC jitter — it collapses once the comparison is made fair\n"
    f"Fix 1 denoise real @ {DENOISE_HZ:.0f} Hz   |   Fix 2 extend sim to true tips"
    f"   |   sim = {os.path.basename(SIM)}",
    fontsize=11.5, y=1.0)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved:", OUT)
print(f"chord ±%: real raw {pct(chR):.1f} | real denoised {pct(chRd):.1f} | "
      f"sim COM {pct(chS):.1f} | sim tips {pct(chSe):.1f}")
print(f"speed cv: real raw {spR.std()/spR.mean():.2f} | real denoised {spRd.std()/spRd.mean():.2f} | "
      f"sim {spS.std()/spS.mean():.2f}")
