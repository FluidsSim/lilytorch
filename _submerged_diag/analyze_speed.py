"""Analyze submerged-vs-surface swim-speed A/B (HP5b).

For each speed_<tag>.csv, report the steady-state forward swim speed over the
last `tail_frac` of the run, measured three robust ways:
  * net-displacement speed  |Δ(x,y)| / Δt  (cycle-jitter-free)
  * mean path-speed         <√(vx²+vy²)>    (instantaneous, turning-robust)
  * mean surge              <vx>            (straight-swim component)
"""
import sys, glob, numpy as np

def analyze(path, tail_frac=0.35):
    d = np.genfromtxt(path, delimiter=",", names=True)
    n = len(d["t"]); i0 = int(n * (1 - tail_frac))
    t, x, y = d["t"], d["x"], d["y"]
    vx, vy = d["vx_fd"], d["vy_fd"] if "vy_fd" in d.dtype.names else None
    tt, xx, yy = t[i0:], x[i0:], y[i0:]
    dt = tt[-1] - tt[0]
    net_disp = np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]) / dt
    # instantaneous velocities by finite difference of position (robust)
    vx_i = np.gradient(x, t); vy_i = np.gradient(y, t)
    path_sp = np.mean(np.hypot(vx_i[i0:], vy_i[i0:]))
    mean_vx = np.mean(vx_i[i0:]); mean_vy = np.mean(vy_i[i0:])
    return dict(n=n, T=t[-1], net_disp=net_disp, path_sp=path_sp,
                mean_vx=mean_vx, mean_vy=mean_vy,
                x0=x[0], xT=x[-1], y0=y[0], yT=y[-1], z=d["z"][-1])

if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(
        "/data/andreaferrario/lilytorch/_submerged_diag/speed_*.csv"))
    print(f"{'tag':<12}{'steps':>6}{'T[s]':>6}{'z':>9}"
          f"{'net|v|':>9}{'path|v|':>9}{'<vx>':>9}{'<vy>':>9}")
    for f in files:
        tag = f.split("speed_")[-1].replace(".csv", "")
        try:
            r = analyze(f)
        except Exception as e:
            print(f"{tag:<12}  ERROR: {e}"); continue
        print(f"{tag:<12}{r['n']:>6}{r['T']:>6.2f}{r['z']:>9.4f}"
              f"{r['net_disp']:>9.4f}{r['path_sp']:>9.4f}"
              f"{r['mean_vx']:>9.4f}{r['mean_vy']:>9.4f}")
