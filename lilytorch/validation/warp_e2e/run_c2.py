"""C2 — end-to-end coupled-trajectory validation: native vs Warp backend.

Runs the 2-D ``_1guillasim`` pinned swimmer (scene a, f64) as a real headless
FARMS/MuJoCo coupled sim, once with the native CUDA backend and once with the
Warp backend (selected via a monkeypatch injected through the sanctioned
``_extra_run_patch`` seam — there is NO config key for the backend), logging the
body trajectory + key fluid fields each step so the two runs can be compared.

Usage (from /data/andreaferrario/lilytorch):
    python -m lilytorch.validation.warp_e2e.run_c2 native --n 120
    python -m lilytorch.validation.warp_e2e.run_c2 warp   --n 120
    python -m lilytorch.validation.warp_e2e.run_c2 compare
    # perf (Warp Kernel-A CUDA-graph fast-path):
    python -m lilytorch.validation.warp_e2e.run_c2 warp   --n 400 --perf

Both the parity and perf runs keep the scene's native ``poisson_method`` (fft)
so both backends use the identical (kernel-agnostic) FFT Poisson — the
Warp-backed ops exercised are advection-flux, Kernel A/B (2-D), apply_bcs and the
force readout.  ``--perf`` turns on the Warp Kernel-A streaming CUDA-graph
(``solver.kernel_cuda_graph``; native ignores the flag).  Forcing mgcg here
destabilises the fft-tuned eel and native ``poisson_cuda_graph`` hits a capture
assert at 1024x128, so the graphed-MGCG (C1) fast-path is timed on its proper
3-D Poisson-bound target (jellyfish) instead, not this 2-D scene.

``--perturb EPS [--perturb-recurring]`` seeds a one-shot (or per-step) relative
velocity perturbation — used to show the coupled system is linearly stable to
uniform perturbations over this horizon (so any backend divergence is a discrete
event, not chaotic amplification).
"""
import argparse
import os
import time

NS_DATA = "/data/andreaferrario/ns_data"
DIRS = {
    "native": os.path.join(NS_DATA, "c2_native_2d"),
    "warp": os.path.join(NS_DATA, "c2_warp_2d"),
}


def _make_config(backend, n_iter, every, field_every, perf, perturb=0.0, perturb_recurring=False):
    from lilytorch.farms_examples._1guillasim.gen_configs_one_pinned_2d import (
        SimConfig as _Base)

    out_dir = DIRS[backend]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "c2_log.npz")

    class C2Config(_Base):
        def __init__(self):
            super().__init__()
            self.headless = True
            self.fast = True
            self.n_iterations = n_iter
            self.bdim_nt = n_iter
            self.save = False
            self.save_frames = False
            self.save_drags = False
            self.diagnostics_every = 0
            # write the run's scratch (configs, run.sh, logs) under ns_data
            self.stack_folder = out_dir
            # NOTE: scene (a) uses the scene's native ``poisson_method`` (fft) in
            # BOTH the parity and perf runs.  Forcing mgcg destabilises this
            # fft-tuned eel (under-converged projection -> body ejected); native
            # ``poisson_cuda_graph`` also hits a capture assert at 1024x128.  The
            # graphed-MGCG (C1) fast-path is validated separately on its proper
            # 3-D Poisson-bound target (jellyfish).  Here ``--perf`` only turns on
            # the Warp Kernel-A streaming CUDA-graph (warp reads it; native
            # ignores the flag), measuring end-to-end wall on the real scene.

        def extra_simulation_extensions(self, output_folder):
            # drop the FlowViewer2D + StreamingCamera (GL/offscreen) — headless.
            return []

        def _bdim_extension(self, output_folder):
            ext = super()._bdim_extension(output_folder)
            # Select the backend via the production ``solver.backend`` config key
            # (BDIMhandler resolves it through src_warp.backend) — no monkeypatch.
            ext["config"]["bdim_yaml"]["solver"]["backend"] = backend
            if perf:
                ext["config"]["bdim_yaml"]["solver"]["kernel_cuda_graph"] = True
            return ext

        def _extra_run_patch(self):
            return (
                "import lilytorch.validation.warp_e2e.c2_hook as _c2;"
                f"_c2.install(backend={backend!r}, out_path={out_path!r}, "
                f"every={every}, field_every={field_every}, "
                f"max_steps={n_iter}, perturb={perturb!r}, "
                f"perturb_recurring={perturb_recurring!r});"
            )

    return C2Config()


def run(backend, n_iter, every, field_every, perf, perturb=0.0, perturb_recurring=False):
    cfg = _make_config(backend, n_iter, every, field_every, perf, perturb, perturb_recurring)
    t0 = time.time()
    cfg.run()
    dt = time.time() - t0
    print(f"[run_c2] backend={backend} perf={perf} n={n_iter} "
          f"wall={dt:.1f}s")


def compare(field_every):
    import numpy as np

    a = np.load(os.path.join(DIRS["native"], "c2_log.npz"), allow_pickle=True)
    b = np.load(os.path.join(DIRS["warp"], "c2_log.npz"), allow_pickle=True)

    def rel(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        den = np.maximum(np.abs(x), np.abs(y))
        den = np.where(den == 0, 1.0, den)
        return np.abs(x - y) / den

    n = min(len(a["step"]), len(b["step"]))
    print(f"\n=== C2 native-vs-warp comparison (first {n} logged steps) ===")

    # trajectory.  ``qpos`` (the generalized coordinates) is the authoritative
    # body trajectory and is enumerated deterministically from the model joints.
    # ``xpos`` (Cartesian body frames) includes the STATIC arena/pool-wall bodies,
    # which MuJoCo enumerates in a model-build-order that can differ between two
    # independently compiled MJCFs — so we align xpos rows by their step-0
    # position (greedy nearest-neighbour) before differencing, removing that pure
    # ordering artifact (the dynamic swimmer links match without it).
    qa, qb = a["qpos"][:n], b["qpos"][:n]
    q_abs = np.abs(qa - qb)
    print(f"qpos : max|Δ|={q_abs.max():.3e}  mean|Δ|={q_abs.mean():.3e}")

    xa, xb = a["xpos"][:n], b["xpos"][:n]
    nb = xa.shape[1]
    perm = -np.ones(nb, dtype=int)
    used = set()
    for i in range(nb):
        d0 = np.linalg.norm(xb[0] - xa[0, i], axis=1)
        for j in np.argsort(d0):
            if j not in used:
                perm[i] = j
                used.add(int(j))
                break
    xb_al = xb[:, perm]
    x_abs = np.abs(xa - xb_al)
    print(f"xpos : max|Δ|={x_abs.max():.3e}  mean|Δ|={x_abs.mean():.3e} "
          f"(arena-body order aligned)")

    # scalar fluid time-series
    for key in ["ke", "umax", "vmax", "div_max", "p_absmax", "p_mean"]:
        if key not in a or key not in b:
            continue
        r = rel(a[key][:n], b[key][:n])
        absd = np.abs(np.asarray(a[key][:n], float) - np.asarray(b[key][:n], float))
        print(f"{key:9s}: max rel={r.max():.3e}  max|Δ|={absd.max():.3e} "
              f"(final native={float(a[key][n-1]):.6g} warp={float(b[key][n-1]):.6g})")

    # field snapshots (L2 over the full field)
    if "p_field" in a and "p_field" in b:
        m = min(len(a["p_field"]), len(b["p_field"]))
        for i in range(m):
            pa, pb = a["p_field"][i], b["p_field"][i]
            ua, ub = a["umag_field"][i], b["umag_field"][i]
            pl2 = np.linalg.norm(pa - pb) / max(np.linalg.norm(pa), 1e-30)
            ul2 = np.linalg.norm(ua - ub) / max(np.linalg.norm(ua), 1e-30)
            fs = int(a["field_step"][i])
            print(f"field@step{fs:4d}: rel-L2  p={pl2:.3e}  |u|={ul2:.3e}")

    # per-step wall-clock (CUDA-synced in the hook).  Skip the first 5 steps
    # (one-time graph capture / kernel warmup) and report the steady-state median.
    if "wall_step" in a and "wall_step" in b:
        wa = np.asarray(a["wall_step"], float)
        wb = np.asarray(b["wall_step"], float)
        sa = wa[5:] if len(wa) > 10 else wa
        sb = wb[5:] if len(wb) > 10 else wb
        print(f"\nper-step wall (steady-state, ms):  "
              f"native median={1e3*np.median(sa):.2f} mean={1e3*sa.mean():.2f}  | "
              f"warp median={1e3*np.median(sb):.2f} mean={1e3*sb.mean():.2f}  | "
              f"ratio(warp/native) median={np.median(sb)/np.median(sa):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["native", "warp", "compare"])
    ap.add_argument("--n", type=int, default=120, help="n_iterations")
    ap.add_argument("--every", type=int, default=1, help="scalar log cadence")
    ap.add_argument("--field-every", type=int, default=30,
                    help="full-field snapshot cadence (0=off)")
    ap.add_argument("--perf", action="store_true",
                    help="enable Warp Kernel-A CUDA-graph (perf run)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="relative velocity perturbation amplitude (chaos test)")
    ap.add_argument("--perturb-recurring", action="store_true",
                    help="apply the perturbation every step (emulate a residual-level backend)")
    args = ap.parse_args()

    if args.mode == "compare":
        compare(args.field_every)
    else:
        run(args.mode, args.n, args.every, args.field_every, args.perf,
            args.perturb, args.perturb_recurring)


if __name__ == "__main__":
    main()
