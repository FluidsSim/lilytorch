"""C2 scene (b) — 3-D jellyfish (f32) end-to-end: native vs Warp backend.

The jellyfish is a STANDALONE driver (no FARMS/MuJoCo): it builds a
``TwoPhaseSolver``, swaps in an analytical pulsing ``JellyfishBody``, and runs
``solver.run_sim()``.  It runs on the **python** step path (a deforming SDF can't
use the rigid streaming Kernel A), so the Warp-backed ops exercised here are the
**advection flux, the (variable-density) Poisson smoother/residual, ``cvof_sweep``
(two-phase VOF), ``apply_bcs`` and the force readout** — NOT Kernel A/B.

Backend selection: the Warp backend is chosen via the ``solver.backend``
resolver (``lilytorch.src_warp.backend.resolve_solver_class``) — the same opt-in
path BDIMhandler and the standalone driver use — which returns the Warp
``src_warp.two_phase_solver.TwoPhaseSolver`` (it injects the Warp
``AdvDiffSolver``/``PoissonSolver`` + ``TwoPhase`` VOF field at construction and
routes the two-phase force readout to the Warp ports).  No ``src/`` edits.

Usage (from /data/andreaferrario/lilytorch):
    python -m lilytorch.validation.warp_e2e.run_c2_jelly native --n 60
    python -m lilytorch.validation.warp_e2e.run_c2_jelly warp   --n 60
    python -m lilytorch.validation.warp_e2e.run_c2_jelly compare
"""
import argparse
import os
import time

import numpy as np
import torch

NS_DATA = "/data/andreaferrario/ns_data"
DIRS = {
    "native": os.path.join(NS_DATA, "c2_native_jelly"),
    "warp": os.path.join(NS_DATA, "c2_warp_jelly"),
}


def _build(config_path, n_iter, dtype, perf=False, backend="native"):
    """Mirror ``run_jellyfish_fluid.build_solver`` but override nt + disable
    frame/drag saving for a short headless validation run."""
    import lilytorch.farms_examples.jellyfish.run_jellyfish_fluid as J
    from lilytorch.util.yaml_operations import yaml2pyobject

    pars = yaml2pyobject(config_path)
    pars["solver"]["nt"] = n_iter
    out = pars.setdefault("output", {})
    out["save_frames"] = False
    out["save"] = False
    out["save_uv"] = False

    # apply the driver's TWO_PHASE settings (mgcg + gravity + two_phase block),
    # then build via the driver's own builder logic by temporarily writing pars.
    s = pars["solver"]
    if J.TWO_PHASE:
        s["solver_method"] = "python"
        s["poisson_method"] = "mgcg"
        s["poisson_smoother"] = "rbgs"
        s.setdefault("poisson_max_mgcg_cycles", 30)
        s.setdefault("poisson_max_cycles", 30)
        s["gravity"] = J.GRAVITY
        s["two_phase"] = {
            "alpha_init": f"lambda X, Y, Z: (Z < {J.SURFACE_Z}).double()",
            "rho_water": J.RHO_WATER, "rho_air": J.RHO_AIR,
            "nu_water": float(s.get("nu", 1.0e-5)), "nu_air": J.NU_AIR,
            "face_density": "harmonic",
        }
        pars.setdefault("jellyfish", {})["gravity"] = J.GRAVITY

    if perf:
        # C1: graphed all-Warp variable-coefficient MGCG preconditioner.  The
        # native solver plumbs ``poisson_cuda_graph`` into the (Warp) PoissonSolver
        # (solver.py:545); the gate's cell cap is raised so 128**3 is not skipped.
        s["poisson_cuda_graph"] = True
        s["poisson_cuda_graph_max_cells"] = 256 ** 3

    out["existing_folder"] = os.path.join(NS_DATA, "_jelly_scratch")
    os.makedirs(out["existing_folder"], exist_ok=True)

    # Select the backend via the production resolver (the same
    # ``solver.backend`` path BDIMhandler / the standalone driver use).  This
    # exercises the real Warp ``TwoPhaseSolver`` subclass, not a scaffolding swap.
    from lilytorch.src_warp.backend import resolve_solver_class
    SolverCls = resolve_solver_class(backend, J.TWO_PHASE)
    solver = SolverCls(pars, dtype=dtype, compute_forces=True)

    from lilytorch.farms_examples.jellyfish.jellyfish_body import (
        JellyfishBody, JellyfishParams)
    jelly = JellyfishBody(
        device=solver.device, x=solver.x, y=solver.y, z=solver.z,
        eps=float(solver.eps), params=JellyfishParams.from_solver_config(pars))
    solver.composite_body = jelly
    solver.n_bodies = len(jelly.bodies)
    jelly.update(solver.starting_time, 0, dt=float(solver.dt))
    solver._recompute_mu_normals()
    jelly.clear_history()
    return solver, jelly


def run(backend, n_iter, field_every, perf=False):
    out_dir = DIRS[backend]
    os.makedirs(out_dir, exist_ok=True)

    import lilytorch.farms_examples.jellyfish.run_jellyfish_fluid as J
    solver, jelly = _build(J.DEFAULT_CONFIG, n_iter, torch.float32, perf=perf,
                           backend=backend)

    mod = type(solver.adv_diff_solver).__module__
    is_warp = "src_warp" in mod
    print(f"[jelly] backend={backend} adv_diff module={mod} (warp={is_warp})")
    assert is_warp == (backend == "warp"), f"backend mismatch: {mod}"
    if backend == "warp":
        assert "src_warp" in type(solver.poisson_solver).__module__
        assert "src_warp" in type(solver.two_phase).__module__

    log = {"ke": [], "umax": [], "wmax": [], "p_absmax": [], "p_mean": [],
           "alpha_sum": [], "field_step": [], "p_field": [], "umag_field": []}
    _orig_fin = solver.finalize_step

    def _fin(u, v, p, iteration, **kw):
        r = _orig_fin(u, v, p, iteration, **kw)
        w = kw.get("w_vel", None)
        with torch.no_grad():
            sq = u * u + v * v + (w * w if w is not None else 0)
            log["ke"].append(0.5 * float(sq.sum().item()))
            log["umax"].append(float(u.abs().max().item()))
            log["wmax"].append(float(w.abs().max().item()) if w is not None else 0.0)
            log["p_absmax"].append(float(p.abs().max().item()))
            log["p_mean"].append(float(p.mean().item()))
            log["alpha_sum"].append(float(solver.two_phase.alpha.sum().item()))
            if field_every and iteration % field_every == 0:
                log["field_step"].append(int(iteration))
                log["p_field"].append(p.detach().to(torch.float64).cpu().numpy().copy())
                log["umag_field"].append(
                    torch.sqrt(sq).detach().to(torch.float64).cpu().numpy().copy())
        return r

    solver.finalize_step = _fin
    t0 = time.time()
    solver.run_sim()
    dt = time.time() - t0

    out = {"backend": backend,
           "com": np.asarray(jelly.com_history, dtype=np.float64),
           "quat": np.asarray(jelly.quaternion_history, dtype=np.float64),
           "linvel": np.asarray(jelly.linear_velocity_history, dtype=np.float64)}
    for k, val in log.items():
        if val:
            out[k] = np.asarray(val)
    np.savez_compressed(os.path.join(out_dir, "jelly_log.npz"), **out)
    print(f"[jelly] backend={backend} n={n_iter} wall={dt:.1f}s "
          f"({len(log['ke'])} steps logged) -> {out_dir}/jelly_log.npz")


def compare():
    a = np.load(os.path.join(DIRS["native"], "jelly_log.npz"), allow_pickle=True)
    b = np.load(os.path.join(DIRS["warp"], "jelly_log.npz"), allow_pickle=True)
    n = min(len(a["com"]), len(b["com"]))
    print(f"\n=== C2(b) jellyfish native-vs-warp (first {n} body states) ===")
    for key in ["com", "quat", "linvel"]:
        d = np.abs(a[key][:n] - b[key][:n])
        sc = np.maximum(np.abs(a[key][:n]).max(), 1e-30)
        print(f"{key:7s}: max|Δ|={d.max():.3e}  rel(/{sc:.2e})={d.max()/sc:.3e}")

    def rel(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        den = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-30)
        return np.abs(x - y) / den
    m = min(len(a["ke"]), len(b["ke"]))
    for key in ["ke", "umax", "wmax", "p_absmax", "alpha_sum"]:
        if key in a and key in b:
            r = rel(a[key][:m], b[key][:m])
            print(f"{key:10s}: max rel={r.max():.3e} "
                  f"(final native={float(a[key][m-1]):.6g} warp={float(b[key][m-1]):.6g})")
    if "p_field" in a and "p_field" in b:
        k = min(len(a["p_field"]), len(b["p_field"]))
        for i in range(k):
            pa, pb = a["p_field"][i], b["p_field"][i]
            ua, ub = a["umag_field"][i], b["umag_field"][i]
            pl = np.linalg.norm(pa - pb) / max(np.linalg.norm(pa), 1e-30)
            ul = np.linalg.norm(ua - ub) / max(np.linalg.norm(ua), 1e-30)
            print(f"field@step{int(a['field_step'][i]):4d}: rel-L2 p={pl:.3e} |u|={ul:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["native", "warp", "compare"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--field-every", type=int, default=20)
    ap.add_argument("--perf", action="store_true",
                    help="enable C1 graphed-MGCG (poisson_cuda_graph) for the warp run")
    args = ap.parse_args()
    if args.mode == "compare":
        compare()
    else:
        run(args.mode, args.n, args.field_every, args.perf)


if __name__ == "__main__":
    main()
