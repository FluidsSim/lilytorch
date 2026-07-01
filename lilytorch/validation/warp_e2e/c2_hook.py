"""C2 end-to-end coupled-trajectory logging hook (native vs Warp backend).

Injected into a FARMS/MuJoCo coupled run via the ``_extra_run_patch`` seam of
``BaseSimConfig`` (so it touches no core source and no FARMS package).  ``install``:

  * (Warp run) swaps ``BDIMhandler.FluidSolver`` / ``BDIMhandler.TwoPhaseSolver``
    for the ``src_warp`` backend **before** the handler constructs the solver —
    the only backend-selection seam (there is no config key); asserts the swap
    took effect on the first logged step.
  * wraps ``BDIMhandler.step`` to record, at a fixed cadence, the MuJoCo body
    trajectory (``qpos``/``xpos``) and key fluid fields (KE, max|u|, pressure
    stats, plus full ``p`` / |u| field snapshots every ``field_every`` steps) so
    a native run and a Warp run of the IDENTICAL config can be compared offline.

Output is an ``.npz`` written on interpreter exit (``atexit``).
"""
import atexit
import time

import numpy as np
import torch


def install(backend, out_path, every=1, field_every=30, max_steps=None,
            perturb=0.0, perturb_recurring=False):
    import lilytorch.integration.BDIMhandler as bh

    # ── backend selection ─────────────────────────────────────────────────────
    # The backend is now chosen by the ``solver.backend`` config key (set by the
    # runner's ``_bdim_extension``), which BDIMhandler resolves via
    # ``src_warp.backend.resolve_solver_class``.  This hook only *verifies* the
    # selection took effect (asserted on the first logged step) and does the
    # per-step logging — no symbol monkeypatching needed.

    log = {
        "step": [], "qpos": [], "xpos": [],
        "ke": [], "umax": [], "vmax": [], "div_max": [],
        "p_absmax": [], "p_min": [], "p_max": [], "p_mean": [],
        "field_step": [], "p_field": [], "umag_field": [],
        "wall_step": [],
    }
    state = {"n": 0, "checked": False, "perturbed": False}
    _orig_step = bh.BDIMhandler.step
    _cuda = torch.cuda.is_available()

    def _wrapped(self, task, physics):
        # Chaos control: seed a one-shot relative perturbation of size ``perturb``
        # into the velocity field on the FIRST step, to measure the coupled
        # system's Lyapunov sensitivity (does a native run perturbed at the
        # backend-difference level diverge like the Warp run?).
        if perturb and (perturb_recurring or not state["perturbed"]):
            first = not state["perturbed"]
            state["perturbed"] = True
            fs = self.fluid_solver
            seed = 1234 + (state["n"] if perturb_recurring else 0)
            with torch.no_grad():
                for f in (fs.u0, fs.v0, getattr(fs, "w0", None)):
                    if f is None:
                        continue
                    g = torch.empty_like(f).normal_(generator=torch.Generator(
                        device=f.device).manual_seed(seed))
                    f.add_(perturb * f.abs() * g)
            if first:
                print(f"[c2_hook] perturbation eps={perturb} "
                      f"recurring={perturb_recurring}")
        if _cuda:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        ret = _orig_step(self, task, physics)
        if _cuda:
            torch.cuda.synchronize()
        log["wall_step"].append(time.perf_counter() - _t0)
        n = state["n"]
        state["n"] = n + 1
        if max_steps is not None and n >= max_steps:
            return ret

        if not state["checked"]:
            mod = type(self.fluid_solver).__module__
            is_warp = "src_warp" in mod
            print(f"[c2_hook] backend={backend!r} fluid_solver module={mod} "
                  f"(warp={is_warp})")
            assert is_warp == (backend == "warp"), (
                f"backend swap mismatch: requested {backend!r} but solver "
                f"module is {mod}")
            state["checked"] = True

        if n % every != 0:
            return ret

        fs = self.fluid_solver
        data = physics.data
        with torch.no_grad():
            u, v = fs.u0, fs.v0
            w = getattr(fs, "w0", None)
            sq = u * u + v * v
            if w is not None:
                sq = sq + w * w
            ke = 0.5 * float(sq.sum().item())
            umax = float(u.abs().max().item())
            vmax = float(v.abs().max().item())
            p = fs.p0
            log["ke"].append(ke)
            log["umax"].append(umax)
            log["vmax"].append(vmax)
            log["p_absmax"].append(float(p.abs().max().item()))
            log["p_min"].append(float(p.min().item()))
            log["p_max"].append(float(p.max().item()))
            log["p_mean"].append(float(p.mean().item()))
            # crude divergence proxy on the staggered interior (du/dx + dv/dy)
            try:
                div = (u[1:, :-1] - u[:-1, :-1]) + (v[:-1, 1:] - v[:-1, :-1])
                log["div_max"].append(float(div.abs().max().item()))
            except Exception:
                log["div_max"].append(float("nan"))

        log["step"].append(n)
        log["qpos"].append(np.asarray(data.qpos, dtype=np.float64).copy())
        log["xpos"].append(np.asarray(data.xpos, dtype=np.float64).copy())

        if field_every and (n % field_every == 0):
            with torch.no_grad():
                pf = fs.p0.detach().to(torch.float64).cpu().numpy().copy()
                um = torch.sqrt(sq).detach().to(torch.float64).cpu().numpy().copy()
            log["field_step"].append(n)
            log["p_field"].append(pf)
            log["umag_field"].append(um)
        return ret

    bh.BDIMhandler.step = _wrapped

    def _save():
        out = {}
        for k, val in log.items():
            if len(val) == 0:
                continue
            try:
                out[k] = np.asarray(val)
            except Exception:
                out[k] = np.asarray(val, dtype=object)
        np.savez_compressed(out_path, backend=backend, **out)
        print(f"[c2_hook] wrote {out_path} ({len(log['step'])} logged steps, "
              f"{len(log['p_field'])} field snapshots)")

    atexit.register(_save)
    print(f"[c2_hook] installed: backend={backend!r} out={out_path} "
          f"every={every} field_every={field_every} max_steps={max_steps}")
