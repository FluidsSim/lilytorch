"""Ad-hoc per-step stability diagnostics for the overlap/force_relaxation study.

Injected into a FARMS sim subprocess via ``_extra_run_patch`` →
``import lilytorch.integration._overlap_diag as _d; _d.install()``.

Monkeypatches ``FluidSolver.check_explosion`` (called once per step, after the
projection and the force integration) to append a CSV row capturing the
quantities relevant to the explicit-coupling instability hypothesis:

    iter, max|u|, max|div u|, max|p|, argmax|p| (i,j,k),
    raw per-body linear-force magnitude (sum and max over bodies)

The per-body force caches read here (``friction_force_lin_*`` / ``pressure_force_*``)
are the RAW per-step forces computed by the solver — i.e. BEFORE the BDIMhandler
``force_relaxation`` low-pass — so the oscillation can be seen even when
relaxation is masking it at the coupling boundary.

Output path: ``$LILY_DIAG_CSV`` if set, else ``./overlap_diag.csv`` (cwd is the
generated run folder).  Non-fatal: any logging error is swallowed so the study
never perturbs the physics.
"""
import os
import csv
import torch

_CSV_PATH = os.environ.get("LILY_DIAG_CSV", "overlap_diag.csv")
_writer = None
_fh = None
_installed = False


def _open():
    global _writer, _fh
    if _writer is not None:
        return
    _fh = open(_CSV_PATH, "w", newline="")
    _writer = csv.writer(_fh)
    _writer.writerow([
        "iter", "umax", "divmax", "pmax", "pargmax_i", "pargmax_j",
        "pargmax_k", "force_lin_sum", "force_lin_max",
        "visc_lin_max", "pres_lin_max",
    ])
    _fh.flush()


def _log(self, iteration):
    try:
        _open()
        u, v = self.u0, self.v0
        w = getattr(self, "w0", None) if self.ndim == 3 else None
        umax = float(torch.stack([a.abs().amax() for a in
                     ((u, v, w) if self.ndim == 3 else (u, v))]).max())
        div = self.divergence(u, v, w=w)
        divmax = float(div.abs().amax())
        p = self.p0
        pabs = p.abs()
        pmax = float(pabs.amax())
        flat = int(torch.argmax(pabs))
        idx = list(torch.unravel_index(torch.tensor(flat), p.shape))
        pi = int(idx[0]); pj = int(idx[1]); pk = int(idx[2]) if self.ndim == 3 else -1

        # raw per-body linear force magnitude (pre-relaxation), split into
        # viscous (friction) and pressure contributions so we can see which
        # term drives the buried-marker runaway.
        if self.ndim == 3:
            vx, vy, vz = self.friction_force_lin_x, self.friction_force_lin_y, self.friction_force_lin_z
            px_, py_, pz_ = self.pressure_force_x, self.pressure_force_y, self.pressure_force_z
            vmag = torch.sqrt(vx*vx + vy*vy + vz*vz)
            pmag = torch.sqrt(px_*px_ + py_*py_ + pz_*pz_)
            fx, fy, fz = vx+px_, vy+py_, vz+pz_
            mag = torch.sqrt(fx*fx + fy*fy + fz*fz)
        else:
            vx, vy = self.friction_force_lin_x, self.friction_force_lin_y
            px_, py_ = self.pressure_force_x, self.pressure_force_y
            vmag = torch.sqrt(vx*vx + vy*vy)
            pmag = torch.sqrt(px_*px_ + py_*py_)
            fx, fy = vx+px_, vy+py_
            mag = torch.sqrt(fx*fx + fy*fy)
        fsum = float(mag.sum())
        fmax = float(mag.amax())
        vmax_f = float(vmag.amax())
        pmax_f = float(pmag.amax())

        _writer.writerow([iteration, f"{umax:.6e}", f"{divmax:.6e}",
                          f"{pmax:.6e}", pi, pj, pk,
                          f"{fsum:.6e}", f"{fmax:.6e}",
                          f"{vmax_f:.6e}", f"{pmax_f:.6e}"])
        if iteration % 20 == 0:
            _fh.flush()
    except Exception as exc:  # never perturb the physics
        print(f"[overlap_diag] log error at iter {iteration}: {exc}")


def install():
    global _installed
    if _installed:
        return
    from lilytorch.src.solver import FluidSolver
    _orig = FluidSolver.check_explosion

    def _patched(self, iteration):
        _log(self, iteration)
        return _orig(self, iteration)

    FluidSolver.check_explosion = _patched
    _installed = True
    print(f"[overlap_diag] installed; logging to {os.path.abspath(_CSV_PATH)}")
