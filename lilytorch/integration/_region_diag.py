"""Region-resolved diagnostics to localize the mu0_projection=True overlap blow-up.

Splits the post-projection fields by the cell-centred BDIM kernel mu0 (computed
from the union SDF) into three regions:
  * interior : mu0 < 0.01      (deep solid; pressure cannot act when mu0-weighted)
  * band     : 0.01 ≤ mu0 ≤ 0.99
  * fluid    : mu0 > 0.99
and logs max|u|, max|div u|, max|p| in each, so we can see which region's field
grows first as the run approaches blow-up.

Inject via _extra_run_patch: ``import lilytorch.integration._region_diag as _d;_d.install()``.
Writes $LILY_REGION_CSV (default ./region_diag.csv).
"""
import os, csv, torch

_CSV = os.environ.get("LILY_REGION_CSV", "region_diag.csv")
_w = None; _fh = None; _installed = False


def _open():
    global _w, _fh
    if _w is not None:
        return
    _fh = open(_CSV, "w", newline="")
    _w = csv.writer(_fh)
    _w.writerow([
        "iter",
        "u_fl", "u_bd", "u_in",      # max|u| per region
        "div_fl", "div_bd", "div_in",  # max|div u| per region
        "p_fl", "p_bd", "p_in",      # max|p| per region
        "nin", "nbd",                # cell counts (interior, band)
    ])
    _fh.flush()


def _mu0_cc(self):
    phi = self.composite_body.sdf_val
    eps = float(self.eps)
    deps = (phi / eps).clamp(-1.0, 1.0)
    return 0.5 * (1.0 + deps + torch.sin(torch.pi * deps) / torch.pi)


def _rmax(field, mask):
    if not bool(mask.any()):
        return 0.0
    return float(field.abs()[mask].amax())


def _log(self, iteration):
    try:
        _open()
        u, v = self.u0, self.v0
        w = getattr(self, "w0", None) if self.ndim == 3 else None
        umag = torch.sqrt(u*u + v*v + (w*w if w is not None else 0.0))
        div = self.divergence(u, v, w=w)
        p = self.p0
        mu0 = _mu0_cc(self)
        interior = mu0 < 0.01
        band = (mu0 >= 0.01) & (mu0 <= 0.99)
        fluid = mu0 > 0.99
        row = [iteration,
               _rmax(umag, fluid), _rmax(umag, band), _rmax(umag, interior),
               _rmax(div, fluid),  _rmax(div, band),  _rmax(div, interior),
               _rmax(p, fluid),    _rmax(p, band),    _rmax(p, interior),
               int(interior.sum()), int(band.sum())]
        _w.writerow([row[0]] + [f"{x:.6e}" if isinstance(x, float) else x
                                for x in row[1:]])
        if iteration % 10 == 0:
            _fh.flush()
    except Exception as exc:
        print(f"[region_diag] err iter {iteration}: {exc}")


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
    print(f"[region_diag] installed → {os.path.abspath(_CSV)}")
