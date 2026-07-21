"""Sweep the eulerian viscous-band shift on ONE frozen zebrafish scene.

The 3-D twin of ``shift_sweep_2d.py``, and the cheap test of the open zebrafish
question in the handoff (§9.5 item 1).  ``streaming_sdf_forces_post_3d`` is a
pure function of (geometry, poses, fluid fields, eps), so a snapshot taken from
a live coupled run (``gen_zfish_snapshot.py``) can be re-read at any band shift
without re-running the simulation.

What it measures: the viscous band is centred at ``φ = eps_solver`` rather than
at the body surface, so the readout is really "σ sampled ~eps_solver off the
wall".  On the cylinder (R/h≈25) that cost ~11% of the viscous force.  The
zebrafish is ~2-3 cells thick, where eps_solver = 2h is a LARGE fraction of the
body radius — if the readout swings O(1) across the sweep here, the offset is
the zebrafish story; if it is flat, the swim-speed gap is something else.

    python -m lilytorch.examples.force_benchmarks.shift_sweep_3d [snap.pt]
"""
from __future__ import annotations

import sys

import torch

from lilytorch.src.native import streaming_sdf_forces_post_3d

DEFAULT_SNAP = ("/data/andreaferrario/ns_data/zfish_force_snapshot/snap.pt")


def _read(snap, off_visc, submethod, device="cpu", off_pres=0.0):
    """Drive the native readout at one band shift -> (B,12) force/torque rows.

    ``off_pres`` defaults to 0 — the production convention, where only the
    viscous channel is shifted.  Pass it explicitly to pin this readout to the
    same sampling locations as the lagrangian one.
    """
    B = snap["n_bodies"]
    out = torch.zeros((B, 12), dtype=torch.float64, device=device)
    nu_rho = torch.tensor([snap["nu"] * snap["rho"]],
                          dtype=snap["u"].dtype, device=device)
    streaming_sdf_forces_post_3d(
        snap["F_flat"].to(device), snap["F_offsets"].to(device),
        snap["body_shapes"].to(device), snap["body_meta"].to(device),
        snap["kin"].to(device), snap["aabb_lo"].to(device),
        snap["aabb_dim"].to(device),
        snap["gx"].to(device), snap["gy"].to(device), snap["gz"].to(device),
        snap["h"], snap["max_vol"],
        snap["sdf_cc"].to(device), snap["interp_method"],
        snap["u"].to(device), snap["v"].to(device),
        snap["w"].to(device), snap["p"].to(device),
        nu_rho,
        snap["eps_body"],          # delta WIDTH: held fixed across the sweep
        off_pres,                  # pressure delta centre
        off_visc,                  # viscous band SHIFT: the swept quantity
        snap["h3"], snap["delta_order"], out,
        submethod, snap["ph_blend_cells"] * snap["h"],
    )
    return out


def _lagrangian(snap, sample_offset, device="cpu", off_pres=None):
    """Drive the lagrangian readout on the same frozen field -> (B,12).

    Only available on snapshots taken from a lagrangian run: the world-frame
    triangulation is built/refreshed by BDIMhandler on that path only.

    ``sample_offset`` moves BOTH channels unless ``off_pres`` is given (the
    legacy single-knob semantics this sweep was written against).
    """
    from lilytorch.src.forces import _viscous_stress_tensor

    if "tri_centroid" not in snap:
        return None
    dt = snap["p"].dtype
    nx, ny, nz = snap["grid"]
    u = snap["u"].to(device).reshape(nx, ny, nz)
    v = snap["v"].to(device).reshape(nx, ny, nz)
    w = snap["w"].to(device).reshape(nx, ny, nz)
    p = snap["p"].to(device).reshape(nx, ny, nz)
    e = _viscous_stress_tensor((u, v, w), snap["h"])
    B = snap["n_bodies"]
    out = torch.zeros((B, 12), dtype=torch.float64, device=device)
    inv = 1.0 / snap["h"]
    torch.ops.lilytorch_kernels.lagrangian_forces_3d.default(
        e[0][0].contiguous(), e[1][1].contiguous(), e[2][2].contiguous(),
        e[0][1].contiguous(), e[0][2].contiguous(), e[1][2].contiguous(),
        p.contiguous(),
        torch.tensor([snap["nu"] * snap["rho"]], dtype=dt, device=device),
        snap["tri_centroid"].to(device, dt), snap["tri_normal"].to(device, dt),
        snap["tri_area"].to(device, dt), snap["tri_offsets"].to(device),
        snap["com_pos"].to(device, dt),
        snap["x0"], snap["y0"], snap["z0"], inv, inv, inv,
        nx, ny, nz, 0,
        float(sample_offset if off_pres is None else off_pres),
        float(sample_offset),
        out,
    )
    return out


def main(path=DEFAULT_SNAP):
    snap = torch.load(path, weights_only=False)
    h = snap["h"]
    live = snap["eps_solver"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"snapshot: {path}")
    print(f"  iteration {snap['iteration']}  grid {snap['grid']}  "
          f"{snap['n_bodies']} bodies  device {dev}")
    print(f"  h = {h:.4e}   live eps_solver = {live:.4e} = {live/h:.2f} h   "
          f"eps_body = {snap['eps_body']:.4e} = {snap['eps_body']/h:.2f} h")
    print(f"  delta_order {snap['delta_order']}  interp {snap['interp_method']}"
          f"  submethod {snap['force_submethod']}")

    # Swimming is along x; total viscous x-force over all links is what sets
    # swim speed, so that is the number to watch.
    print("\n=== eulerian: sweep viscous-band shift eps_solver ===")
    print(f"{'s/h':>5} | {'Fv_x':>12} {'Fv_y':>12} {'Fv_z':>12} | "
          f"{'Fp_x':>12} | {'|Fv|/|Fv@2h|':>13}")
    ref = None
    for s in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        out = _read(snap, s * h, 0, dev).cpu()
        fv = out[:, 0:3].sum(0)
        fp = out[:, 6:9].sum(0)
        if abs(s - 2.0) < 1e-12:
            ref = float(fv[0])
        print(f"{s:5.2f} | {fv[0]:12.5e} {fv[1]:12.5e} {fv[2]:12.5e} | "
              f"{fp[0]:12.5e} | ", end="")
        print(f"{float(fv[0])/ref:13.3f}" if ref else f"{'—':>13}")

    if ref is not None:
        print(f"\n(ratios vs the live setting s=2h, Fv_x={ref:.5e})")

    # Per-link viscous x at the live shift vs the surface limit: the thin tail
    # links are where R/h is smallest, so the offset should bite hardest there.
    print("\n=== per-link viscous Fx: s=0 vs s=2h (live) ===")
    a = _read(snap, 0.0, 0, dev).cpu()
    b = _read(snap, live, 0, dev).cpu()
    print(f"{'link':>4} {'Fv_x(s=0)':>13} {'Fv_x(s=2h)':>13} {'ratio':>8}")
    for i in range(snap["n_bodies"]):
        r = (float(b[i, 0]) / float(a[i, 0])
             if abs(float(a[i, 0])) > 1e-20 else float("nan"))
        print(f"{i:>4} {float(a[i,0]):13.5e} {float(b[i,0]):13.5e} {r:8.3f}")
    print(f"{'sum':>4} {float(a[:,0].sum()):13.5e} {float(b[:,0].sum()):13.5e} "
          f"{float(b[:,0].sum())/float(a[:,0].sum()):8.3f}")

    # The lagrangian readout on the SAME field: its sample offset is the same
    # kind of knob (where off the wall σ is read), so the two readouts should be
    # compared as functions of that distance, not as "two methods".
    if "tri_centroid" in snap:
        print("\n=== lagrangian: sweep sample offset (same frozen field) ===")
        print(f"{'off/h':>6} | {'Fv_x':>12} {'Fp_x':>12}")
        for o in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            out = _lagrangian(snap, o * h, dev)
            if out is None:
                break
            out = out.cpu()
            print(f"{o:6.2f} | {out[:,0].sum():12.5e} {out[:,6].sum():12.5e}")

        # Matched sampling distance.  The tempting idea is that the two
        # readouts are the same "sample sigma at distance s" device and should
        # agree once s == off; they do NOT, and the gap is structural:
        #   eulerian(s)  = closed integral over the OFFSET ISO-SURFACE {phi=s},
        #                  whose measure inflates with s;
        #   lagrangian(o)= closed integral over the TRUE body surface (fixed
        #                  triangulated area), merely SAMPLING sigma at o.
        # On the sphere oracle they agree to 0.5% at s=0 and then diverge as
        # ((R+s)/R)^3 exactly.  See test_forces.py::test_oracle_* and §10.3.
        print("\n=== matched distance: eulerian shift s vs lagrangian offset s ===")
        print(f"{'s/h':>5} {'eul Fv_x':>12} {'lag Fv_x':>12} {'eul/lag':>8}")
        for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            e_ = float(_read(snap, s * h, 0, dev).cpu()[:, 0].sum())
            l_ = float(_lagrangian(snap, s * h, dev).cpu()[:, 0].sum())
            print(f"{s:5.2f} {e_:12.5e} {l_:12.5e} {e_/l_:8.3f}")

        # True triangulated area vs the union iso-surface area (coarea, with
        # the SAME cosine delta the kernel uses).  A ratio well above 1 means
        # the per-link triangulations carry faces buried inside the union --
        # they sample interior fields at full area weight.
        sdf_ = snap["sdf_cc"].double()
        eb = snap["eps_body"]
        d_ = sdf_ - 0.0
        m_ = d_.abs() < eb
        a_eff = float((((1.0 + torch.cos(3.141592653589793 * d_[m_] / eb))
                        / (2 * eb)).sum()) * h ** 3)
        a_tri = float(snap["tri_area"].double().sum())
        print(f"\n  triangulated area   Sum(tri_area) = {a_tri:.4e} m^2")
        print(f"  union iso-surface   A(phi=0)      = {a_eff:.4e} m^2"
              f"   ratio {a_tri/a_eff:.2f}x")
        if a_tri / a_eff > 1.1:
            print("  -> the triangulation carries buried (inter-link) faces; "
                  "they cancel pairwise in the TOTAL but not PER LINK.")

        # Net x-force is what sets swim speed: thrust (pressure) minus drag
        # (viscous).  Comparing the two readouts AT THEIR LIVE SETTINGS is the
        # apples-to-apples number behind "eulerian swims slower".
        e = _read(snap, live, 0, dev).cpu()
        l = _lagrangian(snap, 0.0, dev).cpu()
        en = float(e[:, 0].sum() + e[:, 6].sum())
        ln = float(l[:, 0].sum() + l[:, 6].sum())
        print("\n=== net Fx at the LIVE settings (eulerian s=2h, lagrangian off=0) ===")
        print(f"  eulerian   Fv_x {float(e[:,0].sum()):12.5e}  "
              f"Fp_x {float(e[:,6].sum()):12.5e}  net {en:12.5e}")
        print(f"  lagrangian Fv_x {float(l[:,0].sum()):12.5e}  "
              f"Fp_x {float(l[:,6].sum()):12.5e}  net {ln:12.5e}")
        print(f"  viscous  eulerian/lagrangian = "
              f"{float(e[:,0].sum())/float(l[:,0].sum()):.2f}x")
        print(f"  NET Fx   eulerian/lagrangian = {en/ln:.2f}x  "
              f"<- less net thrust => slower swimming")
    else:
        print("\n(no triangulation in this snapshot — rerun the generator with "
              "ZFISH_SNAP_FORCE_METHOD=lagrangian for the lagrangian sweep)")

    # Geometric context: the delta band has half-width eps_body, so on a body
    # only a few cells thick it is not a surface-localised measure at all.
    sdf = snap["sdf_cc"].double()
    inside = sdf[sdf < 0]
    if inside.numel():
        V0 = float((sdf < 0).sum())
        print(f"\n=== why it is so large here: the body is thinner than the band ===")
        print(f"  max inscribed radius = {-float(inside.min())/h:.2f} h   "
              f"eps_body = {snap['eps_body']/h:.2f} h")
        print(f"  interior cells within eps_body of the surface: "
              f"{float((-inside < snap['eps_body']).double().mean())*100:.1f}%")
        print(f"  enclosed-volume ratio V(phi<eps_solver)/V(phi<0) = "
              f"{float((sdf < snap['eps_solver']).sum())/V0:.3f}"
              f"   (the divergence-theorem inflation factor)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAP)
