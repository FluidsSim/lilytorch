"""THE ORACLE, ON THE REAL FISH GEOMETRY — which readout is RIGHT, not merely different?

Every force oracle in this repo runs on a SPHERE (`BodyAnalytical`, |grad phi|=1,
one convex body, no joints).  The zebrafish is none of those: a 16-link MESH
body, union SDF, sheet-thin (median interior depth 0.61h vs a 2h delta band).
**Nobody has ever run an oracle on it.**  So every fish number to date is
eul-vs-lag — RELATIVE.  Nothing says which one is right, and the lagrangian is
the assumed reference only because it is exact on spheres.

This closes that.  Take the frozen snapshot's real geometry (per-link SDF
tables, kinematics, union SDF, triangulation) and swap the FLUID for an analytic
field with an exact answer via the divergence theorem — same two cases as
`oracle_native_three_way.py`, so the conventions are inherited, not reinvented:

    A)  p = -G*x,     u = 0        ->  F_p = G * V
    B)  u_x = c*y^2,  p = 0        ->  F_v = 2*nu*rho*c * V

Both are origin-independent (`Int n dS = 0` over a closed surface), so the fish
sitting anywhere in the tank does not matter.

WHY THIS IS THE DECISIVE TEST — it splits the error in two, which no
eul-vs-lag comparison can:

  * readout/GEOMETRY error   <- what this measures (analytic field, exact answer)
  * field/BDIM-CONTAMINATION error <- what it deliberately does NOT measure

Trap #0 applies and is the POINT here: an analytic field is clean everywhere,
including inside the body, so this says nothing about band contamination.  That
is the other half; we currently have zero of this half.

Reading the result:
  eul/exact ~ 0.636  => the measure deficit IS the whole readout story, and the
                        live 1.327 over-read is the FLUID FIELD, not geometry.
  eul/exact ~ 1.327  => a second geometric error is live.
  lag/exact ~ 1.0    => the lagrangian earns its reference status ON THIS
                        GEOMETRY (so far only ever earned on spheres).
  lag/exact != 1.0   => the reference everything else is measured against is
                        itself wrong.  Nobody has checked this.

    python -m lilytorch.examples.force_benchmarks.fish_geometry_oracle [snap_lagr.pt]
"""
from __future__ import annotations

import sys

import torch

from lilytorch.examples.force_benchmarks.shift_sweep_3d import _lagrangian, _read

DEFAULT_SNAP = "/data/andreaferrario/ns_data/zfish_force_snapshot/snap_lagr.pt"

G = 1.0        # pressure gradient   p = -G*x
CSH = 1.0      # shear coefficient   u_x = CSH*y^2


def volumes(snap, h):
    """The fish's enclosed volume, two INDEPENDENT ways.

    They bracket the 'exact' answer: the eulerian integrates the union SDF, the
    lagrangian integrates the triangulation, and the exact force is
    proportional to whichever volume that readout's surface actually encloses.
    Memory records these agree to ~0.3% -- far below the 33% effect at issue.
    """
    sdf = snap["sdf_cc"].double()
    V_sdf = float((sdf < 0).sum()) * h ** 3
    c = snap["tri_centroid"].double().T          # stored (3,N) -> (N,3)
    n = snap["tri_normal"].double().T
    a = snap["tri_area"].double()
    V_tri = float(((c * n).sum(1) * a).sum() / 3.0)   # Int x.n dS / 3
    return V_sdf, V_tri


def analytic_fields(snap, dev):
    """(caseA, caseB) as (u, v, w, p) flat tensors on the snapshot's own grid."""
    dt = snap["u"].dtype
    gx = snap["gx"].to(dev, dt)
    gy = snap["gy"].to(dev, dt)
    gz = snap["gz"].to(dev, dt)
    X, Y, _ = torch.meshgrid(gx, gy, gz, indexing="ij")
    z = torch.zeros_like(X).ravel().contiguous()
    pA = (-G * X).ravel().contiguous()
    uB = (CSH * Y ** 2).ravel().contiguous()
    return (z, z.clone(), z.clone(), pA), (uB, z.clone(), z.clone(), z.clone())


def with_fields(snap, fields, delta_order):
    s = dict(snap)
    s["u"], s["v"], s["w"], s["p"] = fields
    s["delta_order"] = delta_order
    return s


def main(path=DEFAULT_SNAP):
    snap = torch.load(path, weights_only=False)
    if "tri_centroid" not in snap:
        raise SystemExit(f"{path} has no triangulation (need a lagrangian snapshot)")
    h = snap["h"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nu_rho = snap["nu"] * snap["rho"]

    V_sdf, V_tri = volumes(snap, h)
    print(f"snapshot: {path}")
    print(f"  grid {snap['grid']}  {snap['n_bodies']} links  h={h:.4e}  "
          f"dtype {snap['u'].dtype}  device {dev}")
    print(f"  nu*rho = {nu_rho:.4e}   eps_body = {snap['eps_body']/h:.2f} h")
    print(f"\n  V(union SDF)      = {V_sdf:.5e} m^3")
    print(f"  V(Int x.n dS / 3) = {V_tri:.5e} m^3   "
          f"(tri/sdf = {V_tri/V_sdf:.4f})")

    fp_true = G * V_sdf
    fv_true = 2 * nu_rho * CSH * V_sdf
    print(f"\n  EXACT (using V_sdf):  F_p = {fp_true:.5e} N   F_v = {fv_true:.5e} N")
    print(f"  (using V_tri they are {G*V_tri:.5e} / "
          f"{2*nu_rho*CSH*V_tri:.5e} -- a {abs(V_tri/V_sdf-1)*100:.1f}% ambiguity,")
    print("   far below the ~33% effect under investigation)")

    A, B = analytic_fields(snap, dev)

    print("\n=== BOTH readouts vs the EXACT answer, at (off_p, off_f) = (0, 0) ===")
    print("  (0,0) is the only setting where both target the same integral.")
    print(f"\n{'readout':>22} | {'F_p':>13} {'F_p/exact':>10} | "
          f"{'F_v':>13} {'F_v/exact':>10}")

    rows = []
    for order in (1, 2):
        e_p = float(_read(with_fields(snap, A, order), 0.0, 0, dev,
                          off_pres=0.0).cpu()[:, 6].sum())
        e_v = float(_read(with_fields(snap, B, order), 0.0, 0, dev,
                          off_pres=0.0).cpu()[:, 0].sum())
        rows.append((f"eulerian order{order}", e_p, e_v))

    l_p = float(_lagrangian(with_fields(snap, A, 1), 0.0, dev,
                            off_pres=0.0).cpu()[:, 6].sum())
    l_v = float(_lagrangian(with_fields(snap, B, 1), 0.0, dev,
                            off_pres=0.0).cpu()[:, 0].sum())
    rows.append(("lagrangian", l_p, l_v))

    for name, fp, fv in rows:
        print(f"{name:>22} | {fp:13.5e} {fp/fp_true:10.3f} | "
              f"{fv:13.5e} {fv/fv_true:10.3f}")

    e1 = rows[0]
    print(f"\n  eulerian(order1)/lagrangian:  F_p {e1[1]/l_p:.3f}   "
          f"F_v {e1[2]/l_v:.3f}")
    print("  -> compare against the LIVE-FIELD ratios (1.327 / 1.551).  If the")
    print("     analytic-field ratio is much SMALLER, the live gap is dominated")
    print("     by the FLUID FIELD (band contamination), not by the geometry.")
    print("\n  Measure deficit for reference: A_coarea/A_tri = 0.636 on this fish.")
    print("  An eulerian at ~0.636x exact would mean the measure is the whole")
    print("  geometric story and NO integrand over-read is needed to explain it.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAP)
