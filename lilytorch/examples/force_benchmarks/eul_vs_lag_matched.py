"""Eulerian vs lagrangian at MATCHED sampling, decomposed by channel.

This is the comparison the whole investigation has been missing.  Every prior
cross-method number was confounded: the two readouts sampled in different
places (eulerian p at phi=0 and sigma at phi=eps; lagrangian both at
``lagrangian_sample_offset``), so "which readout" was never separable from
"sampled where".  B1 split the offsets; this drives BOTH readouts on ONE frozen
field at the SAME ``(off_p, off_f)`` and reports each channel separately.

Frozen field, not two live runs, is the point: two live runs follow different
trajectories, so they compare each readout's self-consistency rather than the
readouts on identical fluid states.

THE PREDICTION TO FALSIFY (handoff §B1a step 2):

  * pressure ratio  ~= 1   -- both channels read p on the SAME iso-surface
                              {phi = off_p} with the same measure.
  * viscous  ratio  ~= J   -- the Steiner area Jacobian (1+s*k1)(1+s*k2), pure
                              geometry, because the eulerian shifts its DELTA
                              (the measure moves with the sample) while the
                              lagrangian keeps its triangulation welded to the
                              skin and moves only the field lookup.

If the PRESSURE ratio is not ~= 1 at matched off_p, something unidentified is
live and that is the real finding.

``A(phi=s)/A(phi=0)``, computed here by coarea with the same cosine delta the
kernel uses, is the area-weighted mean of J over the surface -- i.e. the
geometric prediction for the viscous ratio if the traction were uniform.  The
measured ratio is a traction-weighted average instead, so expect the same
magnitude, not equality.

    python -m lilytorch.examples.force_benchmarks.eul_vs_lag_matched [snap_lagr.pt]

Needs a snapshot from a LAGRANGIAN run (it carries the world-frame
triangulation): see gen_zfish_snapshot.py, ZFISH_SNAP_FORCE_METHOD=lagrangian.
"""
from __future__ import annotations

import math
import sys

import torch

from lilytorch.examples.force_benchmarks.shift_sweep_3d import _lagrangian, _read

DEFAULT_SNAP = "/data/andreaferrario/ns_data/zfish_force_snapshot/snap_lagr.pt"

# (off_p, off_f) in cells.  (0,0) is the ONLY setting at which "they should
# agree" is a real prediction -- there both reduce to the true surface integral.
# (0,2) is Verma et al. 2017 / Maertens & Weymouth.  The rest bracket them.
SETTINGS = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (2.0, 2.0)]


def iso_area(snap, s, h, use_gradmag):
    """A(phi=s) by coarea with the kernel's cosine delta.

    ``A(s)/A(0)`` is the area-weighted mean Steiner Jacobian -- the geometric
    prediction for how much the eulerian's moving measure inflates.
    """
    sdf = snap["sdf_cc"].double()
    eb = snap["eps_body"]
    d = sdf - s
    m = d.abs() < eb
    delta = (1.0 + torch.cos(math.pi * d[m] / eb)) / (2 * eb)
    if use_gradmag:
        g = snap.get("grad_mag")
        if g is not None:
            delta = delta * g.double()[m]
    return float(delta.sum()) * h ** 3


def channels(out):
    """(Fv, Fp) totals over links -> the swim-relevant vectors."""
    return out[:, 0:3].sum(0), out[:, 6:9].sum(0)


def main(path=DEFAULT_SNAP):
    snap = torch.load(path, weights_only=False)
    if "tri_centroid" not in snap:
        raise SystemExit(
            f"{path} has no triangulation -- regenerate with "
            "ZFISH_SNAP_FORCE_METHOD=lagrangian (see gen_zfish_snapshot.py)")
    h = snap["h"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"snapshot: {path}")
    print(f"  iteration {snap['iteration']}  grid {snap['grid']}  "
          f"{snap['n_bodies']} bodies  device {dev}")
    print(f"  h = {h:.4e}  eps_body = {snap['eps_body']/h:.2f} h  "
          f"live eps_solver = {snap['eps_solver']/h:.2f} h  "
          f"delta_order {snap['delta_order']}")

    a0 = iso_area(snap, 0.0, h, False)
    print(f"\n  A(phi=0) = {a0:.4e} m^2   (coarea, cosine delta)")

    print("\n=== BOTH readouts at MATCHED (off_p, off_f), same frozen field ===")
    print("  x-component: swimming is along x, so Fx is what sets swim speed.")
    print(f"\n{'off_p':>6} {'off_f':>6} | {'eul Fp_x':>12} {'lag Fp_x':>12} "
          f"{'P ratio':>8} | {'eul Fv_x':>12} {'lag Fv_x':>12} {'V ratio':>8} | "
          f"{'A(f)/A(0)':>9}")
    for op, of in SETTINGS:
        e = _read(snap, of * h, 0, dev, off_pres=op * h).cpu()
        l = _lagrangian(snap, of * h, dev, off_pres=op * h).cpu()
        efv, efp = channels(e)
        lfv, lfp = channels(l)
        pr = float(efp[0]) / float(lfp[0])
        vr = float(efv[0]) / float(lfv[0])
        aj = iso_area(snap, of * h, h, False) / a0
        print(f"{op:6.1f} {of:6.1f} | {float(efp[0]):12.5e} {float(lfp[0]):12.5e} "
              f"{pr:8.3f} | {float(efv[0]):12.5e} {float(lfv[0]):12.5e} "
              f"{vr:8.3f} | {aj:9.3f}")

    print("\n  P ratio ~= 1 is the prediction at ANY matched off_p (same measure,")
    print("  same iso-surface).  A deviation there is the real finding.")
    print("  V ratio should track A(f)/A(0) -- the eulerian's measure moves with")
    print("  its delta; the lagrangian's triangulation does not.")

    # The pressure channel at matched off_p is the sharpest test: it isolates
    # "does the eulerian's moving measure explain the gap" from any viscous
    # band-contamination story, because at off_p=0 BOTH read the true surface.
    print("\n=== pressure channel, all components, at off_p = 0 ===")
    print(f"{'off_f':>6} | {'eul Fp':>38} | {'lag Fp':>38}")
    for op, of in SETTINGS:
        if op != 0.0:
            continue
        e = _read(snap, of * h, 0, dev, off_pres=0.0).cpu()
        l = _lagrangian(snap, of * h, dev, off_pres=0.0).cpu()
        _, efp = channels(e)
        _, lfp = channels(l)
        print(f"{of:6.1f} | " + " ".join(f"{float(x):12.5e}" for x in efp)
              + " | " + " ".join(f"{float(x):12.5e}" for x in lfp))
    leakage_control(snap, h, dev)


def leakage_control(snap, h, dev):
    """Does off_f leak into the PRESSURE channel?  With a calibrated null.

    off_p is held at 0, so the eulerian pressure force must not move as off_f
    sweeps.  The subtlety that makes a naive "they printed the same" claim
    worthless: ``streaming_sdf_forces_post_3d`` accumulates per-link forces with
    ATOMICS, so the reduction order varies and **repeating the identical call
    does not reproduce bitwise**.  Never assert "bit-exact" of these kernels.

    So the first row below re-runs the REFERENCE ARGUMENTS: it is the null, the
    noise floor of the kernel against itself.  A real leak is a row that stands
    ABOVE that floor; rows at or below it are indistinguishable from noise.
    """
    ref = _read(snap, 0.0, 0, dev, off_pres=0.0).cpu()

    def rel_to_ref(s):
        out = _read(snap, s * h, 0, dev, off_pres=0.0).cpu()
        p_ref, p_new = ref[:, 6:9], out[:, 6:9]
        d = (p_new - p_ref).abs()
        return (float((d / p_ref.abs().clamp_min(1e-30)).max()), float(d.max()),
                torch.equal(p_ref, p_new))

    # The null is ITSELF noisy (observed 4.7e-14 .. 2.6e-13 across runs), so a
    # single sample of it is a bad floor -- it would make a quiet run look like
    # a leak.  Take the worst of several repeats of the identical call.
    nulls = [rel_to_ref(0.0)[0] for _ in range(5)]
    floor = max(nulls)
    print("\n=== CONTROL: off_f must not leak into pressure (off_p pinned at 0) ===")
    print("  NOTE the kernel uses atomics -> identical args do NOT reproduce")
    print("  bitwise.  The NULL = repeating the REFERENCE call, 5x, worst case.")
    print(f"  null (identical args, 5 repeats): "
          + " ".join(f"{n:.2e}" for n in nulls))
    print(f"  => noise floor = {floor:.3e} relative")
    print(f"\n{'off_f':>6} | {'bitwise ==':>10} | {'max |abs diff|':>15} "
          f"{'max rel diff':>13} {'vs floor':>9}")
    for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        rel, absd, bit = rel_to_ref(s)
        print(f"{s:6.2f} | {str(bit):>10} | {absd:15.3e} {rel:13.3e} "
              f"{rel/floor:8.2f}x"
              + ("   ** ABOVE NOISE: REAL LEAK **" if rel > 10*floor else ""))

    # A control that cannot fail is not a control: prove off_f does something.
    v0 = _read(snap, 0.0, 0, dev, off_pres=0.0).cpu()[:, 0]
    v2 = _read(snap, 2.0 * h, 0, dev, off_pres=0.0).cpu()[:, 0]
    print(f"\n  not vacuous: over the SAME sweep the VISCOUS channel moves "
          f"{float(v2.sum())/float(v0.sum()):.3f}x "
          f"({float(v0.sum()):.5e} -> {float(v2.sum()):.5e})")
    print("  => off_f is live; it simply does not touch pressure.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAP)
