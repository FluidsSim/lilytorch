"""Generate a ventral-ballast 1guilla SDF: lower every link's COM by dz.

Keeps mass, inertia and geometry identical to 1guilla_800.sdf (so total weight
and the BDIM buoyancy are unchanged, i.e. still uniform rho=800), but shifts
each link's inertial <pose> z down by dz. That puts the centre of mass below the
centre of buoyancy -> a passive pendulum righting moment that resists roll/pitch
without any active control, like a keel / heavy belly.

MuJoCo uses the SDF <mass>/<inertia>/COM directly; buoyancy (BDIM) depends only
on the mesh shape, so this changes ONLY the roll/pitch stability.

Usage: python make_ballast_sdf.py <dz_metres> <out.sdf>
"""

import re
import sys

SRC = "1guilla_800.sdf"


def main():
    dz = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    out = sys.argv[2] if len(sys.argv) > 2 else "1guilla_ballast.sdf"
    sdf = open(SRC).read()

    def lower_com(m):
        # <inertial> ... <pose>x y z r p y</pose>
        block = m.group(0)

        def shift(pm):
            v = pm.group(1).split()
            v[2] = f"{float(v[2]) - dz:.6g}"      # lower z (down = -z)
            return f"<pose>{' '.join(v)}</pose>"

        return re.sub(r"<pose>([^<]+)</pose>", shift, block, count=1)

    new = re.sub(r"<inertial>.*?</inertial>", lower_com, sdf, flags=re.DOTALL)
    with open(out, "w") as f:
        f.write(new)
    print(f"dz={dz} m -> {out}  (mass/inertia/buoyancy unchanged; COM lowered)")


if __name__ == "__main__":
    main()
