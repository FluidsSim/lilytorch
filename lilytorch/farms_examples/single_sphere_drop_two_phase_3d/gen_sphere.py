"""Set the sphere's MASS (i.e. its density) for the two-phase coupling demo.
SI units: density in kg/m^3, water = 1000.

>>> EDIT ``DENSITY`` BELOW — it is the single sink/float control knob <<<
    DENSITY < 1000  -> lighter than water -> FLOATS  (tennis ball ~370)
    DENSITY > 1000  -> heavier than water -> SINKS
    DENSITY ~ 1000  -> hovers

WHY here and not sphere.yaml: FARMS computes the rigid body's mass/inertia from
the SDF ``<inertial>`` block and IGNORES ``morphology.links.density`` in the
animat yaml when the SDF carries inertials. So the body weight lives in
sphere.sdf, and the yaml ``density`` does nothing. This script rewrites
sphere.sdf's <mass>/<inertia> from DENSITY (solid sphere, the SDF's radius);
run.sh runs it before launch. (rho_body in simulation_config.yaml is a separate
FLUID setting — the band density for projection conditioning, NOT the weight.)
"""
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- control knob (SI, kg/m^3) ----------------------------------------------
DENSITY   = 900.0      # tennis ball ~370 kg/m^3 (floats). >1000 sinks, <1000 floats.
RHO_WATER = 1000.0
# -----------------------------------------------------------------------------


def regen(sdf_path, density):
    sdf = open(sdf_path).read()
    R = float(re.search(r"<radius>\s*([0-9.eE+-]+)\s*</radius>", sdf).group(1))
    V = 4.0 / 3.0 * math.pi * R ** 3
    mass = density * V
    inertia = 0.4 * mass * R ** 2                        # solid sphere: 2/5 m R^2
    sdf = re.sub(r"<mass>[^<]*</mass>", f"<mass>{mass:.12g}</mass>", sdf)
    for tag in ("ixx", "iyy", "izz"):
        sdf = re.sub(rf"<{tag}>[^<]*</{tag}>", f"<{tag}>{inertia:.12g}</{tag}>", sdf)
    sdf = re.sub(r"rho=[^,]*,", f"rho={density},", sdf)  # update the comment
    open(sdf_path, "w").write(sdf)
    return R, mass


if __name__ == "__main__":
    R, m = regen(os.path.join(HERE, "sphere.sdf"), DENSITY)
    verdict = "SINKS" if DENSITY > RHO_WATER else ("FLOATS" if DENSITY < RHO_WATER else "hovers")
    print(f"sphere.sdf: DENSITY={DENSITY} kg/m^3 (R={R} m) -> mass={m:.4e} kg  "
          f"[{verdict} vs water={RHO_WATER}]")
