"""Generate the OPEN TANK arena for the two-phase sphere-drop demo.

Writes (into examples/sdfs/, the shared scratch location used by
base_sim_config):
  * pool/sdf/pool.sdf          -- 4 translucent side walls + floor, OPEN TOP,
                                  sized to the fluid box [0,1]x[0,1]x[0,2.5] so it
                                  encloses the air+water+interface.
  * arena_water/sdf/arena_water.sdf -- a translucent BULK water box filling
                                  z in [0, SURF] (SURF=1.8), leaving the air gap
                                  above visible. (Cosmetic: no FARMS swimming
                                  handler is loaded; the BDIM fluid governs.)

arena_config.yaml points at both. Run this BEFORE ``run.sh`` (run.sh does it for
you). Keep SURF in sync with the fluid ``alpha_init`` / ``water.height`` in
simulation_config.yaml.
"""
from lilytorch.integration.gen_pool_sdf import create_pool_sdf, create_water_sdf

# Fluid box + waterline in SI metres (must match simulation_config.yaml)
XMIN, XMAX = 0.0, 0.3
YMIN, YMAX = 0.0, 0.3
ZMIN, ZMAX = 0.0, 0.6
SURF = 0.4

if __name__ == "__main__":
    pool = create_pool_sdf(
        XMIN, XMAX, YMIN, YMAX, zmin=ZMIN, zmax=ZMAX,
        wall_thickness=0.02, wall_alpha=0.20, plotting=False,
    )
    water = create_water_sdf(
        XMIN, XMAX, YMIN, YMAX, zmin=ZMIN, zmax=SURF,   # bulk water fills 0..SURF
        water_height=0.0, water_alpha=0.30,
    )
    print(f"wrote tank   -> {pool}")
    print(f"wrote water  -> {water}  (fills z in [0, {SURF}])")
