#!/usr/bin/env python3
"""Float-stability DEMO: a compact SPHERE vs an elongated SPHEROID floating at
the waterline, same two-phase config, only the SHAPE differs.

This is the controlled experiment behind the conclusion "static float at the
waterline is the unstable regime, and the instability scales with the body's
waterplane": the sphere (point contact) floats forever; the volume-MATCHED 11:1
spheroid (a long waterline strip) explodes.

Both bodies have IDENTICAL mass (0.26 kg), density (~496 kg/m^3 -> floats half
submerged), volume, and spawn (centre AT the waterline). Run::

    cd lilytorch/farms_examples/boat
    FLOAT_BODY=sphere   python3 gen_float_demo.py     # stable, floats
    FLOAT_BODY=spheroid python3 gen_float_demo.py     # explodes (long waterline)

The live blue air/water surface is the FlowIsoGLViewer (inherited from
SmallSimConfig). Tip: ``pip install torchmcubes`` for a realtime surface.
"""
import math
import os

from gen_configs_small import SmallSimConfig, WATERLINE

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = os.environ.get("FLOAT_BODY", "sphere").lower()
R    = 0.05                       # sphere radius
MASS = 0.26                       # both bodies (density ~496 -> floats half)


def _write_uvsphere_obj(path, sx, sy, sz, nlat=24, nlon=48):
    """UV sphere of radius R scaled by (sx,sy,sz) -> sphere or prolate spheroid."""
    verts = [(0.0, 0.0, -R * sz)]
    for j in range(1, nlat):
        phi = math.pi * j / nlat
        sp, cp = math.sin(phi), math.cos(phi)
        for i in range(nlon):
            th = 2 * math.pi * i / nlon
            verts.append((R * sp * math.cos(th) * sx,
                          R * sp * math.sin(th) * sy, R * cp * sz))
    verts.append((0.0, 0.0, R * sz))
    faces = []
    for i in range(nlon):
        faces.append((1, 2 + i, 2 + (i + 1) % nlon))
    for j in range(nlat - 2):
        rt, rb = 1 + j * nlon, 1 + (j + 1) * nlon
        for i in range(nlon):
            ni = (i + 1) % nlon
            faces.append((rt + i + 1, rb + i + 1, rb + ni + 1))
            faces.append((rt + i + 1, rb + ni + 1, rt + ni + 1))
    vn = len(verts); rl = 1 + (nlat - 2) * nlon
    for i in range(nlon):
        faces.append((rl + i + 1, rl + (i + 1) % nlon + 1, vn))
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fc in faces:
            f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")


def _ensure_assets():
    """Generate the volume-matched OBJ + SDF for the requested body."""
    if BODY == "sphere":
        sx = sy = sz = 1.0
    elif BODY == "spheroid":
        s = 1.0 / math.sqrt(5.0)          # x*5, y,z/sqrt5 -> SAME volume as sphere
        sx, sy, sz = 5.0, s, s
    else:
        raise SystemExit(f"FLOAT_BODY must be 'sphere' or 'spheroid', got {BODY!r}")
    obj = os.path.join(HERE, f"_float_{BODY}.obj")
    _write_uvsphere_obj(obj, sx, sy, sz)
    sdf = os.path.join(HERE, f"toy_float_{BODY}.sdf")
    I = 0.4 * MASS * R * R
    with open(sdf, "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.6"><world name="world"><model name="toy_float_{BODY}">
  <pose>0 0 0 0 0 0</pose>
  <link name="base_link"><pose>0 0 0 0 0 0</pose>
    <inertial><pose>0 0 0 0 0 0</pose><mass>{MASS}</mass>
      <inertia><ixx>{I:.4e}</ixx><ixy>0</ixy><ixz>0</ixz>
               <iyy>{I:.4e}</iyy><iyz>0</iyz><izz>{I:.4e}</izz></inertia></inertial>
    <collision name="c"><pose>0 0 0 0 0 0</pose>
      <geometry><mesh><uri>_float_{BODY}.obj</uri><scale>1 1 1</scale></mesh></geometry></collision>
    <visual name="v"><pose>0 0 0 0 0 0</pose>
      <geometry><mesh><uri>_float_{BODY}.obj</uri><scale>1 1 1</scale></mesh></geometry>
      <material><ambient>0.9 0.5 0.2 1</ambient><diffuse>0.9 0.5 0.2 1</diffuse></material></visual>
  </link></model></world></sdf>
""")
    return sdf


class FloatDemo(SmallSimConfig):
    def __init__(self):
        super().__init__()
        sdf = _ensure_assets()
        # spawn centre AT the waterline -> floats half-submerged from t=0
        self.animats_pars = [{
            "model_name"       : f"toy_float_{BODY}",
            "sdf_file"         : sdf,
            "control_type"     : "torque",
            "gains"            : [0.0, 0.0, 0.0],
            "controller_config": {
                "path": "lilytorch.farms_examples.submarine."
                        "propeller_controller.PropellerController",
                "tau": 0.0,
            },
            "spawn_mode"       : __import__(
                "farms_core.model.options", fromlist=["SpawnMode"]).SpawnMode.FREE,
            "pose"             : [0.30, 0.0, 0.1, 0.0, 0.0, 0.0],
        }]
        # Coarser grid than the small boat (288^3 is too slow for the CPU iso
        # viewer); 192x96x72 (h=0.005) still resolves the body ~10 cells/radius.
        self.Nx, self.Ny, self.Nz = 192, 96, 72
        self.n_iterations = 4000


if __name__ == "__main__":
    print(f"=== FLOAT DEMO: {BODY} (spawn at waterline z={WATERLINE}) ===", flush=True)
    FloatDemo().run()
