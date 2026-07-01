#!/usr/bin/env python3
"""Generate THREE sphere SDFs + striped beach-ball visuals for the two-phase
multi-sphere drop demo.

Each sphere gets:
  - mass/inertia from its density
  - a per-sphere UV-mapped OBJ mesh + MTL + striped PNG texture
  - a per-sphere SDF (collision stays a perfect <sphere>)

Single self-contained script — run via ``run.sh``.
"""
import math
import os
import re
import shutil
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

R_SPHERE = 0.0335
RHO_WATER = 1000.0

SPHERES = [
    dict(name="sphere_heavy",   density=1500.0, colour=(1.0, 0.2, 0.2), label="SINKS"),
    dict(name="sphere_neutral", density=1000.0, colour=(0.2, 1.0, 0.2), label="HOVERS"),
    dict(name="sphere_light",   density= 500.0, colour=(0.2, 0.4, 1.0), label="FLOATS"),
]

N_LAT, N_LON = 32, 64
N_STRIPES = 8
TEX_W, TEX_H = 512, 16

# ── PNG writer (stdlib only) ─────────────────────────────────────────────────

def _write_png(path, w, h, pixels_rgb):
    def _chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack(
            '>I', zlib.crc32(c) & 0xffffffff)
    raw = b''
    for row in range(h):
        raw += b'\x00'
        raw += pixels_rgb[row * w * 3:(row + 1) * w * 3]
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    png = b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr)
    png += _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(png)


def _make_striped_png(path, rgb):
    stripe_w = TEX_W // N_STRIPES
    pixels = bytearray(TEX_W * TEX_H * 3)
    r, g, b = [int(max(0, min(255, c * 255))) for c in rgb]
    for row in range(TEX_H):
        for col in range(TEX_W):
            stripe = min(col // stripe_w, N_STRIPES - 1)
            cr, cg, cb = (r, g, b) if stripe % 2 == 0 else (255, 255, 255)
            idx = (row * TEX_W + col) * 3
            pixels[idx] = cr
            pixels[idx + 1] = cg
            pixels[idx + 2] = cb
    _write_png(path, TEX_W, TEX_H, bytes(pixels))


# ── Shared UV-sphere mesh ───────────────────────────────────────────────────

def _build_shared_mesh():
    """Generate the shared OBJ + MTL templates.  Returns (obj_path, mtl_path)."""
    R = R_SPHERE
    verts = [(0.0, 0.0, -R)]
    uvs   = [(0.5, 0.0)]
    norms = [(0.0, 0.0, -1.0)]
    for j in range(1, N_LAT):
        phi = math.pi * j / N_LAT
        sp, cp = math.sin(phi), math.cos(phi)
        for i in range(N_LON):
            th = 2.0 * math.pi * i / N_LON
            x, y, z = R * sp * math.cos(th), R * sp * math.sin(th), R * cp
            verts.append((x, y, z))
            uvs.append((i / N_LON, 1.0 - j / N_LAT))
            norms.append((x / R, y / R, z / R))
    verts.append((0.0, 0.0, R))
    uvs.append((0.5, 1.0))
    norms.append((0.0, 0.0, 1.0))

    faces = []
    for i in range(N_LON):                      # south pole fan
        faces.append(((0, 0, 0), (1 + i, 1 + i, 1 + i),
                      (1 + (i + 1) % N_LON, 1 + (i + 1) % N_LON, 1 + (i + 1) % N_LON)))
    for j in range(N_LAT - 2):                  # body quads → 2 tris
        rt, rb = 1 + j * N_LON, 1 + (j + 1) * N_LON
        for i in range(N_LON):
            ni = (i + 1) % N_LON
            faces.append(((rt+i, rt+i, rt+i), (rb+i, rb+i, rb+i), (rb+ni, rb+ni, rb+ni)))
            faces.append(((rt+i, rt+i, rt+i), (rb+ni, rb+ni, rb+ni), (rt+ni, rt+ni, rt+ni)))
    vn = len(verts) - 1
    rl = 1 + (N_LAT - 2) * N_LON
    for i in range(N_LON):                      # north pole fan
        faces.append(((rl+i, rl+i, rl+i),
                      (rl+(i+1)%N_LON, rl+(i+1)%N_LON, rl+(i+1)%N_LON), (vn, vn, vn)))

    obj_path = os.path.join(HERE, "_striped_sphere.obj")
    with open(obj_path, 'w') as f:
        f.write('# UV sphere\nmtllib _striped_sphere.mtl\n\n')
        for (x, y, z) in verts:   f.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for (u, v) in uvs:        f.write(f'vt {u:.8f} {v:.8f}\n')
        for (nx, ny, nz) in norms: f.write(f'vn {nx:.8f} {ny:.8f} {nz:.8f}\n')
        f.write('\nusemtl sphere_stripes\ns 1\n')
        for (v0, vt0, vn0), (v1, vt1, vn1), (v2, vt2, vn2) in faces:
            f.write(f'f {v0+1}/{vt0+1}/{vn0+1} '
                    f'{v1+1}/{vt1+1}/{vn1+1} {v2+1}/{vt2+1}/{vn2+1}\n')

    mtl_path = os.path.join(HERE, "_striped_sphere.mtl")
    with open(mtl_path, 'w') as f:
        f.write('newmtl sphere_stripes\nKa 1 1 1\nKd 1 1 1\nKs 0.3 0.3 0.3\n'
                'Ns 64\nd 1\nmap_Kd _stripe_texture.png\n')
    return obj_path, mtl_path


# ── Per-sphere helpers ──────────────────────────────────────────────────────

def _float_str(v):
    return f"{v:.12g}"


def _copy_mesh(name, colour, src_obj, src_mtl):
    """Copy shared OBJ+MTL → per-sphere files + generate per-sphere PNG.
    Also pre-create the ``_composite`` OBJ+MTL that FARMS would generate,
    so FARMS skips trimesh and uses our files directly (same trick as
    the 1guilla fish model)."""
    obj_name = f"_striped_sphere_{name}.obj"
    mtl_name = f"_striped_sphere_{name}.mtl"
    png_name = f"_stripe_texture_{name}.png"

    # per-sphere striped PNG texture
    _make_striped_png(os.path.join(HERE, png_name), colour)

    # per-sphere MTL (references the striped PNG)
    mtl = open(src_mtl).read()
    mtl = re.sub(r"map_Kd .+", f"map_Kd {png_name}", mtl)
    with open(os.path.join(HERE, mtl_name), "w") as f:
        f.write(mtl)

    # per-sphere OBJ (references the per-sphere MTL)
    base = os.path.splitext(obj_name)[0]  # e.g. "_striped_sphere_sphere_heavy"
    obj = open(src_obj).read()
    obj = re.sub(r"mtllib .+", f"mtllib {mtl_name}", obj)
    with open(os.path.join(HERE, obj_name), "w") as f:
        f.write(obj)

    # ── pre-create composite files so FARMS skips trimesh ──────────────
    # FARMS checks for ``{path}_composite.obj``; if it exists, the
    # composite path is skipped entirely.
    comp_obj = f"{base}_composite.obj"
    comp_mtl = f"{base}_composite.mtl"
    shutil.copy2(os.path.join(HERE, obj_name), os.path.join(HERE, comp_obj))
    with open(os.path.join(HERE, comp_mtl), "w") as f:
        f.write(f'newmtl {base}_composite\n'
                f'Ka 1 1 1\nKd 1 1 1\nKs 0.3 0.3 0.3\nNs 64\nd 1\n'
                f'map_Kd {png_name}\n')

    return obj_name


def _regen_sdf(template_path, out_path, name, density, colour, src_obj, src_mtl):
    sdf = open(template_path).read()
    sdf = re.sub(r'<model name="[^"]*">', f'<model name="{name}">', sdf, count=1)
    R = float(re.search(r"<radius>\s*([0-9.eE+-]+)\s*</radius>", sdf).group(1))
    mass = density * (4.0 / 3.0 * math.pi * R ** 3)
    inertia = 0.4 * mass * R ** 2
    sdf = re.sub(r"<mass>[^<]*</mass>", f"<mass>{_float_str(mass)}</mass>", sdf)
    for tag in ("ixx", "iyy", "izz"):
        sdf = re.sub(rf"<{tag}>[^<]*</{tag}>", f"<{tag}>{_float_str(inertia)}</{tag}>", sdf)
    obj_name = _copy_mesh(name, colour, src_obj, src_mtl)
    sdf = re.sub(r"<uri>[^<]*</uri>", f"<uri>{obj_name}</uri>", sdf)
    sdf = re.sub(r"rho=[^,]*,", f"rho={density},", sdf)
    open(out_path, "w").write(sdf)
    return R, mass


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    src_obj, src_mtl = _build_shared_mesh()
    template = os.path.join(HERE, "sphere.sdf")
    for cfg in SPHERES:
        out = os.path.join(HERE, f"{cfg['name']}.sdf")
        shutil.copy2(template, out)
        R, m = _regen_sdf(out, out, cfg["name"], cfg["density"], cfg["colour"],
                          src_obj, src_mtl)
        print(f"{cfg['name']}.sdf: DENSITY={cfg['density']} kg/m^3  "
              f"(R={R} m) -> mass={m:.4e} kg  [{cfg['label']}]")
