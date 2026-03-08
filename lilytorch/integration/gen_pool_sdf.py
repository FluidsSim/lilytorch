import xml.etree.ElementTree as ET
from xml.dom import minidom
from lilytorch.util.paths import lilytorch_repo_root
import os

# ── Material palettes (ambient / diffuse / specular / emissive) ────────

# Warm sandstone walls
WALL_MATERIAL = {
    'ambient':  '0.78 0.74 0.68 1.0',
    'diffuse':  '0.88 0.84 0.78 1.0',
    'specular': '0.30 0.28 0.25 1.0',
    'emissive': '0.0  0.0  0.0  1.0',
}

# Blue-tiled pool floor
FLOOR_MATERIAL = {
    'ambient':  '0.15 0.38 0.58 1.0',
    'diffuse':  '0.22 0.50 0.72 1.0',
    'specular': '0.50 0.50 0.50 1.0',
    'emissive': '0.0  0.0  0.0  1.0',
}

# Translucent water
WATER_MATERIAL = {
    'ambient':  '0.25 0.45 0.75 0.18',
    'diffuse':  '0.30 0.50 0.80 0.18',
    'specular': '0.60 0.60 0.65 0.30',
    'emissive': '0.05 0.08 0.15 0.10',
}


# ── Helpers ────────────────────────────────────────────────────────────

def _add_material(visual_elem, mat_dict):
    """Append <material> with ambient/diffuse/specular/emissive children."""
    mat = ET.SubElement(visual_elem, 'material')
    for tag, value in mat_dict.items():
        el = ET.SubElement(mat, tag)
        el.text = value


def _add_box_link(model, name, pose_text, size_text, mat_dict):
    """Create a <link> with collision + visual box geometry and material."""
    link = ET.SubElement(model, 'link', name=name)
    p = ET.SubElement(link, 'pose')
    p.text = pose_text

    # collision
    col = ET.SubElement(link, 'collision', name=f'{name}_collision')
    cp  = ET.SubElement(col, 'pose')
    cp.text = '0 0 0 0 0 0'
    cg  = ET.SubElement(col, 'geometry')
    cb  = ET.SubElement(cg, 'box')
    cs  = ET.SubElement(cb, 'size')
    cs.text = size_text

    # visual
    vis = ET.SubElement(link, 'visual', name=f'{name}_visual')
    vp  = ET.SubElement(vis, 'pose')
    vp.text = '0 0 0 0 0 0'
    vg  = ET.SubElement(vis, 'geometry')
    vb  = ET.SubElement(vg, 'box')
    vs  = ET.SubElement(vb, 'size')
    vs.text = size_text
    _add_material(vis, mat_dict)

    return link


def _write_sdf(sdf_elem, rel_path):
    """Pretty-print an SDF ElementTree and write it under the sdfs folder."""
    xml_str = minidom.parseString(ET.tostring(sdf_elem)).toprettyxml(indent="  ")
    output_path = os.path.join(
        lilytorch_repo_root, 'farms_examples', 'sdfs', *rel_path.split('/'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(xml_str)
    return output_path


# ── Public API ─────────────────────────────────────────────────────────

def create_pool_sdf(xmin, xmax, ymin, ymax, zmin=None, zmax=None,
                    wall_thickness=None, wall_height=0.3, plotting=False):
    """Generate a rectangular pool SDF with textured walls and floor.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Inner (fluid-domain) boundaries.
    zmin, zmax : float, optional
        Vertical extent.  When given a full 3-D pool is built (4 side walls
        + floor; open top for camera visibility).  When *None* the legacy
        2-D mode is used with *wall_height* in the z-direction.
    wall_thickness : float or None
        Wall thickness.  If *None* it is set to 8 % of the smallest domain
        dimension (>= 0.01 m).
    wall_height : float
        Wall extent in z -- only used when *zmin*/*zmax* are not given.
    plotting : bool
        Pop up a top-view matplotlib figure of the pool.
    """
    is_3d = zmin is not None and zmax is not None

    dx = xmax - xmin
    dy = ymax - ymin
    dz = (zmax - zmin) if is_3d else wall_height

    # Auto-scale wall thickness
    if wall_thickness is None:
        wall_thickness = round(0.08 * min(dx, dy, dz), 4)
        wall_thickness = max(wall_thickness, 0.01)

    wt = wall_thickness
    wz = dz                                                       # wall height
    zc = ((zmin + zmax) / 2) if is_3d else (wall_height / 2)      # wall z-centre

    # ── SDF skeleton ──────────────────────────────────────────────────
    sdf   = ET.Element('sdf', version='1.6')
    world = ET.SubElement(sdf, 'world', name='world')
    model = ET.SubElement(world, 'model', name='pool')
    mp    = ET.SubElement(model, 'pose')
    mp.text = '0 0 0 0 0 0'

    # ---- four side walls ---------------------------------------------
    sides = [
        ('wall_xmin',
         f'{xmin - wt/2} {(ymin+ymax)/2} {zc} 0 0 0',
         f'{wt} {dy + 2*wt} {wz}'),

        ('wall_xmax',
         f'{xmax + wt/2} {(ymin+ymax)/2} {zc} 0 0 0',
         f'{wt} {dy + 2*wt} {wz}'),

        ('wall_ymin',
         f'{(xmin+xmax)/2} {ymin - wt/2} {zc} 0 0 0',
         f'{dx + 2*wt} {wt} {wz}'),

        ('wall_ymax',
         f'{(xmin+xmax)/2} {ymax + wt/2} {zc} 0 0 0',
         f'{dx + 2*wt} {wt} {wz}'),
    ]

    for name, pose_text, size_text in sides:
        _add_box_link(model, name, pose_text, size_text, WALL_MATERIAL)

    # ---- floor (z-min face) ------------------------------------------
    fz = (zmin - wt / 2) if is_3d else (-wt / 2)
    _add_box_link(
        model, 'floor',
        f'{(xmin+xmax)/2} {(ymin+ymax)/2} {fz} 0 0 0',
        f'{dx + 2*wt} {dy + 2*wt} {wt}',
        FLOOR_MATERIAL,
    )

    # ── Optional matplotlib top-view ──────────────────────────────────
    if plotting:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        _, ax = plt.subplots(figsize=(10, 6))

        # floor
        ax.add_patch(patches.Rectangle(
            (xmin - wt, ymin - wt), dx + 2*wt, dy + 2*wt,
            lw=1, ec='brown', fc='tan', alpha=0.3))

        # side walls
        for rect_args in [
            ((xmin - wt, ymin - wt), wt,      dy + 2*wt),
            ((xmax,      ymin - wt), wt,      dy + 2*wt),
            ((xmin - wt, ymin - wt), dx + 2*wt, wt),
            ((xmin - wt, ymax),      dx + 2*wt, wt),
        ]:
            ax.add_patch(patches.Rectangle(
                *rect_args, lw=2, ec='black', fc='gray', alpha=0.5))

        # water area
        ax.add_patch(patches.Rectangle(
            (xmin, ymin), dx, dy, lw=1, ec='blue', fc='lightblue', alpha=0.3))

        m = 0.01
        ax.set_xlim(xmin - wt - m, xmax + wt + m)
        ax.set_ylim(ymin - wt - m, ymax + wt + m)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Pool - Top View')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ── write SDF file ────────────────────────────────────────────────
    return _write_sdf(sdf, 'pool/sdf/pool.sdf')


def create_water_sdf(xmin, xmax, ymin, ymax, zmin=None, zmax=None):
    """Generate a visual-only water-volume SDF sized to the pool interior.

    Returns the absolute path of the written SDF file.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    if zmin is not None and zmax is not None:
        dz = zmax - zmin
        cz = (zmin + zmax) / 2
    else:
        dz, cz = 50.0, -25.0      # legacy: tall slab below z = 0

    sdf   = ET.Element('sdf', version='1.6')
    world = ET.SubElement(sdf, 'world', name='world')
    model = ET.SubElement(world, 'model', name='arena_water')
    mp    = ET.SubElement(model, 'pose')
    mp.text = '0 0 0 0 0 0'

    link = ET.SubElement(model, 'link', name='water')
    lp   = ET.SubElement(link, 'pose')
    lp.text = '0 0 0 0 0 0'

    vis = ET.SubElement(link, 'visual', name='water_visual')
    vp  = ET.SubElement(vis, 'pose')
    vp.text = f'{cx} {cy} {cz} 0 0 0'
    vg  = ET.SubElement(vis, 'geometry')
    vb  = ET.SubElement(vg, 'box')
    vs  = ET.SubElement(vb, 'size')
    vs.text = f'{dx} {dy} {dz}'

    _add_material(vis, WATER_MATERIAL)

    return _write_sdf(sdf, 'arena_water/sdf/arena_water.sdf')
