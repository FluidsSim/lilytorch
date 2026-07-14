import xml.etree.ElementTree as ET
from xml.dom import minidom
from lilytorch.util.paths import lilytorch_repo_root
import os

# ── Material palettes (ambient / diffuse / specular / emissive) ────────

# Warm sandstone walls (transparent)
WALL_MATERIAL = {
    'ambient':  '0.78 0.74 0.68 0.3',
    'diffuse':  '0.88 0.84 0.78 0.3',
    'specular': '0.30 0.28 0.25 0.3',
    'emissive': '0.0  0.0  0.0  0.3',
}

FLOOR_MATERIAL = {
    'ambient':  '0.78 0.74 0.68 0.3',
    'diffuse':  '0.88 0.84 0.78 0.3',
    'specular': '0.30 0.28 0.25 0.3',
    'emissive': '0.0  0.0  0.0  0.3',
}

# Large background ground plane (black)
GROUND_BG_MATERIAL = {
    'ambient':  '0.0 0.0 0.0 1.0',
    'diffuse':  '0.0 0.0 0.0 1.0',
    'specular': '0.0 0.0 0.0 1.0',
    'emissive': '0.0 0.0 0.0 1.0',
}

# Subtle grid lines overlaid on the background floor
GRID_LINE_MATERIAL = {
    'ambient':  '0.55 0.55 0.55 1.0',
    'diffuse':  '0.55 0.55 0.55 1.0',
    'specular': '0.0 0.0 0.0 1.0',
    'emissive': '0.0 0.0 0.0 1.0',
}


# Translucent water
WATER_MATERIAL = {
    'ambient':  '0.25 0.45 0.75 0.18',
    'diffuse':  '0.30 0.50 0.80 0.18',
    'specular': '0.60 0.60 0.65 0.30',
    'emissive': '0.05 0.08 0.15 0.10',
}


# ── Helpers ────────────────────────────────────────────────────────────

def _set_alpha(mat_dict, alpha):
    """Return a copy of *mat_dict* with the 4th (alpha) component overridden."""
    result = {}
    for key, value in mat_dict.items():
        parts = value.split()
        if len(parts) == 4:
            parts[3] = f'{float(alpha):.4f}'
        result[key] = ' '.join(parts)
    return result


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


def _add_collision_only_link(model, name, pose_text, size_text):
    """Create a <link> with collision-only box geometry (no visual)."""
    link = ET.SubElement(model, 'link', name=name)
    p = ET.SubElement(link, 'pose')
    p.text = pose_text

    col = ET.SubElement(link, 'collision', name=f'{name}_collision')
    cp  = ET.SubElement(col, 'pose')
    cp.text = '0 0 0 0 0 0'
    cg  = ET.SubElement(col, 'geometry')
    cb  = ET.SubElement(cg, 'box')
    cs  = ET.SubElement(cb, 'size')
    cs.text = size_text

    return link


def _add_visual_only_link(model, name, pose_text, size_text, mat_dict):
    """Create a <link> with visual-only box geometry (no collision)."""
    link = ET.SubElement(model, 'link', name=name)
    p = ET.SubElement(link, 'pose')
    p.text = pose_text

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
        lilytorch_repo_root, 'examples', 'sdfs', *rel_path.split('/'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(xml_str)
    return output_path


# ── Public API ─────────────────────────────────────────────────────────

def create_pool_sdf(xmin, xmax, ymin, ymax, zmin=None, zmax=None,
                    wall_thickness=None, wall_height=0.3, plotting=False,
                    wall_alpha=None, grid_spacing=None, floor_color=None,
                    lip=None, include_floor=True):
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
    wall_alpha : float or None
        Override the alpha (transparency) of the walls and floor.  ``0.0``
        = fully transparent, ``1.0`` = fully opaque.  *None* keeps the
        defaults from ``WALL_MATERIAL`` / ``FLOOR_MATERIAL`` (0.3).
    grid_spacing : float or None
        When set to a positive value, white grid lines of this spacing (in
        metres) are drawn over the background floor across the inner fluid
        domain.  *None* (default) or ``0`` disables the grid.
    floor_color : list[float] or None
        RGB colour of the large background ground plane, as ``[r, g, b]``
        in [0, 1].  *None* keeps the default black.
    lip : float or None
        Extra height added above *zmax* (3-D) or *wall_height* (2-D) so the
        walls rise above the water surface.  Defaults to *wall_thickness*.
        Pass ``0`` when the water fills the entire pool to keep walls flush.
    include_floor : bool
        If ``False``, omit the floor box and the background ground plane
        (open-bottom pool).  Default ``True``.
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
    if lip is None:
        lip = wt                                                # how much walls rise above water
    wz = dz + lip                                                 # wall height
    zc = ((zmin + zmax) / 2 + lip / 2) if is_3d else ((wall_height + lip) / 2)  # wall z-centre

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

    wall_mat  = _set_alpha(WALL_MATERIAL,  wall_alpha) if wall_alpha is not None else WALL_MATERIAL
    floor_mat = _set_alpha(FLOOR_MATERIAL, wall_alpha) if wall_alpha is not None else FLOOR_MATERIAL

    for name, pose_text, size_text in sides:
        _add_box_link(model, name, pose_text, size_text, wall_mat)

    # ---- floor (z-min face) ------------------------------------------
    if include_floor:
        fz = (zmin - wt / 2) if is_3d else (-wt / 2)
        _add_box_link(
            model, 'floor',
            f'{(xmin+xmax)/2} {(ymin+ymax)/2} {fz} 0 0 0',
            f'{dx + 2*wt} {dy + 2*wt} {wt}',
            floor_mat,
        )

    # ---- large background ground (visual-only, black backdrop) ------
    if include_floor:
        ground_bg_z = (zmin - wt / 2) - wt - 0.01 if is_3d else (-wt / 2) - wt - 0.01
    else:
        ground_bg_z = -100.0  # push far below when no floor
    if floor_color is not None:
        if isinstance(floor_color, str):
            h = floor_color.lstrip('#')
            floor_color = [int(h[i:i+2], 16) / 255.0 for i in range(0, 6, 2)]
        r, g, b = [f'{v:.4f}' for v in floor_color[:3]]
        ground_mat = {
            'ambient':  f'{r} {g} {b} 1.0',
            'diffuse':  f'{r} {g} {b} 1.0',
            'specular': '0.0 0.0 0.0 1.0',
            'emissive': f'{r} {g} {b} 1.0',
        }
    else:
        ground_mat = GROUND_BG_MATERIAL
    _add_visual_only_link(
        model, 'ground_bg',
        f'{(xmin+xmax)/2} {(ymin+ymax)/2} {ground_bg_z} 0 0 0',
        '100 100 0.002',
        ground_mat,
    )

    # ---- white grid lines over the fluid domain ---------------------
    # Guard against grid_spacing <= 0: 0 (or a negative) would otherwise enter
    # this branch and loop forever since `y += grid_spacing` never advances.
    if include_floor and grid_spacing is not None and grid_spacing > 0:
        line_w = grid_spacing * 0.003         # line width = 4 % of spacing
        line_h = 0.0002                       # 0.2 mm thick
        # Place grid lines at the INNER floor surface (inside the tank),
        # visible through the transparent floor material.
        inner_floor_z = fz + wt / 2           # = zmin (3-D) or 0 (2-D)
        gz = inner_floor_z + 0.0001 + line_h / 2

        # Lines running along X (constant y, span full x extent)
        i = 0
        y = ymin
        while y <= ymax + 1e-9:
            _add_visual_only_link(
                model, f'gridh_{i}',
                f'{(xmin + xmax) / 2} {y} {gz} 0 0 0',
                f'{dx} {line_w} {line_h}',
                GRID_LINE_MATERIAL,
            )
            y += grid_spacing
            i += 1

        # Lines running along Y (constant x, span full y extent)
        j = 0
        x = xmin
        while x <= xmax + 1e-9:
            _add_visual_only_link(
                model, f'gridv_{j}',
                f'{x} {(ymin + ymax) / 2} {gz} 0 0 0',
                f'{line_w} {dy} {line_h}',
                GRID_LINE_MATERIAL,
            )
            x += grid_spacing
            j += 1

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


def create_water_sdf(xmin, xmax, ymin, ymax, zmin=None, zmax=None,
                     water_height=0.0, wall_height=0.3, water_alpha=None):
    """Generate a visual-only water-volume SDF sized to the pool interior.

    Parameters
    ----------
    water_height : float
        The value of ``water.height`` used in the arena config.  FARMS
        positions the water body at ``[0, 0, water_height]``, so we
        subtract it from *cz* here so the visual ends up at the correct
        absolute location.
    wall_height : float
        Only used in the 2-D fallback (when *zmin*/*zmax* are *None*).
        The water box is sized to fill the pool from the floor
        (z = 0) up to z = ``wall_height``.

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
        # 2-D mode: water fills from z = 0 up to z = wall_height.
        dz = wall_height
        cz = wall_height / 2

    # Compensate for FARMS water-body offset (water.pos = [0, 0, height])
    cz -= water_height

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

    water_mat = _set_alpha(WATER_MATERIAL, water_alpha) if water_alpha is not None else WATER_MATERIAL
    _add_material(vis, water_mat)

    return _write_sdf(sdf, 'arena_water/sdf/arena_water.sdf')
