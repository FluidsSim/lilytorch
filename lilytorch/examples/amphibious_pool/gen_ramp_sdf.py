"""Generate a ramp SDF for use as a static animat in fluid-rigid coupling.

The ramp is a thin box (collision + visual), fixed to the world via
``SpawnMode.FIXED``.  BDIM computes fluid forces on its surface.  Add it
as an additional animat.

The box thickness is set to at least 2–3 grid cells (default 0.03 m) so
that the BDIM SDF can resolve the interior and enforce a proper no-slip
boundary.  A sub‑grid thickness (e.g. 1 mm) leads to unresolved SDF
values, fluid leakage through the ramp, and spurious vorticity generation.

Usage::

    from lilytorch.examples.amphibious_pool.gen_ramp_sdf import create_ramp_sdf
    create_ramp_sdf(length=4.0, width=3.0)
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from lilytorch.util.paths import lilytorch_repo_root


RAMP_MATERIAL = {
    'ambient':  '0.45 0.42 0.38 1.0',
    'diffuse':  '0.55 0.52 0.48 1.0',
    'specular': '0.10 0.10 0.10 1.0',
    'emissive': '0.0  0.0  0.0  1.0',
}


def _add_material(visual_elem, mat_dict):
    mat = ET.SubElement(visual_elem, 'material')
    for tag, value in mat_dict.items():
        el = ET.SubElement(mat, tag)
        el.text = value


def create_ramp_sdf(
    length: float = 4.0,
    width: float = 3.0,
    thickness: float = 0.03,
    mass: float = 1e6,
) -> str:
    """Generate a single-link ramp SDF — box collision, box visual.

    The link is centred at the origin with local +X along the ramp length,
    +Y across the width, and +Z normal to the ramp surface.  The caller
    sets the world pose (position + pitch) via the animat spawn config.

    Both collision and visual use a box geometry so that the BDIM SDF
    kernel can resolve the interior and enforce a proper no-slip boundary.
    The *thickness* should be at least 2–3 grid cells (e.g. 0.03 m for a
    0.01 m grid).

    Parameters
    ----------
    length : float
        Ramp length along local X (metres).
    width : float
        Ramp width along local Y (metres).
    thickness : float
        Ramp thickness along local Z (metres).  Must be at least 2–3 grid
        cells for BDIM to resolve the surface.
    mass : float
        Mass (default 1e6 kg — effectively immovable).

    Returns
    -------
    str
        Absolute path of the written SDF file.
    """
    # Box full extents as required by SDF <box><size>
    box_size = f'{length} {width} {thickness}'

    # Inertia for a solid rectangular box of full extents (Lx, Ly, Lz).
    # Ixx = M/12 * (Ly² + Lz²), Iyy = M/12 * (Lx² + Lz²), Izz = M/12 * (Lx² + Ly²).
    ixx = (mass / 12.0) * (width * width + thickness * thickness)
    iyy = (mass / 12.0) * (length * length + thickness * thickness)
    izz = (mass / 12.0) * (length * length + width * width)

    sdf = ET.Element('sdf', version='1.6')
    model = ET.SubElement(sdf, 'model', name='ramp')
    mp = ET.SubElement(model, 'pose')
    mp.text = '0 0 0 0 0 0'

    # No joint — SpawnMode.FIXED handles fixing.

    link = ET.SubElement(model, 'link', name='ramp_link')
    lp = ET.SubElement(link, 'pose')
    lp.text = '0 0 0 0 0 0'

    # Inertial
    inertial = ET.SubElement(link, 'inertial')
    imass = ET.SubElement(inertial, 'mass')
    imass.text = str(mass)
    iinertia = ET.SubElement(inertial, 'inertia')
    for tag, val in [('ixx', ixx), ('ixy', 0), ('ixz', 0),
                     ('iyy', iyy), ('iyz', 0), ('izz', izz)]:
        el = ET.SubElement(iinertia, tag)
        el.text = str(val)

    # Collision: box geometry (required for BDIM SDF — plane is unsupported)
    col = ET.SubElement(link, 'collision', name='ramp_collision')
    cp = ET.SubElement(col, 'pose')
    cp.text = '0 0 0 0 0 0'
    cg = ET.SubElement(col, 'geometry')
    cb = ET.SubElement(cg, 'box')
    cs = ET.SubElement(cb, 'size')
    cs.text = box_size

    # Visual: matching box
    vis = ET.SubElement(link, 'visual', name='ramp_visual')
    vp = ET.SubElement(vis, 'pose')
    vp.text = '0 0 0 0 0 0'
    vg = ET.SubElement(vis, 'geometry')
    vb = ET.SubElement(vg, 'box')
    vs = ET.SubElement(vb, 'size')
    vs.text = box_size
    _add_material(vis, RAMP_MATERIAL)

    # Write
    xml_str = minidom.parseString(ET.tostring(sdf)).toprettyxml(indent="  ")
    output_path = os.path.join(
        lilytorch_repo_root, 'examples', 'sdfs', 'ramp', 'sdf', 'ramp.sdf',
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(xml_str)
    return output_path


if __name__ == "__main__":
    path = create_ramp_sdf()
    print(f"Ramp SDF written to: {path}")
