import xml.etree.ElementTree as ET
from xml.dom import minidom
from lilytorch.util.paths import lilytorch_repo_root

def create_pool_sdf(xmin, xmax, ymin, ymax, wall_thickness=0.3, wall_height=0.3):
    # Create root element
    sdf   = ET.Element('sdf', version='1.6')
    world = ET.SubElement(sdf, 'world', name='world')
    model = ET.SubElement(world, 'model', name='pool')

    # Model pose
    model_pose = ET.SubElement(model, 'pose')
    model_pose.text = '0.0 0.0 0.0 0.0 0.0 0.0'

    # Side walls
    # Pool parameters - define the 4 corner points of the inner pool
    # Corners are ordered: bottom-left, bottom-right, top-right, top-left
    corners = [
        (xmin, ymin),  # (x0, y0) - bottom-left
        (xmax, ymin),  # (x1, y1) - bottom-right
        (xmax, ymax),  # (x2, y2) - top-right
        (xmin, ymax)   # (x3, y3) - top-left
    ]


    wall_thickness = 0.3
    wall_height = 0.3
    wall_z = -0.15  # vertical position of walls

    # Calculate pool dimensions from corners
    x0, y0 = corners[0]
    x1, y1 = corners[1]
    x2, y2 = corners[2]
    x3, y3 = corners[3]

    pool_length = x1 - x0  # x-direction (assuming rectangular pool)
    pool_width = y2 - y1   # y-direction

    # Calculate wall positions and sizes
    # Each wall is centered on the pool edge and extends half thickness inward/outward
    sides = [
        # Left wall (x_0): centered at x0, full height in y + extra for corners
        ('side_x_0',
            f'{x0 - wall_thickness/2} {(y0 + y3)/2} {wall_z} 0.0 0.0 0.0',
            f'{wall_thickness} {pool_width + wall_thickness} {wall_height}'),

        # Right wall (x_1): centered at x1, full height in y + extra for corners
        ('side_x_1',
            f'{x1 + wall_thickness/2} {(y1 + y2)/2} {wall_z} 0.0 0.0 0.0',
            f'{wall_thickness} {pool_width + wall_thickness} {wall_height}'),

        # Bottom wall (y_0): centered at y0, full length in x + extra for corners
        ('side_y_0',
            f'{(x0 + x1)/2} {y0 - wall_thickness/2} {wall_z} 0.0 0.0 0.0',
            f'{pool_length + wall_thickness} {wall_thickness} {wall_height}'),

        # Top wall (y_1): centered at y2, full length in x + extra for corners
        ('side_y_1',
            f'{(x2 + x3)/2} {y2 + wall_thickness/2} {wall_z} 0.0 0.0 0.0',
            f'{pool_length + wall_thickness} {wall_thickness} {wall_height}')
    ]

    for name, pose_text, size_text in sides:
        link = ET.SubElement(model, 'link', name=name)
        link_pose = ET.SubElement(link, 'pose')
        link_pose.text = pose_text

        collision   = ET.SubElement(link, 'collision', name=f'{name}_collision')
        c_pose      = ET.SubElement(collision, 'pose')
        c_pose.text = '0.0 0.0 0.0 0.0 0.0 0.0'
        c_geom      = ET.SubElement(collision, 'geometry')
        c_box       = ET.SubElement(c_geom, 'box')
        c_size      = ET.SubElement(c_box, 'size')
        c_size.text = size_text

        visual      = ET.SubElement(link, 'visual', name=f'{name}_visual')
        v_pose      = ET.SubElement(visual, 'pose')
        v_pose.text = '0.0 0.0 0.0 0.0 0.0 0.0'
        v_geom      = ET.SubElement(visual, 'geometry')
        v_box       = ET.SubElement(v_geom, 'box')
        v_size      = ET.SubElement(v_box, 'size')
        v_size.text = size_text
        ET.SubElement(visual, 'material')


        # Floor link
        floor = ET.SubElement(model, 'link', name='floor')
        floor_pose = ET.SubElement(floor, 'pose')
        floor_pose.text = '0.0 0.0 0.0 0.0 0.0 0.0'

        # Calculate floor dimensions and position based on pool corners
        floor_x = (x0 + x1) / 2
        floor_y = (y0 + y2) / 2
        floor_z = -0.35
        floor_length = pool_length + 2 * wall_thickness
        floor_width = pool_width + 2 * wall_thickness

        floor_collision = ET.SubElement(floor, 'collision', name='floor_collision')
        fc_pose = ET.SubElement(floor_collision, 'pose')
        fc_pose.text = f'{floor_x} {floor_y} {floor_z} 0.0 0.0 0.0'
        fc_geom = ET.SubElement(floor_collision, 'geometry')
        fc_box = ET.SubElement(fc_geom, 'box')
        fc_size = ET.SubElement(fc_box, 'size')
        fc_size.text = f'{floor_length} {floor_width} 0.1'

        floor_visual = ET.SubElement(floor, 'visual', name='floor_visual')
        fv_pose = ET.SubElement(floor_visual, 'pose')
        fv_pose.text = f'{floor_x} {floor_y} {floor_z} 0.0 0.0 0.0'
        fv_geom = ET.SubElement(floor_visual, 'geometry')
        fv_box = ET.SubElement(fv_geom, 'box')
        fv_size = ET.SubElement(fv_box, 'size')
        fv_size.text = f'{floor_length} {floor_width} 0.1'
        ET.SubElement(floor_visual, 'material')


    # Visualize the pool dimensions
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    _, ax = plt.subplots(figsize=(10, 6))

    # Draw floor
    floor_patch = patches.Rectangle((x0 - wall_thickness, y0 - wall_thickness),
                                    floor_length, floor_width,
                                    linewidth=1, edgecolor='brown', facecolor='tan', alpha=0.3)
    ax.add_patch(floor_patch)

    # Draw side walls using corner-based parameters
    walls = [
        # Left wall (side_x_0)
        patches.Rectangle((x0 - wall_thickness, y0 - wall_thickness/2),
                            wall_thickness, pool_width + wall_thickness,
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5),
        # Right wall (side_x_1)
        patches.Rectangle((x1, y1 - wall_thickness/2),
                            wall_thickness, pool_width + wall_thickness,
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5),
        # Bottom wall (side_y_0)
        patches.Rectangle((x0 - wall_thickness/2, y0 - wall_thickness),
                            pool_length + wall_thickness, wall_thickness,
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5),
        # Top wall (side_y_1)
        patches.Rectangle((x3 - wall_thickness/2, y2),
                            pool_length + wall_thickness, wall_thickness,
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5)
    ]

    for wall in walls:
        ax.add_patch(wall)

    # Draw water area
    water = patches.Rectangle((x0, y0), pool_length, pool_width,
                                linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.3)
    ax.add_patch(water)

    margin = 1
    ax.set_xlim(x0 - wall_thickness - margin, x1 + wall_thickness + margin)
    ax.set_ylim(y0 - wall_thickness - margin, y2 + wall_thickness + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Pool - Top View')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Pretty print
    xml_str = minidom.parseString(ET.tostring(sdf)).toprettyxml(indent="  ")

    # Write to file
    with open(lilytorch_repo_root + '/farms_examples/sdfs/pool/sdf/pool.sdf', 'w') as f:
        f.write(xml_str)

