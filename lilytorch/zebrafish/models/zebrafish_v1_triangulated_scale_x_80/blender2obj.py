import bpy
import os
import numpy as np


# Set the directory where OBJ files will be saved
export_directory = bpy.path.abspath("/data/andreaferrario/lilytorch/lilytorch/zebrafish/models/zebrafish_v1_triangulated_scale_x_80/sdf/tmp")  # Adjust the path as needed

# Ensure the directory exists
if not os.path.exists(export_directory):
    os.makedirs(export_directory)

# Get selected objects
selected_objects = bpy.context.selected_objects

# Export each selected object individually
for obj in selected_objects:
    if obj.type == 'MESH':  # Ensure only mesh objects are exported
        obj_path = os.path.join(export_directory, f"{obj.name}.obj")

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Select only the current object
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj


        locations = np.copy(obj.location)
        obj.location = [0, 0, 0]

        # Export the object to an OBJ file
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_selected_objects=True,
            apply_modifiers=True,
            export_materials=True,
            export_triangulated_mesh=True,
            forward_axis='Y',
            up_axis='Z',
        )
s
        obj.location = locations

        print(f"Exported {obj.name} to {obj_path}")

print("Exporting complete!")