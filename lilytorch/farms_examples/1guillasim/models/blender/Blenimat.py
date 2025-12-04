

import bpy
import math
import mathutils
import re
import os
import shutil
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Operator,
                       AddonPreferences,
                       PropertyGroup,
                       )
from mathutils import Matrix
import xml.etree.ElementTree as ET
import xml.dom.minidom
import csv
import copy
try:
    import trimesh as tri
    import numpy as np
except ImportError:
    import pip
    import sys
    pip.main(['install', 'trimesh', '--target',
             (sys.exec_prefix) + '\\lib\\site-packages'])
    import trimesh as trir
    import numpy as np
#    except e as e:
#        print({'WARNING'}, 'Failed to install trimesh due to %s.' % (e))
#        raise

'''
Install Trimesh (Inertia calc) manually when admin privillege is needed
WINDOWS: "*\Blender 4.1\4.1\python\bin\python.exe" -m pip install trimesh
LINUX: ./Blender 4.1/4.1/python/bin/python -m pip install trimesh
'''
EPSILON = 1e-6  # small number to clip to zero

bl_info = {
    "name": "farms_blenimat",
    # export_scene.obj deprecates in 4.0; input args changed in 4.1
    "blender": (4, 10, 00),
    "version": (0, 0, 1),
    "category": "Object",
    "author": "Chuanfang Ning",
    "location": "3D Viewport > Sidebar > farms_Blenimat",
    "description": "Toolkit to export rigged model to FARMS model",
    "category": "Development",
}


def update_display(self, context):
    bpy.context.view_layer.objects.active = bpy.data.objects.get('Skeleton')
    bpy.ops.object.mode_set(mode='OBJECT')
    armature = bpy.data.objects.get('Skeleton')
    if armature is not None:
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode=context.scene.ss.display_mode)


def update_mesh_export_path(self, context):
    if not os.path.isabs(context.scene.es.export_path_mesh):
        context.scene.es.export_path_mesh = os.path.abspath(bpy.path.basename(
            context.scene.es.export_path_mesh))


def update_mesh_import_path(self, context):
    if not os.path.isabs(context.scene.ms.import_path_mesh):
        context.scene.ms.import_path_mesh = os.path.abspath(bpy.path.basename(
            context.scene.ms.import_path_mesh))


def update_inertia_import_path(self, context):
    if not os.path.isabs(context.scene.ms.import_path_inertia):
        context.scene.ms.import_path_inertia = os.path.abspath(bpy.path.basename(
            context.scene.ms.import_path_inertia))


def update_sdf_suffix(self, context):
    if not context.scene.es.export_path_sdf.endswith('.sdf'):
        if '.' not in context.scene.es.export_path_sdf:
            context.scene.es.export_path_sdf = context.scene.es.export_path_sdf + '.sdf'
        else:
            context.scene.es.export_path_sdf = ''.join(
                context.scene.es.export_path_sdf.split('.')[:-1] + ['.sdf'])
    if not os.path.isabs(context.scene.es.export_path_sdf):
        context.scene.es.export_path_sdf = os.path.abspath(bpy.path.basename(
            context.scene.es.export_path_sdf))


def update_length_scale(self, context):
    # grid scale for display
    context.space_data.overlay.grid_scale = context.scene.ms.length_scale
    # blender unit scale for accuracy at high zooming
    bpy.context.scene.unit_settings.scale_length = context.scene.ms.length_scale


def update_mesh_mode(self, context):
    if context.scene.gs.mesh_mode == 'VISUAL':
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                if obj.name.endswith('_collision') or obj.name.endswith('_assembly'):
                    obj.hide_set(True)
                else:
                    obj.hide_set(False)

        context.scene.gs.polynum_sum = 0
        for obj in bpy.data.objects:
            if obj.name.endswith('_assembly'):
                continue
            if obj.type == 'MESH' and not obj.name.endswith('collision'):
                context.scene.gs.polynum_sum += len(obj.data.polygons)
        context.scene.gs.polynum = context.scene.gs.polynum_sum
    elif context.scene.gs.mesh_mode == 'COLLISION':
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                if obj.name.endswith('_collision'):
                    obj.hide_set(False)
                else:
                    obj.hide_set(True)
        context.scene.gs.polynum_sum = 0
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.name.endswith('collision'):
                context.scene.gs.polynum_sum += len(obj.data.polygons)
        context.scene.gs.polynum = context.scene.gs.polynum_sum
    else:
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                if obj.name.endswith('_assembly'):
                    obj.hide_set(False)
                else:
                    obj.hide_set(True)
        context.scene.gs.polynum_sum = 0
        context.scene.gs.polynum = 0


def deselect_bones(armature):
    for bone in armature.data.edit_bones:
        bone.select = False
        bone.select_head = False
        bone.select_tail = False


class MeshSettings(PropertyGroup):
    length_scale: FloatProperty(
        name="Length scaling",
        description="Number of meters that 1 Blender unit is equivalent to",
        default=0.001,
        min=1e-6,
        max=1,
        precision=6,
        update=update_length_scale
    )

    import_length_scale: FloatProperty(
        name="Import length scaling",
        description="Number of meters that 1 importing mesh unit is equivalent to",
        default=1,
        min=1e-6,
        max=1,
        precision=6,
    )

    mass: FloatProperty(
        name="Mass of selected mesh(s)",
        description="Mass of selected mesh(s) in kg",
        default=0,
        min=0,
        precision=6
    )

    density: FloatProperty(
        name="Density of selected mesh(s)",
        description="Density of selected mesh(s) in kg/m\u00b3",
        default=0,
        min=0,
        precision=6
    )
    inertia: FloatVectorProperty(
        name="Inertia of selected mesh",
        description="Inertia matrix",
        default=(.0, .0, .0, .0, .0, .0),
        min=.0,
        size=6
    )
    import_path_inertia: StringProperty(
        name="Import inertia (for robots)",
        description="Choose inertia file (csv):",
        default='',
        maxlen=1024,
        subtype='FILE_PATH',
        update=update_inertia_import_path
    )
    import_path_mesh: StringProperty(
        name="Import mesh (for robots)",
        description="Choose mesh list (csv):",
        default='',
        maxlen=1024,
        subtype='FILE_PATH',
        update=update_mesh_import_path
    )


class GeometrySettings(PropertyGroup):
    mesh_mode: EnumProperty(
        name="Mesh Mode",
        description="Mesh Mode",
        items=[('VISUAL', "VISUAL", "Display visual mesh", 'RESTRICT_RENDER_OFF', 1),
               ('COLLISION', "COLLISION", "Display collision mesh ", 'META_CUBE', 2),
               ('ASSEMBLY', "ASSEMBLY", "Display assembly mesh ", 'AUTO', 3),
               ],
        update=update_mesh_mode
    )

    limited_dissolve: FloatProperty(
        name="",
        description="Dissolve degree limit",
        default=5,
        min=0,
        max=90,
        precision=2
    )
    decimate: FloatProperty(
        name="",
        description="Mesh decimate percent",
        default=0.5,
        min=0,
        max=1,
        precision=2
    )

    oct_tree_depth: IntProperty(
        name="",
        description="Remesh voxel size",
        default=4,
        min=1,
        max=6,
    )

    smooth_angle: FloatProperty(
        name="",
        description="Smooth angle",
        default=30.,
        min=0.,
        max=90.,
        precision=2
    )

    polynum: IntProperty(
        name="polynum",
        default=0,
        description="Polygon count of selected mesh")

    polynum_sum: IntProperty(
        name="polynum_sum",
        default=0,
        description="Polygon count of all meshes")


class SkeletonSettings(PropertyGroup):

    display_mode: EnumProperty(
        name="Display Mode",
        description="Display Mode",
        items=[('OBJECT', "OBJECT", "", 'MOD_MESHDEFORM', 1),
               ('POSE', "POSE", "", 'MOD_ARMATURE', 2),
               ],
        update=update_display
    )

    joint_name: StringProperty(
        name="Joint Name",
        description="String Property",
        default="joint_body_00",
        maxlen=1024,
    )
    rotation_mode: EnumProperty(
        name="Rotation Mode",
        description="Rotation Mode",
        items=[('GLOBAL', "GLOBAL", "Rotation axis aligned with global model space (Back X, Right Y, Up Z)", 'ORIENTATION_GLOBAL', 1),  # change convention when defining constraints
               ('LOCAL', "LOCAL", "Rotation axis aligned with local skeleton (Tail X, Local Mesh Z, Right Hand Y) ",
                'ORIENTATION_LOCAL', 2),  # rotate joint when exporting SDF
               ]
    )
    joint_constraints_X: FloatVectorProperty(
        name="X",
        description="Constraint X rotation of joint",
        default=(0.0, 0.0),
        min=-180.,
        max=180,
        size=2,
    )
    joint_constraints_Y: FloatVectorProperty(
        name="Y",
        description="Constraint Y rotation of joint",
        default=(0.0, 0.0),
        min=-180.,
        max=180,
        size=2,
    )
    joint_constraints_Z: FloatVectorProperty(
        name="Z",
        description="Constraint Z rotation of joint",
        default=(0.0, 0.0),
        min=-180.,
        max=180,
        size=2,
    )

    joint_passive_x: BoolProperty(
        name="",
        description="if X is passive",
        default=False
    )
    
    joint_passive_y: BoolProperty(
        name="",
        description="if Y is passive",
        default=False
    )
    
    joint_passive_z: BoolProperty(
        name="",
        description="if Z is passive",
        default=False
    )
    
    rotation_convention: EnumProperty(
        name="Rotation Convention",
        description="Rotation Convention",
        items=[('XYZ', "XYZ", ""),
               ('XZY', "XZY", ""),
               ('YXZ', "YXZ", ""),
               ('YZX', "YZX", ""),
               ('ZXY', "ZXY", ""),
               ('ZYX', "ZYX", ""),
               ]
    )

    auto_increment: BoolProperty(
        name="Auto Increment",
        description="Increment the name if ends with integer",
        default=True
    )
    disconnected: BoolProperty(
        name="Disconnect bone",
        description="Disconnected binding (last tail is not next head)",
        default=False
    )
    extend_axis: EnumProperty(
        name="Extension direction for disconnected bone",
        description="Axis to extend disconnected skeleton in Global Mode",
        items=[('X+', "X+", ""),
               ('X-', "X-", ""),
               ('Y+', "Y+", ""),
               ('Y-', "Y-", ""),
               ('Z+', "Z+", ""),
               ('Z-', "Z-", ""),
               ]
    )

    visual_mesh: EnumProperty(
        name="Visual mesh",
        description="Method with visual mesh binding",
        items=[('SOLID', "SOLID", "", 'BONE_DATA', 1),
               ('SOFT', "SOFT", "", 'OUTLINER_OB_GREASEPENCIL', 2),
               ],
    )


class ExportSettings(PropertyGroup):
    export_path_mesh: StringProperty(
        name="Directory",
        description="Choose export directory:",
        default='',
        maxlen=1024,
        subtype='DIR_PATH',
        update=update_mesh_export_path
    )
    export_path_sdf: StringProperty(
        name="Directory",
        description="Choose export file:",
        default='',
        maxlen=1024,
        subtype='FILE_PATH',
        update=update_sdf_suffix
    )


class Assign_mass(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.assign_mass"
    bl_description = "Set mesh mass and infer density"
    bl_label = "assign mass"

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected.")
            return 0
        elif context.scene.ms.mass == 0:
            cls.poll_message_set("No mass assigned.")
            return 0
        return 1

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                pass
            if obj.rigid_body is None:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add()
            bpy.ops.rigidbody.mass_calculate(
                material='Custom', density=1.0)  # m^3
            volume = bpy.context.object.rigid_body.mass / (context.scene.ms.length_scale**3)
            bpy.context.object.rigid_body.mass = context.scene.ms.mass * (context.scene.ms.length_scale**3)
            density = context.scene.ms.mass / volume
            context.scene.ms.density = density
            obj['density'] = density
        
        return {'FINISHED'}


class Assign_density(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.assign_density"
    bl_description = "Set mesh density and infer mass"
    bl_label = "assign density"

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected.")
            return 0
        elif context.scene.ms.density == 0:
            cls.poll_message_set("No density assigned.")
            return 0
        return 1

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            obj['density'] = context.scene.ms.density
            bpy.context.view_layer.objects.active = obj
            if obj.rigid_body is None:
                bpy.ops.rigidbody.object_add()
            bpy.ops.rigidbody.mass_calculate(
                material='Custom', density=context.scene.ms.density)
        context.scene.ms.mass = bpy.context.object.rigid_body.mass / (context.scene.ms.length_scale**3)
        return {'FINISHED'}


class Import_mesh(bpy.types.Operator):
    bl_idname = "wm.import_mesh"
    bl_label = "OVERWRITE current scene?"
    bl_description = "Import mesh and transform from a csv list"

    @classmethod
    def poll(cls, context):
        if not (os.path.isfile(bpy.path.abspath(context.scene.ms.import_path_mesh)) or context.scene.ms.import_path_mesh.endswith('.csv')):
            cls.poll_message_set("Invalid file format.")
            return 0
        return 1

    def execute(self, context):
        with open(bpy.path.abspath(context.scene.ms.import_path_mesh)) as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            mesh_data = [row for row in reader]
        bpy.ops.object.select_all(action='DESELECT')

        for row in mesh_data:
            name = os.path.splitext(row[0])[0]
            if name in bpy.data.objects.keys():
                bpy.context.scene.objects.get(name).select_set(True)
            if name + '_collision' in bpy.data.objects.keys():
                bpy.context.scene.objects.get(name + '_collision').select_set(True)
            bpy.ops.object.delete()  # delete existing mesh and data
        bpy.ops.outliner.orphans_purge(do_recursive=True)

        mesh_dir = os.path.dirname(context.scene.ms.import_path_mesh)
        for row in mesh_data:
            name = os.path.splitext(row[0])[0]
            file_name = os.path.join(mesh_dir, row[0])
            row_tf = [float(i) for i in row[1:]]
            import_scale = context.scene.ms.import_length_scale/context.scene.ms.length_scale
            if os.path.isfile(file_name):
                bpy.ops.wm.obj_import(
                    filepath=file_name,
                    forward_axis='NEGATIVE_Y',
                    up_axis='Z',
                    validate_meshes=True,
                    #                                global_scale = 1/context.scene.ms.length_scale,
                )
                matrix_rot = mathutils.Matrix()
                matrix_rot[0][0:3] = row_tf[0:3]
                matrix_rot[1][0:3] = row_tf[3:6]
                matrix_rot[2][0:3] = row_tf[6:9]
                matrix_loc = mathutils.Matrix.Translation(
                    [i * import_scale for i in row_tf[9:12]])
                matrix_scale = mathutils.Matrix.Scale(import_scale, 4)
                matrix_out = matrix_loc @ matrix_rot @ matrix_scale
                bpy.context.scene.objects.get(name).matrix_world = matrix_out
                self.report({'INFO'}, 'Successfully imported mesh %s.' % name)
            else:
                self.report(
                    {'WARNING'}, 'Mesh %s does not exist. Skipping' % name)

            bpy.ops.object.select_all(action='DESELECT')
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    obj.select_set(True)
            bpy.ops.object.transform_apply(location=False)
        return {'FINISHED'}

    def invoke(self, context, event):
        if os.path.isfile(context.scene.ms.import_path_inertia):
            wm = context.window_manager
            return wm.invoke_confirm(self, event)
        else:
            return self.execute(context)


class Import_inertia(bpy.types.Operator):
    bl_idname = "wm.import_inertia"
    bl_label = "OVERWRITE current inertia with data from file?"
    bl_description = "Import inertia, mass and CoM from csv file"

    @classmethod
    def poll(cls, context):
        if not (os.path.isfile(bpy.path.abspath(context.scene.ms.import_path_inertia)) or context.scene.ms.import_path_inertia.endswith('.csv')):
            cls.poll_message_set("Invalid file format.")
            return 0
        return 1

    def execute(self, context):
        with open(bpy.path.abspath(context.scene.ms.import_path_inertia)) as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            inertia_data = [row for row in reader]
        for row in inertia_data:
            name = os.path.splitext(row[0])[0]
            if name in bpy.data.objects.keys():
                print('Importing inertia for %s' % name)
                mesh_name = name if name.endswith(
                '_collision') else name + '_collision'
                
                obj = bpy.data.objects[mesh_name]
                if obj.type != 'MESH':
                    continue
                inertia = np.array([[float(row[3]), float(row[4]), float(row[5])],
                                    [float(row[4]), float(row[6]), float(row[7])],
                                    [float(row[5]), float(row[7]), float(row[8])]])
                print(inertia)
                if not Calc_inertia.valid_inertia(inertia):
                    raise ValueError(
                        'Mesh {} has inappropriate inertia {}'.format(
                            obj.name,
                            inertia)
                    )
                # Ixx Ixy Ixz Iyy Iyz Izz
                obj['inertia'] = [inertia[0, 0], inertia[0, 1], inertia[0, 2],
                                  inertia[1, 1], inertia[1, 2],
                                  inertia[2, 2]]
                obj['com'] = [float(row[9]), float(row[10]), float(row[11])]
                obj['left'] = False
                obj['right'] = False
                
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True) 
                if obj.rigid_body is None:
                    bpy.ops.rigidbody.object_add()
                bpy.ops.rigidbody.mass_calculate(
                    material='Custom', density=1)  # kg/m^3
                volume = bpy.context.object.rigid_body.mass * (context.scene.ms.length_scale**3)#m^3
                bpy.context.object.rigid_body.mass = float(row[1]) * (context.scene.ms.length_scale**3)
                density = float(row[1]) / volume
                
                obj['density'] = density
                
        self.report({'INFO'}, 'Successfully imported inertia for %s.' % name)
        return {'FINISHED'}

    def invoke(self, context, event):
        if os.path.isfile(context.scene.ms.import_path_inertia):
            wm = context.window_manager
            return wm.invoke_confirm(self, event)
        else:
            return self.execute(context)


class Set_Left(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.set_left"
    bl_description = "Mark left side"
    bl_label = "set left"

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected.")
            return 0
        return 1

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            obj['left'] = True
            obj['right'] = False
        return {'FINISHED'}

class Set_Right(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.set_right"
    bl_description = "Mark right side"
    bl_label = "set right"

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected.")
            return 0
        return 1

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            obj['right'] = True
            obj['left'] = False
        return {'FINISHED'}

class Clear_side(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.clear_side"
    bl_description = "Clear side marker"
    bl_label = "clear side"

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected.")
            return 0
        return 1

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            obj['left'] = False
            obj['right'] = False
        return {'FINISHED'}

class Center_mesh(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.center_mesh"
    bl_description = "Shift all meshes so that current mesh locates as origin"
    bl_label = "center mesh"
    @classmethod
    def poll(cls, context):
        if 'Skeleton' in bpy.data.objects.keys():
            cls.poll_message_set("Mesh operation should happen before defining skeletons")
            return 0
        else:
            if context.object is None or context.active_object is None:
                cls.poll_message_set("No head reference selected for shifting.")
                return 0
            return 1
    def execute(self, context):
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        offset = copy.copy(context.object.location)
        print(offset)
        for obj in bpy.data.objects:
            print(obj.name)
            if obj.type == 'MESH':
                obj.data.name = obj.name
                print(obj.location)
                obj.location = obj.location - offset
                print(obj.location)
                print(offset)
        
        return {'FINISHED'}
    

class Create_collision(bpy.types.Operator):
    bl_idname = "wm.create_collision"
    bl_label = "OVERWRITE current collision meshes?"
    bl_description = "Initiate copy of meshes for collision models"

    @classmethod
    def poll(cls, context):
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected")
            return 0
        for obj in context.selected_objects:
            if obj.name.endswith('_assembly'):
                continue
            # exist at least 1 visual mesh
            if obj.type == 'MESH' and not obj.name.endswith('_collision'):
                return 1
        cls.poll_message_set("No visual meshes found")
        return 0

    def execute(self, context):

        objs = []
        for obj in context.selected_objects:
            if context.scene.objects.get(obj.name + '_collision') is not None:
                objs.append(context.scene.objects.get(obj.name + '_collision'))
        if len(objs) > 0:
            with bpy.context.temp_override(selected_objects=objs):
                bpy.ops.object.delete()
            bpy.ops.outliner.orphans_purge(do_recursive=True)

        # create collision meshes from visual meshes
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                print(obj.name)
                obj_collision = obj.copy()
                obj_collision.name = obj.name + '_collision'
                obj_collision.data = obj.data.copy()
                context.collection.objects.link(obj_collision)
                self.report(
                    {'INFO'}, 'Successfully created collision mesh for %s.' % obj_collision.name)

        return {'FINISHED'}

    def invoke(self, context, event):
        if any(obj.endswith('_collision') for obj in bpy.context.scene.objects.keys()):
            wm = context.window_manager
            return wm.invoke_confirm(self, event)
        else:
            return self.execute(context)


class Polycount(bpy.types.Operator):
    bl_idname = "wm.polycount"
    bl_label = "Polycount"
    bl_description = "Count polygons in current and in all meshes"

    def execute(self, context):
        is_collision = context.scene.gs.mesh_mode == 'COLLISION'
        context.scene.gs.polynum = 0
        context.scene.gs.polynum_sum = 0
        if len(context.selected_objects) > 0:
            for ob in context.selected_objects:
                if ob.type == 'MESH':
                    context.scene.gs.polynum += len(ob.data.polygons)
        for ob in bpy.data.objects:
            if ob.name.endswith('_assembly'):
                continue
            if ob.type == 'MESH' and not (ob.name.endswith('collision') ^ is_collision):
                context.scene.gs.polynum_sum += len(ob.data.polygons)
        return {'FINISHED'}


class Dissolve(bpy.types.Operator):
    bl_idname = "wm.limited_dissolve"
    bl_label = "Limited dissolve"
    bl_description = "Dissolve selected meshes with specified limit angle"

    @classmethod
    def poll(cls, context):
        if context.object is None or context.active_object is None:
            cls.poll_message_set("No object selected.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        return 1

    def execute(self, context):
        is_collision = context.selected_objects[0].name.endswith('_collision')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.dissolve_limited(
            angle_limit=math.radians(context.scene.gs.limited_dissolve))
        bpy.ops.object.mode_set(mode='OBJECT')
        context.scene.gs.polynum = 0
        context.scene.gs.polynum_sum = 0
        if len(context.selected_objects) > 0:
            for ob in context.selected_objects:
                if ob.type == 'MESH':
                    context.scene.gs.polynum += len(ob.data.polygons)
        for ob in bpy.data.objects:
            if ob.type == 'MESH' and not (ob.name.endswith('collision') ^ is_collision):
                context.scene.gs.polynum_sum += len(ob.data.polygons)
        return {'FINISHED'}


class Decimate(bpy.types.Operator):
    bl_idname = "wm.decimate"
    bl_label = "Decimate"
    bl_description = "Decimate geometry to given percent"

    @classmethod
    def poll(cls, context):
        if context.object is None or context.active_object is None:
            cls.poll_message_set("No object selected.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        return 1

    def execute(self, context):
        is_collision = context.selected_objects[0].name.endswith('_collision')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.decimate(ratio=context.scene.gs.decimate)
        bpy.ops.object.mode_set(mode='OBJECT')
        context.scene.gs.polynum = 0
        context.scene.gs.polynum_sum = 0
        if len(context.selected_objects) > 0:
            for ob in context.selected_objects:
                if ob.type == 'MESH':
                    context.scene.gs.polynum += len(ob.data.polygons)
        for ob in bpy.data.objects:
            if ob.type == 'MESH' and not (ob.name.endswith('collision') ^ is_collision):
                context.scene.gs.polynum_sum += len(ob.data.polygons)
        return {'FINISHED'}


class Remesh(bpy.types.Operator):
    bl_idname = "wm.remesh"
    bl_label = "Remesh"
    bl_description = "Remesh geometry with specified Octree depth"

    @classmethod
    def poll(cls, context):
        if context.object is None or context.active_object is None:
            cls.poll_message_set("No object selected.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        return 1

    def execute(self, context):
        is_collision = context.selected_objects[0].name.endswith('_collision')
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_add(type='REMESH')
                bpy.context.object.modifiers["Remesh"].mode = 'SMOOTH'
                bpy.context.object.modifiers["Remesh"].octree_depth = context.scene.gs.oct_tree_depth
                bpy.context.object.modifiers["Remesh"].scale = 0.9
                bpy.context.object.modifiers["Remesh"].sharpness = 0.5
                bpy.context.object.modifiers["Remesh"].use_remove_disconnected = True
                bpy.context.object.modifiers["Remesh"].threshold = 0.8
#                bpy.context.object.modifiers["Remesh"].adaptivity = context.scene.gs.oct_tree_depth * .5
                bpy.ops.object.modifier_apply(modifier="Remesh")
        context.scene.gs.polynum = 0
        context.scene.gs.polynum_sum = 0
        if len(context.selected_objects) > 0:
            for ob in context.selected_objects:
                if ob.type == 'MESH':
                    context.scene.gs.polynum += len(ob.data.polygons)
        for ob in bpy.data.objects:
            if ob.type == 'MESH' and not (ob.name.endswith('collision') ^ is_collision):
                context.scene.gs.polynum_sum += len(ob.data.polygons)
        return {'FINISHED'}


class Smooth(bpy.types.Operator):
    bl_idname = "wm.smooth"
    bl_label = "Smooth"
    bl_description = "Smooth geometry with bevel operator"

    @classmethod
    def poll(cls, context):
        if context.object is None or context.active_object is None:
            cls.poll_message_set("No object selected.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        return 1

    def execute(self, context):
        is_collision = context.selected_objects[0].name.endswith('_collision')
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_add(type='BEVEL')
                bpy.context.object.modifiers["Bevel"].offset_type = 'PERCENT'
                bpy.context.object.modifiers["Bevel"].width_pct = 25.
                bpy.context.object.modifiers["Bevel"].angle_limit = math.radians(
                    context.scene.gs.smooth_angle)
                bpy.ops.object.modifier_apply(modifier="Bevel")
        context.scene.gs.polynum = 0
        context.scene.gs.polynum_sum = 0
        if len(context.selected_objects) > 0:
            for ob in context.selected_objects:
                if ob.type == 'MESH':
                    context.scene.gs.polynum += len(ob.data.polygons)
        for ob in bpy.data.objects:
            if ob.type == 'MESH' and not (ob.name.endswith('collision') ^ is_collision):
                context.scene.gs.polynum_sum += len(ob.data.polygons)
        return {'FINISHED'}


class Clear_Head(bpy.types.Operator):
    bl_idname = "wm.clear_head"
    bl_label = "Clear head and purge unused amature data"
    bl_description = "Clear head and purge unused amature data"

    def execute(self, context):
        if len(bpy.data.objects) > 0:
            if bpy.context.view_layer.objects.active is None:
                bpy.context.view_layer.objects.active = bpy.data.objects[0]
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.select_by_type(type='ARMATURE')
            bpy.ops.object.delete()
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        for ob in bpy.data.objects:
            ob.parent_bone = ''
            ob.parent_type = 'OBJECT'
            ob.modifiers.clear()

        context.scene.ss.joint_name = "joint_body_00"  # reset input
        return {'FINISHED'}


class Set_Head(bpy.types.Operator):
    bl_idname = "wm.set_head"
    bl_label = "Set current part as head of amature"
    bl_description = "Set current part as head of amature"

    @classmethod
    def poll(cls, context):
        if context.scene.gs.mesh_mode != 'COLLISION':
            cls.poll_message_set("Not in collision mesh mode.")
            return 0
        if context.object is None or context.active_object is None:
            cls.poll_message_set(
                "No object selected. Select the object to set as head.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        return 1

    def execute(self, context):
        ob = bpy.context.active_object
        ob_visual = bpy.data.objects.get(ob.name[:-10])
        if 'Skeleton' not in bpy.data.objects.keys():
            bpy.ops.object.armature_add(
                enter_editmode=False, align='WORLD', location=ob.location, scale=(1, 1, 1))
        armature = bpy.data.objects['Armature']
        armature.name = 'Skeleton'
        armature.data.display_type = 'STICK'
        armature.data.show_names = True
        armature.data.show_axes = True
        armature.show_in_front = True
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        deselect_bones(armature)
        if 'Bone' in armature.data.edit_bones:
            bone = armature.data.edit_bones.get('Bone')
            bone.name = 'Head'
        elif 'Head' in armature.data.edit_bones:
            bone = armature.data.edit_bones.get('Head')
        else:
            bone = armature.data.edit_bones.get('joint_base')
        
        amature_world = armature.matrix_world
        # position bone head to mesh head
        bone.head = amature_world.inverted() @ ob.matrix_world.translation
        ob.parent = armature
        ob.parent_bone = bone.name
        ob.parent_type = 'ARMATURE'
        ob.matrix_parent_inverse = armature.matrix_world.inverted()
        
        if context.scene.ss.visual_mesh == 'SOLID':
            ob_visual.parent = armature
            ob_visual.parent_bone = bone.name
            ob_visual.parent_type = 'ARMATURE'
            ob_visual.matrix_parent_inverse = armature.matrix_world.inverted()
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            ob_visual.hide_set(False)
            bpy.ops.object.select_all(action='DESELECT')
            ob_visual.select_set(True)
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.parent_set(type='ARMATURE_AUTO', xmirror=True)
            ob_visual.hide_set(True)
#        bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


class Bind(bpy.types.Operator):
    bl_idname = "wm.bind_skeleton"
    bl_label = "Bind skeleton"
    bl_description = "Bind selected mesh with amature skeleton"

    @classmethod
    def poll(cls, context):
        if context.scene.gs.mesh_mode != 'COLLISION':
            cls.poll_message_set("Not in collision mesh mode.")
            return 0
        if context.object is None or context.active_object is None:
            cls.poll_message_set("No object selected.")
            return 0
        elif bpy.data.objects.get('Skeleton') is None:
            cls.poll_message_set("No head set.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        elif len(context.selected_objects) != 2:
            cls.poll_message_set(
                '%r instead of a pair of mesh selected' % len(context.selected_objects))
            return 0
        elif context.scene.ss.joint_constraints_X[0] > context.scene.ss.joint_constraints_X[1]:
            cls.poll_message_set('X lower bound %0.2r must be no greater than upper bound %0.2r' % (
                context.scene.ss.joint_constraints_X[0], context.scene.ss.joint_constraints_X[1]))
            return 0
        elif context.scene.ss.joint_constraints_Y[0] > context.scene.ss.joint_constraints_Y[1]:
            cls.poll_message_set('Y lower bound %0.2r must be no greater than upper bound %0.2r' % (
                context.scene.ss.joint_constraints_Y[0], context.scene.ss.joint_constraints_Y[1]))
            return 0
        elif context.scene.ss.joint_constraints_Z[0] > context.scene.ss.joint_constraints_Z[1]:
            cls.poll_message_set('Z lower bound %0.2r must be no greater than upper bound %0.2r' % (
                context.scene.ss.joint_constraints_Z[0], context.scene.ss.joint_constraints_Z[1]))
            return 0
        selected_objects = context.selected_objects
        mesh2 = context.object
        selected_objects.remove(mesh2)
        mesh1 = selected_objects[0]
        if mesh1.parent is not None:
            if mesh1.parent.name == 'Skeleton' or mesh1.parent_bone is None:
                return 1
        if mesh1.name not in bpy.data.objects['Skeleton'].pose.bones.keys():
            cls.poll_message_set(
                'First mesh %s not attached to amature' % (mesh1.name))
            return 0
        if mesh1.name[:-10] not in bpy.data.objects.keys():
            cls.poll_message_set(
                'No visual mesh found for %s.' % (mesh1.name[:-10]))
            return 0
        if mesh2.name[:-10] not in bpy.data.objects.keys():
            cls.poll_message_set(
                'No visual mesh found for %s.' % (mesh2.name[:-10]))
            return 0
        return 1

    def execute(self, context):
        # select bones in object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        armature = bpy.data.objects['Skeleton']
        amature_world_inv = armature.matrix_world.inverted()

        # index meshes
        selected_objects = context.selected_objects
        selected_objects_bkup = selected_objects.copy()
        mesh2 = context.object
        selected_objects.remove(mesh2)
        mesh1 = selected_objects[0]
        mesh1_visual = bpy.data.objects.get(mesh1.name[:-10])
        mesh2_visual = bpy.data.objects.get(mesh2.name[:-10])
        # enter edit mode for skeletons
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        armature = bpy.data.objects['Skeleton']

        # pointer to first bone
        if 'Head' in armature.data.edit_bones.keys():
            bone1 = armature.data.edit_bones.get('Head')
            bone1.name = 'joint_base'
        else:
            bone1 = armature.data.edit_bones.get(mesh1.parent_bone)
        # pointer to second bone
        if mesh2.parent_bone in armature.data.edit_bones.keys():  # override
            bone2 = armature.data.edit_bones.get(mesh2.parent_bone)
            bone2.name = context.scene.ss.joint_name
        elif context.scene.ss.joint_name in armature.data.edit_bones.keys():
            bone2 = armature.data.edit_bones.get(context.scene.ss.joint_name)
        else:
            bone2 = armature.data.edit_bones.new(context.scene.ss.joint_name)
        # check if need to force disconnect
        bone_axis_2 = ['X', 'Y', 'Z'].index(context.scene.ss.extend_axis[0])
        bone_dir_2 = 1 if context.scene.ss.extend_axis[-1] == '+' else -1
        # vector pointing from bone head to bone tail (distal)
        bone_axis_1_vec = list(
            mesh2.matrix_world.to_translation() - mesh1.matrix_world.to_translation())
        bone_axis_1_abs = [abs(i) for i in bone_axis_1_vec]
        bone_axis_1 = bone_axis_1_abs.index(max(bone_axis_1_abs))
        bone_dir_1 = 1 if bone_axis_1_abs[bone_axis_1] > 0 else -1
        mesh2['bone_dir'] = bone_dir_2
        mesh2['bone_axis'] = bone_axis_2
        mesh2['joint_rot'] = mathutils.Vector((0,0,0))
        mesh2['passive'] = [int(context.scene.ss.joint_passive_x),
                            int(context.scene.ss.joint_passive_y),
                            int(context.scene.ss.joint_passive_z)]
        
        force_disconnect = False
        if (not context.scene.ss.disconnected) and context.scene.ss.rotation_mode == 'GLOBAL':
            # check if bone1 bone2 aligned axis
            if (bone_axis_1 != bone_axis_2 or bone_dir_1 != bone_dir_2):
                force_disconnect = True
            # check if bone1 bone2 same axis
            elif bone_axis_1_vec[bone_axis_1] != 0:
                force_disconnect = True

        bone1_name = bone1.name
        bone2_name = bone2.name
        bone2.parent = bone1
        bone2.use_connect = not (
            context.scene.ss.disconnected or force_disconnect)

        # bone2 head to second mesh
        bone2.head = amature_world_inv @ mesh2.matrix_world.translation
        if not bone2.use_connect:  # if disconnected, extend bone2 along extension axis
            temp_bone_length = max(
                (bone2.head - bone1.tail).length, (bone1.head - bone1.tail).length) * 0.5
            temp_ext = mathutils.Vector((0., 0., 0.))
            temp_ext[bone_axis_2] = temp_bone_length * bone_dir_2
            bone2.tail = bone2.head + temp_ext  # bone2 tail extended along extend_axis
            if force_disconnect:
                bone1.tail = bone2.head  # move bone1 tail to mesh2
                match bone_axis_1:  # clear bone1 tail non-axis part
                    case 0:
                        bone1.tail.y = bone1.head.y
                        bone1.tail.z = bone1.head.z
                    case 1:
                        bone1.tail.x = bone1.head.x
                        bone1.tail.z = bone1.head.z
                    case _:
                        bone1.tail.x = bone1.head.x
                        bone1.tail.y = bone1.head.y
        else:  # if connected, extend bone2 along previous bone
            bone1.tail = bone2.head  # connect bone1 tail to bone2 head
            if bone2.tail == mathutils.Vector((0.0, 0.0, 0.0)):
                # extend bone2 along previous bone
                bone2.tail = bone2.head + (bone1.tail - bone1.head) * 0.5
#        return {'FINISHED'}

        if context.scene.ss.rotation_mode == 'LOCAL':
            vec_base = mathutils.Vector((0, 0, 0))
            vec_sign = 1 if context.scene.ss.extend_axis[-1] == '+' else -1
            vec_base['XYZ'.index(
                context.scene.ss.extend_axis[0])] = 1 * vec_sign
            vec_tf = bone1.tail - bone1.head
            rot = vec_base.rotation_difference(vec_tf)
            print('bone1',bone1.name,bone1.tail,bone1.head)
            print(vec_base,vec_tf)
            mesh1['joint_rot'] = rot.to_euler(
                context.scene.ss.rotation_convention)
        else:
            mesh1['joint_rot'] = mathutils.Vector((0,0,0))
        for i, rot in enumerate(mesh1['joint_rot'][:]):
            if abs(rot)<1e-6:
                mesh1['joint_rot'][i] = 0
        bpy.ops.armature.collection_deselect()

        bone2.select = True
        match context.scene.ss.extend_axis:
            case 'X+':
                bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
            case 'X-':
                bpy.ops.armature.calculate_roll(type='GLOBAL_NEG_Z')
            case 'Y+':
                bpy.ops.armature.calculate_roll(type='GLOBAL_NEG_Y')
            case 'Y-':
                bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Y')
            case 'Z+':
                bpy.ops.armature.calculate_roll(type='GLOBAL_NEG_X')
            case _:
                bpy.ops.armature.calculate_roll(type='GLOBAL_POS_X')

        mesh1.parent = armature
        mesh1.parent_bone = bone1_name
        mesh1.parent_type = 'BONE'
        mesh1.matrix_parent_inverse = (
            armature.matrix_world @ Matrix.Translation(bone1.tail - bone1.head) @ bone1.matrix).inverted()

        # temporarily assign mesh 2 parent
        mesh2.parent = armature
        mesh2.parent_type = 'BONE'
        mesh2.parent_bone = bone2_name
        mesh2.matrix_parent_inverse = (
            armature.matrix_world @ Matrix.Translation(bone2.tail - bone2.head) @ bone2.matrix).inverted()

        
        if context.scene.ss.visual_mesh == 'SOLID':
            mesh1_visual.parent = armature
            mesh1_visual.parent_bone = bone1_name
            mesh1_visual.parent_type = 'BONE'
            mesh1_visual.matrix_parent_inverse = (
                armature.matrix_world @ Matrix.Translation(bone1.tail - bone1.head) @ bone1.matrix).inverted()
            mesh2_visual.parent = armature
            mesh2_visual.parent_type = 'BONE'
            mesh2_visual.parent_bone = bone2_name
            mesh2_visual.matrix_parent_inverse = (
                armature.matrix_world @ Matrix.Translation(bone2.tail - bone2.head) @ bone2.matrix).inverted()
        else:
            bpy.ops.object.mode_set(mode='OBJECT') #early enter obj mode
            mesh2_visual.hide_set(False)
            bpy.ops.object.select_all(action='DESELECT')
            mesh2_visual.select_set(True)
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.parent_set(type='ARMATURE_AUTO', xmirror=True)
            mesh2_visual.hide_set(True)

        # refresh to update posebone list
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='EDIT')
        posebone = armature.pose.bones.get(bone2_name)

        if 'Limit Rotation' not in posebone.constraints.keys():
            posebone.constraints.new('LIMIT_ROTATION')
        limit_rotation = posebone.constraints['Limit Rotation']
        limit_rotation.owner_space = 'LOCAL'
        limit_rotation.euler_order = context.scene.ss.rotation_convention

        if bone_axis_2 == 0:
            if bone_dir_2 == 1:  # point to +X: bone x => model -y; bone y => model x
                limit_rotation.min_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[1])
                limit_rotation.max_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[0])
                limit_rotation.min_y = math.radians(
                    context.scene.ss.joint_constraints_X[0])
                limit_rotation.max_y = math.radians(
                    context.scene.ss.joint_constraints_X[1])
                limit_rotation.min_z = math.radians(
                    context.scene.ss.joint_constraints_Z[0])
                limit_rotation.max_z = math.radians(
                    context.scene.ss.joint_constraints_Z[1])
            else:  # point to -X: bone x => model -y; bone y => model -x; bone z => model -z
                limit_rotation.min_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[1])
                limit_rotation.max_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[0])
                limit_rotation.min_y = - \
                    math.radians(context.scene.ss.joint_constraints_X[1])
                limit_rotation.max_y = - \
                    math.radians(context.scene.ss.joint_constraints_X[0])
                limit_rotation.min_z = - \
                    math.radians(context.scene.ss.joint_constraints_Z[1])
                limit_rotation.max_z = - \
                    math.radians(context.scene.ss.joint_constraints_Z[0])
        elif bone_axis_2 == 1:
            # point to +Y (model right-hand side): same rotation axis
            if bone_dir_2 == 1:
                limit_rotation.min_x = math.radians(
                    context.scene.ss.joint_constraints_X[0])
                limit_rotation.max_x = math.radians(
                    context.scene.ss.joint_constraints_X[1])
                limit_rotation.min_y = math.radians(
                    context.scene.ss.joint_constraints_Y[0])
                limit_rotation.max_y = math.radians(
                    context.scene.ss.joint_constraints_Y[1])
                limit_rotation.min_z = math.radians(
                    context.scene.ss.joint_constraints_Z[0])
                limit_rotation.max_z = math.radians(
                    context.scene.ss.joint_constraints_Z[1])
            else:  # point to -Y (model left-hand side): bone x => model -x; bone y => model -y
                limit_rotation.min_x = - \
                    math.radians(context.scene.ss.joint_constraints_X[1])
                limit_rotation.max_x = - \
                    math.radians(context.scene.ss.joint_constraints_X[0])
                limit_rotation.min_y = - \
                    math.radians(context.scene.ss.joint_constraints_Y[1])
                limit_rotation.max_y = - \
                    math.radians(context.scene.ss.joint_constraints_Y[0])
                limit_rotation.min_z = math.radians(
                    context.scene.ss.joint_constraints_Z[0])
                limit_rotation.max_z = math.radians(
                    context.scene.ss.joint_constraints_Z[1])
        else:
            # point to +Z (model upside): bone x => model -y; bone y => model z; bone z => model -x
            if bone_dir_2 == 1:
                limit_rotation.min_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[1])
                limit_rotation.max_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[0])
                limit_rotation.min_y = math.radians(
                    context.scene.ss.joint_constraints_Z[0])
                limit_rotation.max_y = math.radians(
                    context.scene.ss.joint_constraints_Z[1])
                limit_rotation.min_z = - \
                    math.radians(context.scene.ss.joint_constraints_X[1])
                limit_rotation.max_z = - \
                    math.radians(context.scene.ss.joint_constraints_X[0])
            else:  # point to -Z (model downside): bone x => model -y; bone y => model -z; bone z => model x
                limit_rotation.min_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[1])
                limit_rotation.max_x = - \
                    math.radians(context.scene.ss.joint_constraints_Y[0])
                limit_rotation.min_y = - \
                    math.radians(context.scene.ss.joint_constraints_Z[1])
                limit_rotation.max_y = - \
                    math.radians(context.scene.ss.joint_constraints_Z[0])
                limit_rotation.min_z = math.radians(
                    context.scene.ss.joint_constraints_X[0])
                limit_rotation.max_z = math.radians(
                    context.scene.ss.joint_constraints_X[1])
        # enable all rotation limits
        limit_rotation.use_limit_x = True
        limit_rotation.use_limit_y = True
        limit_rotation.use_limit_z = True
        bpy.ops.object.mode_set(mode='OBJECT')

        # auto increment
        self.report({'INFO'}, 'Successfully binded %s with %s using %s.' % (
            mesh1.name, mesh2.name, context.scene.ss.joint_name))
        if context.scene.ss.auto_increment and context.scene.ss.joint_name[-1].isdigit:
            context.scene.ss.joint_name = re.sub(
                r'[0-9]+$', lambda x: f"{str(int(x.group())+1).zfill(len(x.group()))}", context.scene.ss.joint_name)
        # resume selected
        for obj in selected_objects_bkup:
            obj.select_set(True)
        del selected_objects_bkup
        bpy.context.view_layer.objects.active = mesh2
        return {'FINISHED'}


class BindEnd(bpy.types.Operator):
    bl_idname = "wm.bind_end"
    bl_label = "Bind end"
    bl_description = "Bind end part with skeleton between mesh center and 3D cursor"

    @classmethod
    def poll(cls, context):
        if context.scene.gs.mesh_mode != 'COLLISION':
            cls.poll_message_set("Not in collision mesh mode.")
            return 0
        if context.object is None or context.active_object is None:
            cls.poll_message_set("No object selected.")
            return 0
        elif bpy.data.objects.get('Skeleton') is None:
            cls.poll_message_set("No head set.")
            return 0
        elif context.active_object.type != 'MESH':
            cls.poll_message_set("%s instead of MESH selected." %
                                 context.active_object.type)
            return 0
        elif len(context.selected_objects) != 1:
            cls.poll_message_set('%r instead of 1 mesh selected' %
                                 len(context.selected_objects))
            return 0
        mesh = context.object
        if mesh.parent is not None:
            if mesh.parent.name != 'Skeleton' or mesh.parent_bone is None:
                cls.poll_message_set(
                    'Selected mesh %s not attached to amature' % (mesh.name))
                return 0
        return 1

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')
        armature = bpy.data.objects['Skeleton']

        # index meshes
        mesh = context.object
        mesh_visual = bpy.data.objects.get(mesh.name[:-10])
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        deselect_bones(armature)
        armature = bpy.data.objects['Skeleton']

        # pointer to bone
        bone = armature.data.edit_bones.get(mesh.parent_bone)
        bone.tail = armature.matrix_world.inverted() @ bpy.context.scene.cursor.location
        
        
        bpy.ops.armature.collection_deselect()
        bone.select = True
        bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')

        mesh.matrix_parent_inverse = (
            armature.matrix_world @ Matrix.Translation(bone.tail - bone.head) @ bone.matrix).inverted()
        mesh_visual.matrix_parent_inverse = (
            armature.matrix_world @ Matrix.Translation(bone.tail - bone.head) @ bone.matrix).inverted()
        
        #joint_rot
        if context.scene.ss.rotation_mode == 'LOCAL':
            vec_base = mathutils.Vector((0, 0, 0))
            vec_sign = 1 if context.scene.ss.extend_axis[-1] == '+' else -1
            vec_base['XYZ'.index(
                context.scene.ss.extend_axis[0])] = 1 * vec_sign
            vec_tf = bone.tail - bone.head
            rot = vec_base.rotation_difference(vec_tf)
            mesh['joint_rot'] = rot.to_euler(
                context.scene.ss.rotation_convention)
            for i, rot in enumerate(mesh['joint_rot'][:]):
                if abs(rot)<1e-6:
                    mesh['joint_rot'][i] = 0
        else:
            mesh['joint_rot'] = mathutils.Vector((0,0,0))
            
        bpy.ops.object.mode_set(mode='OBJECT')
        # bpy.context.view_layer.objects.active = mesh
        
        
        self.report(
            {'INFO'}, 'Successfully binded %s with current 3D cursor location using.' % (mesh.name))
        return {'FINISHED'}



class Calc_inertia(bpy.types.Operator):
    # density #mass # pose # inert
    bl_idname = "wm.calc_inertia"
    bl_description = "Calculate inertia and add to custom properties"
    bl_label = "OVERWRITE previous inertia?"

    @classmethod
    def poll(cls, context):
        mesh_path = bpy.path.abspath(context.scene.es.export_path_mesh)
        if len(context.selected_objects) == 0:
            cls.poll_message_set("No objects selected.")
            return 0
        else:
            for obj in context.selected_objects:
                if obj.type != 'MESH':
                    continue
                if obj.rigid_body is None:
                    cls.poll_message_set(
                        "Mass of mesh %s undefined." % obj.name)
                    return 0
                elif obj.rigid_body.mass == 0:
                    cls.poll_message_set(
                        "Mass of mesh %s undefined." % obj.name)
                    return 0
        return 1

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            if obj.name.endswith('_assembly'):
                continue
            mesh_path = bpy.path.abspath(context.scene.es.export_path_mesh)
            mesh_name = obj.name if obj.name.endswith(
                '_collision') else obj.name + '_collision'
            mesh = os.path.join(mesh_path, mesh_name + '.stl')
            try:
                tri_mesh = tri.load_mesh(mesh)
            except Exception as e:
                self.report(
                    {'ERROR'}, 'Unable to load collision mesh of %s due to %s.' % (obj.name, e))
                return {'FINISHED'}
            # handle multi-textured combined objects
            tri_mesh = Calc_inertia.as_mesh(tri_mesh)
            tri_mesh.apply_transform(
                tri.transformations.scale_matrix(context.scene.ms.length_scale))
            tri_mesh.density *= obj.rigid_body.mass*(context.scene.ms.length_scale**3)/tri_mesh.mass
            inertia = tri_mesh.moment_inertia
            if not Calc_inertia.valid_inertia(inertia):
                raise ValueError(
                    'Mesh {} has inappropriate inertia {}'.format(
                        obj.name,
                        inertia)
                )
            # Ixx, Ixy, Ixz, Iyy, Iyz, Izz
            obj['inertia'] = [inertia[0, 0], inertia[0, 1], inertia[0, 2],
                              inertia[1, 1], inertia[1, 2], inertia[2, 2]]
            obj['com'] = tri_mesh.center_mass
        self.report({'INFO'}, 'Successfully computed inertia.')
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_confirm(self, event)

    @staticmethod
    def valid_inertia(inertia):
        """ Check if inertia matrix is positive, bounded and non zero. """
        ixx = inertia[0, 0]
        iyy = inertia[1, 1]
        izz = inertia[2, 2]
        (ixx, iyy, izz) = np.linalg.eigvals(inertia)
        positive_definite = np.all([i > 0.0 for i in (ixx, iyy, izz)])
        ineqaulity = (
            (ixx + iyy > izz) and (ixx + izz > iyy) and (iyy + izz > ixx))
        if not ineqaulity or not positive_definite:
            return False
        return True

    @staticmethod
    def as_mesh(tri_mesh):
        if isinstance(tri_mesh, tri.Scene):
            if len(tri_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = tri.util.concatenate(
                    tuple(tri.Trimesh(vertices=g.vertices, faces=g.faces)
                          for g in tri_mesh.geometry.values()))
        else:
            assert (isinstance(tri_mesh, tri.Trimesh))
            mesh = tri_mesh
        return mesh


class ExportMesh(bpy.types.Operator):
    bl_idname = "wm.export_mesh"
    bl_label = "Folder not empty. OVERWRITE?"
    bl_description = "Center and export mesh to file folder"
    scale = 1.0  # no scale here to keep visibility; scale in sdf for correct size

    @classmethod
    def poll(cls, context):
        if os.path.isdir(bpy.path.abspath(context.scene.es.export_path_mesh)):
            return 1
        else:
            cls.poll_message_set("Invalide export path.")
            return 0

    def execute(self, context):
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                obj.hide_set(False)
        mesh_path = bpy.path.abspath(context.scene.es.export_path_mesh)
        # remove old files but keep root folder and sym links
        for filename in os.listdir(mesh_path):
            filepath = os.path.join(mesh_path, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = bpy.data.objects[0]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if obj.name.endswith('_assembly'):
                    continue
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                current_pose = obj.matrix_world.copy()
                obj.matrix_world = Matrix()
                if obj.name.endswith('_collision'):
                    bpy.ops.wm.stl_export(
                        filepath=os.path.join(
                            mesh_path, obj.name + '.stl'),
                        check_existing=False,
                        export_selected_objects=True,
                        forward_axis='Y',
                        up_axis='Z',
                        #                    ascii_format = True, #human readable
                        global_scale=self.scale)
                else:
                    bpy.ops.wm.obj_export(
                        filepath=os.path.join(
                            mesh_path, obj.name + '.obj'),
                        check_existing=False,
                        export_selected_objects=True,
                        forward_axis='Y',
                        up_axis='Z',
                        export_triangulated_mesh=True,
                        global_scale=self.scale,
                        path_mode='RELATIVE')
                obj.select_set(False)
                obj.matrix_world = current_pose
                del current_pose
        update_mesh_mode(self, context)
        self.report({'INFO'}, 'Successfully exported meshes.')
        return {'FINISHED'}

    def invoke(self, context, event):
        if len(os.listdir(bpy.path.abspath(context.scene.es.export_path_mesh))) > 0:
            wm = context.window_manager
            return wm.invoke_confirm(self, event)
        else:
            return self.execute(context)


class ExportSDF(bpy.types.Operator):
    bl_idname = "wm.export_sdf"
    bl_label = "SDF exists. OVERWRITE?"
    bl_description = "Export links and joints properties to SDF"

    @classmethod
    def poll(cls, context):
        if os.path.isdir(os.path.dirname(context.scene.es.export_path_sdf)):
            return 1
        else:
            cls.poll_message_set("Invalide export path.")
            return 0

    def execute(self, context):
        sdf_name = bpy.path.abspath(context.scene.es.export_path_sdf)
        try:
            os.remove(sdf_name)
        except:
            pass

        sdf = ET.Element("sdf", version="1.6")
        world = ET.SubElement(sdf, "world", name="world")
        model = ET.SubElement(world, "model", name=bpy.path.basename(
            bpy.context.blend_data.filepath)[:-6])
        pose = ET.SubElement(model, "pose")
        pose.text = " ".join(['0.0']*6)
        for mesh in bpy.data.objects:
            if mesh.type != 'MESH' or not mesh.name.endswith('_collision'):
                continue
            print(mesh.name[:-10])
            link = ET.SubElement(model, "link", name=mesh.name[:-10])

            link_pose = ET.SubElement(link, "pose")
            link_pose.text = " ".join(
                [str(float('%.6g' % (pos*context.scene.ms.length_scale)))
                 if abs(pos) > EPSILON else '0'
                 for pos in list(mesh.matrix_world.to_translation()[:])]
            ) + " " + " ".join([str(rot)
                                if abs(rot) > EPSILON else '0'
                                for rot in list(mesh.matrix_world.to_euler('XYZ')[:])])

            link_inertial = ET.SubElement(link, "inertial")
            link_center = ET.SubElement(link_inertial, "pose")
            link_center.text = " ".join(
                [str(float('%.6g' % c)) for c in mesh['com']]) + " " + " ".join(['0']*3)
            link_mass = ET.SubElement(link_inertial, "mass")
            link_mass.text = str(float('%.6g' % (mesh.rigid_body.mass / (context.scene.ms.length_scale**3))))
            link_density = ET.SubElement(link_inertial, "density")
            link_density.text = str(float('%.6g' % (mesh['density'])))
            link_inertia = ET.SubElement(link_inertial, "inertia")
            link_inertias = [ET.SubElement(link_inertia, name) for name in [
                "ixx", "ixy", "ixz", "iyy", "iyz", "izz"]]
            for i, inertia in enumerate(link_inertias):
                inertia.text = str(float('%.6g' % mesh['inertia'][i]))
            # Ixx, Iyy, Izz, Ixy, Ixz, Iyz
            link_collision = ET.SubElement(
                link, "collision", name=mesh.name)
            link_collision_pose = ET.SubElement(link_collision, "pose")
            link_collision_pose.text = " ".join(['0.0']*6)
            link_collision_geo = ET.SubElement(link_collision, "geometry")
            link_collision_mesh = ET.SubElement(link_collision_geo, "mesh")
            link_collision_uri = ET.SubElement(link_collision_mesh, "uri")
            link_collision_scale = ET.SubElement(link_collision_mesh, "scale")
            link_collision_scale.text = " ".join(
                ['%s' % float('%.3g' % context.scene.ms.length_scale)]*3)

            mesh_path = bpy.path.abspath(context.scene.es.export_path_mesh)
            link_collision_uri.text = os.path.relpath(
                os.path.join(mesh_path, mesh.name + '.stl'),
                os.path.dirname(bpy.data.filepath))

            link_visual = ET.SubElement(
                link, "visual", name=mesh.name[:-10] + '_visual')
            link_visual_pose = ET.SubElement(link_visual, "pose")
            link_visual_pose.text = link_collision_pose.text
            link_visual_geo = ET.SubElement(link_visual, "geometry")
            link_visual_mesh = ET.SubElement(link_visual_geo, "mesh")
            link_visual_uri = ET.SubElement(link_visual_mesh, "uri")
            link_visual_scale = ET.SubElement(link_visual_mesh, "scale")
            link_visual_scale.text = link_collision_scale.text
            link_visual_uri.text = os.path.relpath(
                os.path.join(mesh_path, mesh.name[:-10] + '.obj'),
                os.path.dirname(bpy.data.filepath))

        bpy.ops.object.mode_set(mode='OBJECT')
        armature = bpy.data.objects['Skeleton']
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        # record end joint info (tail, claws..,) for 3D kinematics
        end_links = []
        end_pose = []
        for bone in armature.data.edit_bones:
            if bone.name == 'joint_base':
                continue
            deselect_bones(armature)
            armature = bpy.data.objects['Skeleton']
            posebone = armature.pose.bones.get(bone.name)
            if posebone.constraints is None:
                raise KeyError(
                    'Rotation constraints UNDEFINED in %s' % bone.name)
            limit_rotation = posebone.constraints['Limit Rotation']

            mesh1 = None
            mesh2 = None
            for mesh in armature.children:
                if not mesh.name.endswith('_collision'):
                    continue
                if mesh1 is not None and mesh2 is not None:
                    break  # user ensures each bone has at most 1 child mesh
                if mesh.parent_type == "BONE" and mesh.parent_bone == bone.name:
                    mesh2 = mesh
                elif mesh.parent_type == "BONE" and mesh.parent_bone == bone.parent.name:
                    mesh1 = mesh
            # get actual rotation limits
            bone_dir = mesh2['bone_dir']
            bone_axis = mesh2['bone_axis']
            if bone_axis == 0:
                if bone_dir == 1:  # point to +X: bone x => model -y; bone y => model x
                    limit_x_lower = limit_rotation.min_y
                    limit_x_upper = limit_rotation.max_y
                    limit_y_lower = -limit_rotation.max_x
                    limit_y_upper = -limit_rotation.min_x
                    limit_z_lower = limit_rotation.min_z
                    limit_z_upper = limit_rotation.max_z
                else:  # point to -X: bone x => model -y; bone y => model -x; bone z => model -z
                    limit_x_lower = -limit_rotation.max_y
                    limit_x_upper = -limit_rotation.min_y
                    limit_y_lower = -limit_rotation.max_x
                    limit_y_upper = -limit_rotation.min_x
                    limit_z_lower = -limit_rotation.max_z
                    limit_z_upper = -limit_rotation.min_z
            elif bone_axis == 1:
                # point to +Y (model right-hand side): same rotation axis
                if bone_dir == 1:
                    limit_x_lower = limit_rotation.min_x
                    limit_x_upper = limit_rotation.max_x
                    limit_y_lower = limit_rotation.min_y
                    limit_y_upper = limit_rotation.max_y
                    limit_z_lower = limit_rotation.min_z
                    limit_z_upper = limit_rotation.max_z
                else:  # point to -Y (model left-hand side): bone x => model -x; bone y => model -y
                    limit_x_lower = -limit_rotation.max_x
                    limit_x_upper = -limit_rotation.min_x
                    limit_y_lower = -limit_rotation.max_y
                    limit_y_upper = -limit_rotation.min_y
                    limit_z_lower = limit_rotation.min_z
                    limit_z_upper = limit_rotation.max_z
            else:
                # point to +Z (model upside): bone x => model -y; bone y => model z; bone z => model -x
                if bone_dir == 1:
                    limit_x_lower = -limit_rotation.max_z
                    limit_x_upper = -limit_rotation.min_z
                    limit_y_lower = -limit_rotation.max_x
                    limit_y_upper = -limit_rotation.min_x
                    limit_z_lower = limit_rotation.min_y
                    limit_z_upper = limit_rotation.max_y
                else:  # point to -Z (model downside): bone x => model -y; bone y => model -z; bone z => model x
                    limit_x_lower = limit_rotation.min_z
                    limit_x_upper = limit_rotation.max_z
                    limit_y_lower = -limit_rotation.max_x
                    limit_y_upper = -limit_rotation.min_x
                    limit_z_lower = -limit_rotation.max_y
                    limit_z_upper = -limit_rotation.min_y

            suffix = [
                '_' + dof for dof in list(limit_rotation.euler_order.lower())]
            #create joint with dof suffix
            for s in suffix.copy():
                if s == '_x':
                    if abs(limit_x_lower)+abs(limit_x_upper) > EPSILON:
                        j = ET.SubElement(
                            model, "joint", name=bone.name + mesh2['passive'][0] * '_passive' + s, type='revolute')
                        if mesh2['passive'][0]:
                            suffix[suffix.index(s)] = '_passive' + s
                    else:
                        suffix.remove(s)
                elif s == '_y':
                    if abs(limit_y_lower)+abs(limit_y_upper) > EPSILON:
                        j = ET.SubElement(
                            model, "joint", name=bone.name + mesh2['passive'][1] * '_passive'+ s, type='revolute')
                        if mesh2['passive'][1]:
                            suffix[suffix.index(s)] = '_passive' + s
                    else:
                        suffix.remove(s)
                elif s == '_z':
                    if abs(limit_z_lower)+abs(limit_z_upper) > EPSILON:
                        j = ET.SubElement(
                            model, "joint", name=bone.name + mesh2['passive'][2] * '_passive'+ s, type='revolute')
                        if mesh2['passive'][2]:
                            suffix[suffix.index(s)] = '_passive' + s
                    else:
                        suffix.remove(s)
            #create link for mdof joints
            
            if len(suffix) > 1:  # create virtual links if multi-dofs
                #                print(suffix)
                #                print('x:,', limit_x_lower, limit_x_upper)
                #                print('y:,', limit_y_lower, limit_y_upper)
                #                print(mesh2.name[:-10])
                link_mesh2 = model.find(
                    ".//link[@name='%s']" % mesh2.name[:-10]) #base mesh locates at end of kinematic chain
                for dof in suffix[:-1]:
                    link = copy.deepcopy(link_mesh2)
                    link.set('name', link.attrib['name'] + dof)
                    for element in link.findall('inertial'):
                        link.remove(element)
                    for element in link.findall('collision'):
                        link.remove(element)
                    for element in link.findall('visual'):
                        link.remove(element)
                    #virtual mesh inserts on top of base mesh -Y, -Z-Y, -Z-X-Y => -Z-X-Y-
                    model.insert(model.findall(
                        ".//link").index(link_mesh2) + 1, link)
                    print(link.get('name'))
                link_mesh2.set('name', link_mesh2.attrib['name']) #base link always no suffix for stable joint convention
            else:
                link_mesh2 = model.find(
                    ".//link[@name='%s']" % mesh2.name[:-10])
                link_mesh2.set('name', link_mesh2.attrib['name'])
            
            count_passive_dof = sum(1 for s in suffix if '_passive' in s)
            count_active_dof = len(suffix) - count_passive_dof
            joint_suffix = suffix.copy()
            for i, dof in enumerate(suffix): #abbreviating all but last dof (base dof) where necessary
                if count_passive_dof == 1 and '_passive' in dof or count_active_dof == 1 and '_passive' not in dof:
                    joint = model.find(
                    ".//joint[@name='%s']" % (bone.name + dof))
                    joint.set('name', joint.attrib['name'][:-2])
                    # link = model.find(
                    # ".//link[@name='%s']" % (mesh2.name[:-10] + dof) if i < len(suffix) - 1 else mesh2.name[:-10])
                    # link.set('name', link.attrib['name'][:-2])
                    joint_suffix[i] = dof[:-2]
            print('suffix',suffix)
            joint = None
            for i, dof in enumerate(joint_suffix):
                joint = model.find(
                    ".//joint[@name='%s']" % (bone.name + dof))
                joint_parent = ET.SubElement(joint, "parent")
                joint_child = ET.SubElement(joint, "child")
                joint_pose = ET.SubElement(joint, "pose")
                joint_pose_temp = [0.0] * 6
                joint_pose_temp[3:] = mesh1['joint_rot']
                joint_pose_temp = ['0' if abs(joint_pose) < EPSILON else str(
                    joint_pose) for joint_pose in joint_pose_temp]
                joint_pose.text = ' '.join(joint_pose_temp)
                joint_axis = ET.SubElement(joint, "axis")
                joint_raxis = ET.SubElement(joint_axis, "xyz")
                joint_rlim = ET.SubElement(joint_axis, "limit")
                joint_rlim_lb = ET.SubElement(joint_rlim, "lower")
                joint_rlim_ub = ET.SubElement(joint_rlim, "upper")
                
                if i == 0: #parent is last link of last link
                    joint_parent.text = mesh1.name[:-10]
                else:
                    joint_parent.text = mesh2.name[:-10] + suffix[i-1]
                if i == len(suffix)-1: #parent is last link of this link
                    joint_child.text = mesh2.name[:-10]
                else:
                    joint_child.text = mesh2.name[:-10] + suffix[i]
                if '_x' in suffix[i]:
                    if mesh2['right']:
                        joint_raxis.text = "-1.0 0.0 0.0"
                        joint_rlim_lb.text = str('%.6g' % -limit_x_upper)
                        joint_rlim_ub.text = str('%.6g' % -limit_x_lower)
                    else:
                        joint_raxis.text = "1.0 0.0 0.0"
                        joint_rlim_lb.text = str('%.6g' % limit_x_lower)
                        joint_rlim_ub.text = str('%.6g' % limit_x_upper)
                elif '_y' in suffix[i]:
                    joint_raxis.text = "0.0 1.0 0.0"
                    joint_rlim_lb.text = str('%.6g' % (limit_y_lower))
                    joint_rlim_ub.text = str('%.6g' % (limit_y_upper))
                else:
                    if mesh2['left']:
                        joint_raxis.text = "0.0 0.0 -1.0"
                        joint_rlim_lb.text = str('%.6g' % -limit_z_upper)
                        joint_rlim_ub.text = str('%.6g' % -limit_z_lower)
                    else:
                        joint_raxis.text = "0.0 0.0 1.0"
                        joint_rlim_lb.text = str('%.6g' % limit_z_lower)
                        joint_rlim_ub.text = str('%.6g' % limit_z_upper)
                    
                # effort, vel not implemented
            if len(bone.children) == 0:
                end_links.append(link_mesh2.attrib['name'])#+ joint_suffix[-1]
                end_pose.append(" ".join(
                [str(float('%.6g' % (pos*context.scene.ms.length_scale)))
                 if abs(pos) > EPSILON else '0'
                 for pos in list(bone.tail)]))
        #store tail bones for 3D kinematics
        for i, end_link in enumerate(end_links):
            linkend = ET.SubElement(model, "linkend", name=end_link)
            linkend_tail = ET.SubElement(linkend, "pose")
            linkend_tail.text = end_pose[i]
                
        bpy.ops.object.mode_set(mode='OBJECT')
        sdf_str = ET.tostring(
            sdf,
            encoding='utf8',
            method='xml'
        ).decode('utf8')
        sdf_pretty = xml.dom.minidom.parseString(sdf_str).toprettyxml()
        with open(context.scene.es.export_path_sdf, "w+") as sdf_file:
            sdf_file.write(sdf_pretty)
        self.report({'INFO'}, 'Successfully exported SDF.')
        return {'FINISHED'}

    def invoke(self, context, event):
        if os.path.isfile(context.scene.es.export_path_sdf):
            wm = context.window_manager
            return wm.invoke_confirm(self, event)
        else:
            return self.execute(context)


class farms_blenimat_panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "farms_Blenimat"  # found in the Sidebar
    bl_options = {"DEFAULT_CLOSED"}


class VIEW3D_PT_farms_blenimat_mesh(farms_blenimat_panel, bpy.types.Panel):
    bl_label = "Mesh"  # found at the top of the Panel

    def draw(self, context):
        """define the layout of the panel"""
        tool = context.scene.ms
        layout = self.layout

        layout.label(text="Modify mesh properties")
        layout.prop(tool, "length_scale", text="Length scale")

        row1 = layout.row()
        row1.prop(tool, "mass", text="Mass")
        row1.prop(tool, "density", text="Density")

        row2 = layout.row()
        op1 = row2.operator(Assign_mass.bl_idname, text="Set Mass")
        op2 = row2.operator(Assign_density.bl_idname, text="Set Density")

        layout.prop(tool, "import_length_scale", text='Import Scale')
#        layout.label(text="Import inertia: (used for robots)")
        row3 = layout.row()
        row3.prop(tool, "import_path_mesh", text="Mesh")
        layout.operator(Import_mesh.bl_idname, text="Import")

        row4 = layout.row()
        row4.prop(tool, "import_path_inertia", text="Inertia")
        row5 = layout.row()
        row5.operator(Import_inertia.bl_idname, text="Import")
        row6 = layout.row()
        row6.operator(Set_Left.bl_idname, text='Set Left')
        row6.operator(Set_Right.bl_idname, text='Set Right')
        row6.operator(Clear_side.bl_idname, text='Clear Side')
        layout.operator(Center_mesh.bl_idname,text = 'Center Mesh')

class VIEW3D_PT_farms_blenimat_geometry(farms_blenimat_panel, bpy.types.Panel):
    bl_label = "Geometry"  # found at the top of the Panel

    def draw(self, context):
        """define the layout of the panel"""
        tool = context.scene.gs
        layout = self.layout
        row7 = layout.row()
        split = row7.split(factor=0.4)
        rwo71 = split.column()
        split = split.split()
        row72 = split.column()
        layout.operator(Create_collision.bl_idname,
                        text="Create collision meshes")
        layout.prop(tool, "mesh_mode", text="Mesh Mode", expand=True)

        rwo71.operator(Polycount.bl_idname, text="Polycount")
        row72.label(text="%r/%r" % (tool.polynum, tool.polynum_sum))
        row75 = layout.row()
        row75.operator(Remesh.bl_idname, text="Remesh")
        row75.prop(tool, 'oct_tree_depth')
        row76 = layout.row()
        row76.operator(Smooth.bl_idname, text='Smooth')
        row76.prop(tool, 'smooth_angle')
        row8 = layout.row()
        row8.operator(Dissolve.bl_idname, text="Dissolve")
        row8.prop(tool, 'limited_dissolve')
        row9 = layout.row()
        row9.operator(Decimate.bl_idname, text="Decimate")
        row9.prop(tool, 'decimate')


class VIEW3D_PT_farms_blenimat_skeleton(farms_blenimat_panel, bpy.types.Panel):
    bl_label = "Skeleton"  # found at the top of the Panel

    def draw(self, context):
        """define the layout of the panel"""
        tool = context.scene.ss
        layout = self.layout
        layout.label(text="Build skeleton tree")
        row1 = layout.row()
        op1 = row1.operator(Clear_Head.bl_idname, text="Clear head")
        op2 = row1.operator(Set_Head.bl_idname, text="Set head")

        layout.prop(tool, "display_mode", text="Display Mode", expand=True)
        layout.prop(tool, "visual_mesh", text="Visual Mesh", expand=True)

        # button2 bind amature
        layout.prop(tool, "rotation_mode", text="Mode")
        layout.prop(tool, "rotation_convention", text="Order")
        layout.prop(tool, "joint_name", text="Name")
        
        lrow = layout.row()
        split = lrow.split(factor = 0.7)
        lrow1 = split.column()
        split = split.split()
        lrow2 = split.column()
        lrow1.prop(tool, "auto_increment", text="Auto-increment")
        lrow2.alignment = 'RIGHT'
        lrow2.label(text='passive')
    
        rowx = layout.row()
        rowx.prop(tool, "joint_constraints_X", text = 'X')
        rowx.prop(tool,"joint_passive_x")
        
        rowy = layout.row()
        rowy.prop(tool, "joint_constraints_Y", text="Y")
        rowy.prop(tool,"joint_passive_y")
        
        rowz = layout.row()
        rowz.prop(tool, "joint_constraints_Z", text="Z")
        rowz.prop(tool,"joint_passive_z")

        row3 = layout.row()
        split = row3.split(factor=0.7)
        rwo31 = split.column()
        split = split.split()
        row32 = split.column()
        rwo31.prop(tool, "disconnected", text="Disconnected")
        row32.prop(tool, "extend_axis", text="")

        row4 = layout.row()
        op3 = row4.operator(Bind.bl_idname, text="Bind Skeleton")
        op4 = row4.operator(BindEnd.bl_idname, text="Bind End Skeleton")
#        row4 = layout.row()
#        op5 = row4.operator(FixName.bl_idname, text="Fixname")


class VIEW3D_PT_farms_blenimat_export(farms_blenimat_panel, bpy.types.Panel):
    bl_label = "Export"  # found at the top of the Panel

    def draw(self, context):
        """define the layout of the panel"""
        tool = context.scene.es
        layout = self.layout
        layout.label(text="Export meshes and sdf")
        layout.prop(tool, "export_path_mesh", text="Mesh")
        layout.prop(tool, "export_path_sdf", text="SDF")
        row1 = self.layout.row()
        op1 = row1.operator(ExportMesh.bl_idname, text="Export mesh")

        row2 = self.layout.row()
        op2 = row2.operator(Calc_inertia.bl_idname, text="Calc inertia")
        op3 = row2.operator(ExportSDF.bl_idname, text="Export SDF")


classes = (
    MeshSettings,
    GeometrySettings,
    SkeletonSettings,
    ExportSettings,

    VIEW3D_PT_farms_blenimat_mesh,
    VIEW3D_PT_farms_blenimat_geometry,
    VIEW3D_PT_farms_blenimat_skeleton,
    VIEW3D_PT_farms_blenimat_export,

    Assign_mass,
    Assign_density,
    Import_mesh,
    Import_inertia,
    Create_collision,
    Polycount,
    Dissolve,
    Decimate,
    Remesh,
    Smooth,
    Set_Left,
    Set_Right,
    Clear_side,
    Center_mesh,

    Clear_Head,
    Set_Head,
    Bind,
    BindEnd,

    Calc_inertia,
    ExportMesh,
    ExportSDF,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)

    bpy.types.Scene.ms = PointerProperty(type=MeshSettings)
    bpy.types.Scene.ss = PointerProperty(type=SkeletonSettings)
    bpy.types.Scene.es = PointerProperty(type=ExportSettings)
    bpy.types.Scene.gs = PointerProperty(type=GeometrySettings)
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            for s in a.spaces:
                if s.type == 'VIEW_3D':
                    s.clip_end = 10000
                    s.clip_start = 0.01
    bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'


def unregister():
    from bpy.utils import unregister_class

    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.ss
    del bpy.types.Scene.es
    del bpy.types.Scene.ms
    del bpy.types.Scene.gs


if __name__ == "__main__":
    register()

