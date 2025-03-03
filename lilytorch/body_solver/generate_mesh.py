
from sdf import *
import trimesh
import matplotlib.pyplot as plt

def def_box():
    f = box((1, 0.3, 2))
    f.name = 'box.obj'
    return f

def def_cylinder():
    radius = 21
    f = capped_cylinder([0,0,-2*radius], [0,0,2*radius], radius).translate((0, 3*radius, 0))
    f.name = 'cylinder.obj'
    return f

f=def_cylinder()


f.save(f.name, samples=2**16)


# 

# mesh_file = "models/textured_simple_reoriented.obj"
# mesh_file = "models/zebrafish_v1/sdf/meshes_zebrafish/link_0.obj"

mesh = trimesh.load_mesh(f.name)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces, Z=mesh.vertices[:,2]) 
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()