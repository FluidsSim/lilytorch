

import pytorch_volumetric as pv
import pytorch_kinematics as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
import trimesh 
from mpl_toolkits.mplot3d import Axes3D

d = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# mesh_file="box.obj"
# mesh_file="cylinder.obj"
# n=1000
# x=torch.linspace(-1.5,1.5,n,dtype=dtype)
# y=torch.linspace(-1.2,1.2,n,dtype=dtype)

mesh_file="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_1.obj"
n=2**10
x=torch.linspace(-0.1,0.1,n,dtype=dtype)
y=torch.linspace(-0.1,0.1,n,dtype=dtype)


X,Y=torch.meshgrid(x,y)

mesh=trimesh.load(mesh_file)
obj=pv.MeshObjectFactory(mesh_file)
sdf=pv.MeshSDF(obj)
sdf_cached=pv.CachedSDF(
    'drill', 
    resolution=1/n, 
    range_per_dim=obj.bounding_box(padding=0.1), 
    gt_sdf=sdf,
    device=d,
    dtype=dtype
    )

coords=[
    x,
    y,
    torch.tensor([0.0], dtype=dtype),
    ]
pts = torch.cartesian_prod(*coords).to(d)


sdf_val, sdf_grad = sdf(pts)
sdf_val = sdf_val.reshape(len(x), len(y))
sdf_grad = sdf_grad.reshape(len(x), len(y), 3)


sdf_val_cached, sdf_grad_cached = sdf_cached(pts)
sdf_val_cached = sdf_val_cached.reshape(len(x), len(y))
sdf_grad_cached = sdf_grad_cached.reshape(len(x), len(y), 3)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces, Z=mesh.vertices[:,2]) 
ax.set_xlabel("x")
ax.set_ylabel("y")


plt.figure()
norm = matplotlib.colors.Normalize(vmin=sdf_val.min(), vmax=sdf_val.max())
cset1 = plt.contourf(X,Y,sdf_val, cmap="Greys")
cset2 = plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
plt.colorbar(cset1)



plt.figure()
norm = matplotlib.colors.Normalize(vmin=sdf_val_cached.min(), vmax=sdf_val_cached.max())
cset1 = plt.contourf(X,Y,sdf_val_cached, cmap="Greys")
cset2 = plt.contour(X,Y,sdf_val_cached, colors='k', levels=[0], linestyles='dashed')
plt.colorbar(cset1)



plt.show()