

import torch
from test_open3d import mesh2sdf
import matplotlib.pyplot as plt 
import numpy as np
import plotting
from pytorch_interp import RegularGridInterpolator

# load mesh file in open3d and compute the sdf 

# mesh_file="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_1.obj"
# min_x=-0.005
# max_x=0.005
# min_y=-0.005
# max_y=0.005

mesh_file="box.obj"
min_x=-1
max_x=1
min_y=-1
max_y=1

m2s = mesh2sdf(mesh_file)
m2s.rototranslate_3d(pos=(0,0,0))

dtype = np.float32


dx=max_x-min_x
dy=max_y-min_y
diag=2*np.sqrt(dx**2+dy**2)
n=2**12
x=np.linspace(min_x,max_x,n).astype(dtype)
y=np.linspace(min_y,max_y,n).astype(dtype)
# x=np.linspace(-diag,diag,n).astype(dtype)
# y=np.linspace(-diag,diag,n).astype(dtype)
X,Y=np.meshgrid(x,y,indexing="ij")


xflat = X.flatten()
yflat = Y.flatten()
zflat = np.zeros_like(xflat)
xflat = xflat.astype(dtype)
yflat = yflat.astype(dtype)
xyz   = np.stack([xflat,yflat,zflat],axis=1)

query_pts=np.array(xyz,dtype=dtype)


sdf_val, sdf_grad=m2s(query_pts)

data_plot = np.stack( (
        xflat,
        yflat,
        sdf_val
        )
    )

plotting.plot2D(
    data_plot,
    ["x","y","z"]
)


# ==== convert data to torch and do bilinear interpolation
device ="cpu"
print("Running on {} device".format(device))

xtorch = torch.from_numpy(x).to(device)
ytorch = torch.from_numpy(y).to(device)
sdf_torch = torch.from_numpy(sdf_val).to(device)


xy_torch = torch.from_numpy( 
    np.stack( 
        (
        x,
        y,
        )
    ).T
).to(device)


interp = RegularGridInterpolator((xtorch,ytorch), sdf_torch, fill_value=None)

N = 2**7

xx = torch.linspace(min_y,max_y,N)
yy = torch.linspace(min_x,max_x,N)
XQ, YQ = torch.meshgrid(xx,yy,indexing="ij")
xxflat = XQ.flatten()
yyflat = YQ.flatten()

points = torch.stack((xxflat,yyflat))

t = [0.,0]
theta = torch.tensor(0 * torch.pi / 180)
s = torch.sin(theta)
c = torch.cos(theta)
rot = torch.stack([torch.stack([c, -s]),
                   torch.stack([s, c])])
trans = torch.stack((t[0]*torch.ones(N**2), t[1]*torch.ones(N**2)))

newpoints=rot.T@points-trans
xquery=newpoints[0]
yquery=newpoints[1]



plt.figure()
plt.subplot(1,2,1)
plt.scatter(points[0],points[1])
plt.subplot(1,2,2)
plt.scatter(newpoints[0],newpoints[1])





# xquery = (max_x-min_x)*torch.rand(N)+min_x
# yquery = (max_y-min_y)*torch.rand(N)+min_y

sdf_query = interp(xquery,yquery)



plt.figure()
data_plot = np.stack( (
        xxflat.cpu(),
        yyflat.cpu(),
        sdf_query.cpu()
        )
    )

plotting.plot2D(
    data_plot,
    ["x","y","z"]
)







plt.show()

# from IPython import embed; embed()







