
import torch
import pytorch_kinematics as pk
# import pytorch_volumetric as pv
from model_to_sdf import RobotSDF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

d = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

file = "lilytorch/body_solver/mesh_examples/URDF_models/kuka_iiwa.urdf"
mesh_folder = "/data/andreaferrario/navier_stokes/lilytorch/body_solver/mesh_examples/URDF_models/"
n=100
x=torch.linspace(-1,0.5,n, dtype=dtype)
y=torch.linspace(-0.2,0.8,n, dtype=dtype)

# file = "/data/andreaferrario/navier_stokes/models/zebrafish_v1_tr/zebrafish.sdf"
# mesh_folder = "/data/andreaferrario/navier_stokes/lilytorch/body_solver/mesh_examples/URDF_models/"
# n=100
# x=torch.linspace(-1,0.5,n, dtype=dtype)
# y=torch.linspace(-0.2,0.8,n, dtype=dtype)


chain = pk.build_serial_chain_from_urdf(open(file).read(), "lbr_iiwa_link_7")
chain = chain.to(dtype=dtype, device=d)

N = 1000
th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)

# order of magnitudes faster when doing FK in parallel
# elapsed 0.008678913116455078s for N=1000 when parallel
# (N,4,4) transform matrix; only the one for the end effector is returned since end_only=True by default
tg_batch = chain.forward_kinematics(th_batch)

s = RobotSDF(chain, path_prefix=mesh_folder)


X,Y=torch.meshgrid(x,y)
coords=[
    x,
    y,
    torch.tensor([0.02], dtype=dtype),
    ]
pts = torch.cartesian_prod(*coords).to(d)
sdfs, grads = s(pts)



# reshape val and grad
for sdf_val, sdf_grad in zip(sdfs, grads):
    sdf_val = sdf_val.reshape(len(x), len(y))
    sdf_grad = sdf_grad.reshape(len(x), len(y), 3)


    plt.figure()
    norm = matplotlib.colors.Normalize(vmin=sdf_val.min(), vmax=sdf_val.max())
    cset1 = plt.contourf(X,Y,sdf_val, cmap="Greys")
    cset2 = plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
    plt.colorbar(cset1)




# # # elapsed 8.44686508178711s for N=1000 when serial
# # for i in range(N):
# #     tg = chain.forward_kinematics(th_batch[i])

# from IPython import embed; embed()



plt.show()











