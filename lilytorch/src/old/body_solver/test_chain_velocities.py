

import torch
import matplotlib.pyplot as plt
import skfmm


def compute_dpdx(p,dx):
    """
    Compute dp/dx
    """
    return torch.gradient(p, spacing=dx, dim=0)[0]

def compute_dpdy(p,dy):
    """
    Compute dp/dy
    """
    return torch.gradient(p, spacing=dy, dim=1)[0]

def vorticity(u, v, dx, dy):
    """
    Compute the vorticity of u,v in 2d
    """
    return compute_dpdx(v,dx)-compute_dpdy(u,dy)

def compute_angles(points):

    dv_par = torch.diff(points,axis=0)
    dv_par = dv_par / torch.linalg.norm(dv_par,axis=1)[:, None]

    dv_ort = torch.zeros_like(dv_par)
    dv_ort[:,0]=-dv_par[:,1]
    dv_ort[:,1]=dv_par[:,0]

    # Project onto previous links
    dv_proj_par = torch.sum(
        dv_par[1:,:] * dv_par[:-1,:],
        axis=1,
    )
    dv_proj_ort = torch.sum(
        dv_par[1:,:] * dv_ort[:-1,:],
        axis=1,
    )

    return torch.arctan2(
        dv_proj_ort,
        dv_proj_par,
    )



def sdUnevenCapsule(xy, r1, r2, h):
    xy[0]=torch.abs(xy[0])
    b=(r1-r2)/h
    a=torch.sqrt(1.0-b*b)
    k = -b*xy[0]+a*xy[1]
    return torch.where(
        k<0.0, torch.sqrt(xy[0]**2+xy[1]**2)-r1,
        torch.where(
            k>a*h, torch.sqrt(xy[0]**2+(xy[1]-h)**2)-r2,
            a*xy[0]+b*xy[1]-r1
        )
    )


nx=2**8
ny=2**7
x=torch.linspace(-12,12,nx)
y=torch.linspace(-5,5,ny)
X,Y = torch.meshgrid(x,y,indexing='ij')
dx=x[1]-x[0]
dy=y[1]-y[0]
xy = torch.stack((X.flatten(),Y.flatten()), axis=0)
dt=0.1
# joint positions
p=torch.tensor([
    [-6.0,0.0],
    [-3.0,0.0],
    [-1.0,1.0],
])
# rotato-translation map in world frames
com_lin_vel=0*torch.tensor([[0.4,0],[0.2,0.4],[0.2,0.4]])
com_ang_vel=torch.tensor([0.1*torch.pi,-0.3*torch.pi,-0.2*torch.pi])



# nx=2**9
# ny=2**9
# x=torch.linspace(-0.005,0.01,nx)
# y=torch.linspace(-0.003,0.003,ny)
# X,Y = torch.meshgrid(x,y,indexing='ij')
# dx=x[1]-x[0]
# dy=y[1]-y[0]
# xy = torch.stack((X.flatten(),Y.flatten()), axis=0)
# dt=1
# com_ang_vel=torch.tensor([-0.0100, -0.0615, -0.0597, -0.0387, -0.0135,  0.0124,  0.0419,  0.0774,
#          0.1107,  0.1214,  0.0823, -0.0235, -0.1902, -0.3893, -0.5885, -0.5885],
#        dtype=torch.float64)
# com_lin_vel=torch.tensor([[ 2.6147e-07, -5.2145e-03],
#         [-1.7992e-06, -5.3326e-03],
#         [ 1.1589e-05, -1.6195e-03],
#         [ 2.3814e-05,  2.0833e-03],
#         [ 3.0270e-05,  4.6426e-03],
#         [ 3.0997e-05,  5.7091e-03],
#         [ 2.9803e-05,  5.4361e-03],
#         [ 3.0223e-05,  3.8072e-03],
#         [ 4.4273e-05,  1.9078e-04],
#         [ 7.9071e-05, -5.7111e-03],
#         [ 1.3530e-04, -1.3072e-02],
#         [ 1.6731e-04, -2.0636e-02],
#         [ 1.1420e-04, -2.3753e-02],
#         [ 7.6819e-05, -1.7107e-02],
#         [ 3.0659e-04,  7.1351e-04],
#         [ 3.0659e-04,  7.1351e-04]], dtype=torch.float64)
# p=torch.tensor([[ 5.4271e-10, -6.0890e-06],
#         [-3.0000e-03, -4.3690e-06],
#         [-4.0000e-03, -8.4266e-07],
#         [-5.0000e-03,  2.5767e-06],
#         [-6.0000e-03,  4.7929e-06],
#         [-7.0000e-03,  5.5644e-06],
#         [-8.0000e-03,  4.8546e-06],
#         [-9.0000e-03,  2.4535e-06],
#         [-1.0000e-02, -1.9821e-06],
#         [-1.1000e-02, -8.3248e-06],
#         [-1.2000e-02, -1.5283e-05],
#         [-1.3000e-02, -2.0000e-05],
#         [-1.4000e-02, -1.8652e-05],
#         [-1.5000e-02, -7.7538e-06],
#         [-1.6000e-02,  1.4549e-05],
#         [-1.6999e-02,  4.8259e-05],
#         ], dtype=torch.float64)



# com_ang_vel=torch.tensor([-0.0100, -0.0615, -0.0597],
#        dtype=torch.float64)
# com_lin_vel=torch.tensor([[ 2.6147e-07, -5.2145e-03],
#         [-1.7992e-06, -5.3326e-03],
#         [ 1.1589e-05, -1.6195e-03]], dtype=torch.float64)
# p=torch.tensor([[ 5.4271e-10, -6.0890e-06],
#         [-3.0000e-03, -4.3690e-06],
#         [-4.0000e-03, -8.4266e-07],
#         ], dtype=torch.float64)



# com_ang_vel=0.1*torch.tensor([-0.3893, -0.5885],
#        dtype=torch.float64)
# com_lin_vel=0.0*torch.tensor([
#     [ 7.6819e-05, -1.7107e-02],
#     [ 3.0659e-04,  7.1351e-04]],
#     dtype=torch.float64
# )
# p=torch.tensor([
#     [-1.5000e-02, -7.7538e-06],
#     [-1.6000e-02,  1.4549e-05],
#     [-1.6999e-02,  4.8259e-05]],
#     dtype=torch.float64
# )




n=p.shape[0]
p_new=torch.zeros_like(p)
thk=0.001
for i in range(n):
    c=torch.cos(com_ang_vel[i])
    s=torch.sin(com_ang_vel[i])
    R=torch.tensor([[c,-s],[s,c]])
    p_new[i]=com_lin_vel[i]+R@p[i]

ds=torch.zeros((n-1,xy.shape[1]))
us=torch.zeros((n-1,X.shape[0],X.shape[1]))
vs=torch.zeros((n-1,X.shape[0],X.shape[1]))
uv=torch.zeros((n-1,2,xy.shape[1]))
for i in range(n-1):
    e=p[i+1]-p[i]
    p_o=xy-p[i][:,None]
    h=torch.clamp((p_o*e[:,None]).sum(axis=0)/torch.dot(e,e),0.0,1.0)
    pq=p_o-e[:,None]*h
    ds[i]=(torch.linalg.norm(pq,axis=0)-thk)

    c=torch.cos(com_ang_vel[i])
    s=torch.sin(com_ang_vel[i])
    R1=torch.tensor([[c,-s],[s,c]])
    c=torch.cos(com_ang_vel[i+1])
    s=torch.sin(com_ang_vel[i+1])
    R2=torch.tensor([[c,-s],[s,c]])

    line_point=e[:,None]*h+p[i][:,None]
    uv[i]=(
        (1-h)*(com_lin_vel[i][:,None]+R1@line_point)+
        h*(com_lin_vel[i+1][:,None]+R2@line_point)+
        -line_point
    ) / dt

    # uv[i]=(com_lin_vel[i][:,None]+R1@line_point-line_point)/dt

# d=ds.min(axis=0)[0].reshape(nx,ny)

idx=ds.argmin(0).unsqueeze(0).expand(ds.shape)
d=ds.gather(0,idx)[0].reshape(nx,ny)
u=uv[:,0,:].gather(0,idx)[0].reshape(nx,ny)
v=uv[:,1,:].gather(0,idx)[0].reshape(nx,ny)



# indexer=ds.argmin(0).unsqueeze(0).expand(ds.shape)
# from IPython import embed; embed()
# d=ds[0].reshape(nx,ny) #torch.gather(ds,0,indexer)[0]
# u=torch.gather(uv[:,0,:],0,indexer)[0]


# oldpos=e[:,None]*h
# newpos=R@oldpos
# uv=newpos-oldpos
# u=uv[0,0].reshape(nx,ny)
# v=uv[0,1].reshape(nx,ny)

# vec=p_new[-1]-p_new[-2]
# print(torch.angle(torch.complex(newpos[0].reshape(nx,ny),newpos[1].reshape(nx,ny))))



# # ==== plotting ====
# var=d
# plt.figure()
# plt.imshow(
#         var.T,
#         extent = (
#             torch.min(x.cpu()), torch.max(x.cpu()),
#             torch.min(y.cpu()), torch.max(y.cpu())
#         ),
#         origin = "lower",
#         cmap = "Greys"
#     )
# plt.colorbar()
# plt.contour(X,Y,var, colors='k', levels=[0], linestyles='-')
# cset1 = plt.contourf(X,Y, var, levels=20, cmap="Greys")
# plt.plot(p[:,0],p[:,1],'r',marker='o')
# # ==== plotting ====
# p[2]=p[1]+R@(p[2]-p[1])


# ==== plotting ====
var=d
plt.figure()
plt.imshow(
        var.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
plt.colorbar()
plt.contour(X,Y,var, colors='k', levels=[0], linestyles='-')
cset1 = plt.contourf(X,Y, var, levels=20, cmap="Greys")
plt.plot(p[:,0],p[:,1],'r',marker='o')
plt.plot(p_new[:,0],p_new[:,1],'b',marker='o')
subsample_n = 2**5
plt.quiver(
    X[::subsample_n,::subsample_n],
    Y[::subsample_n,::subsample_n],
    u[::subsample_n,::subsample_n],
    v[::subsample_n,::subsample_n],
    color='g',
    scale=1/dt, scale_units='xy'
)

dp2=(p_new-p)/dt
plt.quiver(
    p[:,0],
    p[:,1],
    dp2[:,0],
    dp2[:,1],
    color='r',
    scale=1/dt, scale_units='xy'
)

# plt.savefig("sdf_chain_segments.png")




# ==== plotting ====
var=compute_dpdx(v,dx) #vorticity(u,v,dx,dy)
plt.figure()
plt.imshow(
        var.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
plt.plot(p[:,0],p[:,1],'r',marker='o')
plt.plot(p_new[:,0],p_new[:,1],'b',marker='o')
plt.colorbar()
plt.contour(X,Y,d, colors='k', levels=[0], linestyles='-')
# cset1 = plt.contourf(X,Y, var, levels=20, cmap="Greys")
# plt.plot(p[:,0],p[:,1],'r',marker='o')
# plt.plot(p_new[:,0],p_new[:,1],'g',marker='o')
# subsample_n = 2**4
# plt.quiver(
#     X[::subsample_n,::subsample_n],
#     Y[::subsample_n,::subsample_n],
#     u[::subsample_n,::subsample_n],
#     v[::subsample_n,::subsample_n],
#     color='g',
#     scale=dt, scale_units='xy'
# )

# dp2=p_new[2]-p[2]
# plt.quiver(
#     p[2][0],
#     p[2][1],
#     dp2[0],
#     dp2[1],
#     color='r',
#     scale=dt, scale_units='xy'
# )


# # # plt.savefig("vel_chain_segments.png")



plt.show()









#