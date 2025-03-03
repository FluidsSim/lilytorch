

import torch
import matplotlib.pyplot as plt
import skfmm


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
dt=1


# joint positions
p=torch.tensor([
    [-3.0,0.0],
    [-1.0,0.0],
    [5.0,1.1],
])
n=p.shape[0]


# rotato-translation map in world frames
com_lin_vel=torch.tensor([0,0])
com_ang_vel=torch.tensor([0.1*torch.pi,0.1*torch.pi])


p_new=torch.zeros_like(p)
thk=1
for i in range(n-1):
    c=torch.cos(com_ang_vel[i])
    s=torch.sin(com_ang_vel[i])
    R=torch.tensor([[c,-s],[s,c]])
    p_new[i]=com_lin_vel[i]+R@p[i]

p_new[-1]=com_lin_vel[-1]+R@p[-1]



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
    R=torch.tensor([[c,-s],[s,c]])
    line_point=e[:,None]*h+p[i][:,None]
    uv[i]=(com_lin_vel[i]+R@line_point-line_point)/dt
    # uv[i]=(com_lin_vel[i]+R@(p[i][:,None]+e[:,None]*h*0))/dt

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
plt.plot(p_new[:,0],p_new[:,1],'g',marker='o')
subsample_n = 2**3
plt.quiver(
    X[::subsample_n,::subsample_n],
    Y[::subsample_n,::subsample_n],
    u[::subsample_n,::subsample_n],
    v[::subsample_n,::subsample_n],
    color='g',
    scale=dt, scale_units='xy'
)

dp2=p_new[2]-p[2]
plt.quiver(
    p[2][0],
    p[2][1],
    dp2[0],
    dp2[1],
    color='r',
    scale=dt, scale_units='xy'
)

plt.savefig("sdf_chain_segments.png")



# ==== plotting ====
var=u
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


plt.savefig("vel_chain_segments.png")



plt.show()









#