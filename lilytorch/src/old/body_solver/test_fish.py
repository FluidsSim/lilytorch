
import matplotlib.pyplot as plt
import math
from matplotlib import animation
import torch
import scipy.sparse as sp
import numpy as np

print("Using the CPU.")
device = torch.device("cpu")
torch.set_num_threads(8)


N=2**10+1
x=torch.linspace(-0.003,0.03,N)
y=torch.linspace(-0.01,0.01,N)
x = x.to(device)
y = y.to(device)
dx=float(x[1]-x[0])
dy=float(y[1]-y[0])
X, Y = torch.meshgrid(x,y,indexing="ij")
X = torch.tensor(X,requires_grad=False)
Y = torch.tensor(Y,requires_grad=False)
external_grad = torch.ones_like(X,requires_grad=False)

L=0.02
A=0.01
f=1
xshift=-0.00
yshift=0.00

wh=sb=0.06*L
st=0.9*L
wt=0.06*L

c=0.5

ntimes=30
times=torch.linspace(0,1,ntimes)
dt=float(times[1]-times[0])

def envelope(s):
    """
    width lower in the tail
    """
    return c*s

def thk(s,sb,st,wh,wt,L):
    """
    fish width
    """
    return torch.where(
        s<sb,
        torch.sqrt(2*wh*s-s**2),
        torch.where(
            s<st,
            wh-(wh-wt)*(((s-sb)/(st-sb))**2),
            wt*(L-s)/(L-st)
        )
    )

def fish_sdf(x,y):
    s = torch.clamp(x,0,L)
    sdf = torch.sqrt((x-s)**2+y**2) 
    return sdf-0.001#-thk(s,sb,st,wh,wt,L)

def update(x,y,t):
    xc=x-xshift
    yc=y-yshift
    s = torch.clamp(xc,0-A,L+A)
    return xc, yc+A*torch.sin(2*torch.pi*f*torch.tensor(t)) #A*envelope(s/L)*torch.sin(2*torch.pi*f*(s/L-f*t))

def compute_properties(sdf):

    # sdf.backward(gradient=external_grad)
    # nx = X.grad
    # ny = Y.grad
    # X.grad.zero_()
    # Y.grad.zero_()

    nx, ny = torch.gradient(sdf, spacing=[dx,dy])

    norm = torch.sqrt(nx**2+ny**2)

    numerator = torch.gradient(nx, dim=0, spacing=dx)[0]+torch.gradient(ny, dim=1, spacing=dy)[0]
    denominator = (1+nx**2+ny**2)**2

    curvature = numerator/denominator

    # normalize gradients        
    nx=torch.where(norm>0, nx/norm, 0)
    ny=torch.where(norm>0, ny/norm, 0)

    return sdf, nx, ny, curvature

global ctr, sdf, nx, ny, curvature

XOLD=X.clone().detach()
YOLD=Y.clone().detach()
XNEW, YNEW = update(X,Y,times[0])
sdf, nx, ny, curvature = compute_properties(fish_sdf(XNEW, YNEW))


fig = plt.figure()
plt.contourf(X.detach().numpy(),Y.detach().numpy(), sdf.detach().numpy())
# im = plt.imshow(
#         sdf.detach().numpy().T, 
#         extent = (
#             torch.min(x.cpu()), torch.max(x.cpu()),
#             torch.min(y.cpu()), torch.max(y.cpu())
#         ),
#         origin        = "lower",
#         cmap          = "Greys",
#         interpolation = "none",
#     )
subsample_n = 2**5
quiv_normals= plt.quiver(
    X[::subsample_n,::subsample_n].detach().numpy(),
    Y[::subsample_n,::subsample_n].detach().numpy(),
    nx[::subsample_n,::subsample_n].detach().numpy(),
    ny[::subsample_n,::subsample_n].detach().numpy(), 
    color='g',
    # scale=1, scale_units='xy'
)

ctr = plt.contour(X.detach().numpy(),Y.detach().numpy(), sdf.detach().numpy(), colors='k', levels=[0])

plt.show()

torch.set_printoptions(precision=2)
def animate(i):
    global ctr, sdf, nx, ny, curvature, XOLD, YOLD
    XNEW, YNEW = update(X,Y,times[i])
    U=(XNEW-XOLD)/dt
    V=(YNEW-YOLD)/dt

    dm2dx = torch.gradient(YNEW, spacing=dx, dim=0)[0]

    body_u = -U
    body_v = (-V-dm2dx*body_u) 

    # plotting
    for c in ctr.collections:
        c.remove() 
    ctr = plt.contour(X.detach().numpy(),Y.detach().numpy(), sdf.detach().numpy(), colors='k', levels=[0])
    quiv_normals.set_UVC(body_u[::subsample_n,::subsample_n].detach().numpy(), body_v[::subsample_n,::subsample_n].detach().numpy())
    title = plt.title("Time = {}s".format(round(float(times[i]),1)))
    im.set_array(sdf.detach().numpy().T)

    sdf, nx, ny, curvature = compute_properties(fish_sdf(XNEW, YNEW))
    XOLD=XNEW
    YOLD=YNEW

    return im, ctr, title, quiv_normals

animation = animation.FuncAnimation(fig, animate,
                            frames=100, interval=0, blit=False)
# animation.save('fish.gif', writer='imagemagick', fps=100)
plt.show()



subsample_n = 2**2
plot_every=1
scale=1/(plot_every*dt)
sdf_every = sdf.detach().numpy()
U_EVERY=(XNEW-XOLD)/dt
V_EVERY=(YNEW-YOLD)/dt

for i in range(len(times)):
    XNEW, YNEW = update(X,Y,times[i])
    U=(XNEW-XOLD)/dt
    V=(YNEW-YOLD)/dt
    sdf, nx, ny, curvature = compute_properties(fish_sdf(XNEW, YNEW))
    
    dm1dx, dm1dy = torch.gradient(XNEW, spacing=[dx,dy])
    dm2dx, dm2dy = torch.gradient(YNEW, spacing=[dx,dy])

    # dm1dx_flat = dm1dx.flatten().detach().numpy()
    # dm1dy_flat = dm1dy.flatten().detach().numpy()
    # dm2dx_flat = dm2dx.flatten().detach().numpy()
    # dm2dy_flat = dm2dy.flatten().detach().numpy()
    # MAT = sp.vstack([
    #     sp.hstack([sp.diags(dm1dx_flat),sp.diags(dm1dy_flat)]),
    #     sp.hstack([sp.diags(dm2dx_flat),sp.diags(dm2dy_flat)]),
    # ])
    # rhs = -np.hstack([U.detach().numpy().flatten(),V.detach().numpy().flatten()])
    # sol=sp.linalg.spsolve(MAT,rhs)
    # body_u = sol[:N**2].reshape(N,N)
    # body_v = sol[N**2:].reshape(N,N)

    body_u = -U
    body_v = -V-dm2dx*body_u

    # print((dm2dx*body_u).max()) 

    # from IPython import embed; embed()
    

    if not i % plot_every:
        title = plt.title("Time = {}s".format(round(float(times[i]),1)))
        ctr = plt.contour(X.detach().numpy(),Y.detach().numpy(), sdf.detach().numpy(), colors='k', levels=[0])
       
    if not i % plot_every:
        ctr = plt.contour(X.detach().numpy(),Y.detach().numpy(), sdf_every, colors='r', levels=[0])
        plt.quiver(
            X[::subsample_n,::subsample_n].detach().numpy(),
            Y[::subsample_n,::subsample_n].detach().numpy(),
            # body_u[::subsample_n,::subsample_n],
            # body_v[::subsample_n,::subsample_n], 
            np.where(sdf_every<0,body_u,np.nan)[::subsample_n,::subsample_n],
            np.where(sdf_every<0,body_v,np.nan)[::subsample_n,::subsample_n], 

            # torch.where(sdf_every<0,body_u,0)[::subsample_n,::subsample_n].detach().numpy(),
            # torch.where(sdf_every<0,body_v,0)[::subsample_n,::subsample_n].detach().numpy(), 
            color='g',
            scale=scale, scale_units='xy'
        )
        # plt.scatter(XOLD[::subsample_n,::subsample_n].detach().numpy(),YOLD[::subsample_n,::subsample_n].detach().numpy())
        # plt.scatter(XNEW[::subsample_n,::subsample_n].detach().numpy(),YNEW[::subsample_n,::subsample_n].detach().numpy())
        sdf_every = sdf.detach().numpy()
        U_EVERY = U
        V_EVERY = V
        X_every = XNEW.clone().detach()
        Y_every = YNEW.clone().detach()

        plt.show()
        # from IPython import embed; embed()


    XOLD=XNEW
    YOLD=YNEW

