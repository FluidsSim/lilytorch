

import torch
import matplotlib.pyplot as plt


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

def sdUnevenCapsule(X, Y, r1, r2, h):
    Yabs=torch.abs(Y)
    b=(r1-r2)/h
    a=torch.sqrt(1.0-b*b)
    k=-b*Yabs+a*X
    return torch.where(
        k<0.0, torch.sqrt(Yabs**2+X**2)-r1,
        torch.where(
            k>a*h, torch.sqrt(Yabs**2+(X-h)**2)-r2,
            a*Yabs+b*X-r1
        )
    )

def plot_imshow(out, countour=True):
    plt.imshow(
        out.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    plt.colorbar()
    if countour:
        plt.contour(X,Y,out, colors='k', levels=[0], linestyles='-')
        plt.contourf(X,Y, out, levels=20, cmap="Greys")



nx=2**8
ny=2**8
x=torch.linspace(-2,6,nx)
y=torch.linspace(-4,1,ny)

X,Y = torch.meshgrid(x,y,indexing='ij')
dx=x[1]-x[0]
dy=y[1]-y[0]
xy = torch.stack((X.flatten(),Y.flatten()), axis=0)

p=torch.tensor([
    [-1.0,-2.0],
    [2.0,-2.0],
])
r=torch.tensor([
    0.2,
    0.2,
])

origin=torch.tensor([0,0])
n=p.shape[0]
hs=torch.linalg.norm(p[1:]-p[:-1],axis=1)
ds=torch.zeros((n-1,X.shape[0],X.shape[1]))

assert torch.all(torch.diff(r)<hs)

pdiff=p[1:,:]-p[:-1,:]
angles=torch.arctan2(pdiff[:,1],pdiff[:,0])
angles_rel=compute_angles(p)
separation_angles=torch.zeros_like(angles_rel)




for i in range(n-1):
    angle=angles[i]
    c=torch.cos(angle)
    s=torch.sin(angle)
    R=torch.tensor([[c,-s],[s,c]])
    print(R)
    newpos=R.T @ (xy-p[i][:,None])
    ds[i]=sdUnevenCapsule(newpos[1].reshape(nx, ny), newpos[0].reshape(nx, ny), r[i], r[i+1], hs[i])
    if i>0:
        separation_angles[i-1]=angles[i-1]+torch.pi/2+angles_rel[i-1]/2


d=ds.min(axis=0)[0]

plt.figure()
plt.plot(p[:,0],p[:,1],'r',marker='o')
max_norm=torch.sqrt((x[-1]-x[0])**2+(y[-1]-y[0])**2)
for i,alpha in enumerate(separation_angles):
    plt.plot(
        [p[i+1,0]-max_norm*torch.cos(alpha),p[i+1,0]+max_norm*torch.cos(alpha)],
        [p[i+1,1]-max_norm*torch.sin(alpha),p[i+1,1]+max_norm*torch.sin(alpha)],
        'b',
    )
plot_imshow(d)
plt.xlim([x[0],x[-1]])
plt.ylim([y[0],y[-1]])
plt.show()




# from IPython import embed; embed()



