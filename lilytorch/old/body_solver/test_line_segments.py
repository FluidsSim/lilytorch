

import torch
import matplotlib.pyplot as plt
import skfmm

n=2**10
x=torch.linspace(-4,7,n)
y=torch.linspace(-5,2.5,n)
X,Y = torch.meshgrid(x,y,indexing='ij')
dx=x[1]-x[0]
dy=y[1]-y[0]

def lineFromPoints(X, P, Q):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a*(P[0]) + b*(P[1])
    if(b>0):
        return (a*X-c)/b
    else:
        return -(a*X-c)/b

def segment_sdf(X,Y,A,B,r1,r2):
    pa_x=X-A[0]
    pa_y=Y-A[1]
    ba=B-A
    h=torch.clamp((pa_x*ba[0]+pa_y*ba[1])/torch.dot(ba,ba),0.0,1.0)
    r=(r1+h*(r2-r1))*torch.where((h>0)&(h<1),1,0)
    # d=torch.where((h>0)&(h<1),torch.sqrt(
    #     (pa_x-h*ba[0])**2+(pa_y-h*ba[1])**2
    # ),0)
    d=torch.sqrt(
        (pa_x-h*ba[0])**2+(pa_y-h*ba[1])**2
    )-r

    return d

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
    if countour:
        plt.contour(X,Y,out, colors='k', levels=[0], linestyles='-')
        plt.contourf(X,Y, out, levels=50, cmap="Greys")

def plot_segment(A,B):
    plt.plot([A[0],B[0]],[A[1],B[1]],"r")


V=torch.tensor([
    [-3.0,-1.0],
    [0.0,-1.0],
    [5.0,2.0],
    # [3,1],
    # [5,1]
])
thk=0.5*torch.tensor([0.2,4,0.2,0.7,1,1])
# V=torch.tensor([
#     [-3.0,0],
#     [-2,1],
#     [-1,-1],
#     [0,1],
#     [1.0,0.0],
#     [3,1]
# ])
m=V.shape[0]
sdf=torch.zeros((m-1,n,n))
# thk=0.5*torch.ones(m)
# thk=torch.tensor([0.1,0.5,0.1,0.1,0.1,0.3])

plt.figure()
for i in range(m-1):
    sdf[i]=segment_sdf(X,Y,V[i],V[i+1],thk[i],thk[i+1])
    plot_imshow(sdf[i])
    plot_segment(V[i],V[i+1])




plt.figure()
d = sdf.min(axis=0)[0]
plot_imshow(d)
plt.plot(V[:,0],V[:,1],'r',marker='o')



plt.show()

