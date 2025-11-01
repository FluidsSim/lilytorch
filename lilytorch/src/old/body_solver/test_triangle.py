

import torch
import matplotlib.pyplot as plt

def plot_segment(A,B):
    plt.plot([A[0],B[0]],[A[1],B[1]],"r")


def triangle(xy,p0,p1,p2):

    e0=p1-p0
    e1=p2-p1
    e2=p0-p2

    v0=xy-p0[:, None, None]
    v1=xy-p1[:, None, None]
    v2=xy-p2[:, None, None]


    d0=torch.clamp((e0[0]*v0[0]+e0[1]*v0[1])/torch.dot(e0,e0),0.0,1.0)
    pq0x=v0[0]-e0[0]*d0
    pq0y=v0[1]-e0[1]*d0

    d1=torch.clamp((e1[0]*v1[0]+e1[1]*v1[1])/torch.dot(e1,e1),0.0,1.0)
    pq1x=v1[0]-e1[0]*d1
    pq1y=v1[1]-e1[1]*d1

    d2=torch.clamp((e2[0]*v2[0]+e2[1]*v2[1])/torch.dot(e2,e2),0.0,1.0)
    pq2x=v2[0]-e2[0]*d2
    pq2y=v2[1]-e2[1]*d2

    o = (e0[0]*e2[1]-e0[1]*e2[0])

    dx = torch.minimum(
            torch.minimum(
                pq0x*pq0x+pq0y*pq0y,
                pq1x*pq1x+pq1y*pq1y
            ),
            pq2x*pq2x+pq2y*pq2y
    )
    dy = torch.minimum(
            torch.minimum(
                o*(v0[0]*e0[1]-v0[1]*e0[0]),
                o*(v1[0]*e1[1]-v1[1]*e1[0])
            ),
            o*(v2[0]*e2[1]-v2[1]*e2[0])
    )


    return -torch.sqrt(dx)*torch.sign(dy)





n=1300
x=torch.linspace(-4.5,3.5,n)
y=torch.linspace(-1.5,1.5,n)
X,Y = torch.meshgrid(x,y,indexing='ij')
xy = torch.stack((X,Y), axis=0)


p0 = torch.tensor([1,0])
p1 = torch.tensor([0.1,.1])
p2 = torch.tensor([0,0.8])

d=triangle(xy,p0,p1,p2)
# d=two_lines(xy,p0,p1)
plt.figure()
plt.imshow(
        d.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
plt.colorbar()
plt.contour(X,Y,d, colors='k', levels=[0], linestyles='-')
cset1 = plt.contourf(X,Y, d, levels=30, cmap="Greys")

plot_segment(p0,p1)
plot_segment(p1,p2)



plt.show()


