

import torch
import matplotlib.pyplot as plt
import skfmm

def lineFromPoints(X, P, Q):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a*(P[0]) + b*(P[1])
    if(b>0):
        return (a*X-c)/b
    else:
        return -(a*X-c)/b

def triangle(xy, p0, p1, p2):

    e0=p1-p0
    e1=p2-p1
    e2=p0-p2

    v0=xy-p0[:, None, None]
    v1=xy-p1[:, None, None]
    v2=xy-p2[:, None, None]

    dir0=torch.clamp((e0[0]*v0[0]+e0[1]*v0[1])/torch.dot(e0,e0),0.0,1.0)
    pq0x=v0[0]-e0[0]*dir0
    pq0y=v0[1]-e0[1]*dir0

    dir1=torch.clamp((e1[0]*v1[0]+e1[1]*v1[1])/torch.dot(e1,e1),0.0,1.0)
    pq1x=v1[0]-e1[0]*dir1
    pq1y=v1[1]-e1[1]*dir1

    dir2=torch.clamp((e2[0]*v2[0]+e2[1]*v2[1])/torch.dot(e2,e2),0.0,1.0)
    pq2x=v2[0]-e2[0]*dir2
    pq2y=v2[1]-e2[1]*dir2

    o = e0[0]*e2[1]-e0[1]*e2[0]

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


def two_lines(xy, p0, p1):

    e0=p1-p0
    e1=p2-p1

    v0=xy-p0[:, None, None]
    v1=xy-p1[:, None, None]
    v2=xy-p2[:, None, None]

    dir0=torch.clamp((e0[0]*v0[0]+e0[1]*v0[1])/torch.dot(e0,e0),0.0,1.0)
    pq0x=v0[0]-e0[0]*dir0
    pq0y=v0[1]-e0[1]*dir0

    dir1=torch.clamp((e1[0]*v1[0]+e1[1]*v1[1])/torch.dot(e1,e1),0.0,1.0)
    pq1x=v1[0]-e1[0]*dir1
    pq1y=v1[1]-e1[1]*dir1


    o = 0

    dx = torch.minimum(
        pq0x*pq0x+pq0y*pq0y,
        pq1x*pq1x+pq1y*pq1y
    )


    return -torch.sqrt(dx)



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






n=1300
x=torch.linspace(-4.5,3.5,n)
y=torch.linspace(-1.5,1.5,n)
X,Y = torch.meshgrid(x,y,indexing='ij')
xy = torch.stack((Y,X), axis=0)


p0 = torch.tensor([1,0])
p1 = torch.tensor([0.1,.1])
p2 = torch.tensor([0,0.8])

# d=triangle(xy,p0,p1,p2)
# # d=two_lines(xy,p0,p1)
# plt.figure()
# plt.imshow(
#         d.T,
#         extent = (
#             torch.min(x.cpu()), torch.max(x.cpu()),
#             torch.min(y.cpu()), torch.max(y.cpu())
#         ),
#         origin = "lower",
#         cmap = "Greys"
#     )
# plt.colorbar()
# plt.contour(X,Y,d, colors='k', levels=[0], linestyles='-')
# cset1 = plt.contourf(X,Y, d, levels=30, cmap="Greys")

r1=torch.tensor(0.8)
r2=torch.tensor(0.5)
h=torch.tensor(3.)

d=sdUnevenCapsule(xy,r1,r2,h)
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

theta=torch.linspace(0,2*torch.pi,n)
b=(r1-r2)/h
a=torch.sqrt(1.0-b*b)
c=r1*torch.cos(theta)
s=r2*torch.sin(theta)
plt.plot(r1*torch.cos(theta),r1*torch.sin(theta))
plt.plot(r2*torch.cos(theta)+h,r2*torch.sin(theta))
plt.plot(r1*b,r1*a,'*')
plt.plot(r2*b+h,r2*a,'*')
P=torch.tensor([r1*b,r1*a])
Q=torch.tensor([r2*b+h,r2*a])
plt.plot(x, lineFromPoints(x, P, Q),'b')
cset1 = plt.contourf(X,Y, d, levels=30, cmap="Greys")


dx=x[1]-x[0]
dy=y[1]-y[0]

def body(X,Y):
    binary=-torch.zeros_like(X)
    binary[X**2+Y**2<r1**2]=-1
    binary[(torch.abs(Y)<=lineFromPoints(X, P, Q))*(r1*b<=X)*(X<=r2*b+h)]=-1
    binary[(X-h)**2+Y**2<r2**2]=0
    return binary


r1_tail=r2
r2_tail=0.1
h_tail=2
b_tail=(r1_tail-r2_tail)/h_tail
a_tail=torch.sqrt(1.0-b_tail*b_tail)
P_tail=torch.tensor([r1_tail*b_tail,r1_tail*a_tail])
Q_tail=torch.tensor([r2_tail*b_tail+h_tail,r2_tail*a_tail])

def tail(X,Y):
    binary=torch.zeros_like(X)
    binary[X**2+Y**2<r1_tail**2]=-1
    binary[(torch.abs(Y)<=lineFromPoints(X, P_tail, Q_tail))*(r1_tail*b_tail<=X)*(X<=r2_tail*b_tail+h_tail)]=-1
    binary[(X-h_tail)**2+Y**2<r2_tail**2]=-1
    return binary


fish = body(X+h,Y)+tail(X,Y)
fish[fish==0]=1
plt.figure()
plt.imshow(
        fish.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )

# plt.plot(r1*b,r1*a,'*')
# plt.plot(r2*b+h,r2*a,'*')
# plt.contour(X,Y,d, colors='g', levels=[0], linestyles='-')


d=skfmm.distance(fish, dx=[dx,dy])
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






plt.show()


