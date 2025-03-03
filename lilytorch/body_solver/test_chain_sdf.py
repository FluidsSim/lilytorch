

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

def check_computed_angles(p,angles):
    p_predicted=torch.zeros_like(p)
    l=torch.linalg.norm(torch.diff(p,axis=0),axis=1)
    p_predicted[0]=p[0]
    p_predicted[1]=p[1]
    for i in range(2,p.shape[0]):
        c=torch.cos(angles[i-2])
        s=torch.sin(angles[i-2])
        R=torch.tensor([[c,-s],[s,c]])
        # from IPython import embed; embed()
        v=l[i-1]*p_predicted[i-1]
        p_predicted[i]=p_predicted[i-1]+R@v

    print(p,p_predicted)
    return True

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



def two_lines(xy, p0, p1, p2):

    e0=p1-p0
    e1=p2-p1

    v0=xy-p0[:, None, None]
    v1=xy-p1[:, None, None]

    dir0=torch.clamp((e0[0]*v0[0]+e0[1]*v0[1])/torch.dot(e0,e0),0.0,1.0)
    pq0x=v0[0]-e0[0]*dir0
    pq0y=v0[1]-e0[1]*dir0

    dir1=torch.clamp((e1[0]*v1[0]+e1[1]*v1[1])/torch.dot(e1,e1),0.0,1.0)
    pq1x=v1[0]-e1[0]*dir1
    pq1y=v1[1]-e1[1]*dir1

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


n=2**8
x=torch.linspace(-4,7,n)
y=torch.linspace(-5,2.5,n)
X,Y = torch.meshgrid(x,y,indexing='ij')
dx=x[1]-x[0]
dy=y[1]-y[0]
xy = torch.stack((X,Y), axis=0)




# p=torch.tensor([
#     [1.0,0.0],
#     [0.1,0.1],
#     [0,0.8],
# ])

# p0 = torch.tensor([1,0])
# p1 = torch.tensor([0.1,.1])
# p2 = torch.tensor([0,0.8])

# d=two_lines(xy,p0,p1,p2)
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
# plt.plot(p[:,0],p[:,1],'r',marker='o')

# plt.show()



p=torch.tensor([
    [-3.0,-1.0],
    [0.0,-1.0],
    [5.0,2.0],
    # [3,1],
    # [5,1]
])
thk=0.5*torch.tensor([1,1,1,0.7,1,1])

# p=torch.tensor([
#     [-3.0,0],
#     [-2,1],
#     [-1,-1],
#     [0,1],
#     [1.0,0.0],
#     [3,1]
# ])
# thk=torch.tensor([0.1,0.5,0.1,0.1,0.1,0.3])


# from IPython import embed; embed()

e=torch.diff(p,axis=0)
# v=xy-p0[:, None, None]
n=p.shape[0]
ds=torch.zeros((n-1,X.shape[0],X.shape[1]))
angles=compute_angles(p)
# assert check_computed_angles(p,angles)


for i in range(n-1):
    e=p[i+1]-p[i]
    v=xy-p[i][:,None,None]
    h=torch.clamp((e[0]*v[0]+e[1]*v[1])/torch.dot(e,e),0.0,1.0)
    pqx=v[0]-e[0]*h
    pqy=v[1]-e[1]*h
    if i==0:
        mask=torch.where((h<1),1,0)
    elif i==n-2:
        mask=torch.where((h>0),1,0)
    else:
        mask=torch.where((h>0)&(h<1),1,0)
    # mask=torch.where((h>0)&(h<1),1,0)
    mask=1

    xy_angle_ip=torch.atan2(xy[1]-p[i][1], xy[0]-p[i][0])
    tmp=torch.where((xy_angle_ip>0)&(xy_angle_ip<3.14),1,0)

    r=(thk[i]+h*(thk[i+1]-thk[i]))*mask
    d=(torch.sqrt(pqx**2+pqy**2))
    ds[i]=d-r
d=ds.min(axis=0)[0]


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


pi2=torch.pi/2

i=0

xpi=X-p[i,0]
ypi=Y-p[i,1]
bi=p[i+1]-p[i]
hi=torch.clamp((xpi*bi[0]+ypi*bi[1])/torch.dot(bi,bi),0.0,1.0)

xpii=X-p[i+1,0]
ypii=Y-p[i+1,1]
bii=p[i+2]-p[i+1]
hii=torch.clamp((xpii*bii[0]+ypii*bii[1])/torch.dot(bii,bii),0.0,1.0)

xy_angle_ip=torch.atan2(xy[1]-p[i+1][1], xy[0]-p[i+1][0])

c1=(xy_angle_ip>-pi2)&(xy_angle_ip<-pi2+angles[i])
r1=torch.where(c1,1,0)

c2=((xy_angle_ip>pi2+angles[i]/2)|(xy_angle_ip<-pi2))&((hi<1))
r2=torch.where(c2,1,0)

c3=((xy_angle_ip>-pi2+angles[i])&(xy_angle_ip<pi2+angles[i]/2))&((hii>0))
r3=torch.where(c3,1,0)
# r3=torch.where((xy_angle_ip>pi2)&(xy_angle_ip<pi2+angles[i]/2)&(hi>0)&(hi<1),1,0)

r=r1+2*r2+3*r3


# ==== plotting ====
var=r
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
cset1 = plt.contourf(X,Y, var, levels=10, cmap="Greys")
plt.plot(p[:,0],p[:,1],'r',marker='o')


r12=(thk[i]+hi*(thk[i+1]-thk[i]))
r23=(thk[i+1]+hii*(thk[i+2]-thk[i+1]))
xy_angle_ip=torch.atan2(xy[1]-p[i+1][1], xy[0]-p[i+1][0])
# ==== plotting ====
d1=torch.where(
    ((xy_angle_ip>pi2+angles[i]/2)|(xy_angle_ip<-pi2+angles[i])),
    torch.sqrt(
        (xpi-hi*bi[0])**2+(ypi-hi*bi[1])**2
    )-r12,
    torch.sqrt(
        (xpii-hii*bii[0])**2+(ypii-hii*bii[1])**2
    )-r23
)

var=d1



# var=torch.sqrt((xy[1]-p[i+1][1])**2+(xy[0]-p[i+1][0])**2)

# var=r12
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
cset1 = plt.contourf(X,Y, var, levels=10, cmap="Greys")
plt.plot(p[:,0],p[:,1],'r',marker='o')



P=torch.tensor([p[0,0],p[0,1]+thk[0]])
Q=torch.tensor([p[1,0],p[1,1]+thk[1]])
plt.plot(x, lineFromPoints(x, P, Q),'b')
plt.plot(P[0],P[1],'g.')
plt.plot(Q[0],Q[1],'g.')

c=torch.cos(angles[0])
s=torch.sin(angles[0])
R=torch.tensor([[c,-s],[s,c]])
P=p[1]+R @ torch.tensor([0,thk[1]])
d12=torch.sqrt((p[2,0]-p[1,0])**2+(p[2,1]-p[1,1])**2)
Q=p[1]+R @ torch.tensor([d12,thk[2]])

plt.plot(x, lineFromPoints(x, P, Q),'b')
plt.plot(P[0],P[1],'y.')
# plt.plot(Q[0],Q[1],'y.')


plt.xlim([x[0],x[-1]])
plt.ylim([y[0],y[-1]])


plt.show()








# from IPython import embed; embed()