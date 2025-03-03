

import torch
import matplotlib.pyplot as plt
import skfmm

n=1300
x=torch.linspace(-4.5,3.5,n)
y=torch.linspace(-1.5,1.5,n)
X,Y = torch.meshgrid(x,y,indexing='ij')
dx=x[1]-x[0]
dy=y[1]-y[0]

def lineFromPoints(X, P, Q):
    a=Q[1]-P[1]
    b=P[0]-Q[0]
    c=a*(P[0])+b*(P[1])
    if(b>0):
        return (a*X-c)/b
    else:
        return -(a*X-c)/b

def body(X,Y,r1,r2,P,Q,h,b):
    """
    uneven capsule formula
    """
    binary=torch.zeros_like(X)
    binary[X**2+Y**2<r1**2]=-1
    binary[(torch.abs(Y)<=lineFromPoints(X, P, Q))*(r1*b<=X)*(X<=r2*b+h)]=-1
    binary[(X-h)**2+Y**2<r2**2]=-1
    return binary

def body_chain(X,Y,rs,hs):
    nbodies=len(rs)-1
    binaries=torch.zeros(nbodies,X.shape[0],Y.shape[1])
    dh=0
    for i, h in enumerate(hs):
        r1=rs[i]
        r2=rs[i+1]
        b=(r1-r2)/h
        a=torch.sqrt(1.0-b*b)
        P=torch.tensor([r1*b,r1*a])
        Q=torch.tensor([r2*b+h,r2*a])
        binaries[i] = body(X-dh,Y,r1,r2,P,Q,h,b)
        dh+=h
    return binaries

def binaries2sdf(binaries, dx, dy):
    sdfs = torch.zeros(binaries.shape[0],binaries.shape[1],binaries.shape[2])
    for i, binary in enumerate(binaries):
        sdfs[i] = torch.from_numpy(skfmm.distance(convert_minus_plus(binary), dx=[dx,dy]))
    return sdfs

def convert_minus_plus(binary):
    binary_out=torch.ones_like(binary)
    binary_out[binary<0]=-1
    return binary_out

def plot_imshow(out, countour=True):
    plt.figure()
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
        plt.contourf(X,Y, out, levels=30, cmap="Greys")



pos = torch.tensor([0,1,2,3])
rs  = torch.tensor([0.6,0.6,0.5,0.1])
hs  = torch.diff(pos)



binaries = body_chain(X,Y,rs,hs)
sum_binary=convert_minus_plus(binaries.sum(axis=0))
sum_sdf=torch.from_numpy(skfmm.distance(convert_minus_plus(sum_binary), dx=[dx,dy]))
sdfs = binaries2sdf(binaries,dx,dy)
sdf_min = sdfs.min(axis=0)[0]

# all_sdfs =

plot_imshow(sum_binary, countour=False)
plot_imshow(sum_sdf)
plot_imshow(sdf_min)



plt.show()


# d_min = torch.minimum(d1,d2)
# plt.figure()
# plt.imshow(
#         torch.abs(d-d_min).T,
#         extent = (
#             torch.min(x.cpu()), torch.max(x.cpu()),
#             torch.min(y.cpu()), torch.max(y.cpu())
#         ),
#         origin = "lower",
#         cmap = "Greys"
#     )
# plt.colorbar()
# print(torch.abs(d-d_min).max())
# # plt.contour(X,Y,d, colors='k', levels=[0], linestyles='-')
# # cset1 = plt.contourf(X,Y, d, levels=30, cmap="Greys")





# plt.show()


