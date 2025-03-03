

import torch
from torch.autograd.functional import jacobian
from torch.autograd import grad

N=2**10+1
device = "cpu"
x=torch.linspace(-0.05,0.05,N)
y=torch.linspace(-0.05,0.05,N)
X, Y = torch.meshgrid(x,y,indexing="ij")
X = torch.tensor(X,requires_grad=True)
Y = torch.tensor(Y,requires_grad=True)
external_grad = torch.ones_like(X,requires_grad=True)

sdf_val = torch.where(X>=0.01,X,0)+Y-0.001

# sdf_val = (X**2)*torch.sin(Y)+Y**2-0.001

sdf_val.backward(gradient=external_grad)


# map = lambda x,t : (x[0]**2+y[1]**2)*torch.exp(-t)

# gradx, grady = jacobian(sdf, (x,y))

dx=float(x[1]-x[0])
dy=float(y[1]-y[0])
nx, ny = torch.gradient(sdf_val, spacing=[dx,dy])

print(X.grad)

# print((nx-X.grad).max())
# print((ny-Y.grad).max())



import matplotlib.pyplot as plt
plt.figure()
im = plt.imshow(
        sdf_val.detach().numpy().T, 
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin        = "lower",
        interpolation = "none"
    )
subsample_n = 2**6
quiv_normals= plt.quiver(
    X[::subsample_n,::subsample_n].detach().numpy(),
    Y[::subsample_n,::subsample_n].detach().numpy(),
    X.grad[::subsample_n,::subsample_n].detach().numpy(),
    Y.grad[::subsample_n,::subsample_n].detach().numpy(), 
    color='r'
)

plt.figure()
im = plt.imshow(
        sdf_val.detach().numpy().T, 
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin        = "lower",
        interpolation = "none"
    )
subsample_n = 2**6
quiv_normals= plt.quiver(
    X[::subsample_n,::subsample_n].detach().numpy(),
    Y[::subsample_n,::subsample_n].detach().numpy(),
    nx[::subsample_n,::subsample_n].detach().numpy(),
    ny[::subsample_n,::subsample_n].detach().numpy(), 
    color='r'
)


plt.show()


# # inp = torch.stack((X,Y))
# # # out = (sdf).unsqueeze(0)


# # def exp_adder(x, y):
# #     return 2 * x.exp() + 3 * y
# # inputs = (torch.rand(3), torch.rand(3))
# # jacobian(exp_adder, inputs)

