
import torch
import matplotlib.pyplot as plt

N=2**10+1
x = torch.linspace(-100,100,N)
dx = x[1]-x[0]

eps = 2*dx

def f(d):
    d_eps = torch.clamp(d,-1,1)
    # d_eps = d/eps
    s=torch.sin(torch.pi*d_eps)
    c=torch.cos(torch.pi*d_eps)
    mu1 = .25*(1-d_eps**2)-0.5*(d_eps*s+(1+c)/torch.pi)/torch.pi
    return mu1

def g(d):
    s=torch.sin(torch.pi*d/eps)
    c=torch.cos(torch.pi*d/eps)
    return torch.where(
        torch.abs(d)>=eps,
        0,
        eps*( 0.25 - (d/(2*eps))**2 - ( d*s/eps+(1+c)/torch.pi )/(2*torch.pi) )
    )


dd = torch.abs(x)-50

# dd = d[:,int(N/2)]

plt.subplot(1,3,1)
plt.plot(x, dd)

plt.subplot(1,3,2)
plt.plot(x, eps*f(dd/eps))

plt.subplot(1,3,3)
plt.plot(x, g(dd)/eps)

plt.show()


