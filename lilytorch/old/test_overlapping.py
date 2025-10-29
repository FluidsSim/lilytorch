


# circle_d_plot.py
import torch
import matplotlib.pyplot as plt


def box(x,y,xb=20,yb=20):
    qx=torch.abs(x)-xb
    qy=torch.abs(y)-yb
    return torch.sqrt(
        torch.maximum(qx,torch.zeros_like(x))**2 +
        torch.maximum(qy,torch.zeros_like(y))**2
    )+torch.minimum(torch.maximum(qx,qy),torch.zeros_like(x))


def segment_sdf(X,Y,A,B,r1,r2):
    pa_x=X-A[0]
    pa_y=Y-A[1]
    ba=B-A
    h=torch.clamp((pa_x*ba[0]+pa_y*ba[1])/torch.dot(ba,ba),0.0,1.0)
    r=(r1+h*(r2-r1))*torch.where((h>0)&(h<1),1,0)
    return torch.sqrt(
        (pa_x-h*ba[0])**2+(pa_y-h*ba[1])**2
    )-r



# domain
nx, ny = 400, 400
x = torch.linspace(-4, 2, nx)
y = torch.linspace(-2, 2, ny)
X, Y = torch.meshgrid(x, y, indexing="ij")

# circle parameters (center and radius)

# d1 = torch.sqrt((X + 1)**2 + (Y)**2) - 1
# d2 = torch.sqrt((X)**2 + (Y)**2) - 1
# d3 = torch.sqrt((X - 1)**2 + (Y)**2) - 1

d1 = box(X+2,Y,1.1,0.1)
d2 = box(X,Y,1.1,0.1)
# d3 = box(X-2,Y,1.3,0.1)

def gamma(a, b):
    return (0.5*(1+((b-a)/torch.sqrt((b)**2 + (a)**2)).clip(-1,1)))


gamma1 = gamma(d1, d2)
# gamma2_1 = gamma(d2, d1)
# gamma2_2 = gamma(d2, d3)
# gamma2 = gamma2_1 * gamma2_2


# plot
plt.figure(figsize=(6, 6))
# filled contour of the d
cont = plt.contourf(X, Y, d1, levels=100, cmap="RdBu_r")
# zero level set (circle boundary)
plt.contour(X, Y, d1, levels=[0.0], colors="k", linewidths=2)
plt.contour(X, Y, d2, levels=[0.0], colors="k", linewidths=2)
# plt.contour(X, Y, d3, levels=[0.0], colors="k", linewidths=2)
plt.colorbar(cont, label="signed distance")
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

fig2 = plt.figure(num="gamma1_plot", figsize=(6, 6))
ax2 = fig2.add_subplot(1, 1, 1)
c = ax2.imshow(gamma1.T, extent=(x.min().item(), x.max().item(), y.min().item(), y.max().item()),
               origin="lower", cmap="viridis", aspect="equal")
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("gamma1")
fig2.colorbar(c, ax=ax2, label="gamma1")
fig2.tight_layout()

# fig3 = plt.figure(num="gamma2_plot", figsize=(6, 6))
# ax3 = fig3.add_subplot(1, 1, 1)
# c2 = ax3.contourf(X, Y, gamma2, levels=100, cmap="viridis")
# ax3.set_aspect("equal", adjustable="box")
# ax3.set_xlabel("x")
# ax3.set_ylabel("y")
# ax3.set_title("gamma2")
# fig3.colorbar(c2, ax=ax3, label="gamma2")
# fig3.tight_layout()

plt.show()


