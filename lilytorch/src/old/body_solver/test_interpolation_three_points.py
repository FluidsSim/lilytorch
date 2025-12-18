

import torch
import matplotlib.pyplot as plt

x = torch.linspace(-1, 1, 256)
y = torch.linspace(-1, 1, 256)
dx=float(x[1]-x[0])
dy=float(y[1]-y[0])
xx, yy = torch.meshgrid(x, y, indexing='ij')
domain = torch.stack([xx, yy], dim=-1)
# Define u and v velocities in the domain
# Example: vortex flow centered at (0, 0)
u = -yy / (xx**2 + yy**2 + 0.05)-xx
v = xx / (xx**2 + yy**2 + 0.05)


def sdf_circle(points, center=(0.0, 0.0), radius=0.5):
    return torch.linalg.norm(points - torch.tensor(center), dim=-1) - radius

d = sdf_circle(domain)

def barycentric_interpolate(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y):

    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    lambda1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    lambda2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    lambda3 = 1 - lambda1 - lambda2
    return lambda1 * z1 + lambda2 * z2 + lambda3 * z3


solid = (d < 0)
rest = (d >= 0)

identity = torch.ones_like(d)
identity[1:-1, 1:-1] = torch.where(
    d[1:-1,1:-1]>=0,
    torch.where(
        (d[:-2, 1:-1] >= 0) & (d[2:, 1:-1] >= 0) & (d[1:-1, :-2] >= 0) & (d[1:-1, 2:] >= 0),
        1,
        0
    ),
    -1
)

(gradx, grady) = torch.gradient(d, spacing=[dx, dy], edge_order=2)
norm = torch.sqrt(gradx**2+grady**2)
gradx=torch.where(norm>0, gradx/norm, 0)
grady=torch.where(norm>0, grady/norm, 0)

ib_idx = torch.nonzero(identity == 0, as_tuple=False)
gradx_idx = torch.round(torch.sign(gradx[ib_idx[:, 0], ib_idx[:, 1]])).to(torch.int64)
grady_idx = torch.round(torch.sign(grady[ib_idx[:, 0], ib_idx[:, 1]])).to(torch.int64)

nh_idx = torch.stack([ib_idx[:, 0]+gradx_idx, ib_idx[:, 1]], dim=-1)
nh_idx[:, 1] = torch.where(
    identity[nh_idx[:, 0], nh_idx[:, 1]]==0,
    ib_idx[:, 1]+grady_idx,
    ib_idx[:, 1]
)
nv_idx = torch.stack([ib_idx[:, 0], ib_idx[:, 1]+grady_idx], dim=-1)
nv_idx[:, 0] = torch.where(
    identity[nv_idx[:, 0], nv_idx[:, 1]]==0,
    ib_idx[:, 0]+gradx_idx,
    ib_idx[:, 0]
)

xb, yb = x[ib_idx[:, 0]]-gradx[ib_idx[:, 0], ib_idx[:, 1]]*d[ib_idx[:, 0], ib_idx[:, 1]], y[ib_idx[:, 1]]-grady[ib_idx[:, 0], ib_idx[:, 1]]*d[ib_idx[:, 0], ib_idx[:, 1]]
ub = torch.zeros(ib_idx.shape[0])
vb = torch.zeros(ib_idx.shape[0])

u_nh = u[nh_idx[:, 0], nh_idx[:, 1]]
v_nh = v[nh_idx[:, 0], nh_idx[:, 1]]

u_nv = u[nv_idx[:, 0], nv_idx[:, 1]]
v_nv = v[nv_idx[:, 0], nv_idx[:, 1]]


u_ib = barycentric_interpolate(
    xb, yb, ub,
    x[nh_idx[:, 0]], y[nh_idx[:, 1]], u_nh,
    x[nv_idx[:, 0]], y[nv_idx[:, 1]], u_nv,
    x[ib_idx[:, 0]], y[ib_idx[:, 1]]
)

v_ib = barycentric_interpolate(
    xb, yb, vb,
    x[nh_idx[:, 0]], y[nh_idx[:, 1]], v_nh,
    x[nv_idx[:, 0]], y[nv_idx[:, 1]], v_nv,
    x[ib_idx[:, 0]], y[ib_idx[:, 1]]
)



plt.figure(figsize=(6, 6))
plt.imshow(identity.numpy(), extent=[-1, 1, -1, 1], origin='lower', cmap='coolwarm', alpha=0.3, interpolation='none')
plt.colorbar(label='SDF Value')
plt.title('Signed Distance Function of Circle')
plt.xlabel('x')
plt.ylabel('y')
plt.contour(x.numpy(), y.numpy(), d.numpy(), levels=[0], colors='black', linewidths=2, linestyles='--', label='SDF Contour')
plt.legend()

# Select a few example indices (e.g., first 10)
num_examples = min(10, ib_idx.shape[0])
example_indices = torch.randperm(ib_idx.shape[0])[:num_examples]
example_ib = ib_idx[example_indices]
example_nh = nh_idx[example_indices]
example_nv = nv_idx[example_indices]

plt.scatter(x[example_ib[:, 0]].numpy(), y[example_ib[:, 1]].numpy(), color='red', label='IB Points', s=60, marker='o')
plt.scatter(x[example_nh[:, 0]].numpy(), y[example_nh[:, 1]].numpy(), color='blue', label='NH Neighbours', s=40, marker='x')
plt.scatter(x[example_nv[:, 0]].numpy(), y[example_nv[:, 1]].numpy(), color='green', label='NV Neighbours', s=40, marker='^')
plt.scatter(xb[example_indices].numpy(), yb[example_indices].numpy(), color='purple', label='Boundary Points', s=60, marker='s')

plt.legend()


plt.figure(figsize=(6, 6))
plt.imshow(u.numpy().T, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', alpha=0.8, interpolation='none')
plt.colorbar(label='Velocity Magnitude')
plt.contour(x.numpy(), y.numpy(), d.numpy(), levels=[0], colors='black', linewidths=2, linestyles='--')
plt.title('Flow Field with Circle Obstacle')
plt.xlabel('x')
plt.ylabel('y')


plt.figure()
plt.scatter(x[ib_idx[:, 0]].numpy(), y[ib_idx[:, 1]].numpy(), c=u_ib.numpy(), cmap='Reds', label='Interpolated u_ib', s=60)
# plt.scatter(x[ib_idx[:, 0]].numpy(), y[ib_idx[:, 1]].numpy(), c=v_ib.numpy(), cmap='Blues', label='Interpolated v_ib', s=40)
plt.contour(x.numpy(), y.numpy(), d.numpy(), levels=[0], colors='black', linewidths=2, linestyles='--')
plt.legend()



plt.show()


