
import numpy as np
import torch
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

dir = "/data/andreaferrario/ns_data/1guilla_experiments/2025-12-08T06:57:25.488529/"

fig_width = 20
fig_height = 8
curl_max=40

it_end = 150000

def vorticity(u, v, h):

    """
    Compute the vorticity of u,v in 2d - dv/dx-du/dy
    """
    dvdx = torch.zeros_like(u)
    dudy = torch.zeros_like(u)
    dvdx[1:-1, 1:-1] = (v[1:-1, 1:-1]-v[:-2, 1:-1])/h
    dudy[1:-1, 1:-1] = (u[1:-1, 1:-1]-u[1:-1, :-2])/h
    return dvdx-dudy


device="cpu"
dtype=torch.float32


it_spacing = 500
Nx = 2048
Ny = 512
xmin = -0.9
xmax = 5.1
ymin = -0.75
ymax = 0.75
dt = 0.0001
nu = 1e-6
umax = 0.35
Re = umax*0.8/nu

print(f"Reynolds number: {Re}")

dx         = (xmax-xmin)/Nx
dy         = (ymax-ymin)/Ny

assert dx==dy, "dx and dy must be equal to compute vorticity"
h=dx
x=torch.linspace(xmin-dx/2,xmax+dx,Nx+2,device=device,dtype=dtype)
y=torch.linspace(ymin-dy/2,ymax+dy,Ny+2,device=device,dtype=dtype)
X,Y = torch.meshgrid(x,y, indexing="ij")


iterator = range(0,it_end,it_spacing)
npoints = len(iterator)
v_x = torch.zeros(npoints)
v_y = torch.zeros(npoints)

for k, i in enumerate(iterator):

    # load data
    u = torch.from_numpy(np.load(dir+"uv_field/u_"+str(i)+".npy")).to(device=device, dtype=dtype)
    v = torch.from_numpy(np.load(dir+"uv_field/v_"+str(i)+".npy")).to(device=device, dtype=dtype)


    curl = vorticity(u, v, h)
    # curl_plot = curl.clip(-curl_max, curl_max)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(
        curl.cpu().numpy().T,
        extent=[xmin, xmax, ymin, ymax],
        origin='lower',
        cmap=plt.cm.RdBu,
        vmin=-curl_max,
        vmax=curl_max,
        aspect='auto'
    )

    plt.colorbar(label='Vorticity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    for j in range(9):

        body = torch.from_numpy(np.load(dir+f"cnt_field/cnt_{i}_{j}.npy")).to(device=device, dtype=dtype)
        plt.scatter(body[0].cpu(), body[1].cpu(), c="k",s=0.1)

        if j==0: # reference body
            xref = body[0][0]
            yref = body[1][0]

            if i>1:
                v_x[k] = (xref - xold)/(dt*it_spacing)
                v_y[k] = (yref - yold)/(dt*it_spacing)

            xold = xref
            yold = yref






    img_dir = os.path.join(dir, "curl_rendered")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, f"curl_{i}.png"))
    plt.clf()
    plt.close()


plt.figure(figsize=(fig_width, fig_height))
plt.plot(iterator, v_x.cpu().numpy(), label='v_x')
plt.plot(iterator, v_y.cpu().numpy(), label='v_y')
plt.xlabel('Iteration')
plt.ylabel('Velocity')
plt.legend()
plt.title('Reference Point Velocity Over Time')
plt.grid(True)
plt.savefig(os.path.join(dir, "velocity_plot.png"))
plt.close()


#     print(i)


# u = np.load(dir+"uv_field/u_all.npy")





