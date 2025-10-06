
import numpy as np
from lilytorch.util.rw import load_object
import torch
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
from pytorch_interp import RegularGridInterpolator
import os
import matplotlib.pyplot as plt

dir               = "/data/andreaferrario/ns_data/2025-10-06T13:02:56.840205/"
# dir               = "/data/andreaferrario/ns_data/2025-10-06T11:25:04.426659/"


it_spacing = 50
it_start = 0
it_end = 19950

device="cpu"
dtype=torch.float32

fluid_pars = yaml2pyobject(dir+"parameters.yaml")

N    = fluid_pars["solver"]["N"]
xmin = fluid_pars["solver"]["xmin"]
xmax = fluid_pars["solver"]["xmax"]
ymin = fluid_pars["solver"]["ymin"]
ymax = fluid_pars["solver"]["ymax"]

dx         = (xmax-xmin)/N
dy         = (ymax-ymin)/N
x=torch.linspace(xmin-dx/2,xmax+dx/2,N+2,device=device,dtype=dtype)
y=torch.linspace(ymin-dy/2,ymax+dy/2,N+2,device=device,dtype=dtype)
X,Y = torch.meshgrid(x,y, indexing="ij")

n_bodies = 8

# controller = load_object(dir+"/controller0")
# timestep = controller.pars.timestep

timestep = 0.001

u_interp = RegularGridInterpolator(
    (x,y),
    torch.zeros_like(X, device=device, dtype=dtype),
)
v_interp = RegularGridInterpolator(
    (x,y),
    torch.zeros_like(X, device=device, dtype=dtype),
)

tail_particles = torch.tensor([[],[]], device=device, dtype=dtype)

for i in range(it_start,it_end+1,it_spacing):

    # load data
    u = torch.from_numpy(np.load(dir+"uv_field/u_"+str(i)+".npy")).to(device=device, dtype=dtype)
    v = torch.from_numpy(np.load(dir+"uv_field/v_"+str(i)+".npy")).to(device=device, dtype=dtype)
    cnt = []
    for j in range(n_bodies+1):
        cnt.append(torch.from_numpy(np.load(dir+"cnt_field/cnt_"+str(i)+"_"+str(j)+".npy")).to(device=device, dtype=dtype))

    # tail particles
    tail_idx = 10
    tail_particles = torch.cat((
        tail_particles,
        cnt[-1][:,:tail_idx]
    ), dim=1)

    plt.scatter(tail_particles[0], tail_particles[1], color="yellowgreen", s=3)

    for j in range(n_bodies):
        plt.fill(cnt[j][0].cpu(), cnt[j][1].cpu(), color="#000000")



    u_interp.F = u
    v_interp.F = v

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.gca().set_facecolor('#0033cc')  # Shiny dark blue
    plt.gca().patch.set_alpha(0.9)

    u_interp.F = u
    v_interp.F = v

    u_particles = u_interp(tail_particles[0], tail_particles[1])
    v_particles = v_interp(tail_particles[0], tail_particles[1])

    # update the tail particles based on the computed velocities
    tail_particles[0] += u_particles * timestep * it_spacing
    tail_particles[1] += v_particles * timestep * it_spacing

    img_dir = os.path.join(dir, "particle_images")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, f"particles_{i:05d}.png"))
    plt.clf()



#     print(i)


# u = np.load(dir+"uv_field/u_all.npy")





