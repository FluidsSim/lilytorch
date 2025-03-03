
import torch
from matplotlib import pyplot
fig, (ax1, ax2) = pyplot.subplots(1,2,subplot_kw={"projection": "3d"})


n=50
d = torch.linspace(-1, 1, n)
X, Y = torch.meshgrid((d, d))
Z = torch.where(torch.abs(X)+torch.abs(Y)<0.5,1,0)

input = (X-Z*0.3).view(1, 1, n, n).float()

ax1.plot_surface(X, Y, input[0,0], rstride=1, cstride=1,
        linewidth=0, antialiased=False)


# Create grid to upsample input
d = torch.linspace(-1, 1, 50)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0) # add batch dim


output = torch.nn.functional.grid_sample(input, grid, mode='bilinear')

ax2.plot_surface(meshx, meshy, output[0,0], rstride=1, cstride=1,
        linewidth=0, antialiased=False)

pyplot.show()


