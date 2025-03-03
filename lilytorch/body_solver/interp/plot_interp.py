

import torch
import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import matplotlib.pyplot as plt


# Load one tensor
G = torch.load("build/G.zip", weights_only=False).cpu().numpy()
F = torch.load("build/F.zip", weights_only=False).cpu().numpy()
x = torch.load("build/x.zip", weights_only=False).cpu().numpy()
y = torch.load("build/y.zip", weights_only=False).cpu().numpy()
xpt = torch.load("build/xpt.zip", weights_only=False).cpu().numpy()
ypt = torch.load("build/ypt.zip", weights_only=False).cpu().numpy()

print(x[1]-x[0], y[1]-y[0])


[M1,M2]=F.shape
X, Y = np.meshgrid(x, y, indexing="ij")

interp_spline = RegularGridInterpolator((x, y), F, method='linear') #, bounds_error=False, fill_value=None)
G_rgi = interp_spline(np.array([xpt, ypt]).T)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

vmin,vmax = (fun(np.concatenate([F.flatten(),G_rgi])) for fun in (np.min,np.max))

ax[0].contourf(X,Y,F,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[0].set_title("griddata")

ax[1].scatter(xpt,ypt,c=G,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[1].set_title("interpolated")

ax[2].scatter(xpt,ypt,c=G_rgi,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[2].set_title("interpolated")



# x = torch.jit.load("build/x.pt")
# y = torch.jit.load("build/y.pt")


plt.show()

 