
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib
matplotlib.rc('font', **{"size":20})
plt.rcParams["figure.figsize"] = (15,15)

def l2_norm(a, b):
    """Compute the L2 norm of the difference between two arrays."""
    return np.sqrt(np.sum((a - b) ** 2))


def l_infty_norm(a, b):
    """Compute the L-infinity norm of the difference between two arrays."""
    return np.max(np.abs(a - b))

maindir="/data/andreaferrario/ns_data/flow_past_cylinder_error_tests/abdquickest/"

dirs = [ name for name in os.listdir(maindir) if os.path.isdir(os.path.join(maindir, name)) ]

n=len(dirs)
grid_n = np.sort([int(dir) for dir in dirs])


u_best_resolved = np.load(maindir+str(grid_n[-1])+"/uv_field/"+"u.npy")
v_best_resolved = np.load(maindir+str(grid_n[-1])+"/uv_field/"+"v.npy")
p_best_resolved = np.load(maindir+str(grid_n[-1])+"/uv_field/"+"p.npy")


min_grid_n = min(grid_n)
di_best=int(grid_n[-1]/min_grid_n)
errs = np.zeros((n-1,min_grid_n,min_grid_n))
l2_errs_u = np.zeros(n-1)
l_infty_errs_u = np.zeros(n-1)
l2_errs_p = np.zeros(n-1)
l_infty_errs_p = np.zeros(n-1)
for i in range(n):
    dir = str(grid_n[i])
    u = np.load(maindir+dir+"/uv_field/"+"u.npy")
    v = np.load(maindir+dir+"/uv_field/"+"v.npy")
    p = np.load(maindir+dir+"/uv_field/"+"p.npy")
    dx=2/grid_n[i]
    dy=2/grid_n[i]
    w = np.gradient(v, axis=0, edge_order=2)/dx-np.gradient(u, axis=1, edge_order=2)/dy
    plt.figure(1)
    plt.subplot(1,n,i+1)
    plt.imshow(
        w.T,
        extent = (-1, 1, -1, 1),
        origin = "lower",
        cmap = cm.RdBu,
        vmin = -0.2,
        vmax=0.2
    )
    plt.title(f"Grid size: {grid_n[i]}")

    if i<n-1:
        di=int(grid_n[i]/min_grid_n)
        errs[i] = u_best_resolved[1:-1:di_best,1:-1:di_best] - u[1:-1:di,1:-1:di]

        plt.figure(2)
        plt.subplot(1,n-1,i+1)
        plt.imshow(
            errs[i].T,
            extent = (-1, 1, -1, 1),
            origin = "lower",
            cmap = cm.RdBu,
            # vmin = -0.2,
            # vmax=0.2
        )
        plt.title(f"Grid size: {grid_n[i]}")

        l2_errs_u[i] = l2_norm(u_best_resolved[1:-1:di_best,1:-1:di_best], u[1:-1:di,1:-1:di])
        l_infty_errs_u[i] = l_infty_norm(u_best_resolved[1:-1:di_best,1:-1:di_best], u[1:-1:di,1:-1:di])
        l2_errs_p[i] = l2_norm(p_best_resolved[1:-1:di_best,1:-1:di_best], p[1:-1:di,1:-1:di])
        l_infty_errs_p[i] = l_infty_norm(p_best_resolved[1:-1:di_best,1:-1:di_best], p[1:-1:di,1:-1:di])

plt.figure(1)
plt.savefig(maindir+"u_velocity_field_flow_past_cylinder.pdf")

plt.figure(2)
plt.savefig(maindir+"u_velocity_field_error_flow_past_cylinder.pdf")



plt.figure(3)
ms=10
xlim=[10**(-4), 10**(-1)]
x_axis = 1/grid_n[:-1]
plt.subplot(2,2,1)
plt.loglog(x_axis,l2_errs_u,'ko-',markersize=ms)
plt.loglog(xlim,np.float32(xlim)**(2),'r--')
plt.loglog(xlim,np.float32(xlim)**(1),'b--')
plt.xlabel("Grid size")
plt.xlim(xlim)
plt.ylabel("L2 error - u")

plt.subplot(2,2,2)
plt.loglog(x_axis,l_infty_errs_u,'ko-',markersize=ms)
plt.loglog(xlim,np.float32(xlim)**(2),'r--')
plt.loglog(xlim,np.float32(xlim)**(1),'b--')
plt.xlabel("Grid size")
plt.ylabel("L-infty error - u")
plt.xlim(xlim)
plt.tight_layout()

plt.subplot(2,2,3)
plt.loglog(x_axis,l2_errs_p,'go-',markersize=ms)
plt.loglog(xlim,np.float32(xlim)**(2),'r--')
plt.loglog(xlim,np.float32(xlim)**(1),'b--')
plt.xlim(xlim)
plt.xlabel("Grid size")

plt.ylabel("L2 error - p")
plt.subplot(2,2,4)
plt.loglog(x_axis,l_infty_errs_p,'go-',markersize=ms)
plt.loglog(xlim,np.float32(xlim)**(2),'r--')
plt.loglog(xlim,np.float32(xlim)**(1),'b--')
plt.xlabel("Grid size")
plt.ylabel("L-infty error - p")
plt.xlim(xlim)
plt.tight_layout()

plt.savefig(maindir+"flow_past_cylinder_error_analysis.pdf")






