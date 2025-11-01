
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib
import torch
import pytorch_volumetric as pv
import numpy as np

mesh_file = "box.obj"

def auto_body(x, y, mesh_file="box.obj"):
    """
    returns:
    n: the normalised normal direction
    """
    obj = pv.MeshObjectFactory(mesh_file)
    sdf = pv.MeshSDF(obj)

    coords=[x,y,torch.tensor([0.0])]
    pts = torch.cartesian_prod(*coords)
    sdf_val, sdf_grad = sdf(pts)

    # reshape val and grad
    sdf_val = sdf_val.reshape(len(x), len(y)).transpose(0, 1)
    sdf_grad = sdf_grad.reshape(len(x), len(y), 3).permute(1, 0, 2)

    # subsample arrows
    du = sdf_grad[:,:, 0]
    dv = sdf_grad[:,:, 1]
    norm = torch.sqrt(du**2+dv**2)
    du/=norm
    dv/=norm

    return coords, sdf_val, du, dv



def plot2D(results, labels, n_data=300, cmap=None):

    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='nearest',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    # plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='lanczos',
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


if __name__ == "__main__":

    cmap="Greys"

    nx      = 30
    ny      = 60
    dx      = 2 / (nx - 1)
    dy      = 2 / (ny - 1)
    x = torch.linspace(-2, 2, nx)
    y = torch.linspace(-2, 2, ny)


    coords, sdf_val, du, dv = auto_body(
        x,
        y
    )

    interior_padding=0.
    norm = matplotlib.colors.Normalize(vmin=sdf_val.min().cpu() - interior_padding, vmax=sdf_val.max().cpu())
    cset1 = plt.contourf(x, y, sdf_val, cmap=cmap)
    cset2 = plt.contour(x, y, sdf_val, colors='k', levels=[0], linestyles='dashed')

    subsample_n = 1
    plt.quiver(
        x[::subsample_n],
        y[::subsample_n],
        du[::subsample_n,::subsample_n],
        dv[::subsample_n,::subsample_n], 
        color='g'
    )

    plt.clabel(cset2, cset2.levels, inline=True, fontsize=13)
    plt.colorbar(cset1)

    print(sdf_val.shape, du.shape)

    plt.show()











