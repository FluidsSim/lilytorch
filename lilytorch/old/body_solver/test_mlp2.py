
import torch
import torch.nn as nn

from test_open3d import mesh2sdf
import matplotlib as mpl
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata

import numpy as np
import itertools as it

def plot2D(
    results, 
    labels,
    n_data=300, 
    log=False, 
    cmap=None,
    sequence=None,
    interpolation=None,
    **kwargs
):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    savepath = kwargs.pop('savepath', None)    
    closefig      = kwargs.pop('closefig', True)

    x=results[0]
    y=results[1]
    z=results[2]
    xnew = np.linspace(min(x), max(x), n_data)
    ynew = np.linspace(min(y), max(y), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (x, y), z,
        (grid_x, grid_y),
        method='nearest',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation=interpolation,
        norm=mpl.colors.LogNorm() if log else None
    )

    if cmap is not None:
        imgplot.set_cmap(cmap)
    cbar = plt.colorbar()
    cbar.set_label(labels[2])

    if sequence:
        sequence_interp = griddata(
            (x, y), sequence,
            (grid_x, grid_y),
            method='nearest',  # nearest, cubic
        )
        masked_data = np.ma.masked_where(sequence_interp==0, results_interp)
        plt.imshow(
            masked_data,
            extent=extent,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm() if log else None
        )

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()



# mesh_file="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_1.obj"
# min_x=-0.005
# max_x=0.005
# min_y=-0.005
# max_y=0.005

mesh_file="box.obj"
min_x=-1
max_x=1
min_y=-1
max_y=1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on {} device".format(device))

m2s = mesh2sdf(mesh_file)
m2s.rototranslate_3d(pos=(0,0,0))

dtype = np.float32

dx=max_x-min_x
dy=max_y-min_y

# generate data
n=2**14                                                                                                                                                                              
x=min_x+dx*np.random.rand(n)
y=min_y+dy*np.random.rand(n)
z=np.zeros_like(x)
x=x.astype(dtype)
y=y.astype(dtype)
xyz=np.stack([x,y,z],axis=1)

# x=np.linspace(min_x,max_x,n,dtype=dtype)
# y=np.linspace(min_y,max_y,n,dtype=dtype)
# xy = list(it.product(x,y,[0.0]))
query_pts=np.array(xyz,dtype=dtype)
sdf_val, sdf_grad=m2s(query_pts)


xtorch = torch.from_numpy(x).to(device)
ytorch = torch.from_numpy(y).to(device)
sdf_torch = torch.from_numpy(sdf_val).to(device)


xy_torch = torch.from_numpy( 
    np.stack( 
        (
        x,
        y,
        )
    ).T
).to(device)


data_plot = np.stack( (
        x,
        y,
        sdf_val
        )
    )

plot2D(
    data_plot,
    ["x","y","z"]
)


# ==== polynomial method ====
class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.randn((10)))

    def forward(self, x, y):
        return (
            self.p[0] + 
            self.p[1]*x + self.p[2]*y + 
            self.p[3]*x**2 + self.p[4]*y**2 + self.p[5]*x*y +
            self.p[6]*(x**3) + self.p[7]*(x**2)*y + self.p[8]*x*(y**2) + self.p[9]*(y**3)
        )



# ==== MLP ====
class MLP(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nHidden):
        super(MLP, self).__init__()

        self.nHidden = nHidden
        self.linear1 = nn.Linear(2, self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, 1)
        self.ReLU    = nn.ReLU()
        self.flatten = torch.nn.Flatten(0, 1)

        self.layers = nn.Sequential(
            nn.Linear(2, nHidden),
            nn.ReLU(),

            nn.Linear(nHidden, nHidden),
            nn.ReLU(),

            nn.Linear(nHidden, 1),
            nn.Flatten(0, 1)
            )


    def forward(self, x):
        return self.layers(x)
    
        # h1 = self.ReLU(self.linear1(x))
        # h2 = self.ReLU(self.linear2(h1))
        # h3 = self.ReLU(self.linear3(h2))
        # return self.flatten(self.linear4(h3))




model = MLP(100).to(device)
# model = Polynomial3()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)



for t in range(2000):

    # z_pred = model(xtorch,ytorch)

#         # x = input_data[:,1:]
#         # y = input_data[:,0]
#         x, y = input_data
#         x = x.to(device).float()
#         y = y.to(device)

    z_pred = model(xy_torch)

    # Compute and print loss
    loss = criterion(z_pred,sdf_torch)
    if t % 100 == 99:
        print(t, loss.item(), torch.sum((z_pred-sdf_torch)**2))

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.figure()
plot2D(
    np.stack( (
        x,
        y,
        z_pred.cpu().detach().numpy()
        )
    ),
    ["x","y","z"]
)

plt.show()

