
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
mesh_file="box.obj"

m2s = mesh2sdf(mesh_file)
m2s.rototranslate_3d(pos=(0,0,0))

dtype = np.float32

min_x=-1
max_x=1
min_y=-1
max_y=1

dx=max_x-min_x
dy=max_y-min_y

# generate data
n=2**15
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
X,Y=np.meshgrid(x,y,indexing="ij")
data = torch.stack((
    torch.from_numpy(x),
    torch.from_numpy(y),
    torch.from_numpy(sdf_val),
),dim=1)
print(data.shape)


# from IPython import embed; embed()

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

# VALID_RATIO = 0.9
# n_train_examples = int(len(data) * VALID_RATIO)
# n_valid_examples = len(data) - n_train_examples
# train_data, test_data = random_split(data, [n_train_examples, n_valid_examples])

n_training = int(0.9*len(data))
train_data = data[:n_training]
test_data = data[n_training:]


class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind][:2]
        y = self.data[ind][2]
        return x, y
    
class TestDataset(TrainDataset):
    def __getitem__(self, ind):
        x = self.data[ind]
        return x

train_set = TrainDataset(train_data)
test_set  = TestDataset(test_data)


batch_size = 2**9
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)



device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 24),
            nn.LogSigmoid(),
            nn.Linear(24, 12),
            nn.LogSigmoid(),
            nn.Linear(12, 6),
            nn.LogSigmoid(),
            nn.Linear(6, 1)
        ).to(device)
        self.input_fc=nn.Linear(2,250)
        self.hidden_fc=nn.Linear(250,100)
        self.output_fc=nn.Linear(100,1)
        self.relu=nn.ReLU()

    def forward(self, x):
        # from IPython import embed; embed()
        # h_1 = self.relu(self.input_fc(x))
        # h_2 = self.relu(self.hidden_fc(h_1))
        # return self.output_fc(h_2)
        return self.model(x)

model = MLP().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # mean square error


epochs = 10


# model training
model.train()
for epoch in range(epochs):
    losses = []
    for batch_num, input_data in enumerate(train_loader):

        optimizer.zero_grad()
        # x = input_data[:,1:]
        # y = input_data[:,0]
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = loss_fn(output, y)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if batch_num % 10 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))

        

        # plt.figure()
        # plot2D(
        #     np.hstack((
        #         x.cpu().numpy(),
        #         output.cpu().detach().numpy()
        #     )).T,
        #     ["x","y","z"]
        # )
        # plt.figure()
        # plot2D(
        #     np.hstack((
        #         x.cpu().numpy(),
        #         np.expand_dims(y.cpu().detach().numpy(),axis=1)
        #     )).T,
        #     ["x","y","z"]
        # )
        # plt.show()

        # print(torch.sum((output[:,0]-y)**2/len(y)))

        # from IPython import embed; embed()


    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))


x=np.linspace(min_x,max_x,2**10,dtype=dtype)
y=np.linspace(min_y,max_y,2**10,dtype=dtype)
data_predict = torch.tensor(list(it.product(x,y)))

# from IPython import embed; embed()

# data_predict = test_loader.dataset.data[:,:2]
output_predict = model(data_predict)
data_predict_plot = np.hstack((
    data_predict.cpu().numpy(),
    output_predict.cpu().detach().numpy()
)).T

plt.figure()
plot2D(
    data_predict_plot,
    ["x","y","z"]
)
plt.show()




# from IPython import embed; embed()






