
import torch
import torch_interpolations
import matplotlib.pyplot as plt

n=2**7
x = torch.linspace(-1, 1, n)
y = torch.linspace(-1, 1, n)
X, Y = torch.meshgrid(x,y)
points_to_interp = (X.flatten(), Y.flatten())

Z = torch.where(torch.abs(X)+torch.abs(Y)<0.5,1,0)
gi = torch_interpolations.RegularGridInterpolator((x,y), Z)


# points_to_interp = [torch.from_numpy(
#     X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]
fx = gi(points_to_interp)



fig, axes = plt.subplots(1, 2)

axes[0].imshow(Z)
axes[0].set_title("True")
axes[1].imshow(fx.reshape(X.shape))
axes[1].set_title("Interpolated")
plt.show()


