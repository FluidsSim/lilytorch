import torch
import torch.nn.functional as F

# Function to create a Gaussian kernel
def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    x_coord = torch.arange(size)
    x_grid = x_coord.repeat(size).view(size, size)
    y_grid = x_grid.t()#
    
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1./(2.*torch.pi*variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2*variance)
                      )
    
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel

# Parameters for the Gaussian kernel
kernel_size = 5  # Example kernel size
sigma = 1.0  # Standard deviation for Gaussian

# Create the Gaussian kernel
kernel = gaussian_kernel(kernel_size, sigma)

# Reshape the kernel to fit the input data (assuming 1 channel)
kernel = kernel.view(1, 1, kernel_size, kernel_size)

# Example 2D data (e.g., grayscale image)
input_data = torch.randn(1, 1, 100, 100)  # Batch size 1, 1 channel, 100x100 image

# Apply the Gaussian blur using F.conv2d
blurred_data = F.conv2d(input_data, kernel, padding=kernel_size//2)

import matplotlib.pyplot as plt



plt.imshow(blurred_data[0,0])
plt.show()