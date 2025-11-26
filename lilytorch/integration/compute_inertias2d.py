


import numpy as np

# Example SDF: circle of radius 1
def sdf_circle(p, center=(0,0), radius=1.0):
    return np.sqrt((p[...,0] - center[0])**2 + (p[...,1] - center[1])**2) - radius

# Bounding box/region for grid (covers the shape)
xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
N = 1000  # grid resolution
r=1.0

x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
xx, yy = np.meshgrid(x, y)
XY = np.stack([xx, yy], axis=-1)

# Evaluate SDF over grid
inside_mask = sdf_circle(XY, radius=r) < 0

dx = (xmax - xmin) / (N-1)
dy = (ymax - ymin) / (N-1)
dA = dx * dy  # small rectangle area

# Area (mass)
A = np.sum(inside_mask) * dA

# Centroid
x_g = np.sum(xx[inside_mask]) * dA / A
y_g = np.sum(yy[inside_mask]) * dA / A

# Raw moments about the origin
I_x = np.sum((yy[inside_mask]**2) * dA) # around x-axis (horizontal), i.e. y^2 dA
I_y = np.sum((xx[inside_mask]**2) * dA) # around y-axis (vertical), i.e. x^2 dA
I_xy = np.sum((xx[inside_mask]*yy[inside_mask]) * dA)

# Shift to centroid using parallel axis theorem
I_x_centroid = I_x - A * y_g**2
I_y_centroid = I_y - A * x_g**2
I_xy_centroid = I_xy - A * x_g * y_g

print(f"Area: {A:.4f}")
print(f"Centroid: ({x_g:.4f}, {y_g:.4f})")
print(f"I_x about centroid: {I_x_centroid:.4f}")
print(f"I_y about centroid: {I_y_centroid:.4f}")
print(f"I_xy about centroid: {I_xy_centroid:.4f}")


rho=1000
mass = rho * A
mass_real = rho * np.pi * r**2
print(f"Mass (numerical): {mass:.4f}")
print(f"Mass (analytical): {mass_real:.4f}")