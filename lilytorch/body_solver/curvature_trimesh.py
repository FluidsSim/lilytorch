

import matplotlib.pyplot as plt
import numpy as np

import trimesh
from trimesh.curvature import (
    discrete_gaussian_curvature_measure,
    discrete_mean_curvature_measure,
    sphere_ball_intersection,
)


mesh = trimesh.creation.icosphere()

radii = np.linspace(0.1, 2.0, 10)
gauss = np.array(
    [
        discrete_gaussian_curvature_measure(mesh, mesh.vertices, r)
        / sphere_ball_intersection(1, r)
        for r in radii
    ]
)
mean = np.array(
    [
        discrete_mean_curvature_measure(mesh, mesh.vertices, r)
        / sphere_ball_intersection(1, r)
        for r in radii
    ]
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces, Z=mesh.vertices[:,2]) 
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.figure()
plt.plot(radii, gauss.mean(axis=1))
plt.title("Gaussian Curvature")

plt.figure()
plt.plot(radii, mean.mean(axis=1))
plt.title("Mean Curvature")
plt.show()