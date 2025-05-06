
import trimesh
import matplotlib.pyplot as plt
import os

path = "lilytorch/1guillasim/models/1guilla_v1/sdf/meshes/"

scale = [0.001, 0.001, 0.001]
density = 800 # kg/m^3
plotting = True

for i in range(9):
    print("----------")
    print("loading link {}".format(i))
    link = trimesh.load(path+"link{}.obj".format(i))
    link.apply_scale(scale)
    link.density = density # kg/m^3
    link_volume = link.volume
    inertia = link.moment_inertia
    mass = link.mass
    com = link.center_mass

    print("link com: {}".format(com))
    print("link density: {}".format(link.density))
    print("link mass: {}".format(mass))
    print("link inertia: {}".format(inertia))

    if plotting:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(link.vertices[:, 0], link.vertices[:,1], link.vertices[:,2], triangles=link.faces)
        plt.show()
