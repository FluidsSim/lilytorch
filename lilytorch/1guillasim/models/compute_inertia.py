
import trimesh
import matplotlib.pyplot as plt
import os


path = "lilytorch/1guillasim/models/1guilla_v1/sdf/meshes/"

scale = [1,1,1]
density = 800 # kg/m^3
plotting = False

for i in range(9):
    print("----------")
    print("----------")
    print("loading link {}".format(i))
    # link = trimesh.load(path+"link{}_collision.stl".format(i))
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
    print("link volume: {}".format(link_volume))

    print("\
    					<mass>{}</mass>\
    ".format(
        mass
    ))


    print("\
					<inertia>\n\
						<ixx>{}</ixx>\n\
						<ixy>{}</ixy>\n\
						<ixz>{}</ixz>\n\
						<iyy>{}</iyy>\n\
						<iyz>{}</iyz>\n\
						<izz>{}</izz>\n\
					</inertia>\n\
    ".format(
        inertia[0,0],
        inertia[0,1],
        inertia[0,2],
        inertia[1,1],
        inertia[1,2],
        inertia[2,2]
    )
    )


    if plotting:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(link.vertices[:, 0], link.vertices[:,1], link.vertices[:,2], triangles=link.faces)
        plt.show()
