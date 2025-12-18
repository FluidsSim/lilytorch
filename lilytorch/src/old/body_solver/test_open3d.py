import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import itertools as it
from farms_core.io.sdf import ModelSDF
from scipy.spatial.transform import Rotation


class mesh2sdf():
    """
    It is assumed that all vector inputs are numpy arrays
    """
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self._mesh = o3d.io.read_triangle_mesh(mesh_file)

    def translate_3d(self, pos=(0,0,0)):
        self._mesh = self._mesh.translate(np.array(pos))
        self.update_mesh()

    def rototranslate_3d(self, quat=(0,0,0,1), center=(0,0,0), pos=(0,0,0)):
        x, y, z, w = quat # adjust to comply with o3d quaternion
        self._mesh = self._mesh.rotate(o3d.geometry.get_rotation_matrix_from_quaternion((w, x, y, z)),
                                center=np.array(center))
        self._mesh = self._mesh.translate(np.array(pos))
        self.update_mesh()
    
    def update_mesh(self):
        self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh)
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._ = self._raycasting_scene.add_triangles(self._mesht)
        self._mesh.compute_triangle_normals()
        self._face_normals = np.asarray(self._mesh.triangle_normals)
        
    def __call__(self, points_in_object_frame: np.array):

        closest = self._raycasting_scene.compute_closest_points(points_in_object_frame)
        closest_points = closest['points']
        face_ids = closest['primitive_ids']
        pts = closest_points.numpy()
        # negative SDF gradient outside the object and positive SDF gradient inside the object
        gradient = pts - points_in_object_frame

        distance = np.linalg.norm(gradient, axis=-1)
        # normalize gradients
        has_direction = distance > 0
        gradient[has_direction] = gradient[has_direction] / distance[has_direction, None]

        # ensure ray destination is outside the object
        ray_destination = np.repeat(self.bounding_box(padding=1.0)[None, :, 1], points_in_object_frame.shape[0], axis=0)
        # add noise to ray destination, this helps reduce artifacts in the sdf
        ray_destination = ray_destination + 1e-4 * np.random.randn(*points_in_object_frame.shape)
        ray_destination = ray_destination.astype(np.float32)
        # check if point is inside the object
        rays = np.concatenate([points_in_object_frame, ray_destination], axis=-1)
        intersection_counts = self._raycasting_scene.count_intersections(rays).numpy()
        is_inside = intersection_counts % 2 == 1
        distance[is_inside] = distance[is_inside] * -1
        # fix gradient direction to point away from surface outside
        gradient[~is_inside] = gradient[~is_inside] * -1

        # for any points very close to the surface, it is better to use the surface normal as the gradient
        # this is because the closest point on the surface may be noisy when close by
        # e.g. if you are actually on the surface, the closest surface point is itself so you get no gradient info
        on_surface = np.abs(distance) < 1e-3
        surface_normals = self._face_normals[face_ids.numpy()[on_surface]]
        gradient[on_surface] = surface_normals

        return distance, gradient

    def bounding_box(self, padding=0., padding_ratio=0):
        aabb = self._mesh.get_axis_aligned_bounding_box()
        world_min = aabb.get_min_bound()
        world_max = aabb.get_max_bound()
        # already scaled, but we add a little padding
        ranges = np.array(list(zip(world_min, world_max)))
        extents = ranges[:, 1] - ranges[:, 0]
        ranges[:, 0] -= padding + padding_ratio * extents
        ranges[:, 1] += padding + padding_ratio * extents
        return ranges

    def visualize(self):

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(self._mesh)
        # for geometry in geometries:
        #     viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()


class COMPOSITEmesh2sdf():

    def __init__(self, sdf_name, sdf_folder):
        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """
        self.sdf = ModelSDF.read(sdf_folder+sdf_name)[0]
        self.sdfs = []
        for link in self.sdf.links:
            mesh_name = link["visuals"][0]["geometry"]["uri"]
            sdf = mesh2sdf(sdf_folder+mesh_name)
            # initial translation according to the initial poses in the world reference frame (assumes no initial rotation)
            sdf.translate_3d(link.pose[:3])
            self.sdfs.append(sdf)
      
        # for joint in self.sdf.joints:
        #     self.joint_poses.append(joint.pose)

        # from IPython import embed; embed()

    def transform_3d(self, quat_list=[], center_list=[], pos_list=[]):
        for i, (quat, center, pop) in enumerate(zip(quat_list, center_list, pos_list)):
            self.sdf[i].transform_3d()


    def __call__(self, points_in_object_frame: np.array):

        sdfv = []
        sdfg = []
        for i, sdf in enumerate(self.sdfs):
            # B x N for v and B x N x 3 for g
            v, g = sdf(points_in_object_frame)
            # # need to transform the gradient back to the object frame
            # g = self.link_frame_to_obj_frame[i].transform_normals(g)
            sdfv.append(v)
            sdfg.append(g)
        return sdfv, sdfg


    def visualize(self):

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        for sdf in self.sdfs:
            viewer.add_geometry(sdf._mesh)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()




def test_single_mesh():
    # mesh_file="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_1.obj"
    mesh_file="box.obj"

    m2s = mesh2sdf(mesh_file)
    m2s.rototranslate_3d(pos=(0,0,0))

    dtype = np.float32

    # compute sdf on query points
    n=2**10
    x=np.linspace(-1,1,n,dtype=dtype)
    y=np.linspace(-1,1,n,dtype=dtype)
    query_pts=np.array(list(it.product(x,y,[0.0])),dtype=dtype)
    sdf_val, sdf_grad=m2s(query_pts)

    print(sdf_val.shape)
    sdf_val = sdf_val.reshape(len(x), len(y))
    sdf_grad = sdf_grad.reshape(len(x), len(y), 3)

    du = sdf_grad[:,:,0]
    dv = sdf_grad[:,:,1]
    norm = np.sqrt(du**2+dv**2)
    du/=norm
    dv/=norm


    # plotting
    X,Y=np.meshgrid(x,y,indexing='ij')
    plt.figure()
    norm = matplotlib.colors.Normalize(vmin=sdf_val.min(), vmax=sdf_val.max())
    cset1 = plt.contourf(X,Y,sdf_val, cmap="Greys")
    plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
    plt.colorbar(cset1)
    subsample_n = 2**6
    plt.quiver(
        X[::subsample_n,::subsample_n],
        Y[::subsample_n,::subsample_n],
        du[::subsample_n,::subsample_n],
        dv[::subsample_n,::subsample_n], 
        color='g'
    )
    
    m2s.visualize()


    plt.show()



def test_composite_mesh():
    sdf_name = "zebrafish.sdf"
    sdf_folder="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/"

    m2s = COMPOSITEmesh2sdf(sdf_name, sdf_folder)
    m2s.visualize()

    dtype = np.float32

    # compute sdf on query points
    n=2**10
    x=np.linspace(-0.002,0.05,n,dtype=dtype)
    y=np.linspace(-0.002,0.002,n,dtype=dtype)
    query_pts=np.array(list(it.product(x,y,[0.0])),dtype=dtype)
    sdf_vals, sdf_grads=m2s(query_pts)



    # plotting
    X,Y=np.meshgrid(x,y,indexing='ij')
    c=1
    for sdf_val, sdf_grad in zip(sdf_vals, sdf_grads):
        sdf_val = sdf_val.reshape(len(x), len(y))
        sdf_grad = sdf_grad.reshape(len(x), len(y), 3)

        plt.subplot(4,5,c)
        norm = matplotlib.colors.Normalize(vmin=sdf_val.min(), vmax=sdf_val.max())
        cset1 = plt.contourf(X,Y,sdf_val, cmap="Greys")
        plt.title("Link"+str(c-1))
        # cset2 = plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
        plt.colorbar(cset1)
        c=c+1
    plt.show()

if __name__ == "__main__":
    test_single_mesh()
    # test_composite_mesh()