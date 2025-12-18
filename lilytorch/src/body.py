
import os
import torch
import numpy as np
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error) # exclusevely show errors
from scipy.interpolate import CubicSpline
from farms_core.io.sdf import ModelSDF
from pytorch_interpolation import RegularGridInterpolator
import skfmm
from skimage import measure
import math # important to keep this for evaluating math operations for sdfs even if it appears as not used
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm

from lilytorch.src.scripts.zebrafish_files.load_data import get_experimental_signal

"""
Analitical SDFs
"""
def circle(x,y,xt=0,yt=60,r=25):
    return torch.sqrt((x-xt)**2+(y-yt)**2)-r


def sdUnevenCapsule(Y, X, r1, r2, h, side="L"):
    if side=="L":
        X = -X
    Yabs=torch.abs(Y)
    b=(r1-r2)/h
    a=torch.sqrt(1.0-b*b)
    k=-b*Yabs+a*X
    return torch.where(
        k<0.0, torch.sqrt(Yabs**2+X**2)-r1,
        torch.where(
            k>a*h, torch.sqrt(Yabs**2+(X-h)**2)-r2,
            a*Yabs+b*X-r1
        )
    )


def segment(X,Y,A,B,r1,r2):
    pa_x=X-A[0]
    pa_y=Y-A[1]
    ba=B-A
    h=torch.clamp((pa_x*ba[0]+pa_y*ba[1])/torch.dot(ba,ba),0.0,1.0)
    return torch.sqrt(
        (pa_x-h*ba[0])**2+(pa_y-h*ba[1])**2
    )-(r1+h*(r2-r1))

def box(x,y,xb=20,yb=20):
    qx=torch.abs(x)-xb
    qy=torch.abs(y)-yb
    return torch.sqrt(
        torch.maximum(qx,torch.zeros_like(x))**2 +
        torch.maximum(qy,torch.zeros_like(y))**2
    )+torch.minimum(torch.maximum(qx,qy),torch.zeros_like(x))

def resample_contour_exact_spacing(x, y, spacing, closed=True):
    """
    Resample contour with exactly constant spacing
    """
    if closed:
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

    # Convert to points array
    points = np.column_stack([x, y])
    resampled_points = [points[0]]

    # Walk along contour with exact spacing
    current_position = points[0].copy()
    segment_idx = 0
    distance_along_segment = 0.0

    while segment_idx < len(points) - 1:
        # Current segment
        segment_start = points[segment_idx]
        segment_end = points[segment_idx + 1]
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)

        # Distance remaining in current segment
        remaining_in_segment = segment_length - distance_along_segment

        if remaining_in_segment >= spacing:
            # Place next point within current segment
            if segment_length > 1e-12:  # Avoid division by zero
                direction = segment_vector / segment_length
                current_position = segment_start + (distance_along_segment + spacing) * direction
                resampled_points.append(current_position.copy())
                distance_along_segment += spacing
            else:
                # Degenerate segment, skip
                segment_idx += 1
                distance_along_segment = 0.0
        else:
            # Move to next segment
            segment_idx += 1
            distance_along_segment = spacing - remaining_in_segment

    resampled_points = np.array(resampled_points)

    # Compute arc-length coordinates
    if len(resampled_points) > 1:
        diffs = np.diff(resampled_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        s_uniform = np.concatenate(([0], np.cumsum(distances)))
    else:
        s_uniform = np.array([0])

    return resampled_points[:, 0], resampled_points[:, 1], s_uniform

def resample_contour(x, y, spacing, closed=True):
        # if closed:
        #     if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        s_uniform = np.arange(0, s[-1], spacing)
        x_new = np.interp(s_uniform, s, x)
        y_new = np.interp(s_uniform, s, y)
        x_new = np.r_[x_new, x_new[0]]
        y_new = np.r_[y_new, y_new[0]]

        return x_new, y_new, s_uniform

def compute_inertias_2d(sdf_fun, inside_mask, x, y, x_g, y_g, density=1000.0):
    """
    Compute the inertial properties of a 2D shape defined by an SDF over a grid
    sdf_fun: function that takes (N,M,2) array of points and returns (N,M) array of sdf values
    inside_mask: (N,M) boolean array where True indicates points inside the shape
    x: (N,) array of x coordinates of the grid
    y: (M,) array of y coordinates of the grid
    x_g, y_g: coordinates of the centroid
    density: material density
    """
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    xx, yy = np.meshgrid(x, y)

    dA = dx * dy
    mass = density * np.sum(inside_mask) * dA


    # Raw moments about the origin
    I_x = np.sum((yy[inside_mask]**2) * dA) # around x-axis (horizontal), i.e. y^2 dA
    I_y = np.sum((xx[inside_mask]**2) * dA) # around y-axis (vertical), i.e. x^2 dA
    I_xy = np.sum((xx[inside_mask]*yy[inside_mask]) * dA)

    # Shift to centroid using parallel axis theorem
    I_x_centroid = I_x - mass * y_g**2
    I_y_centroid = I_y - mass * x_g**2
    I_xy_centroid = I_xy - mass * x_g * y_g




def body_from_yaml(device, x, y, body_pars, eps=0.05, costum_update=None, starting_time=0, **kwargs):

    if costum_update is not None:
        update_map = costum_update

    type = body_pars["type"]
    if type == "analytical":
        sdf_fun = eval(body_pars["sdf"])
        plotting=body_pars["plotting"]
        update_map = (
            eval(update_maps["rotation"]),
            (eval(update_maps["translation"][0]),eval(update_maps["translation"][1]))
        )
        return BodyAnalytical(
            device,
            x, y,
            sdf_fun,
            update_map,
            eps=eps,
            plotting=plotting
        )

    elif type == "composite_analytical":
        sdf_funs = body_pars["sdf"]
        plotting=body_pars["plotting"]
        update_maps = body_pars["update_maps"]
        return CompositeBodyAnalytical(
            device, x, y,
            [eval(sdf_fun) for sdf_fun in sdf_funs],
            [
                (
                    eval(update_map["rotation"]),
                    (eval(update_map["translation"][0]),eval(update_map["translation"][1]))
                ) for update_map in update_maps
            ],
            eps=eps,
            plotting=plotting
        )

    elif type == "mesh":
        update_map = [None,None]
        mesh_file = body_pars["mesh_file"]
        (nsamples,msamples) = eval(body_pars["n_samples"])
        return BodyMesh(
            device,
            x, y,
            mesh_file,
            update_map,
            eps=eps,
            plotting_meshes=body_pars["plotting_meshes"],
            compute_interp=body_pars["compute_interp"],
            nsamples=nsamples, msamples=msamples
        )

    elif type == "composite_mesh":
        sdf_name = body_pars["sdf_name"]
        sdf_folder = body_pars["sdf_folder"]
        (nsamples,msamples) = eval(body_pars["n_samples"])
        compute_interp = body_pars["compute_interp"]
        plotting= body_pars["plotting"]
        plotting_meshes = body_pars["plotting_meshes"]
        return CompositeBodyMesh(
            device, x, y,
            sdf_folder, sdf_name,
            costum_update,
            eps             = eps,
            compute_interp  = compute_interp,
            nsamples        = nsamples,
            msamples        = msamples,
            plotting        = plotting,
            plotting_meshes = plotting_meshes,
            suit            = body_pars["suit"],
            convexify       = body_pars["convexify"],
            scale           = body_pars["scale"],
            **kwargs
        )

    elif type == "multi_animat":

        (nsamples,msamples) = body_pars["n_samples"]

        return MultiAnimatBodies(
            device, x, y,
            experiment_options = body_pars["experiment_options"],
            eps                = eps,
            compute_interp     = body_pars["compute_interp"],
            nsamples           = nsamples,
            msamples           = msamples,
            plotting           = body_pars["plotting"],
            plotting_meshes    = body_pars["plotting_meshes"],
            suit               = body_pars["suit"],
            convexify          = body_pars["convexify"],
            scale              = body_pars["scale"],
            save_folder        = body_pars["save_folder"],
            **kwargs
        )


    elif type == "fish_analytical":
        control_pars = body_pars["control"]
        return BodyFishAnalytical(
            device, x, y,
            eps=eps,
            L=control_pars["L"], A=control_pars["A"], f=control_pars["f"],
            wavefrequency=control_pars["wavefrequency"],
            c1=control_pars["c1"], c2=control_pars["c2"], c3=control_pars["c3"],
            xshift=control_pars["xshift"], yshift=control_pars["yshift"],
            sb=control_pars["sb"], wh=control_pars["wh"], st=control_pars["st"], wt=control_pars["wt"], thk=control_pars["thk"]
        )

    elif type == "fish_experimental":
        control_pars = body_pars["control"]
        return BodyFishExperimental(
            device, x, y,
            eps             = eps,
            body_length     = control_pars["body_length"],
            folder_name     = control_pars["folder_name"],
            file_name       = control_pars["file_name"],
            save_data       = control_pars["save_data"],
            plot_data       = control_pars["plot_data"],
            target_fish     = control_pars["target_fish"],
            start_recording = control_pars["start_recording"],
            end_recording   = control_pars["end_recording"],
            timestep        = control_pars["timestep"],
            total_duration  = control_pars["total_duration"],
            freq_scaling    = control_pars["freq_scaling"],
            filter_freqs    = control_pars["filter_freqs"],
            xshift          = control_pars["xshift"],
            yshift          = control_pars["yshift"],
            initial_time    = starting_time
        )

    elif type == "composite_segment_body":
        sdf_name = body_pars["sdf_name"]
        sdf_folder = body_pars["sdf_folder"]
        return CompositeSegmentBody(
                    device, x, y,
                    sdf_folder, sdf_name,
                    eps=eps
                )

class mesh2sdf():
    """
    It is assumed that all vector inputs are numpy arrays
    """
    def __init__(self, mesh_file, convexify=True, scale=1):
        self.mesh_file = mesh_file
        self._mesh = o3d.io.read_triangle_mesh(self.mesh_file)
        self.update_mesh(convexify=convexify, scale=scale)

    def update_mesh(self, convexify, scale):
        self._mesh = self._mesh.scale(scale, (0,0,0)) #self._mesh.get_center())
        if convexify:
            self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh.compute_convex_hull()[0])
        else:
            self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh)

        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._ = self._raycasting_scene.add_triangles(self._mesht)
        self._mesh.compute_triangle_normals()
        self._face_normals = np.asarray(self._mesh.triangle_normals)

    def __call__(self, points_in_object_frame: np.array):



        self._raycasting_scene.compute_signed_distance(points_in_object_frame)

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
        ray_destination = np.repeat(self.bounding_box(padding=0.0)[None, :, 1], points_in_object_frame.shape[0], axis=0)
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

    def visualize(self, wireframe=True):

        print("Visualizing the mesh file: {}".format(self.mesh_file))

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        if wireframe:
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(self._mesh)
            viewer.add_geometry(line_set)
        else:
            viewer.add_geometry(self._mesh)
        opt = viewer.get_render_option()
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
            # sdf.translate_3d(link.pose[:3])
            self.sdfs.append(sdf)

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
            viewer.add_geometry(o3d.geometry.LineSet.create_from_triangle_mesh(sdf._mesh))
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()

class Body:

    def __init__(self, device, x, y, eps=0.05):
        """

        """
        self.device=device
        self.dtype = x.dtype

        self.x   = x
        self.y   = y
        self.h = float(x[1]-x[0])

        self.X, self.Y = torch.meshgrid(x,y,indexing="ij")
        self.x_stag = self.x-self.h/2
        self.y_stag = self.y-self.h/2
        [self.Xu_stag, self.Yu_stag] = torch.meshgrid(self.x_stag, self.y, indexing="ij")
        [self.Xv_stag, self.Yv_stag] = torch.meshgrid(self.x, self.y_stag, indexing="ij")

        self.nx  = len(x)
        self.ny  = len(y)
        self.eps = eps
        self.dtype = x.dtype

        self.xflat = self.X.flatten()
        self.yflat = self.Y.flatten()

        self.xu_stag_flat = self.Xu_stag.flatten()
        self.yu_stag_flat = self.Yu_stag.flatten()
        self.xv_stag_flat = self.Xv_stag.flatten()
        self.yv_stag_flat = self.Yv_stag.flatten()

        self.stacked_xy = torch.stack((self.xflat,self.yflat))
        self.stacked_xy_u = torch.stack((self.xu_stag_flat,self.yu_stag_flat))
        self.stacked_xy_v = torch.stack((self.xv_stag_flat,self.yv_stag_flat))


        self.ones_stacked=torch.ones((self.nx*self.ny),device=self.device,dtype=self.dtype)

        # body velocities
        self.sdf = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.body_u = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.body_v = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.old_points = self.stacked_xy.clone().detach()
        self.rad_conv = (torch.pi/180)


    def compute_sdf_properties(self, sdf_val):

        (gradx, grady) = torch.gradient(sdf_val, spacing=[self.h, self.h], edge_order=2)
        norm = torch.sqrt(gradx**2+grady**2)

        # curvature=torch.where(
        #     norm>0,
        #     (torch.gradient(gradx, spacing=self.h, axis=0, edge_order=2)[0]*grady-
        #      torch.gradient(grady, spacing=self.h, axis=1, edge_order=2)[0]*gradx)/
        #     norm**3,
        #     0
        # )

        # curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5


        # compute curvature
        numerator = (
            (grady**2)*torch.gradient(gradx, spacing=self.h, axis=0, edge_order=2)[0]+
            (gradx**2)*torch.gradient(grady, spacing=self.h, axis=1, edge_order=2)[0]+
            -2*gradx*grady*torch.gradient(grady, spacing=self.h, axis=0)[0]
        )
        denominator = norm**3
        curvature = torch.where(denominator>0, numerator/denominator, 0)



        # # compute curvature
        # numerator = (
        #     (grady**2)*torch.gradient(gradx, spacing=self.h, axis=0)[0]+
        #     (gradx**2)*torch.gradient(grady, spacing=self.h, axis=1)[0]+
        #     -2*gradx*grady*torch.gradient(grady, spacing=self.h, axis=0)[0]
        # )
        # denominator = norm**3
        # curvature = torch.where(denominator>0, numerator/denominator, 0)


        # dx_dt   = np.gradient(com_x)
        # dy_dt   = np.gradient(com_y)
        # d2x_dt2 = np.gradient(dx_dt)
        # d2y_dt2 = np.gradient(dy_dt)
        # curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5


        # numerator = torch.gradient(gradx, dim=0, spacing=self.h)[0]+torch.gradient(grady, dim=1, spacing=self.h)[0]
        # denominator = (gradx**2+grady**2)**1.5

        # normalize gradients
        gradx=torch.where(norm>0, gradx/norm, 0)
        grady=torch.where(norm>0, grady/norm, 0)


        return (
            sdf_val,
            gradx,
            grady,
            curvature,
        )

    def phi(self,d):
        # return 0.5+0.5*torch.cos(torch.pi*d.clamp(-1,1))
        return torch.where(
            torch.abs(d)<self.eps,
            ( 1 + torch.cos(torch.pi*d/self.eps) )/( 2*self.eps ),
            0
        )


    def mu_funcs(self, d):
        deps=d/self.eps
        s=torch.sin(torch.pi*deps)
        c=torch.cos(torch.pi*deps)
        mu_0_eps = torch.where(
            d<=-self.eps,
            0,
            torch.where(
                d>=self.eps,
                1,
                0.5*( 1 + deps + s/torch.pi )
            )
        )
        mu_1_eps = torch.where(
            torch.abs(d)>=self.eps,
            0,
            self.eps*( 0.25 - (0.5*deps)**2 - ( s*deps+(1+c)/torch.pi )/(2*torch.pi) )
        )
        return (mu_0_eps, mu_1_eps)





class BodyAnalytical(Body):

    def __init__(self, device, x, y, sdf, update_maps, eps=0.05, plotting=False, pre_update=True):
        super().__init__(device, x, y, eps=eps)
        self.sdf = sdf
        self.update_theta = update_maps[0]
        self.update_translation = update_maps[1]
        self.plotting = plotting
        self.body=self
        self.pre_update = pre_update
        self.initialize()
        self.rad_conv = (torch.pi/180)

    def initialize(self):
        """
        Initialize sdf properties at time 0
        """

        ####### initial sdf at cc nodes to compute contour
        xmid=(self.x.min()+self.x.max())/2
        ymid=(self.y.min()+self.y.max())/2
        xcnt = self.x-xmid
        ycnt = self.y-ymid

        X,Y= torch.meshgrid(xcnt, ycnt,indexing="ij")
        sdf_cnt = self.sdf(X, Y)

        # pos_u = (self.stacked_xy[0]).reshape(self.nx, self.ny)
        # pos_v = (self.stacked_xy[1]).reshape(self.nx, self.ny)
        # self.sdf = self.sdf(pos_u, pos_v)

        # # compute sdf at init
        # (trans, rot) = self.rototranslate_points(torch.tensor(0.0))
        # translpoints=self.stacked_xy-trans
        # newpoints_u=rot.T@translpoints
        # newpos_u = newpoints_u[0].reshape(self.nx, self.ny)
        # newpos_v = newpoints_u[1].reshape(self.nx, self.ny)
        # self.sdf_val = self.sdf(newpos_u, newpos_v)

        # compute sdf at location (0,0)

        # find contour lines
        sdf_np=sdf_cnt.cpu().numpy()
        xnp = xcnt.cpu().numpy()
        ynp = ycnt.cpu().numpy()

        # cnt = np.array(measure.find_contours(sdf_np, 0)[0]).T
        cnt = np.array(measure.find_contours(sdf_np-self.h, 0)[0]).T
        cnt[0]=xnp[0]+cnt[0]*(xnp[1]-xnp[0])
        cnt[1]=ynp[0]+cnt[1]*(ynp[1]-ynp[0])

        curv_coord = np.cumsum(np.sqrt(np.sum(np.diff(cnt, axis=1)**2, axis=0)))

        # # resample contour lines for uniform spacing with spacing self.h
        ds=self.h #0.5*torch.sqrt(torch.tensor(self.h**2+self.h**2))
        # x, y, s_uniform = self.resample_contour(cnt[0], cnt[1], spacing=ds, closed=True)
        x, y, s_uniform = resample_contour(cnt[0], cnt[1], spacing=ds, closed=True)
        del cnt
        cnt=np.array([x, y])

        # Compute ds and cumulative s
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        curv_coord = np.concatenate(([0], np.cumsum(ds)))
        # curv_coord = np.cumsum(np.sqrt(np.sum(np.diff(cnt, axis=1)**2, axis=0)))


        # curv_coord = s_uniform

        self.curv_coord = torch.from_numpy(curv_coord).type(self.dtype).to(self.device)
        self.cnt        = torch.from_numpy(cnt).type(self.dtype).to(self.device)
        self.cnt_update = self.cnt.clone().detach()
        self.ds = self.curv_coord[1]-self.curv_coord[0]

        if self.plotting:

            plt.imshow(
                sdf_np.T,
                extent=(
                    torch.min(self.x.cpu()), torch.max(self.x.cpu()),
                    torch.min(self.y.cpu()), torch.max(self.y.cpu())
                ),
                origin="lower",
                cmap="Greys"
            )
            plt.colorbar()

            # Plot cnt as scatter with color given by a colormap
            cmap = cm.get_cmap('RdBu')
            n_points = self.cnt_update.shape[1]
            colors = cmap(np.linspace(0, 1, n_points))
            plt.plot(self.cnt_update[0].cpu(), self.cnt_update[1].cpu())
            plt.show()


        self.cnt_u=torch.zeros_like(self.cnt_update[0])
        self.cnt_v=torch.zeros_like(self.cnt_update[1])

        self.cnt_f_u=torch.zeros_like(self.cnt_update[0])
        self.cnt_f_v=torch.zeros_like(self.cnt_update[1])
        self.cnt_int_f_u=torch.zeros_like(self.cnt_update[0])
        self.cnt_int_f_v=torch.zeros_like(self.cnt_update[1])
        self.mask = torch.arange(len(self.curv_coord), device=self.device)
        self.com_pos = torch.zeros((2), device=self.device, dtype=self.dtype)
        if self.pre_update:
            self.update(torch.tensor(0.0),0, update_cnt=False)


        return


    def rototranslate_points(self, t):
        """
        Apply rototranslation and update the sdf properties
        Assumes that the rotations happen around the origin of the reference frame (i.e. the center of rotation is (0,0))
        This simply means that com=[transl[0], transl[1]]
        """

        transl = torch.tensor([
            self.update_translation[0](t),
            self.update_translation[1](t)
        ], device=self.device, dtype=self.dtype)

        theta = self.rad_conv*(
            torch.tensor(
                self.update_theta(t),
                device=self.device, dtype=self.dtype
            )
        )

        self.com_pos = transl

        s = torch.sin(theta)
        c = torch.cos(theta)
        rot = torch.stack([torch.stack([c, -s]),
                        torch.stack([s, c])])
        trans = torch.stack((transl[0]*self.ones_stacked, transl[1]*self.ones_stacked))

        return (trans, rot)



    def update(self, t, iteration, dt=1, update_cnt=True):


        (trans, rot) = self.rototranslate_points(t)

        # compute linear and angular velocities using automatic differentiation
        t_var = t.clone().detach().requires_grad_(True)
        vx = self.update_translation[0](t_var)
        vy = self.update_translation[1](t_var)
        w = self.update_theta(t_var) * self.rad_conv

        lin_vel_x = torch.autograd.grad(vx, t_var, create_graph=False)[0]
        lin_vel_y = torch.autograd.grad(vy, t_var, create_graph=False)[0]
        ang_vel   = torch.autograd.grad(w, t_var, create_graph=False)[0]

        # compute sdf at cc locations
        translpoints=self.stacked_xy-trans
        newpoints_u=rot.T@translpoints
        newpos_u = newpoints_u[0].reshape(self.nx, self.ny)
        newpos_v = newpoints_u[1].reshape(self.nx, self.ny)
        self.sdf_val = self.sdf(newpos_u, newpos_v)

        # compute sdf at staggered grid locations (u points -sdf_u and v points-sdf_v)
        translpoints_u=self.stacked_xy_u-trans
        newpoints_u=rot.T@translpoints_u
        newpos_u = newpoints_u[0].reshape(self.nx, self.ny)
        newpos_v = newpoints_u[1].reshape(self.nx, self.ny)
        self.sdf_u = self.sdf(newpos_u, newpos_v)

        translpoints_v=self.stacked_xy_v-trans
        newpoints_v=rot.T@translpoints_v
        newpos_u = newpoints_v[0].reshape(self.nx, self.ny)
        newpos_v = newpoints_v[1].reshape(self.nx, self.ny)
        self.sdf_v = self.sdf(newpos_u, newpos_v)

        # update body velocities (need to be staggered)
        self.body_u = (lin_vel_x - ang_vel*translpoints_u[1]).reshape(self.nx, self.ny)
        self.body_v = (lin_vel_y + ang_vel*translpoints_v[0]).reshape(self.nx, self.ny)

        if update_cnt==True:

            # update contour points and velocities
            self.cnt_update = rot @ self.cnt
            self.cnt_update[0]+=self.com_pos[0]
            self.cnt_update[1]+=self.com_pos[1]
            self.cnt_u=(lin_vel_x-ang_vel*(self.cnt_update[1]-self.com_pos[1]))
            self.cnt_v=(lin_vel_y+ang_vel*(self.cnt_update[0]-self.com_pos[0]))




class CompositeBodyAnalytical(Body):

    def __init__(self, device, x, y, sdf_funs, update_maps, plotting=False, **kwargs):
        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """
        super().__init__(device, x, y, **kwargs)
        self.nbodies = len(sdf_funs)
        assert self.nbodies == len(update_maps), "Number of sdf functions and update maps must be the same"

        self.bodies=[
            BodyAnalytical(
                device, x, y,
                sdf_funs[i],
                update_maps[i],
                plotting=plotting,
                **kwargs
            ) for i in range(self.nbodies)
        ]

        self.mu_funcs = self.bodies[0].mu_funcs
        self.sdf_vals = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.sdf_vals_u = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.sdf_vals_v = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.u_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.v_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.com_pos  = torch.zeros((self.nbodies,2),device=device)
        self.initialize()

    def initialize(self):
        """
        Initialize sdf properties at time 0
        """
        self.update(torch.tensor(0.0,device=self.device,dtype=self.dtype), 0)


    def update(self, t, iteration, dt=1):
        for i, body in enumerate(self.bodies):
            body.update(t, iteration, dt=dt)
            self.sdf_vals[i]   = body.sdf_val
            self.sdf_vals_u[i] = body.sdf_u
            self.sdf_vals_v[i] = body.sdf_v
            self.u_vals[i]   = body.body_u
            self.v_vals[i]   = body.body_v

        self.sdf_val = torch.min(self.sdf_vals,axis=0)[0]
        idx=self.sdf_vals.argmin(0).unsqueeze(0).expand(self.sdf_vals.shape)
        self.sdf_val=self.sdf_vals.gather(0,idx)[0].reshape(self.nx,self.ny)

        self.sdf_val_u = torch.min(self.sdf_vals_u,axis=0)[0]
        idx=self.sdf_vals_u.argmin(0).unsqueeze(0).expand(self.sdf_vals_u.shape)
        self.sdf_val_u=self.sdf_vals_u.gather(0,idx)[0].reshape(self.nx,self.ny)
        self.body_u =self.u_vals.gather(0,idx)[0].reshape(self.nx,self.ny)

        self.sdf_val_v = torch.min(self.sdf_vals_v,axis=0)[0]
        idx=self.sdf_vals_v.argmin(0).unsqueeze(0).expand(self.sdf_vals_v.shape)
        self.sdf_val_v=self.sdf_vals_v.gather(0,idx)[0].reshape(self.nx,self.ny)
        self.body_v =self.v_vals.gather(0,idx)[0].reshape(self.nx,self.ny)


class BodyFishAnalytical(Body):

    def __init__(
        self,
        device,
        x,
        y,
        L             = 0.015,
        A             = 0.015,
        f             = 3,
        wavefrequency = 0.95,
        c1            = +0.05,
        c2            = -0.13,
        c3            = +0.28,
        xshift        = 0.0,
        yshift        = 0.0,
        eps           = 0.05,
        sb            = 0.07,
        wh            = 0.07,
        st            = 0.95,
        wt            = 0.01,
        thk           = False

    ):
        super().__init__(device, x, y, eps=eps)
        """

        """
        self.L=L
        self.A=A
        self.f=f
        self.wavefrequency=wavefrequency
        self.XC=self.X-xshift
        self.YC=self.Y-yshift

        self.wh=wh*L
        self.sb=sb*L
        self.st=st*L
        self.wt=wt*L

        # OLD ENVELOPE (Di Santo et al. 2021 - All Fishes)
        self.c1=c1
        self.c2=c2
        self.c3=c3

        # NEW ENVELOPE (Di Santo et al. 2021 - Danio Rerio)
        self.p0 = 44.5 / 110.0
        self.p1 = self.p0 + 22.0 / 110.0
        self.p2 = 1.0

        self.a0 = 0.04
        self.a1 = 0.16
        self.a2 = 0.24

        self.s1 = (self.a1 - self.a0) / (self.p1 - self.p0)
        self.s2 = (self.a2 - self.a1) / (self.p2 - self.p1)

        self.bodies = [self]
        if thk:
            self.thk = lambda s: thk
        else:
            self.thk = self.thk_nonconst

        self.oldpos_u = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.oldpos_v = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)

        self.initialize()

    def envelope(self, s):
        """
        width lower in the tail
        """

        # NEW ENVELOPE
        # return torch.where(
        #     s < self.p0,
        #     self.a0,
        #     torch.where(
        #         s < self.p1,
        #         self.a0 + self.s1 * (s - self.p0),
        #         self.a1 + self.s2 * (s - self.p1),
        #     )
        # )

        # OLD ENVELOPE
        return self.c1+self.c2*s+self.c3*s**2

    def thk_nonconst(self,s):
        """
        fish width
        """
        return torch.where(
            s<self.sb,
            torch.sqrt(2*self.wh*s-s**2),
            torch.where(
                s<self.st,
                self.wh-(self.wh-self.wt)*(((s-self.sb)/(self.st-self.sb))**2),
                self.wt*(self.L-s)/(self.L-self.st)
            )
        )

    def sdf_fun(self, x,y):
        s = x.clamp(0,self.L)
        sdf = torch.sqrt((x-s)**2+y**2)
        return sdf-self.thk(s)

    def update(self, t, iteration, dt=1):
        """
        Update sdf properties from analytical rototranslation map
        """
        s = self.XC.clamp(0,self.L)
        new_x = self.XC
        new_y = self.YC+self.A*self.envelope(s/self.L)*torch.sin(2*torch.pi*(self.wavefrequency*s/self.L-self.f*t))

        self.body_u=-(new_x-self.oldpos_u)/dt
        self.body_v=-(new_y-self.oldpos_v)/dt

        self.oldpos_u=new_x
        self.oldpos_v=new_y

        self.sdf_val=self.sdf_fun(new_x,new_y)

        self.sdf_vals=[self.sdf_fun(new_x,new_y)]

        # return [self.compute_sdf_properties(self.sdf_fun(new_x,new_y))]

    def initialize(self):
        """
        Initialize sdf properties at time 0
        """
        self.cnt        = torch.zeros((2,1),device=self.device,dtype=self.dtype)
        self.cnt_update = self.cnt.clone().detach()
        self.curv_coord = torch.tensor([0,1],device=self.device,dtype=self.dtype)
        self.com_pos    = torch.tensor([[0,0]],device=self.device,dtype=self.dtype)
        self.update(0,0)

        self.body_u=torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.body_v=torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)

class BodyFishExperimental(Body):

    def __init__(
        self,
        device,
        x,
        y,
        body_length,
        folder_name,
        file_name,
        save_data,
        plot_data,
        target_fish,
        start_recording,
        end_recording,
        timestep,
        total_duration,
        freq_scaling,
        filter_freqs,
        xshift       = -0.0,
        yshift       = 0.0,
        eps          = 0.05,
        initial_time = 0.0
    ):
        super().__init__(device, x, y, eps=eps)
        """

        """
        self.L               = body_length
        self.folder_name     = folder_name
        self.file_name       = file_name
        self.save_data       = save_data
        self.plot_data       = plot_data

        self.target_fish     = target_fish
        self.start_recording = start_recording
        self.end_recording   = end_recording
        self.timestep        = timestep
        self.total_duration  = total_duration
        self.freq_scaling    = freq_scaling
        self.filter_freqs    = filter_freqs
        self.initial_time  = initial_time

        self.XC              = self.X-xshift
        self.YC              = self.Y-yshift


        # TYTELL-LIKE
        self.sb              = 0.07*body_length
        self.st              = 0.95*body_length
        self.wh              = 0.07*body_length
        self.wt              = 0.01*body_length

        # LIU-LIKE
        self.s1 = 0.54
        self.s2 = 0.72
        self.s3 = 0.83
        self.s4 = 0.85
        self.w1 = 0.16
        self.w2 = 0.004

        # Get the signal
        self.points_coords_df = get_experimental_signal(
            folder_name     = self.folder_name,
            file_name       = self.file_name,
            target_fish     = self.target_fish,
            start_recording = self.start_recording,
            end_recording   = self.end_recording,
            timestep        = self.timestep,
            total_duration  = self.total_duration,
            freq_scaling    = self.freq_scaling,
            save_data       = self.save_data,
            plot_data       = self.plot_data,
        )

        self.times   = self.points_coords_df['time'].values
        self.n_steps = len(self.times)

        # Coordinates
        self.points_x    = self.points_coords_df.filter(regex='x_').values * self.L
        self.points_y    = self.points_coords_df.filter(regex='y_').values * self.L
        self.points_x[:] = np.mean(self.points_x, axis=0)

        self.bodies = [self]

    def thk_liu(self, s):
        """
        fish width
        """

        s1, s2, s3, s4, w1, w2 = self.s1, self.s2, self.s3, self.s4, self.w1, self.w2
        s_star                 = s / ( s4 * self.L )

        c0, c1, c2, c3, c4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
        x0, x1, x2, x3, x4 = s_star**0.5, s_star, s_star**2, s_star**3, s_star**4

        return torch.where(
            s < s3 * self.L,
            5 * (s4 * w1) * ( c0*x0 + c1*x1 + c2*x2 + c3*x3 + c4*x4 ),
            w2
        )

    def thk_gazzola(self,s):
        """
        fish width
        """
        return torch.where(
            s<self.sb,
            torch.sqrt(2*self.wh*s-s**2),
            torch.where(
                s<self.st,
                self.wh-(self.wh-self.wt)*(((s-self.sb)/(self.st-self.sb))**2),
                self.wt*(self.L-s)/(self.L-self.st)
            )
        )

    def thk(self,s):
        return self.thk_gazzola(s)

    def sdf_fun(self, x,y):
        s = x.clamp(0,self.L)
        sdf = torch.sqrt((x-s)**2+y**2)
        return sdf-self.thk(s)


    def update(self, t, dt=1):
        """
        Update sdf properties from analytical rototranslation map
        """
        s = self.XC.clamp(0,self.L)
        new_x = self.XC

        # Get coordinates
        t0     = self.times[self.times<=t][-1]
        t1     = self.times[self.times>t][0]
        t0_ind = ( self.times == t0 )
        t1_ind = ( self.times == t1 )

        x0, x1 = self.points_x[t0_ind], self.points_x[t1_ind]
        y0, y1 = self.points_y[t0_ind], self.points_y[t1_ind]

        x_coords_t : np.ndarray = x0 + (x1-x0) * (t-t0) / (t1-t0)
        y_coords_t : np.ndarray = y0 + (y1-y0) * (t-t0) / (t1-t0)

        x_coords_t = x_coords_t.flatten()
        y_coords_t = y_coords_t.flatten()

        # Get coordinates interpolation
        s_coords_t = x_coords_t / x_coords_t[-1]
        # interp_y   = CubicSpline(s_coords_t, y_coords_t)
        interp_y   = lambda s: np.interp(s, s_coords_t, y_coords_t)

        # Get the new y coordinates
        new_y = (
            self.YC +
            torch.tensor(
                interp_y(s/self.L),
                dtype  = torch.float32,
                device = self.device
            )
        )

        self.body_u=0
        self.body_v=-(new_y-self.oldpos_v)/dt

        self.oldpos_v=new_y

        return [self.compute_sdf_properties(self.sdf_fun(new_x,new_y))]

    def initialize(self):
        """
        Initialize sdf properties at initial time
        """
        return self.update(self.initial_time)

    def save_signal(self, folder_name):
        ''' Save the signal to a csv file '''
        self.points_coords_df.to_csv(
            os.path.join(folder_name, 'kinematics_signals.csv'),
            index = False,
        )

class BodyMesh(Body):
    """
    """
    def __init__(self, device, x, y, mesh_file, update_maps, eps=0.05, compute_interp=True, nsamples=500, msamples=500, suit=0, plotting_meshes=False, **kwargs):
        super().__init__(device, x, y, eps=eps)
        self.mesh_file           = mesh_file
        self.compute_interp      = compute_interp
        self.save_folder         = kwargs.pop("save_folder", "")
        os.makedirs(self.save_folder+"interp_data", exist_ok=True)
        self.nsamples            = nsamples
        self.msamples            = msamples
        self.update_theta        = update_maps[0]
        self.update_translation  = update_maps[1]
        self.suit                = suit
        self.plotting            = plotting_meshes
        self.apply_closing_morph = kwargs.pop("apply_closing_morph", True)
        self.m2s                 = mesh2sdf(
            mesh_file,
            convexify=kwargs.pop("convexify", True),
            scale=kwargs.pop("scale", 1)
            )
        self.compute_sdfs()
        del self.m2s
        self.initialize()
        self.bodies = [self]


    def resample_closed_contour(self, points, spacing, keep_duplicate_endpoint=True):
        """
        Resample a closed contour for (approximately) uniform spacing.
        - points: (M,2) numpy array of (x,y). Can be closed (first==last) or open; treated as closed.
        - spacing: desired spacing between resampled points (float > 0).
        - keep_duplicate_endpoint: if True, return N+1 points with last == first (explicit closure).
                                if False, return N points (no duplicate at end).
        Returns:
        - new_pts: (N+1,2) or (N,2) array of resampled points.
        - actual_spacing: total_length / N  (the spacing actually used)
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("points must be an (M,2) array-like")

        if spacing <= 0:
            raise ValueError("spacing must be positive")

        # If not already closed, append first point for segment math
        if not np.allclose(pts[0], pts[-1]):
            pts_closed = np.vstack([pts, pts[0]])
        else:
            pts_closed = pts.copy()

        # segment vectors and lengths
        segs = pts_closed[1:] - pts_closed[:-1]
        seg_lens = np.hypot(segs[:,0], segs[:,1])
        total_length = seg_lens.sum()
        if total_length == 0:
            raise ValueError("zero-length contour")

        # choose number of equal intervals such that spacing ~ requested spacing
        N = max(3, int(round(total_length / spacing)))
        actual_spacing = total_length / N

        # cumulative distances along the closed polyline (start at 0, last = total_length)
        s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        # x,y coordinates corresponding to s
        x = pts_closed[:,0]
        y = pts_closed[:,1]

        # target sample locations: include the final total_length so last interpolates to first point
        target_s = np.linspace(0.0, total_length, N+1)

        # np.interp requires strictly increasing x; s is non-decreasing
        xi = np.interp(target_s, s, x)
        yi = np.interp(target_s, s, y)
        new_pts = np.vstack([xi, yi]).T

        if not keep_duplicate_endpoint:
            return new_pts[:-1], actual_spacing  # return N points
        return new_pts, actual_spacing      # return N+1 points where last==first


    def compute_sdfs(self):
        """
        Initialize the sdf interpolation function
        """
        self.bb = self.m2s.bounding_box()
        if self.compute_interp:

            # xmin=self.x.min().cpu().numpy()
            # xmax=self.x.max().cpu().numpy()
            # ymin=self.y.min().cpu().numpy()
            # ymax=self.y.max().cpu().numpy()
            # diag=np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)
            # xnp = np.linspace(xmin-2*diag,xmax+2*diag,self.nsamples)
            # ynp = np.linspace(ymin-2*diag,ymax+2*diag,self.msamples)

            cx_bb = (self.bb[0,1]+self.bb[0,0])/2
            cy_bb = (self.bb[1,1]+self.bb[1,0])/2
            diag = np.sqrt((self.bb[0,1]-self.bb[0,0])**2+(self.bb[1,1]-self.bb[1,0])**2)
            xnp = np.linspace(cx_bb-2*diag,cx_bb+2*diag,self.nsamples)
            ynp = np.linspace(cy_bb-2*diag,cy_bb+2*diag,self.msamples)

            binary_2d = np.ones((self.nsamples,self.msamples))

            X,Y=np.meshgrid(xnp,ynp,indexing="ij")
            xflat = X.flatten()
            yflat = Y.flatten()
            zflat = 0.0*np.ones_like(xflat)
            xyz   = np.stack([xflat,yflat,zflat],axis=1)
            query_pts=np.array(xyz.astype(np.float32))

            sdf_val_o3d, _=self.m2s(query_pts)
            if self.plotting:
                self.m2s.visualize()

            binary_2d=np.zeros((self.nsamples,self.msamples))
            binary_2d[sdf_val_o3d.reshape(X.shape)<0]=1

            if self.apply_closing_morph:
                gray = (255*binary_2d).astype('uint8')
                im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

                # im = cv2.erode(im, element, iterations = 1)
                im = cv2.dilate(im, element, iterations = 1)
                im = cv2.erode(im, element, iterations = 3)


                im=im[:,:,0]
            else:
                im=binary_2d

            if self.plotting:
                cv2.imshow("window_name", im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            binary_2d=np.where(im==0,1,-1) # this is the inside mask

            # (1) compute the inertial properties of the mesh file in 2d
            dx, dy = xnp[1]-xnp[0], ynp[1]-ynp[0]




            # (2) use skfmm to determine sdf on the full domain
            print("Computing the sdf for {}, with space steps ({},{})".format(self.mesh_file,xnp[1]-xnp[0],ynp[1]-ynp[0]))
            sdf_val = skfmm.distance(binary_2d, dx=[dx,dy])-self.suit

            # sdf_val = cv2.GaussianBlur(sdf_val, (5, 5), 0)


            ######################## Contour computation ########################

            # find contour lines
            cnt = np.array(measure.find_contours(sdf_val, 0)[0]).T
            # cnt = np.array(measure.find_contours(sdf_val-self.eps, 0)[0]).T
            cnt[0]=xnp[0]+cnt[0]*(xnp[1]-xnp[0])
            cnt[1]=ynp[0]+cnt[1]*(ynp[1]-ynp[0])
            curv_coord = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(cnt, axis=1)**2, axis=0)))))


            def signed_area(contour):
                x = contour[0,:]
                y = contour[1,:]
                return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

            def ensure_clockwise(contour):
                A = signed_area(contour)
                if A > 0:  # currently CCW
                    contour = contour[:,::-1]
                return contour

            cnt=ensure_clockwise(cnt)


            # ensure starting point is at the middle of the bounding box
            start_point = np.array([self.bb[0,0], 0]) # assuming y=0 is the centerline
            dists = np.sqrt((cnt[0]-start_point[0])**2+(cnt[1]-start_point[1])**2)
            # Find the closest point to start_point with positive y
            valid_indices = np.where(cnt[1] > 0)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[np.argmin(dists[valid_indices])]
            else:
                idx = np.argmin(dists)

            cnt = np.concatenate((cnt[:, idx+1:], cnt[:, :idx]), axis=1)


            # # resample contour lines for uniform spacing with spacing self.h
            ds=self.h #0.5*torch.sqrt(torch.tensor(self.h**2+self.h**2))
            # x, y, s_uniform = self.resample_contour(cnt[0], cnt[1], spacing=ds, closed=True)
            x, y, s_uniform = resample_contour(cnt[0], cnt[1], spacing=ds, closed=True) # the spacing is approximately ds
            del cnt
            cnt=np.array([x, y])


            # Compute ds and cumulative s
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dx**2 + dy**2)
            curv_coord = np.concatenate(([0], np.cumsum(ds)))

            # Create a vector where points in cnt above the first point
            sign_vec = np.where(cnt[1] >= cnt[1][0], 1, -1)


            if self.plotting:

                var=sdf_val
                plt.figure()
                plt.contourf(
                    X,
                    Y,
                    var
                )
                plt.plot(cnt[0], cnt[1], 'r', linewidth=2)
                plt.colorbar()
                plt.show()


                # Plot cnt as scatter with color given by a colormap
                cmap = cm.get_cmap('RdBu')
                n_points = cnt.shape[1]
                colors = cmap(np.linspace(0, 1, n_points))
                plt.scatter(cnt[0], cnt[1], c=colors, cmap=cmap, s=10)
                plt.show()



            ######################## END contour computation ########################
            print("Computing the interpolation functions for {}".format(self.mesh_file))


            interp_data_dir = "interp_data"
            if not os.path.exists(interp_data_dir):
                os.makedirs(interp_data_dir)

            np.save(self.save_folder+"interp_data/xnp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",xnp)
            np.save(self.save_folder+"interp_data/ynp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",ynp)
            np.save(self.save_folder+"interp_data/sdf_val_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",sdf_val)
            np.save(self.save_folder+"interp_data/cnt_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy", cnt)
            np.save(self.save_folder+"interp_data/curv_coord_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy", curv_coord)
            np.save(self.save_folder+"interp_data/sign_vec_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy", sign_vec)



    def initialize(self):
        xnp = np.load(self.save_folder+"interp_data/xnp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        ynp = np.load(self.save_folder+"interp_data/ynp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        sdf_val = np.load(self.save_folder+"interp_data/sdf_val_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        cnt = np.load(self.save_folder+"interp_data/cnt_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        curv_coord = np.load(self.save_folder+"interp_data/curv_coord_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        sign_vec = np.load(self.save_folder+"interp_data/sign_vec_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")

        self.sdf = RegularGridInterpolator(
            (
                torch.from_numpy(xnp).type(self.dtype).to(self.device),
                torch.from_numpy(ynp).type(self.dtype).to(self.device)
            ),
            torch.from_numpy(sdf_val).type(self.dtype).to(self.device),
            fill_value="nearest",
            method=1 # quadratic
        )

        # self.sdf = self.sdf_interp(
        #     self.stacked_xy[0],
        #     self.stacked_xy[1]
        # ).reshape(self.nx, self.ny)

        self.curv_coord = torch.from_numpy(curv_coord).type(self.dtype).to(self.device)
        self.cnt        = torch.from_numpy(cnt).type(self.dtype).to(self.device)
        self.cnt_update = self.cnt.clone().detach()
        self.cnt_u=torch.zeros_like(self.cnt_update[0])
        self.cnt_v=torch.zeros_like(self.cnt_update[1])
        self.cnt_f_u=torch.zeros_like(self.cnt_update[0])
        self.cnt_f_v=torch.zeros_like(self.cnt_update[1])
        self.cnt_int_f_u=torch.zeros_like(self.cnt_update[0])
        self.cnt_int_f_v=torch.zeros_like(self.cnt_update[1])
        self.r_com=torch.zeros_like(self.cnt_update)
        self.ds = self.curv_coord[1]-self.curv_coord[0]
        # self.ds = np.diff(self.curv_coord)
        self.mask = torch.arange(len(self.curv_coord), device=self.device)
        self.sign_vec = torch.from_numpy(sign_vec).type(self.dtype).to(self.device)

        # return self.update(0)

    # def update(self, t, dt=1):
    #     return [self.update_body(
    #         self.sdf_interp,
    #         self.update_theta(t),
    #         (
    #             self.update_translation[0](t),
    #             self.update_translation[1](t)
    #         ),
    #         dt=dt
    #     )]

    def update(self, iteration, t, dt=1):
        pass

    def visualize(self):
        self.m2s.visualize()

class CompositeBodyMesh(Body):

    def __init__(self, device, x, y, sdf_folder, sdf_name, costum_update, eps=0.05, compute_interp=True, nsamples=2**12, msamples=2**12, plotting=False, plotting_meshes=False, suit=0.0, **kwargs):
        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """
        super().__init__(device, x, y, eps=eps)

        self.sdf_folder      = sdf_folder
        self.sdf             = ModelSDF.read(sdf_folder+sdf_name)[0]
        self.bodies          = []
        self.suit            = suit
        self.plotting        = plotting
        self.plotting_meshes = plotting_meshes
        for link_i, link in enumerate(self.sdf.links):
            # if link_i%2==0:
                mesh_name = link["visuals"][0]["geometry"]["uri"]
                mesh_gpath = sdf_folder+mesh_name
                initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
                update_funcs = (
                    lambda t: 180,
                    [
                        lambda t, initial_pose=initial_pose: -initial_pose[0],
                        lambda t, initial_pose=initial_pose: -initial_pose[1],
                    ]
                    )
                body = BodyMesh(
                        device, x, y,
                        mesh_gpath,
                        update_funcs,
                        eps=eps,
                        compute_interp=compute_interp,
                        nsamples=nsamples, msamples=msamples,
                        suit=suit,
                        plotting_meshes=plotting_meshes,
                        **kwargs
                    )
                body.id = link_i
                self.bodies.append(body)
        self.nbodies = len(self.bodies)
        self.costum_update = costum_update
        self.compute_interp = compute_interp

        self.mu_funcs               = self.bodies[0].mu_funcs
        self.compute_sdf_properties = self.bodies[0].compute_sdf_properties
        self.sdf_vals = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.sdf_vals_u = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.sdf_vals_v = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.u_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.v_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)

        self.sdf_val_u=torch.zeros_like(self.X)
        self.sdf_val_v=torch.zeros_like(self.X)
        self.com_pos  = torch.zeros((self.nbodies,2),device=device)


        if not self.costum_update:
            self.initialize() # initialize the sdf interpolation functions


    def initialize(self):
        self.update(torch.tensor(0.0,device=self.device,dtype=self.dtype), 0)

        # for i, body in enumerate(self.bodies):
        #     body.initialize()
        #     self.sdf_vals[i]=body.sdf

        # self.sdf_val = torch.min(self.sdf_vals,axis=0)[0]

        # if self.plotting:
        #     var=self.sdf_val.cpu()
        #     extent = (
        #         torch.min(self.bodies[0].x.cpu()), torch.max(self.bodies[0].x.cpu()),
        #         torch.min(self.bodies[0].y.cpu()), torch.max(self.bodies[0].y.cpu())
        #     )

        #     # visualize computed interpolation functions over the domain
        #     plt.figure(figsize=(20,10))
        #     plt.imshow(
        #         var.T,
        #         extent = extent,
        #         origin = "lower",
        #         interpolation=None
        #     )
        #     plt.contour(self.bodies[0].X.cpu(),self.bodies[0].Y.cpu(),var, colors='k', levels=[0])
        #     plt.show()

        # self.body_u=torch.zeros_like(self.bodies[0].X)
        # self.body_v=torch.zeros_like(self.bodies[0].X)



    def update(self, t, iteration, dt=1):
        (angles, translations) = self.costum_update(t)
        sdf_properties = []
        for body_i, body in enumerate(self.bodies):
            sdf_properties.append(
                body.update_body(
                    body.sdf_interp,
                    angles[body_i],
                    (
                        translations[body_i,0],
                        translations[body_i,1]
                    ),
                    dt=dt
                )
            )
        self.sdf_val = torch.min(torch.stack([prop[0] for idx, prop in enumerate(sdf_properties)]),axis=0)[0]


        # return sdf_properties


    def visualize(self):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        for body in self.bodies:
            viewer.add_geometry(body.m2s._mesh)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()


    # Function to create a Gaussian kernel
    def gaussian_kernel(self, size: int, sigma: float):
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




class MultiAnimatBodies(Body):

    def __init__(self, device, x, y, experiment_options, eps=0.05, compute_interp=True, nsamples=2**12, msamples=2**12, plotting=False, plotting_meshes=False, suit=0.0, **kwargs):

        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """

        super().__init__(device, x, y, eps=eps)

        self.suit = suit
        self.plotting        = plotting
        self.plotting_meshes = plotting_meshes

        self.body_ids = []
        self.bodies = []


        for animat_i, animat in enumerate(experiment_options.animats):
            sdf = ModelSDF.read(animat.sdf)[0] # this is the sdf content
            sdf_folder      = os.path.dirname(animat.sdf)

            for link_i, link in enumerate(sdf.links):
                geometry = link["collisions"][0]["geometry"]
                if "uri" in geometry:
                    mesh_name = geometry["uri"]
                    mesh_gpath = sdf_folder+"/"+mesh_name
                    initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
                    update_funcs = (
                        lambda t: 180,
                        [
                            lambda t, initial_pose=initial_pose: -initial_pose[0],
                            lambda t, initial_pose=initial_pose: -initial_pose[1],
                        ]
                        )
                    if "scale" in geometry:
                        scale = geometry["scale"]
                        assert scale[0]==scale[1]==scale[2], "Non-uniform scaling not supported."
                        scale = scale[0]
                        kwargs["scale"] = scale
                    body = BodyMesh(
                            device, x, y,
                            mesh_gpath,
                            update_funcs,
                            eps=eps,
                            compute_interp=compute_interp,
                            nsamples=nsamples, msamples=msamples,
                            suit=suit,
                            plotting_meshes=plotting_meshes,
                            **kwargs
                        )
                    self.bodies.append(body)

                elif "radius" in geometry and "length" in geometry:
                    """ Create analytical bodies for capsule """

                    radius = torch.tensor(geometry["radius"],dtype=x.dtype,device=x.device)
                    length = torch.tensor(geometry["length"],dtype=x.dtype,device=x.device)
                    if "L" in link["name"]:
                        sdf_fun = lambda x,y : sdUnevenCapsule(x,y,radius,radius,length, side="L")
                    elif "R" in link["name"]:
                        sdf_fun = lambda x,y : sdUnevenCapsule(x,y,radius,radius,length, side="R")
                    else:
                        raise ValueError("Capsule link name must contain 'L' or 'R' to define the side.")
                    initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
                    update_maps = (lambda t: 0, [lambda t: -initial_pose[0], lambda t: -initial_pose[1]]) # set dummy update maps for initialization (not used)
                    self.bodies.append(
                        BodyAnalytical(
                            device, x, y, sdf_fun, update_maps, eps=eps, plotting=False, pre_update=False
                        )
                    )
                    self.bodies[-1].bb=[[ -radius.cpu(), radius.cpu() ], [ -radius.cpu(), radius.cpu() ], [-radius.cpu(), radius.cpu() ]]

                elif "radius" in geometry and "length" not in geometry:
                    """ Create analytical bodies for spheres """
                    radius = torch.tensor(geometry["radius"],dtype=x.dtype,device=x.device)
                    sdf_fun = lambda x,y : circle(x,y,xt=0,yt=0,r=radius)
                    initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
                    update_maps = (lambda t: 0, [lambda t: -initial_pose[0], lambda t: -initial_pose[1]]) # set dummy update maps for initialization (not used)
                    self.bodies.append(
                        BodyAnalytical(
                            device, x, y, sdf_fun, update_maps, eps=eps, plotting=False, pre_update=False
                        )
                    )
                    self.bodies[-1].bb=[[ -radius.cpu(), radius.cpu() ], [ -radius.cpu(), radius.cpu() ], [-radius.cpu(), radius.cpu() ]]
                else:
                    raise ValueError("Unsupported geometry type in SDF.")
                self.body_ids.append([animat_i,link_i])




        self.nbodies = len(self.bodies)
        self.sdf_vals = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.sdf_vals_u = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.sdf_vals_v = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.u_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.v_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)

        self.sdf_val_u=torch.zeros_like(self.X)
        self.sdf_val_v=torch.zeros_like(self.X)
        self.com_pos  = torch.zeros((self.nbodies,2),device=device)

        # for link_i, link in enumerate(self.sdf.links):
        #     # if link_i%2==0:
        #         mesh_name = link["visuals"][0]["geometry"]["uri"]
        #         mesh_gpath = sdf_folder+mesh_name
        #         initial_pose = np.array(link.pose).astype(x.cpu().numpy().dtype)
        #         update_funcs = (
        #             lambda t: 180,
        #             [
        #                 lambda t, initial_pose=initial_pose: -initial_pose[0],
        #                 lambda t, initial_pose=initial_pose: -initial_pose[1],
        #             ]
        #             )
        #         body = BodyMesh(
        #                 device, x, y,
        #                 mesh_gpath,
        #                 update_funcs,
        #                 eps=eps,
        #                 compute_interp=compute_interp,
        #                 nsamples=nsamples, msamples=msamples,
        #                 suit=suit,
        #                 plotting_meshes=plotting_meshes,
        #                 **kwargs
        #             )
        #         body.id = link_i
        #         self.bodies.append(body)
        # self.nbodies = len(self.bodies)
        # self.costum_update = costum_update
        # self.compute_interp = compute_interp

    #     self.mu_funcs               = self.bodies[0].mu_funcs
    #     self.compute_sdf_properties = self.bodies[0].compute_sdf_properties
    #     self.sdf_vals = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
    #     self.sdf_vals_u = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
    #     self.sdf_vals_v = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
    #     self.u_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
    #     self.v_vals   = torch.zeros((self.nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)

    #     self.sdf_val_u=torch.zeros_like(self.X)
    #     self.sdf_val_v=torch.zeros_like(self.X)
    #     self.com_pos  = torch.zeros((self.nbodies,2),device=device)


    #     if not self.costum_update:
    #         self.initialize() # initialize the sdf interpolation functions


    # def initialize(self):
    #     self.update(torch.tensor(0.0,device=self.device,dtype=self.dtype), 0)

    #     # for i, body in enumerate(self.bodies):
    #     #     body.initialize()
    #     #     self.sdf_vals[i]=body.sdf

    #     # self.sdf_val = torch.min(self.sdf_vals,axis=0)[0]

    #     # if self.plotting:
    #     #     var=self.sdf_val.cpu()
    #     #     extent = (
    #     #         torch.min(self.bodies[0].x.cpu()), torch.max(self.bodies[0].x.cpu()),
    #     #         torch.min(self.bodies[0].y.cpu()), torch.max(self.bodies[0].y.cpu())
    #     #     )

    #     #     # visualize computed interpolation functions over the domain
    #     #     plt.figure(figsize=(20,10))
    #     #     plt.imshow(
    #     #         var.T,
    #     #         extent = extent,
    #     #         origin = "lower",
    #     #         interpolation=None
    #     #     )
    #     #     plt.contour(self.bodies[0].X.cpu(),self.bodies[0].Y.cpu(),var, colors='k', levels=[0])
    #     #     plt.show()

    #     # self.body_u=torch.zeros_like(self.bodies[0].X)
    #     # self.body_v=torch.zeros_like(self.bodies[0].X)








    # def sdf_from_obj(self, mesh_file="box.obj"):
    #     """
    #     No longer used, keep for storing in case
    #     """

    #     m2s = mesh2sdf(mesh_file)
    #     xflat = self.xflat.cpu().numpy().astype(np.float32)
    #     yflat = self.yflat.cpu().numpy().astype(np.float32)
    #     xflat = self.xflat.cpu().numpy().astype(np.float32)
    #     yflat = self.yflat.cpu().numpy().astype(np.float32)
    #     zflat = np.zeros_like(xflat)
    #     xyz   = np.stack([xflat,yflat,zflat],axis=1)

    #     query_pts = np.array(xyz,dtype=np.float32)
    #     sdf_val, sdf_grad= m2s(query_pts)
    #     sdf_val  = torch.from_numpy(sdf_val).to(self.device).reshape(self.nx, self.ny)
    #     sdf_grad = torch.from_numpy(sdf_grad).to(self.device)

    #     # subsample arrows
    #     gradx = sdf_grad[:,0].reshape(self.nx, self.ny)
    #     grady = sdf_grad[:,1].reshape(self.nx, self.ny)
    #     norm  = torch.sqrt(gradx**2+grady**2)

    #     # compute curvature
    #     numerator = (grady**2)*torch.gradient(gradx, spacing=self.h, axis=0)[0] \
    #                 +(gradx**2)*torch.gradient(grady, spacing=self.h, axis=1)[0] \
    #                 -2*gradx*grady*torch.gradient(grady, spacing=self.h, axis=0)[0]
    #     denominator = norm**3
    #     # numerator = torch.gradient(gradx, dim=0, spacing=self.h)[0]+torch.gradient(grady, dim=1, spacing=self.h)[0]
    #     # denominator = (1+gradx**2+grady**2)**2
    #     curvature = numerator/denominator

    #     # normalize gradient
    #     gradx/=norm
    #     grady/=norm


    #     return (
    #         sdf_val,
    #         gradx,
    #         grady,
    #         curvature,
    #     )

    #     # return sdf_val.reshape(self.nx, self.ny), du.reshape(self.nx, self.ny), dv.reshape(self.nx, self.ny)






    # def compute_sdf_from_interp_query(self, xquery, yquery):
    #     return self.compute_sdf_properties(
    #         self.sdf_interp(xquery,yquery).reshape(self.nx, self.ny)
    #     )


    # def update_interp_from_rototranslation2D(self, theta, transl, dt=1):
    #     theta = theta*torch.pi/180
    #     s = torch.sin(torch.tensor(theta, device=self.device))
    #     c = torch.cos(torch.tensor(theta, device=self.device))
    #     rot = torch.stack([torch.stack([c, -s]),
    #                     torch.stack([s, c])]).to(self.device)
    #     trans = torch.stack((transl[0]*self.ones_stacked, transl[1]*self.ones_stacked))
    #     newpoints=rot.T@self.stacked_xy-trans

    #     newpos = rot@self.stacked_xy+trans
    #     newpos_u = newpos[0].reshape(self.nx, self.ny)
    #     newpos_v = newpos[1].reshape(self.nx, self.ny)

    #     # self.body_uprev = self.body_u
    #     # self.body_vprev = self.body_v

    #     self.body_u=(newpos_u-self.oldpos_u)/dt
    #     self.body_v=(newpos_v-self.oldpos_v)/dt

    #     self.oldpos_u = newpos_u
    #     self.oldpos_v = newpos_v

    #     (
    #         new_sdf,
    #         new_nx,
    #         new_ny,
    #         new_curv,
    #     ) = self.compute_sdf_from_interp_query(newpoints[0], newpoints[1])



    #     return new_sdf, new_nx, new_ny, new_curv


class CompositeSegmentBody:

    def __init__(self, device, x, y, sdf_folder, sdf_name, eps=0.05):
        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """
        self.device          = device
        self.thk             = 0.0005
        self.body            = Body(device,x,y,eps=eps)
        self.sdf_folder      = sdf_folder
        self.sdf             = ModelSDF.read(sdf_folder+sdf_name)[0]
        self.n               = len(self.sdf.links)
        self.body            = Body(device,x,y,eps=0.05)
        self.nlinks          = len(self.sdf.links)
        self.initial_poses   = torch.tensor([link.pose[:2] for link in self.sdf.links],device=device)
        self.initial_lin_vel = torch.zeros((self.initial_poses.shape[0]),2,device=device)
        self.initial_ang_vel = torch.zeros(self.initial_poses.shape[0],device=device)

        self.ds=torch.zeros((self.n-1,self.body.stacked_xy.shape[1]),device=device)
        self.us=torch.zeros((self.n-1,self.body.X.shape[0],self.body.X.shape[1]),device=device)
        self.vs=torch.zeros((self.n-1,self.body.X.shape[0],self.body.X.shape[1]),device=device)
        self.uv=torch.zeros((self.n-1,2,self.body.stacked_xy.shape[1]),device=device)

        self.compute_sdf_and_velocities(-self.initial_poses, self.initial_lin_vel, self.initial_ang_vel, dt=1)

    def compute_sdf_and_velocities(self, p, com_lin_vel, com_ang_vel, dt=1, plotting=False):
        """
        p: link poses - dim n
        com_lin_vel: com linear vel - dim n
        com_ang_vel: com ang vel - dim n
        """

        for i in range(self.n-1):
            e=p[i+1]-p[i]
            p_o=self.body.stacked_xy-p[i][:,None]
            h=torch.clamp((p_o*e[:,None]).sum(axis=0)/torch.dot(e,e),0.0,1.0)
            pq=p_o-e[:,None]*h
            self.ds[i]=(torch.linalg.norm(pq,axis=0)-self.thk)

            c=torch.cos(com_ang_vel[i])
            s=torch.sin(com_ang_vel[i])
            R1=torch.tensor([[c,-s],[s,c]],device=self.device)
            c=torch.cos(com_ang_vel[i+1])
            s=torch.sin(com_ang_vel[i+1])
            R2=torch.tensor([[c,-s],[s,c]],device=self.device)

            line_point=e[:,None]*h+p[i][:,None]
            self.uv[i]=(
                (1-h)*(0*com_lin_vel[i][:,None]+R1@line_point)+
                h*(0*com_lin_vel[i+1][:,None]+R2@line_point)+
                -line_point
            ) #/ dt


        idx=self.ds.argmin(0).unsqueeze(0).expand(self.ds.shape)
        self.sdf=self.ds.gather(0,idx)[0].reshape(self.body.nx,self.body.ny)
        self.body_u=self.uv[:,0,:].gather(0,idx)[0].reshape(self.body.nx,self.body.ny)
        self.body_v=self.uv[:,1,:].gather(0,idx)[0].reshape(self.body.nx,self.body.ny)



        # ==== plotting ====
        if plotting:
            import matplotlib.pyplot as plt
            X=self.body.X.cpu()
            Y=self.body.Y.cpu()
            x=self.body.x.cpu()
            y=self.body.y.cpu()
            var=self.sdf.cpu()
            pcpu=p.cpu()
            plt.figure()
            plt.imshow(
                    var.T,
                    extent = (
                        torch.min(x.cpu()), torch.max(x.cpu()),
                        torch.min(y.cpu()), torch.max(y.cpu())
                    ),
                    origin = "lower",
                    cmap = "Greys"
                )
            plt.colorbar()
            plt.contour(X,Y,var, colors='k', levels=[0], linestyles='-')
            cset1 = plt.contourf(X,Y, var, levels=20, cmap="Greys")
            plt.plot(pcpu[:,0],pcpu[:,1],'r',marker='o')
            # plt.plot(p_new[:,0],p_new[:,1],'g',marker='o')
            subsample_n = 2**3
            plt.quiver(
                X[::subsample_n,::subsample_n],
                Y[::subsample_n,::subsample_n],
                self.body_u[::subsample_n,::subsample_n].cpu(),
                self.body_v[::subsample_n,::subsample_n].cpu(),
                color='g',
                scale=dt, scale_units='xy'
            )

            # dp2=p_new[2]-p[2]
            # plt.quiver(
            #     p[2][0],
            #     p[2][1],
            #     dp2[0],
            #     dp2[1],
            #     color='r',
            #     scale=dt, scale_units='xy'
            # )


            # # ==== plotting ====
            # var=self.body_u.cpu()
            # plt.figure()
            # plt.imshow(
            #         var.T,
            #         extent = (
            #             torch.min(x.cpu()), torch.max(x.cpu()),
            #             torch.min(y.cpu()), torch.max(y.cpu())
            #         ),
            #         origin = "lower",
            #         cmap = "Greys"
            #     )
            # plt.colorbar()
            # plt.contour(X,Y,self.sdf.cpu(), colors='k', levels=[0], linestyles='-')
            plt.show()







def test_single_mesh():

    N=2**10

    # mesh_file="box.obj"
    # x=np.linspace(-1,1,N,dtype=dtype)
    # y=np.linspace(-1,1,N,dtype=dtype)

    x=torch.linspace(-0.002,0.002,N)
    y=torch.linspace(-0.002,0.002,N)
    mesh_file = "/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_4.obj"

    m2s = mesh2sdf(mesh_file)

    m2s.visualize(wireframe=False)


    body = BodyMesh("cpu", x, y, mesh_file, (lambda t: 0, [lambda t:0, lambda t:0]),eps=2*(x[1]-x[0]), compute_interp=False,suit=0.0)
    sdf_val = body.initialize()[0]
    sdf_val, du, dv, curv = body.compute_sdf_properties(sdf_val)

    dtype = np.float32

    # compute sdf on query points

    X,Y=np.meshgrid(x,y,indexing="ij")
    # xflat = X.flatten()
    # yflat = Y.flatten()
    # zflat = np.zeros_like(xflat)
    # xflat = xflat.astype(dtype)
    # yflat = yflat.astype(dtype)
    # xyz   = np.stack([xflat,yflat,zflat],axis=1)

    # query_pts=np.array(xyz,dtype=dtype)


    # query_pts=np.array(list(it.product(x,y,[0.0])),dtype=dtype)
    # sdf_val, sdf_grad=m2s(query_pts)


    # sdf_val = sdf_val.reshape(len(x), len(y))
    # sdf_grad = sdf_grad.reshape(len(x), len(y), 3)

    # du = sdf_grad[:,:,0]
    # dv = sdf_grad[:,:,1]
    # norm = np.sqrt(du**2+dv**2)
    # du/=norm
    # dv/=norm

    # zoom
    # x=torch.linspace(-0.0004,-0.00035,N)
    # y=torch.linspace(0.00052,0.00053,N)
    # plotting
    import matplotlib.pyplot as plt
    X,Y=np.meshgrid(x,y,indexing='ij')
    plt.figure()
    plt.imshow(
        sdf_val.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    plt.colorbar()
    plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
    subsample_n = 2**6
    plt.quiver(
        X[::subsample_n,::subsample_n],
        Y[::subsample_n,::subsample_n],
        du[::subsample_n,::subsample_n],
        dv[::subsample_n,::subsample_n],
        color='g'
    )
    plt.savefig("mesh_body_example.pdf")


    plt.figure()
    plt.imshow(
        curv.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    plt.colorbar()


    plt.show()


def test_body_interpolation():

    use_gpu=True

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)



    N=2**10+1

    x=torch.linspace(-0.002,0.002,N)
    y=torch.linspace(-0.002,0.002,N)
    filename = "/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_10.obj"

    # x=torch.linspace(-60,180,N)
    # y=torch.linspace(-60,180,N)
    # filename = "cylinder.obj"

    # x=torch.linspace(-4,4,N)
    # y=torch.linspace(-4,4,N)
    # filename = "box.obj"

    x = x.to(device)
    y = y.to(device)


    N=2**10+1
    N=2**10+1

    body = BodyMesh(device, x, y, filename, (lambda t: 0, [lambda t:0, lambda t:0]),eps=2*(x[1]-x[0]))
    d, nx, ny, curv = body.initialize()[0]

    body.visualize()




    X=body.X.cpu()
    Y=body.Y.cpu()
    d=d.cpu()
    nx=nx.cpu()
    ny=ny.cpu()
    curv=curv.cpu()

    import matplotlib.pyplot as plt

    plt.figure()
    cset1 = plt.contourf(X,Y, d, cmap="Greys")
    plt.colorbar(cset1)
    plt.contour(X,Y, d, colors='k', levels=[0], linestyles='dashed')

    plt.contour(X,Y, d, colors='k', levels=[0], linestyles='dashed')
    subsample_n = 2**7
    plt.quiver(
        X[::subsample_n,::subsample_n],
        Y[::subsample_n,::subsample_n],
        nx.cpu()[::subsample_n,::subsample_n],
        ny.cpu()[::subsample_n,::subsample_n],
        color='g'
    )


    # # d, nx, ny, curv = body.update_fun_from_function(box, torch.tensor(45), [100,0])
    # d, nx, ny, curv = body.update_interp_from_rototranslation2D(torch.tensor(30),[0.005,0.00])
    # (mu0, mu1) = body.mu_funcs(d)


    # X=body.X.cpu()
    # Y=body.Y.cpu()
    # d=d.cpu()
    # nx=nx.cpu()
    # ny=ny.cpu()
    # curv=curv.cpu()
    # mu0=mu0.cpu()
    # mu1=mu1.cpu()

    # plt.figure()
    # cset1 = plt.contourf(X,Y, d, cmap="Greys")
    # plt.colorbar(cset1)
    # plt.contour(X,Y, d, colors='k', levels=[0], linestyles='dashed')
    # subsample_n = 2**7
    # plt.quiver(
    #     X[::subsample_n,::subsample_n],
    #     Y[::subsample_n,::subsample_n],
    #     nx[::subsample_n,::subsample_n],
    #     ny[::subsample_n,::subsample_n],
    #     color='g'
    # )

    plt.figure()
    # cset2 = plt.contourf(X,Y,1/curv, cmap="Greys")
    # plt.colorbar(cset2)
    plt.contour(X,Y,d, colors='k', levels=[0], linestyles='dashed')
    plt.imshow(
            curv.T,
            extent = (
                torch.min(x.cpu()), torch.max(x.cpu()),
                torch.min(y.cpu()), torch.max(y.cpu())
            ),
            origin = "lower",
            cmap = "Greys"
        )

    # plt.figure()
    # plt.imshow(
    #         body.body_u.cpu(),
    #         extent = (
    #             torch.min(x.cpu()), torch.max(x.cpu()),
    #             torch.min(y.cpu()), torch.max(y.cpu())
    #         ),
    #         origin = "lower",
    #         cmap = "Greys"
    #     )

    plt.show()


def test_moving_mesh():

    import matplotlib.pyplot as plt
    import math
    from matplotlib import animation

    print("Using the CPU.")
    device = torch.device("cpu")
    torch.set_num_threads(8)

    N=2**11+1

    x=torch.linspace(-300,300,N)
    y=torch.linspace(-300,300,N)
    x = x.to(device)
    y = y.to(device)


    N=2**10+1
    sdf = lambda x,y : circle (x,y,xt=0,yt=0,r=25)
    update = (lambda i : torch.tensor(1)*i, [lambda i : 0*math.cos(i/10),lambda i : 0*math.sin(i/10)])
    body = BodyAnalytical(device,x,y,sdf,update,eps=2*(x[1]-x[0]))
    d0, nx, ny, curv = body.initialize()

    # x=torch.linspace(-0.01,0.01,N)
    # y=torch.linspace(-0.01,0.01,N)
    # filename = "/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_15.obj"
    # update = (lambda i : torch.tensor(10)*i, [lambda i : 0.00*math.cos(i/10),lambda i : 0.00*math.sin(i/10)])
    # body = BodyMesh(device, x, y, filename, update)
    # d0, nx, ny, curv = body.initialize()

    fig = plt.figure()
    im = plt.imshow(
            d0.T,
            extent = (
                torch.min(x.cpu()), torch.max(x.cpu()),
                torch.min(y.cpu()), torch.max(y.cpu())
            ),
            origin        = "lower",
            cmap          = "Greys",
            interpolation = "none"
        )
    ctr = plt.contour(body.X,body.Y, d0, colors='k', levels=[0], linestyles='dashed')

    subsample_n = 2**7
    sct=plt.scatter(body.X[::subsample_n,::subsample_n].flatten(),body.Y[::subsample_n,::subsample_n].flatten())

    def init():
        d, nx, ny, curv = body.initialize()
        im.set_array(d.T)
        ctr = plt.contour(body.X,body.Y, d, colors='k', levels=[0], linestyles='dashed')
        ctr0 = plt.contour(body.X,body.Y, d0, colors='k', levels=[0], linestyles='dashed')
        quiv= plt.quiver(
            body.X[::subsample_n,::subsample_n],
            body.Y[::subsample_n,::subsample_n],
            body.body_u.cpu()[::subsample_n,::subsample_n],
            body.body_v.cpu()[::subsample_n,::subsample_n],
            color='g'
        )
        return [im,ctr,ctr0,quiv]


    global X0, Y0
    X0=body.X
    Y0=body.Y
    def animate(i):
        global X0, Y0
        d, nx, ny, curv = body.update(i)
        im.set_array(d.T)
        ctr = plt.contour(body.X,body.Y, d, colors='k', levels=[0], linestyles='dashed')
        ctr0 = plt.contour(body.X,body.Y, d0, colors='k', levels=[0], linestyles='dashed')
        quiv = plt.quiver(
            body.oldpos_u[::subsample_n,::subsample_n],
            body.oldpos_v[::subsample_n,::subsample_n],
            body.body_u.cpu()[::subsample_n,::subsample_n],
            body.body_v.cpu()[::subsample_n,::subsample_n],
            color='g'
        )


        u_=body.body_u
        v_=body.body_v
        dt=1
        X=body.oldpos_u #X0+u_*dt
        Y=body.oldpos_v #Y0+v_*dt
        sct.set_offsets(
            torch.stack((
                    X[::subsample_n,::subsample_n].flatten(),
                    Y[::subsample_n,::subsample_n].flatten()
                )
            ).T
        )
        X0=X
        Y0=Y

        return [im,ctr,ctr0,sct,quiv]

    animation = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=100, interval=0, blit=True)

    # pause on click
    global paused
    paused = False
    def toggle_pause(self, *args, **kwargs):
        global paused
        if paused:
            animation.resume()
        else:
            animation.pause()
        paused = not paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)



    plt.show()


def test_body():

    use_gpu=True

    mesh_file = "/data/andreaferrario/lilytorch/lilytorch/1guillasim/models/1guilla_v1/sdf/meshes/link0.obj"

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    N=2**8
    x=torch.linspace(-0.02,0.3,N)
    y=torch.linspace(-0.05,0.05,N)
    X,Y=torch.meshgrid(x,y,indexing="ij")

    x = x.to(device)
    y = y.to(device)

    body = BodyMesh(device, x, y, mesh_file, (lambda t: 0, [lambda t:0, lambda t:0]), eps=2*(x[1]-x[0]), compute_interp=True, convexify=True, plotting_meshes=False)
    body.initialize()
    d, nx, ny, curv = body.compute_sdf_properties(body.sdf)
    (mu0, mu1) = body.mu_funcs(d)

    import matplotlib.pyplot as plt
    import matplotlib

    x=x.cpu()
    y=y.cpu()
    d=d.cpu()
    nx=nx.cpu()
    ny=ny.cpu()
    mu0=mu0.cpu()
    mu1=mu1.cpu()

    plt.figure()
    norm = matplotlib.colors.Normalize(vmin=d.min(), vmax=d.max())
    cset1 = plt.contourf(X,Y,d, cmap="Greys")
    cset2 = plt.contour(X,Y,d, colors='k', levels=[0], linestyles='dashed')
    plt.colorbar(cset1)
    subsample_n = 2**3
    plt.quiver(
        X[::subsample_n,::subsample_n],
        Y[::subsample_n,::subsample_n],
        nx[::subsample_n,::subsample_n],
        ny[::subsample_n,::subsample_n],
        color='g'
    )

    plt.figure()
    cset1 = plt.contourf(X,Y,mu0,cmap="Greys")
    plt.colorbar(cset1)

    plt.figure()
    cset1 = plt.contourf(X,Y,mu1,cmap="Greys")
    plt.colorbar(cset1)

    plt.figure()
    cset1 = plt.contourf(X,Y,nx,cmap="Greys")
    plt.colorbar(cset1)

    plt.figure()
    cset1 = plt.contourf(X,Y,ny,cmap="Greys")
    plt.colorbar(cset1)



    plt.show()


def test_composite_mesh():
    sdf_name = "zebrafish.sdf"
    sdf_folder="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/"

    m2s = COMPOSITEmesh2sdf(sdf_name, sdf_folder)
    m2s.visualize()

    dtype = np.float32

    # compute sdf on query points
    import itertools as it
    n=2**10
    x=np.linspace(-0.002,0.05,n,dtype=dtype)
    y=np.linspace(-0.002,0.002,n,dtype=dtype)
    query_pts=np.array(list(it.product(x,y,[0.0])),dtype=dtype)
    sdf_vals, sdf_grads=m2s(query_pts)


    import matplotlib.pyplot as plt
    import matplotlib

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


def test_fish_mesh():
    sdf_name = "zebrafish.sdf"
    sdf_folder="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/"
    device = torch.device("cpu")
    torch.set_num_threads(8)

    # compute sdf on query points
    import matplotlib.pyplot as plt
    n=2**10
    x=torch.linspace(-0.002,0.05,n)
    y=torch.linspace(-0.002,0.002,n)
    X,Y=torch.meshgrid(x,y,indexing='ij')

    # compute the model boolean's inner part
    composite_body=CompositeSegmentBody(device,x, y, sdf_folder, sdf_name, eps=0.05)
    # sdf_properties = composite_body.initialize()
    # for sdf_property in sdf_properties:
    #     sdf=sdf_property[0]
    #     # plt.contour(X,Y,sdf, colors='k', levels=[0])
    # boolean_model_min = torch.stack([sdf[0] for sdf in sdf_properties]).min(axis=0)[0]<0

    # # define parameters for the line approximation
    # poses = torch.tensor([
    #     [body.update_translation[0](0),
    #     body.update_translation[1](0)]
    # for body in composite_body.bodies]) # translations are obtained from the update translation functions at time 0

    # assert len((sdf_properties))==poses.shape[0]

    # def sdf_approx(V,thk,plotting=False):
    #     """
    #     sdf function for the line segments
    #     """
    #     m=V.shape[0]
    #     sdf=torch.zeros((m-1,n,n))
    #     # thk=torch.tensor([0.5,0.3,0.2,0.7,1])
    #     for i in range(m-1):
    #         sdf[i]=segment(X,Y,V[i],V[i+1],thk[i],thk[i+1])
    #         if plotting:
    #             plt.contour(X,Y,sdf[i], colors='g', levels=[0])
    #     return sdf

    # def cost_fcn(thk):
    #     sdf = sdf_approx(poses, thk)
    #     boolean_approx_min = sdf.min(axis=0)[0]
    #     return torch.abs(boolean_model_min-boolean_approx_min).sum().numpy()<0


    # # sdf = sdf_approx(poses, 0.0005*torch.ones(m))
    # # boolean_approx_min = sdf.min(axis=0)[0]<0
    # # plt.contour(X,Y,boolean_model_min, colors='k', levels=[0])
    # # plt.contour(X,Y,boolean_approx_min, colors='g', levels=[0])
    # #

    # # from scipy.optimize import minimize
    # init_thinkess=0.0005*torch.ones(poses.shape[0])#+0.0001*torch.rand(poses.shape[0])
    # # cons = ({'type': 'ineq', 'fun': lambda x:  x>0})
    # # res = minimize(cost_fcn, init_thinkess, method='COBYLA', constraints=cons)

    # # sdf = sdf_approx(poses, res.x, plotting=True)

    # sdf = sdf_approx(poses, init_thinkess, plotting=True)
    # boolean_approx_min = sdf.min(axis=0)[0]<0
    # plt.contour(X,Y,boolean_model_min, colors='k', levels=[0])
    # plt.contour(X,Y,boolean_approx_min, colors='b', levels=[0])
    # plt.plot(poses[:,0],poses[:,1],'r',marker='o')
    # plt.show()





    # import matplotlib.pyplot as plt
    # import matplotlib

    # # plotting
    # X,Y=np.meshgrid(x,y,indexing='ij')
    # c=1
    # for sdf_val,   in zip(sdf_vals, sdf_grads):
    #     sdf_val = sdf_val.reshape(len(x), len(y))
    #     sdf_grad = sdf_grad.reshape(len(x), len(y), 3)

    #     plt.subplot(4,5,c)
    #     norm = matplotlib.colors.Normalize(vmin=sdf_val.min(), vmax=sdf_val.max())
    #     cset1 = plt.contourf(X,Y,sdf_val, cmap="Greys")
    #     plt.title("Link"+str(c-1))
    #     # cset2 = plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
    #     plt.colorbar(cset1)
    #     c=c+1
    # plt.show()


def test_overlapping_bodies():

    import matplotlib.pyplot as plt
    import math
    import matplotlib

    print("Using the CPU.")
    device = torch.device("cpu")
    torch.set_num_threads(8)

    N=2**11+1

    x=torch.linspace(-100,100,N)
    y=torch.linspace(-100,100,N)
    x = x.to(device)
    y = y.to(device)

    body1 = BodyAnalytical(
        device,x,y,
        lambda x,y : circle (x,y,xt=-30,yt=0,r=30),
        (lambda i : torch.tensor(1)*i, [lambda i : 0*math.cos(i/10),lambda i : 0*math.sin(i/10)]),
        eps=2*(x[1]-x[0])
    )
    d1, _, _, c1 = body1.initialize()[0]
    mu0_1, mu1_1 = body1.mu_funcs(d1)

    body2 = BodyAnalytical(
        device,x,y,
        lambda x,y : circle (x,y,xt=30,yt=0,r=30),
        (lambda i : torch.tensor(1)*i, [lambda i : 0*math.cos(i/10),lambda i : 0*math.sin(i/10)]),
        eps=2*(x[1]-x[0])
    )
    d2, _, _, c2 = body2.initialize()[0]
    mu0_2, mu1_2 = body2.mu_funcs(d2)

    X=body1.X
    Y=body1.Y

    d = torch.where(body1.phi(d1)>0,1/c1,0) #torch.min(torch.stack([d1,d2]),axis=0)[0]

    plt.figure()
    plt.imshow(
                d.T,
                extent = (
                    torch.min(x.cpu()), torch.max(x.cpu()),
                    torch.min(y.cpu()), torch.max(y.cpu())
                ),
                origin = "lower",
                cmap = "Greys"
            )
    plt.colorbar()
    plt.show()


def test_curvature():

    N=2**10
    x=torch.linspace(-0.001,0.001,N)
    y=torch.linspace(-0.001,0.001,N)

    # mesh_file = "/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_10.obj"
    # body = BodyMesh("cpu", x, y, mesh_file, (lambda t: 0, [lambda t:0, lambda t:0]),eps=2*(x[1]-x[0]),suit=0.0)

    sdf = lambda x,y : circle (x,y,xt=0,yt=0,r=0.0007)
    update = (lambda i : torch.tensor(1)*i, [lambda i : 0*math.cos(i/10),lambda i : 0*math.sin(i/10)])
    body = BodyAnalytical("cpu",x,y,sdf,update,eps=2*(x[1]-x[0]))

    sdf_val = body.initialize()[0]
    sdf_val, du, dv, curv = body.compute_sdf_properties(sdf_val)
    R = torch.where(curv>0,1/curv,0)


    import matplotlib.pyplot as plt
    X,Y=np.meshgrid(x,y,indexing='ij')
    plt.figure()
    plt.imshow(
        sdf_val.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    plt.colorbar()
    plt.contour(X,Y,sdf_val, colors='k', levels=[0], linestyles='dashed')
    subsample_n = 2**6
    plt.quiver(
        X[::subsample_n,::subsample_n],
        Y[::subsample_n,::subsample_n],
        du[::subsample_n,::subsample_n],
        dv[::subsample_n,::subsample_n],
        color='g'
    )
    plt.savefig("sphere_body_example.pdf")


    plt.figure()
    plt.imshow(
        R.T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    plt.colorbar()


    plt.show()


if __name__ == "__main__":
    test_body()




