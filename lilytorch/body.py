
import os
import torch
import numpy as np
import open3d as o3d
try:
    from farms_core.io.sdf import ModelSDF
except:
    print("farms_core not installed")
from pytorch_interp import RegularGridInterpolator
import skfmm
import math # important to keep this for evaluating math operations for sdfs even if it appears as not used
import matplotlib.pyplot as plt
import cv2

from lilytorch.scripts.zebrafish_files.load_data import get_experimental_signal

"""
Analitical SDFs
"""
def circle(x,y,xt=0,yt=60,r=25):
    return torch.sqrt((x-xt)**2+(y-yt)**2)-r

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


def body_from_yaml(device, x, y, body_pars, eps=0.05, costum_update=None, starting_time=0, **kwargs):

    if costum_update is not None:
        update_map = costum_update
    else:
        update_maps = body_pars["update_maps"]
        update_map = (
            eval(update_maps["rotation"]),
            (eval(update_maps["translation"][0]),eval(update_maps["translation"][1]))
        )

    type = body_pars["type"]
    if type == "analytical":
        sdf_fun = eval(body_pars["sdf"])
        return BodyAnalytical(
            device,
            x, y,
            sdf_fun,
            update_map,
            eps=eps
        )

    elif type == "mesh":
        mesh_file = body_pars["mesh_file"]
        (nsamples,msamples) = eval(body_pars["n_samples"])
        return BodyMesh(
            device,
            x, y,
            mesh_file,
            update_map,
            eps=eps,
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
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self._mesh = o3d.io.read_triangle_mesh(self.mesh_file)
        self.update_mesh()

    def update_mesh(self, convexify=False):
        if convexify:
            self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh.compute_convex_hull()[0])
        else:
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

    def __init__(self, device, x, y, eps=0.05, **kwargs):
        """

        """
        self.device=device
        self.dtype = x.dtype

        self.x   = x
        self.y   = y
        self.X, self.Y = torch.meshgrid(x,y,indexing="ij")
        self.nx  = len(x)
        self.ny  = len(y)
        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])
        self.eps = eps
        self.dtype = x.dtype

        self.xflat = self.X.flatten()
        self.yflat = self.Y.flatten()
        self.stacked_xy = torch.stack((self.xflat,self.yflat))
        self.ones_stacked=torch.ones((self.nx*self.ny),device=self.device,dtype=self.dtype)

        self.oldpos_u = torch.zeros((self.nx,self.ny),device=self.device)
        self.oldpos_v = torch.zeros((self.nx,self.ny),device=self.device)

        # body velocities
        self.body_u = torch.zeros((self.nx,self.ny),device=self.device)
        self.body_v = torch.zeros((self.nx,self.ny),device=self.device)
        self.old_points = self.stacked_xy.clone().detach()
        self.rad_conv = (torch.pi/180)


    def compute_sdf_properties(self, sdf_val):

        (gradx, grady) = torch.gradient(sdf_val, spacing=[self.dx, self.dy])
        norm = torch.sqrt(gradx**2+grady**2)

        curvature=torch.where(
            norm>0,
            (torch.gradient(gradx, spacing=self.dx, axis=0)[0]*grady-
             torch.gradient(grady, spacing=self.dy, axis=1)[0]*gradx)/
            norm**3,
            0
        )

        # curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5


        # compute curvature
        numerator = (
            (grady**2)*torch.gradient(gradx, spacing=self.dx, axis=0)[0]+
            (gradx**2)*torch.gradient(grady, spacing=self.dy, axis=1)[0]+
            -2*gradx*grady*torch.gradient(grady, spacing=self.dx, axis=0)[0]
        )
        denominator = norm**3
        curvature = torch.where(denominator>0, numerator/denominator, 0)



        # # compute curvature
        # numerator = (
        #     (grady**2)*torch.gradient(gradx, spacing=self.dx, axis=0)[0]+
        #     (gradx**2)*torch.gradient(grady, spacing=self.dy, axis=1)[0]+
        #     -2*gradx*grady*torch.gradient(grady, spacing=self.dx, axis=0)[0]
        # )
        # denominator = norm**3
        # curvature = torch.where(denominator>0, numerator/denominator, 0)


        # dx_dt   = np.gradient(com_x)
        # dy_dt   = np.gradient(com_y)
        # d2x_dt2 = np.gradient(dx_dt)
        # d2y_dt2 = np.gradient(dy_dt)
        # curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5


        # numerator = torch.gradient(gradx, dim=0, spacing=self.dx)[0]+torch.gradient(grady, dim=1, spacing=self.dy)[0]
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
        return torch.where(
            torch.abs(d)<self.eps,
            ( 1 + torch.cos(torch.pi*d/self.eps) )/( 2*self.eps ),
            0
        )

    def mu_funcs(self, d):
        s=torch.sin(torch.pi*d/self.eps)
        c=torch.cos(torch.pi*d/self.eps)
        mu_0_eps = torch.where(
            d<=-self.eps,
            0,
            torch.where(
                d>=self.eps,
                1,
                0.5*( 1 + d/self.eps + s/torch.pi )
            )
        )

        mu_1_eps = torch.where(
            torch.abs(d)>=self.eps,
            0,
            self.eps*( 0.25 - (d/(2*self.eps))**2 - ( d*s/self.eps+(1+c)/torch.pi )/(2*torch.pi) )
        )
        return (mu_0_eps, mu_1_eps)



    def update_body(self, fun, theta, transl, dt=1):
        """
        Update sdf properties from analytical rototranslation map
        """
        theta = torch.tensor(theta*self.rad_conv, device=self.device, dtype=self.dtype).clone().detach()
        s = torch.sin(theta)
        c = torch.cos(theta)
        rot = torch.stack([torch.stack([c, s]),
                        torch.stack([-s, c])])
        trans = torch.stack((transl[0]*self.ones_stacked, transl[1]*self.ones_stacked))

        # newpoints=rot.T@self.stacked_xy-trans

        newpoints=self.stacked_xy-trans
        newpoints=rot@newpoints

        # newpos = self.stacked_xy+trans
        # newpos = rot@newpos

        # newpos=rot@self.stacked_xy+trans

        vel = - rot.T @ (newpoints - self.old_points) / dt

        newpos_u = newpoints[0].reshape(self.nx, self.ny)
        newpos_v = newpoints[1].reshape(self.nx, self.ny)

        # self.body_uprev = self.body_u
        # self.body_vprev = self.body_v

        self.body_u= vel[0].reshape(self.nx, self.ny)
        self.body_v= vel[1].reshape(self.nx, self.ny)

        self.old_points = newpoints

        # self.oldpos_u = newpos_u
        # self.oldpos_v = newpos_v

        # from IPython import embed; embed()

        return fun(newpos_u, newpos_v)



class BodyAnalytical(Body):

    def __init__(self, device, x, y, sdf_fun, update_maps, eps=0.05):
        super().__init__(device, x, y, eps=eps)
        self.sdf_fun = sdf_fun
        self.update_theta = update_maps[0]
        self.update_translation = update_maps[1]
        # self.bodies = [self]
        self.body=self
        self.initialize()

    def initialize(self):
        """
        Initialize sdf properties at time 0
        """
        return self.update(0)

    def update(self, t, dt=1):
        """
        Apply rototranslation and update the sdf properties
        """
        self.sdf=self.update_body(
            self.sdf_fun,
            self.update_theta(t),
            (
                self.update_translation[0](t),
                self.update_translation[1](t)
            ),
            dt=dt
        )

        return [self.update_body(
            self.sdf_fun,
            self.update_theta(t),
            (
                self.update_translation[0](t),
                self.update_translation[1](t)
            ),
            dt=dt
        )]

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

    def update(self, t, dt=1):
        """
        Update sdf properties from analytical rototranslation map
        """
        s = self.XC.clamp(0,self.L)
        new_x = self.XC
        new_y = self.YC+self.A*self.envelope(s/self.L)*torch.sin(2*torch.pi*(self.wavefrequency*s/self.L-self.f*t))

        self.body_u=0
        self.body_v=-(new_y-self.oldpos_v)/dt

        self.oldpos_v=new_y

        return [self.compute_sdf_properties(self.sdf_fun(new_x,new_y))]

    def initialize(self):
        """
        Initialize sdf properties at time 0
        """
        return self.update(0)


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
    def __init__(self, device, x, y, mesh_file, update_maps, eps=0.05, compute_interp=True, nsamples=2**12, msamples=2**12, suit=0, plotting_meshes=False, **kwargs):
        super().__init__(device, x, y, eps=eps)
        self.mesh_file           = mesh_file
        self.compute_interp      = compute_interp
        self.nsamples            = nsamples
        self.msamples            = msamples
        self.update_theta        = update_maps[0]
        self.update_translation  = update_maps[1]
        self.suit                = suit
        self.plotting            = plotting_meshes
        self.apply_closing_morph = kwargs.pop("apply_closing_morph", True)
        self.m2s                 = mesh2sdf(self.mesh_file)
        self.initialize_sdfs()
        del self.m2s
        self.bodies = [self]


    def initialize_sdfs(self):
        """
        Initialize the sdf interpolation function
        """
        self.bb = self.m2s.bounding_box()
        if self.compute_interp:
            # bb = self.m2s.bounding_box()
            # # bb = self.m2s.bounding_box(self.suit)

            # diag = torch.sqrt((self.x[-1]-self.x[0])**2+(self.y[-1]-self.y[0])**2).cpu().detach().numpy()
            # xnp = np.linspace(-diag,diag,self.nsamples)
            # ynp = np.linspace(-diag,diag,self.msamples)

            # cx_bb = (bb[0,1]+bb[0,0])/2
            # cy_bb = (bb[1,1]+bb[1,0])/2
            # # dx_bb = bb[0,1]-bb[0,0]
            # # dy_bb = bb[1,1]-bb[1,0]
            # diag_bb=np.sqrt((bb[0,0]-bb[0,1])**2+(bb[1,0]-bb[1,1])**2)

            # idownsampled = np.where(
            #     np.logical_and(xnp>cx_bb-diag_bb, xnp<cx_bb+diag_bb)
            # )[0]
            # xdownsampled = xnp[idownsampled]
            # jdownsampled = np.where(
            #     np.logical_and(ynp>cy_bb-diag_bb, ynp<cy_bb+diag_bb)
            # )[0]
            # ydownsampled = xnp[jdownsampled]
            # X,Y=np.meshgrid(xdownsampled,ydownsampled,indexing="ij")
            # xflat = X.flatten()
            # yflat = Y.flatten()
            # zflat = np.zeros_like(xflat)
            # xyz   = np.stack([xflat,yflat,zflat],axis=1)
            # query_pts=np.array(xyz.astype(np.float32))


            # sdf_val, _=self.m2s(query_pts)
            # binary_2d = np.ones((self.nsamples,self.msamples))
            # binary_2d[idownsampled[0]:(idownsampled[-1]+1),jdownsampled[0]:(jdownsampled[-1]+1)][sdf_val.reshape(X.shape)<0]=-1


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


                if self.plotting:
                    cv2.imshow("window_name", im)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                im=im[:,:,0]
            else:
                im=binary_2d

            binary_2d=np.where(im==0,1,-1)

            # (2) use skfmm to determine sdf on the full domain
            print("Computing the sdf for {}, with space steps ({},{})".format(self.mesh_file,xnp[1]-xnp[0],ynp[1]-ynp[0]))
            sdf_val = skfmm.distance(binary_2d, dx=[xnp[1]-xnp[0],ynp[1]-ynp[0]])#-self.suit

            if self.plotting:
                self.m2s.visualize()
                var=sdf_val #sdf_val_o3d.reshape(X.shape)
                plt.figure()
                plt.contourf(
                    var
                )
                plt.colorbar()
                plt.contour(var, colors='k', levels=[0], linestyles='dashed')
                plt.show()


            print("Computing the interpolation functions for {}".format(self.mesh_file))

            np.save("interp_data/xnp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",xnp)
            np.save("interp_data/ynp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",ynp)
            np.save("interp_data/sdf_val_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",sdf_val)


        # if self.compute_interp:
        #     bb = self.m2s.bounding_box()
        #     # bb = self.m2s.bounding_box(self.suit)

        #     # (1) use open3d to determine sdf on downsampled data
        #     diag = torch.sqrt((self.x[-1]-self.x[0])**2+(self.y[-1]-self.y[0])**2).cpu().detach().numpy()
        #     xnp = np.linspace(-2*diag,2*diag,self.nsamples)
        #     ynp = np.linspace(-2*diag,2*diag,self.msamples)

        #     cx_bb = (bb[0,1]+bb[0,0])/2
        #     cy_bb = (bb[1,1]+bb[1,0])/2
        #     dx_bb = bb[0,1]-bb[0,0]
        #     dy_bb = bb[1,1]-bb[1,0]
        #     idownsampled = np.where(
        #         np.logical_and(xnp>cx_bb-dx_bb, xnp<cx_bb+dx_bb)
        #     )[0]
        #     xdownsampled = xnp[idownsampled]
        #     jdownsampled = np.where(
        #         np.logical_and(ynp>cy_bb-dy_bb, ynp<cy_bb+dy_bb)
        #     )[0]
        #     ydownsampled = xnp[jdownsampled]
        #     X,Y=np.meshgrid(xdownsampled,ydownsampled,indexing="ij")
        #     xflat = X.flatten()
        #     yflat = Y.flatten()
        #     zflat = np.zeros_like(xflat)
        #     xyz   = np.stack([xflat,yflat,zflat],axis=1)

        #     query_pts=np.array(xyz.astype(np.float32))

        #     sdf_val_o3d, _=self.m2s(query_pts)
        #     binary_2d = np.ones((self.nsamples,self.msamples))
        #     binary_2d = np.ones((self.nsamples,self.msamples))
        #     binary_2d[idownsampled[0]:(idownsampled[-1]+1),jdownsampled[0]:(jdownsampled[-1]+1)][sdf_val_o3d.reshape(X.shape)<0]=-1


        #     var=sdf_val_o3d.reshape(X.shape)
        #     plt.figure()
        #     plt.contourf(
        #         var
        #     )
        #     plt.colorbar()
        #     plt.contour(var, colors='k', levels=[0], linestyles='dashed')
        #     plt.show()


        #     # from IPython import embed; embed()

        #     # (2) use skfmm to determine sdf on the full domain
        #     print("Computing the sdf for {}, with space steps ({},{})".format(self.mesh_file,xnp[1]-xnp[0],ynp[1]-ynp[0]))
        #     sdf_val = skfmm.distance(binary_2d, dx=[xnp[1]-xnp[0],ynp[1]-ynp[0]])#-self.suit


        #     print("Computing the interpolation functions for {}".format(self.mesh_file))

        #     np.save("interp_data/xnp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",xnp)
        #     np.save("interp_data/ynp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",ynp)
        #     np.save("interp_data/sdf_val_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy",sdf_val)

    def initialize(self):
        xnp = np.load("interp_data/xnp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        ynp = np.load("interp_data/ynp_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")
        sdf_val = np.load("interp_data/sdf_val_"+self.mesh_file.split('/')[-1].split('.')[0]+".npy")

        self.sdf_interp = RegularGridInterpolator(
            (
                torch.from_numpy(xnp).type(self.dtype).to(self.device),
                torch.from_numpy(ynp).type(self.dtype).to(self.device)
            ),
            torch.from_numpy(sdf_val).type(self.dtype).to(self.device),
            fill_value="nearest"
        )
        return self.update(0)

    def update(self, t, dt=1):
        return [self.update_body(
            self.sdf_interp,
            self.update_theta(t),
            (
                self.update_translation[0](t),
                self.update_translation[1](t)
            ),
            dt=dt
        )]

    def visualize(self):
        self.m2s.visualize()


class CompositeBodyMesh:

    def __init__(self, device, x, y, sdf_folder, sdf_name, costum_update, eps=0.05, compute_interp=True, nsamples=2**12, msamples=2**12, plotting=False, plotting_meshes=False, suit=0.0, **kwargs):
        """
        sdf_folder = folder of the sdf file
        sdf_name = name of the sdf file
        """
        self.sdf_folder      = sdf_folder
        self.sdf             = ModelSDF.read(sdf_folder+sdf_name)[0]
        self.bodies          = []
        self.suit            = suit
        self.plotting        = plotting
        self.plotting_meshes = plotting_meshes
        for link_i, link in enumerate(self.sdf.links):
            mesh_name = link["visuals"][0]["geometry"]["uri"]
            mesh_gpath = sdf_folder+mesh_name
            initial_pose = np.array(link.pose).astype(np.float32)
            update_funcs = (
                lambda t: 180,
                [
                    lambda t, initial_pose=initial_pose: -initial_pose[0],
                    lambda t, initial_pose=initial_pose: -initial_pose[1],
                ]
                )
            # if link_i == 7:
            #     compute_interp = True
            # else:
            #     compute_interp = False

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
        self.costum_update = costum_update
        self.compute_interp = compute_interp

        self.mu_funcs               = self.bodies[0].mu_funcs
        self.compute_sdf_properties = self.bodies[0].compute_sdf_properties
        nbodies                     = len(self.sdf.links)
        self.sdf_vals               = torch.zeros((nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.u_vals                 = torch.zeros((nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.v_vals                 = torch.zeros((nbodies,self.bodies[0].nx,self.bodies[0].ny),device=device)
        self.com_pos                = torch.zeros((nbodies,2),device=device)

        self.initialize() # initialize the sdf interpolation functions


    def initialize(self):

        for i, body in enumerate(self.bodies):
            self.sdf_vals[i]=body.initialize()[0]

            # plt.contour(self.bodies[i].sdf_interp.F.cpu(), colors='k', levels=[0], linestyles='dashed')
            # plt.show()

        self.sdf_val = torch.min(self.sdf_vals,axis=0)[0]-self.suit

        if self.plotting:
            var=self.sdf_val.cpu()
            extent = (
                torch.min(self.bodies[0].x.cpu()), torch.max(self.bodies[0].x.cpu()),
                torch.min(self.bodies[0].y.cpu()), torch.max(self.bodies[0].y.cpu())
            )

        # if self.compute_interp:
            """
            visualize computed interpolation functions over the domain
            """
            plt.figure(figsize=(20,10))
            plt.imshow(
                var.T,
                extent = extent,
                origin = "lower",
                interpolation=None
            )
            plt.contour(self.bodies[0].X.cpu(),self.bodies[0].Y.cpu(),var, colors='k', levels=[0])
            plt.show()

        self.body_u=torch.zeros_like(self.bodies[0].X)
        self.body_v=torch.zeros_like(self.bodies[0].X)



    def update(self, t, dt=1):
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
    #     numerator = (grady**2)*torch.gradient(gradx, spacing=self.dx, axis=0)[0] \
    #                 +(gradx**2)*torch.gradient(grady, spacing=self.dy, axis=1)[0] \
    #                 -2*gradx*grady*torch.gradient(grady, spacing=self.dx, axis=0)[0]
    #     denominator = norm**3
    #     # numerator = torch.gradient(gradx, dim=0, spacing=self.dx)[0]+torch.gradient(grady, dim=1, spacing=self.dy)[0]
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

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    N=2**7+1
    x=torch.linspace(-60,180,N)
    y=torch.linspace(-60,180,N)
    X,Y=torch.meshgrid(x,y,indexing="ij")

    x = x.to(device)
    y = y.to(device)

    body = Body(device,x,y,eps=2*(x[1]-x[0]))

    d = body.sdf_from_obj(mesh_file="cylinder.obj")
    d, nx, ny, curv = body.compute_sdf_properties(d)
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


    # from IPython import embed; embed()




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
    test_curvature()




