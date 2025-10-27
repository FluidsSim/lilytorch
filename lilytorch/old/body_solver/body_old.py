
import torch
import numpy as np
import open3d as o3d
from farms_core.io.sdf import ModelSDF
from pytorch_interp import RegularGridInterpolator
import skfmm

"""
Analitical SDFs
"""
def circle(x,y,xt=0,yt=60,r=25):
    return torch.sqrt((x-xt)**2+(y-yt)**2)-r

def box(x,y,xb=20,yb=20):
    qx=torch.abs(x)-xb
    qy=torch.abs(y)-yb
    return torch.sqrt(
        torch.maximum(qx,torch.zeros_like(x))**2 +
        torch.maximum(qy,torch.zeros_like(y))**2
    )+torch.minimum(torch.maximum(qx,qy),torch.zeros_like(x))
    
class mesh2sdf():
    """
    It is assumed that all vector inputs are numpy arrays
    """
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self._mesh = o3d.io.read_triangle_mesh(mesh_file)
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
        # opt.show_coordinate_frame = True
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
            viewer.add_geometry(sdf._mesh)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()



class Body:
    """
    Body class
    """

    def __init__(self, device, x, y, eps=0.05):
        """
        
        """
        self.device=device


        self.x   = x
        self.y   = y
        self.X, self.Y = torch.meshgrid(x,y,indexing="ij")

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
        self.ones_stacked=torch.ones(self.nx*self.ny).to(self.device)

        self.oldpos_u = torch.zeros((self.nx,self.ny),device=self.device)
        self.oldpos_v = torch.zeros((self.nx,self.ny),device=self.device)

        self.body_u          = torch.zeros((self.nx,self.ny),device=self.device)
        self.body_v          = torch.zeros((self.nx,self.ny),device=self.device)

    def compute_sdf_properties(self, sdf_val):

        (gradx, grady) = torch.gradient(sdf_val, spacing=[self.dx, self.dy])

        norm = torch.sqrt(gradx**2+grady**2)

        # compute curvature
        # numerator = (grady**2)*torch.gradient(gradx, spacing=self.dx, axis=0)[0]+(gradx**2)*torch.gradient(grady, spacing=self.dy, axis=1)[0]-2*gradx*grady*torch.gradient(grady, spacing=self.dx, axis=0)[0]
        # denominator = norm**3
        
        numerator = torch.gradient(gradx, dim=0, spacing=self.dx)[0]+torch.gradient(grady, dim=1, spacing=self.dy)[0]
        denominator = (1+gradx**2+grady**2)**2

        curvature = numerator/denominator

        # from IPython import embed; embed()

        # normalize gradients
        
        gradx=torch.where(norm>0, gradx/norm, 0)
        grady=torch.where(norm>0, grady/norm, 0)
        # gradx/=norm
        # grady/=norm 

        return (
                    sdf_val, 
                    gradx, 
                    grady,
                    curvature,
                )


    def sdf_from_function(self, fun = lambda x,y : torch.sqrt(x**2+y**2)-1):

        sdf_val = fun(self.X, self.Y)
        return self.compute_sdf_properties(sdf_val)

        

    def sdf_from_obj(self, mesh_file="box.obj"):
        
        m2s = mesh2sdf(mesh_file)
        xflat = self.xflat.cpu().numpy().astype(np.float32)
        yflat = self.yflat.cpu().numpy().astype(np.float32)
        xflat = self.xflat.cpu().numpy().astype(np.float32)
        yflat = self.yflat.cpu().numpy().astype(np.float32)
        zflat = np.zeros_like(xflat)
        xyz   = np.stack([xflat,yflat,zflat],axis=1)

        query_pts = np.array(xyz,dtype=np.float32)
        sdf_val, sdf_grad= m2s(query_pts)
        sdf_val  = torch.from_numpy(sdf_val).to(self.device).reshape(self.nx, self.ny)
        sdf_grad = torch.from_numpy(sdf_grad).to(self.device)

        # subsample arrows
        gradx = sdf_grad[:,0].reshape(self.nx, self.ny)
        grady = sdf_grad[:,1].reshape(self.nx, self.ny)
        norm  = torch.sqrt(gradx**2+grady**2)

        # compute curvature
        numerator = (grady**2)*torch.gradient(gradx, spacing=self.dx, axis=0)[0] \
                    +(gradx**2)*torch.gradient(grady, spacing=self.dy, axis=1)[0] \
                    -2*gradx*grady*torch.gradient(grady, spacing=self.dx, axis=0)[0]
        denominator = norm**3
        # numerator = torch.gradient(gradx, dim=0, spacing=self.dx)[0]+torch.gradient(grady, dim=1, spacing=self.dy)[0]
        # denominator = (1+gradx**2+grady**2)**2
        curvature = numerator/denominator

        # normalize gradient
        gradx/=norm
        grady/=norm


        return (
            sdf_val, 
            gradx, 
            grady,
            curvature,
        )

        # return sdf_val.reshape(self.nx, self.ny), du.reshape(self.nx, self.ny), dv.reshape(self.nx, self.ny)

    def sdf_from_interp(self,mesh_file="box.obj", nsamples=2**12, msamples=2**12):

        print("Computing the interpolation functions for {}".format(mesh_file))
        m2s = mesh2sdf(mesh_file)
        bb = m2s.bounding_box()

        # (1) use open3d to determine sdf on downsampled data 
        diag = torch.sqrt((self.x[-1]-self.x[0])**2+(self.y[-1]-self.y[0])**2).cpu()
        
        xnp = np.linspace(-diag,diag,nsamples)
        ynp = np.linspace(-diag,diag,msamples)
        diag = torch.sqrt((self.x[-1]-self.x[0])**2+(self.y[-1]-self.y[0])**2).cpu()
        
        xnp = np.linspace(-diag,diag,nsamples)
        ynp = np.linspace(-diag,diag,msamples)
        cx_bb = (bb[0,1]+bb[0,0])/2
        cy_bb = (bb[1,1]+bb[1,0])/2
        dx_bb = bb[0,1]-bb[0,0]
        dy_bb = bb[1,1]-bb[1,0]
        idownsampled = np.where(
            np.logical_and(xnp>cx_bb-dx_bb, xnp<cx_bb+dx_bb)
        )[0]
        xdownsampled = xnp[idownsampled]
        jdownsampled = np.where(
            np.logical_and(ynp>cy_bb-dy_bb, ynp<cy_bb+dy_bb)
        )[0]
        ydownsampled = xnp[jdownsampled]
        X,Y=np.meshgrid(xdownsampled,ydownsampled,indexing="ij")
        xflat = X.flatten()
        yflat = Y.flatten()
        zflat = np.zeros_like(xflat)
        xyz   = np.stack([xflat,yflat,zflat],axis=1)

        query_pts=np.array(xyz.astype(np.float32))

        query_pts=np.array(xyz.astype(np.float32))

        sdf_val, _=m2s(query_pts)
        binary_2d = np.ones((nsamples,msamples))
        binary_2d = np.ones((nsamples,msamples))
        binary_2d[idownsampled[0]:(idownsampled[-1]+1),jdownsampled[0]:(jdownsampled[-1]+1)][sdf_val.reshape(X.shape)<0]=-1

        # (2) use skfmm to determine sdf on the full domain
        sdf_val = skfmm.distance(binary_2d, dx=[self.dx,self.dy])

        self.sdf_interp = RegularGridInterpolator(
            (
                torch.from_numpy(xnp).type(self.dtype).to(self.device),
                torch.from_numpy(ynp).type(self.dtype).to(self.device)
            ), 
            torch.from_numpy(sdf_val).type(self.dtype).to(self.device), 
            fill_value=0.0
        )
        self.sdf_interp = RegularGridInterpolator(
            (
                torch.from_numpy(xnp).type(self.dtype).to(self.device),
                torch.from_numpy(ynp).type(self.dtype).to(self.device)
            ), 
            torch.from_numpy(sdf_val).type(self.dtype).to(self.device), 
            fill_value=0.0
        )

        return self.compute_sdf_from_interp_query(self.xflat,self.yflat) # return the initial values of the sdf parameters
        

    
    def compute_sdf_from_interp_query(self, xquery, yquery):
        return self.compute_sdf_properties(
            self.sdf_interp(xquery,yquery).reshape(self.nx, self.ny)
        )


    def update_interp_from_rototranslation2D(self, theta, transl, dt=1):
        theta = theta*torch.pi/180
        s = torch.sin(torch.tensor(theta, device=self.device))
        c = torch.cos(torch.tensor(theta, device=self.device))
        rot = torch.stack([torch.stack([c, -s]),
                        torch.stack([s, c])]).to(self.device)
        trans = torch.stack((transl[0]*self.ones_stacked, transl[1]*self.ones_stacked))
        newpoints=rot.T@self.stacked_xy-trans

        newpos = rot@self.stacked_xy+trans
        newpos_u = newpos[0].reshape(self.nx, self.ny)
        newpos_v = newpos[1].reshape(self.nx, self.ny)

        # self.body_uprev = self.body_u
        # self.body_vprev = self.body_v

        self.body_u=(newpos_u-self.oldpos_u)/dt
        self.body_v=(newpos_v-self.oldpos_v)/dt

        self.oldpos_u = newpos_u
        self.oldpos_v = newpos_v

        (
            new_sdf, 
            new_nx, 
            new_ny,
            new_curv,
        ) = self.compute_sdf_from_interp_query(newpoints[0], newpoints[1])
    


        return new_sdf, new_nx, new_ny, new_curv
    


    def update_fun_from_function(self, fun, theta, transl, dt=1):

        theta = theta*torch.pi/180
        s = torch.sin(torch.tensor(theta, device=self.device))
        c = torch.cos(torch.tensor(theta, device=self.device))
        rot = torch.stack([torch.stack([c, -s]),
                        torch.stack([s, c])]).to(self.device)
        trans = torch.stack((transl[0]*self.ones_stacked, transl[1]*self.ones_stacked))
        newpoints=rot.T@self.stacked_xy-trans
        # vel=(-newpoints+self.stacked_xy)/dt

        newpos = rot@self.stacked_xy+trans
        newpos_u = newpos[0].reshape(self.nx, self.ny)
        newpos_v = newpos[1].reshape(self.nx, self.ny)

        # self.body_uprev = self.body_u
        # self.body_vprev = self.body_v

        self.body_u=(newpos_u-self.oldpos_u)/dt
        self.body_v=(newpos_v-self.oldpos_v)/dt

        self.oldpos_u = newpos_u
        self.oldpos_v = newpos_v

        sdf_val = fun(newpoints[0].reshape(self.nx, self.ny), newpoints[1].reshape(self.nx, self.ny))
    
        (
            new_sdf, 
            new_nx, 
            new_ny,
            new_curv,
        ) = self.compute_sdf_properties(sdf_val)

        return new_sdf, new_nx, new_ny, new_curv
        
    
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




















def test_single_mesh():
    mesh_file="/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_1.obj"
    # mesh_file="box.obj"

    m2s = mesh2sdf(mesh_file)

    dtype = np.float32

    # compute sdf on query points
    n=2**10
    x=np.linspace(-1,1,n,dtype=dtype)
    y=np.linspace(-1,1,n,dtype=dtype)
    X,Y=np.meshgrid(x,y,indexing="ij")
    xflat = X.flatten()
    yflat = Y.flatten()
    zflat = np.zeros_like(xflat)
    xflat = xflat.astype(dtype)
    yflat = yflat.astype(dtype)
    xyz   = np.stack([xflat,yflat,zflat],axis=1)

    query_pts=np.array(xyz,dtype=dtype)


    # query_pts=np.array(list(it.product(x,y,[0.0])),dtype=dtype)
    sdf_val, sdf_grad=m2s(query_pts)
    

    sdf_val = sdf_val.reshape(len(x), len(y))
    sdf_grad = sdf_grad.reshape(len(x), len(y), 3)

    du = sdf_grad[:,:,0]
    dv = sdf_grad[:,:,1]
    norm = np.sqrt(du**2+dv**2)
    du/=norm
    dv/=norm

    # plotting
    import matplotlib.pyplot as plt
    import matplotlib
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


def test_body_interpolation():

    use_gpu=True

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)



    N=2**11+1

    x=torch.linspace(-0.01,0.01,N)
    y=torch.linspace(-0.01,0.01,N)
    filename = "/data/andreaferrario/zebrafish/models/zebrafish_v1_triangulated/sdf/meshes_zebrafish/link_1.obj"

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
    body = Body(device,x,y,eps=2*(x[1]-x[0]))
    d, nx, ny, curv = body.sdf_from_interp(mesh_file=filename) # build interpolation on the domain X,Y (larger than the query domain)
    # 
    # d, nx, ny, curv = body.sdf_from_obj(mesh_file=filename)

    # d, nx, ny, curv = body.sdf_from_function(box)


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
    plt.savefig("figures/box_1.png")


    # d, nx, ny, curv = body.update_fun_from_function(box, torch.tensor(45), [100,0])
    d, nx, ny, curv = body.update_interp_from_rototranslation2D(torch.tensor(30),[0.005,0.00])
    (mu0, mu1) = body.mu_funcs(d)


    X=body.X.cpu()
    Y=body.Y.cpu()
    d=d.cpu()
    nx=nx.cpu()
    ny=ny.cpu()
    curv=curv.cpu()
    mu0=mu0.cpu()
    mu1=mu1.cpu()    
    
    plt.figure()
    cset1 = plt.contourf(X,Y, d, cmap="Greys")
    plt.colorbar(cset1)
    plt.contour(X,Y, d, colors='k', levels=[0], linestyles='dashed')
    subsample_n = 2**7
    plt.quiver(
        X[::subsample_n,::subsample_n],
        Y[::subsample_n,::subsample_n],
        nx[::subsample_n,::subsample_n],
        ny[::subsample_n,::subsample_n], 
        color='g'
    )
    plt.savefig("figures/box_2.png")
    
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

    x=torch.linspace(-100,100,N)
    y=torch.linspace(-100,100,N)
    x = x.to(device)
    y = y.to(device)


    N=2**10+1
    body = Body(device,x,y,eps=2*(x[1]-x[0]))

    fun = lambda x,y : circle (x,y,xt=0,yt=0,r=25)
    # fun = lambda x,y : box (x,y)

    d0, nx, ny, curv = body.sdf_from_function(fun)


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

    subsample_n = 2**8
    sct=plt.scatter(body.X[::subsample_n,::subsample_n].flatten(),body.Y[::subsample_n,::subsample_n].flatten())


    def init():
        d, nx, ny, curv = body.sdf_from_function(fun)
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
        return [im,ctr,ctr0]


    global X0, Y0
    X0=body.X
    Y0=body.Y
    def animate(i):
        global X0, Y0
        d, nx, ny, curv = body.update_fun_from_function(fun, torch.tensor(0)*i, [0,30*math.sin(i)])
        im.set_array(d.T)
        ctr = plt.contour(body.X,body.Y, d, colors='k', levels=[0], linestyles='dashed')
        ctr0 = plt.contour(body.X,body.Y, d0, colors='k', levels=[0], linestyles='dashed')        
        # quiv = plt.quiver(
        #     body.X[::subsample_n,::subsample_n],
        #     body.Y[::subsample_n,::subsample_n],
        #     body.body_u.cpu()[::subsample_n,::subsample_n]-body.body_uprev.cpu()[::subsample_n,::subsample_n],
        #     body.body_v.cpu()[::subsample_n,::subsample_n]-body.body_vprev.cpu()[::subsample_n,::subsample_n], 
        #     color='g'
        # )
        u_=body.body_u
        v_=body.body_v
        dt=1
        X=X0+u_*dt
        Y=Y0+v_*dt
        sct.set_offsets(
            torch.stack((
                    X[::subsample_n,::subsample_n].flatten(),
                    Y[::subsample_n,::subsample_n].flatten()
                )
            ).T
        )
        X0=X
        Y0=Y

        return [im,ctr,ctr0,sct]
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    animation = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=30, interval=0, blit=True)
    
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

    d, nx, ny, curv = body.sdf_from_obj(mesh_file="cylinder.obj")
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


if __name__ == "__main__":
    test_single_mesh()




