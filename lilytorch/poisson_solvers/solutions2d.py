
from tkinter import E
import torch
import numpy as np

"""
2d functions and solutions of the poisson equation
"""
dtype=torch.float64

def sincos_f(N,device=torch.device("cpu")):

    x=torch.linspace(0,1,N,device=device)
    y=torch.linspace(-0.5,0.5,N,device=device)
    [X,Y]=torch.meshgrid(x,y)
    c=torch.ones((N,N),device=device)
    f=-(torch.sin(np.pi*X)*torch.cos(np.pi*Y)+torch.sin(5*np.pi*X)*torch.cos(5*np.pi*Y))
    f[0,:]=0
    f[-1,:]=0
    f[:,0]=0
    f[:,-1]=0

    u_exact = (1/(2*np.pi*np.pi))*torch.sin(np.pi*X)*torch.cos(np.pi*Y)+(1/(50*np.pi*np.pi))*torch.sin(5*np.pi*X)*torch.cos(5*np.pi*Y)
    return X, Y, u_exact, f, c, c, c

def sine_f(N,device=torch.device("cpu")):

    x=torch.linspace(0,1,N,device=device)
    y=torch.linspace(-0.5,0.5,N,device=device)
    [X,Y]=torch.meshgrid(x,y)
    c=torch.ones((N,N),device=device)
    f=-2*(np.pi**2)*torch.sin(np.pi*X)*torch.cos(np.pi*Y)
    f[0,:]=0
    f[-1,:]=0
    f[:,0]=0
    f[:,-1]=0

    u_exact = torch.sin(np.pi*X)*torch.cos(np.pi*Y)
    return X, Y, u_exact, f, c, c, c

def quadratic(N,device=torch.device("cpu")):

    x=torch.linspace(0,1,N,device=device)
    y=torch.linspace(0,1,N,device=device)
    [X,Y]=torch.meshgrid(x,y)
    c=torch.ones((N,N),device=device)

    u_exact = X*Y*(1-X)*(1-Y)

    f=2*X*(X-1)+2*Y*(Y-1)
    f[0,:]=0
    f[-1,:]=0
    f[:,0]=0
    f[:,-1]=0

    return X, Y, u_exact, f, c, c, c

def exp_f(N,device=torch.device("cpu")):
    """
    Care: this is with diriclet BCs!!!
    """

    x=torch.linspace(-1,1,N,device=device)
    y=torch.linspace(-1,1,N,device=device)
    [X,Y]=torch.meshgrid(x,y)
    c=torch.ones((N,N),device=device)

    xhat=0.25
    yhat=-0.5
    u_exact = torch.exp(-(X-xhat)**2-(Y-yhat)**2)

    f=-(4*((X-xhat)**2+(Y-yhat)**2)*u_exact+2*u_exact)
    f[0,:]=0
    f[-1,:]=0
    f[:,0]=0
    f[:,-1]=0

    return X, Y, u_exact, f, c, c, c

def pyro(N,device=torch.device("cpu")):

    x=torch.linspace(0,1,N,device=device)
    y=torch.linspace(0,1,N,device=device)
    [X,Y]=torch.meshgrid(x,y)
    c=torch.ones((N,N),device=device)


    f=(-2.0 * ((1.0 - 6.0 * X**2) * Y**2 * (1.0 - Y**2) +
                   (1.0 - 6.0 * Y**2) * X**2 * (1.0 - X**2)))
    f[0,:]=0
    f[-1,:]=0
    f[:,0]=0
    f[:,-1]=0

    u_exact = (X**2-X**4)*(Y**4-Y**2)

    return X, Y, u_exact, f, c, c, c

def lilypad(N,device=torch.device("cpu")):
    x=torch.linspace(0,N,N+1,device=device)
    y=torch.linspace(0,N,N+1,device=device)
    [X,Y]=torch.meshgrid(x,y)

    u = torch.ones((N,N),device=device)
    v = -0.5*torch.ones((N,N),device=device)
    c = torch.ones((N,N),device=device)

    u[40:50,40:75]=0

    def compute_dpdx(p, dx):
        dpdx = torch.zeros_like(p)
        dpdx[1:-1,:] = (p[2:,:]-p[1:-1,:])/dx
        return dpdx

    def compute_dpdy(p, dy):
        dpdy = torch.zeros_like(p)
        dpdy[:,1:-1] = (p[:,2:]-p[:,1:-1])/dy
        return dpdy

    def divergence(u, v, dx, dy):
        return compute_dpdx(u, dx)+compute_dpdy(v, dy)

    f=divergence(u, v, 1, 1)

    return X, Y, f, f, c, c, c

def variable_coeff(N,device=torch.device("cpu"),create_box=False):

    x=torch.linspace(-1,1,N,device=device)
    y=torch.linspace(-1,1,N,device=device)
    [X,Y]=torch.meshgrid(x,y, indexing="ij")

    from body import BodyMesh, BodyAnalytical, circle
    import sdf
    import trimesh
    from matplotlib import pyplot

    name="box.obj"
    if create_box:
        # f=sdf.sphere((0.5))
        f=sdf.box((0.5, 0.5, 2))
        f.save(name, samples=2**16)
        mesh = trimesh.load_mesh(name)
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces, Z=mesh.vertices[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        pyplot.show()


    body = BodyMesh( device, x, y, name, (lambda t: 0,(lambda t: 0,lambda t: 0)),eps=2/N,nsamples=2**12, msamples=2**12,compute_interp=True)
    body.initialize()
    sdf_fun=body.sdf_interp

    # body = BodyAnalytical( device, x, y, lambda x, y: circle(x,y,xt=0.5,yt=0.5,r=0.25), (lambda t: 0,(lambda t: 0,lambda t: 0)),eps=1/N)
    # body.initialize()
    # sdf_fun =body.sdf_fun

    d = sdf_fun(X,Y)
    (mu0, mu1) = body.mu_funcs(d)

    var=mu0
    pyplot.figure()
    pyplot.imshow(
        var.cpu().T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    pyplot.contour(X.cpu(),Y.cpu(),var.cpu(), colors='k', levels=[0], linestyles='-')
    pyplot.show()

    c=mu0
    f=torch.ones_like(c)
    u_exact = None

    xmid = (x[1:]+x[:-1])/2
    ymid = (y[1:]+y[:-1])/2
    [X_h,Y_h]=torch.meshgrid(xmid,y,indexing="ij")
    [X_v,Y_v]=torch.meshgrid(x,ymid,indexing="ij")

    d_h=sdf_fun(X_h,Y_h)
    d_v=sdf_fun(X_v,Y_v)

    c_h=body.mu_funcs(d_h)[0]
    c_v=body.mu_funcs(d_v)[0]


    return X, Y, f, f, c, c_h, c_v

def variable_coeff_c_hat(N,device=torch.device("cpu"),create_box=False):

    x=torch.linspace(-1,1,N,device=device,dtype=dtype)
    y=torch.linspace(-1,1,N,device=device,dtype=dtype)
    [X,Y]=torch.meshgrid(x,y, indexing="ij")

    from body import BodyMesh, BodyAnalytical, circle
    import sdf
    import trimesh
    from matplotlib import pyplot

    # name="box.obj"
    # if create_box:
    #     f=sdf.sphere((0.5))
    #     # f=sdf.box((0.5, 0.5, 2))
    #     f.save(name, samples=2**16)
    #     mesh = trimesh.load_mesh(name)
    #     fig = pyplot.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], triangles=mesh.faces, Z=mesh.vertices[:,2])
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     pyplot.show()

    # body = BodyMesh( device, x, y, name, (lambda t: 0,(lambda t: 0,lambda t: 0)),eps=2/N,nsamples=2**12, msamples=2**12)
    # body.initialize()
    # sdf_fun=body.sdf_interp

    body = BodyAnalytical( device, x, y, lambda x, y: circle(x,y,xt=0.1,yt=0.,r=0.5), (lambda t: 0,(lambda t: 0,lambda t: 0)),eps=5/N)
    body.initialize()
    sdf_fun =body.sdf_fun

    
    d = sdf_fun(X,Y)
    (mu0, mu1) = body.mu_funcs(d)



    c=mu0

    print(c.dtype)
    ks=0
    kg=1
    sigma=1000
    c=ks+(kg-ks)*(torch.tanh(sigma*d)+1)/2

    var=c
    pyplot.figure()
    pyplot.imshow(
        var.cpu().T,
        extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        ),
        origin = "lower",
        cmap = "Greys"
    )
    pyplot.contour(X.cpu(),Y.cpu(),var.cpu(), colors='k', levels=[0], linestyles='-')
    pyplot.show()




    f=torch.ones_like(c)

    xmid = (x[1:]+x[:-1])/2
    ymid = (y[1:]+y[:-1])/2
    [X_h,Y_h]=torch.meshgrid(xmid,y,indexing="ij")
    [X_v,Y_v]=torch.meshgrid(x,ymid,indexing="ij")

    d_h=sdf_fun(X_h,Y_h)
    d_v=sdf_fun(X_v,Y_v)

    c_h=ks+(kg-ks)*(torch.tanh(sigma*d_h)+1)/2
    c_v=ks+(kg-ks)*(torch.tanh(sigma*d_v)+1)/2



    R=0.25
    circle=torch.sqrt((X)**2+(Y)**2)
    # u_exact = torch.where(
    #     circle<R,
    #     (1/8-R**2*(1-1/ks)/4)-(1/(4*ks))*(circle**2),
    #     1/8-circle**2/4
    # )

    u_exact = c


    return X, Y, u_exact, f, c, c_h, c_v

def multigrid_course(N,device=torch.device("cpu")):
    """
    Multigrid course
    """

    from poisson_2nd_order_2 import PoissonSolver
    h=1/(N-1) # because xh=xl=yh=yl=1 (grid is [0,1]x[0,1])
    h2=h*h
    x=torch.linspace(0,1,N,device=device,dtype=dtype)
    y=torch.linspace(0,1,N,device=device,dtype=dtype)


    [X,Y]=torch.meshgrid(x,y, indexing="ij")

    u_exact=torch.zeros((N+2,N+2),device=device,dtype=dtype)
    u_exact[1:-1,1:-1]=torch.exp(torch.sin(2.0*torch.pi*X)*torch.sin(2.0*torch.pi*Y))-1

    c=torch.ones((N,N),device=device,dtype=dtype)
    solver = PoissonSolver(
        dtype,
        device,
        h,
        verbose=True,
        max_cycles=100,
        nsmoothing=3,
        tol=1e-14,
        w=0.6
    )
    solver.BC(u_exact)
    f, _=solver.FD_operator(u_exact, c, h2)
    return X, Y, u_exact, f, c, c, c
