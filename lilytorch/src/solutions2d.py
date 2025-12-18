
import torch

"""
2d functions and solutions of the poisson equation
"""

def c_half(c):
    ch = (c[1:,1:-1]+c[:-1,1:-1])*0.5
    cv = (c[1:-1,1:]+c[1:-1,:-1])*0.5
    return ch, cv

def grid(N, xlim, ylim, device, dtype):
    h=(xlim[1]-xlim[0])/N
    x=torch.linspace(xlim[0]-h/2,xlim[1]+h/2,N+2,device=device,dtype=dtype)
    y=torch.linspace(ylim[0]-h/2,ylim[1]+h/2,N+2,device=device,dtype=dtype)
    [X,Y]=torch.meshgrid(x,y, indexing="ij")
    return X,Y

##### ========= Example solutions ========= #####
def sincos_f(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[0,1]
    ylim=[-0.5,0.5]

    X, Y = grid(N, xlim, ylim, device, dtype)

    c=torch.ones_like(X)
    ch, cv = c_half(c)

    f=(torch.sin(torch.pi*X)*torch.cos(torch.pi*Y)+torch.sin(5*torch.pi*X)*torch.cos(5*torch.pi*Y))

    u_exact = (1/(2*torch.pi*torch.pi))*torch.sin(torch.pi*X)*torch.cos(torch.pi*Y)+(1/(50*torch.pi*torch.pi))*torch.sin(5*torch.pi*X)*torch.cos(5*torch.pi*Y)
    return X, Y, u_exact, f[1:-1,1:-1], c, ch, cv

def sine_f(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[0,1]
    ylim=[-0.5,0.5]
    X, Y = grid(N, xlim, ylim, device, dtype)

    c=torch.ones_like(X)
    ch, cv = c_half(c)

    u_exact = torch.sin(torch.pi*X)*torch.cos(torch.pi*Y)

    from lilytorch.poisson_test import PoissonSolver
    h=(xlim[1]-xlim[0])/N
    solver = PoissonSolver(
        dtype,
        device,
        h
    )
    solver.BC(u_exact)
    f, _=solver.FD_operator(u_exact, ch, cv, h*h)
    #f=2*(np.pi**2)*torch.sin(np.pi*X)*torch.cos(np.pi*Y)

    return X, Y, u_exact, f, c, ch, cv

def quadratic(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[0,1]
    ylim=[0,1]
    X, Y = grid(N, xlim, ylim, device, dtype)
    c=torch.ones_like(X)
    ch, cv = c_half(c)
    u_exact = X*Y*(1-X)*(1-Y)
    f=-(2*X*(X-1)+2*Y*(Y-1))

    return X, Y, u_exact, f[1:-1,1:-1], c, ch, cv

def exp_f(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[-1,1]
    ylim=[-1,1]
    X, Y = grid(N, xlim, ylim, device, dtype)

    c=torch.ones_like(X)
    ch, cv = c_half(c)

    xhat=0.
    yhat=-0.
    u_exact = torch.exp(-(X-xhat)**2-(Y-yhat)**2)

    f=(4*((X-xhat)**2+(Y-yhat)**2)*u_exact+2*u_exact)

    return X, Y, u_exact, f[1:-1,1:-1], c, ch, cv

def pyro(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[0,1]
    ylim=[0,1]
    X, Y = grid(N, xlim, ylim, device, dtype)
    c=torch.ones_like(X)
    ch, cv = c_half(c)

    f=(-2.0 * ((1.0 - 6.0 * X**2) * Y**2 * (1.0 - Y**2) +
                   (1.0 - 6.0 * Y**2) * X**2 * (1.0 - X**2)))

    u_exact = -(X**2-X**4)*(Y**4-Y**2)

    return X, Y, u_exact, f[1:-1,1:-1], c, ch, cv

def lilypad(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[0,1]
    ylim=[0,1]
    X, Y = grid(N, xlim, ylim, device, dtype)
    c=torch.ones_like(X)
    ch, cv = c_half(c)

    # build rhs function f=divergence(u,v)
    u = torch.ones((N,N),device=device,dtype=dtype)
    v = -0.5*torch.ones((N,N),device=device,dtype=dtype)
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

    dx=(xlim[1]-xlim[0])/N
    dy=(ylim[1]-ylim[0])/N
    f=divergence(u, v, dx, dy)


    return X, Y, None, f, c, ch, cv

def variable_coeff(N,device=torch.device("cpu"), dtype=torch.float32):

    xlim=[-1,1]
    ylim=[-1,1]
    X, Y = grid(N, xlim, ylim, device, dtype)

    from body import BodyAnalytical, circle
    h=(xlim[1]-xlim[0])/N
    xt=0.4
    yt=0.0
    body = BodyAnalytical(
        device, X[:,0], Y[0,:],
        lambda x, y: circle(x,y,xt=xt,yt=yt,r=0.3),
        (
            lambda t: 0*t,
            (
                lambda t: 0*t,
                lambda t: 0*t
            )
        ),
        eps=2*h
    )
    body.initialize()
    sdf_fun =body.sdf_fun


    d = sdf_fun(X,Y)
    (mu0, mu1) = body.mu_funcs(d)
    c=mu0

    # ks=0.0000
    # kg=1
    # sigma=10
    # c=ks+(kg-ks)*(torch.tanh(sigma*d)+1)/2

    # beta=100
    # alpha=1
    # sigma=30


    # c=alpha+(beta-alpha)*(torch.tanh(sigma*d)+1)/2

    # ch, cv = c_half(c)

    f=torch.ones_like(c)

    Xh = (X[1:,1:-1]+X[:-1,1:-1])*0.5
    Yh = (Y[1:,1:-1]+Y[:-1,1:-1])*0.5
    dh = sdf_fun(Xh,Yh)
    ch, _ = body.mu_funcs(dh)

    Xv = (X[1:-1,1:]+X[1:-1,:-1])*0.5
    Yv = (Y[1:-1,1:]+Y[1:-1,:-1])*0.5
    dv = sdf_fun(Xv,Yv)
    cv, _ = body.mu_funcs(dv)

    return X, Y, c, f[1:-1,1:-1], c, ch, cv

def multigrid_course(N,device=torch.device("cpu"), dtype=torch.float32):
    """
    Multigrid course
    """

    xlim=[0,3.2]
    ylim=[0,3.2]

    X, Y = grid(N, xlim, ylim, device, dtype)

    # c  = 1.0+0.5*torch.exp(torch.sin(2.0*torch.pi*X/xlim[-1])*torch.cos(2.0*torch.pi*Y/ylim[-1]))
    c  = torch.ones_like(X)

    ch, cv = c_half(c)

    from lilytorch.poisson_test import PoissonSolver
    h=(xlim[1]-xlim[0])/N
    u_exact=torch.exp(torch.cos(2.0*torch.pi*X/xlim[-1])*torch.cos(2.0*torch.pi*Y/ylim[-1]))
    solver = PoissonSolver(
        dtype,
        device,
        h
    )
    solver.BC(u_exact)
    f, _=solver.FD_operator(u_exact, ch, cv, h*h)

    return X, Y, u_exact, f, c, ch, cv


