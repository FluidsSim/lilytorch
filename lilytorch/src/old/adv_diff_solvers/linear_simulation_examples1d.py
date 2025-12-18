
import torch

def exp():

    nx     = 101
    xstart = 0
    xend   = 1
    bc0    = 0
    bc1    = 0
    x      = torch.linspace(xstart, xend, nx)
    dx     = x[1]-x[0]
    v      = 1

    nu    = 0.0
    dt    = .001 #1*dx/(1+5*nu) #
    nt    = 500
    tstop = nt*dt
    print("dt={}s, dx={}, Cr={}, Pe={}, tstop={}s".format(dt, dx, v*dt/dx, v*dx/nu, tstop))

    u0 = v*torch.exp(-((x-0.25))**2/0.005)
    uexact = v*torch.exp(-((x-0.25+v*tstop))**2/0.005)

    return dt, dx, nt, x, bc0, bc1, nu, u0, uexact


def barba():

    nx     = 100
    xstart = 0
    xend   = 3
    bc0    = 0
    bc1    = 0
    x      = torch.linspace(xstart, xend, nx)
    dx     = x[1]-x[0]
    v      = 1.0

    nu    = 0.0
    dt    = .005
    nt    = 600
    tstop = nt*dt
    print("dt={}s, dx={}, Cr={}, Pe={}, tstop={}s".format(dt, dx, v*dt/dx, v*dx/nu, tstop))

    

    # u0 = torch.zeros_like(x)
    # u0[int(.5 / dx):int(1 / dx + 1)] = 1

    u0 = torch.where(
        torch.logical_and(x<1, x>0.5),
        v,
        0
    )

    uexact =  torch.where(
        torch.logical_and(x<(1+v*tstop), x>(0.5+v*tstop)),
        v,
        0
    )


    return dt, dx, nt, x, bc0, bc1, nu, u0, uexact




def neumann1():
    """
    Example 1 from Neumann et al, 2011 Env Modelling and Software
    """


    nx     = 101
    xstart = 0
    xend   = 1
    bc0    = 1
    bc1    = 0
    x      = torch.linspace(xstart, xend, nx)
    dx     = x[1]-x[0]
    v      = 1.0

    nu    = 0.0
    dt    = .007
    nt    = 100
    tstop = nt*dt
    print("dt={}s, dx={}, Cr={}, Pe={}, tstop={}s".format(dt, dx, v*dt/dx, v*dx/nu, tstop))

    
    u0 = torch.where(
        x<0.,
        v,
        0
    )

    uexact = torch.where(
        x<0.+v*tstop,
        v,
        0
    )


    return dt, dx, nt, x, bc0, bc1, nu, u0, uexact



def neumann2():
    """
    Example 1 from Neumann et al, 2011 Env Modelling and Software
    """


    nx     = 101
    xstart = 0
    xend   = 1
    bc0    = 0
    bc1    = 0
    x      = torch.linspace(xstart, xend, nx)
    dx     = x[1]-x[0]
    v      = 1.0

    nu    = 0.0
    dt    = .001 
    nt    = 150
    tstop = nt*dt
    print("dt={}s, dx={}, Cr={}, Pe={}, tstop={}s".format(dt, dx, v*dt/dx, v*dx/nu, tstop))

    

    u0 = torch.where(
        torch.logical_and(x<20*dx,x>0.),
        v*torch.sin(torch.pi*x/(20*dx))**2,
        0
    )

    uexact = None


    return dt, dx, nt, x, bc0, bc1, nu, u0, uexact



def ferreira():
    """
    Example 1 from Ferreira et al, 2009 page 9
    """


    nx     = 201
    xstart = 0
    xend   = 2
    bc0    = 0
    bc1    = 0
    x      = torch.linspace(xstart, xend, nx, dtype=torch.double)
    dx     = x[1]-x[0]
    v      = 1.0

    nu    = 0.00
    dt    = .005
    nt    = 100
    tstop = nt*dt
    print("dt={}s, dx={}, Cr={}, Pe={}, tstop={}s".format(dt, dx, v*dt/dx, v*dx/nu, tstop))

    

    u0 = torch.zeros_like(x, dtype=torch.double)
    i1=x<=0.2
    i2=torch.logical_and(0.2<=x,x<=0.4)
    i3=torch.logical_and(0.4<=x,x<=0.6)
    i4=torch.logical_and(0.6<=x,x<=0.8)

    u0[i1]=1
    u0[i2]=4*x[i2]-3/5
    u0[i3]=-4*x[i3]+13/5
    u0[i4]=1
    


    uexact = torch.zeros_like(x, dtype=torch.double)
    xexact = x-v*tstop
    i1=xexact<=0.2
    i2=torch.logical_and(0.2<=xexact,xexact<=0.4)
    i3=torch.logical_and(0.4<=xexact,xexact<=0.6)
    i4=torch.logical_and(0.6<=xexact,xexact<=0.8)

    uexact[i1]=1
    uexact[i2]=4*xexact[i2]-3/5
    uexact[i3]=-4*xexact[i3]+13/5
    uexact[i4]=1
    


    return dt, dx, nt, x, bc0, bc1, nu, u0, uexact
