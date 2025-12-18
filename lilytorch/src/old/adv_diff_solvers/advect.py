

import torch

def dudx(u, dx, g):
    g[:,1:-1] = 0.5*(u[:,2:]-u[:,:-2])/dx
    g[:,0] = (u[:,1]-u[:,0])/dx
    g[:,-1] = (u[:,-1]-u[:,-2])/dx

def dudy(u, dy, g):
    g[1:-1,:] = 0.5*(u[2:,:]-u[:-2,:])/dy
    g[0,:] = (u[1,:]-u[0,:])/dy
    g[-1,:] = (u[-1,:]-u[-2,:])/dy

def du2dx2(u, dx, g):
    g[:,1:-1] = (u[:,2:]-2*u[:,1:-1]+u[:,:-2])/(dx**2)
    g[:,0] = 2*(u[:,1]-u[:,0])/(dx**2)
    g[:,-1] = 2*(u[:,-2]-u[:,-1])/(dx**2)

def du2dy2(u, dy, g):
    g[1:-1,:] = (u[2:,:]-2*u[1:-1,:]+u[:-2,:])/(dy**2)
    g[0,:] = 2*(u[1,:]-u[0,:])/(dy**2)
    g[-1,:] = 2*(u[-2,:]-u[-1,:])/(dy**2)
    
        # u[1:-1, 1:-1] = (u[1:-1,1:-1] + 
        #                 nu * dt / dx**2 * 
        #                 () +
        #                 nu * dt / dy**2 * 
        #                 ()



# def dudx(u, dx, g):
#     g[:,0] = (u[:,1]-u[:,0])/dx
#     g[:,1:] = (u[:,1:]-u[:,:-1])/dx
#     return g

# def dudy(u, dy, g):
#     g[0,:] = (u[1,:]-u[0,:])/dy
#     g[1:,:] = (u[1:,:]-u[:-1,:])/dy
#     return g    



def _convection(u, v, g, nu, dt, spacing):
   
    """
    explicit advect solver

    du2_dx2 = torch.gradient( du[0],spacing=spacing[0],dim=0)[0]
    du2_dy2 = torch.gradient( du[1],spacing=spacing[1],dim=1)[0]
    dv2_dx2 = torch.gradient( dv[0],spacing=spacing[0],dim=0)[0]
    dv2_dy2 = torch.gradient( dv[1],spacing=spacing[1],dim=1)[0]

    """
    # return (u-dt*u*dudx(u,spacing[0],g)-dt*v*dudy(u,spacing[1],g), v)

    du = torch.gradient(u, spacing=spacing)
    dv = torch.gradient(v, spacing=spacing)

    # u += - dt*(du[0]*u-du[1]*v) 
    # v += - dt*(dv[0]*u-dv[1]*v) 

    # return (
    #     u - dt*(du[0]*u+du[1]*v),
    #     v - dt*(dv[0]*u+dv[1]*v) 
    # )    

    # return (
    #     u,
    #     v
    # )
    
    return (
        u - dt*(du[0].T*u+du[1].T*v) + nu*dt*(
            torch.gradient( du[0].T,spacing=spacing[0],dim=0)[0]+
            torch.gradient( du[1].T,spacing=spacing[1],dim=1)[0]
        ),
        v - dt*(dv[0].T*u+dv[1].T*v) + nu*dt*(
            torch.gradient( dv[0].T,spacing=spacing[0],dim=0)[0]+
            torch.gradient( dv[1].T,spacing=spacing[1],dim=1)[0]
        )
    )    

    # return (
    #     u-dt*(du[0]*u-du[1]*v),
    #     v-dt*(dv[1]*u-dv[0]*v)
    # )

    # u[1:, 1:] = (u[1:, 1:] - 
    #              (u[1:, 1:] * dt * dudx(u, spacing[0], g)[1:,1:]) -
    #               v[1:, 1:] * dt * dudy(u, spacing[1], g)[1:,1:])

    # return(u,v)



    # # u[1:, 1:] = (u[1:, 1:] - 
    # #              (u[1:, 1:] * dt / spacing[0] * (u[1:, 1:] - u[1:, :-1])) -
    # #               v[1:, 1:] * dt / spacing[1] * (u[1:, 1:] - u[:-1, 1:]))
    # return (u,v)


    # return (
    #     u - nu*dt*(
    #         torch.gradient( du[0].T,spacing=spacing[0],dim=0)[0].T
    #         torch.gradient( du[0].T,spacing=spacing[1],dim=1)[0].T
    #     ),
    #     v - nu*dt*(
    #         torch.gradient( dv[0].T,spacing=spacing[0],dim=0)[0]+
    #         torch.gradient( dv[1].T,spacing=spacing[1],dim=1)[0]
    #     )
    # )



    # u = u
    # u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
    #                 nu * dt / dx**2 * 
    #                 (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
    #                 nu * dt / dy**2 * 
    #                 (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    # return (u,v)

    
if __name__ == "__main__":

    nx      = 31
    ny      = 31
    nt      = 17
    dx      = 2 / (nx - 1)
    dy      = 2 / (ny - 1)
    nu      = 0.05
    dt      = .25 * dx * dy / nu
    spacing = [dx,dy]
    x = torch.linspace(-2, 2, nx)
    y = torch.linspace(-2, 2, ny)
    X, Y = torch.meshgrid(x, y)

    from matplotlib import pyplot, cm
    fig, (ax1,ax2,ax3) = pyplot.subplots(1,3,subplot_kw={"projection": "3d"})
    
    def initial_conditions():
        u = torch.ones((nx,ny))
        v = torch.ones((nx,ny))
        # u=0.1*((X-1)**2+Y**2)
        u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
        return (u,v)
    

    (u,v) = initial_conditions()
    ax1.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
        
    for n in range(nt + 1): 
        g = torch.zeros_like(u)
        dudx(u,spacing[0],g)
        u += -dt*u*g
        dudy(u,spacing[1],g)
        u += -dt*v*g
        du2dx2(u,dx,g)
        u += nu*dt*g
        du2dy2(u,dy,g)
        u += nu*dt*g




    ax2.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)

    (u,v) = initial_conditions()
    for n in range(nt + 1): 
        g = torch.zeros_like(u)
        (u,v) = _convection(u,v,g,torch.tensor([nu]), dt, spacing)
    ax3.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    




    pyplot.show()
