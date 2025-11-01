
import torch
import torch_interpolations

torch.set_num_threads(16)
class AdvectionSolver:
    """
    Solver class for the advection-diffusion equation
    """

    def __init__(self,
                 dt,
                 dx,
                 dy,
                 x, 
                 y,
                 nu,
                 type = "explicit",
                 ):
        """
        x        : x-domain
        y        : y-domain
        dt       : time step
        nu       : diffusion coefficient
        type     : quick or explicit solver
        """
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dtdx = dt / dx
        self.dtdy = dt / dy
        self.dtdx2 = self.dtdx/dx
        self.dtdy2 = self.dtdy/dy
        self.nu = nu

        self.C = 0.1
        self.C2 = self.C**2

        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.nm2 = self.nx-2
        self.ADBzeros = torch.zeros(self.nm2,self.nm2, dtype=torch.float32)
        self.ADBones = torch.ones(self.nm2,self.nm2, dtype=torch.float32)
    
        self.X, self.Y = torch.meshgrid(x,y)
        self.xflat = self.X.flatten()
        self.yflat = self.Y.flatten()
        # dummy initialization
        self.gu = torch_interpolations.RegularGridInterpolator((x,y), torch.zeros_like(X))
        self.gv = torch_interpolations.RegularGridInterpolator((x,y), torch.zeros_like(X))

    def clf(self, u, v):
        vel_max = torch.max(
            torch.max(torch.abs(u)), 
            torch.max(torch.abs(v))
            )
        self.dt = self.dx/(vel_max)
    

    def solve_explicit(self, u, v):
        """
        explicit solver
        """

        u[1:-1, 1:-1] = (
            u[1:-1, 1:-1]-
            self.dtdx*u[1:-1,1:-1]*(u[1:-1,1:-1]-u[:-2,1:-1]) -
            self.dtdy*v[1:-1,1:-1]*(u[1:-1,1:-1]-u[1:-1,:-2]) +
            nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
            nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
        )
        v[1:-1, 1:-1] = (
            v[1:-1, 1:-1]-
            self.dtdx*u[1:-1,1:-1]*(v[1:-1,1:-1]-v[:-2,1:-1]) -
            self.dtdy*v[1:-1,1:-1]*(v[1:-1,1:-1]-v[1:-1,:-2]) +
            nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
            nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
        )   
        return (u,v)
    
    def med(self, a, b, c):
        return torch.max(torch.min(a,b), torch.min(torch.max(a,b), c))

    def FLUXLMT_rule(self, bf, phiU, phiD, phiR): # correspond to (C,D,U) ini LilyPad notation
        bf -= (phiD-2*phiU+phiR)/6
        b1 = phiR+10*(phiU-phiR)
        return self.med(bf, phiU, self.med(phiU, phiD, b1))

    def phi_wLMT(self, F, phi):
        bf=0.5*(phi[:-2,1:-1]+phi[1:-1,1:-1]) 
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[1:-3,2:-2],phi[2:-2,2:-2],phi[:-4,2:-2]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[1:-3,2:-2],phi[3:-1,2:-2]),
        )
        return bf

    def phi_sLMT(self, F, phi):
        bf=0.5*(phi[1:-1,:-2]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,1:-3],phi[2:-2,2:-2],phi[2:-2,:-4]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[1:-3,2:-2],phi[2:-2,3:-1]),
        )
        return bf

    def phi_eLMT(self, F, phi):
        bf=0.5*(phi[2:,1:-1]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[3:-1,2:-2],phi[1:-3,2:-2]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[3:-1,2:-2],phi[2:-2,2:-2],phi[4:,2:-2]),
        )
        return bf

    def phi_nLMT(self, F, phi):
        bf=0.5*(phi[1:-1,2:]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[2:-2,3:-1],phi[2:-2,1:-3]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,3:-1],phi[2:-2,2:-2],phi[2:-2,4:]),
        )
        return bf

    def solve_FLUXLMT(self, u, v):

        uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1]) 
        ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1]) 
        vs = 0.5*(v[1:-1,:-2]+v[1:-1,1:-1]) 
        vn = 0.5*(v[1:-1,2:]+v[1:-1,1:-1])

        u[1:-1,1:-1] = (
                        u[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_wLMT(uw,u)-ue*self.phi_eLMT(ue,u))+
                        self.dtdy*(vs*self.phi_sLMT(vs,u)-vn*self.phi_nLMT(vn,u))+
                        nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )
        v[1:-1,1:-1] = (
                        v[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_wLMT(uw,v)-ue*self.phi_eLMT(ue,v))+
                        self.dtdy*(vs*self.phi_sLMT(vs,v)-vn*self.phi_nLMT(vn,v))+
                        nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )
        return (u,v)

    def psi(self, rf):
        # ADBQUICKEST
        return torch.max(
            torch.zeros_like(rf),
            torch.min(
                2.0*rf*(1.0-self.C),
                torch.min(
                    (2.0+self.C2-3.0*self.C+(1.0-self.C2)*rf)/(3.0-3.0*self.C),
                    2.0*(1.0-self.C)*torch.ones_like(rf)
                    )
                )
            )
        # # CUBISTA
        # return torch.max(
        #     torch.zeros_like(rf),
        #     torch.min(
        #         (3/2)*rf,
        #         torch.min(
        #             (3/4)*rf+1/4,
        #             (3/2)*torch.ones_like(rf)
        #             )
        #         )
        #     )

    def ADBQUICKEST_rule(self, phiU, phiD, phiR):
        return torch.where(
            phiD==phiU,
            phiU,
            phiU+0.5*(phiD-phiU)*self.psi((phiU-phiR)/(phiD-phiU))
        )

    def phi_w(self, F, phi):
        out = torch.zeros((self.nm2,self.nm2))
        out[1:,:]=torch.where(
            F[1:,:]>0, 
            self.ADBQUICKEST_rule(phi[1:-2,1:-1], phi[2:-1,1:-1], phi[:-3,1:-1]),
            self.ADBQUICKEST_rule(phi[2:-1,1:-1], phi[1:-2,1:-1], phi[3:,1:-1])
        )
        out[0,:]=torch.where(
            F[0,:]>0,
            0.5*(phi[0,1:-1]+phi[1,1:-1]),
            self.ADBQUICKEST_rule(phi[1,1:-1], phi[0,1:-1], phi[2,1:-1])
        )
        return out

    def phi_s(self, F, phi):
        out = torch.zeros((self.nm2,self.nm2))
        out[:,1:]=torch.where(
            F[:,1:]>0, 
            self.ADBQUICKEST_rule(phi[1:-1,1:-2], phi[1:-1,2:-1], phi[1:-1,:-3]),
            self.ADBQUICKEST_rule(phi[1:-1,2:-1], phi[1:-1,1:-2], phi[1:-1,3:])
        )
        out[:,0]=torch.where(
            F[:,0]>0,
            0.5*(phi[1:-1,0]+phi[1:-1,1]),
            self.ADBQUICKEST_rule(phi[1:-1,1], phi[1:-1,0], phi[1:-1,2])
        )
        return out

    def phi_e(self, F, phi):
        out = torch.zeros((self.nm2,self.nm2))
        out[:-1,:]=torch.where(
            F[:-1,:]>0, 
            self.ADBQUICKEST_rule(phi[1:-2,1:-1], phi[2:-1,1:-1], phi[:-3,1:-1]),
            self.ADBQUICKEST_rule(phi[2:-1,1:-1], phi[1:-2,1:-1], phi[3:,1:-1])
        )
        out[-1,:]=torch.where(
            F[0,:]<0,
            0.5*(phi[-1,1:-1]+phi[-2,1:-1]),
            self.ADBQUICKEST_rule(phi[-2,1:-1], phi[-1,1:-1], phi[-3,1:-1])
        )
        return out

    def phi_n(self, F, phi):
        out = torch.zeros((self.nm2,self.nm2))
        out[:,:-1]=torch.where(
            F[:,:-1]>0, 
            self.ADBQUICKEST_rule(phi[1:-1,1:-2], phi[1:-1,2:-1], phi[1:-1,:-3]),
            self.ADBQUICKEST_rule(phi[1:-1,2:-1], phi[1:-1,1:-2], phi[1:-1,3:])
        )
        out[:,-1]=torch.where(
            F[:,0]<0,
            0.5*(phi[1:-1,-1]+phi[1:-1,-2]),
            self.ADBQUICKEST_rule(phi[1:-1,-2], phi[1:-1,-1], phi[1:-1,-3])
        )
        return out

    def solve_ADBQUICKEST(self, u, v):

        uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1]) 
        ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1]) 
        vs = 0.5*(v[1:-1,:-2]+v[1:-1,1:-1]) 
        vn = 0.5*(v[1:-1,2:]+v[1:-1,1:-1])

        u[1:-1,1:-1] = (
                        u[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_w(uw,u)-ue*self.phi_e(ue,u))+
                        self.dtdy*(vs*self.phi_s(vs,u)-vn*self.phi_n(vn,u))+
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )
        v[1:-1,1:-1] = (
                        v[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_w(uw,v)-ue*self.phi_e(ue,v))+
                        self.dtdy*(vs*self.phi_s(vs,v)-vn*self.phi_n(vn,v))+
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )
        return (u,v)



    def solve_implicit(self, u, v):
        """
        Implicit solver based on Staam, 1999 where
        u_new(x,y) = u(x-dt*u(x,y), y-dt*v(x,y)) [linearly interpolated]
        """
        xold = self.xflat-u.flatten()*self.dt
        yold = self.yflat-v.flatten()*self.dt
        
        self.gu.values = u
        self.gv.values = v
        
        u = self.gu((xold, yold)).reshape(X.shape)
        v = self.gv((xold, yold)).reshape(X.shape)

        u[1:-1,1:-1] += (
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )
        v[1:-1,1:-1] += (
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )

        return (u,v)

    def Neumann_BC(self, u, v):
        u[0, :]  = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0]  = u[:, 1]
        u[:, -1] = u[:, -2]
        v[0, :]  = v[1, :]
        v[-1, :] = v[-2, :]
        v[:, 0]  = v[:, 1]
        v[:, -1] = v[:, -2]

    def Dirichlet_BC(self, u, v):
        u[0, :]  = 0
        u[-1, :] = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0

if __name__ == "__main__":

    nx      = 2**10
    ny      = 2**10
    nt      = 100
    nu      = 0.0
    dt      = 0.001
    x = torch.linspace(-2, 4, nx)
    y = torch.linspace(-2, 4, ny)
    dx      = x[1]-x[0]
    dy      = y[1]-y[0]
    X, Y = torch.meshgrid(x, y)
    print("dt={}s, dx={}, nx={}".format(dt, dx, nx))

    solver = AdvectionSolver(
        dt,
        dx,
        dy,
        x, 
        y,
        nu,
        type = "explicit",
    )
    
    def initial_conditions():
        # u = torch.zeros((ny,nx))
        # v = torch.zeros((ny,nx))
        # nm=int(nx/2)-int(.5 / dy)
        # np=int(nx/2)+int(.5 / dy)
        # u[nm:np,nm:np] = 1
        # v[nm:np,nm:np] = 1
        u = torch.exp(-X**2-Y**2)
        v = torch.exp(-X**2-Y**2)

        return (u,v)
    

    (u0,v0) = initial_conditions()

    from matplotlib import pyplot
    import time
    fig, (ax_1, ax_2, ax_3, ax_4, ax_5) = pyplot.subplots(1, 5, figsize=(25,5))
    # fig, (ax1,ax2,ax3) = pyplot.subplots(1,3,subplot_kw={"projection": "3d"})

    # ax1.plot_surface(X, Y, u, rstride=1, cstride=1,
    #         linewidth=0, antialiased=False)

    CS1=ax_1.contourf(X, Y, u0, 20)
    ax_1.set_title("Initial")
    fig.colorbar(CS1)
    
    u=u0.clone().detach()
    v=v0.clone().detach()
    start = time.time()
    for n in range(nt + 1): 
        (u,v) = solver.solve_explicit(u,v)
        solver.Neumann_BC(u,v)
    print("Upwind solver took {}s".format(time.time()-start))
        

    CS2=ax_2.contourf(X, Y, u, 20)
    ax_2.set_title("Upwind")
    fig.colorbar(CS2)

    # ax2.plot_surface(X, Y, u, rstride=1, cstride=1,
    #         linewidth=0, antialiased=False)

    # re-set initial conditions and solve using the quick method
    u=u0.clone().detach()
    v=v0.clone().detach()
    start = time.time()
    for n in range(nt + 1): 
        (u,v) = solver.solve_implicit(u,v)
        solver.Neumann_BC(u,v)
    print("Implicit solver took {}s".format(time.time()-start))


    CS3=ax_3.contourf(X, Y, u, 20)
    ax_3.set_title("implicit")
    fig.colorbar(CS3)

    u=u0.clone().detach()
    v=v0.clone().detach()
    start = time.time()
    for n in range(nt + 1): 
        (u,v) = solver.solve_ADBQUICKEST(u,v)
        solver.Neumann_BC(u,v)
    print("ADBquickest solver took {}s".format(time.time()-start))


    CS4=ax_4.contourf(X, Y, u, 20)
    ax_4.set_title("ADBquickest")
    fig.colorbar(CS4)


    # re-set initial conditions and solve using the quick method
    u=u0.clone().detach()
    v=v0.clone().detach()
    start = time.time()
    for n in range(nt + 1): 
        (u,v) = solver.solve_FLUXLMT(u,v)
        solver.Neumann_BC(u,v)
    print("Flux-LMT solver took {}s".format(time.time()-start))


    CS5=ax_5.contourf(X, Y, u, 20)
    ax_5.set_title("Flux LMT")
    fig.colorbar(CS5)






    pyplot.show()