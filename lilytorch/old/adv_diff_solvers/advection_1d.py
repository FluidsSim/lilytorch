
import torch

a=6/8
b=3/8
c=-1/8

def psi(rf, C, C2):
    # # ADBQUICKEST
    # return torch.max(
    #     torch.zeros_like(rf),
    #     torch.min(
    #         2*rf*(1-C),
    #         torch.min(
    #             (2+C2-3*torch.abs(C)+(1-C2)*rf)/(3-3*C),
    #             (2-2*C)*torch.ones_like(rf)
    #             )
    #         )
    #     )
    # CUBISTA
    return torch.max(
        torch.zeros_like(rf),
        torch.min(
            (3/2)*rf,
            torch.min(
                (3/4)*rf+1/4,
                (3/2)*torch.ones_like(rf)
                )
            )
        )


def ADBQUICKEST_rule(phiU, phiD, phiR, C, C2):
    return torch.where(
        phiD!=phiU,
        (phiU+0.5*psi((phiU-phiR)/(phiD-phiU), C, C2)*(phiD-phiU)),
        0
    )
def _solve_explicit(u, dtdx, dtdx2, nu=0):
    """
    explicit solver
    """
    # u[1:] -= dtdx*1*(u[1:]-u[:-1])
    # u[1:] -= dtdx*u[1:]*(u[1:]-u[:-1]) 
    
    u[1:-1] = (
        u[1:-1]-dtdx*1*(u[1:-1]-u[0:-2]) + #u[1:-1]-dtdx*u[1:-1]*(u[1:-1]-u[0:-2]) + 
        nu*dtdx2*(u[2:]-2*u[1:-1]+u[0:-2]) 
        ) 
    return u



def _solve_quick_torch(u, dtdx, dtdx2, nu=0):

    n=len(u)
    v=1
    Fw=v*torch.ones((n-2))
    Fe=v*torch.ones((n-2))
    # Fw=0.5*((u[1:-1]+u[:-2]))
    # Fe=0.5*((u[1:-1]+u[2:]))
    phi_w=0.5*(u[1:-1]+u[:-2])
    phi_w[1:]-=0.125*torch.where(
        Fw[1:]>0, 
        u[:-3]+u[2:-1]-2*u[1:-2], #u[i-2]+u[i]-2*u[i-1]
        u[1:-2]+u[3:]-2*u[2:-1] #u[i-1]+u[i+1]-2*u[i]
        )
    phi_w[0]-=torch.where(
        Fw[0]>0,
        0,
        0.125*(u[0]+u[2]-2*u[1])
    )
    phi_e=0.5*(u[1:-1]+u[2:])
    phi_e[:-1]-=0.125*torch.where(
        Fe[:-1]>0,
        u[:-3]+u[2:-1]-2*u[1:-2], #u[i-1]+u[i+1]-2*u[i]
        u[1:-2]+u[3:]-2*u[2:-1] #u[i]+u[i+2]-2*u[i+1]
        )
    phi_e[-1]-=torch.where(
        Fe[0]<0,
        0,
        0.125*(u[-3]+u[-1]-2*u[-2]) #u[i]+u[i+2]-2*u[i+1]
    )

    u[1:-1] = (
        u[1:-1]-dtdx*(Fe*phi_e-Fw*phi_w) +
        nu*dtdx2*(u[2:]-2*u[1:-1]+u[0:-2]) 
    )
    return u




def _solve_quick(u, dtdx, dtdx2, nu=0):
    """
    QUICK solver
    """
    n=len(u)
    unew=torch.zeros_like(u)

    for i in range(1,n-1):
        Fw=0.5*((u[i]+u[i-1]))
        Fe=0.5*((u[i]+u[i+1]))
        if Fw>0:
            if i==1:
                phi_w = 0.5*(u[i]+u[i-1])
            else:
                phi_w = 0.5*(u[i]+u[i-1])-0.125*(u[i-2]+u[i]-2*u[i-1]) 
                # phi_w = a*u[i-1]+b*u[i]+c*u[i-2] # p_l=a*p_L+b*p_C+c*p_LL
        else:
            phi_w = 0.5*(u[i]+u[i-1])-0.125*(u[i-1]+u[i+1]-2*u[i]) 
            # phi_w = a*u[i]+b*u[i-1]+c*u[i+1] # p_l=a*p_C+b*p_L+c*p_R

        if Fe>0:
            phi_e = 0.5*(u[i]+u[i+1])-0.125*(u[i-1]+u[i+1]-2*u[i]) #
            # phi_e = a*u[i]+b*u[i+1]+c*u[i-1] # p_r=a*p_C+b*p_R+c*p_L
        else:
            if i==n-2:
                phi_e = 0.5*(u[i]+u[i+1])
            else:
                phi_e = 0.5*(u[i]+u[i+1])-0.125*(u[i]+u[i+2]-2*u[i+1]) 
                # phi_e = a*u[i+1]+b*u[i]+c*u[i+2] # p_r=a*p_R+b*p_C+c*p_RR
        
        unew[i] = u[i]-dtdx*(Fe*phi_e-Fw*phi_w)
    return unew



class AdvectionSolver:
    """
    Solver class for the diffusion equation du/dt =nu*laplacian(u)
    """

    def __init__(self,
                 dt,
                 dx,
                 type = "explicit",
                 nu=0.0
                 ):
        """
        x        : x-domain
        y        : y-domain
        dt       : time step
        nu       : diffusion coefficient
        type     : quick or explicit solver
        """
        self.dtdx = dt / dx
        self.dtdx2 = self.dtdx/dx
        self.C = 1*self.dtdx
        self.C2 = self.C**2
        self.nu   = nu

        if type=="quick":
            self.solve = self.solve_quick

        elif type=="explicit":
            self.solve = self.solve_explicit

    def solve_quick(self, u):
        return _solve_quick(u, self.dtdx, self.dtdx2, nu=self.nu)

    def solve_quick_torch(self, u):
        return _solve_quick_torch(u, self.dtdx, self.dtdx2, nu=self.nu)

    def solve_explicit(self, u):
        return _solve_explicit(u, self.dtdx, self.dtdx2, nu=self.nu)


    def solve_ADBQUICKEST(self, u):

        n=len(u)
        v=1
        Fw=v*torch.ones((n-2))
        Fe=v*torch.ones((n-2))
        # Fw=0.5*((u[1:-1]+u[:-2]))
        # Fe=0.5*((u[1:-1]+u[2:]))

        # General rule: phi_f = phiU+0.5*psi((phiU-phiR)/(phiD-phiU))*(phiD-phiU)
        # CASE Fw>0 - since the idx start from 2 (u[-1] does not exist)
        # u[1:-2] # u_{i-1} = phiU
        # u[2:-1] # u_i = phiD
        # u[:-3] # u{i-2} = phiR
        # Fw<0 - 
        # u[1:-2] # u_{i-1} = phiD
        # u[2:-1] # u_i = phiU
        # u[3:] # u{i+1} = phiR

        phi_w = torch.ones((n-2))
        phi_w[1:]=torch.where(
            Fw[1:]>0, 
            ADBQUICKEST_rule(u[1:-2], u[2:-1], u[:-3], self.C, self.C2),
            ADBQUICKEST_rule(u[2:-1], u[1:-2], u[3:], self.C, self.C2)
        )
        phi_w[0]=torch.where(
            Fw[0]>0,
            0.5*(u[0]+u[1]),
            ADBQUICKEST_rule(u[1], u[0], u[2], self.C, self.C2)
        )

        # CASE Fe>0 - since the idx start from 2 (u[-1] does not exist)
        # u[1:-2] # u_{i} = phiU
        # u[2:-1] # u_{i+1} = phiD
        # u[:-3] # u{i-2} = phiR
        # Fe<0 - 
        # u[1:-2] # u_{i} = phiD
        # u[2:-1] # u_{i+1} = phiU
        # u[3:] # u{i+1} = phiR
        phi_e = torch.ones((n-2))
        phi_e[:-1]=torch.where(
            Fe[:-1]>0, 
            ADBQUICKEST_rule(u[1:-2], u[2:-1], u[:-3], self.C, self.C2),
            ADBQUICKEST_rule(u[2:-1], u[1:-2], u[3:], self.C, self.C2)
        )
        phi_e[-1]=torch.where(
            Fe[0]<0,
            0.5*(u[-1]+u[-2]),
            ADBQUICKEST_rule(u[-2], u[-1], u[-3], self.C, self.C2)
        )
        
        # from IPython import embed; embed()

        u[1:-1] = (
            u[1:-1]-self.dtdx*(Fe*phi_e-Fw*phi_w) +
            self.nu*self.dtdx2*(u[2:]-2*u[1:-1]+u[0:-2]) 
        )


        return u

    
if __name__ == "__main__":

    nx      = 101
    x = torch.linspace(0, 1, nx)
    dx      = x[1]-x[0]
    
    nu      = 0.00
    # sigma = .0001
    # dt = sigma * dx / nu
    dt      = .001 #1*dx/(1+5*nu) #
    nt      = 500

    print("dt={}s, dx={}, Cr={}, Pe={}, tstop={}s".format(dt, dx, 1*dt/dx, 1*dx/nu, nt*dt))
    
    solver = AdvectionSolver(
        dt,
        dx,
        type = "explicit",
        nu = nu
    )
    
    def initial_conditions():
        # u = torch.zeros(nx)
        # u[int(.5 / dx) : int(1 / dx + 1)] = 1
        u = 1*torch.exp(-((x-0.25))**2/0.005)
        # u = torch.zeros(nx)
        # u[0:20] = 1*torch.sin(torch.pi*x[0:20]/(20*dx))**2
        # n=int(nx/10)
        # u[int(nx/2)-n:int(nx/2)+n] = torch.cos(torch.pi*x[int(nx/2)-n:int(nx/2)+n]/(120*dx))**2
        return u
    
    from matplotlib import pyplot
    u = initial_conditions()
    fig, ax = pyplot.subplots()
    ax.plot(x,u,'k',label="initial")
    
    for n in range(nt+1): 
        u = solver.solve_explicit(u)
        u[0] = 0
        u[-1] = 0
    ax.plot(x,u,'b--',label="upwind")

    # # re-set initial conditions and solve using the quick method
    # u = initial_conditions()
    # for n in range(nt+1): 
    #     u = solver.solve_quick(u)
    #     u[0] = 0
    #     u[-1] = 0
    # ax.plot(x,u,'r*',label="QUICK")


    # re-set initial conditions and solve using the quick method
    u = initial_conditions()
    for n in range(nt+1): 
        u = solver.solve_quick_torch(u)
        u[0] = 0
        u[-1] = 0
    ax.plot(x,u,'r--',label="QUICK")


    # re-set initial conditions and solve using the quick method
    u = initial_conditions()
    for n in range(nt+1): 
        u = solver.solve_ADBQUICKEST(u)
        u[0] = 0
        u[-1] = 0
    ax.plot(x,u,'g--',label="ADBQUICKEST")
    ax.set_ylim([-0.1,1.3])

    ax.plot(x,1*torch.exp(-((x-0.75))**2/0.005),'y',label="exact")

    ax.legend()
    
    pyplot.show()