
import torch

a=6/8
b=3/8
c=-1/8


def psi_loop(rf, C, C2):
    p = (C-2)/(C-5)
    q = (C+4)/(C+1)
    if 0<rf and rf<p:
        return 2*rf
    elif p<=rf and rf<=q:
        return (2+C2-3*C+(1-C2)*rf)/(3-3*C)
    elif rf>q:
        return 2 
    else:
        return 0


def ADBQUICKEST_rule_normalized(phiU, phiD, phiR, C, C2):
    if phiD==phiR:
        return phiR
    else:
        phiU_hat = (phiU-phiR)/(phiD-phiR)
        rf = phiU_hat/(1-phiU_hat)
        phi_f_hat=phiU_hat+0.5*psi(rf, C, C2)*(1-phiU_hat)
        return (phiD-phiR)*phi_f_hat+phiR

def psi(rf, C, C2):
    # TOPUS
    # return 0.5*(torch.abs(rf)+rf)*(6*rf+2)/((1+torch.abs(rf))**3)

    # # ADBQUICKEST
    # return torch.max(
    #     torch.zeros_like(rf),
    #     torch.min(
    #         2*rf*(1-C),
    #         torch.min(
    #             (2+C2-3*C+(1-C2)*rf)/(3-3*C),
    #             2*(1-C)*torch.ones_like(rf)
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

    # phiU_hat = torch.where(
    #     phiD==phiR,
    #     0,
    #     (phiU-phiR)/(phiD-phiR),
    # )
    # rf = torch.where(
    #     phiD==phiU,
    #     0,
    #     (phiU-phiR)/(phiD-phiU)
    # )
    # phif_hat = phiU_hat+0.5*(1-phiU_hat)*psi(rf, C, C2)
    # return phiR+phif_hat*(phiD-phiR)
    

    # phif_hat = phiU_hat
    # phif_hat[torch.logical_and(0<phiU_hat, phiU_hat<3/8)] = (7/4)*phiU_hat[torch.logical_and(0<phiU_hat, phiU_hat<3/8)]
    # phif_hat[torch.logical_and(3/8<=phiU_hat, phiU_hat<=3/4)] = 3/8+(3/4)*phiU_hat[torch.logical_and(3/8<=phiU_hat, phiU_hat<=3/4)]
    # phif_hat[torch.logical_and(3/4<phiU_hat, phiU_hat<1)] = 3/8+(3/4)*phiU_hat[torch.logical_and(3/4<phiU_hat, phiU_hat<1)]
    # return phiR+(phiD-phiR)*phif_hat

    return torch.where(
        phiD==phiU,
        phiU,
        phiU+0.5*(phiD-phiU)*psi((phiU-phiR)/(phiD-phiU), C, C2)
    )

class AdvectionSolver:
    """
    Solver class for the diffusion equation du/dt =nu*laplacian(u)
    """

    def __init__(self,
                 dt,
                 dx,
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
        self.C2 = self.C*self.C
        self.nu   = nu


    def solve_quick_loop(self, u):
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
            
            unew[i] = u[i]-self.dtdx*(Fe*phi_e-Fw*phi_w)
        return unew


    def solve_ADBQUICKEST_loop(self, u):
        """
        Assumes positive velocities for Fw, Fe
        """
        n=len(u)
        unew=torch.zeros_like(u)

        for i in range(1,n-1):
            if i==1:
                phi_w = 0.5*(u[0]+u[1])
            else:
                phiU=u[i-1]
                phiD=u[i]
                phiR=u[i-2]
                if phiD==phiU:
                    phi_w=phiU
                else:
                    phi_w=ADBQUICKEST_rule_normalized(phiU, phiD, phiR, self.C, self.C2) #phiU+0.5*(phiD-phiU)*psi_loop((phiU-phiR)/(phiD-phiU), self.C, self.C2)

            phiU=u[i]
            phiR=u[i-1]
            phiD=u[i+1]
            if phiD==phiU:
                phi_e=phiU
            else:
                phi_e=ADBQUICKEST_rule_normalized(phiU, phiD, phiR, self.C, self.C2) #phiU+0.5*(phiD-phiU)*psi_loop((phiU-phiR)/(phiD-phiU), self.C, self.C2)

            # unew[i] = u[i]-self.dtdx*(u[i]-u[i-1])
            unew[i] = u[i]-self.dtdx*(phi_e-phi_w)
        return unew


    def solve_quick_torch(self, u):
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
            u[1:-1]-self.dtdx*(Fe*phi_e-Fw*phi_w) +
            self.nu*self.dtdx2*(u[2:]-2*u[1:-1]+u[0:-2]) 
        )
        
        return u


    def solve_explicit(self, u):
        """
        explicit solver
        """
        # u[1:] = u[1:]-self.dtdx*u[1:]*(u[1:]-u[:-1])
        u[1:-1] = (
            u[1:-1]-self.dtdx*u[1:-1]*(u[1:-1]-u[0:-2]) + 
            self.nu*self.dtdx2*(u[2:]-2*u[1:-1]+u[0:-2]) 
            ) 
        return u

    def solve_ADBQUICKEST(self, u):

        n=len(u)
        v=1
        # Fw=v*torch.ones((n-2))
        # Fe=v*torch.ones((n-2))
        Fw=0.5*((u[1:-1]+u[:-2]))
        Fe=0.5*((u[1:-1]+u[2:]))

        # General rule: phi_f = phiU+0.5*psi((phiU-phiR)/(phiD-phiU))*(phiD-phiU)
        # CASE Fw>0 - since the idx start from 2 (u[-1] does not exist)
        # u[1:-2] # u_{i-1} = phiU
        # u[2:-1] # u_i = phiD
        # u[:-3] # u{i-2} = phiR
        # Fw<0 - 
        # u[1:-2] # u_{i-1} = phiD
        # u[2:-1] # u_i = phiU
        # u[3:] # u{i+1} = phiR

        phi_w = torch.zeros((n-2))
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
        phi_e = torch.zeros((n-2))
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
        
        u[1:-1] = (
            u[1:-1]-self.dtdx*(Fe*phi_e-Fw*phi_w) +
            self.nu*self.dtdx2*(u[2:]-2*u[1:-1]+u[0:-2]) 
        )

        return u

    
if __name__ == "__main__":

    import linear_simulation_examples1d as examples

    dt, dx, nt, x, bc0, bc1, nu, u0, uexact = examples.barba()
    
    solver = AdvectionSolver(
        dt,
        dx,
        nu = nu
    )

    from matplotlib import pyplot
    fig, ax = pyplot.subplots()
    ax.plot(x,u0,'k',label="initial")
    
    u=u0.clone().detach()
    for n in range(nt+1): 
        u = solver.solve_explicit(u)
        u[0] = bc0
        u[-1] = bc1
    ax.plot(x,u,'b--',label="upwind")

    # # # re-set initial conditions and solve using the quick method
    # # u=u0.clone().detach()
    # # for n in range(nt+1): 
    # #     u = solver.solve_quick(u)
    # #     u[0] = bc0
    # #     u[-1] = bc1
    # # ax.plot(x,u,'r*',label="QUICK-LOOP")


    # # re-set initial conditions and solve using the quick method
    # u=u0.clone().detach()
    # for n in range(nt+1): 
    #     u = solver.solve_quick_torch(u)
    #     u[0] = bc0
    #     u[-1] = bc1
    # ax.plot(x,u,'r--',label="QUICK")

    # re-set initial conditions and solve using the quick method
    u=u0.clone().detach()
    for n in range(nt+1): 
        u = solver.solve_ADBQUICKEST(u)
        u[0] = bc0
        u[-1] = bc1
    ax.plot(x,u,'g--',label="ADBQUICKEST")

    # # re-set initial conditions and solve using the quick method
    # u=u0.clone().detach()
    # for n in range(nt+1): 
    #     u = solver.solve_ADBQUICKEST_loop(u)
    #     u[0] = bc0
    #     u[-1] = bc1
    # ax.plot(x,u,'g*',label="ADBQUICKEST-LOOP")

    if uexact is not None: 
        ax.plot(x,uexact,'y',label="exact")


    ax.set_ylim([-0.1,1.3])
    ax.legend()
    pyplot.show()