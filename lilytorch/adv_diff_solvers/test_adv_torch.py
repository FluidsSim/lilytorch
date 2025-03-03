
import torch
torch.set_num_threads(8)

C=1
C2=3.2

def solve_ADBQUICKEST(u):
    uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1]) 
    phi_w(uw,u)    
    # u*u*u

def psi(rf):
    # ADBQUICKEST
    return torch.max(
        torch.zeros_like(rf),
        torch.min(
            2.0*rf*(1.0-C),
            torch.min(
                (2.0+C2-3.0*C+(1.0-C2)*rf)/(3.0-3.0*C),
                2.0*(1.0-C)*torch.ones_like(rf)
                )
            )
        )

def ADBQUICKEST_rule(phiU, phiD, phiR):
    return torch.where(
        phiD==phiU,
        phiU,
        phiU+0.5*(phiD-phiU)*psi((phiU-phiR)/(phiD-phiU))
    )

def phi_w(F, phi):
    out = torch.zeros((nm2,nm2))
    out[1:,:]=torch.where(
        F[1:,:]>0, 
        ADBQUICKEST_rule(phi[1:-2,1:-1], phi[2:-1,1:-1], phi[:-3,1:-1]),
        ADBQUICKEST_rule(phi[2:-1,1:-1], phi[1:-2,1:-1], phi[3:,1:-1])
    )
    out[0,:]=torch.where(
        F[0,:]>0,
        0.5*(phi[0,1:-1]+phi[1,1:-1]),
        ADBQUICKEST_rule(phi[1,1:-1], phi[0,1:-1], phi[2,1:-1])
    )
    return out


nx  = 2**8
ny  = 2**8
nt  = 1000
u   = torch.zeros(nx,ny)
nm2 = nx-2

import time

start = time.time()
for n in range(nt + 1): 
    solve_ADBQUICKEST(u)
print("ADBquickest solver took {}s".format(time.time()-start))