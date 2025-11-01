
import torch

"""
SPUDS QUICK scheme solver for convection (See Leonard 1993) and (Weymouth G., 2015)
"""

N=2**10+1
h=1/N
print("Number of elements:{}".format(N))
u=torch.rand((N,N))
v=torch.rand((N,N))



def advection(u, v):

    uw = 0.5*((u[:-2,1:-1]+u[1:-1,1:-1])) # 0.5*(u_{i-1,j}+u_{i,j})
    ue = 0.5*((u[2:,1:-1]+u[1:-1,1:-1])) # 0.5*(u_{i+1,j}+u_{i,j})
    vs = 0.5*((v[1:-1,:-2]+v[1:-1,1:-1])) 
    vn = 0.5*((v[1:-1,2:]+v[1:-1,1:-1])) 
    
    return (
        u+(uw*phi_w(uw, u)-ue*phi_w(ue,u))+(vs*phi_s(vs,u)-vn*phi_n(vs,u)),
        v+(uw*phi_w(uw, v)-ue*phi_w(ue,v))+(vs*phi_s(vs,v)-vn*phi_n(vs,v))
    )



def phi_w(F, phi):
    bf=0.5*(phi[:-2,1:-1]+phi[1:-1,1:-1]) # (N-1)x(N-1)
    bf[1:-1,1:-1] -= torch.where(
        F[1:-1,1:-1]>0,
        (phi[2:-2,2:-2]-2*phi[1:-3,2:-2]+phi[:-4,2:-2])/6,
        (phi[1:-3,2:-2]-2*phi[2:-2,2:-2]+phi[3:-1,2:-2])/6
    )
    return bf

def phi_e(F, phi):
    bf=0.5*(phi[2:,1:-1]+phi[1:-1,1:-1])
    bf[1:-1,1:-1] -= torch.where(
        F[1:-1,1:-1]>0,
        (phi[1:-3,2:-2]-2*phi[2:-2,2:-2]+phi[3:-1,2:-2])/6,
        (phi[2:-2,2:-2]-2*phi[3:-1,2:-2]+phi[4:,2:-2])/6
    )
    return bf

def phi_s(F, phi):
    bf=0.5*(phi[1:-1,2:]+phi[1:-1,1:-1]) # (N-1)x(N-1)
    bf[1:-1,1:-1] -= torch.where(
        F[1:-1,1:-1]>0,
        (phi[2:-2,2:-2]-2*phi[2:-2,1:-3]+phi[2:-2,:-4])/6,
        (phi[2:-2,1:-3]-2*phi[2:-2,2:-2]+phi[2:-2,3:-1])/6
    )
    return bf

def phi_n(F, phi):
    bf=0.5*(phi[1:-1,2:]+phi[1:-1,1:-1])
    bf[1:-1,1:-1] -= torch.where(
        F[1:-1,1:-1]>0,
        (phi[2:-2,1:-3]-2*phi[2:-2,2:-2]+phi[2:-2,3:-1])/6,
        (phi[2:-2,2:-2]-2*phi[2:-2,3:-1]+phi[2:-2,4:])/6
    )
    return bf

(u_new, v_new) = advection(u, v)

from IPython import embed;embed()
