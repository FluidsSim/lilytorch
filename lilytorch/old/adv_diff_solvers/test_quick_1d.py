
import torch

u=torch.rand(2**6+1)
dtdx=0.01

n=len(u)
# v=1
Fw=0.5*((u[1:-1]+u[:-2])) #v*torch.ones((n-2))
Fe=0.5*((u[1:-1]+u[2:])) #v*torch.ones((n-2))
phi_w1=0.5*(u[1:-1]+u[:-2])
phi_w1[1:]-=0.125*torch.where(
    Fw[1:]>0, 
    u[:-3]+u[2:-1]-2*u[1:-2], #u[i-2]+u[i]-2*u[i-1]
    u[1:-2]+u[3:]-2*u[2:-1] #u[i-1]+u[i+1]-2*u[i]
    )
phi_w1[0]-=torch.where(
    Fw[0]>0,
    0,
    0.125*(u[0]+u[2]-2*u[1])
)
phi_e1=0.5*(u[1:-1]+u[2:])
phi_e1[:-1]-=0.125*torch.where(
    Fe[:-1]>0,
    u[0:-3]+u[2:-1]-2*u[1:-2], #u[i-1]+u[i+1]-2*u[i]
    u[1:-2]+u[3:]-2*u[2:-1] #u[i]+u[i+2]-2*u[i+1]
    )
phi_e1[-1]-=torch.where(
    Fe[0]<0,
    0,
    0.125*(u[-3]+u[-1]-2*u[-2]) #u[i]+u[i+2]-2*u[i+1]
)

unew1 = torch.zeros_like(u)
unew1[1:-1] = u[1:-1] - dtdx*(Fe*phi_e1-Fw*phi_w1)

phi_w2=torch.zeros(n-2)
phi_e2=torch.zeros(n-2)
unew2=torch.zeros_like(u)
for i in range(1,n-1):
    Fw=0.5*((u[i]+u[i-1]))
    Fe=0.5*((u[i]+u[i+1]))
    if Fw>0:
        if i==1:
            phi_w2[i-1] = 0.5*(u[i]+u[i-1])
        else:
            phi_w2[i-1] = 0.5*(u[i]+u[i-1])-0.125*(u[i-2]+u[i]-2*u[i-1]) #a*u[i-1]+b*u[i]+c*u[i-2] # p_l=a*p_L+b*p_C+c*p_LL
    else:
        phi_w2[i-1] = 0.5*(u[i]+u[i-1])-0.125*(u[i-1]+u[i+1]-2*u[i]) #a*u[i]+b*u[i-1]+c*u[i+1] # p_l=a*p_C+b*p_L+c*p_R

    if Fe>0:
        phi_e2[i-1] = 0.5*(u[i]+u[i+1])-0.125*(u[i-1]+u[i+1]-2*u[i]) #0.5*(u[i]+u[i+1])-0.125*(u[i-1]+u[i+1]-2*u[i]) #a*u[i]+b*u[i+1]+c*u[i-1] # p_r=a*p_C+b*p_R+c*p_L
    else:
        if i==n-2:
            phi_e2[i-1] = 0.5*(u[i]+u[i+1])
        else:
            phi_e2[i-1] = 0.5*(u[i]+u[i+1])-0.125*(u[i]+u[i+2]-2*u[i+1]) #a*u[i+1]+b*u[i]+c*u[i+2] # p_r=a*p_R+b*p_C+c*p_RR
    unew2[i] = u[i] - dtdx*(Fe*phi_e2[i-1]-Fw*phi_w2[i-1])

print(torch.equal(phi_w1, phi_w2))
print(torch.equal(phi_e1, phi_e2))
print(torch.equal(unew1, unew2))

