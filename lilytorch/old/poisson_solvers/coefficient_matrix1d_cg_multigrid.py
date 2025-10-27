

import torch
import matplotlib.pyplot as plt
import cProfile, pstats
profiler = cProfile.Profile()
from torch.profiler import profile, record_function, ProfilerActivity
import time
import numpy as np
import scipy as sp

torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
N=2**8+1
x=torch.linspace(-1,1,N)
h=x[1]-x[0]
h2=(1/h)**2
c=torch.ones_like(x) 

# c=torch.round(torch.exp(-(x**2)), decimals=2)
# f=np.pi*np.pi*torch.sin(np.pi*x)+torch.sin(5*np.pi*x)
f = torch.exp(-(x-0.25)**2)
f[0]=0
f[-1]=0

u_exact=torch.sin(np.pi*x)


c_hat = (c[1:]+c[:-1])/2
c_sum = c_hat[1:]+c_hat[:-1]

# create A using scipy (only for testing)
data = np.array([
    np.concatenate([[1],c_sum,[1]]),
    -np.concatenate([[0],c_hat[1:-1],[0],[0]]),
    -np.concatenate([[0],[0],c_hat[1:-1],[0]])
    ])
diags = np.array([
    0,
    -1,
    1
    ])
Asp = sp.sparse.spdiags(data, diags, N, N)



tone = torch.tensor([1.])
tzero = torch.tensor([0.])
data = torch.cat([
    torch.unsqueeze(torch.concatenate([tone,c_sum,tone]), dim=0), # Diriclet BC
    torch.unsqueeze(-torch.concatenate([tzero,c_hat[1:-1],tzero,tzero]), dim=0), 
    torch.unsqueeze(-torch.concatenate([tzero,tzero,c_hat[1:-1],tzero]), dim=0)
    ], axis=0)
diags = torch.tensor([
    0,
    -1,
    1
    ])
A = torch.sparse.spdiags(data, diags, (N, N))
D_el = h2*torch.concatenate([tone,c_sum,tone])

print(A.to_dense())
print(Asp.todense())

def Au(u):
    res=torch.zeros_like(u)
    res[1:-1]=c_sum*u[1:-1]
    res[1:-2]-=c_hat[1:-1]*u[2:-1]
    res[2:-1]-=c_hat[1:-1]*u[1:-2]
    res[0]=u[0]
    res[-1]=u[-1]
    return res*h2



# ============ SOLVERS ============
start = time.time()
u=torch.linalg.solve(A.to_dense(),f).reshape(N)
print("Inverse method took {}s".format(time.time()-start))



def CG(A, f, u, tol=1e-2):
    r=f-Au(u)
    d=r
    old_norm=torch.dot(r,r)
    it=0
    while torch.sqrt(old_norm)>tol:
        Ad=Au(d)
        alpha=old_norm/torch.dot(d,Ad)
        u=u+alpha*d
        r=r-alpha*Ad
        new_norm=torch.dot(r,r)
        beta=new_norm/old_norm
        d=r+d*beta
        old_norm=new_norm
        it=it+1
    print("CG method took {} iterations".format(it))
    return u
start = time.time()
u_cg = CG(A, f, torch.zeros_like(f))
print("CG method took {}s".format(time.time()-start))





def CG_jacobi_cond(A, d_el, f, u, tol=1e-2):
    r=f-Au(u)
    z=r/d_el
    d=z
    old_norm=torch.dot(r,z)
    it=0
    while torch.sqrt(old_norm)>tol:
        Ad=Au(d)
        alpha=old_norm/torch.dot(d,Ad)
        u=u+alpha*d
        r=r-alpha*Ad
        z=r/d_el
        new_norm=torch.dot(r,z)
        beta=new_norm/old_norm
        d=z+beta*d
        old_norm=new_norm
        it=it+1
    print("CG-PREC jacobi method took {} iterations".format(it))
    return u
start = time.time()
u_cg_jac = CG_jacobi_cond(A, D_el, f, torch.zeros_like(f))
print("CG-PREC method took {}s".format(time.time()-start))





start = time.time()
Msp = sp.sparse.diags(1.0 / Asp.diagonal())
# u_sp = sp.linalg.solve(Asp.todense(), f)
iters = 0
def nonlocal_iterate(arr):
    global iters
    iters+=1
u_sp, exit_code = sp.sparse.linalg.cg(Asp, f, atol=1e-2, M=Msp, callback=nonlocal_iterate)
print("Scipy method took {}s with {} iterations".format(time.time()-start, iters))



plt.subplot(3,1,1)
plt.plot(x, u)

plt.subplot(3,1,2)
plt.plot(x, u_cg_jac)

plt.subplot(3,1,3)
plt.plot(x, u_sp)

plt.show()

# from IPython import embed; embed()

# # torch.matmul(A, u)