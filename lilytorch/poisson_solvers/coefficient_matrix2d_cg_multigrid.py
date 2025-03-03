
import torch
import matplotlib.pyplot as plt
import time
import solutions2d

torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

N=2**8+1
h=1/N
print("Number of elements:{}".format(N))
u=torch.rand((N,N))

X, Y, u_exact, f, c = solutions2d.quadratic(N)

def build_operators(c, h2):
    """
    Matrix-free 2D operators for the Poisson equation with variable coefficients
    A*u=f,
    where A=f(x,y)
    c = coefficients
    """
    c_hat_L = (c[1:,:]+c[:-1,:])/2 # N x (N+1) - assume that u is (N+1)x(N+1)
    c_hat_R = (c[:,1:]+c[:,:-1])/2 # (N+1) x N
    c_sum_L = c_hat_L[1:,:]+c_hat_L[:-1,:] # (N-1)x(N+1)
    c_sum_R = c_hat_R[:,1:]+c_hat_R[:,:-1] # (N+1)x(N-1)
    c_sum_star_R = c_sum_L[:,1:-1]+c_sum_R[1:-1,:] # (N-1)x(N-1)
    c_hat_star_L = c_hat_L[1:-1,1:-1] # (N-2)x(N-1)
    c_hat_star_R = c_hat_R[1:-1,1:-1] # (N-1)x(N-2)

    # build diagonal Jacobi preconditioner matrix
    J_el = torch.zeros_like(c) # diagonal Jacobi elements
    J_el[1:-1,1:-1] = c_sum_star_R
    J_el[0,:]       = 2*c[0,:]
    J_el[-1,:]      = 2*c[-1,:]
    J_el[:,0]       = 2*c[:,0]
    J_el[:,-1]      = 2*c[:,-1]

    # build lower-upper matrices (note: this is NOT the LU factorization)
    def LU(u):
        res=torch.zeros_like(u)
        res[1:-2,1:-1]+=c_hat_star_L*u[2:-1,1:-1]
        res[2:-1,1:-1]+=c_hat_star_L*u[1:-2,1:-1]
        res[1:-1,1:-2]+=c_hat_star_R*u[1:-1,2:-1]
        res[1:-1,2:-1]+=c_hat_star_R*u[1:-1,1:-2]
        return res

    # build A*U operator
    def Au(u):
        res = -LU(u)
        res[1:-1,1:-1]+=c_sum_star_R*u[1:-1,1:-1] # add diagonals
        res[0,:]=0
        res[-1,:]=0
        res[:,0]=0
        res[:,-1]=0
        return res/h2
    
    return J_el, LU, Au 


def CG(f, u, c, tol=1e-2, maxit=100):
    h2 = h**2
    _, _, Au = build_operators(c, h2)
    r=torch.zeros_like(f)
    Ad=torch.zeros_like(f)
    r=f-Au(u)
    d=r
    old_norm=torch.tensordot(r,r)
    it=0
    while old_norm>tol:
        if it>maxit:
            print("CG did not converge within {} iterations".format(maxit))
            break
        Ad=Au(d)
        alpha=old_norm/torch.tensordot(d,Ad)
        u=u+alpha*d
        r=r-alpha*Ad
        new_norm=torch.tensordot(r,r)
        beta=new_norm/old_norm
        d=r+d*beta
        old_norm=new_norm
        it=it+1
    print("CG method took {} iterations".format(it-1))
    return u
start = time.time()
u_cg = CG(f, u, c)
print("CG method took {}s".format(time.time()-start))



def CG_jacobi_cond(f, u, c, tol=1e-2, maxit=100):
    h2 = h**2
    J_el, _, Au = build_operators(c, h2)
    J_el=J_el/h2
    r=f-Au(u)
    z=r/J_el
    d=z
    old_norm=torch.tensordot(r,z)
    it=0
    while old_norm>tol:
        if it>maxit:
            print("CG-PREC did not converge within {} iterations".format(maxit))
            break
        Ad=Au(d)
        alpha=old_norm/torch.tensordot(d,Ad)
        u=u+alpha*d
        r=r-alpha*Ad
        z=r/J_el
        new_norm=torch.tensordot(r,z)
        beta=new_norm/old_norm
        d=z+beta*d
        old_norm=new_norm
        it=it+1
    print("CG-PREC jacobi method took {} iterations".format(it-1))
    return u
start = time.time()
u_cg_jac = CG_jacobi_cond(f, u, c)
print("CG-PREC method took {}s".format(time.time()-start))


def restrict(r, n):
    coarse_residual = torch.zeros(int(n/2)+1, int(n/2)+1)
    coarse_residual[1:-1, 1:-1] = 0.0625*(r[1:n-1:2, 1:n-1:2]+r[1:n-1:2, 3:n+1:2]+r[3:n+1:2, 1:n-1:2]+r[3:n+1:2, 3:n+1:2])+\
                0.125*(r[2:n:2, 1:n-1:2]+r[2:n:2, 3:n+1:2]+r[1:n-1:2, 2:n:2]+r[3:n+1:2, 2:n:2])+\
                0.25*r[2:n:2, 2:n:2]
    return coarse_residual

def restrict2(r, n):
    coarse_residual = torch.zeros(int(n/2)+1, int(n/2)+1)
    coarse_residual[1:-1, 1:-1] = r[2:n:2, 2:n:2]
    return coarse_residual


def prolong(err_coarse, n):
    err = torch.zeros((n+1,n+1))
    err[2:(n-1):2, 2:(n-1):2] = err_coarse[1:-1,1:-1]
    err[1:n:2, 2:(n-1):2]     = 0.5*(err_coarse[1:,1:-1]+err_coarse[:-1,1:-1])
    err[2:(n-1):2, 1:n:2]     = 0.5*(err_coarse[1:-1,1:]+err_coarse[1:-1,:-1])
    err[1:n:2, 1:n:2]         = 0.25*(err_coarse[1:,1:]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[:-1,:-1])
    return err
    
def multigrid(f, u, c, m1=3):
    """
    2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape
    """
    n  = f.shape[0]-1
    h  = 1/n
    h2 = h**2

    if n>2:

        J_el, LU, Au = build_operators(c, h2)

        # if n==4:
        #     from IPython import embed; embed()
            
        def jacobi(u):
            # out=torch.zeros_like(u)
            # out[1:-1,1:-1]=(u[2:,1:-1]+u[:-2,1:-1]+u[1:-1,2:]+u[1:-1,:-2]+f[1:-1,1:-1]*h2)/4
            # return out
            return (f*h2+LU(u))/J_el

        # Jacobi relaxation
        for _ in range(m1): 
            u=jacobi(u)

        # compute residual        
        # r = torch.zeros_like(u)
        # r[1:-1,1:-1] = -f[1:-1,1:-1]-(u[2:,1:-1]+u[:-2,1:-1]+u[1:-1,2:]+u[1:-1,:-2]-4*u[1:-1,1:-1])/h2
        r = -f+Au(u)
        # from IPython import embed; embed()

        # restrict residual
        coarse_residual = restrict(r, n)
        c_coarse = restrict(c, n)

        # computes the coarse error via relaxation
        err_coarse = multigrid(coarse_residual, torch.zeros_like(coarse_residual), c_coarse) 

        # prolong error
        err = prolong(err_coarse, n)

        # correct u by the error
        u -= err  

        # Jacobi relaxation
        for _ in range(m1): 
            u=jacobi(u)
        
        print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(r)))

    else:
        u[1,1] = -0.25*f[1,1]*h2

    return u 



start = time.time()
u_multigrid = multigrid(f, u, c, m1=100)
print("Multigrid method took {}s".format(time.time()-start))




fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(1, 4, figsize=(20,5))
CS1=ax_1.contourf(X, Y, u_cg, 20)
ax_1.set_title("CG")
CS2=ax_2.contourf(X, Y, u_cg_jac, 20)
ax_2.set_title("CG Jac")
CS3=ax_3.contourf(X, Y, u_multigrid, 20)
ax_3.set_title("Multigrid")
CS4=ax_4.contourf(X, Y, u_exact, 20)
ax_4.set_title("Exact")
fig.colorbar(CS1)
fig.colorbar(CS2)
fig.colorbar(CS3)
fig.colorbar(CS4)

plt.show()


# from IPython import embed; embed()







