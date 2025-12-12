
import torch
import numpy

torch.set_num_threads(16)

def multigrid(f, u, c, w=0.6, m1=300):
    """
    2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape
    """
    n  = f.shape[0]-1
    h  = 1/n
    h2 = h**2


    if n>2:
        
        # Jacobi relaxation
        for _ in range(m1): 
            u[1:-1, 1:-1] = w*(general_relaxation(u)+h2*f[1:-1,1:-1])/4 + (1-w)*u[1:-1, 1:-1]

        # compute residual        
        r = torch.zeros_like(u)
        r[1:-1,1:-1] = -f[1:-1,1:-1]-(general_relaxation(u)-4*u[1:-1,1:-1])/h2

        # restrict residual
        n_coarse = int(n/2)
        coarse_residual = torch.zeros(n_coarse+1, n_coarse+1)
        coarse_residual[1:-1, 1:-1] = 0.0625*(r[1:n-1:2, 1:n-1:2]+r[1:n-1:2, 3:n+1:2]+r[3:n+1:2, 1:n-1:2]+r[3:n+1:2, 3:n+1:2])+\
                    0.125*(r[2:n:2, 1:n-1:2]+r[2:n:2, 3:n+1:2]+r[1:n-1:2, 2:n:2]+r[3:n+1:2, 2:n:2])+\
                    0.25*r[2:n:2, 2:n:2]

        err_coarse0 = torch.zeros_like(coarse_residual)

        # computes the coarse error via relaxation
        err_coarse = multigrid(coarse_residual, err_coarse0, c) 

        # prolong error
        err = torch.zeros_like(u)
        err[2:(n-1):2, 2:(n-1):2] = err_coarse[1:-1,1:-1]
        err[1:n:2, 2:(n-1):2]     = 0.5*(err_coarse[1:,1:-1]+err_coarse[:-1,1:-1])
        err[2:(n-1):2, 1:n:2]     = 0.5*(err_coarse[1:-1,1:]+err_coarse[1:-1,:-1])
        err[1:n:2, 1:n:2]         = 0.25*(err_coarse[1:,1:]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[:-1,:-1])

        # correct u by the error
        u -= err

        # Jacobi relaxation
        for _ in range(m1): 
            u[1:-1, 1:-1] = w*(general_relaxation(u)+h2*f[1:-1,1:-1])/4 + (1-w)*u[1:-1, 1:-1]
        
        print("Steps: {}, Residual: {}".format(n, torch.max(r)))

    else:
        u[1,1] = -0.25*h2*f[1,1]

    return u 


def general_relaxation(m):
    return m[2:,1:-1]+m[:-2,1:-1]+m[1:-1,2:]+m[1:-1,:-2]



if __name__ == "__main__":

    npoints=2**10+1
    # x=torch.linspace(0,1,npoints)
    # y=torch.linspace(0,1,npoints)
    x=torch.linspace(0,1,npoints)
    y=torch.linspace(-0.5,0.5,npoints)    
    h=x[1]-x[0]
    [X,Y]=torch.meshgrid(x,y, indexing="xy")
    c=torch.ones((npoints,npoints))
    f=torch.exp(-(X-0.25)**2-(Y-0.6)**2)

    # f=torch.sin(numpy.pi*X)*torch.cos(numpy.pi*Y)+torch.sin(5*numpy.pi*X)*torch.cos(5*numpy.pi*Y)
    f[0,:]=0
    f[-1,:]=0
    f[:,0]=0
    f[:,-1]=0

    v=torch.rand((npoints,npoints))
    v=torch.zeros((npoints,npoints))
    v[0,:]=0
    v[-1,:]=0
    v[:,-1]=0
    v[:,0]=0

    import time

    start = time.time()
    Z = multigrid(f, v, c)
    print("Multigrid method took {}s".format(time.time()-start))


    
    import matplotlib.pyplot as plt

 

    fig, (ax_1) = plt.subplots(1, 1, figsize=(16,5))
    CS1=ax_1.contourf(X, Y, Z, 20)
    fig.colorbar(CS1)

    plt.show()
