
import torch

class PoissonSolver:
    """
    Solver class for the Poisson equation with variable coefficients
    """

    def __init__(self, device, h, tol=1e-2, max_cycles=2, nsmoothing=5, w=1, verbose=True):
        """
        """
        self.h2         = h*h
        self.device     = device
        self.n_switch   = 2**20
        self.tol        = tol
        self.max_cycles = max_cycles
        self.nsmoothing = nsmoothing
        self.verbose    = verbose
        self.jcap_tol   = 1e-5
        self.w          = w

    def l2_norm(self, r):
        return torch.sqrt((r[1:-1,1:-1]**2).mean())

    def BC(self, q):
        # q[0, :]    = q[1, :]
        # q[-1, :]   = q[-2, :]
        # q[:, 0]    = q[:, 1]
        # q[:, -1]   = q[:, -2]
        q[0, :]    = 0
        q[-1, :]   = 0
        q[:, 0]    = 0
        q[:, -1]   = 0


    def CG_jacobi_cond(self, f, u, c, h2, maxit=100):

        ch=(c[1:,:]+c[:-1,:])/2 # N x (N+1)
        cv=(c[:,1:]+c[:,:-1])/2 # (N+1) x N

        J=torch.zeros_like(f) # J_{i,j}=c_{i-0.5,j}+c_{i+0.5,j}+c_{i,j-0.5}+c_{i,j+0.5}
        J[:-1,:]+=ch
        J[1:,:]+=ch
        J[:,:-1]+=cv
        J[:,1:]+=cv
        J[0,:]+=1
        J[-1,:]+=1
        J[:,0]+=1
        J[:,-1]+=1
        Jinv=torch.where(J<self.jcap_tol,0,1/J)

        def LU(p):
            res=torch.zeros_like(f)
            res[:-1,:]+=ch*p[1:,:]
            res[1:,:]+=ch*p[:-1,:]
            res[:,:-1]+=cv*p[:,1:]
            res[:,1:]+=cv*p[:,:-1]
            res[0,:]+=p[1,:]
            res[-1,:]+=p[-2,:]
            res[:,0]+=p[:,1]
            res[:,-1]+=p[:,-2]
            return res

        def Au(p):
            return (LU(p)-J*p)/h2

        Jinv=Jinv/h2
        r=torch.where(Jinv==0,0,(f-Au(u)))
        z=r*Jinv
        d=z
        old_norm=torch.tensordot(r,z)
        it=0
        while old_norm>self.tol:
            if it>maxit:
                break
            Ad=Au(d)
            alpha=old_norm/torch.tensordot(d,Ad)
            u=u+alpha*d
            r=r-alpha*Ad
            z=r*Jinv
            new_norm=torch.tensordot(r,z)
            beta=new_norm/old_norm
            d=z+beta*d
            old_norm=new_norm
            it=it+1
        u=u-u.mean()
        return u, r

    def smoothing(self, f, p, c, h2):
        """
        2nd order smoothing with Neumann conditions
        """

        ch=(c[1:,:]+c[:-1,:])/2 # N x (N+1)
        cv=(c[:,1:]+c[:,:-1])/2 # (N+1) x N

        J=torch.zeros_like(f) # J_{i,j}=c_{i-0.5,j}+c_{i+0.5,j}+c_{i,j-0.5}+c_{i,j+0.5}

        J[1:-1,1:-1]+=(ch[1:,1:-1]+ch[:-1,1:-1]+cv[1:-1,1:]+cv[1:-1,:-1])

        # J[0,:]=(ch[0,:]+

        # f[0,:]/=2
        # f[-1,:]/=2
        # f[:,0]/=2
        # f[:,-1]/=2


        # J[:-1,:]+=ch
        # J[1:,:]+=ch
        # J[:,:-1]+=cv
        # J[:,1:]+=cv
        # J[0,:]+=1
        # J[-1,:]+=1
        # J[:,0]+=1
        # J[:,-1]+=1



        # J[1:-1,1:-1]+=(ch[1:,1:-1]+ch[:-1,1:-1]+cv[1:-1,1:]+cv[1:-1,:-1])
        # J[0,:]+=1+ch[0,:]
        # J[-1,:]+=1+ch[-1,:]
        # J[:,0]+=1+cv[:,0]
        # J[:,-1]+=1+cv[:,-1]

        # self.BC(J)



        Jinv=torch.where(J<self.jcap_tol,0,1/J)

        def LU(p):
            res=torch.zeros_like(f)

            res[1:-1,1:-1]+=ch[1:,1:-1]*p[2:,1:-1]
            res[1:-1,1:-1]+=ch[:-1,1:-1]*p[:-2,1:-1]
            res[1:-1,1:-1]+=cv[1:-1,1:]*p[1:-1,2:]
            res[1:-1,1:-1]+=cv[1:-1,:-1]*p[1:-1,:-2]

            # res[:-1,:]+=ch*p[1:,:]
            # res[1:,:]+=ch*p[:-1,:]
            # res[:,:-1]+=cv*p[:,1:]
            # res[:,1:]+=cv*p[:,:-1]

            # res[0,:]+=p[1,:]
            # res[-1,:]+=p[-2,:]
            # res[:,0]+=p[:,1]
            # res[:,-1]+=p[:,-2]

            # res[0,:]+=J[0,:]*p[1,:]
            # res[-1,:]+=J[-1,:]*p[-2,:]
            # res[:,0]+=J[:,0]*p[:,1]
            # res[:,-1]+=J[:,-1]*p[:,-2]

            # self.BC(res)

            return res

        # smoothing
        for _ in range(self.nsmoothing):
            Au=(LU(p)-J*p)/h2
            r=f-Au
            p-=r*h2*Jinv
            # self.BC(p)
            # p=(p-p.mean())

        # r=f-(LU(p)-J*p)/h2

        return p, r

    def restrict(self, r, n):
        r_restrict = torch.zeros(int(n/2)+1, int(n/2)+1, device=self.device)
        r_restrict[1:-1, 1:-1] = (
            0.0625*(r[1:-2:2,1:-2:2]+r[1:-2:2,3:-1:2]+r[3:-1:2,1:-2:2]+r[3:-1:2,3:-1:2])+
            0.125*(r[2:-2:2,1:-2:2]+r[1:-2:2,2:-2:2]+r[3:-1:2,2:-2:2]+r[2:-2:2,3:-1:2])+
            0.25*r[2:-2:2,2:-2:2]
        )
        return r_restrict

    def restrict_simple(self, r):
        return r[::2, ::2]

    def prolong_simple(self, err_coarse, n):
        err = torch.zeros((n+1,n+1), device=self.device)
        err[::2, ::2]  = err_coarse
        err[1::2,::2]  = err_coarse[:-1,:]
        err[::2,1::2]  = err_coarse[:,:-1]
        err[1::2,1::2] = err_coarse[:-1,:-1]
        # self.BC(err)
        return err

    def prolong(self, err_coarse, n):
        err = torch.zeros((n+1,n+1), device=self.device)
        err[::2, ::2]  = err_coarse
        err[1::2,::2]  = 0.5*(err_coarse[1:,:]+err_coarse[:-1,:])
        err[::2,1::2]  = 0.5*(err_coarse[:,1:]+err_coarse[:,:-1])
        err[1::2,1::2] = 0.25*(err_coarse[:-1,:-1]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[1:,1:])
        # self.BC(err)
        return err

    def solve_multigrid(self, f, u, c, c_h, c_v):
        cycle=0
        r_err = 1.e33
        while r_err>self.tol and cycle<self.max_cycles:
            u, r = self.multigrid(f, u, c, c_h, c_v, self.h2)
            r_err_new = self.l2_norm(r)
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, r_err_new))
            # if r_err_new>=r_err:
            #     if self.verbose:
            #         print("Multigrid method cannot get any better!")
            #     break
            #

            cycle+=1
            r_err=r_err_new
        return u, r

    def multigrid(self, f, p, c, c_h, c_v, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n=f.shape[0]-1

        # f-=f.mean()

        if n==2:
            p,r=self.smoothing(f, p, c, h2)

        # if n==2:
        #     p, r = self.CG_jacobi_cond(f, p, c, h2, maxit=100)

        # if n==2:
        #     p[1,1]=f[1,1]*h2/(c[1,1]+1e-15)
        #     r=0

        else:

            if n==self.n_switch and self.device=="cuda":
                f=f.cpu()
                p=p.cpu()
                c=c.cpu()

            p,r=self.smoothing(f, p, c, h2)

            if self.verbose:
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            coarse_residual = self.restrict(r,n)
            c_coarse  = self.restrict(c,n)
            # self.BC(coarse_residual)
            # self.BC(c_coarse)
            ch_coarse = self.restrict_simple(c_h)
            cv_coarse = self.restrict_simple(c_v)

            # computes the coarse error via relaxation
            err_coarse, _ = self.multigrid(
                coarse_residual,
                torch.zeros_like(coarse_residual),
                c_coarse,
                ch_coarse,
                cv_coarse,
                4*h2
                )

            if n==self.n_switch and self.device=="cuda":
                f=f.cuda()
                p=p.cuda()
                c=c.cuda()

            # prolong error
            err=self.prolong(err_coarse, n)

            # correct u by the error
            p+=err

            # Jacobi relaxation
            p,r=self.smoothing(f, p, c, h2)

            if self.verbose:
                err_l2 = self.l2_norm(r)
                print("Multigrid - Steps: {}, Residual: {}".format(n,err_l2))

        return p,r




def test_solvers():
    use_gpu=False
    N=2**8+1

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)


    import poisson_solvers.solutions2d
    from matplotlib import pyplot
    import time

    X, Y, u_exact, f, c, c_h, c_v = poisson_solvers.solutions2d.variable_coeff_c_hat(N,device=device)

    h = X[1,0]-X[0,0]
    print("Number of elements:{}, h={}".format(N, h))

    u0=torch.zeros((N,N))


    solver = PoissonSolver(
        device,
        h,
        verbose=True,
        max_cycles=30,
        nsmoothing=6,
        tol=1e-14
    )

    # c_h = c[1:,:]
    # c_v = c[:,1:]

    c       = c.to(device)
    c_h     = c_h.to(device)
    c_v     = c_v.to(device)
    f       = f.to(device)
    u0      = u0.to(device)
    u_exact = u_exact.to(device)

    start = time.time()


    # u, r = solver.smoothing(f, u0, c, h**2)

    u, r = solver.solve_multigrid(f, u0, c, c_h, c_v)

    # u, r = solver.CG_jacobi_cond(f, u0, c, h**2, maxit=100)

    print("Multigrid method took {}s".format(time.time()-start))


    fig, (ax_1, ax_2, ax_3) = pyplot.subplots(1, 3, figsize=(20,5))

    CS1=ax_1.imshow(u.T,origin = "lower")
    ax_1.set_title("pressure")
    fig.colorbar(CS1)


    CS2=ax_2.imshow(r.T,origin = "lower")
    ax_2.set_title("residual")
    fig.colorbar(CS2)


    if u_exact is not None:
        CS3=ax_3.imshow(u_exact.cpu().T,origin = "lower")
        ax_3.set_title("Exact")
        fig.colorbar(CS3)

    pyplot.show()


if __name__ == "__main__":
    test_solvers()