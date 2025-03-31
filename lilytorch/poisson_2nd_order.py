
import torch

class PoissonSolver:
    """
    Solver class for the Poisson equation with variable coefficients
    """

    def __init__(self, device, h, n, tol=1e-2, max_cycles=2, nsmoothing=5, verbose=True):
        """
        """
        self.h2         = h*h
        self.device     = device
        self.n_switch   = 2**20
        self.tol        = tol
        self.max_cycles = max_cycles
        self.nsmoothing = nsmoothing
        self.verbose    = verbose
        self.jcap_tol   = 1e-12

    def smoothing(self, f, p, c, h2):
        """
        2nd order smoothing with Neumann conditions
        """

        ch=(c[1:,:]+c[:-1,:])/2 # N x (N+1)
        cv=(c[:,1:]+c[:,:-1])/2 # (N+1) x N

        J=torch.zeros_like(f)
        J[1:-1,1:-1]+=(ch[1:,1:-1]+ch[:-1,1:-1]+cv[1:-1,1:]+cv[1:-1,:-1])
        J[0,:]+=1+ch[0]
        J[-1,:]+=1+ch[-1]
        J[:,0]+=1+cv[:,0]
        J[:,-1]+=1+cv[:,-1]
        Jinv=torch.where(J<self.jcap_tol,0,1/J)

        # smoothing
        for _ in range(self.nsmoothing):
            res=torch.zeros_like(f)
            res[:-1,:]+=ch*p[1:,:]
            res[1:,:]+=ch*p[1:,:]
            res[0,:]+=p[1,:]
            res[-1,:]+=p[-2,:]

            res[:,:-1]+=cv*p[:,1:]
            res[:,1:]+=cv*p[:,1:]
            res[:,0]+=p[:,1]
            res[:,-1]+=p[:,-2]

            p=(res-f*h2)*Jinv


        # compute residual
        Au=(res-J*p)/h2
        r=torch.where(Jinv==0,0,(f-Au))

        return p, r

    def restrict_simple(self, r):
        return r[::2, ::2]

    def prolong(self, err_coarse, n):
        err = torch.zeros((n+1,n+1), device=self.device)
        err[::2, ::2]  = err_coarse
        err[1::2,::2]  = 0.5*(err_coarse[1:,:]+err_coarse[:-1,:])
        err[::2,1::2]  = 0.5*(err_coarse[:,1:]+err_coarse[:,:-1])
        err[1::2,1::2] = 0.25*(err_coarse[:-1,:-1]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[1:,1:])
        return err

    def solve_multigrid(self, f, u, c, c_h, c_v):
        cycle=0
        r_err = 1.e33
        while r_err>self.tol and cycle<self.max_cycles:
            u, r = self.multigrid(f, u, c, c_h, c_v, self.h2)
            r_err_new = torch.max(torch.abs(r[1:-1,1:-1]))
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, r_err_new))
            # if r_err_new>=r_err:
            #     if self.verbose:
            #         print("Multigrid method cannot get any better!")
            #     break
            #

            cycle+=1
            r_err=r_err_new
        if self.verbose:
            print("Multigrid residual = {}, ncycles = {} \n".format(r_err_new, cycle))
        return u

    def multigrid(self, f, p, c, c_h, c_v, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n=f.shape[0]-1

        # if n==2:
        #     p, r = self.CG_jacobi_cond(f, p, c, c_h, c_v, h2, maxit=100)
        #     self.BC(p)

        if n==2:
            p[1,1] = 0.25*f[1,1]*h2/(c[1,1]+1e-15)
            r = 0

        else:

            if n==self.n_switch and self.device=="cuda":
                f=f.cpu()
                p=p.cpu()
                c=c.cpu()

            p, r = self.smoothing(f, p, c, h2)

            if self.verbose:
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            coarse_residual = r[::2,::2]
            c_coarse  = c[::2,::2]
            ch_coarse = c_h[::2,::2]
            cv_coarse = c_v[::2,::2]

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
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r[1:-1,1:-1]))))

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
        N,
        verbose=True,
        max_cycles=10,
        nsmoothing=100,
        tol=1e-4
    )

    # c_h = c[1:,:]
    # c_v = c[:,1:]

    c       = c.to(device)
    c_h     = c_h.to(device)
    c_v     = c_v.to(device)
    f       = f.to(device)
    u0      = u0.to(device)
    u_exact = u_exact.to(device)


    # ==== COMPUTE THE SOLUTION ======

    # start = time.time()

    # u_cg = solver.CG(f, u0, c, c_h, c_v , h**2, maxit=100).cpu()
    # print("CG method took {}s".format(time.time()-start))

    # start = time.time()
    # u_cg_jac = solver.CG_jacobi_cond(f, u0, c, c_h, c_v , h**2, maxit=15000)[0].cpu()
    # print("CG-PREC method took {}s".format(time.time()-start))

    start = time.time()
    u_multigrid = solver.solve_multigrid(f, u0, c, c_h, c_v).cpu()
    print("Multigrid method took {}s".format(time.time()-start))


    fig, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(1, 4, figsize=(20,5))

    # CS1=ax_1.imshow(u_cg.T,origin = "lower")
    # ax_1.set_title("CG")
    # fig.colorbar(CS1)

    # CS2=ax_2.imshow(u_cg_jac.T,origin = "lower")
    # ax_2.set_title("CG Jac")
    # fig.colorbar(CS2)

    CS3=ax_3.imshow(u_multigrid.T,origin = "lower")
    ax_3.set_title("Multigrid")
    fig.colorbar(CS3)


    if u_exact is not None:
        CS4=ax_4.imshow(u_exact.cpu().T,origin = "lower")
        ax_4.set_title("Exact")
        fig.colorbar(CS4)

    pyplot.show()


if __name__ == "__main__":
    test_solvers()