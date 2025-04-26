
import torch

class PoissonSolver:
    """
    Solver class for the Poisson equation with variable coefficients
    """

    def __init__(self, dtype, device, h, tol=1e-2, max_cycles=2, nsmoothing=5, w=1, verbose=True):
        """
        """
        self.dtype      = dtype
        self.h2         = h*h
        self.device     = device
        self.n_switch   = 2**20
        self.tol        = tol
        self.max_cycles = max_cycles
        self.nsmoothing = nsmoothing
        self.verbose    = verbose
        self.jcap_tol   = 1e-5
        self.w          = w # smoothing factor

    def l2_norm(self, r):
        return torch.sqrt((r**2).mean())

    def BC(self, q):
        q[0, :]    = q[1, :]
        q[-1, :]   = q[-2, :]
        q[:, 0]    = q[:, 1]
        q[:, -1]   = q[:, -2]
        # q[0, :]    = -q[1, :]
        # q[-1, :]   = -q[-2, :]
        # q[:, 0]    = -q[:, 1]
        # q[:, -1]   = -q[:, -2]

    def FD_operator(self, u, c, h2):
        """
        2nd order finite difference operator
        """
        return (u[2:,1:-1]+u[:-2,1:-1]+u[1:-1,2:]+u[1:-1,:-2]-4*u[1:-1,1:-1])/h2

    def compute_residual(self, f, u, c, h2):
        """
        Compute the residual of the Poisson equation
        """
        return f-self.FD_operator(u, c, h2)

    def Jacobi(self, f, p, c, h2):
        """
        Jacobi method
        """
        self.BC(p)
        for _ in range(self.nsmoothing):
            r = self.compute_residual(f, p, c, h2)
            p[1:-1,1:-1] -= self.w*r*h2/4 # following A Multigrid Tutorial, 2nd Edition from Briggs, Henson, and McCormick, 2000
            self.BC(p)
        r = self.compute_residual(f, p, c, h2)
        return p, r

    def restrict(self, r):
        # r_restrict = 0.25*(u(2:2:nf(1)  ,2:2:nf(2)  )...
        #                         + u(1:2:nf(1)-1,2:2:nf(2)  )...
        #                         + u(2:2:nf(1)  ,1:2:nf(2)-1)...
        #                         + u(1:2:nf(1)-1,1:2:nf(2)-1));
        r_restrict = 0.25*(
            r[::2, ::2] +
            r[1::2, ::2] +
            r[::2, 1::2] +
            r[1::2, 1::2]
        )


        # r_restrict = (
        #     0.0625*(r[1:-2:2,1:-2:2]+r[1:-2:2,3:-1:2]+r[3:-1:2,1:-2:2]+r[3:-1:2,3:-1:2])+
        #     0.125*(r[2:-2:2,1:-2:2]+r[1:-2:2,2:-2:2]+r[3:-1:2,2:-2:2]+r[2:-2:2,3:-1:2])+
        #     0.25*r[2:-2:2,2:-2:2]
        # )
        return r_restrict

    def restrict_simple(self, r):
        return r[::2, ::2]

    def prolong_simple(self, err_coarse):
        n=err_coarse.shape[0]
        err = torch.zeros((2*n,2*n), device=self.device)
        err[::2, ::2]  = err_coarse
        err[1::2,::2]  = err_coarse
        err[::2,1::2]  = err_coarse
        err[1::2,1::2] = err_coarse
        return err

    def prolong(self, err_coarse):
        n=err_coarse.shape[0]
        err = torch.zeros((2*n,2*n), device=self.device)
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
            r_err_new = self.l2_norm(r)
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, r_err_new))
            cycle+=1
            r_err=r_err_new
        return u, r

    def multigrid(self, f, p, c, c_h, c_v, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """

        # smoothing
        p, r = self.Jacobi(f, p, c, h2)

        n=f.shape[0]

        if n!=2:

            if self.verbose:
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            coarse_residual = self.restrict(r)

            # computes the coarse error via relaxation
            err_coarse, _ = self.multigrid(
                coarse_residual,
                torch.zeros((n//2+2,n//2+2), device=self.device, dtype=self.dtype),
                c,
                c,
                c,
                4*h2
                )

            # correct u by the error
            p[1:-1,1:-1]+=self.prolong_simple(err_coarse[1:-1,1:-1])

            # Jacobi relaxation
            p, r = self.Jacobi(f, p, c, h2)

            if self.verbose:
                err_l2 = self.l2_norm(r)
                print("Multigrid - Steps: {}, Residual: {}".format(n,err_l2))

        return p, r




def test_solvers():
    use_gpu=True
    N=2**8
    dtype=torch.float64

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    from matplotlib import pyplot


    h=1/N
    h2=h*h
    x=torch.linspace(h/2,1-h/2,N,device=device,dtype=dtype)
    y=torch.linspace(h/2,1-h/2,N,device=device,dtype=dtype)
    [X,Y]=torch.meshgrid(x,y, indexing="ij")
    px=1
    py=1

    u_exact=torch.zeros((N+2,N+2),device=device,dtype=dtype)
    u_exact[1:-1,1:-1]=torch.exp(torch.sin(2.0*torch.pi*X/px)*torch.sin(2.0*torch.pi*Y/py))-1

    c=1
    u0 = torch.zeros((N+2,N+2),device=device,dtype=dtype)
    solver = PoissonSolver(
        dtype,
        device,
        h,
        verbose=True,
        max_cycles=100,
        nsmoothing=3,
        tol=1e-14,
        w=0.6
    )
    solver.BC(u_exact)
    f=solver.FD_operator(u_exact, c, h2)

    u, r = solver.solve_multigrid(f, u0, c, c, c)

    # u, r = solver.CG_jacobi_cond(f, u0, c, h**2, maxit=100)

    # print("Multigrid method took {}s".format(time.time()-start))

    f       = f.cpu()
    u_exact = u_exact.cpu()
    u       = u.cpu()
    r       = r.cpu()

    cmap = "Greys_r"
    fig, (ax_1, ax_2, ax_3) = pyplot.subplots(1, 3, figsize=(20,5))

    CS1=ax_1.imshow(u.T,origin = "lower",cmap=cmap)
    ax_1.set_title("pressure")
    fig.colorbar(CS1)


    CS2=ax_2.imshow(r.T,origin = "lower",cmap=cmap)
    ax_2.set_title("residual")
    fig.colorbar(CS2)


    if u_exact is not None:
        CS3=ax_3.imshow(f.cpu().T,origin = "lower",cmap=cmap)
        ax_3.set_title("Exact")
        fig.colorbar(CS3)

    pyplot.show()


if __name__ == "__main__":
    test_solvers()