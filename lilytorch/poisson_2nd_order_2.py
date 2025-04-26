
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
        self.jcap_tol   = 1e-10
        self.w          = w # smoothing factor

    def l2_norm(self, r):
        return torch.sqrt((r**2).mean())

    def BC(self, q):
        # q[0, :]    = q[1, :]
        # q[-1, :]   = q[-2, :]
        # q[:, 0]    = q[:, 1]
        # q[:, -1]   = q[:, -2]
        q[0, :]    = -q[1, :]
        q[-1, :]   = -q[-2, :]
        q[:, 0]    = -q[:, 1]
        q[:, -1]   = -q[:, -2]

    def FD_operator(self, p, c, h2):
        """
        2nd order finite difference operator
        """
        zc=torch.ones((1,c.shape[0]), device=self.device, dtype=self.dtype)
        zr=torch.ones((c.shape[1],1), device=self.device, dtype=self.dtype)
        ch = torch.vstack((zc,(c[1:,:]+c[:-1,:])/2,zc))
        cv = torch.hstack((zr,(c[:,1:]+c[:,:-1])/2,zr))
        J=(ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]) # J_{i,j}=c_{i-0.5,j}+c_{i+0.5,j}+c_{i,j-0.5}+c_{i,j+0.5}
        Au=ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2]-J*p[1:-1,1:-1]
        return Au/h2, torch.where(J<self.jcap_tol,0,h2/J)  # returns Au and Jinv

        # J  = 4
        # Au = (p[2:,1:-1]+p[:-2,1:-1]+p[1:-1,2:]+p[1:-1,:-2]-J*p[1:-1,1:-1])
        # return Au/h2, h2/J # returns Au and Jinv

    def Jacobi(self, f, p, c, h2):
        """
        Jacobi method
        """
        self.BC(p)
        for i in range(self.nsmoothing):
            Au, Jinv = self.FD_operator(p, c, h2)
            r = f-Au # compute residual
            p[1:-1,1:-1] -= self.w*r*Jinv # following A Multigrid Tutorial, 2nd Edition from Briggs, Henson, and McCormick, 2000
            self.BC(p)
        Au, Jinv = self.FD_operator(p, c, h2)
        r = f-Au # compute residual
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

    def MPCG(self, f, p, c, c_h, c_v):

        Ap,Jinv=self.FD_operator(p, c, self.h2)
        ri=f-Ap
        z=torch.zeros_like(p)
        z[1:-1,1:-1]=ri*Jinv
        self.BC(z)
        # z,r =self.multigrid(f, p, c, c_h, c_v, self.h2)

        d=z
        old_norm=self.l2_norm(ri)
        cycle=0
        while old_norm>self.tol and cycle<self.max_cycles:

            Ad,Jinv=self.FD_operator(d, c, self.h2)
            tdot=torch.tensordot(z[1:-1,1:-1],ri)
            alpha=tdot/torch.tensordot(d[1:-1,1:-1],Ad)
            p+=alpha*d
            ri+=-alpha*Ad

            z[1:-1,1:-1]=ri*Jinv
            # z,_ =self.multigrid(ri, torch.zeros_like(z), c, c_h, c_v, self.h2)
            # z, _ = self.Jacobi(ri, torch.zeros_like(z), c, self.h2)

            beta=torch.tensordot(z[1:-1,1:-1],ri)/tdot
            d=z+beta*d

            old_norm=self.l2_norm(ri)
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, old_norm))
            cycle=cycle+1

        return p, ri

    def solve_multigrid(self, f, u, c, c_h, c_v):
        cycle=0
        r_err = 1.e33
        while r_err>self.tol and cycle<self.max_cycles:
            u, r = self.multigrid(f, u, c, c_h, c_v, self.h2)
            r_err_new = self.l2_norm(r)
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, r_err_new))
            r_err=r_err_new
            cycle+=1
        return u, r

    def multigrid(self, f, p, c, c_h, c_v, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """

        # smoothing
        p, r = self.Jacobi(f, p, c, h2)

        n=f.shape[0]

        if n!=8:

            # if self.verbose:
            #     print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            r_coarse = self.restrict(r)
            c_coarse = self.restrict(c)

            # computes the coarse error via relaxation
            err_coarse, _ = self.multigrid(
                r_coarse,
                torch.zeros((n//2+2,n//2+2), device=self.device, dtype=self.dtype),
                c_coarse,
                c_coarse,
                c_coarse,
                4*h2
                )

            # correct u by the error
            p[1:-1,1:-1]+=self.prolong_simple(err_coarse[1:-1,1:-1])

            # Jacobi relaxation
            p, r = self.Jacobi(f, p, c, h2)

            # if self.verbose:
            #     err_l2 = self.l2_norm(r)
            #     print("Multigrid - Steps: {}, Residual: {}".format(n,err_l2))

        return p, r




def test_solvers():
    use_gpu=False
    N=2**8

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    from matplotlib import pyplot
    import poisson_solvers.solutions2d as examples

    X, Y, u_exact, f, c, _, _ = examples.multigrid_course(N, device=device)

    dtype=f.dtype
    u0 = torch.zeros((N+2,N+2),device=device,dtype=dtype)
    h=X[1,0]-X[0,0]
    h2=h*h

    solver = PoissonSolver(
        dtype,
        device,
        h,
        verbose=True,
        max_cycles=100,
        nsmoothing=10,
        tol=1e-14,
        w=0.6
    )

    # u, r = solver.solve_multigrid(f, u0, c, c, c)
    u, r = solver.MPCG(f, u0, c, c, c)


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
        CS3=ax_3.imshow(u_exact.cpu().T,origin = "lower",cmap=cmap)
        ax_3.set_title("Exact")
        fig.colorbar(CS3)

    pyplot.show()


if __name__ == "__main__":
    test_solvers()