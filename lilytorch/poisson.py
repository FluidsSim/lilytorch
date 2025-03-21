
import torch

class PoissonSolver:
    """
    Solver class for the Poisson equation with variable coefficients
    """

    def __init__(self, device, h, tol=1e-2, max_cycles=2, nsmoothing=5, verbose=True):
        """
        """
        self.h2         = h*h
        self.device     = device
        self.n_switch   = 2**20
        self.BC         = self.Neumann_BC
        self.tol        = tol
        self.max_cycles = max_cycles
        self.nsmoothing = nsmoothing
        self.verbose    = verbose
        self.jcap_tol   = 1e-15

    def Neumann_BC(self, q):
        q[0, :]  = q[1, :]
        q[-1, :] = q[-2, :]
        q[:, 0]  = q[:, 1]
        q[:, -1] = q[:, -2]

    def Dirichlet_BC(self, q):
        q[0, :]  = 0
        q[-1, :] = 0
        q[:, 0]  = 0
        q[:, -1] = 0

    def build_operators(self, c, h2):
        """
        Matrix-free 2D operators for the Poisson equation with variable coefficients
        c*p=-f <=> nabla(c*nabla(p))=-f
        where c=c(x,y) are linearly interpolated coefficients values at midpoints
        """

        c_hat_L = (c[1:,:]+c[:-1,:])/2 # N x (N+1) - assume that u is (N+1)x(N+1)
        c_hat_R = (c[:,1:]+c[:,:-1])/2 # (N+1) x N

        J_diag = c_hat_L[1:,1:-1]+c_hat_L[:-1,1:-1]+c_hat_R[1:-1,1:]+c_hat_R[1:-1,:-1]

        # build diagonal Jacobi preconditioner matrix
        J_el = torch.zeros_like(c) # diagonal Jacobi elements
        J_el[1:-1,1:-1] = J_diag
        J_el_inv = torch.where(J_el<self.jcap_tol,0,1/J_el)

        # build lower-upper matrices (note: this is NOT the LU factorization)
        def LU(u):
            res=torch.zeros_like(u)
            res[:-1,1:-1] += c_hat_L[:,1:-1]*u[1:,1:-1]
            res[1:,1:-1]  += c_hat_L[:,1:-1]*u[:-1,1:-1]
            res[1:-1,:-1] += c_hat_R[1:-1,:]*u[1:-1,1:]
            res[1:-1,1:]  += c_hat_R[1:-1,:]*u[1:-1,:-1]
            self.BC(res)

            return res

        # build A*U operator
        def Au(u):
            res = -LU(u)
            res[1:-1,1:-1]+=J_diag*u[1:-1,1:-1]
            self.BC(res)
            return res/h2
        return J_el_inv, LU, Au



    def CG(self, f, u, c, h2, maxit=100):
        _, _, Au = self.build_operators(c, h2)
        r=torch.zeros_like(f)
        Ad=torch.zeros_like(f)
        r=f-Au(u)
        d=r
        old_norm=torch.tensordot(r,r)
        it=0
        while old_norm>self.tol:
            if it>maxit:
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
        return u, r

    def CG_jacobi_cond(self, f, u, c, h2, maxit=100):
        J_el_inv, _, Au = self.build_operators(c, h2)
        J_el_inv=J_el_inv*h2
        r=f-Au(u)
        z=r*J_el_inv
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
            z=r*J_el_inv
            new_norm=torch.tensordot(r,z)
            beta=new_norm/old_norm
            d=z+beta*d
            old_norm=new_norm
            it=it+1
        return u, r

    def restrict(self, r, n):
        r_restrict = torch.zeros(int(n/2)+1, int(n/2)+1, device=self.device)
        r_restrict[1:-1, 1:-1] = (
            0.0625*(r[1:-2:2,1:-2:2]+r[1:-2:2,3:-1:2]+r[3:-1:2,1:-2:2]+r[3:-1:2,3:-1:2])+
            0.125*(r[2:-2:2,1:-2:2]+r[1:-2:2,2:-2:2]+r[3:-1:2,2:-2:2]+r[2:-2:2,3:-1:2])+
            0.25*r[2:-2:2,2:-2:2]
        )
        r_restrict[0,:]  = r[0, ::2]
        r_restrict[-1,:] = r[-1, ::2]
        r_restrict[:,0]  = r[::2, 0]
        r_restrict[:,-1] = r[::2, -1]
        return r_restrict

    def restrict_simple(self, r):
        r_restrict=r[::2, ::2]
        return r_restrict

    def prolong(self, err_coarse, n):
        err = torch.zeros((n+1,n+1), device=self.device)
        err[::2, ::2] = err_coarse
        err[1::2,::2] = 0.5*(err_coarse[1:,:]+err_coarse[:-1,:])
        err[::2,1::2] = 0.5*(err_coarse[:,1:]+err_coarse[:,:-1])
        err[1::2,1::2] = 0.25*(err_coarse[:-1,:-1]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[1:,1:])
        return err

    def solve_multigrid(self, f, u, c):
        cycle=0
        r_err = 1.e33
        while r_err>self.tol and cycle<self.max_cycles:
            u, r = self.multigrid(f, u, c, self.h2)
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

    def multigrid(self, f, u, c, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n  = f.shape[0]-1

        if n==2:
            u[1,1] = -0.25*f[1,1]*h2/(c[1,1]+1e-15)
            r = 0

        # if n==256:
        #     u, r = self.CG(f, u, c, h2, maxit=100)
        #     self.BC(u)

        else:

            if n==self.n_switch and self.device=="cuda":
                f=f.cpu()
                u=u.cpu()
                c=c.cpu()

            J_el_inv, LU, Au = self.build_operators(c, h2)

            def smooth(u):
                for _ in range(self.nsmoothing):
                    u = (f*h2+LU(u))*J_el_inv

                return u

            # Jacobi relaxation
            u = smooth(u)
            self.BC(u)

            # compute residual
            r = torch.where(J_el_inv==0,0,(f-Au(u)))
            r = r-r.mean()

            # if self.verbose:
            #     print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            # restrict residual
            # coarse_residual = self.restrict(r, n)
            # self.BC(coarse_residual)
            # c_coarse = self.restrict(c, n)
            # self.BC(c_coarse)

            coarse_residual = self.restrict_simple(r)
            self.BC(coarse_residual)
            c_coarse = self.restrict_simple(c)
            self.BC(c_coarse)

            # computes the coarse error via relaxation
            err_coarse, _ = self.multigrid(
                coarse_residual,
                torch.zeros_like(coarse_residual),
                c_coarse,
                4*h2
                )

            if n==self.n_switch and self.device=="cuda":
                f=f.cuda()
                u=u.cuda()
                c=c.cuda()

            # prolong error
            err = self.prolong(err_coarse, n)

            # correct u by the error
            u += err

            # Jacobi relaxation
            u = smooth(u)
            r = (f-Au(u))
            self.BC(u)

            if self.verbose:
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r[1:-1,1:-1]))))


        return u, r

def test_poisson_multiplication():
    use_gpu=False
    N=100
    h=1/N
    c = torch.ones(N,N)
    u = torch.zeros(N,N)
    X,Y = torch.meshgrid(torch.arange(40),torch.arange(40))
    usmall = X**2+Y**2
    u[0:40,0:40] = usmall

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)


    solver = PoissonSolver(
            device,
            h2=1,
            u_exact=None,
            verbose=True
        )
    J_el_inv, LU, Au = solver.build_operators(c, 1)

    res = Au(u).T
    print("Max ={}, idx={}".format(torch.max(res), torch.unravel_index(res.argmax(), res.shape)))

    from matplotlib import pyplot
    pyplot.imshow(res, origin = "lower")
    pyplot.colorbar()



    pyplot.show()


def test_solvers():
    use_gpu=True
    N=2**12+1

    import poisson_solvers.solutions2d
    from matplotlib import pyplot
    import time

    X, Y, u_exact, f, c = poisson_solvers.solutions2d.variable_coeff(N)
    h = X[1,0]-X[0,0]
    print("Number of elements:{}, h={}".format(N, h))

    u0=torch.zeros((N,N))

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    solver = PoissonSolver(
        device,
        h,
        verbose=False,
        nsmoothing=15
    )

    c       = c.to(device)
    f       = f.to(device)
    u0      = u0.to(device)
    u_exact = u_exact.to(device)


    # ==== COMPUTE THE SOLUTION ======

    start = time.time()

    u_cg = solver.CG(f, u0, c, h**2, maxit=100)[0].cpu()
    print("CG method took {}s".format(time.time()-start))

    start = time.time()
    u_cg_jac = solver.CG_jacobi_cond(f, u0, c, h**2, maxit=100)[0].cpu()
    print("CG-PREC method took {}s".format(time.time()-start))

    start = time.time()
    u_multigrid = solver.solve_multigrid(f, u0, c).cpu()
    print("Multigrid method took {}s".format(time.time()-start))


    fig, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(1, 4, figsize=(20,5))
    CS1=ax_1.imshow(u_cg.T,origin = "lower")
    ax_1.set_title("CG")
    fig.colorbar(CS1)

    CS2=ax_2.imshow(u_cg_jac.T,origin = "lower")
    ax_2.set_title("CG Jac")
    fig.colorbar(CS2)

    CS3=ax_3.imshow(u_multigrid.T,origin = "lower")
    ax_3.set_title("Multigrid")
    fig.colorbar(CS3)

    if u_exact is not None:
        CS4=ax_4.imshow(u_exact.cpu().T,origin = "lower")
        ax_4.set_title("Exact")
        fig.colorbar(CS4)

    pyplot.show()


if __name__ == "__main__":
    # test_poisson_multiplication()
    test_solvers()