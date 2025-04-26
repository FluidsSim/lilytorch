
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
        self.jcap_tol   = 1e-12

    def Neumann_BC(self, q):
        # q[0,0]   = 0
        # q[-1,0]  = 0
        # q[0,-1]  = 0
        # q[-1,-1] = 0
        q[0, :]    = q[1, :]
        q[-1, :]   = q[-2, :]
        q[:, 0]    = q[:, 1]
        q[:, -1]   = q[:, -2]

    def Dirichlet_BC(self, q):
        q[0, :]  = 0
        q[-1, :] = 0
        q[:, 0]  = 0
        q[:, -1] = 0

    def build_operators(self, c, c_h, c_v, h2):
        """
        Matrix-free 2D operators for the Poisson equation with variable coefficients
        c*p=-f <=> nabla(c*nabla(p))=-f
        where c=c(x,y) are linearly interpolated coefficients values at midpoints
        """

        c_h = (c[1:,:]+c[:-1,:])/2 # N x (N+1) - assume that u is (N+1)x(N+1)
        c_v = (c[:,1:]+c[:,:-1])/2 # (N+1) x N

        Jdiag = torch.zeros_like(c) # diagonal Jacobi elements
        Jdiag[1:-1,1:-1] += (c_h[1:,1:-1]+c_h[:-1,1:-1]+c_v[1:-1,1:]+c_v[1:-1,:-1])
        Jdiag_inv = torch.where(Jdiag<self.jcap_tol,0,1/Jdiag)


        # J_diag = c_hat_L[1:,1:-1]+c_hat_L[:-1,1:-1]+c_hat_R[1:-1,1:]+c_hat_R[1:-1,:-1]

        # Jdiag[:-1,:] = c[1:,:] # Nx(N+1)
        # Jdiag[1:,:]  = c[:-1,:]
        # Jdiag[:,:-1] = c[:,1:]
        # Jdiag[:,1:]  = c[:,:-1]
        # Jdiag_inv = torch.where(Jdiag<self.jcap_tol,0,1/Jdiag)

        # build lower-upper matrices (note: this is NOT the LU factorization)
        def LU(p):
            res = torch.zeros_like(p)
            # res[:-1,:] += c_h*p[1:,:]
            # res[1:,:]  += c_h*p[:-1,:]
            # res[:,1:]  += c_v*p[:,1:]
            # res[:,:-1] += c_v*p[:,:-1]

            # res[1:-1,:] += (c_h[:-1,:]*p[:-2,:]+c_h[1:,:]*p[2:,:])
            # res[:,1:-1] += (c_v[:,:-1]*p[:,:-2]+c_v[:,1:]*p[:,2:])


            # res[1:-1,:] += (c_h[1:,:]*p[2:,:]+c_h[:-1,:]*p[:-2,:])
            # res[:,1:-1] += (c_v[:,1:]*p[:,2:]+c_v[:,:-1]*p[:,:-2])

            # res[:-1,:] += c_h*p[1:,:]
            # res[1:,:]  += c_h*p[:-1,:]
            # res[:,:-1] += c_v*p[:,1:]
            # res[:,1:]  += c_v*p[:,:-1]

            res[1:-1,1:-1] += c_h[1:,1:-1]*p[2:,1:-1]
            res[1:-1,1:-1] += c_h[:-1,1:-1]*p[:-2,1:-1]
            res[1:-1,1:-1] += c_v[1:-1,1:]*p[1:-1,2:]
            res[1:-1,1:-1] += c_v[1:-1,:-1]*p[1:-1,:-2]
            return res

        # build A*U operator
        def Au(u):
            # res = (LU(u)-Jdiag*u)

            # res = LU(u)+Jdiag*u
            # res += Jdiag*u
            # res[1:-1,1:-1]-=J_diag*u[1:-1,1:-1]

            # self.BC(res)
            # return res/h2
            return (LU(u)-Jdiag*u)/h2

        return Jdiag_inv, LU, Au



    def CG(self, f, u, c, c_h, c_v , h2, maxit=100):
        _, _, Au = self.build_operators(c, c_h, c_v, h2)
        r=torch.zeros_like(f)
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
        return u

    def CG_jacobi_cond(self, f, u, c, c_h, c_v, h2, maxit=100):
        Jdiag_inv, _, Au = self.build_operators(c, c_h, c_v, h2)
        Jdiag_inv=Jdiag_inv/h2
        r=f-Au(u)
        z=r*Jdiag_inv
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
            z=r*Jdiag_inv
            new_norm=torch.tensordot(r,z)
            beta=new_norm/old_norm
            d=z+beta*d
            old_norm=new_norm
            it=it+1
            self.BC(u)
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

    def prolong_simple(self, err_coarse, n):
        err = torch.zeros((n+1,n+1), device=self.device)
        err[::2, ::2] = err_coarse
        err[1::2,::2] = err_coarse[:]
        # err[1::2,::2] = 0.5*(err_coarse[1:,:]+err_coarse[:-1,:])
        # err[::2,1::2] = 0.5*(err_coarse[:,1:]+err_coarse[:,:-1])
        # err[1::2,1::2] = 0.25*(err_coarse[:-1,:-1]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[1:,1:])
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

    def multigrid(self, f, u, c, c_h, c_v, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n  = f.shape[0]-1

        # if n==2:
        #     u, r = self.CG_jacobi_cond(f, u, c, c_h, c_v, h2, maxit=100)
        #     self.BC(u)

        if n==2:
            u[1,1] = 0.25*f[1,1]*h2/(c[1,1]+1e-15)
            r = 0


        else:

            if n==self.n_switch and self.device=="cuda":
                f=f.cpu()
                u=u.cpu()
                c=c.cpu()

            Jdiag_inv, LU, Au = self.build_operators(c, c_h, c_v, h2)

            def smooth(u):
                # u=u*Jdiag_inv
                for _ in range(self.nsmoothing):
                    u = (LU(u)-f*h2)*Jdiag_inv
                    # u = (f*h2)*Jdiag_inv
                    # u=-(LU(u)-f*h2)*Jdiag_inv
                self.BC(u)

                return u

            # Jacobi relaxation
            u = smooth(u)

            # compute residual
            # r=f-Au(u)
            r = torch.where(Jdiag_inv==0,0,(f-Au(u)))
            r = r-r.mean()

            if self.verbose:
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            # restrict residual
            # coarse_residual = self.restrict(r, n)
            # c_coarse  = self.restrict(c, n)
            # ch_coarse = self.restrict(c_h, n)
            # cv_coarse = self.restrict(c_v, n)

            coarse_residual = self.restrict_simple(r)
            c_coarse  = c[::2,::2] #self.restrict_simple(c)
            ch_coarse = c_h[::2,::2] #self.restrict_simple(c_h)
            cv_coarse = c_v[::2,::2] #self.restrict_simple(c_v)

            self.BC(coarse_residual)
            self.BC(c_coarse)
            self.BC(ch_coarse)
            self.BC(cv_coarse)

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
                u=u.cuda()
                c=c.cuda()

            # prolong error
            err = self.prolong(err_coarse, n)

            # correct u by the error
            u += err

            # Jacobi relaxation
            u = smooth(u)
            r = (f-Au(u))

            if self.verbose:
                print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r[1:-1,1:-1]))))

        return u, r




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

    start = time.time()

    u_cg = solver.CG(f, u0, c, c_h, c_v , h**2, maxit=100).cpu()
    print("CG method took {}s".format(time.time()-start))

    start = time.time()
    u_cg_jac = solver.CG_jacobi_cond(f, u0, c, c_h, c_v , h**2, maxit=1000)[0].cpu()
    print("CG-PREC method took {}s".format(time.time()-start))

    start = time.time()
    u_multigrid = solver.solve_multigrid(f, u0, c, c_h, c_v).cpu()
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
    test_solvers()