
from tkinter import E
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
        self.jcap_tol   = 1e-12
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

    def FD_operator(self, p, ch, cv, h2):
        """
        2nd order finite difference operator
        """
        J=(ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]) # J_{i,j}=c_{i-0.5,j}+c_{i+0.5,j}+c_{i,j-0.5}+c_{i,j+0.5}
        Au=ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2]-J*p[1:-1,1:-1]
        return Au/h2, torch.where(J<self.jcap_tol,0,h2/J)  # returns Au and Jinv

    def Jacobi(self, f, p, ch ,cv, h2):
        """
        Jacobi method
        """
        self.BC(p)
        for i in range(self.nsmoothing):
            Au, Jinv = self.FD_operator(p, ch ,cv, h2)
            r = self.compute_residual(Au,f) # compute residual
            p[1:-1,1:-1] -= self.w*r*Jinv # following A Multigrid Tutorial, 2nd Edition from Briggs, Henson, and McCormick, 2000
            self.BC(p)
        Au, Jinv = self.FD_operator(p, ch ,cv, h2)
        r = self.compute_residual(Au,f) # compute residual
        return p, r

    def compute_residual(self,Au,f):
        r=f-Au
        return r

    def restrict_complex(self, r):
        r_coarse=torch.zeros((r.shape[0]//2,r.shape[1]//2), device=self.device, dtype=self.dtype)
        r_coarse[1::,1::]=(
            0.0625*(r[1:-2:2,1:-2:2]+r[1:-2:2,3::2]+r[3::2,1:-2:2]+r[3::2,3::2])+
            0.125*(r[2:-1:2,1:-2:2]+r[1:-2:2,2:-1:2]+r[3::2,2:-1:2]+r[2:-1:2,3::2])+
            0.25*r[2:-1:2,2:-1:2]
        )
        return r_coarse

    def restrict(self, r):
        r_restrict = 0.25*(
            r[::2, ::2] +
            r[1::2, ::2] +
            r[::2, 1::2] +
            r[1::2, 1::2]
        )
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
        err[1:-1:2,::2]  = 0.5*(err_coarse[1:,:]+err_coarse[:-1,:])
        err[::2,1:-1:2]  = 0.5*(err_coarse[:,1:]+err_coarse[:,:-1])
        err[1:-1:2,1:-1:2] = 0.25*(err_coarse[:-1,:-1]+err_coarse[1:,:-1]+err_coarse[:-1,1:]+err_coarse[1:,1:])

        return err

    def extend_cs(self, ch, cv):

        zc=torch.ones((1,ch.shape[1]), device=self.device, dtype=self.dtype)
        zr=torch.ones((cv.shape[0],1), device=self.device, dtype=self.dtype)
        ch_ext=torch.vstack((zc,ch,zc))
        cv_ext=torch.hstack((zr,cv,zr))


        # # expand c to the size of p
        # zc=torch.ones((1,c.shape[0]), device=self.device, dtype=self.dtype)
        # zr=torch.ones((c.shape[1]+2,1), device=self.device, dtype=self.dtype)

        # c_exp=torch.vstack((zc,c,zc))
        # c_exp=torch.hstack((zr,c_exp,zr))

        # # linearly interpolate to find c_{i-0.5,j} and c_{i,j-0.5}
        # ch = (c_exp[1:,:]+c_exp[:-1,:])/2
        # cv = (c_exp[:,1:]+c_exp[:,:-1])/2
        return ch_ext, cv_ext

    def PCG(self, f, p0, c, ch, cv):

        p=p0.clone()

        ch, cv = self.extend_cs(c)


        Ap,Jinv=self.FD_operator(p, ch, cv, self.h2)
        ri=f-Ap
        z=torch.zeros_like(p)
        z[1:-1,1:-1]=ri*Jinv
        self.BC(z)

        d=z
        old_norm=self.l2_norm(ri)
        cycle=0
        while old_norm>self.tol and cycle<self.max_cycles:

            Ad,Jinv=self.FD_operator(d, ch, cv, self.h2)
            tdot=torch.tensordot(z[1:-1,1:-1],ri)
            alpha=tdot/torch.tensordot(d[1:-1,1:-1],Ad)
            p+=alpha*d
            ri+=-alpha*Ad

            z[1:-1,1:-1]=ri*Jinv
            self.BC(z)

            beta=torch.tensordot(z[1:-1,1:-1],ri)/tdot
            d=z+beta*d

            old_norm=self.l2_norm(ri)
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, old_norm))
            cycle=cycle+1

        return p, ri

    def MPCG(self, f, p0, c, ch, cv):

        zc=torch.ones((1,c.shape[0]), device=self.device, dtype=self.dtype)
        zr=torch.ones((c.shape[1]+2,1), device=self.device, dtype=self.dtype)
        c_ext=torch.vstack((zc,c,zc))
        c_ext=torch.hstack((zr,c_ext,zr))

        ch_ext = (c_ext[1:,:]+c_ext[:-1,:])/2
        cv_ext = (c_ext[:,1:]+c_ext[:,:-1])/2

        p=p0.clone()

        # ch, cv = self.extend_cs(c)

        z,ri =self.multigrid(f, p, c, ch_ext, cv_ext, self.h2)

        d=z
        old_norm=self.l2_norm(ri)
        cycle=0
        while old_norm>self.tol and cycle<self.max_cycles:

            Ad,Jinv=self.FD_operator(d, ch_ext, cv_ext, self.h2)
            tdot=torch.tensordot(z[1:-1,1:-1],ri)
            alpha=tdot/torch.tensordot(d[1:-1,1:-1],Ad)
            p+=alpha*d
            ri+=-alpha*Ad

            z,_ =self.multigrid(ri, torch.zeros_like(z), c, ch_ext, cv_ext, self.h2)

            beta=torch.tensordot(z[1:-1,1:-1],ri)/tdot
            d=z+beta*d

            old_norm=self.l2_norm(ri)
            # if self.verbose:
            #     print("Cycle number = {} - residual = {} \n".format(cycle, old_norm))
            cycle=cycle+1
        if self.verbose:
            print("Poisson equation residual = {}/{} reached with {}/{} cycles \n".format(old_norm,self.tol, cycle, self.max_cycles))

        return p, ri

    def solve_multigrid(self, f, p0, c, c_h, c_v):
        u=p0.clone()
        cycle=0
        r_err = 1.e33
        while r_err>self.tol and cycle<self.max_cycles:
            u, r = self.multigrid(f, u, c, c_h, c_v, self.h2)
            r_err_new = self.l2_norm(r)
            # if self.verbose:
            #     print("Cycle number = {} - residual = {} \n".format(cycle, r_err_new))
            r_err=r_err_new
            cycle+=1
        u-=u.mean()
        if self.verbose:
            print("Poisson equation residual = {}/{} reached with {}/{} cycles \n".format(r_err_new,self.tol, cycle, self.max_cycles))
        return u, r

    def multigrid(self, f, p, c, ch, cv, h2):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        # smoothing
        p, r = self.Jacobi(f, p, ch, cv, h2)

        n=f.shape[0]

        if n>8:

            # if self.verbose:
            #     print("Multigrid - Steps: {}, Residual: {}".format(n, torch.max(torch.abs(r))))

            r_coarse = self.restrict_simple(r)
            c_coarse = self.restrict_simple(c)
            # ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            # cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])
            ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])

            # import matplotlib.pyplot as plt

            # plt.imshow(cv_coarse.cpu())
            # plt.show()

            # computes the coarse error via relaxation
            err_coarse, _ = self.multigrid(
                r_coarse,
                torch.zeros((n//2+2,n//2+2), device=self.device, dtype=self.dtype),
                c_coarse,
                ch_coarse,
                cv_coarse,
                4*h2
                )

            # correct u by the error
            p[1:-1,1:-1]+=self.prolong_simple(err_coarse[1:-1,1:-1])

            # Jacobi relaxation
            p, r = self.Jacobi(f, p, ch, cv, h2)

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

    X, Y, u_exact, f, c, ch, cv = examples.multigrid_course(N, device=device)

    dtype=f.dtype
    u0 = torch.zeros((N+2,N+2),device=device,dtype=dtype)
    h=X[1,0]-X[0,0]

    solver = PoissonSolver(
        dtype,
        device,
        h,
        verbose=True,
        max_cycles=20,
        nsmoothing=10,
        tol=1e-5,
        w=0.8
    )

    # u, r = solver.Jacobi(f, u0, ch, cv, h**2)

    print("#############")
    u, r = solver.solve_multigrid(f, u0, c, ch, cv)

    # print("#############")
    # u, r = solver.PCG(f, u0, c, c, c)

    # print("#############")
    # u, r = solver.MPCG(f, u0, c, ch, cv)


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

        # print("L2 norm diff of exact and approx is {}".format(solver.l2_norm(u_exact-u)))
        fig.colorbar(CS3)

    pyplot.show()


if __name__ == "__main__":
    test_solvers()