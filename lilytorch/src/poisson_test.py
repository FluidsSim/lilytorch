
from tkinter import E
import torch

class PoissonSolver:
    """
    Solver class for the Poisson equation with variable coefficients
    """

    def __init__(self, dtype, device, h, tol=1e-2, max_cycles=2, max_vcycles=3, nsmoothing=5, w=1, verbose=True):
        """
        """
        self.dtype       = dtype
        self.h2          = h*h
        self.device      = device
        self.tol         = tol
        self.max_cycles  = max_cycles
        self.max_vcycles = max_vcycles
        self.nsmoothing  = nsmoothing
        self.verbose     = verbose
        self.jcap_tol    = 1e-10
        self.n_switch    = 2**16
        self.w           = w # smoothing factor

    def l2_norm(self, r):
        return torch.tensordot(r,r)/(r.shape[0]*r.shape[1]) #
        # return torch.sqrt((r**2).sum())  # L2 norm of the residual

    def BC(self, q):
        q[0, :]    = q[1, :]
        q[-1, :]   = q[-2, :]
        q[:, 0]    = q[:, 1]
        q[:, -1]   = q[:, -2]

    def FD_operator(self, p, ch, cv, h2):
        """
        2nd order finite difference operator
        """
        J  = ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]
        Au = (ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2])-J*p[1:-1,1:-1]
        return Au/h2, J/h2  # returns FDO and Jacobi operators

    def compute_sum(self,ch, cv, p):
        """
        compute the sum term in the FD operator
        """
        return ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2]

    def compute_J(self, ch, cv):
        """
        compute the J term in the FD operator
        """
        return ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]


    def Jacobi(self, f, p, ch ,cv, h2):
        """
        Jacobi method
        """
        self.BC(p)
        J = self.compute_J(ch, cv)
        Jinv = torch.where(torch.abs(J)<self.jcap_tol,1,1/J)
        for i in range(self.nsmoothing):
            sum = self.compute_sum(ch, cv, p)
            p[1:-1,1:-1] = self.w*(-f*h2+sum)*Jinv+(1-self.w)*p[1:-1,1:-1]
            self.BC(p)

        # compute the residual
        sum=self.compute_sum(ch, cv, p)
        J=self.compute_J(ch, cv)
        Au  = (sum-J*p[1:-1,1:-1])/h2
        r   = f-Au
        return p, r


    def vcycle(self, f, p, c, h2, **kwargs):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n, m = f.shape

        ch=kwargs.pop("ch", 0.5*(c[1:,1:-1]+c[:-1,1:-1]))
        cv=kwargs.pop("cv", 0.5*(c[1:-1,1:]+c[1:-1,:-1]))

        # smoothing
        p, r = self.Jacobi(f, p, ch, cv, h2)

        if n>8 and m>8:

            if n==self.n_switch and self.device=="cuda":
                f  = f.cpu()
                p  = p.cpu()
                ch = ch.cpu()
                cv = cv.cpu()
                h2 = h2.cpu()
                r  = r.cpu()


            # # restriction
            # ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            # cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])

            # ch_coarse, cv_coarse = self.restrict_coeffs(ch,cv,n)

            ch_coarse = 0.5*(ch[::2,::2]+ch[::2,1::2])
            cv_coarse = 0.5*(cv[::2,::2]+cv[1::2,::2])

            # import matplotlib.pyplot as plt
            # plt.imshow(ch_coarse.cpu().T,origin="lower")
            # plt.colorbar()
            # plt.title("ch coarse")
            # plt.show()

            # r_coarse  = r[::2,::2]
            r_coarse = 0.25*(
                r[::2, ::2] +
                r[1::2, ::2] +
                r[::2, 1::2] +
                r[1::2, 1::2]
            )

            # multigrid cycle on the residual
            err_coarse, _ = self.vcycle(
                r_coarse,
                torch.zeros((n//2+2,m//2+2), device=p.device, dtype=p.dtype),
                c,
                4*h2,
                ch=ch_coarse,
                cv=cv_coarse,
                )

            # prolongation
            err = torch.zeros((n,m), device=p.device, dtype=p.dtype)
            err[::2, ::2]   = err_coarse[1:-1,1:-1]
            err[1::2, ::2]  = err_coarse[1:-1,1:-1]
            err[::2, 1::2]  = err_coarse[1:-1,1:-1]
            err[1::2, 1::2] = err_coarse[1:-1,1:-1]

            # correction
            p[1:-1,1:-1]+=err

            if n== self.n_switch and self.device== "cuda":
               f  = f.cuda()
               p  = p.cuda()
               ch = ch.cuda()
               cv = cv.cuda()
               h2 = h2.cuda()
               r  = r.cuda()

            # Jacobi relaxation
            p, r = self.Jacobi(f, p, ch, cv, h2)

        return p, r


    def solve_multigrid(self, f, p0, c, **kwargs):
        # f=f.to(torch.float64)
        # p0=p0.to(torch.float64)
        # c=c.to(torch.float64)

        p=p0.clone().detach()
        for cycle in range(self.max_vcycles):
            p, r = self.vcycle(f, p, c, self.h2, **kwargs)
            r_err = self.l2_norm(r)
            if r_err<self.tol:
                break
        p-=p.mean()
        # p=torch.where(c<self.jcap_tol,0,p-p.mean())
        if self.verbose:
            print("Multigrid residual = {}/{} with {}/{} cycles \n".format(r_err,self.tol, cycle+1, self.max_vcycles))

        # p=p.to(torch.float32)
        return p, r


    def CG(self, f, p0, c, **kwargs):
        p=p0.clone().detach()

        ch=kwargs.pop("ch", (c[1:,1:-1]+c[:-1,1:-1])/2)
        cv=kwargs.pop("cv", (c[1:-1,1:]+c[1:-1,:-1])/2)

        Au,_=self.FD_operator(p, ch, cv, self.h2)
        r=f-Au
        d=torch.zeros_like(p)
        d[1:-1,1:-1]=r
        old_norm=torch.tensordot(r,r)

        for cycle in range(self.max_cycles+1):

            self.BC(d)

            Ad,_=self.FD_operator(d, ch, cv, self.h2)
            alpha=old_norm/torch.tensordot(d[1:-1,1:-1],Ad)

            p=p+alpha*d
            r=r-alpha*Ad

            if self.l2_norm(r)<self.tol:
                break

            new_norm=torch.tensordot(r,r)

            d[1:-1,1:-1]=r+d[1:-1,1:-1]*(new_norm/old_norm)

            old_norm=new_norm

        # p-=p.mean()
        self.BC(p)

        if self.verbose:
            print("CG residual = {}/{} with {}/{} cycles \n".format(self.l2_norm(r),self.tol, cycle, self.max_cycles))

        return p, r

    def PCG(self, f, p0, c, **kwargs):

        p=p0.clone().detach()

        ch=kwargs.pop("ch", (c[1:,1:-1]+c[:-1,1:-1])/2)
        cv=kwargs.pop("cv", (c[1:-1,1:]+c[1:-1,:-1])/2)

        Au,J=self.FD_operator(p, ch, cv, self.h2)
        r=f-Au
        z, r = self.Jacobi(r, torch.zeros_like(p), ch ,cv, self.h2)

        d=z
        old_norm=torch.tensordot(r,z[1:-1,1:-1])

        for cycle in range(self.max_cycles+1):

            self.BC(d)

            Ad,J=self.FD_operator(d, ch, cv, self.h2)
            Jinv = torch.where(torch.abs(J)<self.jcap_tol,0,1/J)

            alpha=old_norm/torch.tensordot(d[1:-1,1:-1],Ad)

            p=p+alpha*d
            r=r-alpha*Ad

            if self.l2_norm(r)<self.tol:
                break

            z, _=self.Jacobi(r, torch.zeros_like(p), ch ,cv, self.h2)

            new_norm=torch.tensordot(r,z[1:-1,1:-1])

            d=z+d*(new_norm/old_norm)

            old_norm=new_norm

        self.BC(p)
        # p-=p.mean()

        if self.verbose:
            print("PCG residual = {}/{} with {}/{} cycles \n".format(self.l2_norm(r),self.tol, cycle, self.max_cycles))

        return p, r

    def MPCG(self, f, p0, c, **kwargs):

        p=p0.clone().detach()

        ch=kwargs.pop("ch", (c[1:,1:-1]+c[:-1,1:-1])/2)
        cv=kwargs.pop("cv", (c[1:-1,1:]+c[1:-1,:-1])/2)

        Au,_=self.FD_operator(p, ch, cv, self.h2)
        r=f-Au
        # z, r = self.Jacobi(r, torch.zeros_like(p), ch ,cv, self.h2)
        z, _ = self.vcycle(r, torch.zeros_like(p), c, self.h2, ch=ch, cv=cv)

        d = z.clone().detach()
        old_z = z.clone().detach()

        for cycle in range(self.max_cycles+1):

            Ad, _ = self.FD_operator(d, ch, cv, self.h2)

            alpha = torch.tensordot(r, z[1:-1,1:-1]) / torch.tensordot(d[1:-1,1:-1], Ad)

            p = p + alpha * d
            r = r - alpha * Ad

            if self.l2_norm(r) < self.tol:
                break

            z, _ = self.vcycle(r, torch.zeros_like(p), c, self.h2, ch=ch, cv=cv)

            beta = torch.tensordot(z[1:-1,1:-1], r) / torch.tensordot(old_z[1:-1,1:-1], r + alpha * Ad)
            d = z + beta * d

            old_z = z.clone().detach()

        # self.BC(p)
        p-=p.mean()

        if self.verbose:
            print("MPCG residual = {}/{} reached with {}/{} cycles \n".format(self.l2_norm(r),self.tol, cycle, self.max_cycles))

        return p, r



def test_solvers():

    use_gpu=True
    dtype=torch.float32
    N=2**9

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    import solutions2d as examples

    X, Y, u_exact, f, c, ch, cv = examples.multigrid_course(N,device=device)
    u0 = torch.zeros((N+2,N+2),device=device,dtype=dtype)
    h = X[1,0]-X[0,0]  # grid spacing

    solver = PoissonSolver(
        dtype,
        device,
        h,
        verbose=True,
        max_cycles=20,
        max_vcycles=20,
        nsmoothing=50,
        tol=1e-5,
        w=0.9
    )


    u_mg, r_mg = solver.solve_multigrid(f, u0, c, ch=ch, cv=cv)

    u_cg, r_cg = solver.CG(f, u0, c, ch=ch, cv=cv)

    u_pcg, r_pcg = solver.PCG(f, u0, c, ch=ch, cv=cv)

    # u_jac, r_jac = solver.Jacobi(f, u0, ch ,cv, h*h)

    u_mpcg, r_mpcg = solver.MPCG(f, u0, c, ch=ch, cv=cv)


    import matplotlib.pyplot as plt
    cmap = "Greys_r"

    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(20,5))

    CS1=ax_1.imshow(u_mg.cpu().T,origin = "lower",cmap=cmap)
    ax_1.set_title("pressure")
    fig.colorbar(CS1)


    CS2=ax_2.imshow(r_mg.cpu().T,origin = "lower",cmap=cmap)
    ax_2.set_title("residual")
    fig.colorbar(CS2)


    if u_exact is not None:
        CS3=ax_3.imshow(u_exact.cpu().cpu().T,origin = "lower",cmap=cmap)
        ax_3.set_title("Exact")
        fig.colorbar(CS3)



    # plot all solutions
    fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(1, 4, figsize=(20,5))

    CS1=ax_1.imshow(u_mg.cpu().T,origin = "lower",cmap=cmap)
    ax_1.set_title("MG")
    fig.colorbar(CS1)

    CS2=ax_2.imshow(u_cg.cpu().T,origin = "lower",cmap=cmap)
    ax_2.set_title("CG")
    fig.colorbar(CS2)

    CS3=ax_3.imshow(u_pcg.cpu().T,origin = "lower",cmap=cmap)
    ax_3.set_title("PCG")
    fig.colorbar(CS3)

    CS4=ax_4.imshow(u_mpcg.cpu().T,origin = "lower",cmap=cmap)
    ax_4.set_title("MPCG")
    fig.colorbar(CS4)




    # plot all solutions
    fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(1, 4, figsize=(20,5))


    CS1=ax_1.imshow(r_mg.cpu().T,origin = "lower",cmap=cmap)
    ax_1.set_title("MG")
    fig.colorbar(CS1)

    CS2=ax_2.imshow(r_cg.cpu().T,origin = "lower",cmap=cmap)
    ax_2.set_title("CG")
    fig.colorbar(CS2)

    CS3=ax_3.imshow(r_pcg.cpu().T,origin = "lower",cmap=cmap)
    ax_3.set_title("PCG")
    fig.colorbar(CS3)

    CS4=ax_4.imshow(r_mpcg.cpu().T,origin = "lower",cmap=cmap)
    ax_4.set_title("MPCG")
    fig.colorbar(CS4)


    plt.show()


if __name__ == "__main__":
    test_solvers()