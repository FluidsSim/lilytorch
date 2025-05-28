
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
        self.n_switch    = 2**20
        self.tol         = tol
        self.max_cycles  = max_cycles
        self.max_vcycles = max_vcycles
        self.nsmoothing  = nsmoothing
        self.verbose     = verbose
        self.jcap_tol    = 1e-12
        self.w           = w # smoothing factor

    def l2_norm(self, r):
        return torch.sqrt((r**2).mean())/(r.shape[0]*r.shape[1])  # L2 norm of the residual

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
        J  = ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]
        Au = -(ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2])+J*p[1:-1,1:-1]
        return Au/h2, J/h2  # returns FDO and Jacobi operators

    def Jacobi(self, f, p, ch ,cv, h2):
        """
        Jacobi method
        """
        self.BC(p)
        for i in range(self.nsmoothing):
            Au, J = self.FD_operator(p, ch ,cv, h2)
            Jinv = torch.where(torch.abs(J)<self.jcap_tol,0,1/J)
            r = f-Au  # compute residual
            p[1:-1,1:-1] += self.w*r*Jinv # following A Multigrid Tutorial, 2nd Edition from Briggs, Henson, and McCormick, 2000
            self.BC(p)
        Au, J = self.FD_operator(p, ch ,cv, h2)
        r = f-Au
        return p, r

    def solve_multigrid(self, f, p0, c, **kwargs):
        p=p0.clone()
        cycle=0
        r_err = 1.e33
        while r_err>self.tol and cycle<self.max_vcycles:
            p, r = self.vcycle(f, p, c, self.h2, **kwargs)
            r_err_new = self.l2_norm(r)
            r_err=r_err_new
            cycle+=1
            p-=p.mean()
            if self.verbose:
                print("Poisson equation residual = {}/{} reached with {}/{} cycles \n".format(r_err_new,self.tol, cycle, self.max_vcycles))
        return p, r

    def vcycle(self, f, p, c, h2, **kwargs):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n=p.shape[0]

        ch=kwargs.pop("ch", (c[1:,1:-1]+c[:-1,1:-1])/2)
        cv=kwargs.pop("cv", (c[1:-1,1:]+c[1:-1,:-1])/2)

        # smoothing
        p, r = self.Jacobi(f, p, ch, cv, h2)

        if n>8:

            # restriction
            ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])
            r_coarse  = r[::2,::2]

            # multigrid cycle on the residual
            err_coarse, _ = self.vcycle(
                r_coarse,
                torch.zeros(((n-2)//2+2,(n-2)//2+2), device=self.device, dtype=self.dtype),
                c,
                4*h2,
                ch=ch_coarse,
                cv=cv_coarse,
                )

            # prolongation
            err = torch.zeros((n-2,n-2), device=self.device, dtype=self.dtype)
            err[::2, ::2]   = err_coarse[1:-1,1:-1]
            err[1::2, ::2]  = err_coarse[1:-1,1:-1]
            err[::2, 1::2]  = err_coarse[1:-1,1:-1]
            err[1::2, 1::2] = err_coarse[1:-1,1:-1]

            # correction
            p[1:-1,1:-1]+=err

            # Jacobi relaxation
            p, r = self.Jacobi(f, p, ch, cv, h2)

        return p, r

    def CG(self, f, p, c, **kwargs):

        ch=kwargs.pop("ch", (c[1:,1:-1]+c[:-1,1:-1])/2)
        cv=kwargs.pop("cv", (c[1:-1,1:]+c[1:-1,:-1])/2)

        Au,_=self.FD_operator(p, ch, cv, self.h2)
        r=torch.zeros_like(p)
        r[1:-1,1:-1]=f-Au
        d=r

        old_norm=torch.tensordot(r,r)
        cycle=0
        while old_norm>self.tol and cycle<3000:
            Ad=torch.zeros_like(p)
            Ad[1:-1,1:-1],_=self.FD_operator(d, ch, cv, self.h2)
            alpha=old_norm/torch.tensordot(d,Ad)

            p=p+alpha*d
            r=r-alpha*Ad

            self.BC(p)
            new_norm=torch.tensordot(r,r)
            beta=new_norm/old_norm
            d=r+d*beta
            
            old_norm=new_norm
            cycle=cycle+1
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, old_norm))

        return p, r


    def PCG(self, f, p, c, **kwargs):

        ch=kwargs.pop("ch", (c[1:,1:-1]+c[:-1,1:-1])/2)
        cv=kwargs.pop("cv", (c[1:-1,1:]+c[1:-1,:-1])/2)

        Au,Jinv=self.FD_operator(p, ch, cv, self.h2)
        ri=f-Au
        z, _ = self.solve_multigrid(ri, torch.zeros_like(p), c, ch=ch, cv=cv)
        # z=torch.zeros_like(p)
        # z[1:-1,1:-1]=ri*Jinv
        # self.BC(z)

        d=z
        old_norm=self.l2_norm(ri)
        cycle=0
        while old_norm>self.tol and cycle<self.max_cycles:

            Ad,Jinv=self.FD_operator(d, ch, cv, self.h2)
            tdot = torch.tensordot(z[1:-1,1:-1],ri)
            alpha=tdot/torch.tensordot(d[1:-1,1:-1],Ad)
            p+=alpha*d
            ri+=-alpha*Ad

            z, _ = self.solve_multigrid(ri, z, c, ch=ch, cv=cv)
            # z[1:-1,1:-1]=ri*Jinv
            # self.BC(z)

            beta=torch.tensordot(z[1:-1,1:-1],ri)/tdot
            d=z+beta*d

            old_norm=self.l2_norm(ri)
            if self.verbose:
                print("Cycle number = {} - residual = {} \n".format(cycle, old_norm))
            cycle=cycle+1

        return p, ri



def test_solvers():
    use_gpu=False
    dtype=torch.float64
    N=2**9

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    xlim=[0,3.2]
    ylim=[0,3.2]
    h=(xlim[1]-xlim[0])/N

    h2=h*h
    x_ext=torch.linspace(xlim[0]-h/2,xlim[1]+h/2,N+2,device=device,dtype=dtype)
    y_ext=torch.linspace(ylim[0]-h/2,ylim[1]+h/2,N+2,device=device,dtype=dtype)


    [X,Y]=torch.meshgrid(x_ext,y_ext, indexing="ij")

    u0 = torch.zeros((N+2,N+2),device=device,dtype=dtype)
    # c  = 1.0+0.5*torch.exp(torch.sin(2.0*torch.pi*X/xlim[-1])*torch.cos(2.0*torch.pi*Y/ylim[-1]))

    c  = torch.ones((N+2,N+2), device=device, dtype=dtype)
    ch = (c[1:,1:-1]+c[:-1,1:-1])/2
    cv = (c[1:-1,1:]+c[1:-1,:-1])/2

    u_exact=torch.exp(torch.cos(2.0*torch.pi*X/xlim[-1])*torch.cos(2.0*torch.pi*Y/ylim[-1]))

    solver = PoissonSolver(
        dtype,
        device,
        h
    )
    solver.BC(u_exact)
    f, _=solver.FD_operator(u_exact, ch, cv, h2)

    dtype=f.dtype
    h=X[1,0]-X[0,0]

    solver = PoissonSolver(
        dtype,
        device,
        h,
        verbose=True,
        max_cycles=10,
        max_vcycles=30,
        nsmoothing=3,
        tol=1e-13,
        w=0.7
    )



    u, r = solver.solve_multigrid(f, u0, c, ch=ch, cv=cv)

    # u, r = solver.PCG(f, u0, c, ch=ch, cv=cv)

    # u, r = solver.CG(f, u0, c, ch=ch, cv=cv)


    f       = f.cpu()
    u_exact = u_exact.cpu()
    u       = u.cpu()
    r       = r.cpu()

    import matplotlib.pyplot as plt
    cmap = "Greys_r"
    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(20,5))

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

    plt.show()


if __name__ == "__main__":
    test_solvers()