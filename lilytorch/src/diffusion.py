
import torch
from pytorch_interp import RegularGridInterpolator

class AdvDiffSolver:
    """
    Solver class for the advection-diffusion equation
    """

    def __init__(self,
                 device,
                 dt,
                 x,
                 y,
                 nu,
                 method="implicit",
                 tol=1e-2,
                 max_cycles=2,
                 max_vcycles=3,
                 nsmoothing=5,
                 w=1,
                 verbose=True
                 ):
        """
        x        : x-domain
        y        : y-domain
        dt       : time step
        nu       : diffusion coefficient
        type     : quick or explicit solver
        """

        self.device=device
        self.dtype=x.dtype

        self.tol         = tol
        self.max_cycles  = max_cycles
        self.max_vcycles = max_vcycles
        self.nsmoothing  = nsmoothing
        self.verbose     = verbose
        self.w           = w # smoothing factor

        self.dt = dt
        dx=x[1]-x[0]
        dy=y[1]-y[0]
        self.dx = dx
        self.dy = dy
        self.dtdx = dt / dx
        self.dtdy = dt / dy
        self.dtdx2 = self.nu*self.dtdx/dx
        self.dtdy2 = self.nu*self.dtdy/dy
        self.nuDT = 0.5*(nu * dt)


        # self.x = x
        # self.y = y
        # self.nx = len(x)
        # self.ny = len(y)
        # self.nm2 = self.nx-2
        # self.ADBzeros = torch.zeros(self.nm2,self.nm2, device=self.device,dtype=self.dtype)
        # self.ADBones = torch.ones(self.nm2,self.nm2, device=self.device,dtype=self.dtype)

        if method == "implicit":
            self.solve = self.solve_implicit
        elif method == "explicit":
            self.solve = self.solve_explicit

        print("Using the {} method for the adv-diff equation".format(method))

    def solve_explicit(self, u, v):
        """
        explicit solver
        """

        u[1:-1,1:-1] += (
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )
        v[1:-1,1:-1] += (
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )
        return (u,v)

    def solve_implicit(self, u, v):
        """
        multigrid implicit solver using Cranc-Nicolson scheme
        """
        for i in range(self.n_smoothing):
            u=self.(1-self.w)*u+self.w*



    def vcycle(self):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n=f.shape[0]

        ch=kwargs.pop("ch", 0.5*(c[1:,1:-1]+c[:-1,1:-1]))
        cv=kwargs.pop("cv", 0.5*(c[1:-1,1:]+c[1:-1,:-1]))

        # smoothing
        p, r = self.Jacobi(f, p, ch, cv, h2)


        if n>8:

            if n==self.n_switch and self.device=="cuda":
                f  = f.cpu()
                p  = p.cpu()
                ch = ch.cpu()
                cv = cv.cpu()
                h2 = h2.cpu()
                r  = r.cpu()


            # restriction
            ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])

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
                torch.zeros((n//2+2,n//2+2), device=p.device, dtype=p.dtype),
                c,
                4*h2,
                ch=ch_coarse,
                cv=cv_coarse,
                )

            # prolongation
            err = torch.zeros((n,n), device=p.device, dtype=p.dtype)
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





class PoissonSolver:
    """
    Solver class for the Poisson equation with variable coefficients
    """

    def l2_norm(self, r):
        return torch.tensordot(r,r)/(r.shape[0]*r.shape[1]) #  torch.sqrt((r**2).sum())  # L2 norm of the residual

    def FD_operator(self, p, ch, cv, h2):
        """
        2nd order finite difference operator
        """
        J  = ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]
        Au = (ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2])-J*p[1:-1,1:-1]
        return Au/h2, J/h2  # returns FDO and Jacobi operators

    def Jacobi(self, f, p, ch ,cv, h2):
        """
        Jacobi method
        """
        self.BC(p)
        for i in range(self.nsmoothing):

            sum  = (ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2])
            J    = ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]
            Jinv = torch.where(torch.abs(J) < self.jcap_tol, torch.zeros_like(J), J.reciprocal())
            p[1:-1,1:-1] = self.w*(-f*h2+sum)*Jinv+(1-self.w)*p[1:-1,1:-1]
            self.BC(p)

        # compute the residual
        sum = (ch[1:,:]*p[2:,1:-1]+ch[:-1,:]*p[:-2,1:-1]+cv[:,1:]*p[1:-1,2:]+cv[:,:-1]*p[1:-1,:-2])
        J   = ch[1:,:]+ch[:-1,:]+cv[:,1:]+cv[:,:-1]
        Au  = (sum-J*p[1:-1,1:-1])/h2
        r   = f-Au
        return p, r


    def vcycle(self, f, p, c, h2, **kwargs):
        """
        2D multigrid solver, assume same grid spacing (n=m), where (n,m)=u.shape with hybrid cpu-gpu implementation
        """
        n=f.shape[0]

        ch=kwargs.pop("ch", 0.5*(c[1:,1:-1]+c[:-1,1:-1]))
        cv=kwargs.pop("cv", 0.5*(c[1:-1,1:]+c[1:-1,:-1]))

        # smoothing
        p, r = self.Jacobi(f, p, ch, cv, h2)


        if n>8:

            if n==self.n_switch and self.device=="cuda":
                f  = f.cpu()
                p  = p.cpu()
                ch = ch.cpu()
                cv = cv.cpu()
                h2 = h2.cpu()
                r  = r.cpu()


            # restriction
            ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])

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
                torch.zeros((n//2+2,n//2+2), device=p.device, dtype=p.dtype),
                c,
                4*h2,
                ch=ch_coarse,
                cv=cv_coarse,
                )

            # prolongation
            err = torch.zeros((n,n), device=p.device, dtype=p.dtype)
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
        f=f.to(torch.float64)
        p0=p0.to(torch.float64)
        c=c.to(torch.float64)

        p=p0.clone().detach()
        for cycle in range(self.max_vcycles):
            p, r = self.vcycle(f, p, c, self.h2, **kwargs)
            r_err = self.l2_norm(r)
            if r_err<self.tol:
                break
        p=torch.where(c<self.jcap_tol, torch.zeros_like(p), p-p.to(torch.float64).mean().to(p.dtype))
        if self.verbose:
            print("Multigrid residual = {}/{} with {}/{} cycles \n".format(r_err,self.tol, cycle+1, self.max_vcycles))

        p=p.to(torch.float32)
        return p, r



