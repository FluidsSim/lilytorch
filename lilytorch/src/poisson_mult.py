
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
        self.jcap_tol    = 1e-12
        self.n_switch    = 2**16
        self.w           = w # smoothing factor

    def l2_norm(self, r):
        # return torch.tensordot(r,r)/(r.shape[0]*r.shape[1]) #
        return torch.sqrt((r**2).sum())  # L2 norm of the residual

    def BC(self, q):
        q[0, :]    = q[1, :]
        q[-1, :]   = q[-2, :]
        q[:, 0]    = q[:, 1]
        q[:, -1]   = q[:, -2]

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
        Jinv = torch.where(torch.abs(J)<self.jcap_tol,0,1/J)
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


            # restriction
            ch_coarse = 0.5*(ch[::2,1::2]+ch[::2,:-1:2])
            cv_coarse = 0.5*(cv[1::2,::2]+cv[:-1:2,::2])

            # r_coarse  = r[::2,::2]
            r_coarse = (
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
                1,
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

        p=p0.clone().detach()
        for cycle in range(self.max_vcycles):
            p, r = self.vcycle(self.h2*f, p, c, 1, **kwargs)
            r_err = self.l2_norm(r)
            if r_err<self.tol:
                break
        p-=p.mean()
        # p=torch.where(c<self.jcap_tol,0,p-p.mean())
        if self.verbose:
            print("Multigrid residual = {}/{} with {}/{} cycles \n".format(r_err,self.tol, cycle+1, self.max_vcycles))

        # p=p.to(torch.float32)
        return p, r

