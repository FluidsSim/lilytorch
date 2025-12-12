
import torch
from pytorch_interpolation import RegularGridInterpolator

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
                 BC_type_u=["D","D","D","D"], # w,o,s,n
                 BC_values_u=[0,0,0,0],
                 BC_type_v=["D","D","D","D"], # w,o,s,n
                 BC_values_v=[0,0,0,0],
                 method="implicit",
                 ):
        """
        x        : x-domain
        y        : y-domain
        dt       : time step
        nu       : diffusion coefficient
        type     : solver type: implicit, explicit, quick, abdquickest, adam_bashforth
        """

        self.device=device
        self.dtype=x.dtype

        self.dt = dt
        dx=x[1]-x[0]
        dy=y[1]-y[0]
        self.dx = float(dx)
        self.dy = float(dy)
        self.dtdx = dt / dx
        self.dtdy = dt / dy
        self.dtdx2 = self.dtdx/dx
        self.dtdy2 = self.dtdy/dy
        self.nu = nu

        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.nm2x = self.nx-2
        self.nm2y = self.ny-2

        # dummy initialization for the abdquickest solver
        self.C = 0.1
        self.C2 = self.C**2


        self.BC_type_u   = BC_type_u
        self.BC_values_u = BC_values_u
        self.BC_type_v   = BC_type_v
        self.BC_values_v = BC_values_v

        if method == "implicit":
            self.solve = self.solve_implicit
            # dummy initialization for the implicit solver
            x_staggered = x - dx/2
            y_staggered = y - dy/2
            self.gu = RegularGridInterpolator((x_staggered,y), torch.zeros((self.nx,self.ny), device=self.device,dtype=self.dtype), fill_value=None)
            self.gv = RegularGridInterpolator((x,y_staggered), torch.zeros((self.nx,self.ny), device=self.device,dtype=self.dtype), fill_value=None)
            self.X_u, self.Y_u = torch.meshgrid(x_staggered,y, indexing="ij")
            self.X_v, self.Y_v = torch.meshgrid(x,y_staggered, indexing="ij")
            self.X_cc, self.Y_cc = torch.meshgrid(x_staggered,y_staggered, indexing="ij")
            self.xflat = self.X_u.flatten().clone().detach()
            self.yflat = self.Y_v.flatten().clone().detach()
        elif method == "explicit":
            self.solve = self.solve_explicit
        elif method == "quick":
            self.solve = self.solve_FLUXLMT
        elif method == "abdquickest":
            self.solve = self.solve_ADBQUICKEST
        elif method == "adam-bashforth":
            self.solve = self.solve_adam_bashforth
            self.HU_prec = torch.zeros((self.nm2x,self.nm2y), device=self.device,dtype=self.dtype)
            self.HV_prec = torch.zeros((self.nm2x,self.nm2y), device=self.device,dtype=self.dtype)
        elif method == "quick-waterlily":
            self.solve = self.solve_quick_waterlily
        else:
            raise("Error: the convection solver method {} does not exist".format(method))

        print("Using the {} method for the adv-diff equation".format(method))

    def clf(self, u, v):
        vel_max = torch.max(
            torch.max(torch.abs(u)),
            torch.max(torch.abs(v))
            )
        self.dt = self.dx/(vel_max*+3*self.nu)

    def solve_explicit(self, u, v, iteration=0):
        """
        explicit solver
        """

        u[1:-1, 1:-1] = (
            u[1:-1, 1:-1]-
            self.dtdx*u[1:-1,1:-1]*(u[1:-1,1:-1]-u[:-2,1:-1]) -
            self.dtdy*v[1:-1,1:-1]*(u[1:-1,1:-1]-u[1:-1,:-2]) +
            self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
            self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
        )
        v[1:-1, 1:-1] = (
            v[1:-1, 1:-1]-
            self.dtdx*u[1:-1,1:-1]*(v[1:-1,1:-1]-v[:-2,1:-1]) -
            self.dtdy*v[1:-1,1:-1]*(v[1:-1,1:-1]-v[1:-1,:-2]) +
            self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
            self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
        )

        return (u,v)

    def med(self, a, b, c):
        return torch.max(torch.min(a,b), torch.min(torch.max(a,b), c))

    def FLUXLMT_rule(self, bf, phiU, phiD, phiR): # correspond to (C,D,U) in LilyPad notation
        bf -= (phiD-2*phiU+phiR)/6
        b1 = phiR+10*(phiU-phiR)
        return self.med(bf, phiU, self.med(phiU, phiD, b1))

    def phi_wLMT(self, F, phi):
        bf=0.5*(phi[:-2,1:-1]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[1:-3,2:-2],phi[2:-2,2:-2],phi[:-4,2:-2]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[1:-3,2:-2],phi[3:-1,2:-2]),
        )
        return bf

    def phi_sLMT(self, F, phi):
        bf=0.5*(phi[1:-1,:-2]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,1:-3],phi[2:-2,2:-2],phi[2:-2,:-4]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[1:-3,2:-2],phi[2:-2,3:-1]),
        )
        return bf

    def phi_eLMT(self, F, phi):
        bf=0.5*(phi[2:,1:-1]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[3:-1,2:-2],phi[1:-3,2:-2]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[3:-1,2:-2],phi[2:-2,2:-2],phi[4:,2:-2]),
        )
        return bf

    def phi_nLMT(self, F, phi):
        bf=0.5*(phi[1:-1,2:]+phi[1:-1,1:-1])
        bf[1:-1,1:-1] = torch.where(
            F[1:-1,1:-1]>0,
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,2:-2],phi[2:-2,3:-1],phi[2:-2,1:-3]),
            self.FLUXLMT_rule(bf[1:-1,1:-1], phi[2:-2,3:-1],phi[2:-2,2:-2],phi[2:-2,4:]),
        )
        return bf

    def solve_FLUXLMT(self, u, v, iteration=0):

        uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1])
        ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1])
        vs = 0.5*(v[1:-1,:-2]+v[1:-1,1:-1])
        vn = 0.5*(v[1:-1,2:]+v[1:-1,1:-1])

        u[1:-1,1:-1] = (
                        u[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_wLMT(uw,u)-ue*self.phi_eLMT(ue,u))+
                        self.dtdy*(vs*self.phi_sLMT(vs,u)-vn*self.phi_nLMT(vn,u))+
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )

        v[1:-1,1:-1] = (
                        v[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_wLMT(uw,v)-ue*self.phi_eLMT(ue,v))+
                        self.dtdy*(vs*self.phi_sLMT(vs,v)-vn*self.phi_nLMT(vn,v))+
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )

        return (u,v)

    def psi(self, rf):
        # ADBQUICKEST
        return torch.max(
            torch.zeros_like(rf, device=self.device),
            torch.min(
                2.0*rf*(1.0-self.C),
                torch.min(
                    (2.0+self.C2-3.0*self.C+(1.0-self.C2)*rf)/(3.0-3.0*self.C),
                    2.0*(1.0-self.C)*torch.ones_like(rf, device=self.device)
                    )
                )
            )
        # # CUBISTA
        # return torch.max(
        #     torch.zeros_like(rf),
        #     torch.min(
        #         (3/2)*rf,
        #         torch.min(
        #             (3/4)*rf+1/4,
        #             (3/2)*torch.ones_like(rf)
        #             )
        #         )
        #     )

    def ADBQUICKEST_rule(self, phiU, phiD, phiR):
        return torch.where(
            phiD==phiU,
            phiU,
            phiU+0.5*(phiD-phiU)*self.psi((phiU-phiR)/(phiD-phiU))
        )

    def phi_w(self, F, phi):
        out = torch.zeros((self.nm2x,self.nm2y), device=self.device)
        out[1:,:]=torch.where(
            F[1:,:]>0,
            self.ADBQUICKEST_rule(phi[1:-2,1:-1], phi[2:-1,1:-1], phi[:-3,1:-1]),
            self.ADBQUICKEST_rule(phi[2:-1,1:-1], phi[1:-2,1:-1], phi[3:,1:-1])
        )
        out[0,:]=torch.where(
            F[0,:]>0,
            0.5*(phi[0,1:-1]+phi[1,1:-1]),
            self.ADBQUICKEST_rule(phi[1,1:-1], phi[0,1:-1], phi[2,1:-1])
        )
        return out

    def phi_s(self, F, phi):
        out = torch.zeros((self.nm2x,self.nm2y), device=self.device)
        out[:,1:]=torch.where(
            F[:,1:]>0,
            self.ADBQUICKEST_rule(phi[1:-1,1:-2], phi[1:-1,2:-1], phi[1:-1,:-3]),
            self.ADBQUICKEST_rule(phi[1:-1,2:-1], phi[1:-1,1:-2], phi[1:-1,3:])
        )
        out[:,0]=torch.where(
            F[:,0]>0,
            0.5*(phi[1:-1,0]+phi[1:-1,1]),
            self.ADBQUICKEST_rule(phi[1:-1,1], phi[1:-1,0], phi[1:-1,2])
        )
        return out

    def phi_e(self, F, phi):
        out = torch.zeros((self.nm2x,self.nm2y), device=self.device)
        out[:-1,:]=torch.where(
            F[:-1,:]>0,
            self.ADBQUICKEST_rule(phi[1:-2,1:-1], phi[2:-1,1:-1], phi[:-3,1:-1]),
            self.ADBQUICKEST_rule(phi[2:-1,1:-1], phi[1:-2,1:-1], phi[3:,1:-1])
        )
        out[-1,:]=torch.where(
            F[0,:]<0,
            0.5*(phi[-1,1:-1]+phi[-2,1:-1]),
            self.ADBQUICKEST_rule(phi[-2,1:-1], phi[-1,1:-1], phi[-3,1:-1])
        )
        return out

    def phi_n(self, F, phi):
        out = torch.zeros((self.nm2x,self.nm2y), device=self.device)
        out[:,:-1]=torch.where(
            F[:,:-1]>0,
            self.ADBQUICKEST_rule(phi[1:-1,1:-2], phi[1:-1,2:-1], phi[1:-1,:-3]),
            self.ADBQUICKEST_rule(phi[1:-1,2:-1], phi[1:-1,1:-2], phi[1:-1,3:])
        )
        out[:,-1]=torch.where(
            F[:,0]<0,
            0.5*(phi[1:-1,-1]+phi[1:-1,-2]),
            self.ADBQUICKEST_rule(phi[1:-1,-2], phi[1:-1,-1], phi[1:-1,-3])
        )
        return out

    def solve_ADBQUICKEST(self, u, v, iteration=0):
        # following lilipad notation
        # uo = 0.5*(x.a[i-1][j]+x.a[i][j]);
        # ue = 0.5*(x.a[i+1][j]+x.a[i][j]);
        # vs = 0.5*(y.a[i][j]+y.a[i-1][j]);
        # vn = 0.5*(y.a[i][j+1]+y.a[i-1][j+1]);
        u_new=torch.zeros_like(u)
        v_new=torch.zeros_like(v)

        uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1])
        ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1])
        vs = 0.5*(v[:-2,1:-1]+v[1:-1,1:-1])
        vn = 0.5*(v[1:-1,2:]+v[:-2,2:])
        u_new[1:-1,1:-1] = (
                        u[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_w(uw,u)-ue*self.phi_e(ue,u))+
                        self.dtdy*(vs*self.phi_s(vs,u)-vn*self.phi_n(vn,u))+
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )

        # uo = 0.5*(x.a[i][j-1]+x.a[i][j]);
        # ue = 0.5*(x.a[i+1][j-1]+x.a[i+1][j]);
        # vs = 0.5*(y.a[i][j-1]+y.a[i][j]);
        # vn = 0.5*(y.a[i][j]+y.a[i][j+1]);
        uw = 0.5*(u[1:-1,:-2]+u[1:-1,1:-1])
        ue = 0.5*(u[2:,:-2]+u[2:,1:-1])
        vs = 0.5*(v[1:-1,:-2]+v[1:-1,1:-1])
        vn = 0.5*(v[1:-1,1:-1]+v[1:-1,2:])
        v_new[1:-1,1:-1] = (
                        v[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_w(uw,v)-ue*self.phi_e(ue,v))+
                        self.dtdy*(vs*self.phi_s(vs,v)-vn*self.phi_n(vn,v))+
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )

        return (u_new,v_new)



    def solve_implicit(self, u, v, iteration=0):
        """
        Implicit solver based on Staam, 1999 where
        u_new(x,y) = u(x-dt*u(x,y), y-dt*v(x,y)) [linearly interpolated]
        """
        xold = self.xflat-u.flatten()*self.dt
        yold = self.yflat-v.flatten()*self.dt

        self.gu.F = u
        u = self.gu(xold, yold).reshape((self.nx,self.ny)).clone().detach()

        self.gv.F = v
        v = self.gv(xold, yold).reshape((self.nx,self.ny)).clone().detach()

        u[1:-1,1:-1] += (
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )
        v[1:-1,1:-1] += (
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )
        return (u,v)

    def solve_adam_bashforth(self, u, v, iteration=0):

        u_new=torch.zeros_like(u)
        v_new=torch.zeros_like(v)

        uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1])
        ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1])
        us = 0.5*(u[1:-1,:-2]+u[1:-1,1:-1])
        un = 0.5*(u[1:-1,2:]+u[1:-1,1:-1])
        fw = uw
        fe = ue
        fs = 0.5*(v[:-2,1:-1]+v[1:-1,1:-1])
        fn = 0.5*(v[:-2,2:]+v[1:-1,2:])
        HU_new = uw*fw-ue*fe+us*fs-un*fn

        vw = 0.5*(v[:-2,1:-1]+v[1:-1,1:-1])
        ve = 0.5*(v[2:,1:-1]+v[1:-1,1:-1])
        vs = 0.5*(v[1:-1,:-2]+v[1:-1,1:-1])
        vn = 0.5*(v[1:-1,2:]+v[1:-1,1:-1])
        fw = 0.5*(u[1:-1,1:-1]+u[1:-1, :-2])
        fe = 0.5*(u[2:,1:-1]+u[2:,:-2])
        fs = vs
        fn = vn
        HV_new = vw*fw-ve*fe+vs*fs-vn*fn

        if iteration==0:
            u_new[1:-1,1:-1] = u[1:-1,1:-1]+self.dt*HU_new/self.dx
            v_new[1:-1,1:-1] = v[1:-1,1:-1]+self.dt*HV_new/self.dy

        else:
            u_new[1:-1,1:-1] = u[1:-1,1:-1]+self.dt*0.5*(3*HU_new - self.HU_prec)/self.dx
            v_new[1:-1,1:-1] = v[1:-1,1:-1]+self.dt*0.5*(3*HV_new - self.HV_prec)/self.dy

        self.HU_prec = HU_new.clone().detach()
        self.HV_prec = HV_new.clone().detach()



        u_new[1:-1,1:-1] += (
                self.nu*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                self.nu*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                )*self.dtdx2
        v_new[1:-1,1:-1] += (
                self.nu*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                self.nu*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                )*self.dtdy2


        return (u_new,v_new)

    def median(self, a, b, c):
        return torch.maximum(
            torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c)
        )

    def lam(self, u, c, d):
        return self.median((5.0*c+2.0*d-u)/6,c,self.median(10.0*c-9.0*u,c,d))

    def phi_U(self, u):
        fw = 0.5*(u[1:-2,1:-1]+u[2:-1,1:-1]) # west flux
        return u[2:-1,1:-1]*torch.where(
            fw>0,
            self.lam(u[:-3,1:-1],u[1:-2,1:-1],u[2:-1,1:-1]),
            self.lam(u[3:,1:-1],u[2:-1,1:-1],u[1:-2,1:-1])
        )


    def solve_quick_waterlily(self, u, v):


        u_new=torch.zeros_like(u)
        v_new=torch.zeros_like(v)


        # uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1])
        # ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1])
        # u_new[1:-1,1:-1]+=self.dtdx*(uw*self.phi_w(uw,u)-ue*self.phi_e(ue,u))

        # ==================== u convection ==============================
        # lower boundary - i=1
        fw = 0.5*(u[0,1:-1]+u[1,1:-1]) # west flux at left boundary
        phi_u = self.dtdx*u[1,1:-1]*torch.where(
            fw>0,
            fw,
            self.lam(u[2,1:-1],u[1,1:-1],u[0,1:-1])
        )
        u_new[1,1:-1] += phi_u

        # inner points
        fw = 0.5*(u[1:-2,1:-1]+u[2:-1,1:-1]) # west flux inside
        phi_u = self.dtdx*u[2:-1,1:-1]*torch.where(
            fw>0,
            self.lam(u[:-3,1:-1],u[1:-2,1:-1],u[2:-1,1:-1]),
            self.lam(u[3:,1:-1],u[2:-1,1:-1],u[1:-2,1:-1])
        )
        u_new[2:-1,1:-1] += phi_u
        u_new[1:-2,1:-1] -= phi_u

        # upper boundary
        fw = 0.5*(u[-1,1:-1]+u[-2,1:-1]) # west flux at right boundary
        phi_u = self.dtdx*u[-1,1:-1]*torch.where(
            fw<0,
            fw,
            self.lam(u[-3,1:-1],u[-2,1:-1],u[-1,1:-1]),
        )
        u_new[-2,1:-1] -= phi_u



        # fs = 0.5*(v[0,1:-1]+v[1,1:-1])
        # phi_u = self.dtdx*u[1,1:-1]*torch.where(
        #     fs<0,
        #     self.lam(u[1:-1, :-2],u[1:-1,1:-1],u[1:-1,2:]),
        #     self.lam(u[1:-1,2:],u[1:-1,1:-1],u[1:-1,:-2])
        # )
        # u_new[1:-1,1:-1] += phi_u


        # ============ v convection ==============================








        vs = 0.5*(v[:-2,1:-1]+v[1:-1,1:-1])
        vn = 0.5*(v[1:-1,2:]+v[:-2,2:])
        u_new[1:-1,1:-1] += (
                        u[1:-1,1:-1]+
                        self.dtdy*(vs*self.phi_s(vs,u)-vn*self.phi_n(vn,u))+
                        self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
                        self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
                        )

        # u_new=torch.zeros_like(u)
        # v_new=torch.zeros_like(v)

        # uw = 0.5*(u[:-2,1:-1]+u[1:-1,1:-1])
        # ue = 0.5*(u[2:,1:-1]+u[1:-1,1:-1])
        # vs = 0.5*(v[:-2,1:-1]+v[1:-1,1:-1])
        # vn = 0.5*(v[1:-1,2:]+v[:-2,2:])
        # u_new[1:-1,1:-1] = (
        #                 u[1:-1,1:-1]+
        #                 self.dtdx*(uw*self.phi_w(uw,u)-ue*self.phi_e(ue,u))+
        #                 self.dtdy*(vs*self.phi_s(vs,u)-vn*self.phi_n(vn,u))+
        #                 self.nu*self.dtdx2*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1]) +
        #                 self.nu*self.dtdx2*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])
        #                 )


        uw = 0.5*(u[1:-1,:-2]+u[1:-1,1:-1])
        ue = 0.5*(u[2:,:-2]+u[2:,1:-1])
        vs = 0.5*(v[1:-1,:-2]+v[1:-1,1:-1])
        vn = 0.5*(v[1:-1,1:-1]+v[1:-1,2:])
        v_new[1:-1,1:-1] = (
                        v[1:-1,1:-1]+
                        self.dtdx*(uw*self.phi_w(uw,v)-ue*self.phi_e(ue,v))+
                        self.dtdy*(vs*self.phi_s(vs,v)-vn*self.phi_n(vn,v))+
                        self.nu*self.dtdx2*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1]) +
                        self.nu*self.dtdx2*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])
                        )
        return (u_new,v_new)



    def set_BCs(self, u, v):

        u[:,0]  = u[:,1]
        u[-1,:] = u[-2,:]
        u[0,:]  = u[1,:]
        u[:,-1] = u[:,-2]

        v[:,0]  = v[:,1]
        v[-1,:] = v[-2,:]
        v[0,:]  = v[1,:]
        v[:,-1] = v[:,-2]

        if self.BC_type_u[0]=="D":
            u[0, :]=self.BC_values_u[0]
        if self.BC_type_u[1]=="D":
            u[-1,:]=self.BC_values_u[1]
        # if self.BC_type_u[2]=="D":
        #     u[:,0]=self.BC_values_u[2]
        # if self.BC_type_u[3]=="D":
        #     u[:,-1]=self.BC_values_u[3]

        if self.BC_type_v[2]=="D":
            v[:,0]=self.BC_values_v[2]
        if self.BC_type_v[3]=="D":
            v[:,-1]=self.BC_values_v[3]
        # if self.BC_type_v[0]=="D":
        #     v[0, :]=self.BC_values_v[0]
        # if self.BC_type_v[1]=="D":
        #     v[-1,:]=self.BC_values_v[1]


        # elif self.BC_type_u[1]=="N":
        #     u[-1,:] = u[-2,:]

        # if self.BC_type_u[2]=="D":
        #     u[:,0]=self.BC_values_u[2]
        # elif self.BC_type_u[2]=="N":
        #     u[:,0] = u[:,1]

        # if self.BC_type_u[3]=="D":
        #     u[:,-1]=self.BC_values_u[3]
        # elif self.BC_type_u[3]=="N":
        #     u[:,-1] = u[:,-2]

        # if self.BC_type_u[0]=="D":
        #     u[0,:]=-u[1,:]+self.BC_values_u[0]
        # elif self.BC_type_u[0]=="N":
        #     u[0,:] = u[1,:]

        # # v
        # if self.BC_type_v[1]=="D":
        #     v[-1,:]=self.BC_values_v[1]
        # elif self.BC_type_v[1]=="N":
        #     v[-1,:] = v[-2,:]

        # if self.BC_type_v[2]=="D":
        #     v[:,0]=self.BC_values_v[2]
        # elif self.BC_type_v[2]=="N":
        #     v[:,0] = v[:,1]

        # if self.BC_type_v[3]=="D":
        #     v[:,-1]=self.BC_values_v[3]
        # elif self.BC_type_v[3]=="N":
        #     v[:,-1] = v[:,-2]

        # if self.BC_type_v[0]=="D":
        #     v[0,:]=-v[1,:]+self.BC_values_v[0]
        # elif self.BC_type_v[0]=="N":
        #     v[0,:] = v[1,:]


if __name__ == "__main__":

    use_gpu=False

    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    N      = 2**8
    nt      = 130
    nu      = 0
    dt      = 0.01
    x=torch.linspace(-60,180,N).to(device)
    y=torch.linspace(-60,180,N).to(device)


    solver = AdvDiffSolver(
        device,
        dt,
        x,
        y,
        nu,
        BC_type_u=["D","D","D","D"],
        BC_values_u=[1,0,0,0],
        BC_type_v=["D","D","D","D"],
        BC_values_v=[0,0,0,0],
        )

    X, Y = solver.X_cc, solver.Y_cc
    dx, dy = solver.dx, solver.dy

    print("dt={}s, dx={}, N={}".format(dt, dx, N))


    def initial_conditions():
        u = torch.zeros((N,N))
        v = torch.zeros((N,N))
        nm=int(N/2)-int(30 / dy)
        np=int(N/2)+int(30 / dy)
        u[nm:np,nm:np] = 1
        v[nm:np,nm:np] = 1
        # Z=(-X**2-Y**2)/0.5
        # u = torch.exp(Z/500-50)
        # v = torch.exp(Z/500-50)
        return (u,v)
    (u0,v0) = initial_conditions()

    u0 = u0.to(device)
    v0 = v0.to(device)

    from matplotlib import pyplot
    import time
    fig, (ax_1, ax_2, ax_3, ax_4, ax_5) = pyplot.subplots(1, 5, figsize=(25,5))

    # CS1=ax_1.contourf(X, Y, u0.cpu(), 20)
    # ax_1.set_title("Initial")
    # fig.colorbar(CS1)

    # u=u0.clone().detach()
    # v=v0.clone().detach()
    # start = time.time()
    # for n in range(nt + 1):
    #     (u,v) = solver.solve_explicit(u,v)
    #     solver.set_BCs(u,v)
    # print("Upwind solver took {}s".format(time.time()-start))


    # CS2=ax_2.contourf(X, Y, u.cpu(), 20)
    # ax_2.set_title("Upwind")
    # fig.colorbar(CS2)


    u=u0.clone().detach()
    v=v0.clone().detach()
    start = time.time()
    for n in range(nt + 1):
        (u,v) = solver.solve_implicit(u,v)
        solver.set_BCs(u,v)
    print("Implicit solver took {}s".format(time.time()-start))


    CS3=ax_3.contourf(X, Y, u.cpu(), 20)
    ax_3.set_title("implicit")
    fig.colorbar(CS3)

    # u=u0.clone().detach()
    # v=v0.clone().detach()
    # start = time.time()
    # for n in range(nt + 1):
    #     (u,v) = solver.solve_ADBQUICKEST(u,v)
    #     solver.set_BCs(u,v)
    # print("ADBquickest solver took {}s".format(time.time()-start))


    # CS4=ax_4.contourf(X, Y, u.cpu(), 20)
    # ax_4.set_title("ADBquickest")
    # fig.colorbar(CS4)


    # # re-set initial conditions and solve using the quick method
    # u=u0.clone().detach()
    # v=v0.clone().detach()
    # start = time.time()
    # for n in range(nt + 1):
    #     (u,v) = solver.solve_FLUXLMT(u,v)
    #     solver.set_BCs(u,v)
    # print("Flux-LMT solver took {}s".format(time.time()-start))


    # CS5=ax_5.contourf(X, Y, u.cpu(), 20)
    # ax_5.set_title("Flux LMT")
    # fig.colorbar(CS5)

    pyplot.show()