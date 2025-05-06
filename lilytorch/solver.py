
from lilytorch.adv_diff import AdvDiffSolver
from lilytorch.poisson import PoissonSolver
from lilytorch.poisson_fft import PoissonSolverFFT

# from lilytorch.poisson_2nd_order import PoissonSolver as PoissonSolver2nd
# from lilytorch.testing_nonapprox_poisson import PoissonSolver as PoissonSolver
from lilytorch.body import body_from_yaml
from lilytorch import plotting
from lilytorch.util.yaml_operations import yaml2pyobject, pyobject2yaml
import lilytorch.parser as parser
import torch
from tqdm import tqdm
import datetime
import os
import torch.nn.functional as F
import numpy as np

class FluidSolver:
    """
    Solver class
    """

    def __init__(self, pars, dtype=torch.float32, costum_update=None, comute_force=True):
        """
        BDIM2 solver for fluid structure interaction
        """

        solver    = pars["solver"]
        bcs       = pars["boundary_conditions"]
        output    = pars["output"]
        body_pars = pars["body"]

        use_gpu=solver["use_gpu"]
        if torch.cuda.is_available() and use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
            self.device = torch.device("cuda")
        else:
            print("Using the CPU.")
            self.device = torch.device("cpu")
            torch.set_num_threads(solver["nthreads"])

        self.dtype = dtype
        self.N    = solver["N"]
        self.xmin = solver["xmin"]
        self.xmax = solver["xmax"]
        self.ymin = solver["ymin"]
        self.ymax = solver["ymax"]

        self.dx         = (self.xmax-self.xmin)/self.N
        self.dy         = (self.ymax-self.ymin)/self.N

        x = torch.arange(self.xmin,self.xmax,self.dx,device=self.device,dtype=self.dtype)
        y = torch.arange(self.ymin,self.ymax,self.dy,device=self.device,dtype=self.dtype)

        x_ext = torch.arange(self.xmin-self.dx,self.xmax+self.dx,self.dx,device=self.device,dtype=self.dtype)
        y_ext = torch.arange(self.ymin-self.dy,self.ymax+self.dy,self.dy,device=self.device,dtype=self.dtype)

        [self.X,self.Y] = torch.meshgrid(x_ext,y_ext, indexing="ij")

        self.nx         = len(x)
        self.ny         = len(y)
        self.dx         = x[1]-x[0]
        self.dy         = y[1]-y[0]
        assert abs(self.dx-self.dy)<1e-7, "dx and dy must be equal"

        self.h2   = self.dx**2
        self.dt   = torch.tensor(solver["dt"], device=self.device, dtype=self.dtype)
        self.dt_np = self.dt.cpu().numpy()

        self.nt   = solver["nt"]
        self.nu = torch.tensor(solver["nu"], device=self.device, dtype=self.dtype) # kinematic viscosity
        self.rho  = torch.tensor(solver["rho"], device=self.device, dtype=self.dtype) # density
        self.eps  = 2*self.dx
        self.visc = self.nu*self.rho # dynamic viscosity
        self.re   = self.rho/self.nu

        self.starting_iteration      = solver.get("starting_iteration", 0)
        self.starting_iteration_path = solver.get("starting_iteration_path", None)
        self.starting_time           = self.starting_iteration * self.dt

        print("Setting dt={}s, dx={}".format(self.dt, x[1]-x[0]))

        # ============= convection solver =============
        self.adv_diff_solver = AdvDiffSolver(
            self.device,
            self.dt,x_ext,y_ext,self.nu,
            BC_type_u = bcs["BC_type_u"], BC_values_u = bcs["BC_values_u"],
            BC_type_v = bcs["BC_type_v"], BC_values_v = bcs["BC_values_v"],
            method    = solver["convection_method"]
        )

        # # =============  poisson solver =============
        # self.poisson_solver  = PoissonSolver(
        #     self.dtype,
        #     self.device,
        #     self.dx,
        #     tol=solver["poisson_tol"],
        #     max_cycles=solver["poisson_max_cycles"],
        #     nsmoothing=solver["poisson_nsmoothing"],
        #     w=solver["jacobi_weight"],
        #     verbose=solver["poisson_verbose"]
        # )

        # =============  poisson solver =============
        self.poisson_solver  = PoissonSolverFFT(
            x_ext,
            y_ext,
            filename=solver["poisson_filename"],
        )


        self.composite_body = body_from_yaml(
            self.device,
            x_ext, y_ext,
            body_pars,
            eps=self.eps,
            costum_update=costum_update,
            starting_time=self.starting_time
        )

        if not hasattr(self.composite_body, 'bodies'): # check if its a single body
            self.composite_body.bodies = [self.composite_body.body]
            self.composite_body.sdf_vals = [self.composite_body.body.sdf]
            self.composite_body.sdf_val = self.composite_body.body.sdf
            self.composite_body.com_pos = torch.tensor([self.xmin+(self.xmax-self.xmin)/2, self.ymin+(self.ymax-self.ymin)/2], device=self.device, dtype=self.dtype).view(1,2)


        # self.sdf_properties = self.composite_body.initialize()
        self.n_bodies=len(self.composite_body.bodies)
        self.friction_force_lin_x = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_lin_y = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_ang_z = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)

        self.pressure_force_x = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_y = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)

        self.body_u = torch.zeros(self.nx+2, self.ny+2,device=self.device,dtype=self.dtype)
        self.body_v = torch.zeros(self.nx+2, self.ny+2,device=self.device,dtype=self.dtype)

        self.zc=torch.ones((1,self.N),device=self.device,dtype=self.dtype)
        self.zr=torch.ones((self.N,1),device=self.device,dtype=self.dtype)

        # # Parameters for the Gaussian kernel
        # kernel_dx=0.005
        # self.kernel_size = round(kernel_dx/self.dx)  # Example kernel size
        # self.kernel_size += 1-self.kernel_size%2
        # sigma = 10  # Standard deviation for Gaussian
        # self.kernel = self.composite_body.gaussian_kernel(self.kernel_size, sigma).view(1, 1, self.kernel_size, self.kernel_size).to(self.device)

        # ===== set initial conditions =====
        self.set_initial_conditions()

        # ===== plotting parameters =====
        self.extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        )

        self.compute_forces = comute_force

        # ===== create folder for frames' storage ====
        self.save_frames = output["save_frames"]
        self.save_every  = output["save_every"]
        self.save_uv     = output["save_uv"]
        self.vmin        = output["vmin"]
        self.vmax        = output["vmax"]
        self.n_quiver_spacing = 2**3

        if self.save_frames or self.save_uv:
            path           = output["save_path"]
            today          = datetime.datetime.now()
            todaystr       = today.isoformat()
            results_folder = f'{path}{todaystr}'
            os.mkdir(results_folder)
            self.save_path = results_folder+"/"

            # Add save path to the parameters
            pars["output"]["results_folder"] = results_folder

            # Save body signal (if available)
            if  getattr(self.composite_body, 'save_signal', None):
                self.composite_body.save_signal(results_folder)

            # Save the parameters as a yaml file
            pyobject2yaml(
                filename = self.save_path+"parameters.yaml",
                pyobject = pars,
            )

    def outside(self, x):
        """
        Return True if all elements in x are outside the domain
        """
        return torch.all(
            torch.logical_and(
                x[:,0]>self.xmin,
                torch.logical_and(
                    x[:,0]<self.xmax,
                    torch.logical_and(
                        x[:,1]>self.ymin,
                        x[:,1]<self.ymax
                    )
                )
            )
        )

    def _load_initial_conditions(self):
        ''' Load initial conditions from a previous simulation '''

        if not self.starting_iteration_path:
            return False

        v0_path = f'{self.starting_iteration_path}/uv_field/v_{self.starting_iteration}.npy'
        u0_path = f'{self.starting_iteration_path}/uv_field/u_{self.starting_iteration}.npy'

        # Verify files
        if not os.path.exists(v0_path) or not os.path.exists(u0_path):
            raise FileNotFoundError(f'Initial conditions not found at {v0_path} or {u0_path}')

        u0 = torch.tensor(np.load(u0_path)).to(self.device)
        v0 = torch.tensor(np.load(v0_path)).to(self.device)
        p0 = torch.zeros((self.nx,self.ny),device=self.device)

        # Verify shape
        shape = (self.nx, self.ny)
        assert u0.shape == shape, f"u0 shape: {u0.shape} != {shape}"
        assert v0.shape == shape, f"v0 shape: {v0.shape} != {shape}"

        # Loaded
        self.u0, self.v0, self.p0 = u0, v0, p0

        return True

    def set_initial_conditions(self):
        """
        initial conditions
        """

        # Load initial conditions
        if self._load_initial_conditions():
            return

        # Set initial conditions
        if self.adv_diff_solver.BC_type_u[0]=="D":
            self.u0 = self.adv_diff_solver.BC_values_u[0]*torch.ones((self.nx+2,self.ny+2),device=self.device,dtype=self.dtype)
        else:
            self.u0 = torch.zeros((self.nx+2,self.ny+2),device=self.device,dtype=self.dtype)
        self.v0 = torch.zeros((self.nx+2,self.ny+2),device=self.device,dtype=self.dtype)
        self.p0 = torch.zeros((self.nx+2,self.ny+2),device=self.device,dtype=self.dtype)

        return

    def compute_dpdx(self,p):
        """
        Compute dp/dx
        """
        return torch.gradient(p, spacing=self.dx, dim=0)[0]

    def compute_dpdy(self,p):
        """
        Compute dp/dy
        """
        return torch.gradient(p, spacing=self.dy, dim=1)[0]

    def gradient(self, var):
        """
        Compute gradient(var)
        """
        return (self.compute_dpdx(var), self.compute_dpdy(var))

    def divergence(self, u, v):
        """
        Compute the divergence
        """
        return self.compute_dpdx(u)+self.compute_dpdy(v)

    def normal_derivative(self, var, normal_x, normal_y):
        """
        Compute the normal derivative dvar/dn
        """
        return normal_x*self.compute_dpdx(var)+normal_y*self.compute_dpdy(var)

    def vorticity(self, u, v):
        """
        Compute the vorticity of u,v in 2d
        """
        return self.compute_dpdx(v)-self.compute_dpdy(u)

    def project(self, u, v, p):

        # ====== solve the pressure poisson equation ======
        self.div_body=torch.zeros_like(u)
        for i, body in enumerate(self.composite_body.bodies[:]):
            sdf_val = self.composite_body.sdf_vals[i]
            mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
            m_m0 = (1-mu0)
            body_u = self.composite_body.u_vals[i]
            body_v = self.composite_body.v_vals[i]

            # (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)

            # uprime += m_m0*body_u+mu1*self.normal_derivative(u-body_u,normal_x,normal_y)
            # vprime += m_m0*body_v+mu1*self.normal_derivative(v-body_v,normal_x,normal_y)

            self.div_body+=m_m0*self.divergence(body_u,body_v)

        mult_factor = self.dt/self.rho
        rhs=(self.divergence(u,v)-self.div_body)/mult_factor
        p=self.poisson_solver.solve(rhs)

        # ====== projection step ======
        (p_x, p_y)=self.gradient(p)
        return (u-mult_factor*p_x, v-mult_factor*p_y)

        # # ====== solve the pressure poisson equation ======
        # rhs = self.div_body[1:-1,1:-1]
        # mult_factor = self.dt/self.rho        # coeff = mult_factor*self.mu0_all[1:-1,1:-1]
        # coeff_horizontal = (coeff[1:,:]+coeff[:-1,:])/2
        # coeff_vertical = (coeff[:,1:]+coeff[:,:-1])/2
        # coeff_horizontal=torch.vstack((mult_factor*self.zc,coeff_horizontal,mult_factor*self.zc))
        # coeff_vertical=torch.hstack((mult_factor*self.zr,coeff_vertical,mult_factor*self.zr))
        # p, _ = self.poisson_solver.solve_multigrid( # f, u, c
        #     rhs,
        #     p,
        #     coeff,
        #     coeff_horizontal,
        #     coeff_vertical,
        # )
        # p = self.project(rhs, p)

        # # ====== projection step ======
        # (p_x, p_y) = self.gradient(p)
        # u = uprime-self.dt/self.rho*self.mu0_all*p_x
        # v = vprime-self.dt/self.rho*self.mu0_all*p_y

        # mult_factor = self.dt/self.rho
        # coeff = mult_factor*self.mu0_all[1:-1,1:-1]
        # coeff_horizontal = (coeff[1:,:]+coeff[:-1,:])/2
        # coeff_vertical = (coeff[:,1:]+coeff[:,:-1])/2
        # coeff_horizontal=torch.vstack((mult_factor*self.zc,coeff_horizontal,mult_factor*self.zc))
        # coeff_vertical=torch.hstack((mult_factor*self.zr,coeff_vertical,mult_factor*self.zr))
        # # p = torch.zeros_like(p)
        # p, _ = self.poisson_solver.solve_multigrid( # f, u, c
        #     rhs,
        #     p,
        #     coeff,
        #     coeff_horizontal,
        #     coeff_vertical,
        # )
        # return p

    def compute_water_forces(self, u, v, p):
        (u_ext, v_ext, p_ext) = (u,v,p)
        # (u_ext, v_ext, p_ext) = self.solver_free(u,v,p)

        # ======= compute stress tensor ======
        dudx, dudy = torch.gradient(u_ext)
        dvdx, dvdy = torch.gradient(v_ext)

        ss_diag = dudy/self.dy+dvdx/self.dx
        ss_11 = 2*dudx/self.dx
        ss_22 = 2*dvdy/self.dy

        for i, body in enumerate(self.composite_body.bodies):

            d=self.composite_body.sdf_vals[i]-self.eps
            # dall=self.composite_body.sdf_val-self.eps


            # self.delta = self.composite_body.bodies[0].phi(d)*self.composite_body.bodies[0].phi(dall)*self.eps*self.eps
            self.delta = self.composite_body.bodies[0].phi(d)*self.eps

            (d, normal_x, normal_y, R) = self.composite_body.compute_sdf_properties(d)

            xstress_tensor = (normal_x*ss_11+normal_y*ss_diag)*self.delta
            ystress_tensor = (normal_x*ss_diag+normal_y*ss_22)*self.delta

            self.friction_force_lin_x[i] = -self.visc*torch.trapz(torch.trapz(xstress_tensor, dx=self.dy), dx=self.dx)
            self.friction_force_lin_y[i] = -self.visc*torch.trapz(torch.trapz(ystress_tensor, dx=self.dy), dx=self.dx)

            # self.friction_force_ang_z[i] = -self.visc*torch.trapz(torch.trapz(
            #     xstress_tensor*(self.X-self.composite_body.com_pos[i,0])+
            #     ystress_tensor*(self.Y-self.composite_body.com_pos[i,1]),
            #     dx=self.dy),
            #     dx=self.dx
            # )

            # self.pressure_force_x[i] = self.visc*torch.trapz(torch.trapz(p_ext*normal_x*self.delta, dx=self.dy), dx=self.dx)
            # self.pressure_force_y[i] = self.visc*torch.trapz(torch.trapz(p_ext*normal_y*self.delta, dx=self.dy), dx=self.dx)

            # delta_plus = self.composite_body.bodies[0].phi(d)/(1+d/R)
            # p_force_par=p_ext-d*self.normal_derivative(p_ext,normal_x,normal_y)
            # self.pressure_force_x[i] = torch.trapz(torch.trapz(p_force_par*normal_x*delta, dx=self.dy), dx=self.dx)
            # self.pressure_force_y[i] = torch.trapz(torch.trapz(p_force_par*normal_y*delta, dx=self.dy), dx=self.dx)

    def solver_iteration_old(self,u,v,p):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)

        self.u_=self.composite_body.bodies[0].body_u
        self.v_=self.composite_body.bodies[0].body_v

        p = torch.zeros_like(u)

        m_m0 = (1-self.mu0)

        # uprime = self.mu0*u + self.body_u + self.mu1_all*self.normal_derivative(u-self.u_,self.normal_x,self.normal_y)
        # vprime = self.mu0*v + self.body_v + self.mu1_all*self.normal_derivative(v-self.v_,self.normal_x,self.normal_y)

        # uprime = self.mu0_all*u + self.body_u + self.mu1_all*self.normal_derivative(u,self.normal_x,self.normal_y) - self.body_un
        # vprime = self.mu0_all*v + self.body_v + self.mu1_all*self.normal_derivative(v,self.normal_x,self.normal_y) - self.body_vn

        uprime = self.mu0*u + m_m0*self.u_+ self.mu1*self.normal_derivative(u-self.u_,self.normal_x,self.normal_y)
        vprime = self.mu0*v + m_m0*self.v_+ self.mu1*self.normal_derivative(v-self.v_,self.normal_x,self.normal_y)

        # ====== solve the pressure poisson equation and project ======
        coeff = self.dt*self.mu0/self.rho
        rhs = -(self.divergence(uprime,vprime)-m_m0*self.divergence(self.u_,self.v_))
        # p = torch.zeros_like(u)
        p = self.poisson_solver.solve_multigrid( # f, u, c
            rhs,
            p,
            coeff
        )
        p=p-p.mean()
        (p_x, p_y) = self.gradient(p)
        u = uprime-coeff*p_x
        v = vprime-coeff*p_y

        self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solver_free(self,u,v,p):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)

        # ====== solve the pressure poisson equation and project ======
        coeff = self.dt/self.rho
        domain = torch.ones_like(u)
        # p = torch.zeros_like(u)
        p_ext, _ = self.poisson_solver.solve_multigrid( # f, u, c
            self.divergence(u,v)/coeff,
            p,
            domain,
            domain,
            domain
        )
        (p_x, p_y) = self.gradient(p_ext)
        u -= coeff*p_x
        v -= coeff*p_y

        self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solver_iteration_test(self,u,v,p,iteration):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)
        self.adv_diff_solver.set_BCs(u,v)

        uprime = self.mu0_all*u
        vprime = self.mu0_all*v

        self.div_body=torch.zeros_like(u)
        for i, body in enumerate(self.composite_body.bodies[:]):
            sdf_val = self.composite_body.sdf_vals[i]
            mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
            m_m0 = (1-mu0)
            body_u = self.composite_body.u_vals[i]
            body_v = self.composite_body.v_vals[i]

            (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)

            uprime += m_m0*body_u+mu1*self.normal_derivative(u-body_u,normal_x,normal_y)
            vprime += m_m0*body_v+mu1*self.normal_derivative(v-body_v,normal_x,normal_y)

            self.div_body+=m_m0*self.divergence(body_u,body_v)

        # self.mu0_all=mu0

        # ====== solve the pressure poisson equation and project ======
        coeff = self.dt/self.rho
        self.div_body=self.divergence(uprime,vprime)-self.div_body
        rhs = self.div_body[1:-1,1:-1]
        p=self.project(rhs, p)
        (p_x, p_y) = self.gradient(p)

        u = uprime-coeff*self.mu0_all*p_x
        v = vprime-coeff*self.mu0_all*p_y

        # u = uprime
        # v = vprime

        self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solver_iteration_test2(self,u,v,p,iteration):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)
        self.adv_diff_solver.set_BCs(u,v)

        uprime = self.mu0_all*u
        vprime = self.mu0_all*v

        self.div_body=torch.zeros_like(u)

        for i, body in enumerate(self.composite_body.bodies):
            if i==self.n_bodies-1:
                sdf_val = self.composite_body.sdf_vals[i]
                mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
                m_m0 = (1-mu0)

                body_u = self.composite_body.u_vals[i]
                body_v = self.composite_body.v_vals[i]

                (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)
                uprime += m_m0*body_u+mu1*self.normal_derivative(u-body_u,normal_x,normal_y)
                vprime += m_m0*body_v+mu1*self.normal_derivative(v-body_v,normal_x,normal_y)

                self.div_body+=m_m0*self.divergence(body_u,body_v)

            else:
                sdf_val = self.composite_body.sdf_vals[i]
                mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
                m_m0 = (1-mu0)

                sdf_val1 = self.composite_body.sdf_vals[i+1]
                mu0p, mu1p = self.composite_body.mu_funcs(sdf_val1)
                m_m0p = (1-mu0p)
                intersect = m_m0*m_m0p

                body_u = self.composite_body.u_vals[i]
                body_v = self.composite_body.v_vals[i]

                (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)
                uprime += (m_m0-intersect)*body_u+mu1*self.normal_derivative(u-body_u,normal_x,normal_y)
                vprime += (m_m0-intersect)*body_v+mu1*self.normal_derivative(v-body_v,normal_x,normal_y)

                self.div_body+=(m_m0-intersect)*self.divergence(body_u,body_v)


        # ====== solve the pressure poisson equation and project ======
        coeff = self.dt/self.rho
        self.div_body=self.divergence(uprime,vprime)-self.div_body
        rhs = self.div_body[1:-1,1:-1]
        p=self.project(rhs, p)
        (p_x, p_y) = self.gradient(p)

        u = uprime-coeff*self.mu0_all*p_x
        v = vprime-coeff*self.mu0_all*p_y
        self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solver_iteration(self,u,v,p,iteration):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)
        uprime = self.mu0_all*u + self.m_m0_all*self.body_u + self.mu1_all*self.normal_derivative(u-self.body_u,self.normal_x,self.normal_y)
        vprime = self.mu0_all*v + self.m_m0_all*self.body_v + self.mu1_all*self.normal_derivative(v-self.body_v,self.normal_x,self.normal_y)
        self.adv_diff_solver.set_BCs(uprime,vprime)
        (u,v) = self.project(uprime,vprime,p)

        # self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solve_heun(self, u, v, p, iteration):
        # (u1,v1,p) = self.solver_iteration(u,v,p,iteration)
        # (u2,v2,p) = self.solver_iteration(u1,v1,p,iteration)
        # u=0.5*(u1+u2)
        # v=0.5*(v1+v2)
        # return (u,v,p)
        (u1,v1,p) = self.solver_iteration(u,v,p,iteration)
        return (u1,v1,p)

    def step_(self, u, v, p, iteration, t):

        (self.mu0_all, self.mu1_all) = self.composite_body.mu_funcs(self.composite_body.sdf_val)
        self.m_m0_all = (1-self.mu0_all)
        (_, self.normal_x, self.normal_y, self.curvature) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val)

        # # smoothen velocities
        # self.body_u = F.conv2d(
        #     self.body_u.view(1, 1, self.N, self.N),
        #     self.kernel,
        #     padding=self.kernel_size//2
        # )[0,0]
        # self.body_v = F.conv2d(
        #     self.body_v.view(1, 1, self.N, self.N),
        #     self.kernel,
        #     padding=self.kernel_size//2
        # )[0,0]

        ##### just for plotting
        self.sdf_properties=[[self.composite_body.sdf_val]]

        if self.compute_forces:
            self.compute_water_forces(u,v,p)
        else:
            self.delta = torch.zeros_like(u)

        (u,v,p) = self.solve_heun(u,v,p,iteration)

        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        # ============ plotting/saving ==========
        if not iteration % self.save_every:
            if self.save_frames:


                # copy u from device to host
                X,Y = self.X.cpu(), self.Y.cpu()
                curl = self.vorticity(u,v).cpu()
                divergence = self.div_body.cpu()
                # d_min = self.composite_body.sdf_val.cpu()
                # d_min = self.d_min.cpu()
                sdf=self.composite_body.sdf_val.cpu()
                pressure = p.cpu()
                # curl_body = (self.vorticity(self.body_u,self.body_v)).cpu()
                vec_x=(self.m_m0_all*self.body_u).cpu()
                vec_y=(self.m_m0_all*self.body_v).cpu()

                tmp=self.delta
                # tmp = (self.div_body).cpu()
                # tmp = (self.m_m0_all*self.divergence(self.body_u,self.body_v)).cpu() #(self.tmp).cpu()



                # plotting.plot2d_imshow(X,Y,(self.vorticity(u,v)/(self.composite_body.bodies[0].L)).cpu(),d_min,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax)

                plotting.plot2d_imshow_composite_quiver(X,Y,curl,self.sdf_properties,0*X,0*X,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # plotting.plot2d_imshow_composite_quiver(X,Y,curl_body,self.sdf_properties,vec_x,vec_y,self.extent,iteration,self.save_path,"curlbody",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                plotting.plot2d_imshow_composite_quiver(X,Y,tmp.cpu(),self.sdf_properties,vec_x,vec_y,self.extent,iteration,self.save_path,"tmp",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # plotting.plot2d_imshow_simple((self.mu1_all/self.eps).cpu(),self.extent,iteration,self.save_path,"mu1",0,0.2)
                # plotting.plot2d_imshow_simple((self.mu0_all).cpu(),self.extent,iteration,self.save_path,"mu0",0,1)

                plotting.plot2d_imshow_composite(X,Y,u.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"u",None, None)
                plotting.plot2d_imshow_composite(X,Y,v.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"v",None, None)

                # plotting.plot2d_imshow_composite(X,Y,vec_x,self.sdf_properties,self.extent,iteration,self.save_path,"bodyu",None, None)
                # plotting.plot2d_imshow_composite(X,Y,vec_y,self.sdf_properties,self.extent,iteration,self.save_path,"bodyv",None, None)

                plotting.plot2d_imshow_only(divergence,self.extent,iteration,self.save_path,"divergence",None, None)

                # plotting.plot2d_imshow(X,Y,pressure,d_min,self.extent,iteration,self.save_path,"pressure",None, None)
                plotting.plot2d_imshow_only(pressure,self.extent,iteration,self.save_path,"pressure",None, None)


                # plotting.plot2d_imshow_only((self.m_m0_all).cpu(),self.extent,iteration,self.save_path,"tmp",None, None)

                plotting.plot2d_imshow_quiver(X,Y,sdf,sdf,vec_x,vec_y,self.extent,iteration,self.save_path,"sdf",None,None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                # plotting.plot2d_imshow_quiver(X,Y,curl,d_min,vec_x,vec_y,self.extent,iteration,self.save_path,"curluv",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                # plotting.plot2d_imshow_only(xstress_tensor.cpu(),self.extent,iteration,self.save_path,"xstress_tensor", None, None)


            if self.save_uv:
                uv_path = f'{self.save_path}/uv_field'
                os.makedirs(uv_path, exist_ok=True)
                np.save(f'{uv_path}/u_{iteration}',u.cpu().numpy())
                np.save(f'{uv_path}/v_{iteration}',v.cpu().numpy())

        continue_sim=self.outside(self.composite_body.com_pos)

        return (u,v,p,continue_sim)

    def run_from_initial(self, u0, v0):
        u=u0
        v=v0
        p=torch.zeros_like(u)
        for iteration in tqdm(range(self.nt)):
            t=iteration*self.dt
            (u,v,p,stop_sim) = self.step_(u, v, p, iteration, t)

    def run_sim(self):
        u=self.u0
        v=self.v0
        p=self.p0
        for iteration in tqdm(range(self.starting_iteration, self.nt)):
            t=iteration*self.dt
            (u,v,p,stop_sim) = self.step_(u, v, p, iteration, t)



if __name__ == "__main__":

    solver = FluidSolver()
    solver.run_sim()


