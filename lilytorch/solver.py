
from lilytorch.adv_diff import AdvDiffSolver
from lilytorch.poisson_test import PoissonSolver
from lilytorch.poisson_fft import PoissonSolverFFT
from pytorch_interp import RegularGridInterpolator
from lilytorch.body import body_from_yaml
from lilytorch import plotting
from lilytorch.util.yaml_operations import pyobject2yaml
import torch
from tqdm import tqdm
import datetime
import os
import numpy as np

class FluidSolver:
    """
    Solver class
    """

    def __init__(self, pars, dtype=torch.float32, costum_update=None, compute_forces=True):
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

        x=torch.linspace(self.xmin-self.dx/2,self.xmax+self.dx/2,self.N+2,device=self.device,dtype=self.dtype)
        y=torch.linspace(self.ymin-self.dy/2,self.ymax+self.dy/2,self.N+2,device=self.device,dtype=self.dtype)

        [self.X,self.Y] = torch.meshgrid(x,y, indexing="ij")

        self.nx         = len(x)
        self.ny         = len(y)
        self.dx         = x[1]-x[0]
        self.dy         = y[1]-y[0]

        self.dxdy = [float(self.dx), float(self.dy)]
        assert abs(self.dx-self.dy)<1e-7, "dx and dy must be equal"

        self.h2   = self.dx**2
        self.dt   = torch.tensor(solver["dt"], device=self.device, dtype=self.dtype)
        self.dt_np = self.dt.cpu().numpy()

        self.nt      = solver["nt"]
        self.nu      = torch.tensor(solver["nu"], device=self.device, dtype=self.dtype) # kinematic viscosity
        self.rho     = torch.tensor(solver["rho"], device=self.device, dtype=self.dtype) # density
        self.eps     = 2*self.dx
        self.visc    = self.nu*self.rho # dynamic viscosity
        self.re      = self.rho/self.nu
        self.p_coeff = self.dt/self.rho

        self.p_fc=torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.p_cc=torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)

        self.brinkmann_k = 1000.0

        self.starting_iteration      = solver.get("starting_iteration", 0)
        self.starting_iteration_path = solver.get("starting_iteration_path", None)
        self.starting_time           = self.starting_iteration * self.dt

        print("Setting dt={}s, dx={}".format(self.dt, x[1]-x[0]))

        # ============= convection solver =============
        self.adv_diff_solver = AdvDiffSolver(
            self.device,
            self.dt,x,y,self.nu,self.rho,
            BC_type_u = bcs["BC_type_u"], BC_values_u = bcs["BC_values_u"],
            BC_type_v = bcs["BC_type_v"], BC_values_v = bcs["BC_values_v"],
            method    = solver["convection_method"]
        )

        # =============  poisson solver =============
        self.poisson_solver  = PoissonSolver(
            self.dtype,
            self.device,
            self.dx,
            tol=solver["poisson_tol"],
            max_vcycles=solver["poisson_max_cycles"],
            nsmoothing=solver["poisson_nsmoothing"],
            w=solver["jacobi_weight"],
            verbose=solver["poisson_verbose"]
        )

        # =============  poisson FFT solver =============
        self.poisson_solverFFT  = PoissonSolverFFT(
            x,
            y,
            filename=solver["poisson_filename"],
            bc_type="free"

        )

        self.composite_body = body_from_yaml(
            self.device,
            x, y,
            body_pars,
            eps=self.eps,
            costum_update=costum_update,
            starting_time=self.starting_time,
        )


        # self.sdf_properties = self.composite_body.initialize()
        self.force_x_interp = RegularGridInterpolator(
            (x,y),
            torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
        )
        self.force_y_interp = RegularGridInterpolator(
            (x,y),
            torch.zeros_like(self.Y, device=self.device, dtype=self.dtype),
        )

        self.n_bodies=len(self.composite_body.bodies)
        self.friction_force_lin_x = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_lin_y = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_ang_z = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.force_x_int = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.force_y_int = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)

        self.pressure_force_x = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_y = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)

        self.body_u = torch.zeros(self.nx, self.ny,device=self.device,dtype=self.dtype)
        self.body_v = torch.zeros(self.nx, self.ny,device=self.device,dtype=self.dtype)

        self.zc=torch.ones((1,self.ny),device=self.device,dtype=self.dtype)
        self.zr=torch.ones((self.nx,1),device=self.device,dtype=self.dtype)

        self.xstress_tensor = torch.zeros_like(self.X, device=self.device, dtype=self.dtype)
        self.ystress_tensor = torch.zeros_like(self.Y, device=self.device, dtype=self.dtype)

        self.viscous_drag_record = torch.zeros((self.n_bodies,2,self.nt),device=self.device,dtype=self.dtype)
        self.pressure_drag_record = torch.zeros((self.n_bodies,2,self.nt),device=self.device,dtype=self.dtype)

        self.force_ctr_x = [torch.zeros((body.cnt.shape[1])) for body in self.composite_body.bodies]
        self.force_ctr_y = [torch.zeros((body.cnt.shape[1])) for body in self.composite_body.bodies]

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

        self.compute_forces = compute_forces

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

            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
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
        return torch.any(
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
        # if self.adv_diff_solver.BC_type_u[0]=="D":
        #     self.u0 = self.adv_diff_solver.BC_values_u[0]*torch.ones((self.nx,self.ny),device=self.device,dtype=self.dtype)
        # else:
        self.u0 = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.v0 = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.p0 = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.adv_diff_solver.set_BCs(self.u0,self.v0)

        return

    def compute_dpdx(self,p):
        """
        Compute dp/dx
        """
        return torch.gradient(p, spacing=self.dx, dim=0, edge_order=2)[0]

    def compute_dpdy(self,p):
        """
        Compute dp/dy
        """
        return torch.gradient(p, spacing=self.dy, dim=1, edge_order=2)[0]

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



    def compute_fluid_forces(self, u, v, p, iteration):

        bodies=self.composite_body.bodies

        # ======= compute stress tensor ======
        dudx, dudy = self.gradient(u)
        dvdx, dvdy = self.gradient(v)
        ss_diag = (dudy+dvdx)
        ss_11 = 2*dudx
        ss_22 = 2*dvdy
        self.xstress_tensor = self.visc*(self.normal_x*ss_11+self.normal_y*ss_diag)
        self.ystress_tensor = self.visc*(self.normal_x*ss_diag+self.normal_y*ss_22)


        # self.force_y_interp.F = -p+self.visc*(normal_x*ss_diag+normal_y*ss_22)

        # import matplotlib.pyplot as plt
        # import matplotlib.colors as colors
        # import matplotlib.cm as cm
        # # plt.figure(1)
        # # ax = plt.axes(projection='3d')
        # vmin, vmax = self.force_x_interp.F.min().cpu(), self.force_x_interp.F.max().cpu()
        # norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # cmap = cm.get_cmap('viridis')

        # phi_all=self.composite_body.bodies[0].phi(self.composite_body.sdf_val)*self.eps

        for i, body in enumerate(bodies):

            # if iteration*self.dt<1:
            #     self.friction_force_lin_x[i] = 0.02
            #     self.friction_force_lin_y[i] = 0.0
            # else:

                # ================ compute viscous forces ================

                d=self.composite_body.sdf_vals[i]
                self.delta = self.composite_body.bodies[0].phi(d)*self.eps
                # self.delta = self.composite_body.bodies[0].phi(self.composite_body.sdf_vals[i])*phi_all*self.eps

                # method 1
                self.force_x_interp.F = 0*self.xstress_tensor
                self.force_y_interp.F = 0*self.ystress_tensor

                f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

                self.friction_force_lin_x[i] = torch.trapezoid(f_x[:-1], body.curv_coord)
                self.friction_force_lin_y[i] = torch.trapezoid(f_y[:-1], body.curv_coord)

                # print("Total force x: ", self.friction_force_lin_x)


                # # method 2
                # dsign=torch.sign(self.composite_body.sdf_vals[i])

                # self.friction_force_lin_x[i] = torch.sum(torch.trapz(dsign*self.xstress_tensor*self.delta)) #*self.dx*self.dy
                # self.friction_force_lin_y[i] = torch.sum(torch.trapz(dsign*self.ystress_tensor*self.delta)) #*self.dx*self.dy


                # # method 3
                # self.force_x_interp.F = self.xstress_tensor
                # self.force_y_interp.F = self.ystress_tensor

                # press_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])

                # alpha=1000
                # beta=20
                # force_x_tmp = torch.trapz(torch.trapz(alpha*(self.composite_body.body_u[i]-u)*self.delta,dx=self.dx),dx=self.dy)
                # force_y_tmp = torch.trapz(torch.trapz(alpha*(self.composite_body.body_v[i]-v)*self.delta,dx=self.dx),dx=self.dy)

                # self.force_x_int[i] += force_x_tmp*self.dt
                # self.force_y_int[i] += force_y_tmp*self.dt

                # self.friction_force_lin_x[i] = - alpha*self.force_x_int[i] - beta*force_x_tmp
                # self.friction_force_lin_y[i] = - alpha*self.force_y_int[i] - beta*force_y_tmp


                # # method 4
                # force_x_tmp = torch.trapz(torch.trapz(u*self.m_m0_all))
                # force_y_tmp = torch.trapz(torch.trapz(v*self.m_m0_all))

                # self.force_x_int[i] += -force_x_tmp*self.dt
                # self.force_y_int[i] += -force_y_tmp*self.dt

                # self.friction_force_lin_x[i] = self.force_x_int[i]
                # self.friction_force_lin_y[i] = self.force_y_int[i]

                # # method 5

                # alpha=1
                # beta=1
                # self.force_x_interp.F = u-self.composite_body.body_u[i]
                # self.force_y_interp.F = v-self.composite_body.body_v[i]


                # f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                # f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

                # self.force_ctr_x[i] += self.dt*f_x
                # self.force_ctr_y[i] += self.dt*f_y

                # combined_f_x = alpha*self.force_ctr_x[i]+beta*f_x
                # combined_f_y = alpha*self.force_ctr_y[i]+beta*f_y


                # self.friction_force_lin_x[i] = torch.trapz(combined_f_x[:-1], body.curv_coord)
                # self.friction_force_lin_y[i] = torch.trapz(combined_f_y[:-1], body.curv_coord)


                self.viscous_drag_record[i,0,iteration] = self.friction_force_lin_x[i]
                self.viscous_drag_record[i,1,iteration] = self.friction_force_lin_y[i]


                # ================ compute pressure forces ================
                self.force_x_interp.F = -p*0.00*self.normal_x
                self.force_y_interp.F = -p*0.00*self.normal_y

                # ================ compute pressure forces ================
                # pforce=p-d*self.normal_derivative(p, self.normal_x, self.normal_y)
                # self.force_x_interp.F = 0.01*pforce*self.normal_x
                # self.force_y_interp.F = 0.01*pforce*self.normal_y


                f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])


                self.pressure_force_x[i] = torch.trapezoid(f_x[:-1], body.curv_coord)
                self.pressure_force_y[i] = torch.trapezoid(f_y[:-1], body.curv_coord)



                self.pressure_drag_record[i,0,iteration] = self.pressure_force_x[i]
                self.pressure_drag_record[i,1,iteration] = self.pressure_force_y[i]


        # print("Total pressure force x: ", self.pressure_force_x)
        # print("Total pressure force y: ", self.pressure_force_y)
        # print("Total viscous force: ", self.viscous_drag_record)


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

        self.mu0_all,_ = self.composite_body.mu_funcs(self.composite_body.sdf_vals[0])

        uprime = 0 #self.mu0_all*u
        vprime = 0 #self.mu0_all*v

        self.div_body=torch.zeros_like(u)
        for i, body in enumerate(self.composite_body.bodies):
            sdf_val = self.composite_body.sdf_vals[i]
            mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
            m_m0 = (1-mu0)
            body_u = self.composite_body.u_vals[i]
            body_v = self.composite_body.v_vals[i]

            (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)

            uprime += mu0*u+m_m0*body_u+mu1*self.normal_derivative(u-body_u,normal_x,normal_y)
            vprime += mu0*v+m_m0*body_v+mu1*self.normal_derivative(v-body_v,normal_x,normal_y)

            self.div_body+=m_m0*self.divergence(body_u,body_v)

        self.adv_diff_solver.set_BCs(uprime,vprime)

        # mult_factor = self.dt/self.rho
        # self.div=(self.divergence(uprime,vprime)-self.div_body)/mult_factor
        # p=self.poisson_solverFFT.solve(self.div)

        # ====== solve the pressure poisson equation ======
        self.div=(self.divergence(uprime,vprime))[1:-1,1:-1]
        mult_factor = self.dt/self.rho
        coeff = mult_factor*torch.ones((self.nx,self.ny),device=self.device) #self.mu0_all[1:-1,1:-1]
        coeff_horizontal = (coeff[1:,:]+coeff[:-1,:])/2
        coeff_vertical = (coeff[:,1:]+coeff[:,:-1])/2
        coeff_horizontal=torch.vstack((mult_factor*self.zc,coeff_horizontal,mult_factor*self.zc))
        coeff_vertical=torch.hstack((mult_factor*self.zr,coeff_vertical,mult_factor*self.zr))
        p, _ = self.poisson_solver.solve_multigrid( # f, u, c
            self.div,
            p,
            coeff,
            coeff_horizontal,
            coeff_vertical,
        )

        # ====== projection step ======
        (p_x, p_y) = self.gradient(p)
        (u,v)=(uprime-self.dt/self.rho*p_x, vprime-self.dt/self.rho*p_y)

        return (u,v,p)

    def integral(self, var):
        """
        Compute the integral of var
        """
        return torch.trapz(torch.trapz(var, dx=self.dx), dx=self.dy)

    def project(self, uprime, vprime, p):

        # coeff = torch.ones((self.nx,self.ny),device=self.device,dtype=self.dtype)
        coeff = self.mu0_all[1:-1,1:-1]
        coeff_horizontal = (coeff[1:,:]+coeff[:-1,:])/2
        coeff_vertical = (coeff[:,1:]+coeff[:,:-1])/2
        coeff_horizontal=torch.vstack((self.zc,coeff_horizontal,self.zc))
        coeff_vertical=torch.hstack((self.zr,coeff_vertical,self.zr))

        p, _ = self.poisson_solver.solve_multigrid( # f, u, c
            self.div,
            p,
            coeff,
            coeff_horizontal,
            coeff_vertical,
        )
        # ====== projection step ======
        (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        (u,v)=(uprime-(self.dt/self.rho)*self.mu0_all*p_x, vprime-(self.dt/self.rho)*self.mu0_all*p_y)

        # self.div=self.divergence(uprime,vprime)
        # p=self.poisson_solverFFT.solve(self.div)
        # (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        # (u,v)=(uprime-self.mu0_all*p_x, vprime-self.mu0_all*p_y)

        return (u,v,p)

    def solver_iteration_test2(self,u,v,p,iteration):
        """
        BDIM2 iteration
        """

        # # ====== convection solver ======
        # (u,v) = self.adv_diff_solver.solve(u,v)

        # uprime = self.mu0_all*u
        # vprime = self.mu0_all*v


        # for i, body in enumerate(self.composite_body.bodies):

        #     if i==self.n_bodies-1:
        #         sdf_val = self.composite_body.sdf_vals[i]
        #         mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
        #         xi0 = 1-mu0
        #         xi1 = mu1
        #         (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)

        #     else:
        #         sdf_val = self.composite_body.sdf_vals[i]
        #         mu0, mu1 = self.composite_body.mu_funcs(sdf_val)
        #         m_m0 = (1-mu0)

        #         sdf_val1 = self.composite_body.sdf_vals[i+1]
        #         mu0p, mu1p = self.composite_body.mu_funcs(sdf_val1)
        #         m_m0p = (1-mu0p)

        #         (_, normal_x, normal_y, _) = self.composite_body.compute_sdf_properties(sdf_val)
        #         xi0 = m_m0-m_m0*m_m0p
        #         xi1 = mu1-(mu1*mu1p)

        #     body_u = self.composite_body.u_vals[i]
        #     body_v = self.composite_body.v_vals[i]
        #     self.div_body=xi0*self.divergence(body_u,body_v)

        #     uprime += xi0*body_u+mu1*self.normal_derivative(uprime-body_u,normal_x,normal_y)
        #     vprime += xi0*body_v+mu1*self.normal_derivative(vprime-body_v,normal_x,normal_y)

        #     # uprime += 1000*(m_m0-intersect)*(body_u-uprime)*self.dt
        #     # vprime += 1000*(m_m0-intersect)*(body_v-vprime)*self.dt

        #     # # brinkmann implicit
        #     # uprime = (self.brinkmann_k*xi0*body_u+uprime/self.dt)/(1/self.dt+self.brinkmann_k*xi0)
        #     # vprime = (self.brinkmann_k*xi0*body_v+vprime/self.dt)/(1/self.dt+self.brinkmann_k*xi0)

        #     # # brinkmann explicit
        #     # uprime += self.brinkmann_k*xi0*(body_u-uprime)*self.dt
        #     # vprime += self.brinkmann_k*xi0*(body_v-vprime)*self.dt

        #     # uprime += 1000*m_m0*(body_u-uprime)*self.dt
        #     # vprime += 1000*m_m0*(body_v-vprime)*self.dt

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)

        # # brinkmann implicit
        # uprime = (self.brinkmann_k*self.m_m0_all*self.composite_body.body_u+u/self.dt)/(1/self.dt+self.brinkmann_k*self.m_m0_all)
        # vprime = (self.brinkmann_k*self.m_m0_all*self.composite_body.body_v+v/self.dt)/(1/self.dt+self.brinkmann_k*self.m_m0_all)

        # ====== BDIM2 =====
        uprime = self.mu0_all*u + self.m_m0_all*self.composite_body.body_u + self.mu1_all*self.normal_derivative(u-self.composite_body.body_u,self.normal_x,self.normal_y)
        vprime = self.mu0_all*v + self.m_m0_all*self.composite_body.body_v + self.mu1_all*self.normal_derivative(v-self.composite_body.body_v,self.normal_x,self.normal_y)

        # self.adv_diff_solver.set_BCs(uprime,vprime)


        self.div_body=self.divergence(self.composite_body.body_u,self.composite_body.body_v)
        self.div=(self.divergence(uprime,vprime)-self.m_m0_all*self.div_body)

        # c = self.mu0_all
        # ch = (c[1:,1:-1]+c[:-1,1:-1])/2
        # cv = (c[1:-1,1:]+c[1:-1,:-1])/2
        # p, _ = self.poisson_solver.solve_multigrid( # f, u, c
        #     self.div[1:-1,1:-1],
        #     torch.zeros_like(u),
        #     c,
        #     ch=ch,
        #     cv=cv,
        # )
        # p/=(self.dt/self.rho)
        # # ====== projection step ======
        # (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        # (u,v)=(uprime-self.mu0_all*(self.dt/self.rho)*p_x, vprime-self.mu0_all*(self.dt/self.rho)*p_y)


        p=self.poisson_solverFFT.solve(self.div/(self.dt/self.rho))
        (p_x, p_y) = self.gradient(p)
        (u,v)=(uprime-(self.dt/self.rho)*p_x, vprime-(self.dt/self.rho)*p_y)

        # rhs_p=self.div
        # p=self.poisson_solverFFT.solve(rhs_p)
        # (p_x, p_y) = self.gradient(p)
        # (u,v)=(uprime-p_x, vprime-p_y)



        self.adv_diff_solver.set_BCs(u,v)


        # self.mu0_all, self.mu1_all = self.composite_body.mu_funcs(self.composite_body.sdf_vals[0])
        if self.compute_forces:
            self.compute_fluid_forces(u,v,p,iteration)

        else:
            self.delta = torch.zeros_like(u)




        return (u,v,p)

    def cc2fc(self,uprime,vprime):
        U=torch.zeros(self.nx+1,self.ny,device=self.device,dtype=self.dtype)
        V=torch.zeros(self.nx,self.ny+1,device=self.device,dtype=self.dtype)
        U[1:-1,:] = 0.5*(uprime[1:,:]+uprime[:-1,:])
        U[0,:]    = 1.5*uprime[0,:]-0.5*uprime[1,:]
        U[-1,:]   = 1.5*uprime[-1,:]-0.5*uprime[-2,:]
        V[:,1:-1] = 0.5*(vprime[:,1:]+vprime[:,:-1])
        V[:,0]    = 1.5*vprime[:,0]-0.5*vprime[:,1]
        V[:,-1]   = 1.5*vprime[:,-1]-0.5*vprime[:,-2]
        return (U,V)

    def fc2cc(self,p):
        u=torch.zeros_like(p)
        u[:-1,:]=0.5*(p[1:,:]+p[:-1,:])
        u[-1,:]=1.5*p[-2,:]-0.5*p[-3,:]
        return u


    def solver_iteration_test3(self,u,v,p,iteration):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)
        self.adv_diff_solver.set_BCs(u,v)

        # # brinkmann implicit
        # uprime = (self.brinkmann_k*self.m_m0_all*self.composite_body.body_u+u/self.dt)/(1/self.dt+self.brinkmann_k*self.m_m0_all)
        # vprime = (self.brinkmann_k*self.m_m0_all*self.composite_body.body_v+v/self.dt)/(1/self.dt+self.brinkmann_k*self.m_m0_all)

        # ====== STEP 1 =====
        uprime = self.mu0_all*u + self.m_m0_all*self.composite_body.body_u + self.mu1_all*self.normal_derivative(u-self.composite_body.body_u,self.normal_x,self.normal_y)
        vprime = self.mu0_all*v + self.m_m0_all*self.composite_body.body_v + self.mu1_all*self.normal_derivative(v-self.composite_body.body_v,self.normal_x,self.normal_y)

        # ====== STEP 2 =====
        (U,V) = self.cc2fc(uprime,vprime)
        self.div=(U[1:,:]-U[:-1,:])/self.dx+(V[:,1:]-V[:,:-1])/self.dy

        # p_fc=self.poisson_solverFFT.solve(self.div/(self.dt/self.rho))
        # p=self.fc2cc(p_fc)

        self.div_body=self.divergence(self.composite_body.body_u,self.composite_body.body_v)
        self.div=(self.divergence(uprime,vprime)-self.m_m0_all*self.div_body)

        c = self.mu0_all
        ch = (c[1:,1:-1]+c[:-1,1:-1])/2
        cv = (c[1:-1,1:]+c[1:-1,:-1])/2
        p, _ = self.poisson_solver.solve_multigrid( # f, u, c
            self.div[1:-1,1:-1], #/(self.dt/self.rho),
            p,
            c,
            ch=ch,
            cv=cv,
        )
        # p/=(self.dt)
        # ====== projection step ======
        (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        (u,v)=(uprime-self.mu0_all*p_x, vprime-self.mu0_all*p_y)


        # p=self.poisson_solverFFT.solve(self.div/(self.dt/self.rho))
        # (p_x, p_y) = self.gradient(p)
        # (u,v)=(uprime-(self.dt/self.rho)*p_x, vprime-(self.dt/self.rho)*p_y)

        # self.adv_diff_solver.set_BCs(u,v)

        if self.compute_forces:
            self.compute_fluid_forces(u,v,p,iteration)

        else:
            self.delta = torch.zeros_like(u)





        return (u,v,p)


    def solver_iteration(self,u,v,p,iteration):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)
        uprime = self.mu0_all*u + self.m_m0_all*self.composite_body.body_u + self.mu1_all*self.normal_derivative(u-self.body_u,self.normal_x,self.normal_y)
        vprime = self.mu0_all*v + self.m_m0_all*self.composite_body.body_v + self.mu1_all*self.normal_derivative(v-self.body_v,self.normal_x,self.normal_y)
        self.adv_diff_solver.set_BCs(uprime,vprime)

        self.div=(self.divergence(uprime,vprime))[1:-1,1:-1]
        coeff = torch.ones((self.nx,self.ny),device=self.device) #*self.mu0_all[1:-1,1:-1]
        coeff_horizontal = (coeff[1:,1:-1]+coeff[:-1,1:-1])/2
        coeff_vertical = (coeff[1:-1,1:]+coeff[1:-1,:-1])/2
        # coeff_horizontal=torch.vstack((self.zc,coeff_horizontal,self.zc))
        # coeff_vertical=torch.hstack((self.zr,coeff_vertical,self.zr))

        p, _ = self.poisson_solver.solve_multigrid( # f, u, c
            self.div,
            p,
            coeff,
            coeff_horizontal,
            coeff_vertical,
        )
        # ====== projection step ======
        (p_x, p_y) = self.gradient(p)
        (u,v)=(uprime-p_x, vprime-p_y)

        # self.div=self.divergence(uprime,vprime)
        # self.div_body=self.divergence(self.composite_body.body_u,self.composite_body.body_v)
        # p=self.poisson_solverFFT.solve(self.div-self.m_m0_all*self.div_body)
        # (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        # (u,v)=(uprime-p_x, vprime-p_y)

        # self.mu0_all, self.mu1_all = self.composite_body.mu_funcs(self.composite_body.sdf_vals[0])
        if self.compute_forces:
            self.compute_fluid_forces(u,v,p,iteration)

        else:
            self.delta = torch.zeros_like(u)

        self.xstress_tensor = self.force_x_interp.F

        # self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solve_heun(self, u, v, p, iteration):
        # (u1,v1,p) = self.solver_iteration(u,v,p,iteration)
        # (u2,v2,p) = self.solver_iteration(u1,v1,p,iteration)
        # u=0.5*(u1+u2)
        # v=0.5*(v1+v2)
        # return (u,v,p)
        (u1,v1,p) = self.solver_iteration_test3(u,v,p,iteration)
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

        # if self.compute_forces:
        #     self.compute_fluid_forces(u,v,p)
        # else:
        #     self.delta = torch.zeros_like(u)

        (u,v,p) = self.solve_heun(u,v,p,iteration)

        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        # ============ plotting/saving ==========
        if not iteration % self.save_every:
            if self.save_frames:


                # copy u from device to host
                X,Y = self.X.cpu(), self.Y.cpu()
                curl = self.vorticity(u,v).cpu()
                divergence = self.div.cpu()
                # d_min = self.composite_body.sdf_val.cpu()
                # d_min = self.d_min.cpu()
                sdf=self.composite_body.sdf_val.cpu()
                pressure = p.cpu()
                # curl_body = (self.vorticity(self.body_u,self.body_v)).cpu()
                vec_x=(self.m_m0_all*self.composite_body.body_u).cpu()
                vec_y=(self.m_m0_all*self.composite_body.body_v).cpu()

                # tmp=p.cpu()
                tmp = (self.delta).cpu()
                # tmp = (self.m_m0_all*self.divergence(self.body_u,self.body_v)).cpu() #(self.tmp).cpu()

                # plotting.plot2d_imshow(X,Y,(self.vorticity(u,v)/(self.composite_body.bodies[0].L)).cpu(),d_min,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax)

                plotting.plot2d_imshow_composite_quiver(X,Y,curl,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # plotting.plot2d_imshow_composite_quiver(X,Y,curl_body,self.sdf_properties,vec_x,vec_y,self.extent,iteration,self.save_path,"curlbody",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                # plotting.plot2d_imshow_composite_quiver(X,Y,tmp.cpu(),self.composite_body.bodies,vec_x,vec_y,self.extent,iteration,self.save_path,"tmp",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # # plotting.plot2d_imshow_simple((self.mu1_all/self.eps).cpu(),self.extent,iteration,self.save_path,"mu1",0,0.2)

                # plotting.plot2d_imshow_composite_quiver(X,Y,self.mu0_all.cpu(),self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"mu0",0,1,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)


                plotting.plot2d_imshow_composite(X,Y,u.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"u",None, None)
                plotting.plot2d_imshow_composite(X,Y,v.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"v",None, None)

                # plotting.plot2d_imshow_composite_quiver(X,Y,vec_x,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"bodyu",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # plotting.plot2d_imshow_composite_quiver(X,Y,vec_y,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"bodyv",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)


                # # plotting.plot_ctrs(self.force_ctr_x,self.composite_body.bodies,self.extent, self.save_path, "contours", iteration,None, None)


                plotting.plot2d_imshow_composite_quiver(X,Y,divergence,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"divergence",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                # # plotting.plot2d_imshow(X,Y,pressure,d_min,self.extent,iteration,self.save_path,"pressure",None, None)
                plotting.plot2d_imshow_composite_quiver(X,Y,pressure,self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"pressure",None,None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)


                # # plotting.plot2d_imshow_only((self.m_m0_all).cpu(),self.extent,iteration,self.save_path,"tmp",None, None)

                # # plotting.plot2d_imshow_composite_quiver(X,Y,sdf,self.composite_body.bodies,vec_x,vec_y,self.extent,iteration,self.save_path,"sdf",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                # # plotting.plot2d_imshow_quiver(X,Y,curl,d_min,vec_x,vec_y,self.extent,iteration,self.save_path,"curluv",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)


                plotting.plot2d_imshow_composite_quiver(X,Y,self.xstress_tensor.cpu(),self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"xstress_tensor",None,None,subsample_n = self.n_quiver_spacing, scale=1)
                # # plotting.plot2d_imshow_composite_quiver(X,Y,self.ystress_tensor.cpu(),self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"ystress_tensor",None,None,subsample_n = self.n_quiver_spacing, scale=1)

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


