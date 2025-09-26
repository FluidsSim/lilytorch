
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
from spreading_operator import spreading_operator_python_parallel, spreading_operator_python_parallel_out
from spreading_operator import interpolation_operator, spreading_operator_python

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

        self.dx         = torch.tensor((self.xmax-self.xmin)/self.N)
        self.dy         = torch.tensor((self.ymax-self.ymin)/self.N)

        x=torch.linspace(self.xmin-self.dx/2,self.xmax+self.dx/2,self.N+2,device=self.device,dtype=self.dtype)
        y=torch.linspace(self.ymin-self.dy/2,self.ymax+self.dy/2,self.N+2,device=self.device,dtype=self.dtype)

        self.x = x
        self.y = y

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
        self.re      = 1/self.nu

        print("Reynolds number: ", self.re)
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

        self.interp_utility = RegularGridInterpolator(
            (x,y),
            torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
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

        # self.force_ctr_x = [torch.zeros((body.cnt.shape[1])) for body in self.composite_body.bodies]
        # self.force_ctr_y = [torch.zeros((body.cnt.shape[1])) for body in self.composite_body.bodies]

        self.force_ctr_x = torch.zeros((self.n_bodies,self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.force_ctr_y = torch.zeros((self.n_bodies,self.nx,self.ny),device=self.device,dtype=self.dtype)

        self.force_f_ibm_x = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)
        self.force_f_ibm_y = torch.zeros((self.nx,self.ny),device=self.device,dtype=self.dtype)

        self.out_f_util = torch.zeros(self.nx*self.ny,device=self.device,dtype=self.dtype)

        self.ds = self.composite_body.bodies[0].ds


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

        # bodies=self.composite_body.bodies

        # # ======= compute stress tensor ======
        # dudx, dudy = self.gradient(u)
        # dvdx, dvdy = self.gradient(v)
        # ss_diag = (dudy+dvdx)
        # ss_11 = 2*dudx
        # ss_22 = 2*dvdy
        # self.xstress_tensor = self.visc*(self.normal_x*ss_11+self.normal_y*ss_diag)
        # self.ystress_tensor = self.visc*(self.normal_x*ss_diag+self.normal_y*ss_22)


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


        self.force_f_ibm_x[:]=0
        self.force_f_ibm_y[:]=0

        for i, body in enumerate(self.composite_body.bodies[:]):

            ####################################################################
            # # compute flow velocity in contours via interpolation

            cnt_f_u=self.interpolate(body.cnt_update[0],body.cnt_update[1],u)
            cnt_f_v=self.interpolate(body.cnt_update[0],body.cnt_update[1],v)

            # cnt_f_u=interpolation_operator(
            #     self.x,self.y,
            #     body.cnt_update[0],body.cnt_update[1],
            #     body.cnt_update.shape[1], self.nx, self.ny,
            #     self.dx, self.dy,
            #     u,
            # )
            # cnt_f_v=interpolation_operator(
            #     self.x,self.y,
            #     body.cnt_update[0],body.cnt_update[1],
            #     body.cnt_update.shape[1], self.nx, self.ny,
            #     self.dx, self.dy,
            #     v,
            # )


            body.force_u_L=((body.cnt_u-cnt_f_u)/self.dt)
            body.force_v_L=((body.cnt_v-cnt_f_v)/self.dt)
            # ds=np.diff(body.curv_coord[mask])

            # du = self.composite_body.body_u[i]-u
            # dv = self.composite_body.body_v[i]-v

            # self.force_ctr_x[i] += self.dt*du
            # self.force_ctr_y[i] += self.dt*dv

            # alpha=50
            # beta=10
            # self.force_x_interp.F = alpha*self.force_ctr_x[i]+beta*du
            # self.force_y_interp.F = alpha*self.force_ctr_y[i]+beta*dv

            # f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
            # f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

            # body.force_u_L=f_x
            # body.force_v_L=f_y

            mask=body.mask
            ds=body.ds


            self.friction_force_lin_x[i]=-torch.sum(body.force_u_L[mask]*ds)
            self.friction_force_lin_y[i]=-torch.sum(body.force_v_L[mask]*ds)
            self.friction_force_ang_z[i]=-torch.sum(self.cross_product_2d(body.r_com[0][mask],body.r_com[1][mask],body.force_u_L[mask],body.force_v_L[mask]))*body.ds

            # spreading_operator_python_parallel_out(
            #     body.cnt_update[0,mask], body.cnt_update[1,mask], self.x, self.y, self.nx, self.ny, self.dx, self.dy, ds, body.force_u_L[mask], self.force_f_ibm_x.flatten()
            # )
            # spreading_operator_python_parallel_out(
            #     body.cnt_update[0,mask], body.cnt_update[1,mask], self.x, self.y, self.nx, self.ny, self.dx, self.dy, ds, body.force_v_L[mask], self.force_f_ibm_y.flatten()
            # )


            ####################################################################
            ####################################################################
            ####################################################################
            ####################################################################

            # # method 2
            # d=self.composite_body.sdf_vals[i]
            # delta=self.composite_body.bodies[0].phi(d)*self.eps
            # dsign=torch.sign(self.composite_body.sdf_vals[i])

            # self.friction_force_lin_x[i] = torch.sum(torch.sum(self.nu*self.xstress_tensor*delta))*self.dx*self.dy


            # self.force_x_interp.F = self.xstress_tensor
            # self.force_y_interp.F = self.ystress_tensor

            # f_x=self.force_x_interp(body.cnt_update[0][body.mask], body.cnt_update[1][body.mask])
            # f_y=self.force_y_interp(body.cnt_update[0][body.mask], body.cnt_update[1][body.mask])

            # self.friction_force_lin_x[i] = f_x.sum()*body.ds
            # self.friction_force_lin_y[i] = f_y.sum()*body.ds

            # print("Total force x: ", self.friction_force_lin_x)




            # if iteration*self.dt<1:
            #     self.friction_force_lin_x[i] = 0.02
            #     self.friction_force_lin_y[i] = 0.0
            # else:

                # ================ compute viscous forces ================

                # d=self.composite_body.sdf_vals[i]
                # self.delta = self.composite_body.bodies[0].phi(d)*self.eps
                # self.delta = self.composite_body.bodies[0].phi(self.composite_body.sdf_vals[i])*phi_all*self.eps

                # # method 1
                # self.force_x_interp.F = self.xstress_tensor
                # self.force_y_interp.F = self.ystress_tensor

                # f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                # f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

                # self.friction_force_lin_x[i] = torch.trapezoid(f_x[:-1], body.curv_coord)
                # self.friction_force_lin_y[i] = torch.trapezoid(f_y[:-1], body.curv_coord)

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

                # du = self.composite_body.body_u[i]-u
                # dv = self.composite_body.body_v[i]-v

                # self.force_ctr_x[i] += self.dt*du
                # self.force_ctr_y[i] += self.dt*dv

                # alpha=50
                # beta=10
                # self.force_x_interp.F = alpha*self.force_ctr_x[i]+beta*du
                # self.force_y_interp.F = alpha*self.force_ctr_y[i]+beta*dv

                # f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                # f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

                # self.friction_force_lin_x[i] = torch.trapz(f_x[:-1], body.curv_coord)
                # self.friction_force_lin_y[i] = torch.trapz(f_y[:-1], body.curv_coord)


                # self.viscous_drag_record[i,0,iteration] = self.friction_force_lin_x[i]
                # self.viscous_drag_record[i,1,iteration] = self.friction_force_lin_y[i]


                # ================ compute pressure forces ================
                # self.force_x_interp.F = -p*self.normal_x
                # self.force_y_interp.F = -p*self.normal_y


                # # pforce=p-d*self.normal_derivative(p, self.normal_x, self.normal_y)
                # # self.force_x_interp.F = 0.01*pforce*self.normal_x
                # # self.force_y_interp.F = 0.01*pforce*self.normal_y


                # f_x=self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                # f_y=self.force_y_interp(body.cnt_update[0], body.cnt_update[1])


                # self.pressure_force_x[i] = torch.trapezoid(f_x[:-1], body.curv_coord)
                # self.pressure_force_y[i] = torch.trapezoid(f_y[:-1], body.curv_coord)



                # self.pressure_drag_record[i,0,iteration] = self.pressure_force_x[i]
                # self.pressure_drag_record[i,1,iteration] = self.pressure_force_y[i]


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


        if self.compute_forces:
            self.compute_fluid_forces(u,v,p,iteration)

        else:
            self.delta = torch.zeros_like(u)

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
                # tmp = (self.delta).cpu()
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

                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    self.force_f_ibm_x.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "force_f_ibm_x",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np,
                    body_contours=False
                )

                plotting.plot2d_imshow_composite_quiver(X,Y,self.xstress_tensor.cpu(),self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"xstress_tensor",None,None,subsample_n = self.n_quiver_spacing, scale=1)
                # # plotting.plot2d_imshow_composite_quiver(X,Y,self.ystress_tensor.cpu(),self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"ystress_tensor",None,None,subsample_n = self.n_quiver_spacing, scale=1)

            if self.save_uv:
                # uv_path = f'{self.save_path}/uv_field'
                # os.makedirs(uv_path, exist_ok=True)
                # np.save(f'{uv_path}/u_{iteration}',u.cpu().numpy())
                # np.save(f'{uv_path}/v_{iteration}',v.cpu().numpy())

                csv_path = f'{self.save_path}/csv'
                os.makedirs(csv_path, exist_ok=True)
                np.savetxt(csv_path + f"/sdf_{iteration}.csv", self.composite_body.sdf_val.cpu().numpy(), delimiter=",")
                np.savetxt(csv_path + f"/normal_x_{iteration}.csv", self.normal_x.cpu().numpy(), delimiter=",")
                np.savetxt(csv_path + f"/normal_y_{iteration}.csv", self.normal_y.cpu().numpy(), delimiter=",")
                np.savetxt(csv_path + f"/body_u_{iteration}.csv", self.composite_body.body_u.cpu().numpy(), delimiter=",")
                np.savetxt(csv_path + f"/body_v_{iteration}.csv", self.composite_body.body_v.cpu().numpy(), delimiter=",")


        continue_sim=self.outside(self.composite_body.com_pos)

        return (u,v,p,continue_sim)


    def interpolate(self,x,y,val):
        self.interp_utility.F = val
        return self.interp_utility(x,y)

    def cross_product_2d(self,ax,ay,bx,by):
        return ax*by-ay*bx

    def step_pibm(self, u, v, p, iteration, t):

        alpha=1000
        beta=10
        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        for i, body in enumerate(self.composite_body.bodies):

            # compute flow velocity in contours via interpolation
            body.cnt_f_u[:]=0
            body.cnt_f_v[:]=0
            mask=body.mask

            # cnt_f_u=self.interpolate(body.cnt_update[0],body.cnt_update[1],u)
            # cnt_f_v=self.interpolate(body.cnt_update[0],body.cnt_update[1],v)

            cnt_f_u=interpolation_operator(
                self.x,self.y,
                body.cnt_update[0],body.cnt_update[1],
                body.cnt_update.shape[1], self.nx, self.ny,
                self.dx, self.dy,
                u,
            )
            cnt_f_v=interpolation_operator(
                self.x,self.y,
                body.cnt_update[0],body.cnt_update[1],
                body.cnt_update.shape[1], self.nx, self.ny,
                self.dx, self.dy,
                v,
            )

            du=body.cnt_u-cnt_f_u
            dv=body.cnt_v-cnt_f_v

            body.cnt_int_f_u+=self.dt*du
            body.cnt_int_f_v+=self.dt*dv

            body.cnt_f_u=alpha*body.cnt_int_f_u+beta*du
            body.cnt_f_v=alpha*body.cnt_int_f_v+beta*dv

            self.friction_force_lin_x[i]=torch.sum(body.cnt_f_u[mask])*body.ds
            self.friction_force_lin_y[i]=-torch.sum(body.cnt_f_v[mask])*body.ds
            # self.friction_force_ang_z[i]=-torch.sum(self.cross_product_2d(body.com_pos[0],body.com_pos[1],body.cnt_f_u[mask],body.cnt_f_v[mask]))*body.ds



        # cnt_all = torch.cat([body.cnt_update[:,body.mask] for body in self.composite_body.bodies],dim=1)
        # f_u_all = torch.cat([body.cnt_f_u[body.mask] for body in self.composite_body.bodies],dim=0)
        # f_v_all = torch.cat([body.cnt_f_v[body.mask] for body in self.composite_body.bodies],dim=0)


        # # compute forces on fluid - spread forces on fluid to eulerian grid
        # self.force_f_ibm_x=spreading_operator_python_parallel(cnt_all[0], cnt_all[1], self.x, self.y, self.nx, self.ny, self.dx, self.dy, self.ds, f_u_all)
        # self.force_f_ibm_y=spreading_operator_python_parallel(cnt_all[0], cnt_all[1], self.x, self.y, self.nx, self.ny, self.dx, self.dy, self.ds, f_v_all)


        # (u,v)=(u+self.dt*self.force_f_ibm_x/self.rho,v+self.dt*self.force_f_ibm_y/self.rho)






        # if iteration==200:
        #     from IPython import embed; embed()



        #     import matplotlib.pyplot as plt
        #     body=self.composite_body.bodies[3]
        #     plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c=body.cnt_f_u, s=10)
        #     plt.show()




        # if iteration==350:
        #     import matplotlib.pyplot as plt

        #     force_grid = self.force_f_ibm_x.detach().cpu().numpy().reshape((self.nx,self.ny))


        #     # Get force on contour (Lagrangian)
        #     cnt_x = cnt_all[0].detach().cpu().numpy()
        #     cnt_y = cnt_all[1].detach().cpu().numpy()
        #     cnt_f_u = f_u_all.detach().cpu().numpy()



        #     # Plot Eulerian force field
        #     fig1, ax1 = plt.subplots(figsize=(8, 6))
        #     im = ax1.imshow(force_grid.T, origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax], cmap='bwr')
        #     ax1.set_title('force_f_ibm_x (grid)')
        #     ax1.set_xlabel('x')
        #     ax1.set_ylabel('y')
        #     plt.colorbar(im, ax=ax1, label='force_f_ibm_x')
        #     plt.tight_layout()

        #     # Plot Lagrangian contour forces as scatter
        #     fig2, ax2 = plt.subplots(figsize=(8, 6))
        #     scatter = ax2.scatter(cnt_x, cnt_y, c=cnt_f_u, cmap='bwr', edgecolor='None', label='cnt_f_u')
        #     ax2.set_title('cnt_f_u (contour)')
        #     ax2.set_xlabel('x')
        #     ax2.set_ylabel('y')
        #     plt.colorbar(scatter, ax=ax2, label='cnt_f_u (contour)')
        #     plt.tight_layout()

        #     plt.show()


        # # from IPython import embed; embed()






        (u,v) = self.adv_diff_solver.solve(u,v)
        self.adv_diff_solver.set_BCs(u,v)

        uprime = u
        vprime = v


        # (self.mu0_all, self.mu1_all) = body.mu_funcs(body.sdf)
        # self.m_m0_all=(1-self.mu0_all)

        # uprime = self.mu0_all*u
        # vprime = self.mu0_all*v




        self.div=self.divergence(uprime,vprime)

        p=self.poisson_solverFFT.solve(self.div/(self.dt/self.rho))
        (p_x, p_y) = self.gradient(p)
        (u,v)=(uprime-(self.dt/self.rho)*p_x, vprime-(self.dt/self.rho)*p_y)


        # c = torch.ones((self.nx,self.ny),device=self.device) #*self.mu0_all
        # ch = (c[1:,1:-1]+c[:-1,1:-1])/2
        # cv = (c[1:-1,1:]+c[1:-1,:-1])/2
        # p, _ = self.poisson_solver.solve_multigrid( # f, u, c
        #     self.div[1:-1,1:-1], #/(self.dt/self.rho),
        #     p,
        #     c,
        #     ch=ch,
        #     cv=cv,
        # )
        # # p/=(self.dt)
        # # ====== projection step ======
        # (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        # (u,v)=(uprime-p_x, vprime-p_y)




        self.adv_diff_solver.set_BCs(u,v)




        # ============ plotting/saving ==========
        if not iteration % self.save_every:
            if self.save_frames:


                # copy u from device to host
                X,Y = self.X.cpu(), self.Y.cpu()
                curl = self.vorticity(u,v).cpu()

                # divergence = self.div.cpu()
                # d_min = self.composite_body.sdf_val.cpu()
                # d_min = self.d_min.cpu()
                # sdf=self.composite_body.sdf_val.cpu()
                # pressure = p.cpu()
                # curl_body = (self.vorticity(self.body_u,self.body_v)).cpu()
                # vec_x=(self.m_m0_all*self.composite_body.body_u).cpu()
                # vec_y=(self.m_m0_all*self.composite_body.body_v).cpu()

                # tmp=p.cpu()
                # tmp = (self.delta).cpu()
                # tmp = (self.m_m0_all*self.divergence(self.body_u,self.body_v)).cpu() #(self.tmp).cpu()

                # plotting.plot2d_imshow(X,Y,(self.vorticity(u,v)/(self.composite_body.bodies[0].L)).cpu(),d_min,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax)

                plotting.plot2d_imshow_composite_quiver(X,Y,curl,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"curl",None,None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                # from IPython import embed; embed()
                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    self.force_f_ibm_x.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "force_f_ibm_x",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np,
                    body_contours=False
                )

                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    u.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "u",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np
                )

                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    v.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "v",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np
                )

                # plotting.plot2d_imshow_composite_quiver(X,Y,curl_body,self.sdf_properties,vec_x,vec_y,self.extent,iteration,self.save_path,"curlbody",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                # plotting.plot2d_imshow_composite_quiver(X,Y,tmp.cpu(),self.composite_body.bodies,vec_x,vec_y,self.extent,iteration,self.save_path,"tmp",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # # plotting.plot2d_imshow_simple((self.mu1_all/self.eps).cpu(),self.extent,iteration,self.save_path,"mu1",0,0.2)

                # plotting.plot2d_imshow_composite_quiver(X,Y,self.mu0_all.cpu(),self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"mu0",0,1,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)


                # plotting.plot2d_imshow_composite(X,Y,u.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"u",None, None)
                # plotting.plot2d_imshow_composite(X,Y,v.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"v",None, None)

                # plotting.plot2d_imshow_composite_quiver(X,Y,vec_x,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"bodyu",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)
                # plotting.plot2d_imshow_composite_quiver(X,Y,vec_y,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"bodyv",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)


                # # plotting.plot_ctrs(self.force_ctr_x,self.composite_body.bodies,self.extent, self.save_path, "contours", iteration,None, None)


                # plotting.plot2d_imshow_composite_quiver(X,Y,divergence,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"divergence",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                # # plotting.plot2d_imshow(X,Y,pressure,d_min,self.extent,iteration,self.save_path,"pressure",None, None)
                # plotting.plot2d_imshow_composite_quiver(X,Y,pressure,self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"pressure",None,None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)


                # # plotting.plot2d_imshow_only((self.m_m0_all).cpu(),self.extent,iteration,self.save_path,"tmp",None, None)

                # # plotting.plot2d_imshow_composite_quiver(X,Y,sdf,self.composite_body.bodies,vec_x,vec_y,self.extent,iteration,self.save_path,"sdf",None, None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                # # plotting.plot2d_imshow_quiver(X,Y,curl,d_min,vec_x,vec_y,self.extent,iteration,self.save_path,"curluv",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)


                # plotting.plot2d_imshow_composite_quiver(X,Y,self.xstress_tensor.cpu(),self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"xstress_tensor",None,None,subsample_n = self.n_quiver_spacing, scale=1)
                # # plotting.plot2d_imshow_composite_quiver(X,Y,self.ystress_tensor.cpu(),self.composite_body.bodies,0*self.normal_x.cpu(),0*self.normal_y.cpu(),self.extent,iteration,self.save_path,"ystress_tensor",None,None,subsample_n = self.n_quiver_spacing, scale=1)

        if hasattr(self.composite_body, "com_pos"):
            continue_sim=self.outside(self.composite_body.com_pos)
        else:
            continue_sim=True

        return (u,v,p,continue_sim)

    def step_fluid_ibdf(self, u, v, p, iteration, t):

        (uprime,vprime) = self.adv_diff_solver.solve(u,v)
        self.adv_diff_solver.set_BCs(uprime,vprime)

        self.div=self.divergence(uprime,vprime)

        p=self.poisson_solverFFT.solve(self.div/(self.dt/self.rho))
        (p_x, p_y) = self.gradient(p)
        return (uprime-(self.dt/self.rho)*p_x, vprime-(self.dt/self.rho)*p_y)


        # c = torch.ones((self.nx,self.ny),device=self.device) #*self.mu0_all
        # ch = (c[1:,1:-1]+c[:-1,1:-1])/2
        # cv = (c[1:-1,1:]+c[1:-1,:-1])/2
        # p, _ = self.poisson_solver.solve_multigrid( # f, u, c
        #     self.div[1:-1,1:-1], #/(self.dt/self.rho),
        #     p,
        #     c,
        #     ch=ch,
        #     cv=cv,
        # )
        # # p/=(self.dt)
        # # ====== projection step ======
        # (p_x, p_y) = torch.gradient(p,spacing=self.dxdy,edge_order=2)
        # return (uprime-p_x, vprime-p_y)


    def build_arc_length_param(self, pts, f, n_points=30):
        """
        Given a set of points (2 x N), reparameterize them so that the output points are equally spaced in arclength.
        Returns new_pts (2 x n_points) and ds (the constant arclength spacing).
        """
        import numpy as np
        import torch

        # Convert to numpy for processing
        pts_np = pts.cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
        n_pts = pts_np.shape[1]

        # Order points by nearest neighbor to approximate arclength order
        ordered_idx = [0]
        used = set(ordered_idx)
        for _ in range(1, n_pts):
            last = ordered_idx[-1]
            dists = np.linalg.norm(pts_np[:, last][:, None] - pts_np, axis=0)
            dists[list(used)] = np.inf  # mask already used
            next_idx = np.argmin(dists)
            ordered_idx.append(next_idx)
            used.add(next_idx)
        pts_ordered = pts_np[:, ordered_idx]

        # Compute cumulative arclength
        diffs = np.diff(pts_ordered, axis=1)
        seg_lengths = np.sqrt((diffs**2).sum(axis=0))
        s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        L = s[-1]

        # Interpolate at equally spaced arclengths
        s_uniform = np.linspace(0, L, n_points)
        xs, ys = pts_ordered[0], pts_ordered[1]
        xq = np.interp(s_uniform, s, xs)
        yq = np.interp(s_uniform, s, ys)
        new_pts = torch.tensor([xq, yq], dtype=pts.dtype, device=pts.device)
        ds = L / (n_points - 1)

        # Interpolate function values f along the new arclength parameterization if f is provided
        if f is not None:
            f_np = f.cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            if f_np.shape[0] == n_pts:
                fq = np.interp(s_uniform, s, f_np[ordered_idx])
                new_f = torch.tensor(fq, dtype=f.dtype, device=f.device)
            else:
                new_f = None
        else:
            new_f = None

        # Optional: plot for debugging and check arclength spacing
        try:
            import matplotlib.pyplot as plt
            plt.scatter(xs, ys, label='Original Points', color='blue', s=10)
            plt.scatter(xq, yq, label='Equally Spaced Points', color='red', s=10)
            plt.legend()
            plt.show()
            # Check arclength spacing
            diffs = np.diff(np.stack([xq, yq], axis=0), axis=1)
            seg_lengths = np.sqrt((diffs**2).sum(axis=0))
            print("Arclength spacing min/max/mean:", seg_lengths.min(), seg_lengths.max(), seg_lengths.mean())
        except ImportError:
            pass

        return new_pts, ds, new_f



        # xs, ys = pts[:,0], pts[:,1]

        # # --- build interpolation ---
        # if SCIPY_AVAILABLE:
        #     bc_type = 'periodic' if close_curve else 'not-a-knot'
        #     csx = CubicSpline(s, xs, bc_type=bc_type)
        #     csy = CubicSpline(s, ys, bc_type=bc_type)
        #     def eval_xy(sq):
        #         sq = np.mod(sq, L) if close_curve else np.clip(sq, 0, L)
        #         return np.array([csx(sq), csy(sq)]).T
        # else:
        #     def eval_xy(sq):
        #         sq = np.mod(sq, L) if close_curve else np.clip(sq, 0, L)
        #         xq = np.interp(sq, s, xs)
        #         yq = np.interp(sq, s, ys)
        #         return np.array([xq, yq]).T

        # def sample_uniform(n, endpoint=True):
        #     grid = np.linspace(0, L, n, endpoint=endpoint)
        #     return eval_xy(grid)

        # return s, pts, eval_xy, L, sample_uniform


    def apply_forcing(self, up, vp, iteration):

        self.force_f_ibm_x[:]=0
        self.force_f_ibm_y[:]=0

        for i, body in enumerate(self.composite_body.bodies[:]):

            # compute flow velocity in contours via interpolation
            body.cnt_f_u[:]=0
            body.cnt_f_v[:]=0

            # cnt_f_u=self.interpolate(body.cnt_update[0],body.cnt_update[1],u)
            # cnt_f_v=self.interpolate(body.cnt_update[0],body.cnt_update[1],v)

            cnt_f_u=interpolation_operator(
                self.x,self.y,
                body.cnt_update[0],body.cnt_update[1],
                body.cnt_update.shape[1], self.nx, self.ny,
                self.dx, self.dy,
                up,
            )
            cnt_f_v=interpolation_operator(
                self.x,self.y,
                body.cnt_update[0],body.cnt_update[1],
                body.cnt_update.shape[1], self.nx, self.ny,
                self.dx, self.dy,
                vp,
            )

            mask=body.mask

            body.force_u_L=((body.cnt_u-cnt_f_u)/self.dt)
            body.force_v_L=((body.cnt_v-cnt_f_v)/self.dt)
            # ds=np.diff(body.curv_coord[mask])
            ds=body.ds

            if False: #iteration<100:
                self.friction_force_lin_y[i]=0.5
            else:
                self.friction_force_lin_x[i]=-torch.sum(body.force_u_L[mask]*ds)
                self.friction_force_lin_y[i]=-torch.sum(body.force_v_L[mask]*ds)
                self.friction_force_ang_z[i]=-torch.sum(self.cross_product_2d(body.r_com[0][mask],body.r_com[1][mask],body.force_u_L[mask],body.force_v_L[mask]))*body.ds

            spreading_operator_python_parallel_out(
                body.cnt_update[0,mask], body.cnt_update[1,mask], self.x, self.y, self.nx, self.ny, self.dx, self.dy, ds, body.force_u_L[mask], self.force_f_ibm_x.flatten()
            )
            spreading_operator_python_parallel_out(
                body.cnt_update[0,mask], body.cnt_update[1,mask], self.x, self.y, self.nx, self.ny, self.dx, self.dy, ds, body.force_v_L[mask], self.force_f_ibm_y.flatten()
            )

        (u,v) = (up+self.dt*self.force_f_ibm_x/self.rho,vp+self.dt*self.force_f_ibm_y/self.rho)
        # (u,v) = (up,vp)



        # cnt_all = torch.cat([body.cnt_update[:,body.mask] for body in self.composite_body.bodies],dim=1)
        # f_u_all = torch.cat([body.force_u_L[body.mask] for body in self.composite_body.bodies],dim=0)
        # f_v_all = torch.cat([body.force_v_L[body.mask] for body in self.composite_body.bodies],dim=0)






        # if iteration==150:

        #     import matplotlib.pyplot as plt
        #     import matplotlib.cm as cm

        #     force_grid = self.force_f_ibm_x.detach().cpu().numpy().reshape((self.nx,self.ny))




        #     # Get force on contour (Lagrangian)
        #     cnt_x = cnt_all[0].detach().cpu().numpy()
        #     cnt_y = cnt_all[1].detach().cpu().numpy()
        #     cnt_f_u = f_u_all.detach().cpu().numpy()



        #     # Plot Eulerian force field
        #     fig1, ax1 = plt.subplots(figsize=(8, 6))
        #     im = ax1.imshow(force_grid.T, origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax], cmap=cm.RdBu)
        #     ax1.set_title('force_f_ibm_x (grid)')
        #     ax1.set_xlabel('x')
        #     ax1.set_ylabel('y')
        #     plt.colorbar(im, ax=ax1, label='force_f_ibm_x')
        #     plt.tight_layout()



        #     # Plot Lagrangian contour forces as scatter
        #     fig2, ax2 = plt.subplots(figsize=(8, 6))
        #     scatter = ax2.scatter(cnt_x, cnt_y, c=cnt_f_u, cmap=cm.RdBu, edgecolor='None', label='cnt_f_u',s=0.5)
        #     ax2.set_title('cnt_f_u (contour)')
        #     ax2.set_xlabel('x')
        #     ax2.set_ylabel('y')
        #     plt.colorbar(scatter, ax=ax2, label='cnt_f_u (contour)')
        #     scatter.set_clim([-800, 200])
        #     plt.tight_layout()


        #     plt.figure()
        #     # body=self.composite_body.bodies[0]
        #     # plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c=body.cnt_f_u, s=10)
        #     for body in self.composite_body.bodies[:]:
        #         scatter = plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c=body.cnt_f_u.cpu(), cmap=cm.RdBu, s=0.5)
        #         scatter.set_clim([-800, 200])
        #     plt.colorbar()
        #     plt.show()



        #     from IPython import embed; embed()


        return (u,v)



    def step_pb_direct_forcing(self, u, v, p, iteration, t):


        (up,vp)=self.step_fluid_ibdf(u,v,p,iteration,t)

        self.composite_body.update(t, iteration, dt=self.dt)

        # for i in range(5):
        (u,v) = self.apply_forcing(up,vp, iteration)




        self.plottting_saving(u, v, p, iteration)


        if hasattr(self.composite_body, "com_pos"):
            continue_sim=self.outside(self.composite_body.com_pos)
        else:
            continue_sim=True

        return (u,v,p,continue_sim)



    def plottting_saving(self, u, v, p, iteration):

        # ============ plotting/saving ==========
        if not iteration % self.save_every:
            if self.save_frames:
                # copy u from device to host
                X,Y = self.X.cpu(), self.Y.cpu()
                curl = self.vorticity(u,v).cpu()

                plotting.plot2d_imshow_composite_quiver(X,Y,curl,self.composite_body.bodies,0*X,0*X,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt_np)

                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    self.force_f_ibm_x.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "force_f_ibm_x",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np,
                    body_contours=False
                )

                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    u.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "u",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np
                )

                plotting.plot2d_imshow_composite_quiver(
                    X, Y,
                    v.cpu(),
                    self.composite_body.bodies,
                    0 * X, 0 * X,
                    self.extent,
                    iteration,
                    self.save_path,
                    "v",
                    None, None,
                    subsample_n=self.n_quiver_spacing,
                    scale=self.save_every * self.dt_np
                )




    def run_from_initial(self, u0, v0):
        u=u0
        v=v0
        p=torch.zeros_like(u)
        for iteration in tqdm(range(self.nt)):
            t=iteration*self.dt
            (u,v,p,stop_sim) = self.step_pibm(u, v, p, iteration, t)

    def run_sim(self):
        u=self.u0
        v=self.v0
        p=self.p0
        for iteration in tqdm(range(self.starting_iteration, self.nt)):
            t=iteration*self.dt
            (u,v,p,stop_sim) = self.step_pb_direct_forcing(u, v, p, iteration, t)



if __name__ == "__main__":

    solver = FluidSolver()
    solver.run_sim()


