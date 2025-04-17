
from lilytorch.adv_diff import AdvDiffSolver
from lilytorch.testing_nonapprox_poisson import PoissonSolver
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

    def __init__(self, pars, costum_update=None, **kwargs):
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

        self.N          = solver["N"]+1
        xmin            = solver["xmin"]
        xmax            = solver["xmax"]
        ymin            = solver["ymin"]
        ymax            = solver["ymax"]
        x               = torch.linspace(xmin,xmax,self.N).to(self.device)
        y               = torch.linspace(ymin,ymax,self.N).to(self.device)

        [self.X,self.Y] = torch.meshgrid(x,y, indexing="ij")
        self.nx         = len(x)
        self.ny         = len(y)
        self.dx         = float(x[1]-x[0])
        self.dy         = float(y[1]-y[0])
        assert abs(self.dx-self.dy)<1e-7, "dx and dy must be equal"

        self.h2         = self.dx**2
        self.dt         = solver["dt"]
        self.nt         = solver["nt"]
        self.nu         = solver["nu"]
        self.rho        = solver["rho"]
        self.eps        = 2*self.dx
        self.visc       = self.nu*self.rho

        self.bcs_u_type = bcs["BC_type_u"]
        self.bcs_u_val  = bcs["BC_values_u"]
        self.bcs_v_type = bcs["BC_type_v"]
        self.bcs_v_val  = bcs["BC_values_v"]

        self.starting_iteration      = solver.get("starting_iteration", 0)
        self.starting_iteration_path = solver.get("starting_iteration_path", None)
        self.starting_time           = self.starting_iteration * self.dt

        print("Setting dt={}s, dx={}".format(self.dt, x[1]-x[0]))

        # ============= convection solver =============
        self.adv_diff_solver = AdvDiffSolver(
            self.device,
            self.dt,x,y,self.nu,
            BC_type_u=self.bcs_u_type, BC_values_u=self.bcs_u_val,
            BC_type_v=self.bcs_v_type, BC_values_v=self.bcs_v_val,
            method=solver["convection_method"]
        )

        # =============  poisson solver =============
        self.poisson_solver  = PoissonSolver(
            self.device,
            self.dx,
            tol=solver["poisson_tol"],
            max_cycles=solver["poisson_max_cycles"],
            nsmoothing=solver["poisson_nsmoothing"],
            verbose=solver["poisson_verbose"]
        )

        self.composite_body = body_from_yaml(
            self.device,
            x, y,
            body_pars,
            eps=self.eps,
            costum_update=costum_update,
            starting_time=self.starting_time
        )
        # self.sdf_properties = self.composite_body.initialize()
        self.n_bodies=len(self.composite_body.bodies)
        self.friction_force_x = torch.zeros(self.n_bodies)
        self.friction_force_y = torch.zeros(self.n_bodies)

        self.pressure_force_x = torch.zeros(self.n_bodies)
        self.pressure_force_y = torch.zeros(self.n_bodies)


        # Parameters for the Gaussian kernel
        kernel_dx=0.005
        self.kernel_size = round(kernel_dx/self.dx)  # Example kernel size
        self.kernel_size += 1-self.kernel_size%2
        sigma = 10  # Standard deviation for Gaussian
        self.kernel = self.composite_body.gaussian_kernel(self.kernel_size, sigma).view(1, 1, self.kernel_size, self.kernel_size).to(self.device)

        # ===== set initial conditions =====
        self.set_initial_conditions()

        # ===== plotting parameters =====
        self.extent = (
            torch.min(x.cpu()), torch.max(x.cpu()),
            torch.min(y.cpu()), torch.max(y.cpu())
        )

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
            todaystr       = today.isoformat().replace(":","-")
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
        if self.bcs_u_type[0]=="D":
            self.u0 = self.bcs_u_val[0]*torch.ones((self.nx,self.ny),device=self.device)
        else:
            self.u0 = torch.zeros((self.nx,self.ny),device=self.device)
        self.v0 = torch.zeros((self.nx,self.ny),device=self.device)
        self.p0 = torch.zeros((self.nx,self.ny),device=self.device)

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
        Compute divergence(u,v)
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

    def stress_tensor(self,u,v,p):
        """
        Compute stress tensor
        """
        dudx, dudy = torch.gradient(u, spacing=[self.dx, self.dy])
        dvdx, dvdy = torch.gradient(v, spacing=[self.dx, self.dy])
        upper = self.visc*(dudy+dvdx)
        return [
            [2*self.visc*dudx, upper],
            [upper, 2*self.visc*dvdy]
        ]

        # dudx, dudy = torch.gradient(u, spacing=[self.dx, self.dy])
        # dvdx, dvdy = torch.gradient(v, spacing=[self.dx, self.dy])
        # upper = self.visc*(dudy+dvdx)
        # # return [
        # #     [2*self.visc*dudx, upper],
        # #     [upper, 2*self.visc*dvdy]
        # # ]
        # return [
        #     [2*self.visc*dudx-p, upper],
        #     [upper, 2*self.visc*dvdy-p]
        # ]

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
        (u_ext,v_ext) = self.adv_diff_solver.solve(u,v)

        # ====== solve the pressure poisson equation and project ======
        coeff = self.dt/self.rho*torch.ones_like(u)
        # p = torch.zeros_like(u)
        p_ext = self.poisson_solver.solve_multigrid( # f, u, c
            self.divergence(u_ext,v_ext),
            p,
            coeff,
            coeff,
            coeff
        )
        (p_x, p_y) = self.gradient(p_ext)
        u_ext = u_ext-coeff*p_x
        v_ext = v_ext-coeff*p_y

        self.adv_diff_solver.set_BCs(u_ext,v_ext)

        return (u_ext,v_ext,p)

    def solver_iteration(self,u,v,p):
        """
        BDIM2 iteration
        """

        # ====== convection solver ======
        (u,v) = self.adv_diff_solver.solve(u,v)

        uprime = self.mu0_all*u + self.m_m0_all*self.body_u + self.mu1_all*self.normal_derivative(u-self.body_u,self.normal_x,self.normal_y)
        vprime = self.mu0_all*v + self.m_m0_all*self.body_v + self.mu1_all*self.normal_derivative(v-self.body_v,self.normal_x,self.normal_y)

        # ====== solve the pressure poisson equation and project ======
        coeff = self.dt*self.mu0_all/self.rho
        rhs = (self.divergence(uprime,vprime)-self.m_m0_all*self.divergence(self.body_u,self.body_v))
        # p = torch.zeros_like(u)
        p = self.poisson_solver.solve_multigrid( # f, u, c
            rhs,
            p,
            coeff,
            coeff,
            coeff
        )
        (p_x, p_y) = self.gradient(p)
        u = uprime-coeff*p_x
        v = vprime-coeff*p_y

        self.adv_diff_solver.set_BCs(u,v)

        return (u,v,p)

    def solve_euler(self, u, v, p):
        return self.solver_iteration(u,v,p)

    def solve_heun(self, u, v, p):
        (u1,v1,p) = self.solver_iteration(u,v,p)
        (u2,v2,p) = self.solver_iteration(u1,v1,p)
        u=0.5*(u1+u2)
        v=0.5*(v1+v2)
        return (u,v,p)
        # (u1,v1,p) = self.solver_iteration(u,v,p)
        # return (u1,v1,p)


    def step(self, u, v, p, iteration, t):

        # update the body to compute the velocity of the body
        sdf_properties = self.composite_body.update(t, dt=self.dt)
        d, normal_x, normal_y, curv = sdf_properties[0]
        (self.mu0, self.mu1) = self.composite_body.bodies[0].mu_funcs(self.d)

        (u,v,p) = self.solver_iteration_old(u,v,p)
        # (u,v,p) = self.solve_heun_old(u,v,p)

        # update body properties
        self.d, self.normal_x, self.normal_y, self.curv = d, normal_x, normal_y, curv

        # ============ plotting ==========
        if self.save_frames and not iteration % self.save_every:
            # copy u from device to host
            X,Y = self.composite_body.bodies[0].X.cpu(), self.composite_body.bodies[0].Y.cpu()
            curl = self.vorticity(u,v).cpu()
            d = d.cpu()
            curl_body = (self.vorticity((1-self.mu0)*self.u_,(1-self.mu0)*self.v_)).cpu()

            plotting.plot2d_imshow(X,Y,curl,d,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax)
            plotting.plot2d_imshow(X,Y,curl_body,d,self.extent,iteration,self.save_path,"body_curl",self.vmin,self.vmax)
            plotting.plot2d_imshow(X,Y,u.cpu(),d,self.extent,iteration,self.save_path,"u",self.vmin,self.vmax)
            plotting.plot2d_imshow(X,Y,v.cpu(),d,self.extent,iteration,self.save_path,"v",self.vmin,self.vmax)
            plotting.plot2d_imshow(X,Y,self.mu0.cpu(),d,self.extent,iteration,self.save_path,"mu0",self.vmin,self.vmax)
            plotting.plot2d_imshow(X,Y,p.cpu(),d,self.extent,iteration,self.save_path,"pressure", None, None)
            plotting.plot2d_imshow_quiver(X,Y,d,d,normal_x.cpu(),normal_y.cpu(),self.extent,iteration,self.save_path,"sdf",self.vmin,self.vmax)

        return (u,v,p)

    def step_(self, u, v, p, iteration, t):

        (self.mu0_all, self.mu1_all) = self.composite_body.mu_funcs(self.composite_body.sdf_val)
        self.m_m0_all = (1-self.mu0_all)
        self.body_u=self.composite_body.body_u
        self.body_v=self.composite_body.body_v
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

        self.composite_body.update(t, dt=self.dt)

        # ======= compute stress tensor ======
        # stress = self.stress_tensor(u,v,p)
        (u_ext, v_ext, p_ext) = self.solver_free(u,v,p)
        dudx, dudy = torch.gradient(u_ext, spacing=[self.dx, self.dy])
        dvdx, dvdy = torch.gradient(v_ext, spacing=[self.dx, self.dy])
        ss_diag = (dudy+dvdx)
        ss_11 = 2*dudx
        ss_22 = 2*dvdy


        for i, body in enumerate(self.composite_body.bodies):

            (d, normal_x, normal_y, R) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_vals[i])
            delta = self.composite_body.bodies[0].phi(d)/(1+d/R)

            xstress_tensor = d*(normal_x*ss_11+normal_y*ss_diag)*delta
            ystress_tensor = d*(normal_x*ss_diag+normal_y*ss_22)*delta

            # delta = self.composite_body.bodies[0].phi(d)*self.eps
            # xstress_tensor = normal_x*(stress[0][0]+stress[0][1])*delta
            # ystress_tensor = normal_y*(stress[0][1]+stress[1][1])*delta
            self.friction_force_x[i] = self.visc*torch.trapz(torch.trapz(xstress_tensor, dx=self.dy), dx=self.dx)
            self.friction_force_y[i] = self.visc*torch.trapz(torch.trapz(ystress_tensor, dx=self.dy), dx=self.dx)

            p_force_par=p_ext-d*self.normal_derivative(p_ext,self.normal_x,self.normal_y)

            self.pressure_force_x[i] = self.visc*torch.trapz(torch.trapz(p_force_par*normal_x*delta, dx=self.dy), dx=self.dx)
            self.pressure_force_y[i] = self.visc*torch.trapz(torch.trapz(p_force_par*normal_y*delta, dx=self.dy), dx=self.dx)


        # (u,v,p) = self.solver_iteration(u,v,p)
        (u,v,p) = self.solve_heun(u,v,p)

        # ============ plotting/saving ==========
        if not iteration % self.save_every:
            if self.save_frames:


                # copy u from device to host
                X,Y = self.X.cpu(), self.Y.cpu()
                curl = self.vorticity(u,v).cpu()
                divergence = self.divergence(u,v).cpu()
                d_min = self.composite_body.sdf_val.cpu()
                # d_min = self.d_min.cpu()
                pressure = p.cpu()
                curl_body = (self.vorticity(self.body_u,self.body_v)).cpu()
                vec_x=self.body_u.cpu()
                vec_y=self.body_v.cpu()


                # plotting.plot2d_imshow(X,Y,(self.vorticity(u,v)/(self.composite_body.bodies[0].L)).cpu(),d_min,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax)

                plotting.plot2d_imshow_composite_quiver(X,Y,curl,self.sdf_properties,0*vec_x,0*vec_y,self.extent,iteration,self.save_path,"curl",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)
                plotting.plot2d_imshow_composite_quiver(X,Y,curl_body,self.sdf_properties,vec_x,vec_y,self.extent,iteration,self.save_path,"curlbody",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                # plotting.plot2d_imshow_simple((self.mu1_all/self.eps).cpu(),self.extent,iteration,self.save_path,"mu1",-0.2,0.2)
                # plotting.plot2d_imshow_simple((self.mu0_all).cpu(),self.extent,iteration,self.save_path,"mu0",0,1)
                plotting.plot2d_imshow_composite(X,Y,u.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"u",None, None)
                plotting.plot2d_imshow_composite(X,Y,v.cpu(),self.sdf_properties,self.extent,iteration,self.save_path,"v",None, None)


                plotting.plot2d_imshow_composite(X,Y,vec_x,self.sdf_properties,self.extent,iteration,self.save_path,"bodyu",None, None)
                plotting.plot2d_imshow_composite(X,Y,vec_y,self.sdf_properties,self.extent,iteration,self.save_path,"bodyv",None, None)

                plotting.plot2d_imshow(X,Y,divergence,d_min,self.extent,iteration,self.save_path,"divergence",None, None)

                # plotting.plot2d_imshow(X,Y,pressure,d_min,self.extent,iteration,self.save_path,"pressure",None, None)
                plotting.plot2d_imshow_only(pressure,self.extent,iteration,self.save_path,"pressure",None, None)


                plotting.plot2d_imshow_only((self.m_m0_all).cpu(),self.extent,iteration,self.save_path,"tmp",None, None)

                plotting.plot2d_imshow_quiver(X,Y,d_min,d_min,vec_x,vec_y,self.extent,iteration,self.save_path,"sdf",None,None,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                plotting.plot2d_imshow_quiver(X,Y,curl,d_min,vec_x,vec_y,self.extent,iteration,self.save_path,"curluv",self.vmin,self.vmax,subsample_n = self.n_quiver_spacing, scale=self.save_every*self.dt)

                plotting.plot2d_imshow_only(
                    xstress_tensor.cpu(),
                    self.extent,iteration,self.save_path,"xstress_tensor", None, None)


            if self.save_uv:
                uv_path = f'{self.save_path}/uv_field'
                os.makedirs(uv_path, exist_ok=True)
                np.save(f'{uv_path}/u_{iteration}',u.cpu().numpy())
                np.save(f'{uv_path}/v_{iteration}',v.cpu().numpy())

        # # update sdf_properties
        # self.sdf_properties = sdf_properties

        return (u,v,p)

    def run_from_initial(self, u0, v0):
        u=u0
        v=v0
        p=torch.zeros_like(u)
        for iteration in tqdm(range(self.nt)):
            t=iteration*self.dt
            (u,v,p) = self.step_(u, v, p, iteration, t)

    def run_sim(self):
        u=self.u0
        v=self.v0
        p=self.p0
        for iteration in tqdm(range(self.starting_iteration, self.nt)):
            t=iteration*self.dt
            (u,v,p) = self.step_(u, v, p, iteration, t)



if __name__ == "__main__":

    solver = FluidSolver()
    solver.run_sim()


