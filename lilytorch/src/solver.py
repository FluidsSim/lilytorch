
from pytorch_interpolation import RegularGridInterpolator
from lilytorch.src.adv_diff import AdvDiffSolver
from lilytorch.src.poisson_mult import PoissonSolver
from lilytorch.src.poisson_fft import PoissonSolverFFT
# from lilytorch.src.poisson_petsc import PoissonSolverPETSc
from lilytorch.src.body import body_from_yaml, _StaggeredGrids
from lilytorch.src import operations as ops
from lilytorch.src import plotting
# from lilytorch.util.rw import save_object
from lilytorch.util.yaml_operations import pyobject2yaml

import torch
from tqdm import tqdm
import datetime
import os
import numpy as np
# from spreading_operator import spreading_operator_python_parallel, spreading_operator_python_parallel_out
# from spreading_operator import interpolation_operator, spreading_operator_python, interp_operator_python_parallel

class FluidSolver:
    """
    Solver class
    """

    # ---- thin wrappers around standalone functions in operations.py ----
    def compute_dpdx(self, p):  return ops.compute_dpdx(p, self.h)
    def compute_dpdy(self, p):  return ops.compute_dpdy(p, self.h)
    def compute_dpdz(self, p):  return ops.compute_dpdz(p, self.h)
    def gradient(self, var):    return ops.gradient(var, self.h, self.ndim)
    def divergence(self, u, v, w=None):
        return ops.divergence(u, v, self.dx, self.dy, w=w, dz=getattr(self, 'dz', None))
    def normal_derivative(self, var, normal_x, normal_y, normal_z=None):
        return ops.normal_derivative(var, self.h, self.ndim, normal_x, normal_y, normal_z)
    def vorticity(self, u, v, w=None):
        return ops.vorticity(u, v, self.h, self.ndim, w=w)
    def vorticity_components(self, u, v, w):
        return ops.vorticity_components(u, v, w, self.h)

    def __init__(self, pars, dtype=torch.float32, costum_update=None, compute_forces=True):
        """
        BDIM2 solver for fluid structure interaction
        """
        solver    = pars["solver"]
        bcs       = pars["boundary_conditions"]
        output    = pars["output"]
        body_pars = pars["body"]

        use_gpu = solver["use_gpu"]
        if torch.cuda.is_available() and use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
            self.device = torch.device("cuda")
        else:
            print("Using the CPU.")
            self.device = torch.device("cpu")
            torch.set_num_threads(solver["nthreads"])

        self.dtype = dtype
        self.nx    = solver["Nx"]+2
        self.ny    = solver["Ny"]+2

        self.xmin  = solver["xmin"]
        self.xmax  = solver["xmax"]
        self.ymin  = solver["ymin"]
        self.ymax  = solver["ymax"]

        self.dx=(self.xmax-self.xmin)/(self.nx-2)
        self.dy=(self.ymax-self.ymin)/(self.ny-2)

        assert abs(float(self.dx-self.dy)) < 1e-10, "Grid spacing in x = {} and y = {} must be equal".format(self.dx, self.dy)
        self.h = self.dx

        self.x = torch.arange(self.xmin-self.h/2, self.xmax+self.h, self.h, device=self.device, dtype=self.dtype)
        self.y = torch.arange(self.ymin-self.h/2, self.ymax+self.h, self.h, device=self.device, dtype=self.dtype)

        # ---- 3D detection ----
        if "Nz" in solver:
            self.ndim = 3
            self.nz   = solver["Nz"] + 2
            self.zmin = solver["zmin"]
            self.zmax = solver["zmax"]
            self.dz   = (self.zmax - self.zmin) / (self.nz - 2)
            assert abs(float(self.dx - self.dz)) < 1e-10, \
                "Grid spacing in x = {} and z = {} must be equal".format(self.dx, self.dz)
            self.z = torch.arange(self.zmin - self.h/2, self.zmax + self.h, self.h,
                                  device=self.device, dtype=self.dtype)
            self.grid_shape = (self.nx, self.ny, self.nz)
        else:
            self.ndim = 2
            self.z = None
            self.grid_shape = (self.nx, self.ny)

        self.h2    = self.h**2
        self.h3    = self.h**3
        self.dt    = torch.tensor(solver["dt"], device=self.device, dtype=self.dtype)
        self.dt_np = self.dt.cpu().numpy()

        self.nt   = solver["nt"]
        self.nu   = torch.tensor(solver["nu"], device=self.device, dtype=self.dtype)   # kinematic viscosity

        self.rho  = torch.tensor(solver["rho"], device=self.device, dtype=self.dtype)  # density
        self.visc = self.nu*self.rho                                                   # dynamic viscosity

        self.eps  = 2*self.h

        # self.re   = 158
        # print("Reynolds number: ", self.re)

        self.p_coeff = self.dt/self.rho

        self.p_fc = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.p_cc = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        self.brinkmann_k = 1.0e5

        self.starting_iteration      = solver.get("starting_iteration", 0)
        self.starting_iteration_path = solver.get("starting_iteration_path", None)
        self.starting_time           = self.starting_iteration * self.dt

        self.perturbation_amplitude  = solver.get("perturbation_amplitude", 0.0)

        print("Setting dt={}s, dx={}".format(self.dt, self.h))

        # ============= convection solver =============
        adv_diff_kwargs = dict(
            BC_type_u=bcs["BC_type_u"], BC_values_u=bcs["BC_values_u"],
            BC_type_v=bcs["BC_type_v"], BC_values_v=bcs["BC_values_v"],
            method=solver["convection_method"],
        )
        if self.ndim == 3:
            adv_diff_kwargs.update(
                z=self.z,
                BC_type_w=bcs.get("BC_type_w", ("D", "D")),
                BC_values_w=bcs.get("BC_values_w", (0.0, 0.0)),
            )
        self.adv_diff_solver = AdvDiffSolver(
            self.device, self.dt, self.x, self.y, self.nu,
            **adv_diff_kwargs,
        )

        # =============  poisson solver =============
        self.poisson_method = solver.get("poisson_method", "multigrid")
        assert self.poisson_method in ("multigrid", "fft"), \
            f"Unknown poisson_method '{self.poisson_method}'. Choose 'multigrid' or 'fft'."
        print(f"Poisson solver: {self.poisson_method}")

        self.poisson_solver  = PoissonSolver(
            self.dtype,
            self.device,
            self.h,
            tol         = solver["poisson_tol"],
            max_cycles  = solver["poisson_max_mgcg_cycles"],
            max_vcycles = solver["poisson_max_cycles"],
            nsmoothing  = solver["poisson_nsmoothing"],
            w           = solver["jacobi_weight"],
            verbose     = solver["poisson_verbose"]
        )

        # Only build the (expensive) FFT solver when it will actually be
        # used.  Gfft + U buffer cost ~834 MB on a 512×128×128 grid.
        if self.poisson_method == "fft":
            if self.ndim == 2:
                self.poisson_solverFFT  = PoissonSolverFFT(
                    self.x,
                    self.y,
                    filename = solver["poisson_folder"],
                    bc_type  = "free"
                )
            else:
                self.poisson_solverFFT  = PoissonSolverFFT(
                    self.x,
                    self.y,
                    z        = self.z,
                    filename = solver["poisson_folder"],
                    bc_type  = "free"
                )
        else:
            self.poisson_solverFFT = None

        # self.poisson_solverPETSc  = PoissonSolverPETSc(
        #     self.nx,
        #     self.ny,
        #     self.x,
        #     self.y,
        #     device=self.device,
        #     dtype=self.dtype
        # )

        # ---- staggered grids (shared by all bodies) ------------------
        self.grids = _StaggeredGrids(self.x, self.y, self.z)

        self.composite_body = body_from_yaml(
            self.device,
            self.x, self.y,
            body_pars,
            z             = self.z,
            eps           = self.eps,
            costum_update = costum_update,
            starting_time = self.starting_time,
            grids         = self.grids,
        )


        self.X, self.Y = self.grids.X, self.grids.Y
        if self.ndim == 3:
            self.Z_grid = self.grids.Z_grid


        # interpolators (2D only — used by force computation)
        if self.ndim == 2:
            self.force_x_interp = RegularGridInterpolator(
                (self.grids.x_stag, self.y),
                torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
                method=1,
                fill_value=None
            )
            self.force_y_interp = RegularGridInterpolator(
                (self.x, self.grids.y_stag),
                torch.zeros_like(self.Y, device=self.device, dtype=self.dtype),
                method=1,
                fill_value=None
            )

            self.interp_utility = RegularGridInterpolator(
                (self.x,self.y),
                torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
                method=1,
                fill_value=None
            )

        self.n_bodies = len(self.composite_body.bodies)

        # low dimensional utilities
        n_force_comp = self.ndim                # 2 or 3
        n_torque_comp = 1 if self.ndim == 2 else 3  # scalar in 2D, vector in 3D
        self.friction_force_lin_x = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_lin_y = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.friction_force_ang_z = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.force_x_int          = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.force_y_int          = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_x     = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_y     = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.pressure_force_ang_z = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        if self.ndim == 3:
            self.friction_force_lin_z  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.force_z_int           = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.pressure_force_z      = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            # torque is a 3-vector in 3D
            self.friction_force_ang_x  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.friction_force_ang_y  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.pressure_force_ang_x  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
            self.pressure_force_ang_y  = torch.zeros(self.n_bodies,device=self.device,dtype=self.dtype)
        self.viscous_drag_record  = torch.zeros((self.n_bodies,n_force_comp,self.nt),device=self.device,dtype=self.dtype)
        self.pressure_drag_record = torch.zeros((self.n_bodies,n_force_comp,self.nt),device=self.device,dtype=self.dtype)

        # high dimensional utilities
        self.body_u = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.body_v = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        if self.ndim == 3:
            self.body_w = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        # self.zc = torch.ones((1,self.ny),device=self.device,dtype=self.dtype)
        # self.zr = torch.ones((self.nx,1),device=self.device,dtype=self.dtype)

        self.xstress_tensor = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.ystress_tensor = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        if self.ndim == 3:
            self.zstress_tensor = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)


        self.div = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        # self.force_ctr_x = [torch.zeros((body.cnt.shape[1])) for body in self.composite_body.bodies]
        # self.force_ctr_y = [torch.zeros((body.cnt.shape[1])) for body in self.composite_body.bodies]

        # self.force_ctr_x = torch.zeros((self.n_bodies,self.nx,self.ny),device=self.device,dtype=self.dtype)
        # self.force_ctr_y = torch.zeros((self.n_bodies,self.nx,self.ny),device=self.device,dtype=self.dtype)

        self.force_f_ibm_x = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.force_f_ibm_y = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        self.out_f_util = torch.zeros(int(torch.tensor(self.grid_shape).prod()), device=self.device, dtype=self.dtype)


        # self.ds = self.composite_body.bodies[0].ds


        # # Parameters for the Gaussian kernel
        # kernel_dx=0.005
        # self.kernel_size = round(kernel_dx/self.h)  # Example kernel size
        # self.kernel_size += 1-self.kernel_size%2
        # sigma = 10  # Standard deviation for Gaussian
        # self.kernel = self.composite_body.gaussian_kernel(self.kernel_size, sigma).view(1, 1, self.kernel_size, self.kernel_size).to(self.device)

          # ===== set initial conditions =====
        self.set_initial_conditions()

          # ===== plotting parameters =====
        self.extent = (
            self.xmin-self.h/2, self.xmax+self.h/2,
            self.ymin-self.h/2, self.ymax+self.h/2
        )
        self.extent_vstag = (
            self.xmin-self.h/2, self.xmax+self.h/2,
            self.ymin-self.h, self.ymax
        )
        self.extent_ustag = (
            self.xmin-self.h, self.xmax,
            self.ymin-self.h/2, self.ymax+self.h/2
        )
        self.extent_curl = (
            self.xmin-self.h, self.xmax,
            self.ymin-self.h, self.ymax
        )



        self.compute_forces = compute_forces
        self.skip_projection = solver.get("skip_projection", False)
        if self.skip_projection:
            print(">>> PROJECTION DISABLED — running without pressure correction <<<")

          # ===== create folder for frames' storage ====
        self.save_frames      = output["save_frames"]
        self.save_every       = output["save_every"]
        self.save_uv          = output["save_uv"]
        self.save_vtk         = output.get("save_vtk", False)
        # vmin/vmax: a number → fixed colour limits; "auto" → auto-scale per field
        _vmin = output["vmin"]
        _vmax = output["vmax"]
        self.vmin = None if _vmin == "auto" else _vmin
        self.vmax = None if _vmax == "auto" else _vmax
        self.n_quiver_spacing = 2**3

        if self.save_frames or self.save_uv:
            path = output["save_path"]
            if "existing_folder" in output:
                results_folder = output["existing_folder"]
            else:
                today          = datetime.datetime.now()
                todaystr       = today.isoformat()
                results_folder = f'{path}{todaystr}'
            os.makedirs(results_folder, exist_ok=True)

            print(f"Frames will be saved in folder: {results_folder}/")

            self.save_path = results_folder+"/"

              # Add save path to the parameters
            pars["output"]["existing_folder"] = results_folder

              # Save body signal (if available)
            if getattr(self.composite_body, 'save_signal', None):
                self.composite_body.save_signal(results_folder)

              # Save the parameters as a yaml file
            pyobject2yaml(
                filename = self.save_path+"parameters.yaml",
                pyobject = pars,
            )

    def inside(self, x):
        """
        Return True if all elements in x are inside the domain
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
        p0 = torch.zeros(self.grid_shape, device=self.device)

          # Verify shape
        assert u0.shape == tuple(self.grid_shape), f"u0 shape: {u0.shape} != {self.grid_shape}"
        assert v0.shape == tuple(self.grid_shape), f"v0 shape: {v0.shape} != {self.grid_shape}"

          # Loaded
        self.u0, self.v0, self.p0 = u0, v0, p0
        if self.ndim == 3:
            w0_path = f'{self.starting_iteration_path}/uv_field/w_{self.starting_iteration}.npy'
            if os.path.exists(w0_path):
                self.w0 = torch.tensor(np.load(w0_path)).to(self.device)
            else:
                self.w0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

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
            self.u0 = self.adv_diff_solver.BC_values_u[0]*torch.ones(self.grid_shape, device=self.device, dtype=self.dtype)
        else:
            self.u0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.v0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        self.p0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)
        if self.ndim == 3:
            self.w0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

        # Add symmetry-breaking perturbation to v (cross-stream) to trigger
        # vortex shedding instead of relying on floating-point roundoff.
        if self.perturbation_amplitude > 0:
            print(f"Adding random perturbation to v0 with amplitude {self.perturbation_amplitude}")
            self.v0 += self.perturbation_amplitude * (2 * torch.rand(self.grid_shape, device=self.device, dtype=self.dtype) - 1)

        # Mask the initial velocity inside the body using the smooth BDIM
        # mask (mu0 = 1 outside, 0 inside).  Without this, the uniform
        # freestream fills the body interior and the first BDIM step creates
        # an impulsive discontinuity that produces a spurious pressure spike
        # and artificial vorticity at t=0.
        if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
            mu0, _ = self.composite_body.mu_funcs(self.composite_body.sdf_val)
            self.u0 = self.u0 * mu0
            self.v0 = self.v0 * mu0
            if self.ndim == 3:
                self.w0 = self.w0 * mu0


    def forces_method1(self, u, v, p, iteration):


        # # ======= compute stress tensor at CC ======
        # u_cc = torch.zeros_like(u)
        # u_cc[:-1,:] = 0.5*(u[1:,:]+u[:-1,:])
        # u_cc[-1,:] = u_cc[-2,:]
        # v_cc = torch.zeros_like(v)
        # v_cc[:,:-1] = 0.5*(v[:,1:]+v[:,:-1])
        # v_cc[:,-1] = v_cc[:,-2]

        # mult=1 #self.visc
        # dudx, dudy = self.compute_dpdx(u_cc), self.compute_dpdy(u_cc)
        # dvdx, dvdy = self.compute_dpdx(v_cc), self.compute_dpdy(v_cc)
        # ss_11 = 2*self.normal_x*dudx
        # ss_12 = self.normal_y*(dudy+dvdx)
        # ss_21 = self.normal_x*(dudy+dvdx)
        # ss_22 = 2*self.normal_y*dvdy
        # self.xstress_tensor = mult*(ss_11+ss_12)
        # self.ystress_tensor = mult*(ss_21+ss_22)

        # pforce_x = -p*self.normal_x
        # pforce_y = -p*self.normal_y


        # dudx, dudy = self.compute_dpdx(u), self.compute_dpdy(u)
        # dvdx, dvdy = self.compute_dpdx(v), self.compute_dpdy(v)
        # ss_diag = (self.normal_x_u*dudy+self.normal_y_v*dvdx)
        # ss_11 = 2*self.normal_x_u*dudx
        # ss_12 = dudy*self.normal_y_u+dvdx*self.normal_x_v
        # ss_22 = 2*self.normal_y_v*dvdy
        # self.xstress_tensor = self.visc*(ss_11+ss_diag)
        # self.ystress_tensor = self.visc*(ss_diag+ss_22)

        # pforce_x = -p*self.normal_x
        # pforce_y = -p*self.normal_y


        # self.force_f_ibm_x[:]=0
        # self.force_f_ibm_y[:]=0


        # ======= compute stress tensor at FC ======
        self.force_y_interp.F = v
        v_xstag = self.force_y_interp(self.grids.Xu_stag, self.grids.Yu_stag)
        self.force_x_interp.F = u
        u_ystag = self.force_x_interp(self.grids.Xv_stag, self.grids.Yv_stag)

        ss_11 = 2*self.compute_dpdx(u)*self.normal_x_u
        ss_12 = (self.compute_dpdy(u)+self.compute_dpdx(v_xstag))*self.normal_y_u
        ss_21 = (self.compute_dpdy(u_ystag)+self.compute_dpdx(v))*self.normal_x_v
        ss_22 = 2*self.compute_dpdy(v)*self.normal_y_v

        # mult=1/self.re
        mult=self.nu*self.rho
        self.xstress_tensor = (ss_11+ss_12)*mult
        self.ystress_tensor = (ss_21+ss_22)*mult

        # u_mask = torch.where(u==0,1,u/torch.abs(u))
        # v_mask = torch.where(v==0,1,v/torch.abs(v))

        # self.xstress_tensor = (ss_11+ss_12)*mult*u_mask
        # self.ystress_tensor = (ss_21+ss_22)*mult*v_mask


        # # ======= compute pressure at staggered locations ======
        # self.interp_utility.F = -p
        # pforce_x = self.interp_utility(self.composite_body.Xu_stag, self.composite_body.Yu_stag)*self.normal_x_u
        # pforce_y = self.interp_utility(self.composite_body.Xv_stag, self.composite_body.Yv_stag)*self.normal_y_v

        self.pforce_x = -p*self.normal_x
        self.pforce_y = -p*self.normal_y


        for i, body in enumerate(self.composite_body.bodies[:]):

                  # method 1
                  mask       = body.mask
                  curv_coord = body.curv_coord

                  moment_arm = [
                      body.cnt_update[0]-body.com_pos[0],
                      body.cnt_update[1]-body.com_pos[1],
                  ]

                  self.force_x_interp.F = self.xstress_tensor
                  self.force_y_interp.F = self.ystress_tensor

                  f_x = self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                  f_y = self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

                  self.friction_force_lin_x[i] = torch.sum(f_x[mask])*body.ds
                  self.friction_force_lin_y[i] = torch.sum(f_y[mask])*body.ds
                  self.friction_force_ang_z[i] = torch.sum(
                      ops.cross_product_2d(
                          moment_arm[0][mask],
                          moment_arm[1][mask],
                          f_x[mask],
                          f_y[mask]
                          )
                      )*body.ds


                  # self.friction_force_lin_x[i] = torch.trapz(f_x[mask],curv_coord)
                  # self.friction_force_lin_y[i] = torch.trapz(f_y[mask],curv_coord)
                  # self.friction_force_ang_z[i] = torch.trapz(
                  #   ops.cross_product_2d(
                  #       moment_arm[0][mask],
                  #       moment_arm[1][mask],
                  #       f_x[mask],
                  #       f_y[mask]
                  #     ),
                  #   curv_coord
                  # )

                  # self.force_x_interp.F = pforce_x
                  # self.force_y_interp.F = pforce_y

                  # f_x = self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
                  # f_y = self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

                  # self.pressure_force_x[i] = torch.trapz(f_x[mask],curv_coord)
                  # self.pressure_force_y[i] = torch.trapz(f_y[mask],curv_coord)
                  # self.pressure_force_ang_z[i] = torch.trapz(
                  #       ops.cross_product_2d(
                  #           moment_arm[0][mask],
                  #           moment_arm[1][mask],
                  #           f_x[mask],
                  #           f_y[mask]
                  #           ), curv_coord
                  #       )



                  moment_arm = [
                      body.cnt_update[0]-body.com_pos[0],
                      body.cnt_update[1]-body.com_pos[1],
                  ]


                  self.interp_utility.F = self.pforce_x
                  f_x = self.interp_utility(body.cnt_update[0], body.cnt_update[1])
                  self.interp_utility.F = self.pforce_y
                  f_y = self.interp_utility(body.cnt_update[0], body.cnt_update[1])

                  # self.pressure_force_x[i] = torch.trapz(f_x[mask],curv_coord)
                  # self.pressure_force_y[i] = torch.trapz(f_y[mask],curv_coord)
                  # self.pressure_force_ang_z[i] = torch.trapz(
                  #       ops.cross_product_2d(
                  #           moment_arm[0][mask],
                  #           moment_arm[1][mask],
                  #           f_x[mask],
                  #           f_y[mask]
                  #           ), curv_coord
                  #       )


                  self.pressure_force_x[i] = torch.sum(f_x[mask])*body.ds
                  self.pressure_force_y[i] = torch.sum(f_y[mask])*body.ds
                  self.pressure_force_ang_z[i] = torch.sum(
                      ops.cross_product_2d(
                          moment_arm[0][mask],
                          moment_arm[1][mask],
                          f_x[mask],
                          f_y[mask]
                          )
                      )*body.ds


                  # # # # # # # self.friction_force_lin_x[i] = self.brinkmann_k*((u-self.composite_body.body_u)*self.m_m0_all_u).sum()*self.h2
                  # # # # # # # self.friction_force_lin_y[i] = self.brinkmann_k*((v-self.composite_body.body_v)*self.m_m0_all_v).sum()*self.h2

                  # # # # method of integration over the grid

                  # # # moment_arm = [
                  # # #     self.composite_body.Xu_stag-body.com_pos[0],
                  # # #     self.composite_body.Yv_stag-body.com_pos[1],
                  # # # ]

                  # # # # method 2
                  # # # d_u=self.composite_body.sdf_val_u-2*self.dx
                  # # # d_v=self.composite_body.sdf_val_v-2*self.dx
                  # # # mu_0_all_u_friction, _ = self.composite_body.mu_funcs(d_u)
                  # # # mu_0_all_v_friction, _ = self.composite_body.mu_funcs(d_v)
                  # # # (_, normal_x_u, normal_y_u, _) = self.composite_body.compute_sdf_properties(d_u)
                  # # # (_, normal_x_v, normal_y_v, _) = self.composite_body.compute_sdf_properties(d_v)


                  # # # mult_x = self.normal_derivative(mu_0_all_u_friction, normal_x_u, normal_y_u)
                  # # # mult_y = self.normal_derivative(mu_0_all_v_friction, normal_x_v, normal_y_v)


                  # # # fforcex_body_i = self.xstress_tensor*mult_x
                  # # # fforcey_body_i = self.ystress_tensor*mult_y


                  # # # self.friction_force_lin_x[i] = fforcex_body_i.sum()*self.h2
                  # # # self.friction_force_lin_y[i] = fforcey_body_i.sum()*self.h2
                  # # # self.friction_force_ang_z[i] = ops.cross_product_2d(
                  # # #         moment_arm[0],
                  # # #         moment_arm[1],
                  # # #         fforcex_body_i,
                  # # #         fforcey_body_i
                  # # #         ).sum()*self.h2



                  # # # # print(self.friction_force_lin_x*2/(1000*0.00275**2*0.2))

                  # # # mult_x = self.normal_derivative(self.mu0_all_u, normal_x_u, normal_y_u)
                  # # # mult_y = self.normal_derivative(self.mu0_all_v, normal_x_v, normal_y_v)

                  # # # pforcex_body_i = pforce_x*mult_x
                  # # # pforcey_body_i = pforce_y*mult_y


                  # # # # if iteration==100:
                  # # # #     from IPython import embed; embed()


                  # # # # import matplotlib.pyplot as plt
                  # # # # import matplotlib.cm as cm

                  # # # # plt.contourf(
                  # # # #     self.composite_body.Xu_stag.cpu(),
                  # # # #     self.composite_body.Yu_stag.cpu(),
                  # # # #     pforcey_body_i,
                  # # # #     levels=50,
                  # # # # )
                  # # # # plt.colorbar()
                  # # # # plt.show()



                  # # # self.pressure_force_x[i] = pforcex_body_i.sum()*self.h2
                  # # # self.pressure_force_y[i] = pforcey_body_i.sum()*self.h2
                  # # # self.pressure_force_ang_z[i] = ops.cross_product_2d(
                  # # #           moment_arm[0],
                  # # #           moment_arm[1],
                  # # #           pforcex_body_i,
                  # # #           pforcey_body_i
                  # # #           ).sum()*self.h2


                  self.viscous_drag_record[i,0,iteration] = self.friction_force_lin_x[i]
                  self.viscous_drag_record[i,1,iteration] = self.friction_force_lin_y[i]

                  self.pressure_drag_record[i,0,iteration] = self.pressure_force_x[i]
                  self.pressure_drag_record[i,1,iteration] = self.pressure_force_y[i]



    def forces_method2(self, u, v, p, iteration):

        # # ======= compute stress tensor at CC ======
        # u_cc = torch.zeros_like(u)
        # u_cc[:-1,:] = 0.5*(u[1:,:]+u[:-1,:])
        # u_cc[-1,:] = u_cc[-2,:]
        # v_cc = torch.zeros_like(v)
        # v_cc[:,:-1] = 0.5*(v[:,1:]+v[:,:-1])
        # v_cc[:,-1] = v_cc[:,-2]

        # mult=self.nu*self.rho # rho might change in time
        # dudx, dudy = self.compute_dpdx(u_cc), self.compute_dpdy(u_cc)
        # dvdx, dvdy = self.compute_dpdx(v_cc), self.compute_dpdy(v_cc)
        # ss_11 = 2*self.normal_x*dudx
        # ss_12 = self.normal_y*(dudy+dvdx)
        # ss_21 = self.normal_x*(dudy+dvdx)
        # ss_22 = 2*self.normal_y*dvdy
        # self.xstress_tensor = mult*(ss_11+ss_12)
        # self.ystress_tensor = mult*(ss_21+ss_22)

        # pforce_x = -p*self.normal_x
        # pforce_y = -p*self.normal_y


        # ======= compute stress tensor at FC ======
        self.force_y_interp.F = v
        v_xstag = self.force_y_interp(self.grids.Xu_stag, self.grids.Yu_stag)
        self.force_x_interp.F = u
        u_ystag = self.force_x_interp(self.grids.Xv_stag, self.grids.Yv_stag)

        ss_11 = 2*self.compute_dpdx(u)*self.normal_x_u
        ss_12 = (self.compute_dpdy(u)+self.compute_dpdx(v_xstag))*self.normal_y_u
        ss_21 = (self.compute_dpdy(u_ystag)+self.compute_dpdx(v))*self.normal_x_v
        ss_22 = 2*self.compute_dpdy(v)*self.normal_y_v

        mult=self.nu*self.rho
        self.xstress_tensor = (ss_11+ss_12)*mult
        self.ystress_tensor = (ss_21+ss_22)*mult



        # # ======= compute stress tensor at FC ======
        # self.force_y_interp.F = v
        # v_xstag = self.force_y_interp(self.composite_body.Xu_stag, self.composite_body.Yu_stag)
        # self.force_x_interp.F = u
        # u_ystag = self.force_x_interp(self.composite_body.Xv_stag, self.composite_body.Yv_stag)

        # ss_11 = 2*self.compute_dpdx(u)*self.normal_x_u
        # ss_12 = (self.compute_dpdy(u)+self.compute_dpdx(v_xstag))*self.normal_y_u
        # ss_21 = (self.compute_dpdy(u_ystag)+self.compute_dpdx(v))*self.normal_x_v
        # ss_22 = 2*self.compute_dpdy(v)*self.normal_y_v

        # mult=self.nu*self.rho
        # self.xstress_tensor = (ss_11+ss_12)*mult
        # self.ystress_tensor = (ss_21+ss_22)*mult

        # # ======= compute pressure at staggered locations ======
        # self.interp_utility.F = -p
        # pforce_x = self.interp_utility(self.composite_body.Xu_stag, self.composite_body.Yu_stag)*self.normal_x_u
        # pforce_y = self.interp_utility(self.composite_body.Xv_stag, self.composite_body.Yv_stag)*self.normal_y_v

        p_outer=torch.where(self.composite_body.sdf_val<0,0,p)
        self.pforce_x = -p_outer*self.normal_x
        self.pforce_y = -p_outer*self.normal_y


        for i, body in enumerate(self.composite_body.bodies[:]):

            moment_arm = [
                self.grids.Xv_stag-body.com_pos[0],
                self.grids.Yu_stag-body.com_pos[1],
            ]


            # fforcex_body_i = self.xstress_tensor*delta
            # fforcey_body_i = self.ystress_tensor*delta

            # self.friction_force_lin_x[i] = (fforcex_body_i).sum()*self.h2
            # self.friction_force_lin_y[i] = (fforcey_body_i).sum()*self.h2
            # self.friction_force_ang_z[i] = ops.cross_product_2d(
            #         moment_arm[0],
            #         moment_arm[1],
            #         fforcex_body_i,
            #         fforcey_body_i
            #         ).sum()*self.h2

            sdf_val_u = body.sdf_u
            sdf_val_v = body.sdf_v

            delta_u = body.phi(sdf_val_u-self.eps)
            delta_v = body.phi(sdf_val_v-self.eps)

            # delta_u = body.phi(sdf_val_u-self.eps)
            # delta_v = body.phi(sdf_val_v-self.eps)

            fforcex_body_i = self.xstress_tensor*delta_u
            fforcey_body_i = self.ystress_tensor*delta_v

            # # method 1
            # mask       = body.mask
            # curv_coord = body.curv_coord

            # moment_arm = [
            #     body.cnt_update[0]-body.com_pos[0],
            #     body.cnt_update[1]-body.com_pos[1],
            # ]

            # self.force_x_interp.F = self.xstress_tensor
            # self.force_y_interp.F = self.ystress_tensor

            # f_x = self.force_x_interp(body.cnt_update[0], body.cnt_update[1])
            # f_y = self.force_y_interp(body.cnt_update[0], body.cnt_update[1])

            # self.friction_force_lin_x[i] = torch.trapz(f_x[mask],curv_coord)
            # self.friction_force_lin_y[i] = torch.trapz(f_y[mask],curv_coord)
            # self.friction_force_ang_z[i] = torch.trapz(
            #   ops.cross_product_2d(
            #       moment_arm[0][mask],
            #       moment_arm[1][mask],
            #       f_x[mask],
            #       f_y[mask]
            #     ),
            #   curv_coord
            # )




            # # method 2
            # d_u=self.composite_body.sdf_val_u-self.dx
            # d_v=self.composite_body.sdf_val_v-self.dx
            # mu_0_all_u_friction, _ = self.composite_body.mu_funcs(d_u)
            # mu_0_all_v_friction, _ = self.composite_body.mu_funcs(d_v)
            # (_, normal_x_u, normal_y_u, _) = self.composite_body.compute_sdf_properties(d_u)
            # (_, normal_x_v, normal_y_v, _) = self.composite_body.compute_sdf_properties(d_v)


            # mult_x = self.normal_derivative(mu_0_all_u_friction, normal_x_u, normal_y_u)
            # mult_y = self.normal_derivative(mu_0_all_v_friction, normal_x_v, normal_y_v)


            # fforcex_body_i = self.xstress_tensor*mult_x
            # fforcey_body_i = self.ystress_tensor*mult_y

            # if iteration==100:
            #     from IPython import embed; embed()




            #     import matplotlib.pyplot as plt
            #     import matplotlib.cm as cm

            #     plt.contourf(
            #         self.composite_body.Xu_stag.cpu(),
            #         self.composite_body.Yu_stag.cpu(),
            #         delta_u.cpu(),
            #         levels=300,
            #     )
            #     plt.colorbar()
            #     plt.show()





            #     plt.scatter(body.cnt_update[0],body.cnt_update[1],c=f_x[mask])
            #     plt.colorbar()


            # sdf_val_u = self.composite_body.sdf_vals[i]

            self.friction_force_lin_x[i] = fforcex_body_i.sum()*self.h2
            self.friction_force_lin_y[i] = fforcey_body_i.sum()*self.h2
            self.friction_force_ang_z[i] = ops.cross_product_2d(
                    moment_arm[0],
                    moment_arm[1],
                    fforcex_body_i,
                    fforcey_body_i
                    ).sum()*self.h2


            moment_arm = [
                self.grids.X-body.com_pos[0],
                self.grids.Y-body.com_pos[1],
            ]

            delta = body.phi(body.sdf_val)
            self.pforcex_body_i = self.pforce_x*delta
            self.pforcey_body_i = self.pforce_y*delta



            # mult = self.normal_derivative(self.m_m0_all, self.normal_x, self.normal_y)
            # pforcex_body_i = pforce_x*mult
            # pforcey_body_i = pforce_y*mult

            # if iteration==100:
            #     from IPython import embed; embed()



            # import matplotlib.pyplot as plt
            # import matplotlib.cm as cm

            # plt.contourf(
            #     self.composite_body.X.cpu(),
            #     self.composite_body.Y.cpu(),
            #     mult,
            #     levels=100,
            # )
            # plt.colorbar()
            # plt.show()


            # sdf_p = self.composite_body.sdf_val-2*self.dx
            # mu_0_all_p, _ = self.composite_body.mu_funcs(sdf_p)
            # m_mu0_all_p = 1 - mu_0_all_p
            # (_, normal_x_p, normal_y_p, _) = self.composite_body.compute_sdf_properties(sdf_p)


            # mult = -self.normal_derivative(m_mu0_all_p, normal_x_p, normal_y_p)
            # pforcex_body_i = -p*normal_x_p*mult
            # pforcey_body_i = -p*normal_y_p*mult


            # sdf_p = self.composite_body.sdf_val
            # (_, normal_x_p, normal_y_p, _) = self.composite_body.compute_sdf_properties(sdf_p)
            # delta = self.composite_body.phi(sdf_p)
            # integrand = (-p+sdf_p*self.normal_derivative(p,normal_x_p,normal_y_p))
            # self.pforcex_body_i = integrand*normal_x_p*delta
            # self.pforcey_body_i = integrand*normal_y_p*delta


            # mult = self.normal_derivative(self.m_m0_all, self.normal_x, self.normal_y)
            # pforcex_body_i = pforce_x*mult
            # pforcey_body_i = pforce_y*mult



            self.pressure_force_x[i] = (self.pforcex_body_i).sum()*self.h2
            self.pressure_force_y[i] = (self.pforcey_body_i).sum()*self.h2
            self.pressure_force_ang_z[i] = ops.cross_product_2d(
                      moment_arm[0],
                      moment_arm[1],
                      self.pforcex_body_i,
                      self.pforcey_body_i
                      ).sum()*self.h2




            # method of integration over the grid

            # moment_arm = [
            #     self.composite_body.Xu_stag-body.com_pos[0],
            #     self.composite_body.Yv_stag-body.com_pos[1],
            # ]

            # d_u=self.composite_body.sdf_val_u-2*self.dx
            # d_v=self.composite_body.sdf_val_v-2*self.dx
            # mu_0_all_u_friction, _ = self.composite_body.mu_funcs(d_u)
            # mu_0_all_v_friction, _ = self.composite_body.mu_funcs(d_v)
            # (_, normal_x_u, normal_y_u, _) = self.composite_body.compute_sdf_properties(d_u)
            # (_, normal_x_v, normal_y_v, _) = self.composite_body.compute_sdf_properties(d_v)


            # mult_x = self.normal_derivative(mu_0_all_u_friction, normal_x_u, normal_y_u)
            # mult_y = self.normal_derivative(mu_0_all_v_friction, normal_x_v, normal_y_v)


            # fforcex_body_i = self.xstress_tensor*mult_x
            # fforcey_body_i = self.ystress_tensor*mult_y


            # self.friction_force_lin_x[i] = fforcex_body_i.sum()*self.h2
            # self.friction_force_lin_y[i] = fforcey_body_i.sum()*self.h2
            # self.friction_force_ang_z[i] = ops.cross_product_2d(
            #         moment_arm[0],
            #         moment_arm[1],
            #         fforcex_body_i,
            #         fforcey_body_i
            #         ).sum()*self.h2



            # print(self.friction_force_lin_x*2/(1000*0.00275**2*0.2))

            # mult_x = self.normal_derivative(self.mu0_all_u, normal_x_u, normal_y_u)
            # mult_y = self.normal_derivative(self.mu0_all_v, normal_x_v, normal_y_v)

            # pforcex_body_i = pforce_x*mult_x
            # pforcey_body_i = pforce_y*mult_y


            # if iteration==100:
            #     from IPython import embed; embed()


            # import matplotlib.pyplot as plt
            # import matplotlib.cm as cm

            # plt.contourf(
            #     self.composite_body.Xu_stag.cpu(),
            #     self.composite_body.Yu_stag.cpu(),
            #     pforcey_body_i,
            #     levels=50,
            # )
            # plt.colorbar()
            # plt.show()



            # self.pressure_force_x[i] = pforcex_body_i.sum()*self.h2
            # self.pressure_force_y[i] = pforcey_body_i.sum()*self.h2
            # self.pressure_force_ang_z[i] = ops.cross_product_2d(
            #           moment_arm[0],
            #           moment_arm[1],
            #           pforcex_body_i,
            #           pforcey_body_i
            #           ).sum()*self.h2



            # delta_u = self.composite_body.phi(self.composite_body.sdf_val_u-self.eps)
            # delta_v = self.composite_body.phi(self.composite_body.sdf_val_v-self.eps)

            # fforcex_body_i = self.xstress_tensor*delta_u
            # fforcey_body_i = self.ystress_tensor*delta_v

            # self.friction_force_lin_x[i] = fforcex_body_i.sum()*self.h2
            # self.friction_force_lin_y[i] = fforcey_body_i.sum()*self.h2
            # self.friction_force_ang_z[i] = ops.cross_product_2d(
            #         moment_arm[0],
            #         moment_arm[1],
            #         fforcex_body_i,
            #         fforcey_body_i
            #         ).sum()*self.h2


            # delta_u = self.composite_body.phi(self.composite_body.sdf_val_u)
            # delta_v = self.composite_body.phi(self.composite_body.sdf_val_v)

            # pforcex_body_i = pforce_x*delta_u
            # pforcey_body_i = pforce_y*delta_v

            # moment_arm = [
            #     self.composite_body.X-body.com_pos[0],
            #     self.composite_body.Y-body.com_pos[1],
            # ]


            # delta = self.composite_body.phi(self.composite_body.sdf_val)

            # pforcex_body_i = -p*delta*self.normal_x
            # pforcey_body_i = -p*delta*self.normal_y


            # self.pressure_force_x[i] = (pforcex_body_i).sum()*self.h2
            # self.pressure_force_y[i] = (pforcey_body_i).sum()*self.h2
            # self.pressure_force_ang_z[i] = ops.cross_product_2d(
            #           moment_arm[0],
            #           moment_arm[1],
            #           pforcex_body_i,
            #           pforcey_body_i
            #           ).sum()*self.h2




            self.viscous_drag_record[i,0,iteration] = self.friction_force_lin_x[i]
            self.viscous_drag_record[i,1,iteration] = self.friction_force_lin_y[i]

            self.pressure_drag_record[i,0,iteration] = self.pressure_force_x[i]
            self.pressure_drag_record[i,1,iteration] = self.pressure_force_y[i]


    # ==================================================================
    # 3-D force computation  (volume-integral with smoothed delta)
    # ==================================================================
    def forces_method2_3d(self, u, v, w, p, iteration):
        """Compute viscous and pressure forces/torques on each body in 3-D.

        Uses the same smoothed-delta volume-integration approach as the 2-D
        ``forces_method2`` but extended to three dimensions:

        Viscous force:
            F_visc = ∫ (σ · n) δ_ε(d - ε) dV
        where σ_{ij} = ν ρ (∂u_i/∂x_j + ∂u_j/∂x_i) is the viscous stress.

        Pressure force:
            F_pres = -∫ p n δ_ε(d) dV

        Torques are computed about each body's centre of mass via r × f.
        """
        mult = self.nu * self.rho
        h3   = self.h3
        dpdx = self.compute_dpdx
        dpdy = self.compute_dpdy
        dpdz = self.compute_dpdz

        # ---- velocity gradients (cell-centred via torch.gradient) ----
        dudx = dpdx(u);  dudy = dpdy(u);  dudz = dpdz(u)
        dvdx = dpdx(v);  dvdy = dpdy(v);  dvdz = dpdz(v)
        dwdx = dpdx(w);  dwdy = dpdy(w);  dwdz = dpdz(w)

        # ---- viscous stress tensor (σ·n)_i  at cell centres ----------
        # σ_{ij} n_j  summed over j  for each component i
        nx, ny, nz = self.normal_x, self.normal_y, self.normal_z
        self.xstress_tensor = mult * (2*dudx*nx + (dudy + dvdx)*ny + (dudz + dwdx)*nz)
        self.ystress_tensor = mult * ((dvdx + dudy)*nx + 2*dvdy*ny + (dvdz + dwdy)*nz)
        self.zstress_tensor = mult * ((dwdx + dudz)*nx + (dwdy + dvdz)*ny + 2*dwdz*nz)

        # ---- pressure force density at CC ----------------------------
        p_outer = torch.where(self.composite_body.sdf_val < 0, 0, p)
        self.pforce_x = -p_outer * nx
        self.pforce_y = -p_outer * ny
        self.pforce_z = -p_outer * nz

        # ---- per-body integration ------------------------------------
        X   = self.composite_body.X
        Y   = self.composite_body.Y
        Z   = self.composite_body.Z_grid

        for i, body in enumerate(self.composite_body.bodies):
            # -- smoothed delta for viscous (shifted) and pressure ------
            sdf_i       = self.composite_body.sdf_vals[i]
            delta_visc  = body.phi(sdf_i - self.eps)
            delta_pres  = body.phi(sdf_i)

            # -- viscous contribution -----------------------------------
            fvisc_x = self.xstress_tensor * delta_visc
            fvisc_y = self.ystress_tensor * delta_visc
            fvisc_z = self.zstress_tensor * delta_visc

            self.friction_force_lin_x[i] = fvisc_x.sum() * h3
            self.friction_force_lin_y[i] = fvisc_y.sum() * h3
            self.friction_force_lin_z[i] = fvisc_z.sum() * h3

            # moment arm from body CoM
            rx = X - body.com_pos[0]
            ry = Y - body.com_pos[1]
            rz = Z - body.com_pos[2]

            # torque = ∫ r × f dV
            tx, ty, tz = ops.cross_product_3d(rx, ry, rz, fvisc_x, fvisc_y, fvisc_z)
            self.friction_force_ang_x[i] = tx.sum() * h3
            self.friction_force_ang_y[i] = ty.sum() * h3
            self.friction_force_ang_z[i] = tz.sum() * h3

            # -- pressure contribution ----------------------------------
            fpres_x = self.pforce_x * delta_pres
            fpres_y = self.pforce_y * delta_pres
            fpres_z = self.pforce_z * delta_pres

            self.pressure_force_x[i] = fpres_x.sum() * h3
            self.pressure_force_y[i] = fpres_y.sum() * h3
            self.pressure_force_z[i] = fpres_z.sum() * h3

            tx_p, ty_p, tz_p = ops.cross_product_3d(rx, ry, rz, fpres_x, fpres_y, fpres_z)
            self.pressure_force_ang_x[i] = tx_p.sum() * h3
            self.pressure_force_ang_y[i] = ty_p.sum() * h3
            self.pressure_force_ang_z[i] = tz_p.sum() * h3

            # -- record for post-processing -----------------------------
            self.viscous_drag_record[i, 0, iteration]  = self.friction_force_lin_x[i]
            self.viscous_drag_record[i, 1, iteration]  = self.friction_force_lin_y[i]
            self.viscous_drag_record[i, 2, iteration]  = self.friction_force_lin_z[i]

            self.pressure_drag_record[i, 0, iteration] = self.pressure_force_x[i]
            self.pressure_drag_record[i, 1, iteration] = self.pressure_force_y[i]
            self.pressure_drag_record[i, 2, iteration] = self.pressure_force_z[i]


    def integral(self, var):
        """
        Compute the integral of var
        """
        return torch.trapz(torch.trapz(var, dx=self.h), dx=self.h)


    def project(self, u, v, p, w_vel=None, w=1.0):

        # skip projection if requested (diagnostic mode)
        if self.skip_projection:
            if self.ndim == 2:
                return (u, v, p)
            else:
                return (u, v, w_vel, p)

        # for general deforming bodies
        if self.ndim == 2:
            self.div  = self.divergence(u, v)
        else:
            self.div  = self.divergence(u, v, w_vel)

        coeff = w*self.dt/self.rho

        if self.poisson_method == "fft":
            # ---- FFT solver (constant-coefficient Poisson) ----
            p = self.poisson_solverFFT.solve(self.div / coeff)
            if self.ndim == 2:
                (p_x, p_y) = self.gradient(p)
                u = u - coeff * p_x
                v = v - coeff * p_y
            else:
                (p_x, p_y, p_z) = self.gradient(p)
                u     = u - coeff * p_x
                v     = v - coeff * p_y
                w_vel = w_vel - coeff * p_z
        else:
            # ---- Multigrid solver (variable-coefficient Poisson) ----
            c = torch.ones_like(u)
            ch = coeff*self.mu0_all_u
            cv = coeff*self.mu0_all_v

            if self.ndim == 2:
                p, _    = self.poisson_solver.solve_multigrid(
                    self.div[1:-1,1:-1],
                    torch.zeros_like(u),
                    coeff*c,
                    ch = ch[1:,1:-1],
                    cv = cv[1:-1,1:],
                )
                # ====== projection step ======
                (p_x, p_y) = self.gradient(p)
                u          = u - ch * p_x
                v          = v - cv * p_y
            else:
                cw = coeff * self.mu0_all_w
                p, _ = self.poisson_solver.solve_multigrid(
                    self.div[1:-1, 1:-1, 1:-1],
                    torch.zeros_like(u),
                    coeff * c,
                    ch=ch[1:, 1:-1, 1:-1],
                    cv=cv[1:-1, 1:, 1:-1],
                    cw=cw[1:-1, 1:-1, 1:],
                )
                # ====== projection step ======
                (p_x, p_y, p_z) = self.gradient(p)
                u    = u - ch * p_x
                v    = v - cv * p_y
                w_vel = w_vel - cw * p_z

        if self.ndim == 2:
            return (u, v, p)
        else:
            return (u, v, w_vel, p)



    def solver_iteration_heun(self, u, v, p, iteration, w_vel=None):
        """
        Heun (RK2 predictor-corrector) time integration with BDIM2.

        Matches the WaterLily.jl ``mom_step!`` algorithm:
          1. Predictor: adv-diff → BDIM → project(w=1)
          2. Corrector: adv-diff at predicted vel → rebase from u^n →
             BDIM → average with predictor → project(w=0.5)
        """

        if self.ndim == 2:
            # ====== PREDICTOR ======
            (uprime, vprime) = self.adv_diff_solver.solve(u, v)

            # BDIM2 meta-equation
            uprime = (self.mu0_all_u * uprime
                      + self.m_m0_all_u * self.composite_body.body_u
                      + self.mu1_all_u * self.normal_derivative(
                          uprime - self.composite_body.body_u,
                          self.normal_x_u, self.normal_y_u))
            vprime = (self.mu0_all_v * vprime
                      + self.m_m0_all_v * self.composite_body.body_v
                      + self.mu1_all_v * self.normal_derivative(
                          vprime - self.composite_body.body_v,
                          self.normal_x_v, self.normal_y_v))

            # Save BDIM'd pre-projection velocities for averaging
            uprime_bdim = uprime.clone()
            vprime_bdim = vprime.clone()

            self.adv_diff_solver.set_BCs(uprime, vprime)
            (u1, v1, p1) = self.project(uprime, vprime, p)

            # ====== CORRECTOR ======
            # Evaluate RHS at the projected predicted velocity
            (uprime2, vprime2) = self.adv_diff_solver.solve(u1, v1)
            # adv_diff.solve returns u1 + dt*RHS(u1).
            # Heun needs u^n + dt*RHS(u1), so rebase from u^n:
            uprime2 = u + (uprime2 - u1)
            vprime2 = v + (vprime2 - v1)

            # BDIM2 meta-equation on corrector
            uprime2 = (self.mu0_all_u * uprime2
                       + self.m_m0_all_u * self.composite_body.body_u
                       + self.mu1_all_u * self.normal_derivative(
                           uprime2 - self.composite_body.body_u,
                           self.normal_x_u, self.normal_y_u))
            vprime2 = (self.mu0_all_v * vprime2
                       + self.m_m0_all_v * self.composite_body.body_v
                       + self.mu1_all_v * self.normal_derivative(
                           vprime2 - self.composite_body.body_v,
                           self.normal_x_v, self.normal_y_v))

            # Average the BDIM'd pre-projection velocities (WaterLily style)
            u_avg = 0.5 * (uprime_bdim + uprime2)
            v_avg = 0.5 * (vprime_bdim + vprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg)
            (u_out, v_out, p_out) = self.project(u_avg, v_avg, p, w=0.5)

            return (u_out, v_out, p_out)

        else:  # 3D
            # ====== PREDICTOR ======
            (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w_vel)

            # BDIM2 meta-equation for u
            uprime = (self.mu0_all_u * uprime
                      + self.m_m0_all_u * self.composite_body.body_u
                      + self.mu1_all_u * self.normal_derivative(
                          uprime - self.composite_body.body_u,
                          self.normal_x_u, self.normal_y_u, self.normal_z_u))
            # BDIM2 meta-equation for v
            vprime = (self.mu0_all_v * vprime
                      + self.m_m0_all_v * self.composite_body.body_v
                      + self.mu1_all_v * self.normal_derivative(
                          vprime - self.composite_body.body_v,
                          self.normal_x_v, self.normal_y_v, self.normal_z_v))
            # BDIM2 meta-equation for w
            wprime = (self.mu0_all_w * wprime
                      + self.m_m0_all_w * self.composite_body.body_w
                      + self.mu1_all_w * self.normal_derivative(
                          wprime - self.composite_body.body_w,
                          self.normal_x_w, self.normal_y_w, self.normal_z_w))

            # Save BDIM'd pre-projection velocities for averaging
            uprime_bdim = uprime.clone()
            vprime_bdim = vprime.clone()
            wprime_bdim = wprime.clone()

            self.adv_diff_solver.set_BCs(uprime, vprime, wprime)
            (u1, v1, w1, p1) = self.project(uprime, vprime, p, w_vel=wprime)

            # ====== CORRECTOR ======
            (uprime2, vprime2, wprime2) = self.adv_diff_solver.solve(u1, v1, w1)
            # Rebase from u^n
            uprime2 = u     + (uprime2 - u1)
            vprime2 = v     + (vprime2 - v1)
            wprime2 = w_vel + (wprime2 - w1)

            # BDIM2 meta-equation on corrector
            uprime2 = (self.mu0_all_u * uprime2
                       + self.m_m0_all_u * self.composite_body.body_u
                       + self.mu1_all_u * self.normal_derivative(
                           uprime2 - self.composite_body.body_u,
                           self.normal_x_u, self.normal_y_u, self.normal_z_u))
            vprime2 = (self.mu0_all_v * vprime2
                       + self.m_m0_all_v * self.composite_body.body_v
                       + self.mu1_all_v * self.normal_derivative(
                           vprime2 - self.composite_body.body_v,
                           self.normal_x_v, self.normal_y_v, self.normal_z_v))
            wprime2 = (self.mu0_all_w * wprime2
                       + self.m_m0_all_w * self.composite_body.body_w
                       + self.mu1_all_w * self.normal_derivative(
                           wprime2 - self.composite_body.body_w,
                           self.normal_x_w, self.normal_y_w, self.normal_z_w))

            # Average the BDIM'd pre-projection velocities
            u_avg = 0.5 * (uprime_bdim + uprime2)
            v_avg = 0.5 * (vprime_bdim + vprime2)
            w_avg = 0.5 * (wprime_bdim + wprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg, w_avg)
            (u_out, v_out, w_out, p_out) = self.project(u_avg, v_avg, p, w_vel=w_avg, w=0.5)

            return (u_out, v_out, p_out, w_out)

    def solve_heun(self, u, v, p, iteration, w_vel=None):
        if self.ndim == 2:
            return self.solver_iteration_heun(u, v, p, iteration)
        else:
            return self.solver_iteration_heun(u, v, p, iteration, w_vel=w_vel)

    def step_(self, u, v, p, iteration, t, w_vel=None):

        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        (self.mu0_all, self.mu1_all) = self.composite_body.mu_funcs(self.composite_body.sdf_val)
        self.m_m0_all                = (1-self.mu0_all)

        # --- mu / normals at u-staggered ---
        (self.mu0_all_u, self.mu1_all_u) = self.composite_body.mu_funcs(self.composite_body.sdf_val_u)
        self.m_m0_all_u                  = (1-self.mu0_all_u)

        # --- mu / normals at v-staggered ---
        (self.mu0_all_v, self.mu1_all_v) = self.composite_body.mu_funcs(self.composite_body.sdf_val_v)
        self.m_m0_all_v                  = (1-self.mu0_all_v)

        if self.ndim == 2:
            (_, self.normal_x, self.normal_y, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val)
            (_, self.normal_x_u, self.normal_y_u, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val_u)
            (_, self.normal_x_v, self.normal_y_v, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val_v)
        else:
            (_, self.normal_x, self.normal_y, self.normal_z, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val)
            (_, self.normal_x_u, self.normal_y_u, self.normal_z_u, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val_u)
            (_, self.normal_x_v, self.normal_y_v, self.normal_z_v, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val_v)
            # --- mu / normals at w-staggered ---
            (self.mu0_all_w, self.mu1_all_w) = self.composite_body.mu_funcs(self.composite_body.sdf_val_w)
            self.m_m0_all_w                  = (1-self.mu0_all_w)
            (_, self.normal_x_w, self.normal_y_w, self.normal_z_w, _) = self.composite_body.compute_sdf_properties(self.composite_body.sdf_val_w)

        ##### just for plotting
        self.sdf_properties = [[self.composite_body.sdf_val_u]]

        if self.ndim == 2:
            (u, v, p) = self.solve_heun(u, v, p, iteration)

            if self.compute_forces:
                self.forces_method2(u, v, p, iteration)

        else:
            (u, v, p, w_vel) = self.solve_heun(u, v, p, iteration, w_vel=w_vel)

            if self.compute_forces:
                self.forces_method2_3d(u, v, w_vel, p, iteration)

        # ---- plotting / saving  (works for 2D and 3D) ----
        terminate = self.plotting_and_saving(u, v, p, iteration, w_vel=w_vel)

        if self.ndim == 2:
            return (u, v, p, terminate)
        else:
            return (u, v, p, w_vel, terminate)

    # ------------------------------------------------------------------
    #   Unified plotting / saving   (replaces old plotting_debug)
    # ------------------------------------------------------------------

    # Default plot specifications.
    # Each entry: (name, field_lambda, vmin, vmax, body_contours)
    # field_lambda receives (solver, u, v, p, w_vel) and returns a CPU tensor/array.
    # vmin/vmax per-spec:  None or "yaml" → use the global output.vmin/vmax
    #                      float          → fixed limit for this field
    # The YAML output.vmin/vmax can be a number (fixed) or "auto" (auto-scale).

    DEFAULT_PLOT_SPECS = [
        ("curl",       lambda s, u, v, p, w: (
            s.vorticity(u, v, w).cpu() if s.ndim == 2
            else s.vorticity_components(u, v, w)["omega_z"].cpu()
        ), None, None, True),
        ("pressure",   lambda s, u, v, p, w: p.cpu(),                    None, None, True),
        ("divergence", lambda s, u, v, p, w: s.divergence(u, v, w).cpu(),None, None, False),
    ]

    # Additional specs used only for 3-D isosurface rendering.
    # Signed vorticity components give much better 3-D visualisation
    # than the magnitude (which loses rotational direction).
    # We cache the vorticity dict on the solver so it's computed once per frame.
    @staticmethod
    def _cached_vort(s, u, v, w):
        """Compute vorticity_components once per frame and cache on solver."""
        if not hasattr(s, "_vort_cache") or s._vort_cache_id != id(u):
            s._vort_cache = s.vorticity_components(u, v, w)
            s._vort_cache_id = id(u)
        return s._vort_cache

    @staticmethod
    def _vel_mag(s, u, v, w):
        """Velocity magnitude.  u, v, w all share the same (Nx+2, Ny+2, Nz+2)
        shape in this solver, so we can compute |V| element-wise.  The
        O(dx/2) stagger offset is negligible for visualisation."""
        return (u**2 + v**2 + w**2).sqrt()

    DEFAULT_3D_ISO_SPECS = [
        ("omega_x",    lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_x"].cpu(), None, True),
        ("omega_y",    lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_y"].cpu(), None, True),
        ("omega_z",    lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_z"].cpu(), None, True),
        ("omega_mag",  lambda s, u, v, p, w: FluidSolver._cached_vort(s, u, v, w)["omega_mag"].cpu(), None, True),
        ("vel_mag",    lambda s, u, v, p, w: FluidSolver._vel_mag(s, u, v, w).cpu(),                 None,   True),
        ("pressure",   lambda s, u, v, p, w: p.cpu(),                                                None,   True),
    ]

    def plotting_and_saving(self, u, v, p, iteration, *, w_vel=None, check_termination=True):
        """
        Unified plotting + data saving for 2-D and 3-D.
        Replaces the old ``plotting_debug`` and ``plotting_saving`` methods.
        """
        if iteration % self.save_every != 0:
            if check_termination:
                return self.check_termination(iteration, u, v, p)
            return False

        # ---- frame plots ----
        if self.save_frames:
            specs = getattr(self, "plot_specs", self.DEFAULT_PLOT_SPECS)
            bodies = getattr(self.composite_body, "bodies", None) if hasattr(self, "composite_body") else None

            if self.ndim == 2:
                for (name, field_fn, vmin, vmax, show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    # Per-spec None → fall back to YAML global; float → fixed
                    eff_vmin = self.vmin if vmin is None else vmin
                    eff_vmax = self.vmax if vmax is None else vmax
                    plotting.plot_field_2d(
                        field.detach().numpy() if hasattr(field, 'detach') else np.asarray(field),
                        self.extent,
                        name, iteration, self.save_path,
                        vmin=eff_vmin,
                        vmax=eff_vmax,
                        bodies=bodies if show_body else None,
                    )
            else:
                coords = {
                    "x": self.x.cpu().numpy(),
                    "y": self.y.cpu().numpy(),
                    "z": self.z.cpu().numpy(),
                }
                for (name, field_fn, vmin, vmax, _show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().numpy() if hasattr(field, 'detach') else np.asarray(field)
                    # Per-spec None → fall back to YAML global; float → fixed
                    eff_vmin = self.vmin if vmin is None else vmin
                    eff_vmax = self.vmax if vmax is None else vmax
                    plotting.plot_field_3d_slices(
                        field_np, coords,
                        name, iteration, self.save_path,
                        vmin=eff_vmin, vmax=eff_vmax,
                        bodies=None,   # body contours not meaningful in 3-D slices
                    )

                # ---- VTK export (3-D only) ----
                if self.save_vtk:
                    vtk_fields = {"u": u.cpu().numpy(), "v": v.cpu().numpy(), "p": p.cpu().numpy()}
                    if w_vel is not None:
                        vtk_fields["w"] = w_vel.cpu().numpy()
                    plotting.save_vtk(vtk_fields, coords, iteration, self.save_path)

                # ---- 3-D isosurface renders ----
                sdf_np = None
                if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                    sdf_np = self.composite_body.sdf_val.cpu().numpy()
                iso_specs = getattr(self, "iso_3d_specs",
                                    self.DEFAULT_3D_ISO_SPECS if self.ndim == 3
                                    else specs)
                for iso_entry in iso_specs:
                    name, field_fn = iso_entry[0], iso_entry[1]
                    # iso_entry format: (name, field_fn, iso_thresh, show_body)
                    # iso_thresh can be: None (auto), a float, or "vmax" (use self.vmax)
                    iso_thresh = iso_entry[2] if len(iso_entry) > 2 else None
                    if iso_thresh == "vmax":
                        iso_thresh = getattr(self, "vmax", None)
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().numpy() if hasattr(field, 'detach') else np.asarray(field)
                    plotting.plot_field_3d(
                        field_np, coords,
                        name, iteration, self.save_path,
                        sdf_3d=sdf_np,
                        iso_value=iso_thresh,
                    )

        # ---- raw data save ----
        if self.save_uv:
            self.save_results(u, v, p, iteration, w_vel=w_vel)

        if check_termination:
            return self.check_termination(iteration, u, v, p)
        return False

    # keep old name as an alias so call-sites in alternative solver variants still work
    def plotting_debug(self, u, v, p, iteration, check_termination=True):
        return self.plotting_and_saving(u, v, p, iteration, check_termination=check_termination)

    def check_termination(self, iteration, u, v, p):
        if iteration == self.nt - 1 or torch.isnan(u).any():
                print("Termination condition met: max_iter or NaN")
                terminate = True
        else:
            if hasattr(self.composite_body, "com_pos"):
                    terminate = not self.inside(self.composite_body.com_pos)
                    if terminate:
                        print("Termination condition met: body exited domain")
            else:
                terminate = False
        return terminate

    def plotting_saving(self, u, v, p, iteration):
        """Legacy alias — delegates to the unified method."""
        self.plotting_and_saving(u, v, p, iteration, check_termination=False)

    def save_results(self, u, v, p, iteration, *, w_vel=None):
        if self.save_uv:
          uv_path = f'{self.save_path}/uv_field'
          os.makedirs(uv_path, exist_ok=True)
          np.save(f'{uv_path}/u_{iteration}',u.cpu().numpy())
          np.save(f'{uv_path}/v_{iteration}',v.cpu().numpy())
          np.save(f'{uv_path}/p_{iteration}',p.cpu().numpy())
          if w_vel is not None:
              np.save(f'{uv_path}/w_{iteration}', w_vel.cpu().numpy())
          if iteration==0:
              np.save(f'{uv_path}/x_grid',self.x.cpu().numpy())
              np.save(f'{uv_path}/y_grid',self.y.cpu().numpy())
              if self.z is not None:
                  np.save(f'{uv_path}/z_grid',self.z.cpu().numpy())
          if self.ndim == 2:
              cnt_path = f'{self.save_path}/cnt_field'
              os.makedirs(cnt_path, exist_ok=True)
              for i, body in enumerate(self.composite_body.bodies):
                  np.save(f'{cnt_path}/cnt_{iteration}_{i}',body.cnt_update.cpu().numpy())

    def run_from_initial(self, u0, v0, w0=None):
        u = u0
        v = v0
        p = torch.zeros_like(u)
        if self.ndim == 2:
            for iteration in tqdm(range(self.nt)):
                t                = iteration*self.dt
                (u,v,p,stop_sim) = self.step_(u, v, p, iteration, t)
        else:
            w = w0 if w0 is not None else torch.zeros_like(u)
            for iteration in tqdm(range(self.nt)):
                t                      = iteration*self.dt
                (u,v,p,w,stop_sim) = self.step_(u, v, p, iteration, t, w_vel=w)

    def run_sim(self):
        u = self.u0
        v = self.v0
        p = self.p0
        if self.ndim == 2:
            for iteration in tqdm(range(self.starting_iteration, self.nt)):
                t                = iteration*self.dt
                (u,v,p,stop_sim) = self.step_(u, v, p, iteration, t)
        else:
            w = self.w0
            for iteration in tqdm(range(self.starting_iteration, self.nt)):
                t                      = iteration*self.dt
                (u,v,p,w,stop_sim) = self.step_(u, v, p, iteration, t, w_vel=w)

        if self.compute_forces and self.save_uv:
            uv_path = f'{self.save_path}'
            np.save(f'{uv_path}/viscous_drags',self.viscous_drag_record.cpu().numpy())
            np.save(f'{uv_path}/pressure_drags',self.pressure_drag_record.cpu().numpy())


if __name__ == "__main__":

    solver = FluidSolver()
    solver.run_sim()


