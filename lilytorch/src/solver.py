
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
from concurrent.futures import ThreadPoolExecutor
# from spreading_operator import spreading_operator_python_parallel, spreading_operator_python_parallel_out
# from spreading_operator import interpolation_operator, spreading_operator_python, interp_operator_python_parallel

# ======================================================================
# Compilable force-computation kernels  (module-level, for torch.compile)
# ======================================================================

def _forces_shared_3d(u, v, w, p, sdf_val, nx, ny, nz, nu_rho, h):
    """Compute velocity gradients, viscous stress·n, and pressure force density.

    All arguments are plain tensors or scalars — no ``self`` access — so this
    function is safe for ``torch.compile(mode='reduce-overhead')``.

    Returns
    -------
    xstress, ystress, zstress : viscous-stress × normal  (σ·n)_i
    pforce_x, pforce_y, pforce_z : -p_outer * n_i
    """
    # 9 velocity gradients (central difference, edge_order=2)
    dudx = torch.gradient(u, spacing=h, dim=0, edge_order=2)[0]
    dudy = torch.gradient(u, spacing=h, dim=1, edge_order=2)[0]
    dudz = torch.gradient(u, spacing=h, dim=2, edge_order=2)[0]
    dvdx = torch.gradient(v, spacing=h, dim=0, edge_order=2)[0]
    dvdy = torch.gradient(v, spacing=h, dim=1, edge_order=2)[0]
    dvdz = torch.gradient(v, spacing=h, dim=2, edge_order=2)[0]
    dwdx = torch.gradient(w, spacing=h, dim=0, edge_order=2)[0]
    dwdy = torch.gradient(w, spacing=h, dim=1, edge_order=2)[0]
    dwdz = torch.gradient(w, spacing=h, dim=2, edge_order=2)[0]

    # viscous stress tensor  σ_{ij} n_j  (summed over j, for each i)
    xstress = nu_rho * (2 * dudx * nx + (dudy + dvdx) * ny + (dudz + dwdx) * nz)
    ystress = nu_rho * ((dvdx + dudy) * nx + 2 * dvdy * ny + (dvdz + dwdy) * nz)
    zstress = nu_rho * ((dwdx + dudz) * nx + (dwdy + dvdz) * ny + 2 * dwdz * nz)

    # pressure force density (outside body only)
    p_outer  = torch.where(sdf_val < 0, 0.0, p)
    pforce_x = -p_outer * nx
    pforce_y = -p_outer * ny
    pforce_z = -p_outer * nz

    return xstress, ystress, zstress, pforce_x, pforce_y, pforce_z


def _forces_body_integrate_3d(
    xstress, ystress, zstress,
    pforce_x, pforce_y, pforce_z,
    sdf_i, eps_body, eps_solver,
    com_x, com_y, com_z,
    X, Y, Z, h3,
):
    """Integrate viscous + pressure force / torque for ONE body.

    Inlines ``body.phi`` (smoothed delta) and ``cross_product_3d`` for
    maximal fusion under ``torch.compile``.

    Returns 18 scalars: (fv_x,fv_y,fv_z, tv_x,tv_y,tv_z,
                          fp_x,fp_y,fp_z, tp_x,tp_y,tp_z).
    """
    # smoothed delta — viscous (shifted by eps_solver)
    d_visc = sdf_i - eps_solver
    delta_visc = torch.where(
        torch.abs(d_visc) < eps_body,
        (1.0 + torch.cos(torch.pi * d_visc / eps_body)) / (2.0 * eps_body),
        0.0,
    )
    # smoothed delta — pressure
    delta_pres = torch.where(
        torch.abs(sdf_i) < eps_body,
        (1.0 + torch.cos(torch.pi * sdf_i / eps_body)) / (2.0 * eps_body),
        0.0,
    )

    # viscous forces
    fvisc_x = xstress * delta_visc
    fvisc_y = ystress * delta_visc
    fvisc_z = zstress * delta_visc
    fv_x = fvisc_x.sum() * h3
    fv_y = fvisc_y.sum() * h3
    fv_z = fvisc_z.sum() * h3

    # moment arms from CoM
    rx = X - com_x
    ry = Y - com_y
    rz = Z - com_z

    # torque  r × f_visc
    tv_x = (ry * fvisc_z - rz * fvisc_y).sum() * h3
    tv_y = (rz * fvisc_x - rx * fvisc_z).sum() * h3
    tv_z = (rx * fvisc_y - ry * fvisc_x).sum() * h3

    # pressure forces
    fpres_x = pforce_x * delta_pres
    fpres_y = pforce_y * delta_pres
    fpres_z = pforce_z * delta_pres
    fp_x = fpres_x.sum() * h3
    fp_y = fpres_y.sum() * h3
    fp_z = fpres_z.sum() * h3

    # torque  r × f_pres
    tp_x = (ry * fpres_z - rz * fpres_y).sum() * h3
    tp_y = (rz * fpres_x - rx * fpres_z).sum() * h3
    tp_z = (rx * fpres_y - ry * fpres_x).sum() * h3

    return (fv_x, fv_y, fv_z, tv_x, tv_y, tv_z,
            fp_x, fp_y, fp_z, tp_x, tp_y, tp_z)


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

    def __init__(self, pars, dtype=torch.float32, custom_update=None, compute_forces=True):
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

        # ---- optional torch.compile for adv-diff + BDIM kernels -----
        self._compile_adv_diff = solver.get("compile_adv_diff", False)
        if self._compile_adv_diff and self.device.type == "cuda":
            self.adv_diff_solver.solve = torch.compile(
                self.adv_diff_solver.solve, mode="reduce-overhead",
            )
            self._bdim_meta_compiled = torch.compile(
                FluidSolver._bdim_meta, mode="reduce-overhead",
            )
            print("  [compile] adv_diff_solver.solve + BDIM meta-equation compiled (reduce-overhead)")
        else:
            self._bdim_meta_compiled = FluidSolver._bdim_meta

        # ---- optional torch.compile for force computation -----
        self._compile_forces = solver.get("compile_forces", False)
        if self._compile_forces and self.device.type == "cuda":
            self._forces_shared_compiled = torch.compile(
                _forces_shared_3d, mode="reduce-overhead",
            )
            self._forces_body_compiled = torch.compile(
                _forces_body_integrate_3d, mode="reduce-overhead",
            )
            print("  [compile] forces_shared + forces_body_integrate compiled (reduce-overhead)")
        else:
            self._forces_shared_compiled = _forces_shared_3d
            self._forces_body_compiled = _forces_body_integrate_3d

        # ---- optional torch.compile for SDF / mu / normals ----
        self._compile_sdf = solver.get("compile_sdf", False)
        if self._compile_sdf and self.device.type == "cuda":
            print("  [compile] SDF rotation, staggering, mu+normals compiled (reduce-overhead)")

        # =============  poisson solver =============
        self.poisson_method = solver.get("poisson_method", "multigrid")
        assert self.poisson_method in ("multigrid", "mgcg", "fft"), \
            f"Unknown poisson_method '{self.poisson_method}'. Choose 'multigrid', 'mgcg', or 'fft'."
        print(f"Poisson solver: {self.poisson_method}")

        self.poisson_solver  = PoissonSolver(
            self.dtype,
            self.device,
            self.h,
            tol             = solver["poisson_tol"],
            max_cycles      = solver["poisson_max_mgcg_cycles"],
            max_vcycles     = solver["poisson_max_cycles"],
            nsmoothing      = solver["poisson_nsmoothing"],
            w               = solver["jacobi_weight"],
            verbose         = solver["poisson_verbose"],
            precond_vcycles = solver.get("poisson_precond_vcycles", 1),
            smoother        = solver.get("poisson_smoother", "jacobi"),
            compile_smoother= solver.get("poisson_compile", False),
        )

        # Warm-start: reuse previous pressure as Poisson initial guess
        self.poisson_warm_start = solver.get("poisson_warm_start", False)

        # Only build the FFT solver when it will actually be used.
        # Gfft + U buffer cost ~834 MB on a 512x128x128 grid.
        self.poisson_bc_type = solver.get("poisson_bc_type", "free")
        assert self.poisson_bc_type in ("free", "neumann"), \
            f"Unknown poisson_bc_type '{self.poisson_bc_type}'. Choose 'free' or 'neumann'."

        if self.poisson_method == "fft":
            fft_kwargs = dict(
                bc_type  = self.poisson_bc_type,
                filename = solver["poisson_folder"],
            )
            if self.ndim == 2:
                self.poisson_solverFFT = PoissonSolverFFT(
                    self.x, self.y, **fft_kwargs,
                )
            else:
                self.poisson_solverFFT = PoissonSolverFFT(
                    self.x, self.y, z=self.z, **fft_kwargs,
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
            custom_update = custom_update,
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

        # NOTE: xstress_tensor, ystress_tensor, zstress_tensor, and div
        # are created on-the-fly in forces_method* and project() respectively
        # (no pre-allocation needed — they are rebound, not written in-place).


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

        # Background thread pool for async I/O (saving + plotting)
        self._io_executor = ThreadPoolExecutor(max_workers=2)
        self._io_futures = []  # track pending I/O tasks

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
        in_xy = torch.logical_and(
            x[:,0]>self.xmin,
            torch.logical_and(
                x[:,0]<self.xmax,
                torch.logical_and(
                    x[:,1]>self.ymin,
                    x[:,1]<self.ymax
                )
            )
        )
        if self.ndim == 3:
            in_xy = torch.logical_and(
                in_xy,
                torch.logical_and(x[:,2]>self.zmin, x[:,2]<self.zmax)
            )
        return torch.all(in_xy)

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

        # ---- CC normals (computed on-the-fly, not cached on self) ------
        normal_x, normal_y = self.composite_body.compute_normals(
            self.composite_body.sdf_val
        )


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

        self.pforce_x = -p*normal_x
        self.pforce_y = -p*normal_y


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

        # ---- CC normals (computed on-the-fly, not cached on self) ------
        normal_x, normal_y = self.composite_body.compute_normals(
            self.composite_body.sdf_val
        )

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
        self.pforce_x = -p_outer*normal_x
        self.pforce_y = -p_outer*normal_y


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

        When ``compile_forces=True``, the heavy tensor work is delegated to
        two ``torch.compile``-d kernels (``_forces_shared_3d`` and
        ``_forces_body_integrate_3d``) that fuse ~40 CUDA kernels into one
        or two CUDA-graph launches, giving ~6× wall-clock speedup.
        """
        nu_rho = self.nu * self.rho
        h      = self.h
        h3     = self.h3

        # ---- CC normals (reuse cached values when available) ---------
        nx = getattr(self, 'normal_x', None)
        if nx is None:
            nx, ny, nz = self.composite_body.compute_normals(
                self.composite_body.sdf_val)
        else:
            ny, nz = self.normal_y, self.normal_z

        # ---- shared part: gradients + stress + pressure density ------
        (xstress, ystress, zstress,
         pforce_x, pforce_y, pforce_z) = self._forces_shared_compiled(
            u, v, w, p, self.composite_body.sdf_val, nx, ny, nz,
            nu_rho, h,
        )

        # When compiled with CUDA graphs (reduce-overhead), the returned
        # tensors live in the graph's replay buffer and will be overwritten
        # by the next compiled call.  Clone them here so the body-integrate
        # kernel can safely read them.
        if self._compile_forces:
            xstress  = xstress.clone()
            ystress  = ystress.clone()
            zstress  = zstress.clone()
            pforce_x = pforce_x.clone()
            pforce_y = pforce_y.clone()
            pforce_z = pforce_z.clone()

        # Cache stress / pforce for post-processing if needed
        self.xstress_tensor = xstress
        self.ystress_tensor = ystress
        self.zstress_tensor = zstress
        self.pforce_x = pforce_x
        self.pforce_y = pforce_y
        self.pforce_z = pforce_z

        # ---- per-body integration ------------------------------------
        X   = self.composite_body.X
        Y   = self.composite_body.Y
        Z   = self.composite_body.Z_grid

        for i, body in enumerate(self.composite_body.bodies):
            sdf_i    = self.composite_body.sdf_vals[i]
            eps_body = body.eps          # smoothing width (same for all bodies)

            (fv_x, fv_y, fv_z,
             tv_x, tv_y, tv_z,
             fp_x, fp_y, fp_z,
             tp_x, tp_y, tp_z) = self._forces_body_compiled(
                xstress, ystress, zstress,
                pforce_x, pforce_y, pforce_z,
                sdf_i, eps_body, self.eps,
                body.com_pos[0], body.com_pos[1], body.com_pos[2],
                X, Y, Z, h3,
            )

            # store linear forces
            self.friction_force_lin_x[i] = fv_x
            self.friction_force_lin_y[i] = fv_y
            self.friction_force_lin_z[i] = fv_z

            # store viscous torques
            self.friction_force_ang_x[i] = tv_x
            self.friction_force_ang_y[i] = tv_y
            self.friction_force_ang_z[i] = tv_z

            # store pressure forces
            self.pressure_force_x[i] = fp_x
            self.pressure_force_y[i] = fp_y
            self.pressure_force_z[i] = fp_z

            # store pressure torques
            self.pressure_force_ang_x[i] = tp_x
            self.pressure_force_ang_y[i] = tp_y
            self.pressure_force_ang_z[i] = tp_z

            # record for post-processing
            self.viscous_drag_record[i, 0, iteration]  = fv_x
            self.viscous_drag_record[i, 1, iteration]  = fv_y
            self.viscous_drag_record[i, 2, iteration]  = fv_z

            self.pressure_drag_record[i, 0, iteration] = fp_x
            self.pressure_drag_record[i, 1, iteration] = fp_y
            self.pressure_drag_record[i, 2, iteration] = fp_z


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
            # ---- Multigrid / MGCG solver (variable-coefficient Poisson) ----
            c = torch.ones_like(u)
            ch = coeff*self.mu0_all_u
            cv = coeff*self.mu0_all_v

            # Select solve method: MGCG or standalone multigrid
            _poisson_solve = (self.poisson_solver.solve_mgcg
                              if self.poisson_method == "mgcg"
                              else self.poisson_solver.solve_multigrid)

            # Warm-start: reuse previous pressure as initial guess
            p0 = p if self.poisson_warm_start else torch.zeros_like(u)

            if self.ndim == 2:
                p, _    = _poisson_solve(
                    self.div[1:-1,1:-1],
                    p0,
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
                p, _ = _poisson_solve(
                    self.div[1:-1, 1:-1, 1:-1],
                    p0,
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



    @staticmethod
    def _bdim_meta(
        phi, mu0, body_vel, mu1, *normals_and_extras,
    ):
        """BDIM2 meta-equation (compilable, elementwise).

        phi_out = mu0 * (phi - body_vel) + body_vel
                  + mu1 * normal_derivative(phi - body_vel, ...)

        Algebraically equivalent to mu0*phi + (1-mu0)*body_vel + mu1*nd
        but avoids materialising the (1 - mu0) tensor.

        Parameters
        ----------
        normals_and_extras : variable-length
            2-D: (normal_x, normal_y, h_scalar as 0-d tensor, ndim_int=2)
            3-D: (normal_x, normal_y, normal_z, h_scalar, ndim_int=3)
        Last two positional args are always (h, ndim) for normal_derivative.
        """
        # unpack: last two are (h, ndim), preceding are normals
        h    = normals_and_extras[-2]
        ndim = normals_and_extras[-1]
        normals = normals_and_extras[:-2]
        normal_x = normals[0]
        normal_y = normals[1]
        normal_z = normals[2] if len(normals) == 3 else None
        diff = phi - body_vel
        nd = ops.normal_derivative(diff, h, ndim, normal_x, normal_y, normal_z)
        return mu0 * diff + body_vel + mu1 * nd

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

            # BDIM2 meta-equation (fused when compiled)
            _bdim = self._bdim_meta_compiled
            _h    = self.h
            uprime = _bdim(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            )
            vprime = _bdim(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            )

            # Keep references to BDIM'd velocities for Heun averaging.
            # No .clone() needed: project() does not modify its inputs
            # in-place, and set_BCs() only touches boundary cells that
            # get overwritten by the final set_BCs(u_avg) anyway.
            uprime_bdim = uprime
            vprime_bdim = vprime

            self.adv_diff_solver.set_BCs(uprime, vprime)
            (u1, v1, p1) = self.project(uprime, vprime, p)

            # ====== CORRECTOR ======
            # Evaluate RHS at the projected predicted velocity
            (uprime2, vprime2) = self.adv_diff_solver.solve(u1, v1)
            # adv_diff.solve returns u1 + dt*RHS(u1).
            # Heun needs u^n + dt*RHS(u1), so rebase from u^n:
            uprime2 = u + (uprime2 - u1)
            vprime2 = v + (vprime2 - v1)

            # BDIM2 meta-equation on corrector (fused when compiled)
            uprime2 = _bdim(
                uprime2, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            )
            vprime2 = _bdim(
                vprime2, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            )

            # Average the BDIM'd pre-projection velocities (WaterLily style)
            u_avg = 0.5 * (uprime_bdim + uprime2)
            v_avg = 0.5 * (vprime_bdim + vprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg)
            (u_out, v_out, p_out) = self.project(u_avg, v_avg, p, w=0.5)

            return (u_out, v_out, p_out)

        else:  # 3D
            # ====== PREDICTOR ======
            (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w_vel)

            # BDIM2 meta-equation (fused when compiled)
            _bdim = self._bdim_meta_compiled
            _h    = self.h
            uprime = _bdim(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u, _h, 3,
            )
            vprime = _bdim(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v, _h, 3,
            )
            wprime = _bdim(
                wprime, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w, _h, 3,
            )

            # Keep references to BDIM'd velocities for Heun averaging.
            # No .clone() needed — see 2-D comment above.
            uprime_bdim = uprime
            vprime_bdim = vprime
            wprime_bdim = wprime

            self.adv_diff_solver.set_BCs(uprime, vprime, wprime)
            (u1, v1, w1, p1) = self.project(uprime, vprime, p, w_vel=wprime)

            # ====== CORRECTOR ======
            (uprime2, vprime2, wprime2) = self.adv_diff_solver.solve(u1, v1, w1)
            # Rebase from u^n
            uprime2 = u     + (uprime2 - u1)
            vprime2 = v     + (vprime2 - v1)
            wprime2 = w_vel + (wprime2 - w1)

            # BDIM2 meta-equation on corrector (fused when compiled)
            uprime2 = _bdim(
                uprime2, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u, _h, 3,
            )
            vprime2 = _bdim(
                vprime2, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v, _h, 3,
            )
            wprime2 = _bdim(
                wprime2, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w, _h, 3,
            )

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

    # ------------------------------------------------------------------
    #   BDIM field cleanup  —  free intermediate tensors between steps
    # ------------------------------------------------------------------
    _BDIM_FIELD_NAMES = (
        # staggered-grid mu / normals (u, v, w)
        'mu0_all_u', 'mu1_all_u', 'mu0_all_v', 'mu1_all_v',
        'normal_x_u', 'normal_y_u', 'normal_x_v', 'normal_y_v',
        # 3-D only (harmlessly absent in 2-D)
        'mu0_all_w', 'mu1_all_w',
        'normal_x_w', 'normal_y_w',
        'normal_z_u', 'normal_z_v', 'normal_z_w',
        # CC-grid mu / normals (recomputed in _recompute_mu_normals_3d)
        'mu0_all', 'mu1_all', 'm_m0_all',
        'normal_x', 'normal_y', 'normal_z',
        # force intermediates (recomputed in forces_method2 / forces_method2_3d)
        'xstress_tensor', 'ystress_tensor', 'zstress_tensor',
        'pforce_x', 'pforce_y', 'pforce_z',
        # divergence from project()
        'div',
    )

    def _release_bdim_fields(self):
        """Set BDIM intermediate fields to *None* so their GPU memory
        can be reclaimed between time-steps (they are recomputed at the
        beginning of every step anyway)."""
        for attr in self._BDIM_FIELD_NAMES:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def step_(self, u, v, p, iteration, t, w_vel=None):

        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        # --- mu / normals at u-staggered ---
        (self.mu0_all_u, self.mu1_all_u) = self.composite_body.mu_funcs(self.composite_body.sdf_val_u)

        # --- mu / normals at v-staggered ---
        (self.mu0_all_v, self.mu1_all_v) = self.composite_body.mu_funcs(self.composite_body.sdf_val_v)

        if self.ndim == 2:
            # CC normals are computed on-the-fly inside forces_method1/2
            (self.normal_x_u, self.normal_y_u) = self.composite_body.compute_normals(self.composite_body.sdf_val_u)
            (self.normal_x_v, self.normal_y_v) = self.composite_body.compute_normals(self.composite_body.sdf_val_v)
        else:
            # CC normals are computed on-the-fly inside forces_method2_3d
            (self.normal_x_u, self.normal_y_u, self.normal_z_u) = self.composite_body.compute_normals(self.composite_body.sdf_val_u)
            (self.normal_x_v, self.normal_y_v, self.normal_z_v) = self.composite_body.compute_normals(self.composite_body.sdf_val_v)
            # --- mu / normals at w-staggered ---
            (self.mu0_all_w, self.mu1_all_w) = self.composite_body.mu_funcs(self.composite_body.sdf_val_w)
            (self.normal_x_w, self.normal_y_w, self.normal_z_w) = self.composite_body.compute_normals(self.composite_body.sdf_val_w)

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

        # ---- free BDIM fields to reclaim GPU memory between steps ----
        self._release_bdim_fields()

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

    def _submit_io(self, fn, *args, **kwargs):
        """Submit *fn* to the background I/O pool and track the future."""
        # Reap already-finished futures to avoid unbounded growth
        self._io_futures = [f for f in self._io_futures if not f.done()]
        fut = self._io_executor.submit(fn, *args, **kwargs)
        self._io_futures.append(fut)
        return fut

    def flush_io(self):
        """Block until all pending background I/O tasks have completed."""
        for fut in self._io_futures:
            fut.result()          # re-raises any exception from the worker
        self._io_futures.clear()

    def plotting_and_saving(self, u, v, p, iteration, *, w_vel=None, check_termination=True):
        """
        Unified plotting + data saving for 2-D and 3-D.
        Replaces the old ``plotting_debug`` and ``plotting_saving`` methods.

        Plotting and saving are offloaded to a background thread so the
        solver loop is not blocked by synchronous disk I/O.
        """
        if iteration % self.save_every != 0:
            if check_termination:
                return self.check_termination(iteration, u, v, p)
            return False

        # ---- snapshot tensors to CPU numpy *once* (still on main thread
        #      so the GPU transfer is overlapped with the previous kernel)
        # We clone/detach to decouple from the live computation graph.

        # ---- frame plots ----
        if self.save_frames:
            specs = getattr(self, "plot_specs", self.DEFAULT_PLOT_SPECS)
            bodies = getattr(self.composite_body, "bodies", None) if hasattr(self, "composite_body") else None

            if self.ndim == 2:
                for (name, field_fn, vmin, vmax, show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    eff_vmin = self.vmin if vmin is None else vmin
                    eff_vmax = self.vmax if vmax is None else vmax
                    extent   = self.extent  # plain tuple, safe to share
                    save_path = self.save_path
                    _bodies  = bodies if show_body else None
                    self._submit_io(
                        plotting.plot_field_2d,
                        field_np, extent,
                        name, iteration, save_path,
                        vmin=eff_vmin, vmax=eff_vmax, bodies=_bodies,
                    )
            else:
                coords = {
                    "x": self.x.cpu().numpy().copy(),
                    "y": self.y.cpu().numpy().copy(),
                    "z": self.z.cpu().numpy().copy(),
                }
                for (name, field_fn, vmin, vmax, _show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    eff_vmin = self.vmin if vmin is None else vmin
                    eff_vmax = self.vmax if vmax is None else vmax
                    save_path = self.save_path
                    self._submit_io(
                        plotting.plot_field_3d_slices,
                        field_np, coords,
                        name, iteration, save_path,
                        vmin=eff_vmin, vmax=eff_vmax, bodies=None,
                    )

                # ---- VTK export (3-D only) ----
                if self.save_vtk:
                    vtk_fields = {
                        "u": u.detach().cpu().numpy().copy(),
                        "v": v.detach().cpu().numpy().copy(),
                        "p": p.detach().cpu().numpy().copy(),
                    }
                    if w_vel is not None:
                        vtk_fields["w"] = w_vel.detach().cpu().numpy().copy()
                    self._submit_io(plotting.save_vtk, vtk_fields, coords, iteration, self.save_path)

                # ---- 3-D isosurface renders ----
                sdf_np = None
                if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                    sdf_np = self.composite_body.sdf_val.cpu().numpy().copy()
                iso_specs = getattr(self, "iso_3d_specs",
                                    self.DEFAULT_3D_ISO_SPECS if self.ndim == 3
                                    else specs)
                for iso_entry in iso_specs:
                    name, field_fn = iso_entry[0], iso_entry[1]
                    iso_thresh = iso_entry[2] if len(iso_entry) > 2 else None
                    if iso_thresh == "vmax":
                        iso_thresh = getattr(self, "vmax", None)
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    self._submit_io(
                        plotting.plot_field_3d,
                        field_np, coords,
                        name, iteration, self.save_path,
                        sdf_3d=sdf_np, iso_value=iso_thresh,
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

          # Snapshot tensors to CPU numpy arrays on the main thread (single
          # GPU→CPU transfer), then hand the actual np.save() to a worker.
          u_np = u.detach().cpu().numpy().copy()
          v_np = v.detach().cpu().numpy().copy()
          p_np = p.detach().cpu().numpy().copy()

          def _save_fields(path, it, u_a, v_a, p_a, w_a):
              np.save(f'{path}/u_{it}', u_a)
              np.save(f'{path}/v_{it}', v_a)
              np.save(f'{path}/p_{it}', p_a)
              if w_a is not None:
                  np.save(f'{path}/w_{it}', w_a)

          w_np = w_vel.detach().cpu().numpy().copy() if w_vel is not None else None
          self._submit_io(_save_fields, uv_path, iteration, u_np, v_np, p_np, w_np)

          if iteration == 0:
              x_np = self.x.cpu().numpy().copy()
              y_np = self.y.cpu().numpy().copy()
              z_np = self.z.cpu().numpy().copy() if self.z is not None else None

              def _save_grids(path, x_a, y_a, z_a):
                  np.save(f'{path}/x_grid', x_a)
                  np.save(f'{path}/y_grid', y_a)
                  if z_a is not None:
                      np.save(f'{path}/z_grid', z_a)

              self._submit_io(_save_grids, uv_path, x_np, y_np, z_np)

          if self.ndim == 2:
              cnt_path = f'{self.save_path}/cnt_field'
              os.makedirs(cnt_path, exist_ok=True)
              for i, body in enumerate(self.composite_body.bodies):
                  cnt_np = body.cnt_update.cpu().numpy().copy()
                  self._submit_io(np.save, f'{cnt_path}/cnt_{iteration}_{i}', cnt_np)

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
            vd_np = self.viscous_drag_record.cpu().numpy().copy()
            pd_np = self.pressure_drag_record.cpu().numpy().copy()
            self._submit_io(np.save, f'{uv_path}/viscous_drags', vd_np)
            self._submit_io(np.save, f'{uv_path}/pressure_drags', pd_np)

        # Block until all background I/O is complete before returning
        self.flush_io()


if __name__ == "__main__":

    solver = FluidSolver()
    solver.run_sim()


