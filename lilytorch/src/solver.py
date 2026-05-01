
import datetime
import logging
import os
import threading
import warnings

import h5py
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from lilytorch.src.kernels import RegularGridInterpolator
from tqdm import tqdm

from lilytorch.src.adv_diff import AdvDiffSolver
from lilytorch.src.body import (body_from_yaml, _StaggeredGrids,
                                _mu_normals_batched_3d,
                                _mu_normals_batched_3d_compiled)
from lilytorch.src import operations as ops
from lilytorch.src import plotting
from lilytorch.src.plotting import PlottingMixin
from lilytorch.src.poisson_fft import PoissonSolverFFT
from lilytorch.src.poisson_mult import PoissonSolver
from lilytorch.util.yaml_operations import pyobject2yaml

logger = logging.getLogger(__name__)


# ======================================================================
# Flow diagnostics — energy, enstrophy, divergence, CFL monitoring
# ======================================================================

class FlowDiagnostics:
    """Lightweight monitor for kinetic energy, enstrophy, max-divergence,
    and CFL number.  Records scalar time-series and optionally warns when
    energy grows beyond a user-specified factor of its initial value.

    Parameters
    ----------
    nt : int
        Total number of time steps (for pre-allocation).
    ndim : int
        Spatial dimension (2 or 3).
    h : float or Tensor
        Uniform grid spacing.
    device, dtype
        Torch device / dtype for the record arrays.
    check_every : int
        Diagnostics are computed every *check_every* steps.  1 = every step.
    energy_growth_factor : float
        Issue a warning when E_k exceeds *energy_growth_factor* × E_k(0).
        Set to ``None`` or ``inf`` to disable the energy blow-up check.
    """

    def __init__(self, nt, ndim, h, device, dtype,
                 check_every=1, energy_growth_factor=10.0):
        self.nt    = nt
        self.ndim  = ndim
        self.h     = float(h)
        self.hd    = self.h ** ndim          # cell volume  h^d
        self.device = device
        self.dtype  = dtype

        self.check_every = max(1, int(check_every))
        self.energy_growth_factor = energy_growth_factor

        # Pre-allocated record arrays (filled with NaN so uncomputed slots
        # are visually obvious when plotted).
        self.kinetic_energy = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.enstrophy      = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.max_divergence  = torch.full((nt,), float('nan'), device=device, dtype=dtype)
        self.cfl_number      = torch.full((nt,), float('nan'), device=device, dtype=dtype)

        self._ek0 = None   # E_k at the first computed step (baseline)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, iteration, u, v, p, dt, nu, divergence_fn, vorticity_fn,
               w=None):
        """Compute and record diagnostics for the current step.

        Parameters
        ----------
        iteration : int
            Current time-step index.
        u, v : Tensor
            Velocity components.  *w* is ``None`` in 2-D.
        p : Tensor
            Pressure (unused for now; kept for future pressure-energy).
        dt : float or Tensor
            Time step size (for CFL).
        nu : float or Tensor
            Kinematic viscosity (for CFL).
        divergence_fn : callable(u, v, w=None) -> Tensor
            Solver's divergence method.
        vorticity_fn  : callable(u, v, w=None) -> Tensor
            Solver's vorticity method (returns scalar in 2-D,
            magnitude in 3-D).
        w : Tensor or None
            z-velocity component (3-D only).
        """
        if iteration % self.check_every != 0:
            return

        h  = self.h
        hd = self.hd
        dt_val = float(dt)
        nu_val = float(nu)

        # ---- kinetic energy  E_k = 0.5 * h^d * Σ(u² + v² [+ w²]) ----
        ke = u.square() + v.square()
        if w is not None:
            ke = ke + w.square()
        ek = 0.5 * hd * ke.sum()
        self.kinetic_energy[iteration] = ek

        # ---- enstrophy  Z = 0.5 * h^d * Σ ω² ----
        omega = vorticity_fn(u, v, w)
        enst = 0.5 * hd * omega.square().sum()
        self.enstrophy[iteration] = enst

        # ---- max |div(u)| ----
        div = divergence_fn(u, v, w=w)
        self.max_divergence[iteration] = div.abs().max()

        # ---- CFL = u_max * dt / h ----
        vel_max = u.abs().max()
        vel_max = max(vel_max, v.abs().max())
        if w is not None:
            vel_max = max(vel_max, w.abs().max())
        self.cfl_number[iteration] = float(vel_max) * dt_val / h

        # ---- energy blow-up warning ----
        if self._ek0 is None:
            self._ek0 = float(ek) if float(ek) > 0 else 1.0
        if (self.energy_growth_factor is not None
                and float(ek) > self.energy_growth_factor * self._ek0):
            warnings.warn(
                f"[FlowDiagnostics] E_k = {float(ek):.6e} at iter {iteration} "
                f"exceeds {self.energy_growth_factor}x initial "
                f"({self._ek0:.6e}).  Possible blow-up.",
                RuntimeWarning, stacklevel=2,
            )

        # ---- CFL warning ----
        cfl_val = float(self.cfl_number[iteration])
        if cfl_val > 0.5:
            warnings.warn(
                f"[FlowDiagnostics] CFL = {cfl_val:.3f} > 0.5 at iter {iteration}",
                RuntimeWarning, stacklevel=2,
            )

    # ------------------------------------------------------------------
    def save_h5(self, path, lock):
        """Write diagnostics to ``<path>/diagnostics.h5``."""
        h5_path = os.path.join(path, "diagnostics.h5")
        data = {
            "kinetic_energy": self.kinetic_energy.cpu().numpy().copy(),
            "enstrophy":      self.enstrophy.cpu().numpy().copy(),
            "max_divergence":  self.max_divergence.cpu().numpy().copy(),
            "cfl_number":      self.cfl_number.cpu().numpy().copy(),
        }
        with lock:
            with h5py.File(h5_path, "w") as f:
                for name, arr in data.items():
                    f.create_dataset(name, data=arr)
        logger.info("Saved flow diagnostics to %s", h5_path)

# Module-level force kernels and method implementations are now
# in lilytorch/src/forces.py (item #8). Re-import the kernels so
# torch.compile setup in FluidSolver.__init__ continues to work
# unchanged via the module-local names.
from lilytorch.src.forces import (
    _forces_shared_3d, _forces_body_integrate_3d,
    _forces_body_batch_3d,
    _forces_shared_2d, _forces_body_batch_2d,
)
from lilytorch.src import forces, extras

# Pre-built dicts for releasing BDIM intermediate tensors in the FSI hot path.
# Used by _fluid_step_3d and check_explosion via __dict__.update().
_FS_FREE_AFTER_BDIM_3D = {
    'mu1_all_u': None, 'mu1_all_v': None, 'mu1_all_w': None,
    'normal_x_u': None, 'normal_y_u': None, 'normal_z_u': None,
    'normal_x_v': None, 'normal_y_v': None, 'normal_z_v': None,
    'normal_x_w': None, 'normal_y_w': None, 'normal_z_w': None,
    'mu1_all': None, 'm_m0_all': None,
}
_FS_FREE_AFTER_VAR_DENS_3D = {
    'mu0_all_u': None, 'mu0_all_v': None, 'mu0_all_w': None,
    'mu0_all': None,
}


class FluidSolver(PlottingMixin):
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

    def __init__(self, pars, dtype=None, custom_update=None, compute_forces=True):
        """
        BDIM2 solver for fluid structure interaction.

        Parameters
        ----------
        pars : dict
            Full parameter dict.  ``pars['solver']['dtype']`` may be one of
            ``"float32"`` / ``"float64"`` (or the equivalent torch dtypes)
            and is honoured when the ``dtype`` keyword argument is left as
            ``None`` (the default).
        dtype : torch.dtype or str or None
            Explicit override for the floating-point precision.  When
            ``None`` (default) the value is read from
            ``pars['solver']['dtype']`` if present, otherwise falls back
            to ``torch.float32``.  Accepts ``"float32"`` / ``"float64"``
            strings as a convenience for YAML-driven configs.
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

        # ---- Resolve dtype: explicit kwarg > YAML > float32 default ----
        # The Python kwarg takes precedence so existing callers that pass
        # ``dtype=torch.float64`` keep working unchanged.  Otherwise the
        # YAML key ``solver.dtype`` is consulted (matching BDIMhandler's
        # behaviour) so that ``base_sim_config.dtype = "float64"`` alone
        # is sufficient to switch the entire solver to double precision.
        if dtype is None:
            dtype = solver.get("dtype", "float32")
        if isinstance(dtype, str):
            _dtype_map = {"float32": torch.float32, "float64": torch.float64,
                          "double": torch.float64, "single": torch.float32}
            if dtype not in _dtype_map:
                raise ValueError(
                    f"Unknown dtype string '{dtype}'. "
                    f"Expected one of {sorted(_dtype_map)}."
                )
            dtype = _dtype_map[dtype]
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"FluidSolver dtype must be torch.float32 or torch.float64, "
                f"got {dtype}."
            )
        self.dtype = dtype
        print(f"Using dtype: {self.dtype}")
        self.nx    = solver["Nx"]+2
        self.ny    = solver["Ny"]+2

        self.xmin  = solver["xmin"]
        self.xmax  = solver["xmax"]
        self.ymin  = solver["ymin"]
        self.ymax  = solver["ymax"]

        self.dx=(self.xmax-self.xmin)/(self.nx-2)
        self.dy=(self.ymax-self.ymin)/(self.ny-2)

        assert abs(float(self.dx-self.dy)) < 1e-10, "Grid spacing in x = {} and y = {} must be equal".format(self.dx, self.dy)
        self.h = torch.tensor(self.dx, device=self.device, dtype=self.dtype)

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

        self.eps  = solver.get("eps_multiplier",
                                torch.tensor(2.0, device=self.device, dtype=self.dtype)) * self.h

        self.starting_iteration      = solver.get("starting_iteration", 0)
        self.starting_iteration_path = solver.get("starting_iteration_path", None)
        self.starting_time           = self.starting_iteration * self.dt

        self.perturbation_amplitude  = solver.get("perturbation_amplitude", 0.0)

        # ============= Smagorinsky LES model =============
        self.smagorinsky_cs = solver.get("smagorinsky_cs", 0.0)
        self.use_smagorinsky = self.smagorinsky_cs > 0
        if self.use_smagorinsky:
            print(f"Smagorinsky LES model enabled: Cs = {self.smagorinsky_cs}")

        # ============= Carreau non-Newtonian model =============
        carreau = solver.get("carreau", None)
        if carreau is not None:
            self.use_carreau = True
            self.carreau_nu_0   = carreau["nu_0"]
            self.carreau_nu_inf = carreau["nu_inf"]
            self.carreau_lam    = carreau["lam"]
            self.carreau_n      = carreau["n"]
            self.carreau_tau_y  = carreau.get("tau_y", 0.0)
            # Diffusion CFL stability limit: Δt < h² / (2·ndim·ν_max)
            # → ν_max = safety · h² / (2·ndim·Δt),  safety = 0.4
            cfl_nu_max = 0.4 * float(self.h)**2 / (2.0 * self.ndim * float(self.dt))
            self.carreau_nu_max = carreau.get("nu_max", cfl_nu_max)
            model_name = "Herschel-Bulkley–Carreau" if self.carreau_tau_y > 0 else "Carreau"
            print(f"{model_name} model enabled: "
                  f"nu_0={self.carreau_nu_0}, nu_inf={self.carreau_nu_inf}, "
                  f"lam={self.carreau_lam}, n={self.carreau_n}")
            if self.carreau_tau_y > 0:
                print(f"  yield stress tau_y={self.carreau_tau_y} Pa, "
                      f"nu_max={self.carreau_nu_max:.4e} (CFL limit={cfl_nu_max:.4e})")
            if self.use_smagorinsky:
                raise ValueError("Cannot use both Smagorinsky and Carreau simultaneously.")
            # Override self.nu to nu_inf so that the nu_t pathway
            # (nu_eff = self.nu + nu_t) yields nu_t >= 0 everywhere.
            # The user-supplied "nu" in the config is ignored when Carreau is
            # active — the zero-shear and infinite-shear viscosities are
            # specified entirely by the Carreau parameters.
            if float(self.nu) != self.carreau_nu_inf:
                print(f"  [Carreau] Overriding solver nu={float(self.nu):.6e} "
                      f"→ nu_inf={self.carreau_nu_inf:.6e} for consistency.")
                self.nu   = torch.tensor(self.carreau_nu_inf, device=self.device, dtype=self.dtype)
                self.visc = self.nu * self.rho
        else:
            self.use_carreau = False

        # Whether any variable-viscosity model is active
        self.use_variable_viscosity = self.use_smagorinsky or self.use_carreau
        # Cached spatially-varying ν·ρ field for force computation
        # (set each step when use_variable_viscosity is True, else None)
        self._nu_rho_field = None

        # ============= Yield-stress damping =============
        # Implicit penalty that drives velocity toward zero in unyielded
        # (low-shear) regions, mimicking the solid-like behaviour of a
        # yield-stress fluid.  Enabled only when the user explicitly sets
        # a "yield_damping" dict in the solver config:
        #   yield_damping:
        #     gamma_c  : 0.0625   # critical strain rate [1/s]
        #     strength : 1.1      # max damping coefficient σ_max [1/s]
        yield_damping_cfg = solver.get("yield_damping", None)
        if yield_damping_cfg is not None:
            self.use_yield_damping = True
            self._yield_gamma_c  = yield_damping_cfg["gamma_c"]
            self._yield_strength = yield_damping_cfg["strength"]
            print(f"Yield-stress damping enabled: "
                  f"gamma_c={self._yield_gamma_c:.4f} s^-1, "
                  f"strength={self._yield_strength:.4f} s^-1")
        else:
            self.use_yield_damping = False

        # ============= Sponge / damping layer =============
        sponge = solver.get("sponge", None)
        if sponge is not None:
            self.use_sponge = True
            sponge_width    = sponge.get("width", 0.15)    # [m] thickness of the sponge layer
            sponge_strength = sponge.get("strength", 50.0)  # [1/s] max damping coefficient σ_max
            sponge_axes     = sponge.get("axes", None)      # None = all axes, or list e.g. ["x"]
            # Build quadratic σ(x,y,z) fields on each staggered grid.
            # σ = σ_max · (max(0, Ls - d) / Ls)²
            # where d = distance from the nearest domain boundary.
            self._sponge_sigma_u, self._sponge_sigma_v, self._sponge_sigma_w = \
                self._build_sponge_fields(sponge_width, sponge_strength, axes=sponge_axes)
            axes_str = ",".join(sponge_axes) if sponge_axes else "all"
            print(f"Sponge layer enabled: width={sponge_width} m, strength={sponge_strength} 1/s, axes={axes_str}")
        else:
            self.use_sponge = False
            self._sponge_sigma_u = None
            self._sponge_sigma_v = None
            self._sponge_sigma_w = None

        # ============= time integration =============
        self.time_integration = solver.get("time_integration", "euler")
        assert self.time_integration in ("heun", "euler"), \
            f"Unknown time_integration '{self.time_integration}'. Choose 'heun' or 'euler'."
        # ---- time integration dispatch (set once, used every step) ----
        self._solve = self.solve_heun if self.time_integration == "heun" else self.solve_euler

        print("Setting dt={}s, dx={}".format(self.dt, self.h))
        print(f"Time integration: {self.time_integration}")

        # ---- FSI variable-density and explosion-detection state ----
        self.rho_body = float(solver.get("rho_body", 1000.0))
        _heun_flag = solver.get("heun", None)
        if _heun_flag is not None:
            self._fsi_use_heun = bool(_heun_flag) and (self.ndim == 2)
        else:
            self._fsi_use_heun = (self.time_integration == "heun") and (self.ndim == 2)
        self.terminate = False

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

        _u_inlet = float(self.adv_diff_solver.BC_values_u[1])
        self._vmax_abort = float(solver.get("vmax_abort", max(100.0 * abs(_u_inlet), 100.0)))

        # ---- optional torch.compile for adv-diff + BDIM kernels -----
        self._compile_adv_diff = solver.get("compile_adv_diff", False)
        if self._compile_adv_diff and self.device.type == "cuda":
            self.adv_diff_solver.solve = torch.compile(
                self.adv_diff_solver.solve, mode="reduce-overhead",
            )
            self._bdim_meta_compiled = torch.compile(
                FluidSolver._bdim_meta, mode="reduce-overhead",
            )
            # Dynamic-shape variant for the union-AABB crop path
            # (sub-block shape varies with body kinematics).
            self._bdim_meta_dyn_compiled = torch.compile(
                FluidSolver._bdim_meta, dynamic=True,
            )
            print("  [compile] adv_diff_solver.solve + BDIM meta-equation compiled (reduce-overhead)")
        else:
            self._bdim_meta_compiled = FluidSolver._bdim_meta
            self._bdim_meta_dyn_compiled = FluidSolver._bdim_meta

        # ---- optional Towers (2008) 2nd-order delta correction -----------
        # When force_delta_order=2, the smoothed delta is divided by |∇SDF|
        # so that the volume integral gives the correct surface measure even
        # when the numerical SDF deviates from unit gradient.
        # For analytical bodies |∇SDF|=1 exactly, so order 2 is a no-op;
        # it matters for mesh bodies or near geometric corners.
        self.force_delta_order = int(solver.get("force_delta_order", 1))
        if self.force_delta_order not in (1, 2):
            raise ValueError(f"force_delta_order must be 1 or 2, got {self.force_delta_order}")

        # ---- optional torch.compile for force computation -----
        self._compile_forces = solver.get("compile_forces", False)

        # =====================================================================
        # Solver mode: pure-Python  vs  C++/CUDA kernel path
        # =====================================================================
        # ``use_kernels`` is the SINGLE user-facing switch that selects
        # between the two solver variants.  It is independent of
        # ``use_gpu`` (which selects the torch device).
        #
        #   * ``use_kernels = False`` -- pure-Python / pure-PyTorch path.
        #     Suboptimal but reference: no batching, no per-body cropping,
        #     no streaming fused kernels.  Works on CPU and CUDA.  All
        #     four ``compile_*`` flags (compile_adv_diff, compile_forces,
        #     compile_sdf, poisson_compile) remain independent toggles.
        #
        #   * ``use_kernels = True`` -- streaming C++/CUDA kernels path.
        #     Activates the per-body streaming SDF + fused force kernels,
        #     the union-AABB crops for shared stress / mu / normals /
        #     BDIM meta-equation, and the per-body custom trilinear
        #     samplers.  Requires the compiled ``lilytorch.src.kernels``
        #     extension to be available.
        #
        # All previously-individual variant flags
        # (force_narrow_band, force_narrow_batch, force_shared_union,
        # mu_normals_union, bdim_union, streaming_sdf_3d,
        # streaming_forces_3d, streaming_sdf_2d, streaming_forces_2d)
        # are removed as user-facing keys; the corresponding internal
        # ``self._...`` attributes are derived directly from
        # ``use_kernels`` here so that downstream dispatch in
        # ``forces.py`` and ``BDIMhandler.py`` keeps working unchanged.
        self._use_kernels = bool(solver.get("use_kernels", True))
        _uk = self._use_kernels
        # Shared-stress union-AABB crop.
        self._forces_shared_union = _uk
        # mu + normals union-AABB crop.
        self._mu_normals_union = _uk
        self._mu_union_ready = False   # persistent buffers allocated lazily
        # BDIM meta-equation union-AABB crop.
        self._bdim_union = _uk
        # Phase B (3-D streaming SDF) and Phase D (3-D fused forces).
        self._streaming_sdf_3d = _uk
        self._streaming_forces_3d = _uk
        # Fused Phase C+D: SDF + inline lagged force in one kernel pass.
        # Eliminates sparse_cc_flat and union-AABB stress tensors.
        # Disabled when use_kernels=False or fused_sdf_forces=False.
        self._fused_sdf_forces_3d = _uk and bool(
            solver.get("fused_sdf_forces", True)
        )
        # The streaming 3-D path requires the per-body C++/CUDA trilinear
        # samplers (built in ``BDIMhandler._init_custom_trilinear_3d``).
        self._custom_trilinear_3d = _uk
        # 2-D analogues (streaming SDF + fused forces).
        self._streaming_sdf_2d = _uk
        self._streaming_forces_2d = _uk
        # Body-SDF sampling method used inside the streaming C++/CUDA
        # kernels (``streaming_sdf_min_3d`` / ``..._multi``):
        #   * ``"trilinear"`` (default) -- 2x2x2 stencil, matches the
        #     historical behaviour;
        #   * ``"triquadratic"`` -- 3x3x3 Lagrange stencil for higher-order
        #     SDF accuracy near the body surface; falls back to trilinear
        #     in the boundary layer of the body grid.
        # Accepts either the string form or an int (0 / 1) for
        # convenience from CLI / YAML configs.
        _interp_raw = solver.get("sdf_interp_method", "trilinear")
        if isinstance(_interp_raw, str):
            _interp_map = {"trilinear": 0, "triquadratic": 1}
            if _interp_raw not in _interp_map:
                raise ValueError(
                    f"sdf_interp_method must be one of {list(_interp_map)} "
                    f"(got {_interp_raw!r})"
                )
            self._sdf_interp_method = _interp_map[_interp_raw]
        else:
            self._sdf_interp_method = int(_interp_raw)
            if self._sdf_interp_method not in (0, 1):
                raise ValueError(
                    "sdf_interp_method int form must be 0 (trilinear) or "
                    f"1 (triquadratic); got {self._sdf_interp_method}"
                )
        # Lazy-allocated per-body compiled-wrapper plumbing.
        if self._compile_forces and self.device.type == "cuda":
            # 3-D kernels
            self._forces_shared_compiled = torch.compile(
                _forces_shared_3d, mode="reduce-overhead",
            )
            # ``_forces_body_integrate_3d`` runs on per-body AABB sub-blocks
            # whose shapes change slowly as the body rotates.  Use
            # ``dynamic=True`` so we get kernel fusion without a recompile
            # for every orientation (CUDA-graph mode is incompatible with
            # variable shapes).
            self._forces_body_compiled = torch.compile(
                _forces_body_integrate_3d, dynamic=True,
            )
            self._forces_body_batch_compiled = torch.compile(
                _forces_body_batch_3d, mode="reduce-overhead",
            )
            # Dynamic-shape shared kernel for the union-AABB crop path
            # (sub-block shape varies with body kinematics).
            self._forces_shared_dyn_compiled = torch.compile(
                _forces_shared_3d, dynamic=True,
            )
            # Dynamic-shape batched mu/normals kernel for the union-AABB
            # crop path (same rationale).
            self._mu_normals_batched_3d_dyn_compiled = torch.compile(
                _mu_normals_batched_3d, dynamic=True,
            )
            # 2-D kernels
            self._forces_shared_2d_compiled = torch.compile(
                _forces_shared_2d, mode="reduce-overhead",
            )
            self._forces_body_batch_2d_compiled = torch.compile(
                _forces_body_batch_2d, mode="reduce-overhead",
            )
            print(
                "  [compile] forces_shared + forces_body_batch compiled "
                "(reduce-overhead, 2D+3D)"
                + ("  [use_kernels=True: streaming SDF + Phase D forces + "
                   "shared-stress / mu-normals / BDIM-meta union crops "
                   "(2D+3D)]"
                   if self._use_kernels else
                   "  [use_kernels=False: pure-PyTorch path]")
            )
        else:
            self._forces_shared_compiled = _forces_shared_3d
            self._forces_body_compiled = _forces_body_integrate_3d
            self._forces_body_batch_compiled = _forces_body_batch_3d
            self._forces_shared_dyn_compiled = _forces_shared_3d
            self._mu_normals_batched_3d_dyn_compiled = _mu_normals_batched_3d
            self._forces_shared_2d_compiled = _forces_shared_2d
            self._forces_body_batch_2d_compiled = _forces_body_batch_2d

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
            smoother        = solver.get("poisson_smoother", "rbgs"),
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
                method="quadratic",
                fill_value=None
            )
            self.force_y_interp = RegularGridInterpolator(
                (self.x, self.grids.y_stag),
                torch.zeros_like(self.Y, device=self.device, dtype=self.dtype),
                method="quadratic",
                fill_value=None
            )

            self.interp_utility = RegularGridInterpolator(
                (self.x,self.y),
                torch.zeros_like(self.X, device=self.device, dtype=self.dtype),
                method="quadratic",
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
        # Torques about each body's COM (scalar in 2-D, 3-vector in 3-D).
        # Populated only in 3-D BDIM paths for now.
        self.viscous_torque_record  = torch.zeros((self.n_bodies,n_torque_comp,self.nt),device=self.device,dtype=self.dtype)
        self.pressure_torque_record = torch.zeros((self.n_bodies,n_torque_comp,self.nt),device=self.device,dtype=self.dtype)

        # ===== flow diagnostics (energy, enstrophy, CFL) =====
        _diag_every  = solver.get("diagnostics_every", 1)
        _energy_grow = solver.get("energy_growth_factor", 10.0)
        self.diagnostics = FlowDiagnostics(
            nt      = self.nt,
            ndim    = self.ndim,
            h       = self.h,
            device  = self.device,
            dtype   = self.dtype,
            check_every          = _diag_every,
            energy_growth_factor = _energy_grow,
        )

        # NOTE: xstress_tensor, ystress_tensor, zstress_tensor, and div
        # are created on-the-fly in forces_method* and project() respectively
        # (no pre-allocation needed — they are rebound, not written in-place).

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

          # ===== create folder for frames' storage ====
        self.save_frames      = output["save_frames"]
        self.save_every       = output["save_every"]
        # Unified save flag (replaces old save_uv + save_vtk).
        # Backward compat: accept legacy keys from old YAML configs.
        self.save             = output.get("save",
                                    output.get("save_uv", False)
                                    or output.get("save_vtk", False))
        # vmin/vmax: a number → fixed colour limits; "auto" → auto-scale per field
        _vmin = output["vmin"]
        _vmax = output["vmax"]
        self.vmin = None if _vmin == "auto" else _vmin
        self.vmax = None if _vmax == "auto" else _vmax
        self.plot_specs = self._resolve_plot_specs(output.get("plot_specs"))
        self.iso_3d_specs = self._resolve_iso_3d_specs(
            output.get("iso_3d_specs"),
            global_iso_value=output.get("iso_3d_value"),
        )
        self.n_quiver_spacing = 2**3

        # Background thread pool for async I/O (saving + plotting)
        self._io_executor = ThreadPoolExecutor(max_workers=2)
        self._io_futures = []  # track pending I/O tasks
        self._hdf5_lock  = threading.Lock()  # serialise HDF5 writes

        if self.save_frames or self.save:
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

        u0 = torch.tensor(np.load(u0_path)).to(device=self.device, dtype=self.dtype)
        v0 = torch.tensor(np.load(v0_path)).to(device=self.device, dtype=self.dtype)
        p0 = torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype)

          # Verify shape
        assert u0.shape == tuple(self.grid_shape), f"u0 shape: {u0.shape} != {self.grid_shape}"
        assert v0.shape == tuple(self.grid_shape), f"v0 shape: {v0.shape} != {self.grid_shape}"

          # Loaded
        self.u0, self.v0, self.p0 = u0, v0, p0
        if self.ndim == 3:
            w0_path = f'{self.starting_iteration_path}/uv_field/w_{self.starting_iteration}.npy'
            if os.path.exists(w0_path):
                self.w0 = torch.tensor(np.load(w0_path)).to(device=self.device, dtype=self.dtype)
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
        # and artificial vorticity at t=0.  Each staggered velocity component
        # is masked with its own staggered-grid mu0 for consistency with the
        # rest of the BDIM pipeline.
        if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
            cb = self.composite_body
            mu0_u, _ = cb.mu_funcs(cb.sdf_val_u)
            mu0_v, _ = cb.mu_funcs(cb.sdf_val_v)
            self.u0 = self.u0 * mu0_u
            self.v0 = self.v0 * mu0_v
            if self.ndim == 3:
                mu0_w, _ = cb.mu_funcs(cb.sdf_val_w)
                self.w0 = self.w0 * mu0_w


    # --- moved to lilytorch/src/forces.py (item #8) ---
    forces_method1 = forces.forces_method1
    forces_method2 = forces.forces_method2
    forces_method2_3d = forces.forces_method2_3d




    def project(self, u, v, p, w_vel=None, w=1.0, *,
                ch=None, cv=None, cw=None, ch_cc=None):
        """Pressure-Poisson projection.

        Parameters
        ----------
        u, v, p : tensors
            Velocity & pressure fields.
        w_vel : tensor or None
            z-velocity (3-D only).
        w : float
            Heun weight (1.0 = predictor, 0.5 = corrector).
        ch, cv, cw : tensor or None
            Pre-computed Poisson coefficients for each staggered grid
            (``dt / rho_eff`` on the respective face grids).
            When *None* (default), the standard BDIM coefficients
            ``(w*dt/rho) * mu0`` are used.  Pass custom coefficients
            for variable-density formulations (e.g. FARMS coupling where
            ``ch = dt / (rho_body + drho * mu0_u)``).
        ch_cc : tensor or None
            Cell-centred coefficient ``dt / rho_eff_cc`` for the FFT
            Poisson RHS.  When provided the FFT path solves
            ``∇²p = div / ch_cc`` (i.e. ``div * rho_eff_cc / dt``) and
            then corrects using the staggered *ch/cv/cw*.  When *None*
            (default) the FFT path falls back to a single scalar
            coefficient (constant-density behaviour).
        """

        # for general deforming bodies
        if self.ndim == 2:
            self.div  = self.divergence(u, v)
        else:
            self.div  = self.divergence(u, v, w_vel)

        coeff = w * self.dt / self.rho

        if self.poisson_method == "fft":
            # ---- FFT solver ----
            if ch_cc is not None:
                # Variable-density path: RHS uses cell-centred density,
                # correction uses the staggered ch/cv/cw coefficients.
                p = self.poisson_solverFFT.solve(self.div / ch_cc)
                if self.ndim == 2:
                    (p_x, p_y) = self.gradient(p)
                    u = u - ch * p_x
                    v = v - cv * p_y
                else:
                    (p_x, p_y, p_z) = self.gradient(p)
                    u     = u - ch * p_x
                    v     = v - cv * p_y
                    w_vel = w_vel - cw * p_z
            else:
                # Constant-density fallback: single scalar coefficient.
                # If ch was provided, use it as the scalar (backward compat).
                fft_coeff = coeff if ch is None else ch
                p = self.poisson_solverFFT.solve(self.div / fft_coeff)
                if self.ndim == 2:
                    (p_x, p_y) = self.gradient(p)
                    u = u - fft_coeff * p_x
                    v = v - fft_coeff * p_y
                else:
                    (p_x, p_y, p_z) = self.gradient(p)
                    u     = u - fft_coeff * p_x
                    v     = v - fft_coeff * p_y
                    w_vel = w_vel - fft_coeff * p_z
        else:
            # ---- Multigrid / MGCG solver (variable-coefficient Poisson) ----
            has_custom_coeffs = any(arr is not None for arr in (ch, cv, cw))
            if ch is None:
                ch = coeff * self.mu0_all_u
            if cv is None:
                cv = coeff * self.mu0_all_v

            # Select solve method: MGCG or standalone multigrid
            _poisson_solve = (self.poisson_solver.solve_mgcg
                              if self.poisson_method == "mgcg"
                              else self.poisson_solver.solve_multigrid)

            # Variable-density custom coefficients are coupled to a moving
            # immersed geometry; reusing the previous pressure field can carry
            # stale body-interior/interface values and destabilize the solve.
            if self.poisson_warm_start and not has_custom_coeffs:
                p0 = p
            else:
                p0 = torch.zeros_like(p)

            if self.ndim == 2:
                p, _ = _poisson_solve(
                    self.div[1:-1,1:-1],
                    p0,
                    ch = ch[1:,1:-1],
                    cv = cv[1:-1,1:],
                )
                # ====== projection step ======
                (p_x, p_y) = self.gradient(p)
                u          = u - ch * p_x
                v          = v - cv * p_y
            else:
                if cw is None:
                    cw = coeff * self.mu0_all_w
                p, _ = _poisson_solve(
                    self.div[1:-1, 1:-1, 1:-1],
                    p0,
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



    # --- moved to lilytorch/src/extras.py (item #4) ---
    _build_sponge_fields = extras._build_sponge_fields
    apply_sponge_damping = extras.apply_sponge_damping
    apply_yield_damping = extras.apply_yield_damping
    _compute_smagorinsky_nu_t = extras._compute_smagorinsky_nu_t
    _compute_carreau_nu_t = extras._compute_carreau_nu_t
    _compute_nu_t = extras._compute_nu_t
    _compute_nu_rho_for_forces = extras._compute_nu_rho_for_forces

    @staticmethod
    def _bdim_meta(
        phi, mu0, body_vel, mu1, *normals_and_extras,
    ):
        """BDIM2 meta-equation (compilable, elementwise).

        phi_out = mu0 * (phi - body_vel) + body_vel
                  + mu1 * normal_derivative(phi - body_vel, ...)
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

    # ------------------------------------------------------------------
    #   3-D BDIM apply with optional union-AABB narrow band
    # ------------------------------------------------------------------
    def _bdim_apply_3d(self, phi, mu0, body_vel, mu1,
                      normal_x, normal_y, normal_z):
        """Apply the BDIM2 meta-equation to a single 3-D staggered grid.

        When ``self._bdim_union`` is on AND a union AABB is available
        (cached on ``self._bdim_union_aabb``), only the union sub-block
        is touched.  Outside the union mu0=1, mu1=0, body_vel=0 makes the
        meta-equation the identity, so phi[outside] is left unchanged
        (we slice-write the cropped result back into phi).

        Otherwise falls back to the full-grid CUDA-graph kernel.
        Returns a tensor that the caller can safely consume; the caller
        is responsible for cloning if it needs to keep a reference past
        the next CUDA-graph replay (full-grid path) — in the union path
        the returned tensor is the input ``phi`` itself (mutated in
        place), which is already an owned tensor.
        """
        _h = self.h
        if self._bdim_union and getattr(self, '_bdim_union_aabb', None) is not None:
            ui0, ui1, uj0, uj1, uk0, uk1 = self._bdim_union_aabb
            usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))
            sub = self._bdim_meta_dyn_compiled(
                phi[usl].contiguous(),
                mu0[usl].contiguous(),
                body_vel[usl].contiguous(),
                mu1[usl].contiguous(),
                normal_x[usl].contiguous(),
                normal_y[usl].contiguous(),
                normal_z[usl].contiguous(),
                _h, 3,
            )
            phi[usl] = sub
            return phi
        return self._bdim_meta_compiled(
            phi, mu0, body_vel, mu1,
            normal_x, normal_y, normal_z, _h, 3,
        ).clone()

    # ------------------------------------------------------------------
    #   Union AABB across all body sub-blocks (3-D)
    # ------------------------------------------------------------------
    def _compute_union_aabb_3d(self, halo=2, bucket=16):
        """Return (i0,i1,j0,j1,k0,k1) union AABB over all body sparse
        SDFs, expanded by ``halo`` cells and clipped to grid extent.
        Returns ``None`` if any body lacks a sparse AABB.

        When ``bucket > 1`` each extent (i1-i0, j1-j0, k1-k0) is rounded
        up to a multiple of ``bucket`` by expanding the high side first
        and, if we hit the grid boundary, the low side.  This stabilizes
        the sub-block shape to a small discrete set so that
        ``dynamic=True`` compiled kernels only pay the recompile cost a
        bounded number of times (once per bucket combination seen during
        warmup) instead of every time the swimmer deforms.
        """
        comp = self.composite_body
        sparse = getattr(comp, '_sdf_sparse', None)
        u_i0 = u_j0 = u_k0 = 1 << 30
        u_i1 = u_j1 = u_k1 = -1
        if not sparse or sparse[0] is None:
            # Fused SDF+forces path does not populate _sdf_sparse; instead it
            # stores the union AABB directly so the cheap sub-block path can
            # still activate without the CC-SDF per-body slabs.
            raw = getattr(comp, '_fused_union_aabb', None)
            if raw is None:
                return None
            u_i0, u_i1, u_j0, u_j1, u_k0, u_k1 = raw
        else:
            for entry in sparse:
                if entry is None:
                    return None
                aabb_i = entry[0]
                if aabb_i is None:
                    return None
                i0, i1, j0, j1, k0, k1 = aabb_i
                if i0 < u_i0: u_i0 = i0
                if j0 < u_j0: u_j0 = j0
                if k0 < u_k0: u_k0 = k0
                if i1 > u_i1: u_i1 = i1
                if j1 > u_j1: u_j1 = j1
                if k1 > u_k1: u_k1 = k1
        Ni, Nj, Nk = comp.sdf_val.shape
        u_i0 = max(0, u_i0 - halo); u_i1 = min(Ni, u_i1 + halo)
        u_j0 = max(0, u_j0 - halo); u_j1 = min(Nj, u_j1 + halo)
        u_k0 = max(0, u_k0 - halo); u_k1 = min(Nk, u_k1 + halo)

        if bucket is not None and bucket > 1:
            def _pad(lo, hi, N, b):
                extent = hi - lo
                target = ((extent + b - 1) // b) * b
                if target > N:
                    target = N
                pad = target - extent
                # Expand high side first, then spill to low side if clipped.
                new_hi = hi + pad
                if new_hi > N:
                    over = new_hi - N
                    new_hi = N
                    lo = max(0, lo - over)
                return lo, new_hi
            u_i0, u_i1 = _pad(u_i0, u_i1, Ni, bucket)
            u_j0, u_j1 = _pad(u_j0, u_j1, Nj, bucket)
            u_k0, u_k1 = _pad(u_k0, u_k1, Nk, bucket)

        # Cropping is only a net win when the union AABB covers a small
        # fraction of the grid: each BDIM apply pays a fixed launch-
        # overhead cost (7 .contiguous() slice copies + 1 slice-assign,
        # ~9 kernel launches) that only beats the full-grid kernel when
        # the saved kernel work exceeds ~9 × launch_overhead. Empirically
        # the break-even point is around 50 % of the full volume; above
        # that, return None so the caller falls back to the full-grid
        # compiled kernel.
        sub_vol  = (u_i1 - u_i0) * (u_j1 - u_j0) * (u_k1 - u_k0)
        full_vol = Ni * Nj * Nk
        if sub_vol > 0.5 * full_vol:
            return None

        return (u_i0, u_i1, u_j0, u_j1, u_k0, u_k1)

    def solver_iteration_heun(self, u, v, p, iteration, w_vel=None):
        """
        Heun (RK2 predictor-corrector) time integration with BDIM2.

        1. **Predictor** (w = 1):
            adv-diff on u^n → BDIM → project → BCs → u_pred (div-free)

        2. **Corrector** (w = 0.5):
            adv-diff on u_pred → rebase from u^n → BDIM →
            average with **projected** predictor → project(w=0.5)

        The corrector average is ``0.5*(u_pred + BDIM(u^n + dt·RHS(u_pred)))``.
        Because ``u_pred`` is divergence-free, ``div(u_avg)`` equals half
        the corrector's BDIM divergence.  ``w=0.5`` compensates this
        halving in the Poisson coefficient so that the stored pressure
        equals the physical dynamic pressure (needed for correct force
        computation).  The velocity correction is ``w``-independent.
        """

        if self.ndim == 2:
            # ====== PREDICTOR ======
            nu_t = self._compute_nu_t(u, v)
            (uprime, vprime) = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()

            # BDIM2 meta-equation (fused when compiled)
            _bdim = self._bdim_meta_compiled
            _h    = self.h
            uprime = _bdim(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vprime = _bdim(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()

            self.adv_diff_solver.set_BCs(uprime, vprime)
            (u1, v1, p1) = self.project(uprime, vprime, p)
            # Re-apply BCs after projection
            # BC!(a.u,...) after every project! call).
            self.adv_diff_solver.set_BCs(u1, v1)

            # ====== CORRECTOR ======
            # Evaluate RHS at the projected predicted velocity
            nu_t = self._compute_nu_t(u1, v1)
            (uprime2, vprime2) = self.adv_diff_solver.solve(u1, v1, nu_t=nu_t)
            # adv_diff.solve returns u1 + dt*RHS(u1).
            # Rebase from u^n: u^n + dt*RHS(u_pred), matching
            # with u⁰ = u^n (saved once at the top of mom_step!).
            uprime2 = u + (uprime2 - u1)
            vprime2 = v + (vprime2 - v1)

            # BDIM2 meta-equation on corrector (fused when compiled)
            uprime2 = _bdim(
                uprime2, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vprime2 = _bdim(
                vprime2, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()

            # Average the PROJECTED predictor with the corrector's BDIM
            # halves  u_pred + BDIM(u^n + dt*RHS(u_pred)).
            u_avg = 0.5 * (u1 + uprime2)
            v_avg = 0.5 * (v1 + vprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg)
            # w=0.5: the corrector average halves the divergence (u_pred
            # is div-free), so w=0.5 doubles the Poisson coefficient to
            # recover the physical pressure.  Velocity correction is
            # w-independent; only the stored pressure changes.
            (u_out, v_out, p_out) = self.project(u_avg, v_avg, p, w=0.5)

            # Sponge damping (2-D)
            if self.use_sponge:
                (u_out, v_out) = self.apply_sponge_damping(u_out, v_out)

            # Yield-stress damping (2-D)
            if self.use_yield_damping:
                (u_out, v_out) = self.apply_yield_damping(u_out, v_out)

            return (u_out, v_out, p_out)

        else:  # 3D
            # ====== PREDICTOR ======
            nu_t = self._compute_nu_t(u, v, w_vel)
            (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()
            wprime = wprime.clone()

            # Cache union AABB for both BDIM passes (predictor + corrector).
            # Cleared at end of step.  Cheap (Python loop over <~10 bodies).
            self._bdim_union_aabb = (
                self._compute_union_aabb_3d(halo=2)
                if self._bdim_union else None
            )

            # BDIM2 meta-equation (fused when compiled)
            _bdim = self._bdim_meta_compiled
            _h    = self.h
            uprime = self._bdim_apply_3d(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u,
            )
            vprime = self._bdim_apply_3d(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v,
            )
            wprime = self._bdim_apply_3d(
                wprime, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w,
            )

            self.adv_diff_solver.set_BCs(uprime, vprime, wprime)
            (u1, v1, w1, p1) = self.project(uprime, vprime, p, w_vel=wprime)
            # Re-apply BCs after projection
            self.adv_diff_solver.set_BCs(u1, v1, w1)

            # ====== CORRECTOR ======
            nu_t = self._compute_nu_t(u1, v1, w1)
            (uprime2, vprime2, wprime2) = self.adv_diff_solver.solve(u1, v1, w1, nu_t=nu_t)
            # Rebase from u^n
            uprime2 = u     + (uprime2 - u1)
            vprime2 = v     + (vprime2 - v1)
            wprime2 = w_vel + (wprime2 - w1)

            # BDIM2 meta-equation on corrector (fused when compiled)
            uprime2 = self._bdim_apply_3d(
                uprime2, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u,
            )
            vprime2 = self._bdim_apply_3d(
                vprime2, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v,
            )
            wprime2 = self._bdim_apply_3d(
                wprime2, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w,
            )

            # Drop the cached union AABB now that both BDIM passes are done.
            self._bdim_union_aabb = None

            # Free mu1 + staggered normals after both BDIM passes
            for _attr in ('mu1_all_u', 'mu1_all_v', 'mu1_all_w',
                          'normal_x_u', 'normal_y_u', 'normal_z_u',
                          'normal_x_v', 'normal_y_v', 'normal_z_v',
                          'normal_x_w', 'normal_y_w', 'normal_z_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            # Average the PROJECTED predictor (div-free) with the
            # corrector's BDIM output
            u_avg = 0.5 * (u1 + uprime2)
            v_avg = 0.5 * (v1 + vprime2)
            w_avg = 0.5 * (w1 + wprime2)

            self.adv_diff_solver.set_BCs(u_avg, v_avg, w_avg)
            # w=0.5: see 2-D comment.
            (u_out, v_out, w_out, p_out) = self.project(u_avg, v_avg, p, w_vel=w_avg, w=0.5)

            # Free mu0 after project
            for _attr in ('mu0_all_u', 'mu0_all_v', 'mu0_all_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            # Sponge damping: damp velocity near domain boundaries
            if self.use_sponge:
                (u_out, v_out, w_out) = self.apply_sponge_damping(u_out, v_out, w_out)

            # Yield-stress damping (3-D)
            if self.use_yield_damping:
                (u_out, v_out, w_out) = self.apply_yield_damping(u_out, v_out, w_out)

            return (u_out, v_out, p_out, w_out)

    def solve_heun(self, u, v, p, iteration, w_vel=None):
        if self.ndim == 2:
            return self.solver_iteration_heun(u, v, p, iteration)
        else:
            return self.solver_iteration_heun(u, v, p, iteration, w_vel=w_vel)

    def solver_iteration_euler(self, u, v, p, iteration, w_vel=None):
        """Forward Euler time integration with BDIM2.

        Single-stage scheme:
          adv-diff → BDIM → project(w=1)

        Cheaper per step than Heun (one RHS evaluation instead of two),
        but only first-order accurate in time.
        """

        _bdim = self._bdim_meta_compiled
        _h    = self.h

        if self.ndim == 2:
            nu_t = self._compute_nu_t(u, v)
            (uprime, vprime) = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()

            # BDIM2 meta-equation
            uprime = _bdim(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vprime = _bdim(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()

            self.adv_diff_solver.set_BCs(uprime, vprime)
            (u_out, v_out, p_out) = self.project(uprime, vprime, p)

            # Sponge damping (2-D)
            if self.use_sponge:
                (u_out, v_out) = self.apply_sponge_damping(u_out, v_out)

            # Yield-stress damping (2-D Euler)
            if self.use_yield_damping:
                (u_out, v_out) = self.apply_yield_damping(u_out, v_out)

            return (u_out, v_out, p_out)

        else:  # 3D
            nu_t = self._compute_nu_t(u, v, w_vel)
            (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
            # Clone CUDA-graph outputs before passing to _bdim.
            uprime = uprime.clone()
            vprime = vprime.clone()
            wprime = wprime.clone()

            # Cache union AABB for the BDIM pass.
            self._bdim_union_aabb = (
                self._compute_union_aabb_3d(halo=2)
                if self._bdim_union else None
            )

            # BDIM2 meta-equation
            uprime = self._bdim_apply_3d(
                uprime, self.mu0_all_u,
                self.composite_body.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, self.normal_z_u,
            )
            vprime = self._bdim_apply_3d(
                vprime, self.mu0_all_v,
                self.composite_body.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, self.normal_z_v,
            )
            wprime = self._bdim_apply_3d(
                wprime, self.mu0_all_w,
                self.composite_body.body_w, self.mu1_all_w,
                self.normal_x_w, self.normal_y_w, self.normal_z_w,
            )
            self._bdim_union_aabb = None

            # Free mu1 and staggered normals — no longer needed after
            # BDIM.  project() only uses mu0_{u,v,w}, and forces
            # recomputes CC normals on-the-fly.  This releases
            # 12 × grid_shape × 4 bytes ≈ 1.5 GB for typical 3-D grids.
            for _attr in ('mu1_all_u', 'mu1_all_v', 'mu1_all_w',
                          'normal_x_u', 'normal_y_u', 'normal_z_u',
                          'normal_x_v', 'normal_y_v', 'normal_z_v',
                          'normal_x_w', 'normal_y_w', 'normal_z_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            self.adv_diff_solver.set_BCs(uprime, vprime, wprime)
            (u_out, v_out, w_out, p_out) = self.project(uprime, vprime, p, w_vel=wprime)

            # Free mu0 — no longer needed after project.  Reduces peak
            # memory during force computation by another ~0.4 GB.
            for _attr in ('mu0_all_u', 'mu0_all_v', 'mu0_all_w'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

            # Sponge damping: damp velocity near domain boundaries
            if self.use_sponge:
                (u_out, v_out, w_out) = self.apply_sponge_damping(u_out, v_out, w_out)

            # Yield-stress damping (3-D Euler)
            if self.use_yield_damping:
                (u_out, v_out, w_out) = self.apply_yield_damping(u_out, v_out, w_out)

            return (u_out, v_out, p_out, w_out)

    def solve_euler(self, u, v, p, iteration, w_vel=None):
        if self.ndim == 2:
            return self.solver_iteration_euler(u, v, p, iteration)
        else:
            return self.solver_iteration_euler(u, v, p, iteration, w_vel=w_vel)

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
        # When mu/normals union crop is active, keep the persistent
        # full-grid mu/normal buffers alive across steps — they hold the
        # outside-body default values that never change, and only the
        # union sub-block is overwritten each step.
        keep = set()
        if getattr(self, '_mu_normals_union', False):
            keep = {
                'mu0_all_u', 'mu1_all_u', 'mu0_all_v', 'mu1_all_v',
                'mu0_all_w', 'mu1_all_w', 'mu0_all', 'mu1_all',
                'm_m0_all',
                'normal_x_u', 'normal_y_u', 'normal_z_u',
                'normal_x_v', 'normal_y_v', 'normal_z_v',
                'normal_x_w', 'normal_y_w', 'normal_z_w',
                'normal_x', 'normal_y', 'normal_z',
            }
        for attr in self._BDIM_FIELD_NAMES:
            if attr in keep:
                continue
            if hasattr(self, attr):
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    #   mu / normal recomputation  (shared by step_() and BDIMhandler)
    # ------------------------------------------------------------------
    def _recompute_mu_normals_2d(self):
        """Recompute mu0/mu1 and normals on u- and v-staggered grids (2-D).

        CC-grid normals are computed on-the-fly inside forces_method1/2.
        """
        comp = self.composite_body

        (self.mu0_all_u, self.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
        (self.normal_x_u, self.normal_y_u) = comp.compute_normals(comp.sdf_val_u)

        (self.mu0_all_v, self.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
        (self.normal_x_v, self.normal_y_v) = comp.compute_normals(comp.sdf_val_v)

        # CC-grid mu0 — used for smooth pressure masking in forces_method2
        (self.mu0_all, self.mu1_all) = comp.mu_funcs(comp.sdf_val)

    def _recompute_mu_normals_3d(self):
        """Recompute mu0/mu1 and normals on all staggered + CC grids (3-D).

        When ``_compile_sdf`` is enabled, uses a batched+compiled kernel that
        processes all four grids (u, v, w, CC) in a single fused CUDA graph.
        When ``_mu_normals_union`` is enabled, the kernel runs only on the
        union AABB of all body sub-blocks (with halo) and results are
        slice-written into persistent full-grid buffers pre-filled with
        the outside-body defaults (mu0=1, mu1=0, normals=0).
        """
        comp = self.composite_body

        # ------------------------------------------------------------------
        # Union-AABB crop path — outside the union SDF is _FAR so mu0=1,
        # mu1=0, normals=0 (these defaults never change between steps).
        # Uses the shared _compute_union_aabb_3d helper which rounds the
        # sub-block extents up to a bucket multiple so the dynamic-shape
        # compiled kernel only recompiles a bounded number of times.
        # ------------------------------------------------------------------
        u_aabb = None
        if self._mu_normals_union:
            u_aabb = self._compute_union_aabb_3d(halo=2, bucket=16)

        if u_aabb is not None:
            # Lazy-allocate a single packed persistent buffer of shape
            # (21, Nx, Ny, Nz) pre-filled with outside-body defaults
            # (mu0=1, mu1=0, normals=0, m_m0=0).  All downstream mu /
            # normal attributes are *views* into this packed tensor, so
            # the union sub-block can be written with a single slice
            # assignment instead of 21 separate slice-writes.
            #
            # Pack layout along dim-0:
            #    0- 3  : mu0 for [u, v, w, cc]
            #    4- 7  : mu1 for [u, v, w, cc]
            #    8-11  : normal_x for [u, v, w, cc]
            #   12-15  : normal_y for [u, v, w, cc]
            #   16-19  : normal_z for [u, v, w, cc]
            #      20  : m_m0_all  (= 1 - mu0_cc)
            if getattr(self, '_mu_pack', None) is None or \
               self._mu_pack.shape[1:] != comp.sdf_val.shape or \
               self._mu_pack.dtype != comp.sdf_val.dtype or \
               self._mu_pack.device != comp.sdf_val.device:
                pack = torch.zeros(
                    (21, *comp.sdf_val.shape),
                    device=comp.sdf_val.device, dtype=comp.sdf_val.dtype,
                )
                pack[0:4].fill_(1.0)  # mu0 defaults to 1 outside body
                self._mu_pack = pack
                self._mu_union_ready = True
            pack = self._mu_pack

            # (Re-)alias every step: cheap Python, and robust to any
            # non-union path overwriting these attributes.
            self.mu0_all_u, self.mu0_all_v, self.mu0_all_w, self.mu0_all = (
                pack[0], pack[1], pack[2], pack[3])
            self.mu1_all_u, self.mu1_all_v, self.mu1_all_w, self.mu1_all = (
                pack[4], pack[5], pack[6], pack[7])
            self.normal_x_u, self.normal_x_v, self.normal_x_w, self.normal_x = (
                pack[8], pack[9], pack[10], pack[11])
            self.normal_y_u, self.normal_y_v, self.normal_y_w, self.normal_y = (
                pack[12], pack[13], pack[14], pack[15])
            self.normal_z_u, self.normal_z_v, self.normal_z_w, self.normal_z = (
                pack[16], pack[17], pack[18], pack[19])
            self.m_m0_all = pack[20]

            ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
            usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))

            mu0_s, mu1_s, nx_s, ny_s, nz_s = self._mu_normals_batched_3d_dyn_compiled(
                comp.sdf_val_u[usl].contiguous(),
                comp.sdf_val_v[usl].contiguous(),
                comp.sdf_val_w[usl].contiguous(),
                comp.sdf_val[usl].contiguous(),
                comp.h, comp.eps,
            )

            # Fused slice-write: stack all 21 sub-block outputs along
            # dim-0 and scatter into the packed buffer with ONE assign.
            # Order must match the pack layout above.
            stacked = torch.cat(
                (mu0_s, mu1_s, nx_s, ny_s, nz_s, 1.0 - mu0_s[3:4]),
                dim=0,
            )  # (21, sub_Nx, sub_Ny, sub_Nz)
            self._mu_pack[:, ui0:ui1, uj0:uj1, uk0:uk1] = stacked
            return

        if self._compile_sdf:
            # ── Batched + compiled path: all 4 grids in one fused pass ──
            _fn = _mu_normals_batched_3d_compiled
            mu0, mu1, nx, ny, nz = _fn(
                comp.sdf_val_u, comp.sdf_val_v, comp.sdf_val_w,
                comp.sdf_val, comp.h, comp.eps,
            )
            # Clone outputs — CUDA graph buffers are overwritten on
            # subsequent replays, so we must detach before storing.
            mu0, mu1 = mu0.clone(), mu1.clone()
            nx, ny, nz = nx.clone(), ny.clone(), nz.clone()

            # Unstack: order is [u, v, w, cc]
            self.mu0_all_u, self.mu1_all_u = mu0[0], mu1[0]
            self.normal_x_u, self.normal_y_u, self.normal_z_u = nx[0], ny[0], nz[0]

            self.mu0_all_v, self.mu1_all_v = mu0[1], mu1[1]
            self.normal_x_v, self.normal_y_v, self.normal_z_v = nx[1], ny[1], nz[1]

            self.mu0_all_w, self.mu1_all_w = mu0[2], mu1[2]
            self.normal_x_w, self.normal_y_w, self.normal_z_w = nx[2], ny[2], nz[2]

            self.mu0_all, self.mu1_all = mu0[3], mu1[3]
            self.m_m0_all = 1 - self.mu0_all
            self.normal_x, self.normal_y, self.normal_z = nx[3], ny[3], nz[3]
        else:
            # ── Eager path: 4 × individual mu_funcs + compute_normals ──
            # u-grid
            (self.mu0_all_u, self.mu1_all_u) = comp.mu_funcs(comp.sdf_val_u)
            (self.normal_x_u, self.normal_y_u, self.normal_z_u) = comp.compute_normals(comp.sdf_val_u)

            # v-grid
            (self.mu0_all_v, self.mu1_all_v) = comp.mu_funcs(comp.sdf_val_v)
            (self.normal_x_v, self.normal_y_v, self.normal_z_v) = comp.compute_normals(comp.sdf_val_v)

            # w-grid
            (self.mu0_all_w, self.mu1_all_w) = comp.mu_funcs(comp.sdf_val_w)
            (self.normal_x_w, self.normal_y_w, self.normal_z_w) = comp.compute_normals(comp.sdf_val_w)

            # CC-grid (p) — cached for forces_method2_3d
            (self.mu0_all, self.mu1_all) = comp.mu_funcs(comp.sdf_val)
            self.m_m0_all = 1 - self.mu0_all
            (self.normal_x, self.normal_y, self.normal_z) = comp.compute_normals(comp.sdf_val)

    def _recompute_mu_normals(self):
        """Dispatch to 2-D or 3-D mu/normal recomputation."""
        if self.ndim == 2:
            self._recompute_mu_normals_2d()
        else:
            self._recompute_mu_normals_3d()

    # ==================================================================
    #  Variable-density FSI fluid step  (called by BDIMhandler.step)
    # ==================================================================

    def _compute_variable_density_coefficients(self, timestep):
        """Compute variable-density Poisson coefficients for FSI coupling.

        Returns ``(ch, cv, ch_cc)`` for 2-D or ``(ch, cv, cw, ch_cc)``
        for 3-D, where:

            * ``ch, cv, cw`` -- staggered ``dt / rho_eff`` on face grids.
            * ``ch_cc`` -- cell-centred ``dt / rho_eff_cc`` for FFT RHS.

        Effective density: ``rho_eff(x) = rho_body + (rho_fluid - rho_body) * mu0(x)``.

        Narrow-band fast-path (3-D, mu_normals_union active)
        ---------------------------------------------------
        Outside the union AABB ``mu0 = 1`` everywhere, so
        ``ch = cv = cw = ch_cc = dt / rho_fluid`` is constant.  Persistent
        full-grid buffers are pre-filled once with that default and only the
        union sub-block is overwritten each step, avoiding four full-grid
        divisions.
        """
        _drho = float(self.rho) - self.rho_body

        if (self.ndim == 3
                and getattr(self, '_mu_normals_union', False)
                and self.mu0_all_u is not None
                and self.mu0_all_v is not None
                and self.mu0_all_w is not None
                and self.mu0_all   is not None):
            u_aabb = self._compute_union_aabb_3d(halo=2, bucket=16)
            if u_aabb is not None:
                _dt_over_rhofluid = float(timestep / float(self.rho))
                mu0_u = self.mu0_all_u
                needs_realloc = (
                    getattr(self, '_ch_persist', None) is None
                    or self._ch_persist.shape  != mu0_u.shape
                    or self._ch_persist.dtype  != mu0_u.dtype
                    or self._ch_persist.device != mu0_u.device
                    or self._ch_outside_val    != _dt_over_rhofluid
                )
                if needs_realloc:
                    self._ch_persist     = torch.full_like(self.mu0_all_u, _dt_over_rhofluid)
                    self._cv_persist     = torch.full_like(self.mu0_all_v, _dt_over_rhofluid)
                    self._cw_persist     = torch.full_like(self.mu0_all_w, _dt_over_rhofluid)
                    self._ch_cc_persist  = torch.full_like(self.mu0_all,   _dt_over_rhofluid)
                    self._ch_outside_val = _dt_over_rhofluid

                ui0, ui1, uj0, uj1, uk0, uk1 = u_aabb
                usl = (slice(ui0, ui1), slice(uj0, uj1), slice(uk0, uk1))

                # Use per-cell winning body density when available (fused path).
                # winning_rho_cc[x] = rho_body of the body closest to x,
                # pre-filled with rho_fluid outside all bodies.
                _winning = getattr(self.composite_body, '_winning_rho_cc', None)
                if _winning is not None:
                    _rho_b_u   = _winning[usl]
                    _rho_fluid = float(self.rho)
                    self._ch_persist[usl]    = timestep / (
                        _rho_b_u * (1 - self.mu0_all_u[usl]) +
                        _rho_fluid * self.mu0_all_u[usl])
                    self._cv_persist[usl]    = timestep / (
                        _winning[usl] * (1 - self.mu0_all_v[usl]) +
                        _rho_fluid * self.mu0_all_v[usl])
                    self._cw_persist[usl]    = timestep / (
                        _winning[usl] * (1 - self.mu0_all_w[usl]) +
                        _rho_fluid * self.mu0_all_w[usl])
                    self._ch_cc_persist[usl] = timestep / (
                        _winning[usl] * (1 - self.mu0_all[usl]) +
                        _rho_fluid * self.mu0_all[usl])
                else:
                    self._ch_persist[usl]    = timestep / (self.rho_body + _drho * self.mu0_all_u[usl])
                    self._cv_persist[usl]    = timestep / (self.rho_body + _drho * self.mu0_all_v[usl])
                    self._cw_persist[usl]    = timestep / (self.rho_body + _drho * self.mu0_all_w[usl])
                    self._ch_cc_persist[usl] = timestep / (self.rho_body + _drho * self.mu0_all[usl])
                return (self._ch_persist, self._cv_persist,
                        self._cw_persist, self._ch_cc_persist)

        ch    = timestep / (self.rho_body + _drho * self.mu0_all_u)
        cv    = timestep / (self.rho_body + _drho * self.mu0_all_v)
        ch_cc = timestep / (self.rho_body + _drho * self.mu0_all)
        if self.ndim == 3:
            cw = timestep / (self.rho_body + _drho * self.mu0_all_w)
            return ch, cv, cw, ch_cc
        return ch, cv, ch_cc

    def fluid_step(self, *args):
        """One FSI fluid step (advect-BDIM-project).  Called by BDIMhandler."""
        if self.ndim == 3:
            return self._fluid_step_3d(*args)
        return self._fluid_step_2d(*args)

    def _fluid_step_2d(self, u, v, p, timestep):
        _bdim = self._bdim_meta_compiled
        _h    = self.h
        comp  = self.composite_body

        _ch, _cv, _ch_cc = self._compute_variable_density_coefficients(timestep)

        def _advect_bdim(u_in, v_in, nu_t=None, u_rebase=None, v_rebase=None):
            (up, vp) = self.adv_diff_solver.solve(u_in, v_in, nu_t=nu_t)
            up = up.clone()
            vp = vp.clone()
            if u_rebase is not None:
                up = u_rebase + (up - u_in)
            if v_rebase is not None:
                vp = v_rebase + (vp - v_in)
            up = _bdim(
                up, self.mu0_all_u,
                comp.body_u, self.mu1_all_u,
                self.normal_x_u, self.normal_y_u, _h, 2,
            ).clone()
            vp = _bdim(
                vp, self.mu0_all_v,
                comp.body_v, self.mu1_all_v,
                self.normal_x_v, self.normal_y_v, _h, 2,
            ).clone()
            self.adv_diff_solver.set_BCs(up, vp)
            return up, vp

        nu_t = self._compute_nu_t(u, v)

        if self._fsi_use_heun:
            # Heun (RK2) predictor-corrector — matches WaterLily.jl mom_step!
            uprime, vprime = _advect_bdim(u, v, nu_t=nu_t)
            u1, v1, p1 = self.project(uprime, vprime, p, ch=_ch, cv=_cv, ch_cc=_ch_cc)
            self.adv_diff_solver.set_BCs(u1, v1)

            nu_t = self._compute_nu_t(u1, v1)
            uprime2, vprime2 = _advect_bdim(u1, v1, nu_t=nu_t, u_rebase=u, v_rebase=v)

            u_avg = 0.5 * (u1 + uprime2)
            v_avg = 0.5 * (v1 + vprime2)
            self.adv_diff_solver.set_BCs(u_avg, v_avg)

            u_out, v_out, p_out = self.project(
                u_avg, v_avg, p1,
                ch=0.5 * _ch, cv=0.5 * _cv, ch_cc=0.5 * _ch_cc,
            )
        else:
            uprime, vprime = _advect_bdim(u, v, nu_t=nu_t)
            u_out, v_out, p_out = self.project(uprime, vprime, p,
                                               ch=_ch, cv=_cv, ch_cc=_ch_cc)

        if self.use_sponge:
            (u_out, v_out) = self.apply_sponge_damping(u_out, v_out)
        if self.use_yield_damping:
            (u_out, v_out) = self.apply_yield_damping(u_out, v_out)

        self.adv_diff_solver.set_BCs(u_out, v_out)
        return (u_out, v_out, p_out)

    def _fluid_step_3d(self, u, v, w, p, timestep):
        nu_t = self._compute_nu_t(u, v, w)
        (uprime, vprime, wprime) = self.adv_diff_solver.solve(u, v, w, nu_t=nu_t)
        uprime = uprime.clone()
        vprime = vprime.clone()
        wprime = wprime.clone()
        self.adv_diff_solver.set_BCs(uprime, vprime, wprime)

        self._bdim_union_aabb = (
            self._compute_union_aabb_3d(halo=2) if self._bdim_union else None
        )
        uprime = self._bdim_apply_3d(
            uprime, self.mu0_all_u,
            self.composite_body.body_u, self.mu1_all_u,
            self.normal_x_u, self.normal_y_u, self.normal_z_u,
        )
        vprime = self._bdim_apply_3d(
            vprime, self.mu0_all_v,
            self.composite_body.body_v, self.mu1_all_v,
            self.normal_x_v, self.normal_y_v, self.normal_z_v,
        )
        wprime = self._bdim_apply_3d(
            wprime, self.mu0_all_w,
            self.composite_body.body_w, self.mu1_all_w,
            self.normal_x_w, self.normal_y_w, self.normal_z_w,
        )
        self._bdim_union_aabb = None

        self.__dict__.update(_FS_FREE_AFTER_BDIM_3D)

        ch, cv, cw, ch_cc = self._compute_variable_density_coefficients(timestep)

        self.__dict__.update(_FS_FREE_AFTER_VAR_DENS_3D)

        if self.poisson_method == "fft":
            (u, v, w, p) = self.project(uprime, vprime, p,
                                        w_vel=wprime, ch=ch, cv=cv, cw=cw, ch_cc=ch_cc)
        else:
            (u, v, w, p) = self.project(uprime, vprime, p,
                                        w_vel=wprime, ch=ch, cv=cv, cw=cw)

        if self.use_sponge:
            (u, v, w) = self.apply_sponge_damping(u, v, w)
        if self.use_yield_damping:
            (u, v, w) = self.apply_yield_damping(u, v, w)

        self.adv_diff_solver.set_BCs(u, v, w)
        return (u, v, w, p)

    def check_explosion(self, iteration):
        """Abort if fluid fields are non-finite or velocities exceed _vmax_abort."""
        if self.ndim == 3:
            field_names = ("u", "v", "w", "p")
            field_arrs  = (self.u0, self.v0, self.w0, self.p0)
            n_vel       = 3
        else:
            field_names = ("u", "v", "p")
            field_arrs  = (self.u0, self.v0, self.p0)
            n_vel       = 2

        finite_flags = torch.stack([torch.isfinite(a).all() for a in field_arrs])
        vmax_per_vel = torch.stack([a.abs().amax() for a in field_arrs[:n_vel]])
        diag = torch.cat((finite_flags.to(vmax_per_vel.dtype), vmax_per_vel)).cpu().numpy()
        finite_np = diag[:len(field_arrs)]
        vmax_np   = diag[len(field_arrs):]

        for name, ok in zip(field_names, finite_np):
            if not bool(ok):
                self.terminate = True
                raise RuntimeError(
                    f"[BDIM] Fluid explosion at iteration {iteration}: "
                    f"non-finite values in field '{name}'. Likely cause: "
                    f"body intersecting a domain wall (Poisson ill-conditioned) "
                    f"or CFL violation."
                )

        vmax = float(vmax_np.max())
        if vmax > self._vmax_abort:
            self.terminate = True
            raise RuntimeError(
                f"[BDIM] Fluid explosion at iteration {iteration}: "
                f"|u|_max = {vmax:.3e} > vmax_abort = {self._vmax_abort:.3e}."
            )

    def _apply_force_feedback(self, iteration, t):
        """Advance any standalone body state that consumes solver loads.

        This is intentionally separate from the FARMS / MuJoCo path: it is
        only used by ``FluidSolver.step_()`` after viscous and pressure loads
        have been accumulated on the current standalone composite body.
        """
        callback = getattr(self.composite_body, "apply_force_feedback", None)
        if callback is None:
            return

        force_x = self.friction_force_lin_x + self.pressure_force_x
        force_y = self.friction_force_lin_y + self.pressure_force_y

        if self.ndim == 2:
            force = torch.stack((force_x, force_y), dim=-1)
            torque = self.friction_force_ang_z + self.pressure_force_ang_z
        else:
            force_z = self.friction_force_lin_z + self.pressure_force_z
            torque_x = self.friction_force_ang_x + self.pressure_force_ang_x
            torque_y = self.friction_force_ang_y + self.pressure_force_ang_y
            torque_z = self.friction_force_ang_z + self.pressure_force_ang_z
            force = torch.stack((force_x, force_y, force_z), dim=-1)
            torque = torch.stack((torque_x, torque_y, torque_z), dim=-1)

        callback(
            force=force,
            torque=torque,
            iteration=iteration,
            time=t,
            dt=self.dt,
            solver=self,
        )

    def step_(self, u, v, p, iteration, t, w_vel=None):

        # update sdf_properties
        self.composite_body.update(t, iteration, dt=self.dt)

        # --- recompute mu / normals on all staggered grids ---
        self._recompute_mu_normals()

        ##### just for plotting
        self.sdf_properties = [[self.composite_body.sdf_val_u]]

        if self.ndim == 2:
            (u, v, p) = self._solve(u, v, p, iteration)

            if self.compute_forces:
                self.forces_method2(u, v, p, iteration)
                self._apply_force_feedback(iteration, t)

        else:
            (u, v, p, w_vel) = self._solve(u, v, p, iteration, w_vel=w_vel)

            if self.compute_forces:
                self.forces_method2_3d(u, v, w_vel, p, iteration)
                self._apply_force_feedback(iteration, t)

        # ---- flow diagnostics (energy, enstrophy, CFL, divergence) ----
        self.diagnostics.update(
            iteration, u, v, p, self.dt, self.nu,
            divergence_fn=self.divergence,
            vorticity_fn=self.vorticity,
            w=w_vel,
        )

        # ---- free BDIM fields to reclaim GPU memory between steps ----
        self._release_bdim_fields()

        # ---- flush CUDA allocator cache to reduce nvidia-smi usage ----
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # ---- plotting / saving  (works for 2D and 3D) ----
        terminate = self.plotting_and_saving(u, v, p, iteration, w_vel=w_vel)

        if self.ndim == 2:
            return (u, v, p, terminate)
        else:
            return (u, v, p, w_vel, terminate)

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

        if self.compute_forces and self.save:
            self.save_drags_h5()

        # ---- save flow diagnostics ----
        if self.save:
            self._submit_io(self.diagnostics.save_h5,
                            self.save_path, self._hdf5_lock)

        # Block until all background I/O is complete before returning
        self.flush_io()

