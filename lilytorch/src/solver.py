
import datetime
import logging
import os
import threading
import warnings
import h5py
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from lilytorch.src.kernels import (
    streaming_sdf_stag_3d_multi,
    bdim_vardens_3d,
    bdim_vardens_sigma_3d,
    streaming_sdf_stag_2d_multi,
    bdim_vardens_2d,
    bdim_vardens_sigma_2d,
)
from lilytorch.src.adv_diff import AdvDiffSolver
from lilytorch.src.body import (body_from_yaml,
                                _mu_normals_batched)
from lilytorch.src.free_surface import FreeSurface
from lilytorch.src import operations as ops
from lilytorch.src.plotting import PlottingMixin
from lilytorch.src.poisson_fft import PoissonSolverFFT
from lilytorch.src.poisson_mult import PoissonSolver
from lilytorch.util.yaml_operations import pyobject2yaml
from lilytorch.src.forces import (
    _forces_shared, _forces_body_batch,
    _forces_body_integrate_3d,
)
from lilytorch.src import forces, extras

logger = logging.getLogger(__name__)


# ======================================================================
# Flow diagnostics — energy, enstrophy, divergence, CFL monitoring
# ======================================================================


# unused for now
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


def _build_fs_free_dicts(ndim):
    """Build the ``_FS_FREE_AFTER_BDIM`` / ``_FS_FREE_AFTER_VAR_DENS`` dicts
    for a given spatial dimensionality.

    The hot path uses ``self.__dict__.update(<dict>)`` to drop references to
    BDIM intermediates between sub-steps so the CUDA allocator can reclaim
    them.  Building the dicts here from ``ndim`` keeps the field list in
    one place and makes the 2-D path mechanically parallel to 3-D.
    """
    axes = ('u', 'v', 'w')[:ndim]
    norms = ('x', 'y', 'z')[:ndim]
    after_bdim = {f'mu1_all_{a}': None for a in axes}
    for a in axes:
        for n in norms:
            after_bdim[f'normal_{n}_{a}'] = None
    after_bdim['mu1_all'] = None
    after_var_dens = {f'mu0_all_{a}': None for a in axes}
    after_var_dens['mu0_all'] = None
    return after_bdim, after_var_dens


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

        # Device selection
        use_gpu = solver["use_gpu"]
        if torch.cuda.is_available() and use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
            self.device = torch.device("cuda")
        else:
            print("Using the CPU.")
            self.device = torch.device("cpu")
            torch.set_num_threads(solver["nthreads"])

        # dtype selection
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

        # BDIM-σ (Lauber et al. 2022): per-body Poisson-coefficient shift
        # to enforce mu0_poisson → 0 inside thin bodies (r < eps).  When
        # enabled, ``_sigma_shifts`` is lazily computed at the first
        # fluid step (once body SDFs are populated) and static thereafter.
        self.apply_bdim_sigma = bool(solver.get("apply_bdim_sigma", False))
        self._sigma_shifts    = None   # lazily computed at first fluid step

        # BDIM2 mu0-weighted Poisson coefficient.  When True (default) the
        # variable-density coefficient is ``dt*mu0/rho_eff`` (Maertens &
        # Weymouth 2015), which sharpens the body interface but makes the
        # Poisson operator degenerate inside/near the body.  When False the
        # coefficient is the plain ``dt/rho_eff`` (no mu0 numerator), keeping
        # the operator non-degenerate — required for stable multibody
        # swimmers where inter-link velocity seams create divergence in
        # low-mu0 cells the degenerate solve cannot remove.
        self.bdim_mu0_projection = bool(solver.get("bdim_mu0_projection", True))

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

        self.rho_body = float(solver.get("rho_body", 1000.0))

        self.terminate = False   # flag for early termination (e.g. from NaN detection)

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

        # ---- optional torch.compile for adv-diff -----
        self._compile_adv_diff = solver.get("compile_adv_diff", False)
        if self._compile_adv_diff and self.device.type == "cuda":
            self.adv_diff_solver.solve = torch.compile(
                self.adv_diff_solver.solve, mode="default",
            )

        # Dynamic BDIM META compilation for the union-AABB crop path
        # (sub-block shape varies with body kinematics).
        self._bdim_meta_dyn_compiled = torch.compile(
            FluidSolver._bdim_meta, dynamic=True,
        )
        self._bdim_union_aabb = None

        # ---- optional Towers (2008) 2nd-order delta correction -----------
        # When force_delta_order=2, the smoothed delta is divided by |∇SDF|
        # so that the volume integral gives the correct surface measure even
        # when the numerical SDF deviates from unit gradient.
        # For analytical bodies |∇SDF|=1 exactly, so order 2 is a no-op;
        # it matters for mesh bodies or near geometric corners.
        self.force_delta_order = int(solver.get("force_delta_order", 1))
        if self.force_delta_order not in (1, 2):
            raise ValueError(f"force_delta_order must be 1 or 2, got {self.force_delta_order}")
        # Force-integration method.
        #
        #   "eulerian"   — volumetric ∫ σ·δ_ε(φ) dV / pressure ∫ -p n δ_ε(φ) dV
        #                  (default).  Implemented by ``forces_method2`` /
        #                  ``forces_method2_3d``; works in both python and
        #                  kernel solver modes.
        #   "lagrangian" — surface integral ∫ σ·n dS on per-body Lagrangian
        #                  markers (2-D: arc-length contour ``cnt_update``;
        #                  3-D: per-body triangulation
        #                  ``tri_centroid_world``/``tri_normal_world``/
        #                  ``tri_area``).  Implemented by ``forces_lagrangian_2d``
        #                  / ``forces_lagrangian_3d``.
        #
        # Legacy aliases (``method1`` → ``lagrangian``, ``method2`` →
        # ``eulerian``) are accepted with a one-time DeprecationWarning so
        # existing configs continue to load.
        _fm_raw = solver.get("force_method", "eulerian")
        _fm_aliases = {"method1": "lagrangian", "method2": "eulerian"}
        if _fm_raw in _fm_aliases:
            import warnings
            warnings.warn(
                f"force_method={_fm_raw!r} is deprecated; use "
                f"{_fm_aliases[_fm_raw]!r} instead.",
                DeprecationWarning, stacklevel=2,
            )
            _fm_raw = _fm_aliases[_fm_raw]
        if _fm_raw not in ("eulerian", "lagrangian"):
            raise ValueError(
                f"solver.force_method must be one of "
                f"('eulerian', 'lagrangian'), got {_fm_raw!r}."
            )
        self.force_method = _fm_raw
        # Distance to offset the Lagrangian-force sample point along the
        # outward surface normal (see lagrangian_forces.cu).  0 (default)
        # = sample exactly at the centroid/contour marker (legacy
        # behaviour, biased by BDIM band contamination).  Set to ~eps
        # via config to escape the band.
        self.lagrangian_sample_offset = float(
            solver.get("lagrangian_sample_offset", 0.0))
        self.zero_pressure_inside = solver.get("zero_pressure_inside", False)
        # Smooth body-velocity blend in the overlap band (kernel path).
        # Width given in grid cells; <=0 / None → legacy hard running-min
        # winner-take-all.  See BDIMhandler / streaming_sdf.cu.
        _bvb = solver.get("body_velocity_blend_eps_cells", None)
        self._body_vel_blend_cells = float(_bvb) if _bvb else 0.0

        self._solver_method = solver.get("solver_method", "kernel")
        method = solver.get("solver_method", "kernel")
        if method not in ("python", "kernel"):
            raise ValueError(
                f"solver.solver_method must be one of "
                f"('python', 'kernel'), got {method!r}."
            )

        _method = self._solver_method
        _kernel = (_method == "kernel")

        self._use_kernels    = _kernel
        self._mu_union_ready = False

        _METHOD_DESCR = {
            "python": "Python reference path",
            "kernel": ("native streamed path (update-only geometry + "
                       "post-fluid-step force kernel)"),
        }
        print(f"  [solver_method={_method!r}] {_METHOD_DESCR[_method]}")
        # Body-SDF sampling method used inside the streaming C++/CUDA
        # kernels (``streaming_sdf_stag_{2d,3d}_multi``):
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

        # self._mu_normals_dyn_compiled = torch.compile(
        #         _mu_normals_batched, dynamic=True,
        #     )

        self._mu_normals_dyn_compiled = _mu_normals_batched

        self._forces_shared_compiled     = _forces_shared
        self._forces_body_batch_compiled = _forces_body_batch
        self._forces_shared_dyn_compiled = _forces_shared
        self._forces_body_compiled       = _forces_body_integrate_3d

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
            use_kernels     = self._use_kernels,
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

        self.composite_body = body_from_yaml(
            self.device,
            self.x, self.y,
            body_pars,
            z             = self.z,
            eps           = self.eps,
            custom_update = custom_update,
            starting_time = self.starting_time,
            use_kernels   = self._use_kernels,
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
        self.save_drags       = output.get("save_drags", False)
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

        # ----------------------------------------------------------------
        # Dim-dispatch table.  Bind the right per-step method once so the
        # FSI hot path does not branch on ``self.ndim`` every step.  The
        # ``if`` here lives in __init__ — that's the only place it is
        # allowed to remain (Step 1 of the 2-D/3-D unification refactor).
        # ----------------------------------------------------------------
        # ``_fluid_step``, ``_recompute_mu_normals`` and ``_bdim_apply``
        # are dim-agnostic class methods — no per-D dispatch binding needed.

        # ----------------------------------------------------------------
        # Per-instance BDIM-intermediate free-dicts (Step 4 of unification).
        # Replaces the module-level ``_FS_FREE_AFTER_BDIM_3D`` /
        # ``_FS_FREE_AFTER_VAR_DENS_3D`` constants — these are now built
        # from ``self.ndim`` so the 2-D path can use the same mechanism.
        # ----------------------------------------------------------------
        (self._FS_FREE_AFTER_BDIM,
         self._FS_FREE_AFTER_VAR_DENS) = _build_fs_free_dicts(self.ndim)

        # Pre-built per-axis attribute names for ``_apply_bdim_all_axes``.
        # Stored once at __init__ so the hot path is a single attribute
        # read + getattr loop, with no per-step Python branching on ndim.
        self._bdim_axis_names   = ('u', 'v', 'w')[:self.ndim]
        self._bdim_normal_names = ('x', 'y', 'z')[:self.ndim]

        # Eagerly allocate the kernel-mode persistent face-coefficient
        # buffers (_ch/_cv/_cw_persist) so the ~1.5 GiB cost at 512³ fp32
        # is paid HERE (and counted in the persistent baseline) instead of
        # mid-step on the first fluid_step.  If the runtime timestep differs
        # from self.dt these buffers will be reallocated on the first call,
        # at no worse cost than the previous lazy-init path.
        if self._use_kernels and self.ndim == 3:
            self._init_var_dens_persist_3d(self.dt)
        elif self._use_kernels and self.ndim == 2:
            self._init_var_dens_persist_2d(self.dt)

        # =====================================================================
        # Free-surface (level-set fluid-air) + gravity body force
        # ---------------------------------------------------------------------
        # Both are opt-in via the ``solver.free_surface`` and ``solver.gravity``
        # YAML blocks.  When neither is set every code path below short-circuits
        # to its pre-existing behaviour, so the BDIM solid-body pipeline is
        # unaffected.
        #
        #   solver.free_surface:
        #     phi_init: "lambda X, Y: Y - 0.5"   # negative in fluid, positive in air
        #     theta_min:    0.01      # GFM cut-face fluid-fraction clamp
        #     band_cells:   4         # narrow-band half-width (informational)
        #     reinit_iters: 4         # explicit reinit sub-steps per call
        #     reinit_every: 5         # reinitialise every N solver steps
        #     extend_iters: 4         # velocity-extension sub-steps per call
        #     extend_every: 1
        #
        #   solver.gravity: [gx, gy]      # 2-D
        #   solver.gravity: [gx, gy, gz]  # 3-D
        # =====================================================================
        self._init_gravity(solver.get("gravity", None))
        self._init_free_surface(solver.get("free_surface", None))

    # =====================================================================
    # Free-surface + gravity body force (opt-in)
    # =====================================================================
    def _init_gravity(self, gravity_cfg):
        """Parse the optional ``solver.gravity`` block.

        ``solver.gravity`` is a list of length ``ndim`` giving the gravity
        body acceleration vector in SI units (m/s²).  When ``None`` the
        body force is disabled and ``self.use_gravity`` is False.
        """
        self.use_gravity = False
        self._gravity = None
        if gravity_cfg is None:
            return
        g = list(gravity_cfg)
        if len(g) != self.ndim:
            raise ValueError(
                f"solver.gravity must have {self.ndim} components in "
                f"{self.ndim}-D, got {g}."
            )
        self._gravity = tuple(float(c) for c in g)
        self.use_gravity = any(c != 0.0 for c in self._gravity)
        if self.use_gravity:
            print(f"Gravity body force enabled: g = {self._gravity}")

    def _init_free_surface(self, fs_cfg):
        """Parse ``solver.free_surface`` and build the :class:`FreeSurface`.

        When the block is absent ``self.free_surface`` is ``None`` and
        every downstream hook short-circuits.  See :class:`FreeSurface`
        for the full list of accepted parameters.
        """
        self.free_surface = None
        self._fs_reinit_every = 5
        self._fs_extend_every = 1
        if fs_cfg is None:
            return
        phi_src = fs_cfg.get("phi_init")
        if phi_src is None:
            raise ValueError(
                "solver.free_surface.phi_init is required when "
                "solver.free_surface is set."
            )
        # Allow either a raw callable or a YAML lambda string evaluated
        # in a torch-only namespace (same convention as body.sdf).
        if isinstance(phi_src, str):
            phi_init = eval(phi_src, {"torch": torch})
        else:
            phi_init = phi_src
        if not callable(phi_init):
            raise ValueError(
                "solver.free_surface.phi_init must be a callable or a "
                "lambda-string evaluating to one."
            )
        kwargs = dict(
            theta_min   = float(fs_cfg.get("theta_min", 0.01)),
            band_cells  = int(fs_cfg.get("band_cells", 4)),
            reinit_iters= int(fs_cfg.get("reinit_iters", 4)),
            extend_iters= int(fs_cfg.get("extend_iters", 4)),
            device      = self.device,
            dtype       = self.dtype,
        )
        if self.ndim == 2:
            self.free_surface = FreeSurface(
                self.x, self.y, self.h, phi_init, **kwargs,
            )
        else:
            self.free_surface = FreeSurface(
                self.x, self.y, self.h, phi_init, z=self.z, **kwargs,
            )
        self._fs_reinit_every = int(fs_cfg.get("reinit_every", 5))
        self._fs_extend_every = int(fs_cfg.get("extend_every", 1))
        print(
            f"Free-surface enabled: theta_min={kwargs['theta_min']}, "
            f"reinit every {self._fs_reinit_every} steps "
            f"({kwargs['reinit_iters']} sub-iters), "
            f"velocity extension every {self._fs_extend_every} steps "
            f"({kwargs['extend_iters']} sub-iters)."
        )

    @torch.no_grad()
    def _apply_gravity_body_force(self, *vels):
        """Add ``dt * g`` to each velocity component (predictor-side
        forward-Euler body force).  Called inside ``step_`` right before
        ``fluid_step`` so the projection can balance it through the
        Poisson solve.

        When the free surface is active the body force is gated by the
        cell-centred *fluid* mask: gravity only accelerates fluid cells.
        Without this gate the air narrow-band cells free-fall through
        the box and dominate ``|u|_max``, even though their pressure is
        pinned to ``p_atm = 0`` by the ghost-fluid layer.
        """
        if not self.use_gravity:
            return vels
        # MAC-staggered: ``u`` is on x-faces, ``v`` on y-faces, etc.
        # A uniform gravity vector adds the same constant to every face
        # on the corresponding component grid; this is consistent with
        # the cell-centred pressure projection that follows.
        fluid_mask = (None if self.free_surface is None
                      else self.free_surface.fluid_mask_cc)
        out = []
        for vel, g_comp in zip(vels, self._gravity):
            if g_comp == 0.0:
                out.append(vel)
            else:
                if fluid_mask is not None:
                    # The cell-centred fluid mask is a defensible proxy
                    # for the staggered MAC component: gravity is a
                    # local body force, so leaving the air band's MAC
                    # cells un-accelerated is enough to keep them quiet
                    # (the velocity extension step then propagates the
                    # fluid value into the narrow band each step).
                    vel.add_(
                        (float(self.dt) * g_comp) * fluid_mask.to(vel.dtype)
                    )
                else:
                    # In-place add to avoid an extra allocation.
                    vel.add_(float(self.dt) * g_comp)
                out.append(vel)
        return tuple(out)

    @torch.no_grad()
    def _fs_apply_gfm_to_coeffs(self, ch, cv, cw):
        """Multiply staggered Poisson face coefficients in-place by the
        ghost-fluid face scales.

        When the host coefficient arrays are full-grid (legacy python
        path) the GFM scales — which are face-grid sized — are inserted
        into the appropriate inner slice; when they are face-grid
        (kernel path) we multiply directly.
        """
        if self.free_surface is None:
            return ch, cv, cw
        scales = self.free_surface.ghost_fluid_face_scales()
        s_u = scales[0]
        s_v = scales[1]
        s_w = scales[2] if self.ndim == 3 else None
        # Detect face-grid vs full-grid: project()'s ``_face_grid`` check
        # is ``ch.shape[0] < grid_shape[0]`` — use the same here so the
        # scaling slice matches.
        if self.ndim == 2:
            _face_grid = (ch.shape[0] < self.nx)
            if _face_grid:
                ch = ch * s_u[:, 1:-1]
                cv = cv * s_v[1:-1, :]
            else:
                ch[1:, 1:-1] = ch[1:, 1:-1] * s_u[:, 1:-1]
                cv[1:-1, 1:] = cv[1:-1, 1:] * s_v[1:-1, :]
        else:
            _face_grid = (ch.shape[0] < self.nx)
            if _face_grid:
                ch = ch * s_u[:, 1:-1, 1:-1]
                cv = cv * s_v[1:-1, :, 1:-1]
                cw = cw * s_w[1:-1, 1:-1, :]
            else:
                ch[1:, 1:-1, 1:-1] = ch[1:, 1:-1, 1:-1] * s_u[:, 1:-1, 1:-1]
                cv[1:-1, 1:, 1:-1] = cv[1:-1, 1:, 1:-1] * s_v[1:-1, :, 1:-1]
                cw[1:-1, 1:-1, 1:] = cw[1:-1, 1:-1, 1:] * s_w[1:-1, 1:-1, :]
        return ch, cv, cw

    @torch.no_grad()
    def _fs_post_step(self, u, v, w_vel, iteration):
        """Free-surface bookkeeping invoked once per outer time step:

        1. advect ``phi_fs`` with the projected (divergence-free) velocity;
        2. periodically reinitialise to ``|∇phi| = 1``;
        3. periodically extend the cell-centred velocity into the air
           narrow band (used by next-step advection of ``phi_fs`` and
           by the divergence stencil at cut faces).

        The cell-centred velocity extension is applied to **MAC velocity
        components** in-place (a constant-along-normal extension on the
        MAC grid is a defensible first approximation as long as the
        narrow band has only a few cells of air on the air side of the
        interface — within that band the staggered offset matters less
        than the upwind direction itself).
        """
        if self.free_surface is None:
            return
        fs = self.free_surface
        if self.ndim == 2:
            fs.advect(u, v, dt=float(self.dt))
        else:
            fs.advect(u, v, w_vel, dt=float(self.dt))

        if self._fs_reinit_every > 0 and (iteration % self._fs_reinit_every == 0):
            fs.reinitialize()

        if self._fs_extend_every > 0 and (iteration % self._fs_extend_every == 0):
            if self.ndim == 2:
                fs.extend_velocity(u, v)
            else:
                fs.extend_velocity(u, v, w_vel)

    def inside(self, x):
        """
        Return True if all bodies' x are inside the domain
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
        # In Phase-I kernel-mode 3-D the composite body does not carry
        # persistent staggered SDF tensors, so this masking step is
        # skipped (BDIM correction on the very first solver step will
        # apply the equivalent in-body mask).
        if (hasattr(self, "composite_body")
                and hasattr(self.composite_body, "sdf_val")
                and hasattr(self.composite_body, "sdf_val_u")):
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
    # Lagrangian (surface-integral) force methods — phase 2 of force_method
    # rework.  See ``forces.forces_lagrangian_2d`` / ``forces_lagrangian_3d``.
    forces_lagrangian_2d = forces.forces_lagrangian_2d
    forces_lagrangian_3d = forces.forces_lagrangian_3d




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

        self.div  = self.divergence(u, v, w=w_vel)

        coeff = w * self.dt / self.rho

        # ---- Free-surface (level-set) ghost-fluid integration ----------
        # When the optional free surface is active, multiply the
        # staggered Poisson coefficients (face-grid `dt/ρ_eff` arrays)
        # elementwise by the GFM scales: 1 in fluid, 0 across air-air
        # faces (decouples air cells), 1/θ across cut faces (Dirichlet
        # `p_atm = 0` on the interpolated zero-crossing of φ_fs).
        # Also force the divergence RHS to 0 in air cells so the
        # smoother sees no spurious driving term where p ≡ 0.
        if self.free_surface is not None:
            # If the legacy multigrid path will fall back to its default
            # constant-density coefficients, materialise them first so
            # the GFM rescale can be applied uniformly.
            if self.poisson_method != "fft":
                if ch is None:
                    ch = coeff * self.mu0_all_u
                if cv is None:
                    cv = coeff * self.mu0_all_v
                if self.ndim == 3 and cw is None:
                    cw = coeff * self.mu0_all_w
            ch, cv, cw = self._fs_apply_gfm_to_coeffs(ch, cv, cw)
            # Mask div in air cells (cell-centred).  This is a no-op on the
            # boundary slice the Poisson solver actually sees, but it
            # avoids any non-zero RHS leaking into ill-conditioned air
            # cells when `_face_grid` slicing changes.
            air = self.free_surface.air_mask_cc
            self.div = torch.where(air, torch.zeros_like(self.div), self.div)
            # Hand the **inner** air mask to the Poisson solver as a
            # Dirichlet pin (p_air ≡ 0).  The GFM cut-face scaling makes
            # the air cells adjacent to the interface non-degenerate
            # (J ≠ 0), so the smoother's J=0 mechanism alone is NOT
            # enough to enforce the Dirichlet BC there.  The mask hooks
            # into PoissonSolver.smooth to zero p in those cells after
            # every sweep at every multigrid level.
            if self.ndim == 2:
                self.poisson_solver.dirichlet_mask = air[1:-1, 1:-1].contiguous()
            else:
                self.poisson_solver.dirichlet_mask = air[1:-1, 1:-1, 1:-1].contiguous()
        else:
            self.poisson_solver.dirichlet_mask = None

        if self.poisson_method == "fft":
            # ---- FFT solver ----
            if ch_cc is not None:
                # Variable-density path: RHS uses cell-centred density,
                # correction uses the staggered ch/cv/cw coefficients.
                # With bdim_mu0_projection=True, ch_cc=0 inside the body
                # (mu0=0) — body cells decouple from the projection.
                # Mask those cells (RHS=0) so we don't divide by zero;
                # FFT then finds a harmonic extension that matches the
                # fluid Poisson solution. Inside-body p doesn't affect
                # u_new there because the correction is multiplied by
                # ch (which is also zero).
                _body_decoupled = (ch_cc <= 0)
                _ch_safe = torch.where(_body_decoupled,
                                       torch.ones_like(ch_cc), ch_cc)
                _rhs = torch.where(_body_decoupled,
                                   torch.zeros_like(self.div),
                                   self.div / _ch_safe)
                p = self.poisson_solverFFT.solve(_rhs)
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
                # If ch was provided as a scalar, use it (backward compat).
                # If ch is a face-grid tensor (kernel mode), fall back to
                # the scalar coeff for the Poisson RHS and apply in-place
                # face-grid correction (same logic as the multigrid path).
                _face_grid_fft = (ch is not None
                                  and isinstance(ch, torch.Tensor)
                                  and ch.shape[0] < u.shape[0])
                fft_coeff = coeff if (ch is None or _face_grid_fft) else ch
                # With bdim_mu0_projection=True, fft_coeff can be 0 inside
                # the body (mu0=0). The straight division self.div /
                # fft_coeff then yields 0/0 = NaN there, corrupting the
                # FFT solve. Mask body-decoupled cells (RHS=0); FFT
                # produces a harmonic extension and the subsequent
                # correction multiplies by fft_coeff=0 inside the body,
                # so u_new equals u_star = u_body there as intended.
                if isinstance(fft_coeff, torch.Tensor):
                    _body_decoupled = (fft_coeff <= 0)
                    _safe = torch.where(_body_decoupled,
                                        torch.ones_like(fft_coeff), fft_coeff)
                    _rhs = torch.where(_body_decoupled,
                                       torch.zeros_like(self.div),
                                       self.div / _safe)
                else:
                    _rhs = self.div / fft_coeff
                p = self.poisson_solverFFT.solve(_rhs)
                if self.ndim == 2:
                    (p_x, p_y) = self.gradient(p)
                    u = u - fft_coeff * p_x
                    v = v - fft_coeff * p_y
                else:
                    (p_x, p_y, p_z) = self.gradient(p)
                    if _face_grid_fft:
                        u[1:, 1:-1, 1:-1]     -= ch * p_x[1:, 1:-1, 1:-1]
                        v[1:-1, 1:, 1:-1]     -= cv * p_y[1:-1, 1:, 1:-1]
                        w_vel[1:-1, 1:-1, 1:] -= cw * p_z[1:-1, 1:-1, 1:]
                    else:
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
                # Detect face-grid coefficients (shape Ngx-1, Ngy-2, Ngz-2)
                # vs padded coefficients (shape Ngx, Ngy, Ngz).  The kernel
                # path stores ch/cv/cw directly at the staggered face-grid
                # size so the Poisson solver receives contiguous arrays.
                _face_grid = ch.shape[0] < u.shape[0]
                p, _ = _poisson_solve(
                    self.div[1:-1, 1:-1, 1:-1],
                    p0,
                    ch=(ch               if _face_grid else ch[1:, 1:-1, 1:-1]),
                    cv=(cv               if _face_grid else cv[1:-1, 1:, 1:-1]),
                    cw=(cw               if _face_grid else cw[1:-1, 1:-1, 1:]),
                )
                if bool(int(os.environ.get('LILYTORCH_MEM_DBG', '0'))):
                    torch.cuda.synchronize()
                    _gb = torch.cuda.memory_allocated() / 1024**3
                    _mx = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"[MEM_DBG] {'project:after multigrid solve (pre-gradient)':55s}"
                          f"  cur={_gb:.3f} GiB  peak={_mx:.3f} GiB", flush=True)
                # ====== projection step ======
                # Per-axis fused correction: avoids materialising (p_x, p_y, p_z)
                # as three full-grid tensors (~1.5 GB transient at 512³ float32).
                # Each axis allocates ONE diff tensor, applied with addcmul_,
                # then released before the next axis. Boundary face is zeroed
                # to preserve the original gradient() semantics (the operator
                # returns 0 at index 0 and at index N-1; the slice [1:] picks
                # up indices 1..N-1, so the LAST element of the diff must be 0
                # to match — set_BCs overwrites it shortly afterward anyway).
                inv_h = 1.0 / self.h
                if _face_grid:
                    diff = p[1:, 1:-1, 1:-1] - p[:-1, 1:-1, 1:-1]
                    diff[-1, :, :] = 0
                    u[1:, 1:-1, 1:-1].addcmul_(ch, diff, value=-inv_h)
                    del diff
                    diff = p[1:-1, 1:, 1:-1] - p[1:-1, :-1, 1:-1]
                    diff[:, -1, :] = 0
                    v[1:-1, 1:, 1:-1].addcmul_(cv, diff, value=-inv_h)
                    del diff
                    diff = p[1:-1, 1:-1, 1:] - p[1:-1, 1:-1, :-1]
                    diff[:, :, -1] = 0
                    w_vel[1:-1, 1:-1, 1:].addcmul_(cw, diff, value=-inv_h)
                    del diff
                else:
                    # Padded-coefficient path (non-kernel): keep the original
                    # gradient() call. The padded coefficients are full-grid
                    # so the savings vs the inline path are marginal.
                    (p_x, p_y, p_z) = self.gradient(p)
                    u     = u - ch * p_x
                    v     = v - cv * p_y
                    w_vel = w_vel - cw * p_z
                    del p_x, p_y, p_z
                if bool(int(os.environ.get('LILYTORCH_MEM_DBG', '0'))):
                    torch.cuda.synchronize()
                    _gb = torch.cuda.memory_allocated() / 1024**3
                    _mx = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"[MEM_DBG] {'project:after inline projection correction':55s}"
                          f"  cur={_gb:.3f} GiB  peak={_mx:.3f} GiB", flush=True)

        # Free-surface post-projection: restore the Dirichlet ``p == 0``
        # gauge in air cells.  The multigrid solver subtracts the field
        # mean at the end of every solve to remove the constant null
        # space (correct for purely Neumann boundaries); when a GFM
        # Dirichlet BC is active that mean-shift breaks the gauge.  We
        # undo it here by shifting the whole field by the post-solve
        # air-cell mean (which, in absence of the shift, should be 0).
        if self.free_surface is not None:
            air = self.free_surface.air_mask_cc
            if air.any():
                offset = p[air].mean()
                p = p - offset
            self.free_surface.apply_pressure_mask(p)

        if self.ndim == 2:
            return (u, v, p)
        else:
            return (u, v, w_vel, p)
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
    #   BDIM apply with optional union-AABB narrow band  (dim-agnostic)
    # ------------------------------------------------------------------
    def _bdim_apply(self, phi, mu0, body_vel, mu1, *normals):
        """Apply the BDIM2 meta-equation to a single staggered grid.

        Dim-agnostic: handles both 2-D (two normals) and 3-D (three
        normals).  When a union AABB is available (cached on
        ``self._bdim_union_aabb`` as a flat ``(lo_0, hi_0, lo_1, hi_1, ...)``
        tuple of length ``2 * ndim``), only the union sub-block is
        touched.  Outside the union mu0=1, mu1=0, body_vel=0 makes the
        meta-equation the identity, so phi[outside] is left unchanged
        (we slice-write the cropped result back into phi).

        Otherwise falls back to the full-grid kernel.  Returns a tensor
        the caller can safely consume; the caller is responsible for
        cloning if it needs to keep a reference past the next CUDA-graph
        replay (full-grid path) — in the union path the returned tensor
        is the input ``phi`` itself (mutated in place), which is already
        an owned tensor.
        """
        D = self.ndim
        _h = self.h
        u_aabb = self._bdim_union_aabb
        if u_aabb is None:
            return self._bdim_meta_dyn_compiled(
                phi, mu0, body_vel, mu1, *normals, _h, D,
            )

        usl = tuple(
            slice(u_aabb[2 * d], u_aabb[2 * d + 1]) for d in range(D)
        )
        sub = self._bdim_meta_dyn_compiled(
            phi[usl].contiguous(),
            mu0[usl].contiguous(),
            body_vel[usl].contiguous(),
            mu1[usl].contiguous(),
            *(n[usl].contiguous() for n in normals),
            _h, D,
        )
        phi[usl] = sub
        return phi

    # ------------------------------------------------------------------
    #   Per-axis BDIM apply — dim-agnostic loop over MAC axes
    # ------------------------------------------------------------------
    def _apply_bdim_all_axes(self, vels):
        """Apply BDIM to each velocity component on its own staggered grid.

        For each axis ``a`` in ``('u', 'v'[, 'w'])`` reads the per-axis
        ``mu0_all_a``, ``mu1_all_a``, ``body_a`` and per-(axis,normal-component)
        ``normal_<n>_<a>`` attributes, and dispatches through the
        dim-agnostic ``self._bdim_apply``.  Used by ``_fluid_step`` to
        avoid per-axis duplication.
        """
        comp = self.composite_body
        out = []
        for i, ax in enumerate(self._bdim_axis_names):
            mu0 = getattr(self, f'mu0_all_{ax}')
            mu1 = getattr(self, f'mu1_all_{ax}')
            body_vel = getattr(comp, f'body_{ax}')
            normals = tuple(
                getattr(self, f'normal_{n}_{ax}')
                for n in self._bdim_normal_names
            )
            out.append(self._bdim_apply(vels[i], mu0, body_vel, mu1, *normals))
        return tuple(out)

    # ------------------------------------------------------------------
    #   Union AABB across all body sub-blocks (dim-agnostic)
    # ------------------------------------------------------------------
    def _compute_union_aabb(self, halo=2, bucket=16):
        """Return the union AABB over all body sparse SDFs as a flat
        ``(lo_0, hi_0, lo_1, hi_1, ...)`` tuple of length ``2 * ndim``,
        expanded by ``halo`` cells and clipped to grid extent.  Returns
        ``None`` if no body has a sparse AABB OR if the union covers more
        than 50 % of the full grid volume (above that, the full-grid
        kernel is faster than the cropped-then-slice-write path).

        Reads ``comp._sdf_sparse`` (per-body slabs) when available; falls
        back to ``comp._combined_union_aabb`` (fused-kernel populated) when
        ``_sdf_sparse`` is absent or empty.

        When ``bucket > 1`` each extent is rounded up to a multiple of
        ``bucket`` by expanding the high side first and spilling into the
        low side if clipped at the grid boundary.  This stabilizes the
        sub-block shape to a bounded set so that ``dynamic=True``
        compiled kernels only pay the recompile cost once per bucket
        combination seen during warmup, instead of every step.
        """
        comp = self.composite_body
        D = self.ndim
        grid_shape = comp.sdf_val.shape  # (N0, N1[, N2])

        lo = [1 << 30] * D
        hi = [-1] * D
        sparse = getattr(comp, '_sdf_sparse', None)

        if not sparse or sparse[0] is None:
            # Fused SDF+forces path stores the union AABB directly so the
            # cheap sub-block path can still activate without per-body slabs.
            raw = getattr(comp, '_combined_union_aabb', None)
            if raw is None:
                return None
            for d in range(D):
                lo[d], hi[d] = raw[2 * d], raw[2 * d + 1]
        else:
            for entry in sparse:
                if entry is None:
                    return None
                aabb_i = entry[0]
                if aabb_i is None:
                    return None
                # aabb_i is a flat 2D-tuple (lo0, hi0, lo1, hi1, ...)
                for d in range(D):
                    if aabb_i[2 * d] < lo[d]:
                        lo[d] = aabb_i[2 * d]
                    if aabb_i[2 * d + 1] > hi[d]:
                        hi[d] = aabb_i[2 * d + 1]

        # Halo expansion + clipping
        for d in range(D):
            lo[d] = max(0, lo[d] - halo)
            hi[d] = min(grid_shape[d], hi[d] + halo)

        # Bucket-rounding so dynamic-shape compiled kernels see a small
        # discrete set of sub-block shapes.
        if bucket is not None and bucket > 1:
            for d in range(D):
                extent = hi[d] - lo[d]
                target = ((extent + bucket - 1) // bucket) * bucket
                N = grid_shape[d]
                if target > N:
                    target = N
                pad = target - extent
                new_hi = hi[d] + pad
                new_lo = lo[d]
                if new_hi > N:
                    over = new_hi - N
                    new_hi = N
                    new_lo = max(0, new_lo - over)
                lo[d], hi[d] = new_lo, new_hi

        # Skip the crop fast-path when the sub-block covers >50 % of the
        # full grid volume — at that point the slice-and-write overhead
        # exceeds the saved kernel work.
        sub_vol = 1
        full_vol = 1
        for d in range(D):
            sub_vol *= (hi[d] - lo[d])
            full_vol *= grid_shape[d]
        if sub_vol > 0.5 * full_vol:
            return None

        out = []
        for d in range(D):
            out.append(lo[d])
            out.append(hi[d])
        return tuple(out)

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
        # CC-grid mu / normals (recomputed in _recompute_mu_normals)
        'mu0_all', 'mu1_all',
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
        beginning of every step anyway).

        Phase I: the kernel-mode paths (2-D and 3-D) no longer
        materialise any of the staggered or CC mu / normal tensors as
        full-grid buffers — they live only in CUDA thread registers
        inside Kernel B — so the keep-set is empty in both modes.
        """
        for attr in self._BDIM_FIELD_NAMES:
            if hasattr(self, attr):
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    #   mu / normal recomputation  (shared by step_() and BDIMhandler)
    # ------------------------------------------------------------------
    def _recompute_mu_normals(self):
        """Recompute mu0/mu1 and CC/staggered unit normals — dim-agnostic.

        Processes all ``ndim + 1`` SDF grids ([u, v, [w,] cc]) in a single
        batched pass via the dim-agnostic ``_mu_normals_batched`` helper.

        When the kernel mode is on AND the union-AABB sub-block covers
        less than 50 % of the grid (see :meth:`_compute_union_aabb`),
        the compiled kernel runs only on the union sub-block and results
        are slice-written into a single persistent packed buffer
        ``self._mu_pack`` pre-filled with outside-body defaults
        (mu0 = 1, mu1 = 0, normals = 0).  Every downstream
        ``mu0_all_<a>`` / ``mu1_all_<a>`` / ``normal_<n>_<a>`` attribute
        is a *view* into the pack, so a sub-block update is one fused
        slice-write rather than ``(2 + ndim) * (ndim + 1) + 1`` separate
        slice writes.

        Pack layout along dim-0 (D = ndim, n_grids = D + 1):
            ``[0 .. n_grids)``                  : mu0  for [u, v, [w,] cc]
            ``[n_grids .. 2*n_grids)``          : mu1  for the same grids
            ``[2*n_grids .. 3*n_grids)``        : normal_x for the same
            ``[3*n_grids .. 4*n_grids)``        : normal_y for the same
            ``[4*n_grids .. 5*n_grids)``        : normal_z (3-D only)

        When kernel mode is off or the union covers >50 % of the grid,
        falls back to the full-grid compiled kernel (no pack buffer).
        """
        comp   = self.composite_body
        D      = self.ndim
        ngrids = D + 1
        cc     = D                       # CC slot index along the stack
        axes   = self._bdim_axis_names   # ('u', 'v') or ('u', 'v', 'w')
        norms  = self._bdim_normal_names # ('x', 'y') or ('x', 'y', 'z')

        # SDF attribute names: (sdf_val_u, sdf_val_v, [sdf_val_w,] sdf_val)
        sdf_attrs = tuple(f'sdf_val_{a}' for a in axes) + ('sdf_val',)

        # ------------------------------------------------------------------
        # Union-AABB crop path — outside the union SDF is _FAR so mu0=1,
        # mu1=0, normals=0 (these defaults never change between steps).
        # Slice each SDF to the sub-block BEFORE stacking so the torch.stack
        # allocation is O(sub-block) rather than O(full grid).
        # ------------------------------------------------------------------
        if self._use_kernels:
            u_aabb = self._compute_union_aabb(halo=2, bucket=16)
            if u_aabb is not None:
                n_pack = (2 + D) * ngrids   # 12 for 2-D, 20 for 3-D

                if (getattr(self, '_mu_pack', None) is None
                    or self._mu_pack.shape[1:] != comp.sdf_val.shape
                    or self._mu_pack.shape[0] != n_pack
                    or self._mu_pack.dtype  != comp.sdf_val.dtype
                    or self._mu_pack.device != comp.sdf_val.device):
                    pack = torch.zeros(
                        (n_pack, *comp.sdf_val.shape),
                        device=comp.sdf_val.device, dtype=comp.sdf_val.dtype,
                    )
                    pack[0:ngrids].fill_(1.0)   # mu0 defaults to 1 outside body
                    self._mu_pack = pack
                    self._mu_union_ready = True
                pack = self._mu_pack

                # (Re-)alias every step — cheap Python, robust to any
                # non-union path overwriting these attributes.
                for i, ax in enumerate(axes):
                    setattr(self, f'mu0_all_{ax}', pack[i])
                    setattr(self, f'mu1_all_{ax}', pack[ngrids + i])
                    for d, n in enumerate(norms):
                        setattr(self, f'normal_{n}_{ax}',
                                pack[(2 + d) * ngrids + i])
                self.mu0_all = pack[cc]
                self.mu1_all = pack[ngrids + cc]
                for d, n in enumerate(norms):
                    setattr(self, f'normal_{n}', pack[(2 + d) * ngrids + cc])

                slices = tuple(
                    slice(u_aabb[2 * d], u_aabb[2 * d + 1]) for d in range(D)
                )
                # Stack only the sub-block — O(sub-block * ngrids) copy,
                # not O(full-grid * ngrids).
                sdf_sub = torch.stack(
                    [getattr(comp, n)[slices] for n in sdf_attrs]
                )

                out = self._mu_normals_dyn_compiled(sdf_sub, comp.h, comp.eps)
                mu0_s, mu1_s = out[0], out[1]
                normals_s = out[2:]   # length-D tuple of (ngrids, *sub_spatial)

                # Fused slice-write: stack mu0, mu1, normals along dim-0 and
                # scatter into the packed buffer with ONE assign.
                stacked = torch.cat(
                    (mu0_s, mu1_s) + tuple(normals_s),
                    dim=0,
                )
                self._mu_pack[(slice(None),) + slices] = stacked
                return

        # ── Full-grid path: batched + compiled, all grids in one pass ──
        sdf_stack = torch.stack([getattr(comp, n) for n in sdf_attrs])
        out = self._mu_normals_dyn_compiled(sdf_stack, comp.h, comp.eps)
        mu0, mu1 = out[0], out[1]
        normals = out[2:]

        for i, ax in enumerate(axes):
            setattr(self, f'mu0_all_{ax}', mu0[i])
            setattr(self, f'mu1_all_{ax}', mu1[i])
            for d, n in enumerate(norms):
                setattr(self, f'normal_{n}_{ax}', normals[d][i])
        self.mu0_all = mu0[cc]
        self.mu1_all = mu1[cc]
        for d, n in enumerate(norms):
            setattr(self, f'normal_{n}', normals[d][cc])

    def _compute_sigma_shifts(self):
        """Compute per-body BDIM-σ shifts (Lauber et al. 2022).

        For each body ``b``, ``shift_b = max(0, eps + phi_min_b)`` where
        ``phi_min_b`` is the minimum body SDF.  Thick bodies (r ≥ eps)
        get ``shift = 0`` and behave unchanged; thin bodies (r < eps)
        get a positive shift so the Poisson coefficient
        ``mu0(phi - shift_b)`` reaches 0 inside the body.

        Stored as a float32 device tensor for direct ``data_ptr<float>()``
        consumption by the CUDA/CPU σ kernels.
        """
        shifts = []
        eps_val = float(self.eps)
        comp    = self.composite_body

        # In kernel mode body.sdf_val is never populated.  The per-body SDF
        # table lives in _kernel_static_{3,2}d['F_flat'] (mesh bodies: h/2
        # resolution, accurate; analytical bodies: h resolution).  In Python
        # mode fall back to the full-domain body.sdf_val.
        sm = getattr(comp, '_kernel_static_3d', None) \
            or getattr(comp, '_kernel_static_2d', None)

        ndim_b           = self.ndim
        F_flat           = sm['F_flat']           if sm is not None else None
        F_offsets        = sm['F_offsets']        if sm is not None else None
        body_shapes_flat = sm['body_shapes'].reshape(-1) if sm is not None else None

        for b, body in enumerate(comp.bodies):
            if F_flat is not None:
                i0   = int(F_offsets[b])
                size = int(body_shapes_flat[b * ndim_b:(b + 1) * ndim_b].prod().item())
                smin = float(F_flat[i0:i0 + size].min().item())
                shifts.append(max(0.0, smin + eps_val))
            else:
                sdf_val = getattr(body, 'sdf_val', None)
                if sdf_val is not None:
                    shifts.append(max(0.0, float(sdf_val.min().item()) + eps_val))
                else:
                    shifts.append(0.0)

        self._sigma_shifts = torch.tensor(
            shifts, dtype=torch.float32, device=self.device)
        nonzero = [(i, s) for i, s in enumerate(shifts) if s > 0]
        print(f"[BDIM-σ] sigma_shifts computed: {len(nonzero)} thin "
              f"body/bodies need correction")
        for i, s in nonzero:
            print(f"  body {i}: shift = {s*1e3:.4f} mm")

    def _compute_sigma_mu_grids(self, mu_grids):
        """Recompute ``mu0`` from σ-shifted union SDFs for each stagger axis.

        Used by ``_compute_variable_density_coefficients`` (Python-mode
        path) to substitute ``mu0_poisson`` for ``mu0_all_*`` when
        BDIM-σ is enabled.  The velocity BDIM mu0 is unchanged.

        Returns a tuple of the same length / shape as ``mu_grids``
        (``(mu0_u, mu0_v, [mu0_w,] mu0_cc)``).  Falls back to the
        original ``mu_grids`` when the per-body SDF attributes
        (``sdf_val_u/v/w``, ``sdf_val``) required to rebuild the
        union SDF are not populated (kernel mode does not store them).
        """
        comp  = self.composite_body
        axes  = self._bdim_axis_names
        sdf_attrs = tuple(f'sdf_val_{a}' for a in axes) + ('sdf_val',)
        eps_val   = float(self.eps)

        # Without per-body SDFs (kernel mode skips populating them)
        # there is no Python-side σ correction to apply.
        if not all(getattr(b, sdf_attrs[-1], None) is not None
                   for b in comp.bodies):
            return mu_grids

        # Re-build a σ-shifted union SDF per stagger axis by min-reducing
        # over per-body ``sdf_val_<attr> - shift_b``.  Start from a copy
        # of the unshifted union and tighten with each thin body.
        unions = []
        for attr in sdf_attrs:
            base = getattr(comp, attr, None)
            if base is None:
                return mu_grids
            unions.append(base.clone())

        for b, body in enumerate(comp.bodies):
            shift = float(self._sigma_shifts[b])
            if shift <= 0.0:
                continue
            for u_idx, attr in enumerate(sdf_attrs):
                sdf_b = getattr(body, attr, None)
                if sdf_b is None:
                    continue
                torch.minimum(unions[u_idx], sdf_b - shift, out=unions[u_idx])

        # Smooth Heaviside mu0(d/eps) — same formula as Body.mu_funcs.
        new_mu = []
        for u in unions:
            d    = u
            mu0  = (d >= 0).to(d.dtype)
            band = (d > -eps_val) & (d < eps_val)
            deps = d[band] / eps_val
            mu0[band] = 0.5 * (1.0 + deps + torch.sin(torch.pi * deps) / torch.pi)
            new_mu.append(mu0)
        return tuple(new_mu)

    # ==================================================================
    #  Variable-density FSI fluid step  (called by BDIMhandler.step)
    # ==================================================================

    # ------------------------------------------------------------------
    #   Persistent-buffer attribute names for the var-density fast path.
    #   Position 0..D-1 → face-grid (axis u, v[, w]); position D → CC.
    # ------------------------------------------------------------------
    _VAR_DENS_PERSIST_NAMES = ('_ch_persist', '_cv_persist', '_cw_persist',
                               '_ch_cc_persist')

    def _compute_variable_density_coefficients(self, timestep):
        """Compute variable-density Poisson coefficients for FSI coupling.

        Returns ``(ch, cv, ch_cc)`` for 2-D or ``(ch, cv, cw, ch_cc)``
        for 3-D, where:

            * ``ch, cv, cw`` -- staggered ``dt * mu0 / rho_eff`` on face grids.
            * ``ch_cc`` -- cell-centred ``dt * mu0 / rho_eff_cc`` for FFT RHS.

        BDIM2 mu0-weighted, variable-density Poisson coefficient:
        ``ch = dt * mu0 / (rho_body + (rho_fluid - rho_body) * mu0)``.
        The ``mu0`` factor makes the velocity-correction vanish EXACTLY inside
        the body (mu0=0), preserving the imposed body velocity and avoiding the
        ill-conditioned band Poisson.  When ``rho_body == rho_fluid`` it reduces
        to the original BDIM2 form ``dt * mu0 / rho`` (Maertens & Weymouth 2015).

        Narrow-band fast-path (kernel mode + sparse bodies, 2-D and 3-D)
        ----------------------------------------------------------------
        Outside the union AABB ``mu0 = 1`` everywhere, so the coefficients
        are constant (``dt / rho_fluid``).  Persistent full-grid buffers
        (``_ch_persist``, ``_cv_persist``, ``_cw_persist`` (3-D only),
        ``_ch_cc_persist``) are pre-filled once with that default and only
        the union sub-block is overwritten each step, avoiding ``D + 1``
        full-grid divisions.

        When the composite body has populated ``_winning_rho_cc`` (per-cell
        winning-body density, from the fused SDF+forces kernel) the
        sub-block uses that per-cell density; otherwise it falls back to
        the scalar ``self.rho_body``.
        """
        D       = self.ndim
        _drho   = float(self.rho) - self.rho_body
        # mu0-weighted (BDIM2) numerator vs plain dt/rho_eff — see the
        # ``bdim_mu0_projection`` flag set in __init__.
        _mu0w   = self.bdim_mu0_projection
        axes    = self._bdim_axis_names
        # All D+1 grids ([u, v, [w,] cc]) share the same shape in this
        # codebase (staggering is a coord offset, not a shape change),
        # so a single slice tuple indexes every grid identically.
        mu_grids = tuple(getattr(self, f'mu0_all_{a}') for a in axes) + (self.mu0_all,)

        # ---- BDIM-σ: substitute σ-shifted mu0 for the Poisson coefficient.
        # The velocity BDIM (mu0_all_* used in _bdim_apply) is *not* changed
        # — only the Poisson coefficient line uses these shifted grids.
        if (self.apply_bdim_sigma
                and self._sigma_shifts is not None
                and bool(self._sigma_shifts.any())):
            mu_grids = self._compute_sigma_mu_grids(mu_grids)

        # ---- narrow-band fast path -------------------------------------
        if self._use_kernels and all(m is not None for m in mu_grids):
            u_aabb = self._compute_union_aabb(halo=2, bucket=16)
            if u_aabb is not None:
                _dt_over_rhofluid = float(timestep / float(self.rho))
                names    = self._VAR_DENS_PERSIST_NAMES
                face_names = names[:D]
                cc_name    = names[3]   # always '_ch_cc_persist'
                mu_ref     = mu_grids[0]

                needs_realloc = (
                    getattr(self, '_ch_persist', None) is None
                    or self._ch_persist.shape  != mu_ref.shape
                    or self._ch_persist.dtype  != mu_ref.dtype
                    or self._ch_persist.device != mu_ref.device
                    or self._ch_outside_val    != _dt_over_rhofluid
                )
                if needs_realloc:
                    for name, mu in zip(face_names + (cc_name,), mu_grids):
                        setattr(self, name, torch.full_like(mu, _dt_over_rhofluid))
                    self._ch_outside_val = _dt_over_rhofluid

                usl = tuple(
                    slice(u_aabb[2 * d], u_aabb[2 * d + 1]) for d in range(D)
                )

                # Per-cell winning body density when the fused kernel
                # populated it; otherwise scalar rho_body.
                _winning = getattr(self.composite_body, '_winning_rho_cc', None)
                if _winning is not None:
                    _rho_fluid = float(self.rho)
                    _rho_b     = _winning[usl]
                    for name, mu in zip(face_names + (cc_name,), mu_grids):
                        mu_sub = mu[usl]
                        getattr(self, name)[usl] = (
                            (timestep * mu_sub if _mu0w else timestep)
                            / (_rho_b * (1 - mu_sub) + _rho_fluid * mu_sub)
                        )
                else:
                    for name, mu in zip(face_names + (cc_name,), mu_grids):
                        getattr(self, name)[usl] = (
                            (timestep * mu[usl] if _mu0w else timestep)
                            / (self.rho_body + _drho * mu[usl])
                        )

                return (*(getattr(self, n) for n in face_names),
                        getattr(self, cc_name))

        # ---- full-grid fallback ----------------------------------------
        return tuple((timestep * mu if _mu0w else timestep)
                     / (self.rho_body + _drho * mu) for mu in mu_grids)

    # ------------------------------------------------------------------
    #   Phase-I kernel-mode 3-D fluid step  (Kernel A + Kernel B)
    # ------------------------------------------------------------------
    def _init_var_dens_persist_3d(self, timestep):
        """Lazy-allocate the persistent ``ch/cv/cw`` Poisson-coefficient
        buffers for the kernel-mode 3-D path.

        Outside any immersed body ``mu0 = 1`` everywhere, so the
        coefficients reduce to the constant ``dt / rho_fluid``.  The
        buffers are pre-filled once with that default; Kernel B
        overwrites only the dirty AABB sub-block each step.
        """
        Ngx, Ngy, Ngz = self.grid_shape
        device = self.device
        dtype  = self.dtype
        _dt_over_rhofluid = float(timestep) / float(self.rho)
        # Face-grid shapes: each buffer covers only the staggered faces of
        # the interior region, excluding ghost-cell rows.  The kernel writes
        # directly into these shapes (no padded ghost-cell rows needed).
        ch_gs = (Ngx - 1, Ngy - 2, Ngz - 2)   # x-faces: (Nx+1, Ny, Nz)
        cv_gs = (Ngx - 2, Ngy - 1, Ngz - 2)   # y-faces: (Nx, Ny+1, Nz)
        cw_gs = (Ngx - 2, Ngy - 2, Ngz - 1)   # z-faces: (Nx, Ny, Nz+1)
        needs_realloc = (
            getattr(self, '_ch_persist', None) is None
            or self._ch_persist.shape  != ch_gs
            or self._ch_persist.dtype  != dtype
            or self._ch_persist.device != device
            or getattr(self, '_ch_outside_val', None) != _dt_over_rhofluid
        )
        if needs_realloc:
            self._ch_persist = torch.full(ch_gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._cv_persist = torch.full(cv_gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._cw_persist = torch.full(cw_gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._ch_outside_val = _dt_over_rhofluid

    def _init_var_dens_persist_2d(self, timestep):
        """2-D analogue of :meth:`_init_var_dens_persist_3d`."""
        gs = self.grid_shape
        device = self.device
        dtype  = self.dtype
        _dt_over_rhofluid = float(timestep) / float(self.rho)
        needs_realloc = (
            getattr(self, '_ch_persist', None) is None
            or self._ch_persist.shape  != gs
            or self._ch_persist.dtype  != dtype
            or self._ch_persist.device != device
            or getattr(self, '_ch_outside_val', None) != _dt_over_rhofluid
        )
        if needs_realloc:
            self._ch_persist = torch.full(gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._cv_persist = torch.full(gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._ch_outside_val = _dt_over_rhofluid

    def _fluid_step_kernel_2d(self, u, v, p, timestep):
        """Phase-I 2-D kernel fluid step.  2-D analogue of
        :meth:`_fluid_step_kernel_3d`; see that method for the full
        rationale.  Calls Kernel A (streaming SDF + body face velocities
        into per-step temporaries) then Kernel B (fused BDIM2 update +
        variable-density Poisson coefficients).
        """
        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_2d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_2d

        # BDIM-σ: lazily compute per-body sigma shifts on the first
        # fluid step (body SDFs are populated by BDIMhandler.update by
        # now).  Static thereafter.
        if self.apply_bdim_sigma and self._sigma_shifts is None:
            self._compute_sigma_shifts()

        # 1-2. eddy viscosity + advection-diffusion.
        nu_t   = self._compute_nu_t(u, v)
        primes = self.adv_diff_solver.solve(u, v, nu_t=nu_t)

        # Copy primes -> persistent u0/v0 so cells outside the dirty
        # AABB hold the advdiff result (Kernel B only touches the AABB).
        self.u0.copy_(primes[0])
        self.v0.copy_(primes[1])

        # 3. Init persistent var-dens coefficients (once / on resize).
        self._init_var_dens_persist_2d(timestep)

        # 4. Per-step temporaries for Kernel A -> Kernel B.
        _opts = dict(device=self.device, dtype=self.dtype)
        _FAR  = 1e4
        gs    = self.grid_shape
        sdf_u_tmp = torch.full(gs, _FAR, **_opts)
        sdf_v_tmp = torch.full(gs, _FAR, **_opts)
        bU_tmp    = torch.zeros(gs, **_opts)
        bV_tmp    = torch.zeros(gs, **_opts)
        # int64 key scratch buffers for Kernel A (packed body-id + SDF).
        # 2-D keys are full-grid sized so Kernel B's σ path can index by
        # the same flat ``g`` as the SDF/body tensors.
        Ngrid = int(gs[0]) * int(gs[1])
        _key_opts = dict(dtype=torch.int64, device=self.device)
        key_cc_t  = torch.empty(Ngrid, **_key_opts)
        key_u_t   = torch.empty(Ngrid, **_key_opts)
        key_v_t   = torch.empty(Ngrid, **_key_opts)
        # Velocity-blend accumulators (full-grid, like the 2-D keys; zeroed).
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        if blend_eps > 0.0:
            num_u_t = torch.zeros(Ngrid, **_opts); num_v_t = torch.zeros(Ngrid, **_opts)
            den_u_t = torch.zeros(Ngrid, **_opts); den_v_t = torch.zeros(Ngrid, **_opts)
        else:
            num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
            den_u_t = torch.empty(1, **_opts); den_v_t = torch.empty(1, **_opts)

        # 5. Kernel A.
        streaming_sdf_stag_2d_multi(
            sm['F_flat'], sm['F_offsets'],
            sm['body_shapes'], sm['body_meta'], ks['kin'],
            ks['aabb_lo'], ks['aabb_dim'],
            ks['gx'], ks['gy'],
            float(comp.h), int(ks['max_vol']),
            comp.sdf_val, sdf_u_tmp, sdf_v_tmp,
            bU_tmp, bV_tmp,
            key_cc_t, key_u_t, key_v_t,
            int(getattr(self, '_sdf_interp_method', 0)),
            int(ks['dirty_i0']), int(ks['dirty_j0']),
            int(ks['dirty_Ai']), int(ks['dirty_Aj']),
            num_u_t, num_v_t, den_u_t, den_v_t, float(blend_eps),
        )

        # 6. Kernel B: fused BDIM2 + variable-density coefficients.
        if (self.apply_bdim_sigma
                and self._sigma_shifts is not None
                and bool(self._sigma_shifts.any())):
            bdim_vardens_sigma_2d(
                primes[0], primes[1],
                sdf_u_tmp, sdf_v_tmp,
                bU_tmp, bV_tmp,
                self.u0, self.v0,
                self._ch_persist, self._cv_persist,
                key_u_t, key_v_t,
                self._sigma_shifts,
                float(comp.eps), float(self.rho_body), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']),
                int(self.bdim_mu0_projection),
            )
        else:
            bdim_vardens_2d(
                primes[0], primes[1],
                sdf_u_tmp, sdf_v_tmp,
                bU_tmp, bV_tmp,
                self.u0, self.v0,
                self._ch_persist, self._cv_persist,
                float(comp.eps), float(self.rho_body), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']),
                int(self.bdim_mu0_projection),
            )

        # 7. Free per-step temporaries before the pressure projection.
        del sdf_u_tmp, sdf_v_tmp, bU_tmp, bV_tmp, primes
        del key_cc_t, key_u_t, key_v_t

        # 8. Boundary conditions on the BDIM-corrected velocity.
        self.adv_diff_solver.set_BCs(self.u0, self.v0)

        # 9. Pressure projection.
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
        )
        vels_out = out[:-1]
        p_out    = out[-1]

        # 10. Optional sponge / yield damping + final BC pass.
        if self.use_sponge:
            vels_out = self.apply_sponge_damping(*vels_out)
        if self.use_yield_damping:
            vels_out = self.apply_yield_damping(*vels_out)
        self.adv_diff_solver.set_BCs(*vels_out)

        return (*vels_out, p_out)

    def _fluid_step_kernel_3d(self, u, v, w_vel, p, timestep):
        """Phase-I 3-D kernel fluid step.

        Replaces the chain
            _apply_bdim_all_axes -> _compute_variable_density_coefficients
        with two CUDA kernels (Kernel A + Kernel B).  Kernel A streams
        the union SDF and rigid body face velocities into per-step
        temporaries; Kernel B fuses the BDIM2 velocity update with the
        variable-density Poisson coefficient calculation, computing
        ``mu0``, ``mu1`` and the unit normals in CUDA thread registers
        only.  No persistent staggered SDF / body-velocity /
        winning_rho_cc / mu-pack tensors are required.
        """
        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_3d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_3d

        # Lightweight memory debug helper (enabled via LILYTORCH_MEM_DBG=1).
        _mem_dbg = bool(int(os.environ.get('LILYTORCH_MEM_DBG', '0')))
        def _chk(tag, reset=False):
            if _mem_dbg:
                torch.cuda.synchronize()
                if reset:
                    torch.cuda.reset_peak_memory_stats()
                gb = torch.cuda.memory_allocated() / 1024**3
                mx = torch.cuda.max_memory_allocated() / 1024**3
                print(f"[MEM_DBG] {tag:55s}  cur={gb:.3f} GiB  peak={mx:.3f} GiB",
                      flush=True)

        _chk("0-baseline (step start)", reset=True)
        # BDIM-σ: lazily compute per-body sigma shifts on the first
        # fluid step (body SDFs are populated by BDIMhandler.update by
        # now).  Static thereafter.
        if self.apply_bdim_sigma and self._sigma_shifts is None:
            self._compute_sigma_shifts()

        # 1-2. eddy viscosity + advection-diffusion.
        nu_t   = self._compute_nu_t(u, v, w_vel)
        _chk("1-after compute_nu_t")
        primes = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
        _chk("2-after adv_diff.solve (primes allocated)")
        del nu_t   # no longer needed; free before the large temporaries below
        # primes are fresh tensors from solve(); use them as the
        # advdiff inputs to Kernel B and (after the copy below) as the
        # ``u'/v'/w'`` outside-AABB values written into u0/v0/w0.

        # Copy primes -> persistent u0/v0/w0.  Cells outside the dirty
        # AABB carry the advdiff result; Kernel B overrides the AABB
        # sub-block with the BDIM2 result.  This is the same total
        # number of writes the python path performs via clone+inplace.
        self.u0.copy_(primes[0])
        self.v0.copy_(primes[1])
        self.w0.copy_(primes[2])

        # 3. Init persistent var-dens coefficients (once / on resize).
        self._init_var_dens_persist_3d(timestep)

        # 4. Per-step temporaries for Kernel A -> Kernel B.  Allocated
        # only inside the step; freed (via del below) before the pressure
        # projection so its peak working set is not stacked on top of them.
        _opts = dict(device=self.device, dtype=self.dtype)
        _FAR  = 1e4
        gs    = self.grid_shape
        # int64 key scratch buffers for Kernel A (packed body-id + SDF).
        # Sized to dirty_vol (AABB-local indexing) — much smaller than
        # the full grid, reducing transient peak by ~4×(n_grid - dirty_vol)×8 B.
        dirty_vol = int(ks['dirty_Ai']) * int(ks['dirty_Aj']) * int(ks['dirty_Ak'])
        sdf_u_tmp = torch.full(gs, _FAR, **_opts)
        sdf_v_tmp = torch.full(gs, _FAR, **_opts)
        sdf_w_tmp = torch.full(gs, _FAR, **_opts)
        bU_tmp    = torch.zeros(gs, **_opts)
        bV_tmp    = torch.zeros(gs, **_opts)
        bW_tmp    = torch.zeros(gs, **_opts)
        # int64 key scratch buffers for Kernel A (packed body-id + SDF).
        # Allocated per step so they are freed before the pressure projection
        # and do not contribute to persistent baseline memory.
        _key_opts = dict(dtype=torch.int64, device=self.device)
        key_cc_t = torch.empty(dirty_vol, **_key_opts)
        key_u_t  = torch.empty(dirty_vol, **_key_opts)
        key_v_t  = torch.empty(dirty_vol, **_key_opts)
        key_w_t  = torch.empty(dirty_vol, **_key_opts)
        # Velocity-blend accumulators (dirty-vol-local, zeroed each step).
        # Only allocated when the blend is enabled; otherwise tiny stubs so
        # the op signature is satisfied (blend_eps<=0 → kernel ignores them).
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        if blend_eps > 0.0:
            num_u_t = torch.zeros(dirty_vol, **_opts)
            num_v_t = torch.zeros(dirty_vol, **_opts)
            num_w_t = torch.zeros(dirty_vol, **_opts)
            den_u_t = torch.zeros(dirty_vol, **_opts)
            den_v_t = torch.zeros(dirty_vol, **_opts)
            den_w_t = torch.zeros(dirty_vol, **_opts)
        else:
            # 6 distinct stubs — the op marks num/den as mutable (l!..q!),
            # so they must not alias the same storage even when unused.
            num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
            num_w_t = torch.empty(1, **_opts); den_u_t = torch.empty(1, **_opts)
            den_v_t = torch.empty(1, **_opts); den_w_t = torch.empty(1, **_opts)
        _chk("4-after alloc sdf/bUVW/key temps (before Kernel A)")

        # 5. Kernel A: stream SDF + body velocities into the temps,
        #    CC SDF into comp.sdf_val (persistent, used by forces).
        streaming_sdf_stag_3d_multi(
            sm['F_flat'], sm['F_offsets'],
            sm['body_shapes'], sm['body_meta'], ks['kin'],
            ks['aabb_lo'], ks['aabb_dim'],
            ks['gx'], ks['gy'], ks['gz'],
            float(comp.h), int(ks['max_vol']),
            comp.sdf_val, sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
            bU_tmp, bV_tmp, bW_tmp,
            key_cc_t, key_u_t, key_v_t, key_w_t,
            int(getattr(self, '_sdf_interp_method', 0)),
            int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
            int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
            num_u_t, num_v_t, num_w_t, den_u_t, den_v_t, den_w_t,
            float(blend_eps),
        )

        # 6. Kernel B: fused BDIM2 + variable-density coefficients.
        #    Reads primes / SDF / body face velocities; writes u0/v0/w0
        #    and ch/cv/cw inside the dirty AABB.  mu0/mu1/normals are
        #    computed only in registers.
        if (self.apply_bdim_sigma
                and self._sigma_shifts is not None
                and bool(self._sigma_shifts.any())):
            bdim_vardens_sigma_3d(
                primes[0], primes[1], primes[2],
                sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
                bU_tmp, bV_tmp, bW_tmp,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                key_u_t, key_v_t, key_w_t,
                self._sigma_shifts,
                float(comp.eps), float(self.rho_body), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
                int(self.bdim_mu0_projection),
            )
        else:
            bdim_vardens_3d(
                primes[0], primes[1], primes[2],
                sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
                bU_tmp, bV_tmp, bW_tmp,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                float(comp.eps), float(self.rho_body), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
                int(self.bdim_mu0_projection),
            )

        # 7. Free per-step temporaries before the pressure projection
        #    so its peak working set is not stacked on top of them.
        #    Key tensors must outlive Kernel B (read by the BDIM-σ path),
        #    so they are freed here together with the other temporaries.
        del sdf_u_tmp, sdf_v_tmp, sdf_w_tmp, bU_tmp, bV_tmp, bW_tmp, primes
        del key_cc_t, key_u_t, key_v_t, key_w_t
        _chk("7-after del temps+primes (pre-project baseline)")

        # 8. Boundary conditions on the BDIM-corrected velocity.
        self.adv_diff_solver.set_BCs(self.u0, self.v0, self.w0)

        # 9. Pressure projection.  Multigrid only needs ch/cv/cw;
        #    FFT also wants ch_cc but Phase I does not allocate a
        #    cell-centred coefficient buffer.
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist, cw=self._cw_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
            w_vel=self.w0,
        )
        vels_out = out[:-1]
        p_out    = out[-1]
        _chk("9-after project()")

        # 10. Optional sponge / yield damping + final BC pass.
        if self.use_sponge:
            vels_out = self.apply_sponge_damping(*vels_out)
        if self.use_yield_damping:
            vels_out = self.apply_yield_damping(*vels_out)
        self.adv_diff_solver.set_BCs(*vels_out)

        return (*vels_out, p_out)

    def fluid_step(self, *args):
        """One FSI fluid step (advect-BDIM-project).  Called by BDIMhandler.

        Dim-agnostic.  Signature: ``(u, v, p, timestep)`` in 2-D,
        ``(u, v, w, p, timestep)`` in 3-D.  Pipeline (same for both D):

        1. eddy viscosity (Smagorinsky / Carreau) on the cell-centred field
        2. advection-diffusion → clone outputs
        3. BDIM2 forcing on staggered grids, with union-AABB narrow band
           when ``_use_kernels`` is on and bodies are sparse
        4. ``set_BCs`` after BDIM so ghost cells are consistent for project
        5. free BDIM intermediates (mu1, normals) before var-density solve
        6. variable-density Poisson coefficients on staggered + CC grids
        7. free mu0 fields before project (no longer needed)
        8. pressure projection
        9. optional sponge / yield damping
        10. final ``set_BCs`` on the projected velocity
        """
        D = self.ndim
        vels     = args[:D]
        p        = args[D]
        timestep = args[D + 1]

        # Phase-I fused-kernel fast path (2-D and 3-D).  The python
        # reference path stays on the legacy implementation below.
        if self._use_kernels:
            if D == 3:
                return self._fluid_step_kernel_3d(vels[0], vels[1], vels[2],
                                                  p, timestep)
            if D == 2:
                return self._fluid_step_kernel_2d(vels[0], vels[1],
                                                  p, timestep)

        # 1-2. eddy viscosity + advection-diffusion
        nu_t   = self._compute_nu_t(*vels)
        primes = self.adv_diff_solver.solve(*vels, nu_t=nu_t)
        primes = tuple(pp.clone() for pp in primes)

        # 3. BDIM with optional union-AABB narrow band (2-D harmlessly
        #    falls back to full-grid when no sparse SDFs are available).
        self._bdim_union_aabb = (
            self._compute_union_aabb(halo=2) if self._use_kernels else None
        )
        primes = self._apply_bdim_all_axes(primes)
        self._bdim_union_aabb = None

        # 4. enforce BCs on the post-BDIM field (BDIM may touch ghosts
        #    when bodies straddle the domain boundary).
        self.adv_diff_solver.set_BCs(*primes)

        # 5. drop mu1 / normals — projected coefficients only need mu0.
        self.__dict__.update(self._FS_FREE_AFTER_BDIM)

        # 6. variable-density Poisson coefficients.
        #    2-D returns (ch, cv, ch_cc); 3-D returns (ch, cv, cw, ch_cc).
        coeffs       = self._compute_variable_density_coefficients(timestep)
        ch_cc        = coeffs[-1]
        face_coeffs  = coeffs[:-1]

        # 7. drop mu0 fields — projection consumes ch/cv/cw/ch_cc only.
        self.__dict__.update(self._FS_FREE_AFTER_VAR_DENS)

        # 8. pressure projection.  ``ch_cc`` is consumed by the FFT path
        #    only; the multigrid/MGCG path ignores it, so passing it
        #    unconditionally is harmless and removes a branch.
        proj_kwargs = {'ch': face_coeffs[0], 'cv': face_coeffs[1], 'ch_cc': ch_cc}
        if D == 3:
            proj_kwargs['cw']    = face_coeffs[2]
            proj_kwargs['w_vel'] = primes[2]
        out      = self.project(primes[0], primes[1], p, **proj_kwargs)
        vels_out = out[:-1]
        p_out    = out[-1]

        # 9. boundary / yield damping.
        if self.use_sponge:
            vels_out = self.apply_sponge_damping(*vels_out)
        if self.use_yield_damping:
            vels_out = self.apply_yield_damping(*vels_out)

        # 10. final BC pass on the projected velocity.
        self.adv_diff_solver.set_BCs(*vels_out)

        return (*vels_out, p_out)

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
        self.composite_body.update(t, iteration, dt=self.dt)
        # Phase I: in kernel mode (2-D and 3-D), mu0/mu1 and normals are
        # computed in CUDA thread registers inside Kernel B during
        # fluid_step.  No persistent mu/normal buffers are allocated, so
        # the python ``_recompute_mu_normals`` is unused (and would
        # allocate the per-axis mu/normal pack it depends on).
        if not self._use_kernels:
            self._recompute_mu_normals()
        # self.sdf_properties = [[self.composite_body.sdf_val_u]]

        # Gravity body force (predictor-side forward-Euler).  Applied
        # BEFORE fluid_step so the adv-diff + projection see the new
        # momentum and the pressure projection can balance it through
        # the (optional) free-surface ghost-fluid Dirichlet BC.
        if self.use_gravity:
            if self.ndim == 2:
                u, v = self._apply_gravity_body_force(u, v)
            else:
                u, v, w_vel = self._apply_gravity_body_force(u, v, w_vel)

        if self.ndim == 2:
            (u, v, p) = self.fluid_step(u, v, p, self.dt)
            if self.zero_pressure_inside:
                p = torch.where(self.composite_body.sdf_val < 0, 0, p)
            self.u0, self.v0, self.p0 = u, v, p

            if self.compute_forces:
                if self.force_method == "lagrangian":
                    self.forces_lagrangian_2d(u, v, p, iteration)
                else:
                    self.forces_method2(u, v, p, iteration)

        else:
            (u, v, w_vel, p) = self.fluid_step(u, v, w_vel, p, self.dt)
            if self.zero_pressure_inside:
                p = torch.where(self.composite_body.sdf_val < 0, 0, p)
            self.u0, self.v0, self.w0, self.p0 = u, v, w_vel, p

            if self.compute_forces:
                if self.force_method == "lagrangian":
                    self.forces_lagrangian_3d(u, v, w_vel, p, iteration)
                else:
                    self.forces_method2_3d(u, v, w_vel, p, iteration)

        self._apply_force_feedback(iteration, t)
        self.check_explosion(iteration)

        # ---- free-surface bookkeeping (advect + periodic reinit/extend) ----
        if self.free_surface is not None:
            self._fs_post_step(u, v, w_vel, iteration)

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

        if self.compute_forces and self.save_drags:
            self.save_drags_h5()

        # Block until all background I/O is complete before returning
        self.flush_io()
