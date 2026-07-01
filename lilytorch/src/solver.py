
import datetime
import logging
import os
import threading
import warnings
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from lilytorch.src.kernels import (
    streaming_sdf_stag_3d_multi,
    bdim_coeff_3d,
    bdim_coeff_sigma_3d,
    streaming_sdf_stag_2d_multi,
    bdim_coeff_2d,
    bdim_coeff_sigma_2d,
)
from lilytorch.src.advection import AdvDiffSolver, SCHEMES as _ADV_SCHEMES
from lilytorch.src.diagnostics import FlowDiagnostics
from lilytorch.src.body import (body_from_yaml,
                                _mu_normals_batched)
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


def _build_fs_free_dicts(ndim):
    """Build the ``_FS_FREE_AFTER_BDIM`` / ``_FS_FREE_AFTER_BDIM_COEFF`` dicts
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
    after_bdim_coeff = {f'mu0_all_{a}': None for a in axes}
    after_bdim_coeff['mu0_all'] = None
    return after_bdim, after_bdim_coeff


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

        # FlowDiagnostics cadence: compute energy / enstrophy / max-divergence
        # / CFL every ``diagnostics_every`` steps and warn on blow-up / CFL>0.5.
        # 0 (default) disables the monitor entirely; the actual FlowDiagnostics
        # object is created after the output block (it reuses save_path/lock).
        self.diagnostics_every = int(solver.get("diagnostics_every", 0) or 0)
        self.diagnostics       = None

        # Per-step overhead controls (H1/H2 TODO items).
        # check_explosion_every: GPU→CPU sync cadence for NaN/vmax checks.
        #   Default 50 — fast enough to catch blow-ups before they cascade.
        # empty_cache_every: how often to flush the CUDA allocator cache.
        #   Default 200 — reduces nvidia-smi pressure without per-step churn.
        self.check_explosion_every = int(solver.get("check_explosion_every", 50))
        self.empty_cache_every     = int(solver.get("empty_cache_every", 200))

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
        # Maertens–Weymouth body-velocity-divergence RHS correction. Keeps the
        # mu0-weighted projection consistent for overlapping/deforming bodies
        # (no-op for single rigid bodies). See _mw_body_div_correction.
        self._bdim_body_div_correction = bool(
            solver.get("bdim_body_div_correction", False))

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

        # NOTE: ``solver.rho_body`` is intentionally NOT read. The body density
        # is no longer a fluid property: the BDIM Poisson coefficient is the
        # constant-density ``dt*mu0/rho_fluid`` (Weymouth & Yue), the body
        # entering only through ``mu0``. Body weight / buoyancy is the coupling's
        # concern (MuJoCo mass + the external Archimedes term in BDIMhandler,
        # which uses the ANIMAT density). Any ``rho_body`` key in a config is
        # silently ignored.

        self.terminate = False   # flag for early termination (e.g. from NaN detection)

        # ============= convection solver =============
        self.convection_method = solver["convection_method"]
        adv_diff_kwargs = dict(
            BC_type_u=bcs["BC_type_u"], BC_values_u=bcs["BC_values_u"],
            BC_type_v=bcs["BC_type_v"], BC_values_v=bcs["BC_values_v"],
            method=self.convection_method,
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

        # ---- multi-stream advection (CUDA only) -----
        # Each velocity component is dispatched on a separate CUDA stream so
        # u/v/w adv-diff can overlap.  Peak intermediate memory is ~ndim× the
        # sequential path (all rhs tensors are live simultaneously); disable
        # on memory-constrained runs.  No-op on CPU.
        # NOTE: set before the torch.compile decision below — it changes
        # which path solve() takes (and thus whether compile is worthwhile).
        if solver.get("adv_diff_streams", False) and self.device.type == "cuda":
            self.adv_diff_solver._use_streams = True

        # ---- optional torch.compile for adv-diff -----
        # Skip when solve() will take the fused CUDA ``advect_flux_add`` kernel
        # path: that path is a custom op + host syncs, so compiling it gives no
        # speedup and trips dynamo's speculation log ("SpeculationLog diverged"
        # AssertionError on graph-break restart).  Compile only helps the
        # pure-PyTorch fallback (e.g. semi-Lagrangian, multi-stream ND>1).
        self._compile_adv_diff = solver.get("compile_adv_diff", False)
        if (self._compile_adv_diff and self.device.type == "cuda"
                and not self.adv_diff_solver.uses_cuda_flux_kernel):
            self.adv_diff_solver.solve = torch.compile(
                self.adv_diff_solver.solve, mode="default",
            )

        # ---- optional torch.compile for project() -----
        # Fuses divergence + Poisson + velocity-correction ops into fewer
        # kernels.  Graph-breaks at Python branches (poisson_method check,
        # body_div_corr guard), so benefit is partial but non-zero.
        if solver.get("compile_project", False) and self.device.type == "cuda":
            self.project = torch.compile(self.project, mode="default")

        # ---- CUDA graph capture of adv-diff solve -----
        # Eliminates Python dispatch overhead on every adv-diff call.
        # Only supported for constant-viscosity runs (nu_t=None) and schemes
        # without host syncs (not abdquickest).  Lazily captured on the first
        # fluid_step call via _adv_graph_pending flag.
        self._use_cuda_graphs = (
            solver.get("use_cuda_graphs", False)
            and self.device.type == "cuda"
            and not self._compile_adv_diff          # graphs + compile = redundant
            and not solver.get("adv_diff_streams", False)  # graphs + streams = conflict
        )
        self._adv_graph_captured = False

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
        # Eulerian pressure-force readout sub-method (only meaningful when
        # ``force_method == "eulerian"``):
        #
        #   "ndelta" — F = -Σ p·n·δ_ε(φ_body)  (per-body smoothed-delta band
        #              integral; the historical default).
        #   "deltaH" — F = -Σ p·∂_iH_ε(φ_union)  partial-Heaviside readout: the
        #              pressure force density is taken from the UNION SDF (one
        #              closed surface, no internal inter-link seams) so it obeys
        #              summation-by-parts and does not leak the hydrostatic
        #              baseline; it is split back to the individual bodies by a
        #              softmin partition of unity (w_b = softmax(-φ_b/τ),
        #              τ = ``force_ph_blend_cells``·h).  Viscous force/torque are
        #              unchanged (still the per-body δ_ε integral).
        #
        # Mirrors ``TwoPhaseSolver._apply_partition_heaviside`` (python path) in
        # the native CUDA/CPU force kernels.
        _fsm_raw = solver.get("force_submethod", "ndelta")
        if _fsm_raw not in ("ndelta", "deltaH"):
            raise ValueError(
                f"solver.force_submethod must be one of "
                f"('ndelta', 'deltaH'), got {_fsm_raw!r}."
            )
        self.force_submethod = _fsm_raw
        # Softmin partition temperature for the deltaH readout, in grid cells.
        self.force_ph_blend_cells = float(
            solver.get("force_ph_blend_cells", 1.5))
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
        assert self.poisson_method in ("multigrid", "mgcg", "rmgcg", "fft"), \
            f"Unknown poisson_method '{self.poisson_method}'. Choose 'multigrid', 'mgcg', 'rmgcg', or 'fft'."
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
            recycle_k       = solver.get("poisson_recycle_k", 0),
            cuda_graph      = solver.get("poisson_cuda_graph", False),
            cuda_graph_max_cells = solver.get("poisson_cuda_graph_max_cells", 64 ** 3),
        )
        # Degenerate-cell freeze threshold. Cells with |diagonal| < jcap_tol are
        # frozen (iD=0, residual zeroed) — WaterLily's iszero(D) guard. The
        # default 1e-12 is an ABSOLUTE threshold; for the mu0-weighted operator
        # (dt*mu0/rho ~ 1e-7 scale) it only catches the exact-zero interior and
        # leaves the tiny-mu0 band ill-conditioned. Raise it (relative to the
        # ~dt/rho fluid coefficient) to also freeze the near-degenerate band.
        _jct = solver.get("poisson_jcap_tol", None)
        if _jct is not None:
            self.poisson_solver.jcap_tol = float(_jct)

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

          # ===== stacked velocity storage (Step 5 unification) =====
        # u0/v0[/w0] are compat @property views into this single
        # (D, *grid) buffer; see the property block below.  Allocated
        # before set_initial_conditions so the property setters have a
        # backing store to write into.
        self._vel = torch.zeros(
            (self.ndim, *self.grid_shape), device=self.device, dtype=self.dtype)

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

        # ---- flow diagnostics monitor (opt-in via diagnostics_every>0) ----
        if self.diagnostics_every > 0:
            self.diagnostics = FlowDiagnostics(
                nt=self.nt, ndim=self.ndim, h=self.h,
                device=self.device, dtype=self.dtype,
                check_every=self.diagnostics_every,
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
        # ``_FS_FREE_AFTER_BDIM_COEFF_3D`` constants — these are now built
        # from ``self.ndim`` so the 2-D path can use the same mechanism.
        # ----------------------------------------------------------------
        (self._FS_FREE_AFTER_BDIM,
         self._FS_FREE_AFTER_BDIM_COEFF) = _build_fs_free_dicts(self.ndim)

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
            self._init_bdim_coeff_persist_3d(self.dt)
        elif self._use_kernels and self.ndim == 2:
            self._init_bdim_coeff_persist_2d(self.dt)

        # =====================================================================
        # Gravity body force (opt-in via the ``solver.gravity`` block).
        #   solver.gravity: [gx, gy]      # 2-D
        #   solver.gravity: [gx, gy, gz]  # 3-D
        # Used by the two-phase solver (water+air) and any gravity-driven
        # run; ``None`` disables it.
        # =====================================================================
        self._init_gravity(solver.get("gravity", None))

        # Warp Kernel-A CUDA-graph fast path (opt-in) + persistent kernel-step
        # buffer caches (allocated lazily by _kernel_bufs_{2,3}d in graph mode).
        self._kernel_cuda_graph = bool(solver.get("kernel_cuda_graph", False))
        self._kbuf2d = None
        self._kbuf3d = None
        # Periodic MGCG convergence check (1 = check every iter); threaded onto
        # the Poisson sub-solver (its constructor takes no such kwarg).
        if getattr(self, "poisson_solver", None) is not None:
            self.poisson_solver.cg_check_every = int(
                solver.get("poisson_cg_check_every", 1))

    # =====================================================================
    # Gravity body force (opt-in)
    # =====================================================================
    def _init_gravity(self, gravity_cfg):
        """Parse the optional ``solver.gravity`` block.

        ``solver.gravity`` is a list of length ``ndim`` giving the gravity
        body acceleration vector in SI units (m/s²).  When ``None`` the
        body force is disabled and ``self.use_gravity`` is False.
        """
        self.use_gravity = False
        self._gravity = None
        # Well-balanced-gravity attributes default OFF so no-gravity runs
        # (the vast majority) skip the hydrostatic split entirely and stay
        # byte-identical.
        self._wb_gravity = False
        self.p_h         = None
        self._ph_grad    = None
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

        # ---- well-balanced (hydrostatic-split) gravity ------------------
        # Stage 1 (single-phase / uniform density): pre-subtract the analytic
        # hydrostatic gradient from the predictor body force so the Poisson
        # never sees the stiff hydrostatic and produces no spurious flow.
        # ``_build_hydrostatic_reference`` returns None for variable-density
        # solvers (two-phase), which then keep the legacy uniform ``dt*g``
        # body force untouched (Stage 2).  No-gravity runs are unaffected.
        if self.use_gravity:
            p_h = self._build_hydrostatic_reference()
            if p_h is not None:
                self.p_h      = p_h
                self._ph_grad = self.gradient(p_h)   # ∇p_h on the face grids
                self._wb_gravity = True
                print("Well-balanced gravity: hydrostatic split active "
                      "(single-phase, uniform density).")

    def _build_hydrostatic_reference(self):
        """Cell-centred analytic hydrostatic pressure ``p_h`` with
        ``∇p_h = rho*g`` for UNIFORM single-phase density.

        Returns ``rho * (g . x)`` on the cell-centred grid.  The predictor
        subtracts ``(dt/rho)*∇p_h`` (cancelling ``dt*g`` in the interior to
        machine precision -- uniform-density incompressible flow is
        gravity-invariant).  The body-force readout uses the DYNAMIC pressure
        ``p_d`` alone (NOT ``p_d + p_h``): single-phase buoyancy is handled by
        the rigid-body integrator (external Archimedes / MuJoCo), and feeding
        ``p_h`` through the non-gauge-invariant band quadrature leaks a spurious
        horizontal force.  ``self.p_h`` is kept only for optional physical-
        pressure reconstruction (``p_d + p_h``) in plotting.  Variable-density
        solvers (two-phase) override this to return ``None`` -> the legacy
        uniform ``dt*g`` body force is used (Stage 2, see
        milestones/hydrostatic_gravity_stage2_handoff.md).
        """
        g   = self._gravity
        rho = float(self.rho)
        if self.ndim == 2:
            X = self.x.view(-1, 1)
            Y = self.y.view(1, -1)
            p_h = rho * (g[0] * X + g[1] * Y)
        else:
            X = self.x.view(-1, 1, 1)
            Y = self.y.view(1, -1, 1)
            Z = self.z.view(1, 1, -1)
            p_h = rho * (g[0] * X + g[1] * Y + g[2] * Z)
        return p_h.expand(self.grid_shape).contiguous()

    @torch.no_grad()
    def _apply_gravity_body_force(self, *vels):
        """Predictor-side gravity body force, applied inside ``step_`` right
        before ``fluid_step`` so the projection can balance it.

        Legacy path (default; variable-density / two-phase): add the uniform
        ``dt*g`` to every face (``u`` on x-faces, ``v`` on y-faces, ...),
        consistent with the cell-centred projection that follows.

        Well-balanced path (``self._wb_gravity``, single-phase uniform
        density): add ``dt*g - (dt/rho)*∇p_h`` instead.  The analytic
        ``∇p_h = rho*g`` cancels the body force in the INTERIOR to machine
        precision, so the Poisson never sees the stiff hydrostatic gradient
        and produces no spurious flow.  Boundary faces (where the
        backward-difference ``∇p_h`` is zeroed, exactly as in the
        projection's correction) keep the legacy ``dt*g`` and are governed
        by the BCs -- i.e. boundary behaviour is unchanged.
        """
        if not self.use_gravity:
            return vels
        dt = float(self.dt)
        if not self._wb_gravity:
            out = []
            for vel, g_comp in zip(vels, self._gravity):
                if g_comp != 0.0:
                    vel.add_(dt * g_comp)           # in-place
                out.append(vel)
            return tuple(out)
        # ---- well-balanced (hydrostatic-split) gravity ----
        c = dt / float(self.rho)
        out = []
        for vel, g_comp, ph_g in zip(vels, self._gravity, self._ph_grad):
            if g_comp != 0.0:
                vel.add_(dt * g_comp)               # body force
                vel.sub_(c * ph_g)                  # subtract hydrostatic ∇p_h
            out.append(vel)
        return tuple(out)

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


    # --- stacked velocity compat accessors (Step 5 unification) --------
    # u0/v0[/w0] are views into the single ``self._vel`` (D, *grid) buffer.
    # Reads return a (contiguous) row view; assignments copy into the row in
    # place (so external aliases of the buffer stay valid).  The 3-D ``w0``
    # raises AttributeError in 2-D so ``hasattr/getattr(fs, 'w0')`` keeps the
    # legacy "absent in 2-D" semantics that the viewers rely on.
    @property
    def u0(self):
        return self._vel[0]

    @u0.setter
    def u0(self, value):
        self._vel[0] = value

    @property
    def v0(self):
        return self._vel[1]

    @v0.setter
    def v0(self, value):
        self._vel[1] = value

    @property
    def w0(self):
        if self.ndim < 3:
            raise AttributeError("w0 is undefined for a 2-D solver")
        return self._vel[2]

    @w0.setter
    def w0(self, value):
        if self.ndim < 3:
            raise AttributeError("w0 is undefined for a 2-D solver")
        self._vel[2] = value

    # --- moved to lilytorch/src/forces.py (item #8) ---
    forces_method1 = forces.forces_method1
    forces_method2 = forces.forces_method2
    forces_method2_3d = forces.forces_method2_3d
    # Lagrangian (surface-integral) force methods — phase 2 of force_method
    # rework.  See ``forces.forces_lagrangian_2d`` / ``forces_lagrangian_3d``.
    forces_lagrangian_2d = forces.forces_lagrangian_2d
    forces_lagrangian_3d = forces.forces_lagrangian_3d




    def _mw_body_div_correction(self, bU, bV, bW=None):
        """Maertens–Weymouth body-velocity-divergence source term.

        Returns the cell-centred field ``(1 - mu0) * (∇·u_body)`` that must be
        **subtracted** from the Poisson RHS so the projection enforces
        ``∇·u = (1-mu0)∇·u_b`` (fluid solenoidal; the solid is allowed to carry
        the body's own divergence) instead of forcing ``∇·u = 0`` even inside
        the body.  For a single rigid body ``∇·u_b = 0`` so this is a no-op; it
        is nonzero only for overlapping / smoothed-overlap (e.g. convexify)
        links, where it is exactly the term that keeps the mu0-weighted
        (degenerate-inside) Poisson operator consistent and prevents the
        seam blow-up.  See ``docs/immersed_boundary.rst``.
        """
        div_b = self.divergence(bU, bV, w=bW)        # ∇·u_b at cell centres
        phi   = self.composite_body.sdf_val          # CC union SDF
        eps   = float(self.eps)
        deps  = (phi / eps).clamp(-1.0, 1.0)
        # Smoothed Heaviside mu0 (matches bdim_one_axis / _mu_normals).
        mu0   = 0.5 * (1.0 + deps + torch.sin(torch.pi * deps) / torch.pi)
        return (1.0 - mu0) * div_b

    def project(self, u, v, p, w_vel=None, w=1.0, *,
                ch=None, cv=None, cw=None, ch_cc=None, body_div_corr=None):
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
            (``dt * mu0 / rho_fluid`` on the respective face grids).
            When *None* (default), the standard BDIM coefficients
            ``(w*dt/rho) * mu0`` are used.  Pass custom coefficients
            for variable-density formulations (e.g. the two-phase solver,
            where ``rho_fluid`` is the VOF water/air blend).  The body enters
            only through ``mu0`` -- its density is never a fluid coefficient.
        ch_cc : tensor or None
            Cell-centred coefficient ``dt / rho_eff_cc`` for the FFT
            Poisson RHS.  When provided the FFT path solves
            ``∇²p = div / ch_cc`` (i.e. ``div * rho_eff_cc / dt``) and
            then corrects using the staggered *ch/cv/cw*.  When *None*
            (default) the FFT path falls back to a single scalar
            coefficient (constant-density behaviour).
        """

        coeff = w * self.dt / self.rho

        self.poisson_solver.dirichlet_mask = None

        if self.poisson_method == "fft":
            # ---- FFT solver: CONSTANT-coefficient projection ----
            # Full-grid div is computed here (FFT solver needs the ghost-cell
            # wrapper; body_div_corr is also full-grid).
            # The FFT solver only solves the constant-coefficient Poisson
            # ``∇²p = div/c``.  It does NOT solve the variable-coefficient
            # (mu0-weighted / variable-density) BDIM Poisson — that is the
            # multigrid / MGCG path.  Using a mu0-weighted or variable-
            # density coefficient in the velocity correction is therefore
            # *inconsistent* with the constant-coefficient pressure: the
            # mu0/ρ_eff terms belong to the BDIM variable-coefficient
            # operator, which FFT does not invert.  A consistent FFT
            # projection uses a single SCALAR coefficient for BOTH the RHS
            # divisor and the velocity correction, which is exactly
            # divergence-free:
            #     ∇·(u − c∇p) = div − c∇²p = div − c·(div/c) = 0.
            # i.e. a clean constant-density Chorin projection (WaterLily-
            # style).  ``ch_cc`` and any mu0-weighted ``ch/cv/cw`` tensors
            # are intentionally ignored on the FFT path; variable-density /
            # BDIM-weighted projection requires poisson_method
            # "multigrid"/"mgcg".  (Because the correction no longer
            # multiplies the face-grid ``ch/cv/cw`` buffers, the FFT path is
            # automatically compatible with kernel-mode coefficients too.)
            # The RHS divisor is inherently bounded (a scalar), so the
            # f650945 band-singularity cannot occur here.
            div = self.divergence(u, v, w=w_vel)
            if body_div_corr is not None:
                div = div - body_div_corr
            c_scalar = ch if (ch is not None and not isinstance(ch, torch.Tensor)) else coeff
            _rhs = div / c_scalar
            del div
            p = self.poisson_solverFFT.solve(_rhs)
            if self.ndim == 2:
                (p_x, p_y) = self.gradient(p)
                u = u - c_scalar * p_x
                v = v - c_scalar * p_y
            else:
                (p_x, p_y, p_z) = self.gradient(p)
                u     = u - c_scalar * p_x
                v     = v - c_scalar * p_y
                w_vel = w_vel - c_scalar * p_z
        else:
            # ---- Multigrid / MGCG solver (variable-coefficient Poisson) ----
            # T3a: compute interior-only divergence directly — no full-grid
            # ghost-cell wrapper — so the buffer is interior-sized from the
            # start.  For the Python path, pre-scale by h² in-place and pass
            # pre_scaled=True to skip the redundant h²·f copy inside
            # solve_multigrid / solve_mgcg (~1 interior-field saved at peak).
            # Maertens–Weymouth correction is sliced to the interior before
            # the optional in-place h² scaling.
            div = ops.divergence_interior(
                u, v, self.dx, self.dy,
                w=w_vel, dz=getattr(self, 'dz', None),
            )
            if body_div_corr is not None:
                _sl = (slice(1, -1),) * self.ndim
                div.sub_(body_div_corr[_sl])

            # Pre-scale only on the Python path; the native CUDA solver
            # applies h² internally (h2=self.h2 kernel parameter) so
            # passing a pre-scaled RHS would double-scale.
            _pre_scaled = not self.poisson_solver.use_kernels
            if _pre_scaled:
                div.mul_(self.poisson_solver.h2)

            has_custom_coeffs = any(arr is not None for arr in (ch, cv, cw))
            if ch is None:
                ch = coeff * self.mu0_all_u
            if cv is None:
                cv = coeff * self.mu0_all_v

            # Select solve method: recycled MGCG, MGCG, or standalone multigrid
            _poisson_solve = {
                "rmgcg": self.poisson_solver.solve_rmgcg,
                "mgcg":  self.poisson_solver.solve_mgcg,
            }.get(self.poisson_method, self.poisson_solver.solve_multigrid)

            # Variable-density custom coefficients are coupled to a moving
            # immersed geometry; reusing the previous pressure field can carry
            # stale body-interior/interface values and destabilize the solve.
            # (Confirmed by A/B on the two-phase surface-pool case: warm-start
            # made the solve 2.6-4x SLOWER and less stable -- with
            # zero_pressure_inside zeroing the body interior, the previous p is
            # a poor guess whose sharp body-boundary mismatch the smoother must
            # undo, so multigrid runs its full cycle budget every step instead
            # of early-exiting. Kept disabled for the custom-coeff path.)
            if self.poisson_warm_start and not has_custom_coeffs:
                p0 = p
            else:
                p0 = torch.zeros_like(p)

            if self.ndim == 2:
                p, _ = _poisson_solve(
                    div,
                    p0,
                    ch=ch[1:, 1:-1],
                    cv=cv[1:-1, 1:],
                    pre_scaled=_pre_scaled,
                )
                del div                   # T3a: free interior RHS before correction
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
                    div,
                    p0,
                    ch=(ch  if _face_grid else ch[1:, 1:-1, 1:-1]),
                    cv=(cv  if _face_grid else cv[1:-1, 1:, 1:-1]),
                    cw=(cw  if _face_grid else cw[1:-1, 1:-1, 1:]),
                    pre_scaled=_pre_scaled,
                )
                del div                   # T3a: free interior RHS before correction
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

        Used by ``_compute_bdim_coefficients`` (Python-mode
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
    _BDIM_COEFF_PERSIST_NAMES = ('_ch_persist', '_cv_persist', '_cw_persist',
                               '_ch_cc_persist')

    def _compute_bdim_coefficients(self, timestep):
        """BDIM2 Poisson coefficients ``c = dt * mu0 / rho_fluid`` (FSI).

        Returns ``(ch, cv, ch_cc)`` for 2-D or ``(ch, cv, cw, ch_cc)``
        for 3-D, where:

            * ``ch, cv, cw`` -- staggered ``dt * mu0 / rho_fluid`` on faces.
            * ``ch_cc`` -- cell-centred ``dt / rho_fluid`` (FFT RHS divisor).

        This is the Weymouth & Yue (2011) / Maertens & Weymouth (2015) form:
        the Poisson coefficient is ``(1 - delta^B) / rho = mu0 / rho_fluid``.
        The ``mu0`` factor makes the velocity-correction vanish EXACTLY inside
        the body (mu0=0), preserving the imposed body velocity and avoiding the
        ill-conditioned band Poisson.  The body enters ONLY through ``mu0`` (its
        geometry) -- the body **density never appears here**: its weight /
        inertia / buoyancy is the rigid-body coupling's concern (MuJoCo + the
        external Archimedes term in ``BDIMhandler``), not a fluid property.
        (``TwoPhaseSolver`` overrides this with the VOF water/air density.)

        Narrow-band fast-path (kernel mode + sparse bodies, 2-D and 3-D)
        ----------------------------------------------------------------
        Outside the union AABB ``mu0 = 1`` everywhere, so the coefficients
        are constant (``dt / rho_fluid``).  Persistent full-grid buffers
        (``_ch_persist``, ``_cv_persist``, ``_cw_persist`` (3-D only),
        ``_ch_cc_persist``) are pre-filled once with that default and only
        the union sub-block is overwritten each step, avoiding ``D + 1``
        full-grid divisions.
        """
        D       = self.ndim
        _rho_f  = float(self.rho)
        # mu0-weighted (BDIM2) numerator vs plain dt/rho — see the
        # ``bdim_mu0_projection`` flag set in __init__.
        #
        # The cell-centred ``ch_cc`` (FFT-Poisson RHS divisor) must stay
        # BOUNDED: the constant-coefficient FFT solver divides by it
        # (``div / ch_cc``), and a mu0 factor drives ``ch_cc -> 0`` in the
        # BDIM band -> singular -> the f650945 explosion.  So ch_cc omits mu0.
        # The staggered correction coefficients (faces) KEEP mu0 so the
        # velocity correction vanishes inside the body (mu0=0) and preserves
        # the imposed no-slip body velocity.  ``_cc_no_mu0`` marks the cc grid.
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
            # Reuse the AABB cached in _fluid_step_kernel_{2,3}d when available
            # (it was computed at step 3 and kept alive through step 6).
            u_aabb = (self._bdim_union_aabb
                      if self._bdim_union_aabb is not None
                      else self._compute_union_aabb(halo=2, bucket=16))
            if u_aabb is not None:
                _dt_over_rhofluid = float(timestep / float(self.rho))
                names    = self._BDIM_COEFF_PERSIST_NAMES
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

                # Constant-density BDIM2: faces ``dt*mu0/rho``, cc ``dt/rho``.
                # The body enters via mu0 only -- no body density.
                for name, mu in zip(face_names + (cc_name,), mu_grids):
                    _num = timestep if (name == cc_name or not _mu0w) else timestep * mu[usl]
                    getattr(self, name)[usl] = _num / _rho_f

                return (*(getattr(self, n) for n in face_names),
                        getattr(self, cc_name))

        # ---- full-grid fallback ----------------------------------------
        # cc grid (FFT RHS, last element) omits mu0 so ch_cc stays bounded;
        # the correction faces keep mu0 (preserve no-slip body velocity).
        #
        # When ``bdim_mu0_projection`` is False the coefficient is the
        # constant ``dt/rho`` everywhere; we must still produce a tensor
        # of the same shape as the mu0 grid so that downstream slicing
        # (``ch[1:, 1:-1]`` etc.) works.  The reference shape is taken
        # from ``mu0_all_u`` (the first staggered mu0 grid).
        _ref_shape = mu_grids[0].shape if mu_grids[0] is not None else self.grid_shape
        _scalar_coeff = timestep / _rho_f
        out = []
        for i, mu in enumerate(mu_grids):
            is_cc = (i == len(mu_grids) - 1)
            if is_cc or not _mu0w:
                out.append(torch.full(_ref_shape, _scalar_coeff,
                                      dtype=self.dtype, device=self.device))
            else:
                out.append(timestep * mu / _rho_f)
        return tuple(out)

    # ------------------------------------------------------------------
    #   Phase-I kernel-mode 3-D fluid step  (Kernel A + Kernel B)
    # ------------------------------------------------------------------
    def _init_bdim_coeff_persist_3d(self, timestep):
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

    def _init_bdim_coeff_persist_2d(self, timestep):
        """2-D analogue of :meth:`_init_bdim_coeff_persist_3d`."""
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

    def _kernel_bufs_2d(self, gs, Ngrid, blend_on):
        """Lazily-allocated persistent streaming temporaries for the 2-D graph
        path (sdf_u/v, body_u/v, key_*, num/den), reused every step."""
        c = self._kbuf2d
        if c is None or c["gs"] != gs or c["dtype"] != self.dtype \
                or c["blend"] != blend_on:
            o = dict(device=self.device, dtype=self.dtype)
            ki = dict(dtype=torch.int64, device=self.device)
            nb = Ngrid if blend_on else 1
            c = dict(gs=gs, dtype=self.dtype, blend=blend_on,
                     sdf_u=torch.empty(gs, **o), sdf_v=torch.empty(gs, **o),
                     bU=torch.empty(gs, **o), bV=torch.empty(gs, **o),
                     key_cc=torch.empty(Ngrid, **ki), key_u=torch.empty(Ngrid, **ki),
                     key_v=torch.empty(Ngrid, **ki),
                     num_u=torch.empty(nb, **o), num_v=torch.empty(nb, **o),
                     den_u=torch.empty(nb, **o), den_v=torch.empty(nb, **o))
            self._kbuf2d = c
        return c

    def _kernel_bufs_3d(self, gs, blend_on):
        """3-D analogue of :meth:`_kernel_bufs_2d` (adds w-axis buffers).  The
        graph path is non-σ, so the key buffers are unused — they are size-1
        dummies, keeping the cache independent of the per-step dirty_vol (which
        changes as the body moves, and would otherwise thrash the graph)."""
        c = self._kbuf3d
        if c is None or c["gs"] != gs or c["dtype"] != self.dtype \
                or c["blend"] != blend_on:
            o = dict(device=self.device, dtype=self.dtype)
            ki = dict(dtype=torch.int64, device=self.device)
            nb = 1  # blend stays on the eager path → no full-grid num/den here
            c = dict(gs=gs, dtype=self.dtype, blend=blend_on,
                     sdf_u=torch.empty(gs, **o), sdf_v=torch.empty(gs, **o),
                     sdf_w=torch.empty(gs, **o),
                     bU=torch.empty(gs, **o), bV=torch.empty(gs, **o),
                     bW=torch.empty(gs, **o),
                     key_cc=torch.empty(1, **ki), key_u=torch.empty(1, **ki),
                     key_v=torch.empty(1, **ki), key_w=torch.empty(1, **ki),
                     num_u=torch.empty(nb, **o), num_v=torch.empty(nb, **o),
                     num_w=torch.empty(nb, **o), den_u=torch.empty(nb, **o),
                     den_v=torch.empty(nb, **o), den_w=torch.empty(nb, **o))
            self._kbuf3d = c
        return c

    # ── Kernel A/B (2-D) on Warp ─────────────────────────────────────────────
    def _fluid_step_kernel_2d(self, u, v, p, timestep):
        """2-D kernel fluid step with Kernel A (streaming SDF) and Kernel B
        (fused BDIM2 + variable-density Poisson coefficients) routed to the Warp
        single-source ports via :mod:`lilytorch.src.kernels`.

        Body copied verbatim from ``lilytorch.src.solver.FluidSolver``; the only
        changes are: (1) the two ``streaming_sdf_stag_2d_multi`` /
        ``bdim_coeff_2d`` calls dispatch to the Warp ports; (2) the σ path
        runs on Warp too — the streaming bridge emits the winning body-id into
        ``key_u/key_v`` (``emit_keys``) and the Warp σ Kernel B reads it (Item
        5); (3) ``comp.sdf_val`` is pre-filled to ``+FAR`` (the Warp
        ``atomic_min`` needs it; the native op initialised it internally)."""
        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_2d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_2d

        # BDIM-σ: lazily compute the per-body thin-body shifts on first use,
        # then decide whether the σ Kernel B path is active this step.
        if self.apply_bdim_sigma and self._sigma_shifts is None:
            self._compute_sigma_shifts()
        sigma_active = (self.apply_bdim_sigma
                        and self._sigma_shifts is not None
                        and bool(self._sigma_shifts.any()))

        # 1-2. eddy viscosity + advection-diffusion.
        nu_t   = self._compute_nu_t(u, v)
        primes = self.adv_diff_solver.solve(u, v, nu_t=nu_t)
        self.u0.copy_(primes[0])
        self.v0.copy_(primes[1])

        # 3. Init persistent var-dens coefficients (once / on resize).
        self._init_bdim_coeff_persist_2d(timestep)

        # 4. Per-step temporaries for Kernel A -> Kernel B.
        _opts = dict(device=self.device, dtype=self.dtype)
        _FAR  = 1e4
        gs    = self.grid_shape
        Ngrid = int(gs[0]) * int(gs[1])
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        blend_on = blend_eps > 0.0
        # Opt-in CUDA-graph fast path needs PERSISTENT buffers (stable pointers).
        # σ / blend stay on the eager path (graph capture is non-σ, non-blend).
        graph_mode = (getattr(self, "_kernel_cuda_graph", False)
                      and not sigma_active and not blend_on)
        if graph_mode:
            # Persistent buffers; the SDF→FAR / body→0 resets are folded into the
            # bridge's captured graph (no per-step torch fills here).
            c = self._kernel_bufs_2d(gs, Ngrid, blend_on)
            sdf_u_tmp, sdf_v_tmp = c["sdf_u"], c["sdf_v"]
            bU_tmp, bV_tmp = c["bU"], c["bV"]
            key_cc_t, key_u_t, key_v_t = c["key_cc"], c["key_u"], c["key_v"]
            num_u_t, num_v_t = c["num_u"], c["num_v"]
            den_u_t, den_v_t = c["den_u"], c["den_v"]
        else:
            sdf_u_tmp = torch.full(gs, _FAR, **_opts)
            sdf_v_tmp = torch.full(gs, _FAR, **_opts)
            bU_tmp    = torch.zeros(gs, **_opts)
            bV_tmp    = torch.zeros(gs, **_opts)
            _key_opts = dict(dtype=torch.int64, device=self.device)
            key_cc_t  = torch.empty(Ngrid, **_key_opts)
            key_u_t   = torch.empty(Ngrid, **_key_opts)
            key_v_t   = torch.empty(Ngrid, **_key_opts)
            if blend_on:
                num_u_t = torch.zeros(Ngrid, **_opts); num_v_t = torch.zeros(Ngrid, **_opts)
                den_u_t = torch.zeros(Ngrid, **_opts); den_v_t = torch.zeros(Ngrid, **_opts)
            else:
                num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
                den_u_t = torch.empty(1, **_opts); den_v_t = torch.empty(1, **_opts)

        # Warp Kernel A's atomic-min needs the CC SDF pre-filled to +FAR.  In
        # graph mode the bridge folds this reset into the captured graph.
        if not graph_mode:
            comp.sdf_val.fill_(_FAR)

        # 5. Kernel A (Warp).
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
            emit_keys=sigma_active, use_graph=graph_mode,
        )

        # 6. Kernel B (Warp): fused BDIM2 + variable-density coefficients.
        #    σ variant (thin bodies) reads the body-id keys emitted by Kernel A.
        if sigma_active:
            bdim_coeff_2d(
                primes[0], primes[1],
                sdf_u_tmp, sdf_v_tmp,
                bU_tmp, bV_tmp,
                self.u0, self.v0,
                self._ch_persist, self._cv_persist,
                float(comp.eps), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']),
                int(self.bdim_mu0_projection),
                key_u=key_u_t, key_v=key_v_t,
                sigma_shifts=self._sigma_shifts,
            )
        else:
            bdim_coeff_2d(
                primes[0], primes[1],
                sdf_u_tmp, sdf_v_tmp,
                bU_tmp, bV_tmp,
                self.u0, self.v0,
                self._ch_persist, self._cv_persist,
                float(comp.eps), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']),
                int(self.bdim_mu0_projection),
            )

        # 6b. Maertens–Weymouth body-divergence RHS correction (before free).
        _body_div_corr = (
            self._mw_body_div_correction(bU_tmp, bV_tmp)
            if self._bdim_body_div_correction else None)

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
            body_div_corr=_body_div_corr,
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

    # ── Kernel A/B (3-D) on Warp ─────────────────────────────────────────────
    def _fluid_step_kernel_3d(self, u, v, w_vel, p, timestep):
        """3-D kernel fluid step with Kernel A (streaming SDF) and Kernel B
        (fused BDIM2 + variable-density Poisson coefficients) routed to the Warp
        single-source ports via :mod:`lilytorch.src.kernels`.

        Body copied from ``lilytorch.src.solver.FluidSolver._fluid_step_kernel_3d``;
        the only changes are: (1) the two ``streaming_sdf_stag_3d_multi`` /
        ``bdim_coeff_3d`` calls dispatch to the Warp ports; (2) the σ path
        runs on Warp too — the streaming bridge emits the winning body-id into
        the dirty-local ``key_{u,v,w}`` (``emit_keys``) and the Warp σ Kernel B
        reads it (Item 5); (3) ``comp.sdf_val`` is pre-filled to ``+FAR`` (the
        Warp ``atomic_min`` needs it; the native op initialised it internally);
        (4) the verbose ``_chk`` memory-debug instrumentation is dropped (it is
        not part of the kernel contract)."""
        comp = self.composite_body
        ks = getattr(comp, '_kernel_step', None)
        if ks is None or 'dirty_i0' not in ks:
            raise RuntimeError(
                "_fluid_step_kernel_3d called but composite_body has no "
                "Phase-I _kernel_step bookkeeping; was BDIMhandler.update() "
                "invoked first?"
            )
        sm = comp._kernel_static_3d

        # BDIM-σ: lazily compute per-body thin-body shifts, then decide whether
        # the σ Kernel B path is active this step.
        if self.apply_bdim_sigma and self._sigma_shifts is None:
            self._compute_sigma_shifts()
        sigma_active = (self.apply_bdim_sigma
                        and self._sigma_shifts is not None
                        and bool(self._sigma_shifts.any()))

        # 1-2. eddy viscosity + advection-diffusion.
        nu_t   = self._compute_nu_t(u, v, w_vel)
        primes = self.adv_diff_solver.solve(u, v, w_vel, nu_t=nu_t)
        del nu_t
        self.u0.copy_(primes[0])
        self.v0.copy_(primes[1])
        self.w0.copy_(primes[2])

        # 3. Init persistent var-dens coefficients (once / on resize).
        self._init_bdim_coeff_persist_3d(timestep)

        # 4. Per-step temporaries for Kernel A -> Kernel B.
        _opts = dict(device=self.device, dtype=self.dtype)
        _FAR  = 1e4
        gs    = self.grid_shape
        dirty_vol = int(ks['dirty_Ai']) * int(ks['dirty_Aj']) * int(ks['dirty_Ak'])
        blend_eps = self._body_vel_blend_cells * float(comp.h)
        blend_on = blend_eps > 0.0
        graph_mode = (getattr(self, "_kernel_cuda_graph", False)
                      and not sigma_active and not blend_on)
        if graph_mode:
            c = self._kernel_bufs_3d(gs, blend_on)
            sdf_u_tmp, sdf_v_tmp, sdf_w_tmp = c["sdf_u"], c["sdf_v"], c["sdf_w"]
            bU_tmp, bV_tmp, bW_tmp = c["bU"], c["bV"], c["bW"]
            key_cc_t, key_u_t = c["key_cc"], c["key_u"]
            key_v_t, key_w_t = c["key_v"], c["key_w"]
            num_u_t, num_v_t, num_w_t = c["num_u"], c["num_v"], c["num_w"]
            den_u_t, den_v_t, den_w_t = c["den_u"], c["den_v"], c["den_w"]
            # SDF→FAR / body→0 resets are folded into the bridge's captured graph.
        else:
            sdf_u_tmp = torch.full(gs, _FAR, **_opts)
            sdf_v_tmp = torch.full(gs, _FAR, **_opts)
            sdf_w_tmp = torch.full(gs, _FAR, **_opts)
            bU_tmp    = torch.zeros(gs, **_opts)
            bV_tmp    = torch.zeros(gs, **_opts)
            bW_tmp    = torch.zeros(gs, **_opts)
            _key_opts = dict(dtype=torch.int64, device=self.device)
            key_cc_t = torch.empty(dirty_vol, **_key_opts)
            key_u_t  = torch.empty(dirty_vol, **_key_opts)
            key_v_t  = torch.empty(dirty_vol, **_key_opts)
            key_w_t  = torch.empty(dirty_vol, **_key_opts)
            if blend_on:
                num_u_t = torch.zeros(dirty_vol, **_opts)
                num_v_t = torch.zeros(dirty_vol, **_opts)
                num_w_t = torch.zeros(dirty_vol, **_opts)
                den_u_t = torch.zeros(dirty_vol, **_opts)
                den_v_t = torch.zeros(dirty_vol, **_opts)
                den_w_t = torch.zeros(dirty_vol, **_opts)
            else:
                num_u_t = torch.empty(1, **_opts); num_v_t = torch.empty(1, **_opts)
                num_w_t = torch.empty(1, **_opts); den_u_t = torch.empty(1, **_opts)
                den_v_t = torch.empty(1, **_opts); den_w_t = torch.empty(1, **_opts)

        # Warp Kernel A's atomic-min needs the CC SDF pre-filled to +FAR.  In
        # graph mode the bridge folds this reset into the captured graph.
        if not graph_mode:
            comp.sdf_val.fill_(_FAR)

        # 5. Kernel A (Warp).
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
            emit_keys=sigma_active, use_graph=graph_mode,
        )

        # 6. Kernel B (Warp): fused BDIM2 + variable-density coefficients.
        #    σ variant (thin bodies) reads the body-id keys emitted by Kernel A.
        if sigma_active:
            bdim_coeff_3d(
                primes[0], primes[1], primes[2],
                sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
                bU_tmp, bV_tmp, bW_tmp,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                float(comp.eps), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
                int(self.bdim_mu0_projection),
                key_u=key_u_t, key_v=key_v_t, key_w=key_w_t,
                sigma_shifts=self._sigma_shifts,
            )
        else:
            bdim_coeff_3d(
                primes[0], primes[1], primes[2],
                sdf_u_tmp, sdf_v_tmp, sdf_w_tmp,
                bU_tmp, bV_tmp, bW_tmp,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                float(comp.eps), float(self.rho),
                float(timestep), float(comp.h),
                int(ks['dirty_i0']), int(ks['dirty_j0']), int(ks['dirty_k0']),
                int(ks['dirty_Ai']), int(ks['dirty_Aj']), int(ks['dirty_Ak']),
                int(self.bdim_mu0_projection),
            )

        # 6b. Maertens–Weymouth body-divergence RHS correction (before free).
        _body_div_corr = (
            self._mw_body_div_correction(bU_tmp, bV_tmp, bW_tmp)
            if self._bdim_body_div_correction else None)

        # 7. Free per-step temporaries before the pressure projection.
        del sdf_u_tmp, sdf_v_tmp, sdf_w_tmp, bU_tmp, bV_tmp, bW_tmp, primes
        del key_cc_t, key_u_t, key_v_t, key_w_t

        # 8. Boundary conditions on the BDIM-corrected velocity.
        self.adv_diff_solver.set_BCs(self.u0, self.v0, self.w0)

        # 9. Pressure projection.
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist, cw=self._cw_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
            w_vel=self.w0,
            body_div_corr=_body_div_corr,
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

        # Lazy CUDA graph capture: first call after construction, once the
        # body SDFs are valid (so warmup runs inside the capture are physical).
        if self._use_cuda_graphs and not self._adv_graph_captured:
            self._capture_adv_cuda_graph()
            self._adv_graph_captured = True

        # 1-2. eddy viscosity + advection-diffusion
        nu_t   = self._compute_nu_t(*vels)
        primes = self.adv_diff_solver.solve(*vels, nu_t=nu_t)
        primes = tuple(pp.clone() for pp in primes)

        # 3. BDIM with optional union-AABB narrow band (2-D harmlessly
        #    falls back to full-grid when no sparse SDFs are available).
        # Compute the AABB once and keep it alive through step 6 so that
        # _compute_bdim_coefficients can reuse it without a second call.
        self._bdim_union_aabb = (
            self._compute_union_aabb(halo=2) if self._use_kernels else None
        )
        primes = self._apply_bdim_all_axes(primes)
        # Do NOT reset _bdim_union_aabb here — _compute_bdim_coefficients reuses it.

        # 4. enforce BCs on the post-BDIM field (BDIM may touch ghosts
        #    when bodies straddle the domain boundary).
        self.adv_diff_solver.set_BCs(*primes)

        # 5. drop mu1 / normals — projected coefficients only need mu0.
        self.__dict__.update(self._FS_FREE_AFTER_BDIM)

        # 6. variable-density Poisson coefficients.
        #    2-D returns (ch, cv, ch_cc); 3-D returns (ch, cv, cw, ch_cc).
        coeffs       = self._compute_bdim_coefficients(timestep)
        self._bdim_union_aabb = None   # release after both step-3 and step-6 are done
        ch_cc        = coeffs[-1]
        face_coeffs  = coeffs[:-1]

        # 7. drop mu0 fields — projection consumes ch/cv/cw/ch_cc only.
        self.__dict__.update(self._FS_FREE_AFTER_BDIM_COEFF)

        # 8. pressure projection.  ``ch_cc`` is consumed by the FFT path
        #    only; the multigrid/MGCG path ignores it, so passing it
        #    unconditionally is harmless and removes a branch.
        proj_kwargs = {'ch': face_coeffs[0], 'cv': face_coeffs[1], 'ch_cc': ch_cc}
        if D == 3:
            proj_kwargs['cw']    = face_coeffs[2]
            proj_kwargs['w_vel'] = primes[2]
        if self._bdim_body_div_correction:
            _cb = self.composite_body
            proj_kwargs['body_div_corr'] = self._mw_body_div_correction(
                _cb.body_u, _cb.body_v,
                getattr(_cb, 'body_w', None) if D == 3 else None)
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

    def _capture_adv_cuda_graph(self):
        """Capture a CUDA graph for the adv-diff solve (constant-viscosity path).

        Uses ``torch.cuda.make_graphed_callables``, which internally runs
        ``num_warmup_iters`` un-captured passes then records one replay graph.
        Input tensors are copied in each call; outputs are freshly allocated.

        Constraints (checked here, silently skip if violated):
        - Scheme must not have host syncs (abdquickest uses .item()).
        - nu_t must be None (Smagorinsky not supported in graph path).
        - Not compatible with multi-stream (separate streams inside a graph
          are not captured by make_graphed_callables).
        """
        adv = self.adv_diff_solver
        if adv._scheme_name == 'abdquickest':
            print("[cuda_graph] skip — abdquickest requires host sync for CFL")
            return
        if getattr(self, '_smagorinsky_cs', 0.0) > 0:
            print("[cuda_graph] skip — Smagorinsky requires nu_t tensor input")
            return

        D = self.ndim
        samples = (self.u0.clone(), self.v0.clone())
        if D == 3:
            samples = (*samples, self.w0.clone())

        _base = adv.solve  # bind current solve (possibly already compiled)

        # make_graphed_callables needs a module or plain function with only
        # Tensor positional args.  Wrap with a class to avoid closure issues.
        class _Wrapper(torch.nn.Module):
            def forward(self_, *v):  # noqa: N805
                return _base(*v)

        wrapper = _Wrapper().to(self.device)
        try:
            graphed = torch.cuda.make_graphed_callables(
                wrapper, samples, num_warmup_iters=3,
            )
        except Exception as e:
            print(f"[cuda_graph] capture failed ({e}); falling back to eager")
            return

        adv.solve = lambda *v, nu_t=None, iteration=0: graphed(*v)
        print(f"[cuda_graph] adv-diff graph captured ({D}D, scheme={adv._scheme_name})")

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

    def get_loads(self):
        """Assemble the net hydrodynamic ``(force, torque)`` per body from
        the viscous + pressure components computed by the most recent
        :meth:`advance_and_compute_loads` (i.e. the ``forces_*`` call).

        Returns
        -------
        (force, torque) or None
            2-D: ``force`` is ``(B, 2)``, ``torque`` is ``(B,)`` about the
            out-of-plane axis.  3-D: ``force`` is ``(B, 3)``, ``torque`` is
            ``(B, 3)``.  Returns ``None`` when no loads are available
            (``compute_forces`` disabled).

        The fluid solver only *reports* loads here; **applying** them to a
        body is the consumer's job — the standalone rigid-body integrator,
        FARMS/MuJoCo (``BDIMhandler``), or the implicit coupling driver.
        """
        if not getattr(self, "compute_forces", False):
            return None
        if not hasattr(self, "friction_force_lin_x"):
            return None

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

        return force, torque

    def _apply_force_feedback(self, iteration, t):
        """Hand the freshly computed loads to a *standalone* body's own
        integrator via its ``apply_force_feedback`` callback.

        This is the **explicit-coupling feedback** used by the standalone
        orchestrators (:meth:`run_sim` / :meth:`run_from_initial`); it is
        deliberately **not** called from :meth:`step_` / the fluid core, so
        the fluid step itself never applies loads.  It is a no-op for
        bodies without the callback (FARMS path, prescribed-motion
        validation bodies), where load application is handled elsewhere.
        """
        callback = getattr(self.composite_body, "apply_force_feedback", None)
        if callback is None:
            return
        loads = self.get_loads()
        if loads is None:
            return
        force, torque = loads
        callback(
            force=force,
            torque=torque,
            iteration=iteration,
            time=t,
            dt=self.dt,
            solver=self,
        )

    def advance_and_compute_loads(self, u, v, p, iteration, t, w_vel=None):
        """Repeatable per-step fluid core: body update → gravity → fluid_step
        → (optional zero-pressure) → load computation.

        Sets ``self.u0/v0/(w0)/p0`` and the ``friction_force_*`` /
        ``pressure_force_*`` attributes, but does **not** apply the loads,
        advance the free surface, release BDIM fields, or plot — those are
        the once-per-step tail in :meth:`finalize_step`.

        It is therefore safe to call this repeatedly inside a single time
        step (implicit / strongly-coupled FSI) *provided the fluid fields
        are restored to the start-of-step state between calls*, and the
        loads can be consumed by any structure integrator without the
        solver committing them.

        Returns ``(u, v, p, w_vel)`` (``w_vel`` is ``None`` in 2-D).
        """
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

        # NOTE (well-balanced gravity): the Poisson now solves only the DYNAMIC
        # pressure ``p_d`` (the analytic hydrostatic ``p_h`` is pre-balanced out
        # of the predictor).  The body-force readout deliberately uses ``p_d``
        # alone -- exactly as the no-gravity reference does -- so it carries the
        # dynamic load (thrust / form drag / added mass) WITHOUT the hydrostatic
        # baseline.  Adding ``p_h`` back here would feed the large hydrostatic
        # head through the discretely non-gauge-invariant band quadrature
        # ``Σ -p n δ_ε`` and leak a spurious horizontal force (verified: it
        # reverses the single-phase swim).  Buoyancy is the rigid-body
        # integrator's concern (external Archimedes / MuJoCo), unchanged from
        # the no-gravity case.  Physical pressure, if ever needed for plotting,
        # is ``p_d + self.p_h``.
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

        return u, v, p, w_vel

    def finalize_step(self, u, v, p, iteration, w_vel=None):
        """Once-per-step tail: stability check, free-surface advection,
        BDIM-field release, optional allocator flush, and plotting/saving.

        For implicit coupling, call this exactly once *after* the coupling
        iteration has converged, on the converged fluid state.  Returns the
        ``terminate`` flag from :meth:`plotting_and_saving`.
        """
        if iteration % self.check_explosion_every == 0:
            self.check_explosion(iteration)

        # ---- flow diagnostics on the post-projection field (every N steps) ----
        # Runs before the BDIM-field release so it sees the converged u,v,[w],p
        # and warns on energy blow-up / CFL>0.5 before they cascade to NaN.
        if self.diagnostics is not None:
            cb = self.composite_body
            self.diagnostics.update(
                iteration, u, v, p, self.dt, self.nu,
                self.divergence, self.vorticity, w=w_vel,
                sdf_cc=getattr(cb, "sdf_val", None),
                mu_fn=getattr(cb, "mu_funcs", None),
            )

        # ---- free BDIM fields to reclaim GPU memory between steps ----
        self._release_bdim_fields()

        # ---- flush CUDA allocator cache to reduce nvidia-smi usage ----
        if self.device.type == "cuda" and iteration % self.empty_cache_every == 0:
            torch.cuda.empty_cache()

        # ---- plotting / saving  (works for 2D and 3D) ----
        return self.plotting_and_saving(u, v, p, iteration, w_vel=w_vel)

    def step_(self, u, v, p, iteration, t, w_vel=None):
        # One fluid time step: advance the fluid + compute the loads, then
        # run the once-per-step tail.  The fluid step does **not** apply the
        # loads to any body — that is the orchestrator's job (run_sim /
        # BDIMhandler / the implicit coupler), which reads them via
        # get_loads().  The three pieces are separate methods so the
        # implicit / strongly-coupled driver can repeat the core
        # (advance_and_compute_loads) under a checkpoint/restore loop and
        # run the tail (finalize_step) only once, on convergence.
        u, v, p, w_vel = self.advance_and_compute_loads(
            u, v, p, iteration, t, w_vel=w_vel
        )
        terminate = self.finalize_step(u, v, p, iteration, w_vel=w_vel)

        if self.ndim == 2:
            return (u, v, p, terminate)
        else:
            return (u, v, p, w_vel, terminate)

    def _save_diagnostics_h5(self):
        """Persist the FlowDiagnostics time-series to ``diagnostics.h5`` in the
        run's save folder, if diagnostics are enabled and a save path exists."""
        if self.diagnostics is None:
            return
        save_dir = getattr(self, "save_path", None)
        if save_dir:
            self.diagnostics.save_h5(save_dir, self._hdf5_lock)

    def run_from_initial(self, u0, v0, w0=None):
        """
        Run the standalone (explicit) simulation loop starting from the
        given initial velocity fields.

        As an explicit-coupling orchestrator this drives the fluid core,
        feeds the computed loads back to a standalone body, and runs the
        once-per-step tail — in that order (load feedback before the tail,
        matching the legacy ``step_`` sequencing).
        """
        u = u0
        v = v0
        p = torch.zeros_like(u)
        if self.ndim == 2:
            for iteration in tqdm(range(self.nt)):
                t = iteration*self.dt
                u, v, p, _ = self.advance_and_compute_loads(u, v, p, iteration, t)
                self._apply_force_feedback(iteration, t)
                self.finalize_step(u, v, p, iteration)
        else:
            w = w0 if w0 is not None else torch.zeros_like(u)
            for iteration in tqdm(range(self.nt)):
                t = iteration*self.dt
                u, v, p, w = self.advance_and_compute_loads(u, v, p, iteration, t, w_vel=w)
                self._apply_force_feedback(iteration, t)
                self.finalize_step(u, v, p, iteration, w_vel=w)

        self._save_diagnostics_h5()

    def run_sim(self):
        """
        Standalone (explicit-coupling) simulation loop.

        Same orchestration as :meth:`run_from_initial`: fluid core → load
        feedback to the standalone body → once-per-step tail.
        """
        u = self.u0
        v = self.v0
        p = self.p0
        if self.ndim == 2:
            for iteration in tqdm(range(self.starting_iteration, self.nt)):
                t = iteration*self.dt
                u, v, p, _ = self.advance_and_compute_loads(u, v, p, iteration, t)
                self._apply_force_feedback(iteration, t)
                self.finalize_step(u, v, p, iteration)
        else:
            w = self.w0
            for iteration in tqdm(range(self.starting_iteration, self.nt)):
                t = iteration*self.dt
                u, v, p, w = self.advance_and_compute_loads(u, v, p, iteration, t, w_vel=w)
                self._apply_force_feedback(iteration, t)
                self.finalize_step(u, v, p, iteration, w_vel=w)

        if self.compute_forces and self.save_drags:
            self.save_drags_h5()

        self._save_diagnostics_h5()

        # Block until all background I/O is complete before returning
        self.flush_io()
