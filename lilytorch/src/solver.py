
import datetime
import logging
import os
import threading
import warnings
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from lilytorch.src.native import (
    bdim_apply_3d,
    bdim_apply_2d,
)
from lilytorch.src.advection import AdvDiffSolver
from lilytorch.src.diagnostics import FlowDiagnostics
from lilytorch.src.body import (body_from_yaml,
                                _mu_normals_batched)
from lilytorch.src import operations as ops
from lilytorch.src.plotting import PlottingMixin
from lilytorch.src.poisson_fft import PoissonSolverFFT
from lilytorch.src.poisson_mult import PoissonSolver
from lilytorch.util.yaml_operations import pyobject2yaml
from lilytorch.src import forces, extras
from lilytorch.src.graph_capture import NativeWholeStepGraphRunner

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

        # Debug flag — force the eager (non-graph) path even on CUDA.
        self._graph_capture_debug = bool(solver.get("graph_capture_debug", False))
        if self._graph_capture_debug:
            print("Whole-step graph capture DISABLED (graph_capture_debug=True).")

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

        # NOTE: every numerical kernel (advection, diffusion, Poisson,
        # body-update) is an always-on native CUDA / C++ op with a CPU twin —
        # there is no torch.compile / adv-diff CUDA-graph toggle any more.  The
        # native ops launch on torch's current stream, so the whole pre-Poisson
        # region is captured as one CUDA graph instead (see
        # ``NativeWholeStepGraphRunner``).

        # Dynamic BDIM META compilation for the union-AABB crop path
        # (sub-block shape varies with body kinematics).
        self._bdim_meta_dyn_compiled = torch.compile(
            FluidSolver._bdim_meta, dynamic=True,
        )
        self._bdim_union_aabb = None

        # ---- optional Towers (2008) 2nd-order delta correction -----------
        # When force_delta_order=2, the smoothed delta is MULTIPLIED by |∇SDF|
        # so that the volume integral gives the correct surface measure even
        # when the numerical SDF deviates from unit gradient.  This is the
        # coarea identity ∮_{φ=0} g dS = ∫ g δ(φ)|∇φ| dV.
        # For analytical bodies |∇SDF|=1 exactly, so order 2 is a no-op;
        # it matters for mesh bodies or near geometric corners.
        # (Until 2026-07-16 this DIVIDED by |∇SDF|, doubling the error it was
        # meant to remove — invisible to every analytical-body test.  See
        # milestones/force_readout_agreement_handoff.md §8 / §10.3c.)
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
        # Eulerian force readout: which normal the per-link split rides on (only
        # meaningful when ``force_method == "eulerian"``).  BOTH channels are
        # always read on the UNION smoothed-Heaviside band measure ∂_iH_ε(φ_union)
        # — one closed surface with no internal inter-link seams, so the net obeys
        # summation-by-parts and never leaks the hydrostatic baseline — and the
        # union force is split back to the individual links by a softmin partition
        # of unity keyed on the body-velocity blend width.
        #
        #   "union" — F_i = union band measure × the union ∂_iH direction.  The
        #             net is exact by summation-by-parts and the readout is
        #             gauge-safe (constant p → 0), so this is the default and the
        #             only readout allowed with the two-phase solver.
        #   "body"  — F_i = union coarea magnitude × each link's own ANALYTIC
        #             surface normal.  More accurate per-link forces on thin /
        #             multi-link bodies, but gauge-UNSAFE (constant p leaks a
        #             large spurious force), so it is analysis-only and forbidden
        #             with the two-phase solver.
        #
        # Maps to the native op's ``force_submethod`` int (0 = union, 2 = body;
        # value 1 was the retired deltaH readout — hence the numbering gap).
        _fln_raw = solver.get("force_link_normal", "union")
        if _fln_raw not in ("union", "body"):
            raise ValueError(
                f"solver.force_link_normal must be one of "
                f"('union', 'body'), got {_fln_raw!r}."
            )
        self.force_link_normal = _fln_raw
        self.force_submethod = 0 if _fln_raw == "union" else 2
        if self.force_submethod == 2 and solver.get("two_phase") is not None:
            raise ValueError(
                "force_link_normal='body' (per-body analytic normal, sm2) is "
                "gauge-unsafe (constant p leaks a spurious force) and must not "
                "be used with the two-phase solver; use 'union'."
            )
        # Distance to offset the Lagrangian-force sample point along the
        # outward surface normal (see lagrangian_forces.cu).  0 (default)
        # = sample exactly at the centroid/contour marker (legacy
        # behaviour, biased by BDIM band contamination).  Set to ~eps
        # via config to escape the band.
        #
        # DEPRECATED in favour of the per-channel knobs below, which apply to
        # BOTH readouts.  Kept because it is the single knob every existing
        # lagrangian config sets, and it moves p and sigma together.
        self.lagrangian_sample_offset = float(
            solver.get("lagrangian_sample_offset", 0.0))

        # ---- per-channel sampling locations, shared by BOTH readouts -----
        # phi = offset is the iso-surface on which each channel is read.  The
        # two channels are contaminated differently inside the BDIM band, and
        # the two readouts have historically disagreed about where to sample:
        #
        #   eulerian:   p at phi=0,   sigma at phi=eps   (Maertens & Weymouth)
        #   lagrangian: both at phi=lagrangian_sample_offset
        #
        # so any cross-method comparison confounded "which readout" with
        # "sampled where".  These knobs exist to eliminate that: setting them
        # pins both readouts to the same locations.  None → each readout keeps
        # its own legacy convention, so defaults are bit-for-bit unchanged.
        #
        # Given in CELLS, not metres: the right value scales with h, so a
        # value fixed in metres means a different thing on every grid and a
        # grid-convergence study would silently change the physics.
        _h_f = float(self.h)
        _off_p_cells = solver.get("sample_offset_pressure_cells", None)
        _off_f_cells = solver.get("sample_offset_friction_cells", None)
        _lso = self.lagrangian_sample_offset

        if _off_p_cells is None:
            self.eul_sample_offset_pressure  = 0.0
            self.lagr_sample_offset_pressure = _lso
        else:
            _p = float(_off_p_cells) * _h_f
            self.eul_sample_offset_pressure  = _p
            self.lagr_sample_offset_pressure = _p

        if _off_f_cells is None:
            self.eul_sample_offset_friction  = float(self.eps)
            self.lagr_sample_offset_friction = _lso
        else:
            _f = float(_off_f_cells) * _h_f
            self.eul_sample_offset_friction  = _f
            self.lagr_sample_offset_friction = _f

        self.zero_pressure_inside = solver.get("zero_pressure_inside", False)
        # Smooth body-velocity blend in the overlap band (kernel path).
        # Width given in grid cells; <=0 / None → legacy hard running-min
        # winner-take-all.  See BDIMhandler / streaming_sdf.cu.
        _bvb = solver.get("body_velocity_blend_eps_cells", None)
        self._body_vel_blend_cells = float(_bvb) if _bvb else 0.0

        # ``solver_method`` is DEPRECATED and IGNORED: the former "python" /
        # "kernel" modes were collapsed into the single fused (body-agnostic)
        # path.  The key is still parsed so the many existing configs that set
        # it keep loading; it only triggers a warning.
        if solver.get("solver_method") is not None:
            warnings.warn(
                "solver.solver_method is deprecated and ignored: the python "
                "and kernel modes were collapsed into the single fused "
                "(body-agnostic) fluid path.",
                DeprecationWarning, stacklevel=2,
            )
        self._mu_union_ready = False
        # Body-SDF sampling method used inside the streaming C++/CUDA
        # kernels (``body_update_{2d,3d}``):
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
            recycle_k       = solver.get("poisson_recycle_k", 0),
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
        )
        # Belt-and-suspenders: ensure the body's eps exactly matches the
        # solver's Maertens-Weymouth eps_multiplier * h, even if a subclass
        # or downstream handler replaces the body after construction.
        self.composite_body.eps = float(self.eps)

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
        # Persistent device buffer for the per-step dirty rect (int32[4/6]).
        # Allocated once; ``stage()`` copies fresh values in before each step.
        self._bdim_rect_dev = torch.empty(2 * self.ndim, dtype=torch.int32,
                                          device=self.device)
        # CPU-side staging buffer — avoids a per-step GPU tensor allocation
        # inside stage() (the old path did ``torch.tensor([...], device=...)``
        # for every step, which is ~80k cudaMalloc/cudaFree calls over a run).
        self._stage_rect_cpu = torch.empty(2 * self.ndim, dtype=torch.int32,
                                           device='cpu', pin_memory=True)

        # =====================================================================
        # Gravity body force (opt-in via the ``solver.gravity`` block).
        #   solver.gravity: [gx, gy]      # 2-D
        #   solver.gravity: [gx, gy, gz]  # 3-D
        # Used by the two-phase solver (water+air) and any gravity-driven
        # run; ``None`` disables it.
        # =====================================================================
        self._init_gravity(solver.get("gravity", None))

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

        Used by the two-phase solver.  BOTH fused single-phase steps compute
        this term INSIDE the ``bdim_apply_{2,3}d`` kernel (``mw_on``
        inputs, written to ``_mw_div_corr_persist``) — this method is their
        torch oracle (parity-tested in
        ``tests/test_bdim.py::test_mw_div_corr_fold_{2d,3d}``).
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
            # In-place corrections keep u/v/w_vel (== self.u0/v0/w0)
            # pointer-stable across steps -> the data_ptr-keyed CUDA-graph
            # caches replay instead of leaking a fresh graph per allocator
            # reshuffle.  See the multigrid branch note below.
            if self.ndim == 2:
                (p_x, p_y) = self.gradient(p)
                u.add_(p_x, alpha=-float(c_scalar))
                v.add_(p_y, alpha=-float(c_scalar))
            else:
                (p_x, p_y, p_z) = self.gradient(p)
                u.add_(p_x, alpha=-float(c_scalar))
                v.add_(p_y, alpha=-float(c_scalar))
                w_vel.add_(p_z, alpha=-float(c_scalar))
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

            # The multigrid driver applies h² internally (f_scaled = h²·f
            # inside solve_*), so pass the raw divergence RHS (pre_scaled=False)
            # to avoid double-scaling.
            _pre_scaled = False
            if _pre_scaled:
                div.mul_(self.poisson_solver.h2)

            if ch is None:
                ch = coeff * self.mu0_all_u
            if cv is None:
                cv = coeff * self.mu0_all_v

            # Select solve method: recycled MGCG, MGCG, or standalone multigrid
            _poisson_solve = {
                "rmgcg": self.poisson_solver.solve_rmgcg,
                "mgcg":  self.poisson_solver.solve_mgcg,
            }.get(self.poisson_method, self.poisson_solver.solve_multigrid)

            # Warm start: reuse the previous pressure as the initial guess.
            #
            # This used to be gated on ``not has_custom_coeffs``, from a time
            # when custom ch/cv/cw meant "two-phase".  The streaming BDIM path
            # now ALWAYS passes persistent coefficient buffers, so that guard
            # silently disabled the flag for every config on that path,
            # single-phase included.
            #``
            # The guess is not a mere speed knob here: ``poisson_tol`` is an
            # ABSOLUTE residual, and a solve that runs out its cycle cap stops
            # wherever it happens to be.  A two-phase pressure carries
            # hydrostatics (O(1e3) Pa), so cold-starting from zero rebuilds
            # that whole field every step and never converges.  Measured,
            # surface-pool 900x300x52 over 60 steps: cold, tol 1e-8 leaves
            # max|div u| = 6.17 at 182 ms/step; warm leaves 0.24 at the same
            # 182 ms (tol 1e-6), or 0.93 at 25 ms (tol 1e-4).
            #
            # ``zero_pressure_inside`` is the one combination to watch: it
            # zeroes the body interior of the stored p, so the warm guess
            # carries a sharp mismatch at the body boundary that the smoother
            # must undo -- there, cold-starting can genuinely be faster.
            if self.poisson_warm_start:
                p0 = p
            elif self.poisson_method == "multigrid":
                # The native multigrid driver serves p0=None from a persistent
                # zeroed buffer -- skip the per-step ``zeros_like`` alloc +
                # copy-into-graph (host-wrapper trim; the MGCG/RMGCG cores need
                # a real tensor for ``x = p0.clone()``, so only multigrid opts
                # out here).
                p0 = None
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
                # In-place correction (u/v are self.u0/self.v0): keeps the
                # velocity buffers pointer-stable across steps so the
                # data_ptr-keyed CUDA-graph caches (fused SL advect, apply_bcs)
                # replay one graph instead of capturing+pinning a fresh graph
                # every allocator reshuffle (empty_cache_every) -> steady GPU
                # growth -> wp_cuda_graph_create_exec OOM.  Mirrors the 3-D
                # face-grid path's addcmul_ correction below.
                (p_x, p_y) = self.gradient(p)
                u.addcmul_(ch, p_x, value=-1.0)
                v.addcmul_(cv, p_y, value=-1.0)
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
                    u.addcmul_(ch, p_x, value=-1.0)
                    v.addcmul_(cv, p_y, value=-1.0)
                    w_vel.addcmul_(cw, p_z, value=-1.0)
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

    def _preproj_graph_safe(self):
        """May the pre-projection region be recorded into a CUDA graph?

        Only if every kernel it launches goes to torch's CURRENT CUDA stream.
        A kernel launched on any OTHER stream is never recorded by torch stream
        capture: it executes during the capture pass and is then silently
        DROPPED from every replay — the physics just vanishes, with no error and
        no NaN (measured once at ~47% velocity error).  Both advection paths
        (flux and semi-Lagrangian) are native end-to-end and capture correctly;
        the guard stays as the gate for any future non-capturable solve.
        """
        safe = getattr(self.adv_diff_solver, 'graph_capturable', False)
        if not safe and not getattr(self, '_preproj_graph_warned', False):
            self._preproj_graph_warned = True
            warnings.warn(
                "Pre-projection CUDA graph DISABLED: convection_method "
                f"'{getattr(self.adv_diff_solver, '_scheme_name', '?')}' "
                "launches kernels that torch's stream capture cannot record "
                "(they would be dropped from every graph replay). Running "
                "the region eagerly.",
                RuntimeWarning, stacklevel=2,
            )
        return safe

    def _release_bdim_fields(self):
        """Set BDIM intermediate fields to *None* so their GPU memory
        can be reclaimed between time-steps (they are recomputed at the
        beginning of every step anyway).

        Phase I: the kernel-mode paths (2-D and 3-D) no longer
        materialise any of the staggered or CC mu / normal tensors as
        full-grid buffers — they live only in CUDA thread registers
        inside the fused ``bdim_apply`` kernel — so the keep-set is
        empty in both modes.
        """
        for attr in self._BDIM_FIELD_NAMES:
            if hasattr(self, attr):
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    #   mu / normal recomputation  (python-style force readout only)
    # ------------------------------------------------------------------
    def _needs_python_mu_normals(self):
        """True when the per-step ``_recompute_mu_normals`` pack must be
        built: standalone (non-streaming) bodies whose forces are read by
        the python force integration, which consumes ``mu0_all``/normals.
        Streaming bodies (``_kernel_static_{2,3}d`` present) use the
        post-step force kernels and never need the full-grid pack.
        ``TwoPhaseSolver`` extends this with the consistent-momentum path
        (which needs the staggered mu/normals for the velocity BDIM)."""
        comp = self.composite_body
        streaming = (getattr(comp, '_kernel_static_3d', None) is not None
                     or getattr(comp, '_kernel_static_2d', None) is not None)
        return (not streaming) and bool(self.compute_forces)

    def _recompute_mu_normals(self):
        """Recompute mu0/mu1 and CC/staggered unit normals — dim-agnostic.

        Processes all ``ndim + 1`` SDF grids ([u, v, [w,] cc]) in a single
        batched pass via the dim-agnostic ``_mu_normals_batched`` helper.

        When the union-AABB sub-block covers less than 50 % of the grid
        (see :meth:`_compute_union_aabb`),
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

        When no sparse AABB is available or the union covers >50 % of the
        grid, falls back to the full-grid compiled kernel (no pack buffer).
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
        # allocation is O(sub-block) rather than O(full grid).  Activates
        # whenever a sparse AABB is available (``_sdf_sparse`` or
        # ``_combined_union_aabb``); bodies without one fall through to the
        # full-grid path below.
        # ------------------------------------------------------------------
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

        # ---- narrow-band fast path -------------------------------------
        if all(m is not None for m in mu_grids):
            # Reuse the AABB cached earlier in the step when available
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
    #   Fused fluid step — native bdim_apply
    # ------------------------------------------------------------------
    def _init_bdim_coeff_persist_3d(self, timestep):
        """Lazy-allocate the persistent ``ch/cv/cw`` Poisson-coefficient
        buffers for the kernel-mode 3-D path.

        Outside any immersed body ``mu0 = 1`` everywhere, so the
        coefficients reduce to the constant ``dt / rho_fluid``.  The
        buffers are pre-filled once with that default; the fused
        ``bdim_apply`` kernel overwrites only the dirty AABB sub-block
        each step.
        """
        Ngx, Ngy, Ngz = self.grid_shape
        device = self.device
        dtype  = self.dtype
        _dt_over_rhofluid = (self._cached_float('dt', timestep)
                             / self._cached_float('rho', self.rho))
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
            or self._ch_persist.device.type != device.type
            or getattr(self, '_ch_outside_val', None) != _dt_over_rhofluid
        )
        if needs_realloc:
            self._ch_persist = torch.full(ch_gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._cv_persist = torch.full(cv_gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._cw_persist = torch.full(cw_gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._ch_outside_val = _dt_over_rhofluid
        gs = self.grid_shape
        if self._bdim_body_div_correction and (
                getattr(self, '_mw_div_corr_persist', None) is None
                or self._mw_div_corr_persist.shape != gs
                or self._mw_div_corr_persist.dtype != dtype
                or self._mw_div_corr_persist.device.type != device.type):
            # Full-grid MW body-divergence output, written by the fused
            # bdim_apply kernel every step (so no stale-cell hazard).
            self._mw_div_corr_persist = torch.zeros(gs, device=device, dtype=dtype)

    def _cached_float(self, key, value):
        """Python-float mirror of a per-run scalar that may live as a 0-d GPU
        tensor (``self.rho``/``self.nu``/``self.eps``/``comp.h``/``self.dt``…).

        ``float(gpu_tensor)`` is a host↔device sync; doing it every step at
        kernel-marshalling sites (bdim_apply, the fused force readout) stalls
        the CUDA pipeline mid-step — profiling the salamander 2-D run showed
        ~16 such hidden syncs/step.  These scalars are constant for the life of
        the run, so convert once and reuse the cached float.

        Always checks the cache first — even for non-tensor scalars — to
        return the SAME float every call (critical for persistent-buffer
        realloc guards and CUDA-graph key stability)."""
        cache = self.__dict__.setdefault('_float_scalar_cache', {})
        v = cache.get(key)
        if v is None:
            v = float(value)
            cache[key] = v
        return v

    def _init_bdim_coeff_persist_2d(self, timestep):
        """2-D analogue of :meth:`_init_bdim_coeff_persist_3d`."""
        gs = self.grid_shape
        device = self.device
        dtype  = self.dtype
        _dt_over_rhofluid = (self._cached_float('dt', timestep)
                             / self._cached_float('rho', self.rho))
        needs_realloc = (
            getattr(self, '_ch_persist', None) is None
            or self._ch_persist.shape  != gs
            or self._ch_persist.dtype  != dtype
            or self._ch_persist.device.type != device.type
            or getattr(self, '_ch_outside_val', None) != _dt_over_rhofluid
        )
        if needs_realloc:
            self._ch_persist = torch.full(gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._cv_persist = torch.full(gs, _dt_over_rhofluid, device=device, dtype=dtype)
            self._ch_outside_val = _dt_over_rhofluid
        if self._bdim_body_div_correction and (
                getattr(self, '_mw_div_corr_persist', None) is None
                or self._mw_div_corr_persist.shape != gs
                or self._mw_div_corr_persist.dtype != dtype
                or self._mw_div_corr_persist.device.type != device.type):
            # Full-grid MW body-divergence output, written by the fused
            # bdim_apply kernel every step (so no stale-cell hazard).
            self._mw_div_corr_persist = torch.zeros(gs, device=device, dtype=dtype)

    # ── Fused BDIM step (2-D): native bdim_apply ──────────────────────────
    def _fluid_step_fused_2d(self, u, v, p, timestep):
        """2-D fused fluid step: a single native kernel (``bdim_apply_2d``)
        applies the BDIM2 meta-equation and assembles the variable-density
        Poisson coefficients, followed by the pressure projection.

        Body-agnostic: the step computes no body geometry itself.  It consumes
        the contract fields the preceding ``composite_body.update()`` published
        on the composite body — whether that update is the body's own (python
        field evaluation, e.g. deforming bodies) or the rigid streaming
        ``body_update`` launched by ``BDIMhandler``:

          * ``sdf_val_u`` / ``sdf_val_v`` — staggered union SDFs;
          * ``body_u`` / ``body_v``       — staggered body velocities;
          * ``bdim_dirty``  (optional)    — narrow-band dirty rect
            ``{'i0','j0','Ai','Aj'}``; ``None``/absent → full grid;
          * ``bdim_fields_scratch`` (optional) — ``True`` when the fields
            are per-step scratch buffers this step must release before the
            pressure projection (streaming eager path).
        """
        comp  = self.composite_body
        sdf_u = getattr(comp, 'sdf_val_u', None)
        sdf_v = getattr(comp, 'sdf_val_v', None)
        bU    = getattr(comp, 'body_u', None)
        bV    = getattr(comp, 'body_v', None)
        if sdf_u is None or sdf_v is None or bU is None or bV is None:
            raise RuntimeError(
                "fluid_step: composite_body.update() must publish the "
                "staggered BDIM fields (sdf_val_u/v, body_u/v) before the "
                "fluid step."
            )

        # Dirty rect: provider-supplied narrow band, else the full grid.
        gs = self.grid_shape
        d  = getattr(comp, 'bdim_dirty', None)
        if d is None:
            d = {'i0': 0, 'j0': 0, 'Ai': int(gs[0]), 'Aj': int(gs[1])}

        mw_on = self._bdim_body_div_correction

        use_graph = (u.is_cuda and not self._graph_capture_debug
                     and self._preproj_graph_safe())

        # ── Lazy-init whole-step runner with correct mode ──────────
        # The runner itself decides whether to capture a CUDA graph or
        # run eagerly — solver.py has no branching on use_graph.
        if getattr(self, '_preproj_graph_2d', None) is None:
            self._preproj_graph_2d = NativeWholeStepGraphRunner(
                use_cuda_graph=use_graph)
        else:
            self._preproj_graph_2d._use_cuda_graph = use_graph
        runner = self._preproj_graph_2d

        # ── Shared pre-computation (CPU + CUDA) ─────────────────────
        # Compute nu_t (Smagorinsky / Carreau) natively, then form the
        # total effective viscosity nu_eff = nu + nu_t in a persistent
        # buffer.  The torch.add is cheap on CPU; on CUDA it runs before
        # graph capture and synchronises implicitly.
        _nu_t_field = self._compute_nu_t(u, v)
        if _nu_t_field is not None:
            if (getattr(self, '_nu_eff_graph', None) is None
                    or self._nu_eff_graph.shape != u.shape
                    or self._nu_eff_graph.dtype != u.dtype
                    or self._nu_eff_graph.device != u.device):
                self._nu_eff_graph = torch.empty_like(u)
            torch.add(float(self.nu), _nu_t_field, out=self._nu_eff_graph)
            _nu_eff = self._nu_eff_graph
        else:
            _nu_eff = None

        # Init persistent BDIM coefficient buffers (fallback for standalone
        # runs; BDIMhandler normally allocates these before the fluid step).
        if getattr(self, '_ch_persist', None) is None:
            self._init_bdim_coeff_persist_2d(timestep)

        # ── Core pre-projection operations (always the same) ─────
        # advection + diffusion + BDIM forcing + set_BCs — all native.
        # The runner dispatches to either graph replay or eager launch.
        def _run_preproj():
            primes_loc = self.adv_diff_solver.solve(u, v, nu_eff=_nu_eff)
            bdim_apply_2d(
                primes_loc[0], primes_loc[1],
                sdf_u, sdf_v,
                bU, bV,
                self.u0, self.v0,
                self._ch_persist, self._cv_persist,
                self._cached_float('comp_eps', comp.eps),
                self._cached_float('rho', self.rho),
                self._cached_float('dt', timestep),
                self._cached_float('comp_h', comp.h),
                int(d['i0']), int(d['j0']),
                int(d['Ai']), int(d['Aj']),
                int(self.bdim_mu0_projection),
                sdf_cc=(comp.sdf_val if mw_on else None),
                div_corr=(self._mw_div_corr_persist if mw_on else None),
                eps_mw=(self._cached_float('eps', self.eps) if mw_on else 1.0),
                inv_dx=1.0 / self._cached_float('dx', self.dx),
                inv_dy=1.0 / self._cached_float('dy', self.dy),
                rect_dev=self._bdim_rect_dev,
            )
            self.adv_diff_solver.set_BCs(self.u0, self.v0)

        # ── Staging: copy the per-step dirty rect into the persistent
        # device buffer OUTSIDE the graph capture (stable pointer, no
        # torch ops on the default stream → no CUDA error 900).
        # Uses a pre-allocated CPU-pinned buffer to avoid a per-step
        # GPU tensor allocation (cudaMalloc + cudaFree per step).
        def stage():
            cpu = self._stage_rect_cpu
            cpu[0] = int(d['i0'])
            cpu[1] = int(d['j0'])
            cpu[2] = int(d['Ai'])
            cpu[3] = int(d['Aj'])
            self._bdim_rect_dev.copy_(cpu)

        key = (
            u.data_ptr(), v.data_ptr(),
            self.u0.data_ptr(), self.v0.data_ptr(),
            sdf_u.data_ptr(), sdf_v.data_ptr(),
            bU.data_ptr(), bV.data_ptr(),
            self._ch_persist.data_ptr(),
            self._cv_persist.data_ptr(),
            self._bdim_rect_dev.data_ptr(),
            (self._mw_div_corr_persist.data_ptr() if mw_on else 0),
            (comp.sdf_val.data_ptr()
             if mw_on and comp.sdf_val is not None else 0),
            self._cached_float('dt', timestep), str(u.dtype),
        )
        device_str = f"cuda:{u.device.index}" if u.is_cuda else "cpu"
        runner.run(key, device_str, _run_preproj, stage)

        # 5b. MW body-divergence RHS correction — now written in-kernel.
        _body_div_corr = self._mw_div_corr_persist if mw_on else None

        # 6. Release per-step scratch fields before the pressure projection
        #    (streaming eager path — mirrors the former temporary free).
        del sdf_u, sdf_v, bU, bV
        if getattr(comp, 'bdim_fields_scratch', False):
            comp.sdf_val_u = comp.sdf_val_v = None
            comp.body_u = comp.body_v = None

        # 8. Pressure projection.
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
            body_div_corr=_body_div_corr,
        )
        vels_out = out[:-1]
        p_out    = out[-1]

        # 9. Optional sponge / yield damping + final BC pass.
        if self.use_sponge:
            vels_out = self.apply_sponge_damping(*vels_out)
        if self.use_yield_damping:
            vels_out = self.apply_yield_damping(*vels_out)
        self.adv_diff_solver.set_BCs(*vels_out)

        return (*vels_out, p_out)

    # ── Fused BDIM step (3-D): native bdim_apply ──────────────────────────
    def _fluid_step_fused_3d(self, u, v, w_vel, p, timestep):
        """3-D analogue of :meth:`_fluid_step_fused_2d` (adds the w axis and
        the ``k0``/``Ak`` dirty extents).  Same body-agnostic contract: the
        step consumes the staggered fields published by the preceding
        ``composite_body.update()`` and never computes body geometry itself.

        Whole-step pre-Poisson graph (CUDA) captures SL/convective advection
        + diffusion + bdim_apply + set_BCs as ONE CUDA-graph replay,
        collapsing ~5 host-side captures into a single replay (~3 µs)."""
        comp  = self.composite_body
        sdf_u = getattr(comp, 'sdf_val_u', None)
        sdf_v = getattr(comp, 'sdf_val_v', None)
        sdf_w = getattr(comp, 'sdf_val_w', None)
        bU    = getattr(comp, 'body_u', None)
        bV    = getattr(comp, 'body_v', None)
        bW    = getattr(comp, 'body_w', None)
        if (sdf_u is None or sdf_v is None or sdf_w is None
                or bU is None or bV is None or bW is None):
            raise RuntimeError(
                "fluid_step: composite_body.update() must publish the "
                "staggered BDIM fields (sdf_val_u/v/w, body_u/v/w) before "
                "the fluid step."
            )

        # Dirty rect: provider-supplied narrow band, else the full grid.
        gs = self.grid_shape
        d  = getattr(comp, 'bdim_dirty', None)
        if d is None:
            d = {'i0': 0, 'j0': 0, 'k0': 0,
                 'Ai': int(gs[0]), 'Aj': int(gs[1]), 'Ak': int(gs[2])}

        mw_on = self._bdim_body_div_correction

        use_graph = (u.is_cuda and not self._graph_capture_debug
                     and self._preproj_graph_safe())

        # ── Lazy-init whole-step runner with correct mode ──────────
        # The runner itself decides whether to capture a CUDA graph or
        # run eagerly — solver.py has no branching on use_graph.
        if getattr(self, '_preproj_graph_3d', None) is None:
            self._preproj_graph_3d = NativeWholeStepGraphRunner(
                use_cuda_graph=use_graph)
        else:
            self._preproj_graph_3d._use_cuda_graph = use_graph
        runner = self._preproj_graph_3d

        # ── Shared pre-computation (CPU + CUDA) ─────────────────────
        # Compute nu_t (Smagorinsky / Carreau) natively, then form the
        # total effective viscosity nu_eff = nu + nu_t in a persistent
        # buffer.  The torch.add is cheap on CPU; on CUDA it runs before
        # graph capture and synchronises implicitly.
        _nu_t_field = self._compute_nu_t(u, v, w_vel)
        if _nu_t_field is not None:
            if (getattr(self, '_nu_eff_graph', None) is None
                    or self._nu_eff_graph.shape != u.shape
                    or self._nu_eff_graph.dtype != u.dtype
                    or self._nu_eff_graph.device != u.device):
                self._nu_eff_graph = torch.empty_like(u)
            torch.add(float(self.nu), _nu_t_field, out=self._nu_eff_graph)
            _nu_eff = self._nu_eff_graph
        else:
            _nu_eff = None

        # Init persistent BDIM coefficient buffers (fallback for standalone
        # runs; BDIMhandler normally allocates these before the fluid step).
        if getattr(self, '_ch_persist', None) is None:
            self._init_bdim_coeff_persist_3d(timestep)

        # ── Core pre-projection operations (always the same) ─────
        # advection + diffusion + BDIM forcing + set_BCs — all native.
        # The runner dispatches to either graph replay or eager launch.
        def _run_preproj():
            primes_loc = self.adv_diff_solver.solve(
                u, v, w_vel, nu_eff=_nu_eff)
            bdim_apply_3d(
                primes_loc[0], primes_loc[1], primes_loc[2],
                sdf_u, sdf_v, sdf_w,
                bU, bV, bW,
                self.u0, self.v0, self.w0,
                self._ch_persist, self._cv_persist, self._cw_persist,
                self._cached_float('comp_eps', comp.eps),
                self._cached_float('rho', self.rho),
                self._cached_float('dt', timestep),
                self._cached_float('comp_h', comp.h),
                int(d['i0']), int(d['j0']), int(d['k0']),
                int(d['Ai']), int(d['Aj']), int(d['Ak']),
                int(self.bdim_mu0_projection),
                sdf_cc=(comp.sdf_val if mw_on else None),
                div_corr=(self._mw_div_corr_persist if mw_on else None),
                eps_mw=(self._cached_float('eps', self.eps) if mw_on else 1.0),
                inv_dx=1.0 / self._cached_float('dx', self.dx),
                inv_dy=1.0 / self._cached_float('dy', self.dy),
                inv_dz=1.0 / self._cached_float('dz', self.dz),
                rect_dev=self._bdim_rect_dev,
            )
            self.adv_diff_solver.set_BCs(self.u0, self.v0, self.w0)

        # ── Staging: copy the per-step dirty rect into the persistent
        # device buffer OUTSIDE the graph capture (stable pointer, no
        # torch ops on the default stream → no CUDA error 900).
        # Uses a pre-allocated CPU-pinned buffer to avoid a per-step
        # GPU tensor allocation (cudaMalloc + cudaFree per step).
        def stage():
            cpu = self._stage_rect_cpu
            cpu[0] = int(d['i0'])
            cpu[1] = int(d['j0'])
            cpu[2] = int(d['k0'])
            cpu[3] = int(d['Ai'])
            cpu[4] = int(d['Aj'])
            cpu[5] = int(d['Ak'])
            self._bdim_rect_dev.copy_(cpu)

        key = (
            u.data_ptr(), v.data_ptr(), w_vel.data_ptr(),
            self.u0.data_ptr(), self.v0.data_ptr(), self.w0.data_ptr(),
            sdf_u.data_ptr(), sdf_v.data_ptr(), sdf_w.data_ptr(),
            bU.data_ptr(), bV.data_ptr(), bW.data_ptr(),
            self._ch_persist.data_ptr(),
            self._cv_persist.data_ptr(),
            self._cw_persist.data_ptr(),
            self._bdim_rect_dev.data_ptr(),
            (self._mw_div_corr_persist.data_ptr() if mw_on else 0),
            (comp.sdf_val.data_ptr()
             if mw_on and comp.sdf_val is not None else 0),
            self._cached_float('dt', timestep), str(u.dtype),
        )
        device_str = f"cuda:{u.device.index}" if u.is_cuda else "cpu"
        runner.run(key, device_str, _run_preproj, stage)

        # 5b. MW body-divergence RHS correction — now written in-kernel.
        _body_div_corr = self._mw_div_corr_persist if mw_on else None

        # 6. Release per-step scratch fields before the pressure projection
        #    (streaming eager path — mirrors the former temporary free).
        del sdf_u, sdf_v, sdf_w, bU, bV, bW
        if getattr(comp, 'bdim_fields_scratch', False):
            comp.sdf_val_u = comp.sdf_val_v = comp.sdf_val_w = None
            comp.body_u = comp.body_v = comp.body_w = None

        # 8. Pressure projection (set_BCs already done inside the graph/eager
        #    branch above; no separate step 7 needed).
        out = self.project(
            self.u0, self.v0, p,
            ch=self._ch_persist, cv=self._cv_persist, cw=self._cw_persist,
            ch_cc=getattr(self, '_ch_cc_persist', None),
            w_vel=self.w0,
            body_div_corr=_body_div_corr,
        )
        vels_out = out[:-1]
        p_out    = out[-1]

        # 9. Optional sponge / yield damping + final BC pass.
        if self.use_sponge:
            vels_out = self.apply_sponge_damping(*vels_out)
        if self.use_yield_damping:
            vels_out = self.apply_yield_damping(*vels_out)
        self.adv_diff_solver.set_BCs(*vels_out)

        return (*vels_out, p_out)

    def fluid_step(self, *args):
        """One FSI fluid step (advect-BDIM-project).  Called by BDIMhandler.

        Dim-agnostic.  Signature: ``(u, v, p, timestep)`` in 2-D,
        ``(u, v, w, p, timestep)`` in 3-D.  Dispatches to the single fused
        (body-agnostic) step ``_fluid_step_fused_{2,3}d``, which consumes
        the contract fields published by the preceding
        ``composite_body.update()`` — whether that update is a python-style
        field evaluation (standalone / deforming bodies) or the rigid
        streaming ``body_update`` launched by ``BDIMhandler``.
        """
        if self.ndim == 3:
            return self._fluid_step_fused_3d(*args)
        return self._fluid_step_fused_2d(*args)

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
        # The fused step computes mu0/mu1 and normals in CUDA thread
        # registers inside bdim_apply, so the full-grid mu/normal pack is
        # only materialised when a downstream consumer actually reads it:
        # the python force readout of standalone (non-streaming) bodies.
        # Streaming (FARMS) runs read their forces from the post-step force
        # kernels and must NOT start paying the mu/normals memory peak.
        if self._needs_python_mu_normals():
            self._recompute_mu_normals()

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
