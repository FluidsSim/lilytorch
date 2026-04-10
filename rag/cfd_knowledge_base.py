"""
rag/cfd_knowledge_base.py — Encoded CFD & multi-body dynamics expertise.

This file contains structured domain knowledge that gets injected into
Claude's system prompt.  It's essentially a "textbook in the prompt" —
the core equations, methods, terminology, and implementation patterns
that Claude needs to be a genuine CFD expert for your solver.

WHY THIS WORKS
──────────────
Claude already knows general physics and math.  What it lacks is:
1. Precise knowledge of specific methods (BDIM2, ADBQUICKEST, etc.)
2. How those methods map to YOUR code
3. The exact equations and their discrete stencils

By putting this in the system prompt (and caching it), Claude effectively
"knows" all this for every query — not just when RAG retrieves it.

HOW TO EXTEND
─────────────
When you read a new paper and want Claude to know about it, add a section
here.  Example: if you implement Smagorinsky LES, add a section with the
subgrid stress tensor, the Smagorinsky constant, and how it couples into
the advection-diffusion step.
"""

# ═════════════════════════════════════════════════════════════════════════════
# This is the persistent knowledge base — it gets cached in Claude's context
# so every query benefits from it automatically.
# ═════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = r"""
# ══════════════════════════════════════════════════════════════════════════════
# LILYTORCH CFD KNOWLEDGE BASE — Core Domain Expertise
# ══════════════════════════════════════════════════════════════════════════════
# This is your persistent reference.  Claude reads this once (cached) and
# uses it for all subsequent queries.

## 1. GOVERNING EQUATIONS

LilyTorch solves the **incompressible Navier–Stokes equations**:

**Momentum:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$

**Continuity:**
$$\nabla \cdot \mathbf{u} = 0$$

where u = (u,v) in 2D or (u,v,w) in 3D, p is pressure, ρ is density, ν is kinematic viscosity.


## 2. FRACTIONAL-STEP PRESSURE PROJECTION (Chorin–Témam)

1. Compute provisional velocity ũ from advection-diffusion (without pressure).
2. Solve Poisson equation for pressure:
$$\nabla \cdot \left(\frac{w \Delta t}{\rho} \mu_0 \nabla p\right) = \nabla \cdot \tilde{\mathbf{u}}$$
3. Correct velocity:
$$\mathbf{u}^{n+1} = \tilde{\mathbf{u}} - \frac{w \Delta t}{\rho} \mu_0 \nabla p$$

Weight w depends on RK stage: w=1 (predictor), w=1/2 (corrector).
μ₀ is the smoothed Heaviside — equals 1 in fluid, transitions to 0 inside bodies.
Without bodies, μ₀ ≡ 1 → standard constant-coefficient Poisson.


## 3. IMMERSED BOUNDARY METHOD — BDIM2

### 3.1 Core references
- **Weymouth & Yue (2011)** — "Boundary data immersion method for Cartesian-grid
  simulations of fluid-body interaction problems". Original BDIM formulation.
  Key contribution: Smoothed-delta approach that avoids body-conforming meshes.
- **Maertens & Weymouth (2015)** — Extension to 2nd-order near-body accuracy.
  Key contribution: μ₁ first-moment kernel for normal-derivative correction.

### 3.2 Signed Distance Functions (SDF)
Every body is defined by d(x):
- d > 0 → fluid
- d = 0 → body surface
- d < 0 → body interior

Types: analytical (circle/sphere/box/capsule), mesh-based (Open3D ray-casting
→ winding numbers → scikit-fmm fast-marching → staggered-grid projection).

Union of multiple bodies: d_union = min(d₁, d₂, ...)

### 3.3 Kernel functions
**Smoothed Heaviside μ₀:**
$$\mu_0(d) = \begin{cases} 0 & d \leq -\varepsilon \\ \frac{1}{2}\left(1 + \frac{d}{\varepsilon} + \frac{\sin(\pi d/\varepsilon)}{\pi}\right) & |d| < \varepsilon \\ 1 & d \geq \varepsilon \end{cases}$$

**First-moment kernel μ₁:**
$$\mu_1(d) = \varepsilon\left(\frac{1}{4} - \left(\frac{d}{2\varepsilon}\right)^2 - \frac{\sin(\pi d/\varepsilon) + (1+\cos(\pi d/\varepsilon))/\pi}{2\pi}\right)$$

Kernel half-width: ε = 1.5h (where h is grid spacing).

### 3.4 BDIM2 meta-equation
$$\phi_{\text{out}} = \mu_0 (\phi - v_b) + v_b + \mu_1 \hat{n} \cdot \nabla(\phi - v_b)$$

Where:
- φ is the provisional velocity field
- v_b is the prescribed body velocity
- n̂ is the outward surface normal (= ∇d/|∇d|)
- Inside body (d ≪ -ε): φ_out = v_b (body velocity imposed)
- In fluid (d ≫ ε): φ_out = φ (unchanged)
- Transition layer: smooth blending with normal-derivative correction (2nd-order)

The μ₁ term is what makes BDIM2 second-order near the body (vs first-order BDIM1).

### 3.5 Pressure projection with BDIM
$$\nabla \cdot \left(\frac{w\Delta t}{\rho} \mu_0 \nabla p\right) = \nabla \cdot \tilde{\mathbf{u}}$$

Face coefficients c = (wΔt/ρ)μ₀ ensure zero pressure gradient inside body.
This is a variable-coefficient Poisson equation when bodies are present.


## 4. GRID LAYOUT — Staggered MAC Grid

**Marker-and-Cell (MAC) staggered grid** (Harlow & Welch 1965):
- Pressure p: cell-centred
- Velocity u: x-face-centred (staggered +h/2 in x)
- Velocity v: y-face-centred (staggered +h/2 in y)
- Velocity w (3D): z-face-centred (staggered +h/2 in z)

One ghost cell per side → array shapes (Nx+2, Ny+2) or (Nx+2, Ny+2, Nz+2).
Interior: [1:-1, 1:-1, ...].

**Discrete operators** (all 2nd-order central):
- Gradient (backward diff): (p_i - p_{i-1})/h
- Divergence (forward diff): (u_{i+1} - u_i)/h
- Laplacian: (u_{i+1} - 2u_i + u_{i-1})/h²
- Vorticity 2D: ω_z = ∂v/∂x - ∂u/∂y
- Vorticity 3D: (ω_x, ω_y, ω_z, |ω|), clipped [2:-2]


## 5. TIME INTEGRATION

### 5.1 Heun's method (RK2) — mirrors WaterLily.jl's mom_step!

**Predictor (w=1):**
1. Advection-diffusion on u^n → u*
2. BDIM meta-equation: u* ← BDIM(u*, u^n_body)
3. Pressure projection → ũ

**Corrector (w=1/2):**
1. Advection-diffusion on ũ → u**
2. Rebase from u^n: u** ← u^n + (u** − ũ) = u^n + dt·RHS(ũ)
3. BDIM meta-equation → u**_bdim
4. Average with projected predictor: u_avg ← ½(ũ + u**_bdim)
5. Pressure projection with w=½ → u^{n+1}

Because ũ is divergence-free, div(u_avg) = ½ div(u**_bdim).
w = ½ doubles the Poisson coefficient so the pressure is at the
correct physical scale (the 0.5 cancels).

Heun achieves 2nd-order temporal accuracy. Forward Euler is also supported.

### 5.2 Adams-Bashforth (multistep)
AB2: u^{n+1} = u^n + Δt(3/2 f^n - 1/2 f^{n-1})
Requires storing previous RHS. Second-order, larger stability region than RK2 for diffusion.


## 6. ADVECTION SCHEMES

### 6.1 QUICK (Leonard 1979)
3rd-order Quadratic Upstream Interpolation for Convective Kinematics.
Uses 3-point upstream-biased stencil. Applied with median limiter for boundedness.

### 6.2 ADBQUICKEST (Leonard 1991)
Adaptive Bounded QUICK Estimated Streaming. 3rd-order TVD scheme.
Uses Courant-number-dependent interpolation + TVD limiter.
Key advantage: 3rd-order accuracy + guaranteed boundedness (no spurious oscillations).
This is the default and recommended scheme in lilytorch.

The flux limiter ensures Total Variation Diminishing (TVD) behaviour:
monotone, no new extrema created. The Courant number C = u·Δt/h
determines the interpolation coefficients adaptively.

### 6.3 Other schemes
- CUBISTA (Alves et al. 2003): 2nd-order TVD
- Van Leer: 2nd-order TVD flux limiter
- CDS: 2nd-order central (unbounded — only for smooth problems)
- Semi-Lagrangian (Stam 1999): unconditionally stable backtracing

### 6.4 CFL condition
$$\Delta t \leq \frac{\min(h)}{v_{\max} + 3\nu}$$


## 7. POISSON SOLVERS

### 7.1 FFT — Neumann BCs (DCT-II)
Eigenvalues: λ_k = (2/h²)(cos(πk/N) - 1), k = 0,1,...,N-1
Solve: DCT-II → divide by eigenvalues → inverse DCT-II. Spectral accuracy.

### 7.2 FFT — Free-space (Green's function convolution)
2D Green's function (Hejlesen et al. 2013, 8th-order algebraic smoothing):
$$G(r) = \frac{1}{2\pi}\left[-\frac{1}{2}\ln(r^2+\sigma^2) + \sum_{k=1}^{4}\frac{1}{2k}\frac{\sigma^{2k}}{(r^2+\sigma^2)^k}\right]$$

3D Green's function (Gaussian/erf regularisation):
$$G(r) = -\frac{1}{4\pi r}\operatorname{erf}\left(\frac{r}{\sqrt{2}\sigma}\right)$$

where σ = h. Implemented as zero-padded FFT convolution.

### 7.3 Geometric Multigrid
**V-cycle structure:**
1. Pre-smooth: n_smoothing sweeps of Jacobi (ω=0.7)
2. Restrict: full-weighting restriction to coarse grid
3. Recurse down to 2×2(×2) coarsest grid
4. Prolongate: bilinear/trilinear interpolation
5. Post-smooth: n_smoothing sweeps

**Variable-coefficient stencil** (dimension d):
$$\frac{c_{d+} p_{i+1} - (c_{d+} + c_{d-}) p_i + c_{d-} p_{i-1}}{h^2}$$

where face coefficients c_{d±} are harmonic averages of μ₀ at cell faces.

**Weighted Jacobi smoother:**
$$p_i^{new} = p_i + \omega \frac{r_i}{D_i}$$
where r_i is the residual, D_i is the diagonal coefficient, ω = 0.7.
Spectral radius ≈ cos(π/N) ≈ 1 - π²/2N² for Jacobi on Poisson.

**Red-Black Gauss-Seidel:** Updates odd/even points in two sweeps.
Spectral radius ≈ cos²(π/N) — converges ~2× faster than Jacobi.

### 7.4 MGCG (Multigrid-preconditioned Conjugate Gradient)
Standard CG iteration where each preconditioning step is one V-cycle.
Combines multigrid's fast low-frequency convergence with CG's Krylov
minimality — robust when multigrid alone stalls.

### 7.5 Tuning parameters
- poisson_tol: convergence tolerance (default 1e-5 to 1e-7)
- poisson_max_cycles: max V-cycles (5-30)
- poisson_nsmoothing: smoothing sweeps per level (5-10)
- jacobi_weight: ω for weighted Jacobi (0.6-0.8, default 0.7)
- poisson_warm_start: reuse previous pressure as initial guess


## 8. FORCE COMPUTATION

### 8.1 Viscous forces (contour integral)
$$\mathbf{F}_{\text{visc}} = \int (\boldsymbol{\sigma} \cdot \hat{\mathbf{n}}) \delta_\varepsilon(d - \varepsilon) \, dV$$

### 8.2 Pressure forces
$$\mathbf{F}_{\text{pres}} = -\int p \hat{\mathbf{n}} \delta_\varepsilon(d) \, dV$$

### 8.3 Smoothed delta function
$$\delta_\varepsilon(d) = \frac{1 + \cos(\pi d/\varepsilon)}{2\varepsilon}$$

### 8.4 Torque decomposition
$$\tau_x = \sum(Y f_z) h^3 - \text{com}_y F_z - \sum(Z f_y) h^3 + \text{com}_z F_y$$

### 8.5 Buoyancy (FARMS coupling)
$$F_{z,\text{buoy}} = -\rho_w \frac{m_{\text{link}}}{\rho_b} g \cdot \min\left(\frac{z_s + h_l - z_l}{2 h_l}, 1\right)$$


## 9. BOUNDARY CONDITIONS

Face ordering (axis-minor): 0: x_min, 1: x_max, 2: y_min, 3: y_max, 4: z_min, 5: z_max

- **Dirichlet ("D"):** φ_ghost = 2φ_bc - φ_interior (image/mirror, 2nd-order)
- **Neumann ("N"):** φ_ghost = φ_interior (zero normal gradient)
- Poisson BCs: always Neumann (∂p/∂n = 0)
- Immersed-body BCs: via BDIM2 meta-equation, not ghost cells

Common setups:
- Uniform inflow: D on inlet, D on outlet (or convective)
- Channel: D on walls, N on streamwise
- Free-swimming: all N + free-space Poisson


## 10. VARIABLE-DENSITY COUPLING

When body density ρ_b ≠ fluid density ρ_f:
$$\rho(\mathbf{x}) = \rho_b + (\rho_f - \rho_b) \mu_0(\mathbf{x})$$

Face-centred Poisson coefficients: c_u = Δt/ρ_u, etc.
This creates density jumps at the body interface that the multigrid solver
must handle — the variable-coefficient stencil (§7.3) supports this.


## 11. FARMS / MUJOCO COUPLING

Two-way fluid-structure interaction for articulated bodies:

**Each timestep:**
1. Read link poses from MuJoCo (physics.data.xpos, xmat)
2. Recompute per-link SDFs, take union → composite body SDF
3. Run fluid step: advection-diffusion → BDIM → pressure projection
4. Integrate hydrodynamic forces on each link (pressure + viscous)
5. Write forces back to physics.data.xfrc_applied

AABB narrow-band optimisation: only evaluate SDF near each link's bounding box.

**Body hierarchy (Python classes):**
- Body (base) → BodyAnalytical, BodyMesh, MultiAnimatBodies
- Composites: CompositeBodyAnalytical, CompositeBodyMesh, CompositeSegmentBody
- Fish: BodyFishAnalytical, BodyFishExperimental

**BDIMhandler:** Bridge between FARMS/MuJoCo and the fluid solver.
- _fluid_step_2d / _fluid_step_3d: full fluid update
- apply_forces: integrates and applies hydrodynamic wrenches

**FluidExtension:** FARMS TaskExtension that calls BDIMhandler.step() each
MuJoCo timestep. Handles HDF5 logging.


## 12. PYTORCH / GPU IMPLEMENTATION PATTERNS

### 12.1 Tensor layout
- All fields are PyTorch tensors on GPU (cuda) or CPU
- Shapes: (1, 1, Nx+2, Ny+2) in 2D, (1, 1, Nx+2, Ny+2, Nz+2) in 3D
- Batch dimensions (1,1) are for potential batch processing
- dtype: float32 (critical for stability with torch.compile)

### 12.2 torch.compile
- mode="reduce-overhead" (uses CUDA graphs)
- Targets: adv_diff_solver.solve, forces_shared, forces_body_batch, V-cycle
- Must use module-level functions (not class methods) for compilation
- One-time warm-up cost per compiled function
- float32 required (float64 causes recompilation issues)

### 12.3 Memory management
- 3D simulations on 1024×256×128 grids use ~13 GB VRAM
- CUDA graph private pools consume additional memory
- Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for fragmentation


## 13. CONFIGURATION SYSTEM

**BaseSimConfig** — shared base class for all simulation configs.
Key fields: Nx/Ny/Nz, xmin-zmax, timestep, density, convection_method,
poisson_method/tol/max_cycles, BC types and values, compile flags.

**SimConfig(BaseSimConfig)** — per-experiment config (e.g. single swimmer,
validation case). Overrides __init__ to set experiment-specific parameters.

**Pattern for new simulation:**
```python
class SimConfig(BaseSimConfig):
    def __init__(self):
        super().__init__()
        self.data_folder = ...
        # Grid, physics, BCs, body parameters
        # Extensions (FlowViewer, CameraRecording)
    def extra_simulation_extensions(self, output_folder):
        return [...]
```

## 14. KEY NUMERICAL STABILITY CONSIDERATIONS

1. **CFL condition**: Δt ≤ h/(v_max + 3ν). Violated → blowup.
2. **Poisson convergence**: if residual doesn't drop below tol, velocity
   field retains divergence → pressure oscillations → instability.
3. **torch.compile + float64**: causes recompilation; stick to float32.
4. **Zero pressure inside bodies**: set zero_pressure_inside=True to avoid
   pressure buildup inside immersed bodies.
5. **Warm-start Poisson**: reusing previous p as initial guess significantly
   reduces iteration count (factor 2-5×).
6. **Density ratio**: large ρ_b/ρ_f ratios (>10) can cause conditioning issues
   in the variable-coefficient Poisson solve.


## 15. REFERENCES

- Weymouth & Yue (2011) — BDIM original formulation
- Maertens & Weymouth (2015) — BDIM2, second-order near-body accuracy
- Chorin (1968) — Pressure projection method
- Harlow & Welch (1965) — MAC staggered grid
- Leonard (1979) — QUICK scheme
- Leonard (1991) — ULTIMATE/ADBQUICKEST
- Alves et al. (2003) — CUBISTA scheme
- Hejlesen et al. (2013) — Regularised Green's functions for Poisson
- Stam (1999) — Semi-Lagrangian advection
- Peskin (2002) — Immersed boundary methods review
- Mittal & Iaccarino (2005) — IBM comparison and taxonomy
- Coquerelle & Cottet (2008) — Vortex penalisation method
- Gazzola et al. (2011) — Simulations of optimised swimming
- Briggs, Henson & McCormick (2000) — Multigrid tutorial
- Trottenberg, Oosterlee & Schüller (2001) — Multigrid textbook
"""


# ═════════════════════════════════════════════════════════════════════════════
# Expert system prompts — these tell Claude HOW to use the knowledge base
# ═════════════════════════════════════════════════════════════════════════════

EXPERT_SYSTEM_CHAT = r"""You are a senior research engineer with deep expertise \
in computational fluid dynamics and fluid-structure interaction. You have \
thorough knowledge of the lilytorch codebase (a PyTorch-based BDIM2 solver) \
described in the knowledge base above.

## How to answer:
- Reference specific equations from the knowledge base by section number.
- When retrieved context includes paper excerpts, cite the paper by name \
  and mention equation/section numbers.
- When retrieved context includes code, reference the file and function name.
- Be precise with numerical methods terminology: distinguish spatial order, \
  temporal order, and convergence rate.
- When discussing stability, reference CFL conditions and spectral radii.
- If the knowledge base + retrieved context isn't sufficient, say so honestly.
"""

EXPERT_SYSTEM_CODE = r"""You are a senior research software engineer working on \
lilytorch (a PyTorch-based BDIM2 CFD solver). You have deep knowledge of the \
codebase architecture described in the knowledge base above.

## When generating code:
- Follow existing patterns exactly: BaseSimConfig inheritance, Body hierarchy, \
  FluidSolver API, staggered MAC grid indexing conventions.
- Use PyTorch idioms: tensor ops, proper device handling, dtype consistency.
- Only call functions/methods that exist in retrieved code. If unsure, say so.
- Handle both 2D and 3D cases (check ndim) when existing code does.
- For SimConfig classes, follow the field naming from the knowledge base §13.
- Docstrings following existing style.
"""

EXPERT_SYSTEM_THEORY = r"""You are a computational physics professor advising a \
PhD student building an IBM-based CFD solver. You have deep knowledge of the \
methods described in the knowledge base above.

## When explaining theory:
- Write equations in LaTeX ($..$ inline, $$...$$ display).
- Always cite source papers by name and year. Reference equation numbers if \
  available in retrieved context.
- Explain physical intuition, not just math.
- Connect theory to implementation: after explaining a method, note how it \
  maps to lilytorch code (reference knowledge base sections).
- Be quantitative about numerical properties: give order of accuracy, \
  convergence rates, spectral radii, stability bounds.
- Flag open research problems when relevant.
"""

EXPERT_SYSTEMS = {
    "chat":   EXPERT_SYSTEM_CHAT,
    "code":   EXPERT_SYSTEM_CODE,
    "theory": EXPERT_SYSTEM_THEORY,
}
