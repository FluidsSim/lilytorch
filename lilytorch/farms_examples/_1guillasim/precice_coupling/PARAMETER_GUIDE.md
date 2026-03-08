# preCICE–OpenFOAM Coupling: Parameter Guide

This document explains every parameter in the OpenFOAM + preCICE coupling
setup and maps each one to its BDIM equivalent (from
`gen_configs_one_pinned.py`).

---

## 1. Domain & Mesh

### 1.1 blockMeshDict — Background mesh

| OpenFOAM parameter | Current value | BDIM equivalent | Notes |
|---|---|---|---|
| `xmin, xmax` | -0.9, 1.5 | `xmin=-0.9, xmax=1.5` | Identical domain extent |
| `ymin, ymax` | -0.3, 0.3 | `ymin=-0.3, ymax=0.3` | Identical domain extent |
| `zmin, zmax` | 0.15, 0.25 | *(2D — no z)* | Thin 3D slab centred at z=0.2 (swimmer spawn height) |
| Cell counts `(240 60 10)` | Nx=240, Ny=60, Nz=10 | `Nx=1024, Ny=256` | See resolution comparison below |

**Resolution comparison:**

The domain is 2.4 m × 0.6 m (× 0.1 m in z).

| | BDIM | OpenFOAM background | OpenFOAM after snappy level 1 | After level 2 | After level 3 |
|---|---|---|---|---|---|
| Δx, Δy | 2.4/1024 ≈ **2.3 mm** | 2.4/240 = **10 mm** | 5 mm | **2.5 mm** | **1.25 mm** |

- The **background mesh** (blockMeshDict) is ~4× coarser than BDIM.
- **snappyHexMesh** then refines near the swimmer body at levels 2–3, giving
  2.5–1.25 mm cells near the surface — comparable to or finer than BDIM.
- Far from the body the mesh is coarse (10 mm), but that region only sees
  smooth, slowly-varying flow, so coarse is fine.

**How to make it finer or coarser:**

To match BDIM resolution everywhere (not just near the body), you would set:
```
hex (0 1 2 3 4 5 6 7) (1024 256 10) simpleGrading (1 1 1)
```
But this gives ~2.6M background cells and is very expensive in 3D.
The current 240×60×10 + snappy refinement is the standard OpenFOAM approach:
coarse far field + fine near body.

---

### 1.2 snappyHexMeshDict — Adaptive refinement near the swimmer

| Parameter | Value | Meaning |
|---|---|---|
| `refinementSurfaces → swimmer → level (2 3)` | min level 2, max level 3 | Cells touching swimmer surface are refined 2–3 times. Each level halves the cell size: level 2 = Δx/4, level 3 = Δx/8 |
| `refinementRegions → refinementBox → levels ((1e15 1))` | level 1 inside box | The entire refinement box (wake region) gets 1 level of refinement (Δx/2 = 5 mm) |
| `refinementBox min/max` | (-0.9, -0.1, 0.15) to (1.5, 0.1, 0.25) | Region around swimmer + wake that gets level 1 refinement |
| `locationInMesh` | (0.5, 0.2, 0.2) | A point that must be **outside** the swimmer STL but **inside** the domain. snappyHexMesh keeps cells on this side. |
| `nCellsBetweenLevels` | 3 | Number of buffer cells between refinement levels (ensures smooth transition) |

**BDIM equivalent:** BDIM uses a uniform grid — there is no adaptive refinement.
The entire domain has the same Δx. In OpenFOAM, you get the same effect by
using a very fine blockMesh + no snappy, but that's expensive in 3D.

**To adjust:**

- `level (3 4)` → finer near body (1.25–0.625 mm), but ~4× more cells
- `level (1 2)` → coarser near body (5–2.5 mm), ~4× fewer cells
- Add more refinement regions (e.g., a tighter box around the swimmer)

---

## 2. Time Stepping

### 2.1 controlDict — OpenFOAM time control

| Parameter | Current value | BDIM equivalent | Notes |
|---|---|---|---|
| `deltaT` | **0.005** | `timestep = 0.0005` | OpenFOAM timestep is 10× larger. Acceptable because OpenFOAM uses an implicit pressure solver (PIMPLE) which is unconditionally stable, unlike BDIM's explicit `abdquickest` scheme which requires small CFL. |
| `endTime` | 10 | `n_iterations × timestep = 20001 × 0.0005 = 10 s` | Same physical end time |
| `writeInterval` | **10** | `save_every = 200` | OpenFOAM writes every 10 timesteps = every 0.05 s. BDIM writes every 200 × 0.0005 = 0.1 s. So OpenFOAM currently saves 2× more frequently. |
| `writeControl` | `timeStep` | — | Could also be `runTime` to write at fixed physical time intervals |
| `writeFormat` | `ascii` | — | Use `binary` for ~2× smaller files + faster I/O |

**Key difference:** BDIM uses an explicit time integration (`abdquickest` =
Adams–Bashforth with QUICK spatial scheme) which is CFL-limited — the timestep
must satisfy Δt < Δx / U_max. With Δx ≈ 2.3 mm and U ≈ 0.22 m/s, this gives
Δt < ~0.01 s, so BDIM's 0.0005 is conservative.

OpenFOAM's **PIMPLE** algorithm is implicit and unconditionally stable, so you
can use much larger timesteps. The limit becomes accuracy, not stability:
- Δt = 0.005 → CFL ≈ U×Δt/Δx = 0.22 × 0.005 / 0.0023 ≈ 0.5 (fine for PIMPLE)
- Δt = 0.001 → CFL ≈ 0.1 (very accurate)
- Δt = 0.0005 → CFL ≈ 0.05 (matches BDIM, very conservative)

**To adjust:**

- Decrease `deltaT` for more temporal accuracy (but slower)
- Increase for speed (up to CFL ~ 2–5 with PIMPLE, but coupling may need smaller steps)
- Must match `time-window-size` in `precice-config.xml` (see §4)
- Must match `timestep` in `gen_configs_precice.py` (FARMS side)

---

## 3. Solver Settings

### 3.1 fvSchemes — Discretisation schemes

| Scheme | Current | BDIM equivalent | Notes |
|---|---|---|---|
| `ddtSchemes → Euler` | 1st-order implicit time | `abdquickest` = 2nd-order explicit (Adams–Bashforth) | Euler is only 1st-order but very stable. For 2nd-order in OpenFOAM, use `backward` or `CrankNicolson 0.9`. |
| `divSchemes → Gauss upwind` | 1st-order upwind convection | QUICK (3rd-order) | Upwind is diffusive but robust. For 2nd-order: `Gauss linearUpwind grad(U)`. For less diffusion: `Gauss LUST grad(U)`. |
| `gradSchemes → Gauss linear` | 2nd-order central | Central differences | Same |
| `laplacianSchemes → Gauss linear limited corrected 0.5` | 2nd-order with non-orthogonal correction | Central differences | The `0.5` limits the non-orthogonal correction for stability on skewed cells |

**To improve accuracy (at cost of stability):**
```
ddtSchemes    { default backward; }                    // 2nd-order time
divSchemes    { div(phi,U) Gauss linearUpwind grad(U); } // 2nd-order convection
```

### 3.2 fvSolution — Linear solvers & PIMPLE algorithm

| Parameter | Current value | Meaning |
|---|---|---|
| **Pressure solver (p)** | PCG + DIC, tol=1e-6, relTol=0.01 | Preconditioned Conjugate Gradient with Diagonal Incomplete Cholesky. Solves the pressure Poisson equation. |
| **Velocity solver (U)** | PBiCGStab + DILU, tol=1e-6, relTol=0.01 | Stabilised Bi-Conjugate Gradient with Diagonal Incomplete LU. Solves momentum equation. |
| **cellDisplacement** | GAMG + GaussSeidel, tol=1e-8 | Geometric-Algebraic Multi-Grid. Solves the mesh motion Laplacian (how internal mesh points move). |

**BDIM equivalent:** BDIM uses a multigrid Poisson solver (`solve_multigrid`).
OpenFOAM's PCG+DIC is iterative (no multigrid for pressure by default). You
could switch to GAMG for pressure too:
```
p { solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.01; }
```

#### PIMPLE algorithm

| Parameter | Current value | Meaning | BDIM equivalent |
|---|---|---|---|
| `nOuterCorrectors` | **2** | Number of outer PIMPLE loops per timestep. More = more accurate but slower. | BDIM does 1 explicit step (no iteration) |
| `nCorrectors` | **1** | Number of pressure-velocity correction steps inside each outer loop | *(part of BDIM projection step)* |
| `nNonOrthogonalCorrectors` | **1** | Extra pressure solves to account for non-orthogonal mesh cells | *(not needed in BDIM — Cartesian grid)* |

**PIMPLE explained:** Each timestep, PIMPLE does:
1. Solve momentum (predict U)
2. Solve pressure Poisson (correct p)
3. Correct velocity for pressure gradient
4. Repeat steps 1–3 `nOuterCorrectors` times

This is like multiple "sub-iterations" per timestep. With `nOuterCorrectors=2`,
it's essentially doing 2 complete solves, which allows larger timesteps.

**Relaxation factors:**
| Parameter | Value | Notes |
|---|---|---|
| `U.*` | 0.7 | Under-relax velocity by 30% between outer iterations |
| `p.*` | 0.7 | Under-relax pressure by 30% |

These prevent oscillations in the PIMPLE outer loop. Only active when
`nOuterCorrectors > 1`. Set to 1.0 for the `Final` corrector automatically.

---

## 4. preCICE Coupling

### 4.1 precice-config.xml — Coupling scheme

| Parameter | Current value | Meaning |
|---|---|---|
| `time-window-size` | **0.005** | Size of each coupling window. **Must equal** `deltaT` in controlDict and `timestep` in gen_configs_precice.py |
| `max-time` | 10.0 | Total simulation time |
| `max-iterations` | 5 | Max implicit sub-iterations per coupling window. If convergence isn't reached in 5 iterations, it moves on. |
| `relative-convergence-measure limit` | 1e-3 | Coupling converges when both Displacement and Force change by < 0.1% between sub-iterations |
| `serial-implicit` | — | FARMS computes first (displacement), then OpenFOAM (forces). They iterate until converged. |

**BDIM equivalent:** BDIM has no coupling — the fluid and body are solved
monolithically in the same code. preCICE partitioned coupling adds overhead
(multiple implicit iterations per timestep).

**Acceleration (under-relaxation):**

| Parameter | Value | Meaning |
|---|---|---|
| `acceleration:aitken` | — | Aitken adaptive under-relaxation to speed up convergence |
| `initial-relaxation` | 0.5 | First iteration uses 50% relaxation; Aitken adjusts automatically |
| `data: Force` | — | Relaxation is applied to the Force data |

**To adjust:**
- Increase `max-iterations` (e.g., 10) for tighter convergence
- Decrease `limit` (e.g., 1e-5) for more accuracy
- Use `serial-explicit` instead to skip sub-iterations entirely (fastest, but less stable)

### 4.2 preciceDict — OpenFOAM adapter config

| Parameter | Value | Meaning |
|---|---|---|
| `participant` | OpenFOAM | This participant's name in the preCICE config |
| `modules (FSI)` | — | Use the Fluid-Structure Interaction module |
| `patches (swimmer)` | — | OpenFOAM patches coupled to preCICE |
| `locations faceCenters` | — | Coupling data is exchanged at face centres (not vertices) |
| `readData: Displacement` | — | OpenFOAM reads body displacement from FARMS |
| `writeData: Force` | — | OpenFOAM sends fluid forces back to FARMS |
| `rho [1 -3 0 0 0 0 0] 1000` | — | Fluid density for force computation (must match transportProperties) |

### 4.3 Mapping

| Parameter | Value | Meaning |
|---|---|---|
| `mapping:nearest-neighbor` | — | Map data between FARMS mesh (STL vertices) and OpenFOAM mesh (face centres) using nearest point |
| `geometric-filter="no-filter"` | — | Required for parallel OpenFOAM (decomposed mesh). Without this, some coupling points may be filtered out. |

---

## 5. Parallel Execution

| Parameter | Value | File | BDIM equivalent |
|---|---|---|---|
| `numberOfSubdomains` | 16 | decomposeParDict | BDIM runs on GPU (single device) |
| `method` | scotch | decomposeParDict | Automatic domain decomposition |
| MPI ranks | 16 | run.sh (`mpirun -np 16`) | — |

---

## 6. Physical Properties

| Property | OpenFOAM | BDIM | File |
|---|---|---|---|
| Kinematic viscosity ν | 1.0e-6 m²/s | `nu = 1.0e-6` | transportProperties |
| Fluid density ρ | 1000 kg/m³ | `density = 800` (body), `rho = 1000` (fluid in solver YAML) | preciceDict, transportProperties |
| Inlet velocity | 0.215971 m/s | `u_inlet = 0.215971` | 0.orig/U |
| Reynolds number | Re = U·L/ν ≈ 0.216 × 0.8 / 1e-6 ≈ **173,000** | Same | — |

---

## 7. Output Fields

OpenFOAM writes **all registered fields** at each write time:
- `U` — velocity vector (3 components) ← **this is your (u,v,w)**
- `p` — pressure (kinematic, i.e. p/ρ) ← **this is your p**
- `cellDisplacement` — mesh motion field (internal OpenFOAM use)
- `pointDisplacement` — mesh motion at vertices (internal)

The `cellDisplacement` and `pointDisplacement` fields are small overhead.
The `fieldAverage` function (which computed UMean, pMean, etc.) has been
**removed** to save disk space.

**BDIM equivalent:** BDIM saves `(u, v)` velocity components and `p` pressure.
OpenFOAM's `U` contains `(Ux, Uy, Uz)` — the 3D equivalent.

---

## 8. Quick Reference: Changing Resolution/Speed

### Want BDIM-equivalent uniform resolution?
```python
# blockMeshDict:
hex (0 1 2 3 4 5 6 7) (1024 256 10) simpleGrading (1 1 1)
# snappyHexMeshDict: disable (set castellatedMesh false; snap false;)
# Result: ~2.6M cells, Δx ≈ 2.3 mm everywhere, VERY slow
```

### Want faster (coarser)?
```python
# blockMeshDict:
hex (0 1 2 3 4 5 6 7) (120 30 10) simpleGrading (1 1 1)
# snappyHexMeshDict: level (1 2) instead of (2 3)
# Result: ~75K cells, Δx ≈ 20 mm background, 5–2.5 mm near body
```

### Want more temporal accuracy?
```python
# controlDict:      deltaT 0.001;
# precice-config:   <time-window-size value="0.001" />
# gen_configs:      timestep = 0.001; n_iterations = 10001
```

### Want even faster time-stepping?
```python
# controlDict:      deltaT 0.01;     (CFL ≈ 1)
# precice-config:   <time-window-size value="0.01" />
# gen_configs:      timestep = 0.01; n_iterations = 1001
# fvSolution:       nOuterCorrectors 3;  (need more PIMPLE iters at high CFL)
```

---

## 9. Summary of Current vs BDIM Settings

| Parameter | Current OpenFOAM | BDIM (gen_configs_one_pinned.py) | Ratio |
|---|---|---|---|
| Domain x | [-0.9, 1.5] | [-0.9, 1.5] | Same |
| Domain y | [-0.3, 0.3] | [-0.3, 0.3] | Same |
| Domain z | [0.15, 0.25] (3D) | *(2D)* | — |
| Δx near body | ~1.25–2.5 mm (snappy level 2–3) | ~2.3 mm | ~Same |
| Δx far field | 10 mm (background) | 2.3 mm | 4× coarser |
| Timestep | 0.005 s | 0.0005 s | 10× larger |
| Time scheme | Euler (1st-order implicit) | abdquickest (2nd-order explicit) | Different |
| Convection scheme | upwind (1st-order) | QUICK (3rd-order) | Lower order |
| Pressure solver | PCG iterative | Multigrid direct | Different |
| Save frequency | every 0.05 s | every 0.1 s | 2× more |
| Parallelism | 16 MPI ranks (CPU) | GPU (single device) | Different |

---

## 10. File Locations

All template files are in:
```
lilytorch/farms_examples/_1guillasim/precice_coupling/
├── gen_configs_precice.py          # Main launcher — sets timestep, domain, n_iterations
├── precice-config.xml              # preCICE coupling scheme
├── precice_handler.py              # FARMS-side coupling code
├── precice_extension.py            # FARMS extension wrapper
├── plot_results.py                 # Post-processing
└── openfoam_case/
    ├── system/
    │   ├── blockMeshDict           # Background mesh (domain, cell counts)
    │   ├── snappyHexMeshDict       # Adaptive refinement near body
    │   ├── controlDict             # Time stepping, output control
    │   ├── fvSchemes               # Discretisation schemes
    │   ├── fvSolution              # Linear solvers, PIMPLE settings
    │   ├── decomposeParDict        # Parallel decomposition
    │   └── preciceDict             # OpenFOAM-preCICE adapter config
    ├── constant/
    │   ├── transportProperties     # Fluid properties (nu)
    │   ├── dynamicMeshDict         # Mesh motion solver
    │   └── triSurface/swimmer.stl  # Swimmer geometry
    ├── 0.orig/                     # Initial/boundary conditions
    │   ├── U                       # Velocity BCs
    │   ├── p                       # Pressure BCs
    │   └── pointDisplacement       # Mesh motion BCs
    └── prepare_mesh.sh             # Mesh generation script
```

**Key rule:** When changing the timestep, you must update **three files** consistently:
1. `gen_configs_precice.py` → `timestep = ...`
2. `openfoam_case/system/controlDict` → `deltaT ...;`
3. `precice-config.xml` → `<time-window-size value="..." />`
