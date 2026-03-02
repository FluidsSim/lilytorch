# Lilytorch

**Lilytorch** is a GPU-accelerated 2D computational fluid dynamics (CFD) package built on [PyTorch](https://pytorch.org/), implementing a **BDIM2 (Boundary Data Immersion Method)** solver for fluid–structure interaction. It integrates with [MuJoCo](https://mujoco.org/) via the [FARMS](https://github.com/farmsim) framework, enabling two-way coupled simulations of articulated bodies (robots, animals) immersed in viscous fluids.



## Overview

Lilytorch solves the 2D incompressible Navier-Stokes equations on a Cartesian grid with immersed bodies represented via Signed Distance Functions (SDFs). When coupled with FARMS/MuJoCo, the fluid solver runs alongside the multibody dynamics engine: at each timestep it reads body poses from MuJoCo, solves the fluid equations, computes hydrodynamic forces (pressure + viscous drag), and applies them back as external wrenches — achieving closed-loop fluid–structure coupling.

### Architecture

```
YAML Config
    │
    ▼
┌──────────────────────────────────────────────┐
│  FluidSolver (solver.py)                     │
│  ├── AdvDiffSolver     (velocity transport)  │
│  ├── PoissonSolverFFT  (pressure projection) │
│  ├── PoissonSolver     (multigrid fallback)  │
│  └── CompositeBody     (immersed bodies)     │
│       └── Body / BodyMesh / BodyFish...      │
│            (SDF geometry + kinematics)        │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │  Standalone mode    │  Coupled mode (FARMS)
    │  (solver.step_)     │
    │                     ▼
    │          ┌─────────────────────┐
    │          │  FluidExtension     │
    │          │  (extensions.py)    │
    │          └────────┬────────────┘
    │                   │
    │          ┌────────▼────────────┐
    │          │  BDIMhandler        │
    │          │  (per-example)      │
    │          │  ┌────────────────┐ │
    │          │  │ FluidSolver    │ │
    │          │  └────────────────┘ │
    │          └────────┬────────────┘
    │                   │ forces
    │          ┌────────▼────────────┐
    │          │  FARMS / MuJoCo     │
    │          │  (multibody + neuro)│
    │          └─────────────────────┘
    │
    ▼
  Output: velocity/pressure fields, forces,
          frames → video, HDF5 logs, metrics
```

- **Standalone mode**: The `FluidSolver` runs the full Navier-Stokes loop (advection-diffusion → pressure Poisson → velocity correction → body update → force computation) independently.
- **Coupled mode**: `FluidExtension` hooks into the FARMS/MuJoCo simulation loop via the `BDIMhandler`, which bridges body kinematics from FARMS sensors to the fluid solver and applies computed hydrodynamic forces back to MuJoCo bodies.

## Package Structure

### Core Solver (`lilytorch/src/`)

| Module | Description |
|---|---|
| `solver.py` | Main `FluidSolver` class implementing the BDIM2 Navier-Stokes solver: grid setup, time-stepping (Heun, Adams-Bashforth), pressure projection, IBM forcing, and force computation on immersed bodies. |
| `body.py` | Immersed body representation via SDFs. Class hierarchy: `Body` → `BodyAnalytical`, `BodyMesh`, `BodyFishAnalytical`, `BodyFishExperimental`, plus composite wrappers for multi-link articulated models. Includes `body_from_yaml()` factory. |
| `adv_diff.py` | Advection-diffusion solver for velocity transport. Supports implicit/explicit/QUICK/ABDQUICKEST/Adams-Bashforth schemes with Dirichlet/Neumann BCs. |
| `diffusion.py` | Stand-alone diffusion solver using multigrid V-cycles. |
| `poisson_fft.py` | FFT-based Poisson solver for the pressure equation using Green's function convolution (via `torch_dct`). Pre-computes and caches Green's functions to disk. |
| `poisson_mult.py` | Multigrid Poisson solver with Jacobi smoothing for variable-coefficient problems. |
| `poisson_multigrid.py` | Extended multigrid implementation with full V-cycle hierarchy, restriction, and prolongation. |
| `poisson_petsc.py` | PETSc-based Poisson solver using sparse KSP solvers (optional, requires `petsc4py`). |
| `dynamic_water.py` | `WaterDynamicsCallback` that feeds CFD-computed velocity fields back to MuJoCo as spatially-varying water drag for closed-loop coupling. |
| `plotting.py` | Visualization of velocity fields, vorticity, pressure, SDFs, and time histories. |
| `parser.py` | CLI argument parser for selecting YAML config files. |
| `video_from_png.py` | Utility to assemble saved PNG frames into MP4 videos. |

### FARMS/MuJoCo Integration (`lilytorch/integration/`)

| Module | Description |
|---|---|
| `extensions.py` | `FluidExtension` — FARMS `TaskExtension` subclass that initializes the fluid solver at episode start and applies hydrodynamic forces at each `before_step`. Also includes `DataLogger` for HDF5 logging. |
| `gen_pool_sdf.py` | Generates SDF XML files defining rectangular pool arenas with collision walls for MuJoCo. |

### Utilities (`lilytorch/util/`)

| Module | Description |
|---|---|
| `mp_util.py` | Multiprocessing utility for 1D/2D parameter sweeps. |
| `yaml_operations.py` | YAML read/write helpers. |
| `paths.py` | Canonical path constants for the repository. |

### Example Simulations (`lilytorch/farms_examples/`)

| Directory | Description |
|---|---|
| `_1guillasim/` | Anguilliform (eel-like) swimmer — 1 and 2 swimmer configurations. |
| `zebrafishsim/` | Zebrafish larva swimmer. |
| `salamander/` | Salamander swimming and paddling gaits. |
| `amphibot/` | Amphibot robot experimental data and analysis. |
| `single_sphere_drop_*` | Validation benchmarks — sphere sedimentation test cases (Coquerelle & Gazzola). |
| `sphere/` | Falling cylinder experiment. |
| `sdfs/` | Pre-built SDF arena and animat descriptions. |

Each example contains a `BDIMhandler.py` (fluid–body interface), config generators, PD controllers, and plotting scripts.

### FARMS Submodules (`lilytorch/FARMS_V2/`)

Four git submodules from [farmsim](https://github.com/farmsim), pinned to the `amphibious_v0.2` branch:

| Submodule | Role |
|---|---|
| `farms_core` | Core framework: simulation options, sensor conventions, data structures, extensions API. |
| `farms_mujoco` | MuJoCo backend: physics simulation, task management, swimming/drag handlers. |
| `farms_sim` | Simulation launcher: `setup_from_clargs()`, `run_simulation()`. |
| `farms_amphibious` | Amphibious animal models: animat options, kinematics, neural controllers. |

## Installation

> It is strongly recommended to use a virtual environment.

### Prerequisites

- Python ≥ 3.8
- A C compiler (for Cython extensions in FARMS)
- CUDA-capable GPU (recommended for performance, CPU also supported)

### Steps

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules <repo-url>
   cd lilytorch
   ```

   If already cloned without submodules:
   ```bash
   git submodule update --init --recursive
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install PyTorch** (select the appropriate CUDA version for your system):
   https://pytorch.org/get-started/locally/

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install the FARMS submodules:**
   ```bash
   cd lilytorch/FARMS_V2
   python setup_farms.py
   cd -
   ```
   This installs `farms_core`, `farms_mujoco`, `farms_sim`, and `farms_amphibious` in editable mode with their Cython extensions compiled.

6. **Install lilytorch in editable mode:**
   ```bash
   pip install -e .
   ```

### Key Dependencies

| Dependency | Purpose |
|---|---|
| [PyTorch](https://pytorch.org/) | GPU-accelerated tensor computation for the CFD solver |
| [MuJoCo](https://mujoco.org/) / `dm_control` | Rigid-body multibody dynamics engine |
| [FARMS](https://github.com/farmsim) | Neuromechanical simulation framework |
| NumPy / SciPy | Array operations, splines, signal processing, ODE integration |
| [Open3D](http://www.open3d.org/) | 3D mesh loading and SDF computation |
| scikit-fmm | Fast marching method for SDF computation |
| matplotlib / OpenCV | Visualization and video generation |
| PETSc / petsc4py | Parallel sparse Poisson solver (optional) |

```




