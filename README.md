# Lilytorch

**Lilytorch** is a GPU-accelerated 2D computational fluid dynamics (CFD) package built on [PyTorch](https://pytorch.org/), implementing a **BDIM2 (Boundary Data Immersion Method)** solver for fluid–structure interaction. It integrates with [MuJoCo](https://mujoco.org/) via the [FARMS](https://github.com/farmsim) framework, enabling two-way coupled simulations of articulated bodies (robots, animals) immersed in viscous fluids.

![Anguilliform Swimming](images/curl_1guilla.png)

## Overview

Lilytorch solves the 2D/3D incompressible Navier-Stokes equations on a Cartesian grid with immersed bodies represented via Signed Distance Functions (SDFs) using a second order Boundary Data Immersion Method. When coupled with FARMS/MuJoCo, the fluid solver runs alongside the multibody dynamics engine: at each timestep it reads body poses from MuJoCo, solves the fluid equations, computes hydrodynamic forces (pressure + viscous drag), and applies them back as external wrenches — achieving closed-loop fluid–structure coupling.

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
| `video_postprocess.py` | Utility to assemble saved PNG frames into MP4 or GIF videos. |

### FARMS/MuJoCo Integration (`lilytorch/integration/`)

| Module | Description |
|---|---|
| `extensions.py` | `FluidExtension` — FARMS `TaskExtension` subclass that initializes the fluid solver at episode start and applies hydrodynamic forces at each `before_step`. Also includes `DataLogger` for HDF5 logging. |
| `flow_viewer.py` | `FlowViewer` — FARMS `TaskExtension` that renders fluid fields (vorticity, pressure, velocity) as coloured spheres directly inside the MuJoCo viewer window. See [FlowViewer](#flowviewer--in-viewer-flow-visualisation). |
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

3. **Install PyTorch:**

   There are two installation modes:
   - **CPU-only**: No additional prerequisites — install the CPU build of PyTorch.
   - **CPU/CUDA**: Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) to be installed on your system. Install the CUDA build of PyTorch matching your CUDA version. Use this mode if you want to run the simulation on the GPU.

   Visit the official selector to get the right install command for your setup:
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
| [PyTorch](https://pytorch.org/) | GPU/-accelerated tensor computation for the CFD solver |
| [MuJoCo](https://mujoco.org/) / `dm_control` | Rigid-body multibody dynamics engine |
| [FARMS](https://github.com/farmsim) | Neuromechanical simulation framework |
| NumPy / SciPy | Array operations, splines, signal processing, ODE integration |
| [Open3D](http://www.open3d.org/) | 3D mesh loading and SDF computation |
| scikit-fmm | Fast marching method for SDF computation |
| matplotlib / OpenCV | Visualization and video generation |
| PETSc / petsc4py | Parallel sparse Poisson solver (optional) |

## Generating Videos from Simulation Output

After a simulation run, the solver saves PNG frames for each field (vorticity, pressure, velocity magnitude, etc.) into sub-folders under the output directory. Use `video_postprocess.py` to assemble these frames into MP4 or GIF videos.

### Basic Usage

```bash
# Generate videos for ALL fields in a specific run directory
python lilytorch/src/video_postprocess.py /path/to/save_path/2026-03-05_12-00-00/

# Pass the parent save_path — the script auto-picks the latest run
python lilytorch/src/video_postprocess.py /path/to/save_path/
```

### Selecting Specific Fields

```bash
# Only generate videos for selected fields
python lilytorch/src/video_postprocess.py /data/andreaferrario/ns_data/1guilla_self_propelled/swimming2 --fields omega_z_3d vel_mag_3d --format gif
```

Available field names depend on the simulation (2D vs 3D). Common ones include:
- **2D:** `omega_z`, `vel_mag`, `pressure`
- **3D:** `omega_x_3d`, `omega_y_3d`, `omega_z_3d`, `omega_mag_3d`, `vel_mag_3d`, `pressure_3d`

### Options

| Flag | Default | Description |
|---|---|---|
| `--fields F1 F2 ...` | all | Only generate videos for the listed sub-folders |
| `--fps N` | auto | Override frame rate (by default computed from `dt * save_every`) |
| `--slow-factor S` | 1.0 | Real-time multiplier. 1.0 = physical time equals video time |
| `--no-overlay` | off | Disable the simulation-time text overlay on each frame |
| `--crf N` | 18 | H.264 quality (lower = better; 18 ≈ visually lossless) |
| `--format {mp4,gif}` | mp4 | Output format: `mp4` (H.264 video) or `gif` (animated GIF) |

### Examples

```bash
# Slow-motion (5× slower than real time), high quality
python lilytorch/src/video_postprocess.py /path/to/run_dir --slow-factor 5.0 --crf 15

# Fixed 30 FPS, no timestamp overlay
python lilytorch/src/video_postprocess.py /path/to/run_dir --fps 30 --no-overlay

# 3D cylinder wake — only vorticity z-component and velocity magnitude
python lilytorch/src/video_postprocess.py /data/andreaferrario/ns_data/1guilla_self_propelled/swimming1 --fields omega_z_3d vel_mag_3d

# Generate GIF instead of MP4
python lilytorch/src/video_postprocess.py /path/to/run_dir --format gif

# GIF of a specific field with slow-motion
python lilytorch/src/video_postprocess.py /path/to/run_dir --fields omega_z_3d --format gif --slow-factor 5.0
```

### Notes

- The script reads `parameters.yaml` from the run directory to compute the FPS from `dt` and `save_every`. If the file is missing, it defaults to 10 FPS.
- **ffmpeg** is the preferred backend (H.264, concat demuxer, optional drawtext overlay). If ffmpeg is not installed, the script falls back to OpenCV `VideoWriter`.
- Output files are saved alongside the PNG sub-folders inside the run directory (e.g. `omega_z_3d.mp4` or `omega_z_3d.gif`).
- GIF output uses a two-pass palette approach via ffmpeg for high quality. If ffmpeg is unavailable, the script falls back to Pillow.

## FlowViewer — In-Viewer Flow Visualisation

FlowViewer is a FARMS `TaskExtension` that renders a fluid field (e.g. vorticity, pressure, velocity magnitude) as coloured spheres directly inside the MuJoCo viewer window **and** in the recorded video, overlaid on the swimming animat. This gives real-time visual feedback of the flow during a coupled simulation without requiring a separate plotting window.

Spheres are injected into both:
- The **interactive viewer's** `user_scn` (visible in the MuJoCo GUI window).
- The **CameraRecording** extension's offscreen renderer (visible in the saved MP4 video).

### How It Works

1. During `initialize_episode`, FlowViewer pre-allocates a fixed number of sphere geoms in the MuJoCo viewer's `user_scn`.
2. At every `update_every` timesteps (defaults to the solver's `save_every`), it:
   - Extracts the chosen scalar field from the fluid solver.
   - Applies Gaussian smoothing and crops boundary cells.
   - Masks out the body interior using the SDF.
   - Finds grid points where the field exceeds `iso_fraction × peak|field|`.
   - Sub-samples to the sphere budget and updates sphere positions and colours.
3. **Bipolar fields** (vorticity components): positive values are red, negative values are blue.
4. **Non-negative fields** (velocity magnitude, pressure): displayed in orange.

### Prerequisites

- FlowViewer must be listed **after** `FluidExtension` in the extensions list (it reads the solver state that `FluidExtension` computes in `before_step`).
- A 3D fluid solver must be active (`w0` must exist); 2D simulations are skipped.
- For the **interactive viewer**: set `headless: false`.
- For the **recorded video**: include a `CameraRecording` extension in the extensions list. FlowViewer automatically patches its offscreen renderer.
- Both modes work simultaneously when both the viewer and CameraRecording are present.

### Configuration

Add FlowViewer to the `extensions` list in your `gen_configs` file:

```python
extensions = [
    # ... FluidExtension must come first ...
    {
        "loader": "lilytorch.integration.flow_viewer.FlowViewer",
        "config": {
            "field":         "omega_z",   # scalar field to display
            "max_spheres":   4000,         # sphere budget (max visual geoms)
            "iso_fraction":  0.15,         # threshold = fraction × peak |field|
            "smooth_sigma":  2.5,          # Gaussian smoothing (grid cells)
            "crop_boundary": 3,            # cells to crop from each domain face
            "sphere_size":   0.004,        # radius of each sphere (MuJoCo units)
            "update_every":  None,         # None → uses solver.save_every
        },
    },
]
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `field` | `"omega_z"` | Which scalar field to visualise. See available fields below. |
| `max_spheres` | `4000` | Maximum number of sphere geoms to allocate. MuJoCo's hard limit is 100 000 geoms per scene. Higher values give denser coverage but cost more rendering time. |
| `iso_fraction` | `0.15` | Isosurface threshold as a fraction of the peak absolute field value. Lower values show more of the field; higher values show only the strongest features. |
| `smooth_sigma` | `2.5` | Standard deviation (in grid cells) for Gaussian smoothing of the field before thresholding. Reduces noise and produces cleaner isosurfaces. Set to `0` to disable. |
| `crop_boundary` | `3` | Number of grid cells to discard from each face of the domain. Removes boundary artefacts from the visualisation. |
| `sphere_size` | `0.004` | Radius of each sphere in MuJoCo world units. Adjust relative to the body size for visual clarity. |
| `update_every` | `None` | How often (in solver iterations) to refresh the spheres. `None` defaults to the solver's `save_every` value. |

### Available Fields

| Field name | Description | Colour scheme |
|---|---|---|
| `omega_x` | Vorticity x-component | Bipolar (red +, blue −) |
| `omega_y` | Vorticity y-component | Bipolar (red +, blue −) |
| `omega_z` | Vorticity z-component | Bipolar (red +, blue −) |
| `omega_mag` | Vorticity magnitude | Orange |
| `vel_mag` | Velocity magnitude | Orange |
| `pressure` | Pressure field | Bipolar (red +, blue −) |

### Example

In `gen_configs_one_pinned_3d.py`:

```python
headless = False   # must be False for FlowViewer

simulation_dict = {
    # ... other config ...
    "extensions": [
        {
            "loader": "lilytorch.integration.extensions.FluidExtension",
            "config": { ... },
        },
        {
            "loader": "lilytorch.integration.flow_viewer.FlowViewer",
            "config": {
                "field":        "omega_z",
                "max_spheres":  4000,
                "iso_fraction": 0.15,
                "sphere_size":  0.004,
            },
        },
    ],
}
```

### Tuning Tips

- **Too few / too many spheres visible?** Adjust `iso_fraction`. A value of `0.10` shows more of the wake; `0.25` highlights only the strongest vortices.
- **Noisy / speckled appearance?** Increase `smooth_sigma` (e.g. `3.0`–`4.0`).
- **Spheres too small or too large?** Scale `sphere_size` relative to your swimmer's body length.
- **Slow rendering?** Reduce `max_spheres` or increase `update_every`.
- **Want to see pressure instead of vorticity?** Change `field` to `"pressure"` or `"vel_mag"`.




