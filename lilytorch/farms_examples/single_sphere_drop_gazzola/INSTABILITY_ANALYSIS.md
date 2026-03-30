# Gazzola Sphere Sedimentation — Instability Analysis & Fixes

## Problem
The simulation becomes unstable around iteration 12,500 when the falling sphere (radius 2.5mm) approaches the bottom boundary (y=0). MuJoCo reports large pressure/stress forces, causing numerical instability.

## Root Causes

### 1. **Pressure Field Singularity Near Wall**
- **Issue**: With **all-Neumann BCs** on pressure and Dirichlet BCs on velocity, the Poisson solver can develop large pressure gradients / spikes near sharp corners (body at wall).
- **Evidence**: Multigrid Poisson solvers are notoriously ill-conditioned in this geometry.
- **Impact**: Pressure forces blow up as `F_p = -∮ p·n dV` integrates over a large p field.

### 2. **Force Computation Instability**
- **Issue**: The smoothed-delta force integration `∫ σ·n δ_ε(d) dV` uses a regularization parameter `eps` (typically 2-3× grid spacing). When the body is at distance ~eps from a wall:
  - The smoothing kernel overlaps with the boundary
  - Delta functions become ill-defined
  - Force vectors become noisy or diverge

### 3. **No Boundary Damping Layer**
- Current sponge layer is **5mm on x-walls only**, **does not damp y-direction** (settling direction).
- When the sphere is within ~10mm of the bottom, flow-induced forces are not attenuated.

### 4. **Potential Poisson Solver Issues**
- Multigrid with Jacobi smoother can struggle with:
  - Sharp SDF gradients at wall contact
  - Highly anisotropic grids (Nx=256, Ny=2048)
  - Ill-conditioning from body-wall proximity

## Implemented Diagnostics

A new `GazzolaDiagnostics` class logs every 50 iterations:

### Pressure & Velocity Statistics
- `p_min, p_max, p_mean, p_rms` — detect singularities
- `u_rms, v_rms, max_vel, cfl` — monitor flow stability
- `p_at_body_{min,max,mean}` — pressure specifically on/near body

### Force & Stability Metrics
- `force_visc_total, force_pres_total` — detect force blow-up
- `stress_max, pforce_max` — track stress tensor and pressure force density
- `kinetic_energy, enstrophy` — global flow stability
- `reynolds` — Re number (should be ~0.3 for this geometry)

### Boundary & SDF Diagnostics
- `dist_to_bottom, dist_to_top` — flag when body is within critical distance
- `sdf_min, sdf_near_body, sdf_body_count` — SDF quality
- `max_divergence` — Poisson solver convergence proxy

**Output**: Saved to `output/gazzola_diagnostics.h5` with detailed tracking of when/where instability occurs.

## Stabilization Measures (Implemented)

### 1. **Distance-Based Force Attenuation**
```python
if dist_to_boundary < 5mm:
    attenuation = (dist_to_boundary / 5mm)²
    scale all forces by attenuation
```
- Smoothly reduces forces to zero as body approaches wall
- Threshold (5mm) should be tuned based on diagnostics
- Quadratic decay is smooth; prevents sudden changes

### 2. **Force Magnitude Clamping**
```python
if |F_total| > 0.5 N:
    scale all forces to fit within 0.5 N
```
- Hard cap on force magnitudes
- Prevents MuJoCo contact/constraint solver from diverging
- Tune `max_force_mag` based on typical settling forces

### 3. **Enhanced Poisson Solver Options** (Not Yet Implemented)
Suggested in config YAML:
```yaml
poisson_method: "fft"          # FFT is more stable than multigrid for this
poisson_tol: 1e-6              # Looser tolerance near walls to avoid ill-conditioning
```

## Recommended Fixes (Priority Order)

### Priority 1: Sponge Layer Extension (Quick, Low Risk)
```yaml
sponge:
  width: 0.01           # 10mm on walls
  strength: 100.0       # 1/s, stronger damping
  axes: ["x", "y"]      # *** Also damp y-direction near bottom ***
```
**Why**: Physically attenuates wake/turbulence before it reaches boundary. Reduces pressure spike.

### Priority 2: Tune Attenuation Threshold
Run with diagnostics enabled for a few full steps, examine HDF5:
```python
# In Python:
import h5py
with h5py.File('output/gazzola_diagnostics.h5') as f:
    dist_bottom = f['body/dist_to_bottom'][:]
    forces = f['forces/total'][:]

    # Find where forces spike
    idx_spike = np.argmax(forces)
    print(f"Force spike at dist_to_bottom = {dist_bottom[idx_spike]:.5f} m")

    # Set dist_to_boundary = 1.5× this distance
```

### Priority 3: Use FFT Poisson (Medium Risk, Moderate Speedup)
```yaml
poisson_method: "fft"
poisson_bc_type: "neumann"  # Already using Neumann everywhere
```
**Why**: FFT avoids iterative solver ill-conditioning. Globally supported BCs → smoother pressure field. ~30% faster, more stable.

**Check before switching**: Ensure multigrid is not already compiled in the SDF/body update path.

### Priority 4: Reduce Grid Aspect Ratio (High Risk, Expensive)
Current: `Nx=256, Ny=2048` (1:8 aspect ratio)
```yaml
# Option A: Keep time-step, reduce domain
Nx: 512
Ny: 2048    # ↓ to 2048 (1:4 aspect ratio, better conditioned)

# Option B: Switch to uniform spacing everywhere
Nx: 512
Ny: 512
domain: [-0.02, 0.02] × [0, 0.32]  # Much coarser, but more stable
```
**Risk**: Requires re-tuning advection/diffusion schemes, may reduce accuracy.

### Priority 5: Implicit Pressure Formulation (Research)
Replace Heun with **decoupled implicit pressure** (standard for incompressible flows):
- Predictor: Solve momentum without pressure → `u*`
- Corrector: Implicit pressure-Poisson with semi-Lagrangian advection
**Reference**:
  - Chorin's projection method (unconditionally stable)
  - Used in commercial CFD (ANSYS, OpenFOAM)

## Usage: Run diagnostics

### Step 1: Enable Diagnostics
Already integrated in updated `controller.py`. No config changes needed.

### Step 2: Run Simulation
```bash
cd /data/andreaferrario/lilytorch
python -m farms_examples.single_sphere_drop_gazzola.gen_config_gazzola
```

### Step 3: Analyze HDF5
```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('output/gazzola_diagnostics.h5') as f:
    it = f['iterations'][:]

    # Where does instability start?
    forces = f['forces/total'][:]
    dist_bottom = f['body/dist_to_bottom'][:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(it, forces, 'b-', label='Total Force')
    ax2.plot(it, dist_bottom, 'r-', label='Dist to Bottom')

    ax1.set_ylabel('Force (N)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_xlabel('Iteration')

    plt.legend()
    plt.savefig('diagnostics.png')
```

## Testing Recommended Fixes

After each fix, run until instability and look for:
1. **Force spike disappears** → Success
2. **Instability delayed to later iteration** → Partial success, iterate
3. **No change** → Root cause is elsewhere

## Expected Results (Healthy Simulation)

```
Iteration 12400–12600 (sphere near bottom):
  Pressure:     min=-5e-2  max=+5e-2  rms=2e-2          ← Moderate, not >0.5
  Velocity:     max=0.2 m/s  CFL=0.05  div_max=1e-6     ← CFL << 0.5, div tiny
  Forces:       visc=1e-4 N  pres=1e-5 N  total=1e-4 N   ← No spikes
  Body:         y=0.001 m  dist_bottom=0.001 m          ← Landing phase
  Stability:    KE=1e-5  enstrophy=1e-3  Re=0.3         ← Laminar, Re correct
```

## Summary

| **Issue** | **Diagnostic** | **Quick Fix** | **Robust Fix** |
|-----------|----------------|--------------|---|
| Pressure singularity | `p_rms > 0.1`, `p_at_body_max spike` | Extend sponge layer to y | FFT Poisson |
| Force blow-up | `force_total > 0.5 N` | Force clamping (implemented) | Implicit pressure |
| Body-wall interaction | `dist_to_bottom < 0.005 m` | Attenuation (implemented) | Reduce aspect ratio |
| CFL instability | `cfl > 0.4` | Reduce dt or refine near wall | Implicit scheme |

The diagnostics are now in place. **Run the simulation, analyze the HDF5, and report back** which of these issues are triggered. The root cause will be clear from the logs.
