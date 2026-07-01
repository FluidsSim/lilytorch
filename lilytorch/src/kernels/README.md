# lilytorch.src.kernels

Native CUDA/CPU ops for the lilytorch streamed BDIM path.

## Active operators

| op | Python wrapper | purpose |
| --- | --- | --- |
| `streaming_sdf_min_rho_3d_multi` | `kernels.streaming_sdf_min_rho_3d_multi` | 3-D memory-saving SDF/face-velocity update with winning density |
| `streaming_sdf_forces_post_3d` | `kernels.streaming_sdf_forces_post_3d` | 3-D post-fluid-step force/torque integration |
| `streaming_sdf_min_rho_2d_multi` | `kernels.streaming_sdf_min_rho_2d_multi` | 2-D memory-saving SDF/face-velocity update with winning density |
| `streaming_sdf_forces_post_2d` | `kernels.streaming_sdf_forces_post_2d` | 2-D post-fluid-step force/torque integration |
| `apply_bcs_3d` / `apply_bcs_2d` | `kernels.apply_bcs_3d` / `kernels.apply_bcs_2d` | boundary-condition writes |
| `interpolate_3d` / `interpolate_2d` | `kernels.interp_3d` / `kernels.interp_2d` | uniform-grid interpolation |

The SDF update and force integration are separate kernels in both 2-D and
3-D. The memory-saving SDF update avoids Python-visible per-body CC SDF slabs;
the post force pass re-samples body-local SDF support and uses current union SDF
normals.

## Building

```bash
pip install -e . --no-build-isolation
# or
PYTHON=$(which python) bash lilytorch/src/kernels/build.sh
```

The extension must be compiled against the same PyTorch installation used at
runtime.
