# lilytorch.src.kernels

Native CUDA ops for BDIM-IB CFD, ported from
[`pytorch_interpolation`](https://github.com/ferrarioa5/pytorch_interpolation).
This sub-package owns the kernels lilytorch needs at run time so that
the repository no longer has to depend on the external project for the
SDF / force / BC fused operators.

## Operators

Registered under the `lilytorch_kernels` torch library:

| op | Python wrapper | purpose |
| --- | --- | --- |
| `streaming_sdf_min_3d`       | `kernels.streaming_sdf_min_3d`       | one-body fused SDF + face-velocity running-min update |
| `streaming_sdf_min_3d_multi` | `kernels.streaming_sdf_min_3d_multi` | multi-body fused SDF + face-velocity running-min update |
| `bdim_forces_3d_multi`       | `kernels.bdim_forces_3d_multi`       | per-body force / torque integration |
| `apply_bcs_3d`               | `kernels.apply_bcs_3d`               | fused Neumann + Dirichlet BC writes |

All ops have CUDA and CPU implementations. The CPU path is OpenMP-parallelised and mirrors the CUDA kernels line-for-line; it covers float32 / float64 (half precision is CUDA-only).

## Layout

```
kernels/
  __init__.py            # imports _C and re-exports Python wrappers
  ops.py                 # thin Python wrappers around torch.ops.lilytorch_kernels.*
  build.sh               # convenience in-place rebuild
  csrc/
    ops.cpp              # PyInit__C, schemas
    streaming_sdf_cpu.cpp # OpenMP CPU kernels + launchers + CPU registration
    cuda/
      streaming_sdf.cu   # all CUDA kernels + launchers + CUDA registration
```

## Building

The extension is built by the top-level `setup.py`:

```bash
# editable / in-place build (preferred during development)
pip install -e .              # builds lilytorch.src.kernels._C
# or just rebuild the extension after editing CUDA / cpp sources:
bash lilytorch/src/kernels/build.sh
```

You can disable the CUDA build entirely by exporting `LILYTORCH_NO_CUDA=1`
before installing — the extension will be skipped and the Python
wrappers will raise on first call.

## Usage

```python
from lilytorch.src.kernels import (
    streaming_sdf_min_3d_multi,
    bdim_forces_3d_multi,
    apply_bcs_3d,
)
```

Signatures match the original `pytorch_interpolation` exports
1-for-1, so migrating call sites is a `from ... import` change.
