# LilyTorch

LilyTorch is a PyTorch-based 2-D/3-D incompressible-flow solver for
fluid–structure interaction. It combines a Cartesian MAC grid, BDIM2 immersed
boundaries, native C++/CUDA operators, FFT or geometric-multigrid pressure
solvers, and optional FARMS/MuJoCo coupling.

The maintained user guide, numerical documentation, API inventory, examples
index, paper links, and milestone index live in the unified
[`FluidsSim/lilytorch-docs`](https://github.com/FluidsSim/lilytorch-docs)
repository. This repository intentionally does not duplicate those pages.

## Related repositories

- [`FluidsSim/lilytorch_examples`](https://github.com/FluidsSim/lilytorch_examples)
  contains runnable cases, tests, benchmarks, and validation studies.
- [`FluidsSim/paper`](https://github.com/FluidsSim/paper) contains the manuscript
  and supplementary material.
- [`FluidsSim/milestones`](https://github.com/FluidsSim/milestones) contains
  historical design and investigation records.

## Installation

Python 3.9 or newer and a C++ compiler are required. Install the appropriate
CPU or CUDA build of PyTorch first; a matching CUDA toolkit is required when
building CUDA operators.

```bash
git clone https://github.com/FluidsSim/lilytorch.git
git clone https://github.com/FluidsSim/lilytorch_examples.git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -r lilytorch/requirements.txt
python -m pip install -e lilytorch --no-build-isolation
python -m pip install -e lilytorch_examples
```

`--no-build-isolation` is intentional: the native extension must be compiled
against the same PyTorch installation used at runtime. Set
`LILYTORCH_NO_CUDA=1` to force a CPU-only extension build.

Verify the install:

```bash
python -c "import lilytorch; from lilytorch.src import native; print(lilytorch.__version__)"
```

## FARMS/MuJoCo coupling

Initialize the pinned submodules and install the standard FARMS packages:

```bash
git -C lilytorch submodule update --init --recursive
(cd lilytorch/lilytorch/FARMS_V2 && python setup_farms.py)
```

The helper installs `farms_core`, `farms_mujoco`, and `farms_sim`. Install
`farms_amphibious` separately for cases that import it:

```bash
python -m pip install -e lilytorch/lilytorch/FARMS_V2/farms_amphibious \
  --no-build-isolation
```

## Run and test

From the directory containing both checkouts, run a maintained standalone case:

```bash
python -m lilytorch_examples.examples.standalone.runsim \
  flow_past_circle_2d.yaml
```

Run LilyTorch's package tests with:

```bash
python -m pytest lilytorch/tests
```

The production implementation uses native CPU/CUDA operators. There is no
separate `diffusion.py`, `kernels/` package, or selectable Python-versus-native
solver path; old `solver_method` and `use_kernels` values are compatibility
inputs only.

## License

MIT
