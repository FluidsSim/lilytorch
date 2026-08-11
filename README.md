# LilyTorch

LilyTorch is a PyTorch-based 2-D/3-D incompressible-flow solver for
fluid–structure interaction. It combines a Cartesian MAC grid, BDIM2 immersed
boundaries, native C++/CUDA operators, FFT or geometric-multigrid pressure
solvers, and optional FARMS/MuJoCo coupling.

The maintained user guide, numerical documentation, API inventory, examples
index, paper links, and milestone index live in the unified
[`FluidsSim/lilytorch-docs`](https://github.com/FluidsSim/lilytorch-docs)
repository. This repository intentionally does not duplicate those pages.

## Examples

An set of examples will be soon made available. In the meantime, some simulation
examples are shown below.

<table>
<tr>
<td width="50%">

**Boat**

<video src="https://github.com/user-attachments/assets/36c3d06b-aced-45c4-a61c-3aaa5a3ddbad" width="100%" height="240" controls muted></video>

</td>
<td width="50%">

**Submarine**

<video src="https://github.com/user-attachments/assets/2f4ddde8-4d87-49ed-954d-24a5cabd732b" width="100%" height="240" controls muted></video>

</td>
</tr>
<tr>
<td width="50%">

**Salamander — swimming 2d**

<video src="https://github.com/user-attachments/assets/95f012d1-1b49-4c24-a9ae-dc4e41c26db8" width="100%" height="240" controls muted></video>

</td>
<td width="50%">

**Zebrafish — 3-D, vorticity magnitude**

<video src="https://github.com/user-attachments/assets/d2998e68-b97c-45b0-9c7e-e9cd7db5cb7e" width="100%" height="240" controls muted></video>

</td>
</tr>
<tr>
<td width="50%">

**Three spheres VOF**

<video src="https://github.com/user-attachments/assets/a4f9c598-ad5d-401d-8b3f-7abf634adfc4" width="100%" height="240" controls muted></video>

</td>
<td width="50%">

**1guilla robot surface swimming**

<video src="https://github.com/user-attachments/assets/de08af64-c821-40f0-a54a-7bc89bf82886" width="100%" height="240" controls muted></video>

</td>
</tr>
</table>


## Validation with robot experiments

### 1guilla in water - Rhodamine dye - Re~160000

<video src="https://github.com/user-attachments/assets/71332af2-63d4-4604-8bd4-fbdac16104ec" width="100%" height="600" controls muted></video>

### 1guilla in carboxymethyl cellulose and water thick solution - Fluorescein dye - Re~100

<video src="https://github.com/user-attachments/assets/88a075d3-9c22-4f99-a7d5-cf6ff57be1ad" width="100%" height="600" controls muted></video>


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

Run LilyTorch's companion test suite from the directory containing both
checkouts with:

```bash
python -m pytest lilytorch_examples/lilytorch_examples/tests
```

The production implementation uses native CPU/CUDA operators. There is no
separate `diffusion.py`, `kernels/` package, or selectable Python-versus-native
solver path; old `solver_method` and `use_kernels` values are compatibility
inputs only.

## License

MIT
