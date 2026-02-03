# lilytorch
Lilytorch implements a 2D rigid-fluid solver in pytorch and integrated with MuJoCo via the FARMS framework (https://github.com/farmsim). In a nutshell, this package allows for the simulation of 2d fluids on top of the MuJoCo physics enging. This coupling allows the simulation of multi-rigid body systems and their interactions, including robots, animals. 

It is strongly recommended to set up a virtual environment to use this repository.


# installation steps
Install FARMS:
Run the following from the repository root:

1. Install PyTorch from https://pytorch.org/get-started/locally/ 

2. Install the necessary requirements:
```bash
python -m pip install -r requirements.txt
```

3. Initialize submodules and install FARMS:
```bash
git submodule update --init --recursive
cd lilytorch/FARMS_V2
python setup_farms.py
cd -
```

4. Install lilytorch in editable mode:
```bash
pip install -e .
```




