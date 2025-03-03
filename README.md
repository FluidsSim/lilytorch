# lilytorch
Lilytorch implements a rigid-fluid solver in pytorch

# installation steps
It is strongly recommended to set up a virtual environment to use this repository.
Follow the following instruction in the right order (the order is important):
1. Install torch following the instructions at https://pytorch.org/get-started/locally/
2. Install the necessary FARMS packages: enter in the lilytorch/FARMS directory and run
> python setup_farms.py
3. Install the requirements via
> pip install -r requirements.txt
4. Install the lilytorch library by runnning
> pip install -e .

# example scripts
1. Test the correct farms installation by running the file `lilytorch/lilytorch/zebrafish/example_single.py`:
> python lilytorch/lilytorch/zebrafish/example_single.py
2. Test the correct integration with FARMS by running `lilytorch/lilytorch/zebrafish/example_fluid.py`:
> python example_fluid.py



