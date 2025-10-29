# Instructions for FARMS installation

FARMS must first be installed to run the simulation experiments. You will first
need a working [Python](https://www.python.org/) installation with
[`pip`](https://pip.pypa.io/en/stable/installation/) available. Installing FARMS
can then be done by running `setup_farms.py` from within the `/lilytorch/FARMS/`
directory, where the necessary FARMS packages will be automatically installed.
It is highly recommended to run this script within a [Python virtual
environment](https://docs.python.org/3/library/venv.html).

Once the environment is set up, you can run one of the experiments in the
following directories with `sh run.sh`:

- `/lilytorch/1guillasim/1guilla_swm_pos_fst/`
- `/lilytorch/zebrafishsim/zebrafish_trv_wtr_swm_cpl_fst/`

For some preliminary documentation, please check the [FARMS CORE
README](https://github.com/farmsim/farms_core/blob/amphibious_v0.2/README.md)
under the amphibious_v0.2 branch.

Happy simulating!
