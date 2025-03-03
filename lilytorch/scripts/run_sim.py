

from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject


pars = yaml2pyobject("lilytorch/scripts/flow_past_cylinder.yaml")

# modify parameters
# pars["body"]["control"]["f"]  = 10
# pars["output"]["save_frames"] = True
# pars["output"]["save_uv"]     = False
# pars["output"]["save_path"]   = "/data/andreaferrario/ns_data/"

solver = FluidSolver(pars)
solver.run_sim()












