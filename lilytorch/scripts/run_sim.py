

from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch

# pars = yaml2pyobject("lilytorch/scripts/fish_analytical.yaml")

# pars = yaml2pyobject("lilytorch/scripts/flow_past_cylinder.yaml")
pars = yaml2pyobject("lilytorch/scripts/moving_cylinder_mesh.yaml")

# modify parameters
# pars["body"]["control"]["f"]  = 10
# pars["output"]["save_frames"] = True
# pars["output"]["save_uv"]     = False
# pars["output"]["save_path"]   = "/data/andreaferrario/ns_data/"

solver = FluidSolver(pars, dtype=torch.float32, compute_forces=True)
solver.run_sim()










