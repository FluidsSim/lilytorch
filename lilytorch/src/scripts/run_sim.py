

from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch

# pars = yaml2pyobject("lilytorch/src/scripts/configs/fish_analytical.yaml")

# pars = yaml2pyobject("lilytorch/src/scripts/configs/flow_past_cylinder.yaml")
# pars = yaml2pyobject("lilytorch/src/scripts/configs/moving_cylinder_mesh.yaml")

# pars = yaml2pyobject("lilytorch/src/scripts/configs/flow_past_cylinder_nondimensional.yaml")

# pars = yaml2pyobject("lilytorch/src/scripts/configs/moving_stick.yaml")

pars = yaml2pyobject("lilytorch/src/scripts/configs/moving_cylinder_analytical.yaml")


# modify parameters
# pars["body"]["control"]["f"]  = 10
# pars["output"]["save_frames"] = True
# pars["output"]["save_uv"]     = False
# pars["output"]["save_path"]   = "/data/andreaferrario/ns_data/"

solver = FluidSolver(pars, dtype=torch.float32, compute_forces=False)
solver.run_sim()










