

from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch

pars = yaml2pyobject("lilytorch/scripts/flow_past_cylinder.yaml")

solver = FluidSolver(pars, dtype=torch.float32, comute_forces=True)
solver.run_sim()


#plot drag
import numpy as np
import matplotlib.pyplot as plt
drag_forces = solver.drag_record.cpu().numpy()

time=np.arange(solver.nt)*solver.dt_np

U = pars["boundary_conditions"]["BC_values_u"][0]
D = 0.2
nu = pars["solver"]["nu"]
Re = U*D/nu
print("Reynolds number: ",Re)
# drag coefficient
drag_coeff = 2*drag_forces[0,0,:]/(pars["solver"]["rho"]*U**2*D)
convective_time = time*U/D


plt.figure(figsize=(8, 4))
plt.title("Re = %.2f" % Re)
plt.plot(convective_time,drag_forces[0,0,:])
plt.xlim([convective_time[0],convective_time[-1]])
plt.xlabel("Convective Time [t*U/D]")
# plt.ylim([0,1.6])
plt.ylabel("Drag Coefficient")
plt.savefig("drag_force_flow_past_cylinder.png")
plt.show()









