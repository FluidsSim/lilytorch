

from petsc4py import PETSc
import torch

device = torch.device("cpu")
a_tensor = torch.tensor(([1., 2.], [3.,4.]), dtype=torch.float64, device=device)
a_vec = PETSc.Vec().createWithDLPack(a_tensor) #use a_tensor.detach().c
a_tensor[1][1] = -4.
a_vec.view()
print(a_tensor)


import torch.utils.dlpack as dlpack
b_vec = PETSc.Vec().createWithArray([5.,6.,7.,8.])
b_vec.attachDLPackInfo(a_vec)
b_tensor = dlpack.from_dlpack(b_vec)
print(b_tensor)

