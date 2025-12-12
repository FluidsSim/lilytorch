
import torch
import scipy.sparse as sp


Nx=8
D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
D_1d = sp.lil_matrix(D_1d)
D_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)



D1_torch = torch.from_numpy(D_1d.todense()).to_sparse_csr()

# vec=torch.ones(2,Nx-1)
# vec[0,:]=-1

# D1_torch = torch.sparse.spdiags(vec, torch.tensor([-1, 1]), (Nx,Nx), layout=torch.sparse_csr)
# D1_torch[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
# D1_torch[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)

print(D1_torch)

print(D_1d.todense())
print(D1_torch.to_dense())

