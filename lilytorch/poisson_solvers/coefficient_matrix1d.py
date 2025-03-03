



import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy as sp


N=2**9+1
x=np.linspace(0,1,N)
h=x[1]-x[0]
c=np.ones_like(x) #np.round(np.exp(-(x-0.25)**2),2)

c_hat = (c[1:]+c[:-1])/2
c_sum = c_hat[1:]+c_hat[:-1]

A = np.zeros((N,N))

A[1:-1,1:-1] = np.diag(c_sum, k=0)-np.diag(c_hat[1:-1], k=-1)-np.diag(c_hat[1:-1], k=1)



A = (
    np.diag(np.concatenate([[1],c_sum,[1]]), k=0)
    -np.diag(np.concatenate([[0],c_hat[1:-1],[0]]), k=-1)
    -np.diag(np.concatenate([[0],c_hat[1:-1],[0]]), k=1)
)


print(A)




data = np.array([
    np.concatenate([[1],c_sum,[1]]),
    -np.concatenate([[0],c_hat[1:-1],[0],[0]]),
    -np.concatenate([[0],[0],c_hat[1:-1],[0]])
    ])
diags = np.array([
    0,
    -1,
    1
    ])
print(sp.sparse.spdiags(data, diags, N, N).todense())


# equivalent to:
Asp = ( 
    sp.sparse.spdiags([np.concatenate([[1],c_sum,[1]])], [0], N, N)
    +sp.sparse.spdiags([-np.concatenate([[0],c_hat[1:-1],[0]])], [-1], N, N)
    +sp.sparse.spdiags([-np.concatenate([[0],[0],c_hat[1:-1],[0]])], [1], N, N)
)
print(Asp.todense())

tone = torch.tensor([1.])
tzero = torch.tensor([0.])
data = torch.cat([
    torch.unsqueeze(torch.concatenate([tone,torch.from_numpy(c_sum),tone]), dim=0), # Diriclet BC
    torch.unsqueeze(-torch.concatenate([tzero,torch.from_numpy(c_hat[1:-1]),tzero,tzero]), dim=0), 
    torch.unsqueeze(-torch.concatenate([tzero,tzero,torch.from_numpy(c_hat[1:-1]),tzero]), dim=0)
    ], axis=0)
diags = torch.tensor([
    0,
    -1,
    1
    ])
Atorch = torch.sparse.spdiags(data, diags, (N, N))
print(Atorch.to_dense())

f=torch.exp(-(torch.from_numpy(x)-0.25)**2)

u=torch.linalg.solve(Atorch.to_dense(),f).reshape(N)

plt.plot(x, u)
plt.show()