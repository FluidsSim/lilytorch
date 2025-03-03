import numpy as np
import matplotlib.pyplot as plt
from pyro.mesh import patch
import pyro.multigrid.MG as MG

N = 256

mg = MG.CellCenterMG2d(N, N,
                       xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                       yl_BC_type="dirichlet", yr_BC_type="dirichlet", verbose=1)

def rhs(x, y):
    return x-x+1
    # return -2.0 * ((1.0 - 6.0 * x**2) * y**2 * (1.0 - y**2) +
    #                (1.0 - 6.0 * y**2) * x**2 * (1.0 - x**2))

mg.init_RHS(rhs(mg.x2d, mg.y2d))

mg.init_zeros()

mg.solve(rtol=1.e-11)
phi = mg.get_solution()
fig, ax = plt.subplots()
plt.imshow(
    np.transpose(phi.v()), 
    origin="lower",
    vmin=-0.1, 
    vmax=0,
    # extent=self.extent
    )
plt.colorbar()

# ax.contourf(mg.x2d, mg.y2d, phi.v(),10)
# plt.colorbar()



plt.show()









