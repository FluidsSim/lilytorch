
import numpy as np
import matplotlib.pyplot as plt
import pyro.multigrid.variable_coeff_MG as MG
from pyro.mesh import patch
import pyro.mesh.boundary as bnd


N = 2**7
ks=0
kg=1
sigma=30
R=0.25

def alpha(x, y):
    d=np.sqrt((x-0.3)**2+y**2)-R
    return ks+(kg-ks)*(np.tanh(sigma*d)+1)/2
    # return np.ones((N+2,N+2))


def f(x, y):
    # return np.ones_like(x)
    return -2.0*((1.0-6.0*x**2)*y**2*(1.0-y**2) + (1.0-6.0*y**2)*x**2*(1.0-x**2))


g = patch.Grid2d(N, N, ng=1,xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
d = patch.CellCenterData2d(g)
bc_alpha = bnd.BC(xlb="neumann", xrb="neumann",
                  ylb="neumann", yrb="neumann")
d.register_var("alpha", bc_alpha)
d.create()
a = d.get_var("alpha")
a[:, :] = alpha(g.x2d, g.y2d)


rhs = f(g.x2d, g.y2d)

mg = MG.VarCoeffCCMG2d(N, N,
                       xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                       xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                       coeffs=a, coeffs_bc=bc_alpha,
                       verbose=1, vis=0)

mg.init_zeros()

rhs = f(mg.x2d, mg.y2d)
mg.init_RHS(rhs)

# solve
mg.solve(rtol=1.e-5)



# plotting
v = mg.get_solution()
fig, ax = plt.subplots(1,2)
im = ax[0].imshow(a.T,
              interpolation="none", origin="lower",
              extent=[mg.xmin, mg.xmax, mg.ymin, mg.ymax])
plt.colorbar(im, ax=ax[0])


im = ax[1].imshow(v.T,
              interpolation="nearest", origin="lower",
              extent=[mg.xmin, mg.xmax, mg.ymin, mg.ymax])
plt.colorbar(im, ax=ax[1])


plt.show()


