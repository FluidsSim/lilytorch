
import numpy as np
import matplotlib.pyplot as plt
import pyro.multigrid.variable_coeff_MG as MG
from pyro.mesh import patch
import pyro.mesh.boundary as bnd


N = 2**8
ks=100
kg=1
sigma=100
R=0.25

def true(x, y):
    circle=np.sqrt((x-0.5)**2+(y-0.5)**2)
    return np.where(
        circle<R,
        (1/8-R**2*(1-1/ks)/4)-(1/(4*ks))*(circle**2),
        1/8-circle**2/4
    )
    # return np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)

def alpha(x, y):
    d=np.sqrt((x-0.5)**2+(y-0.5)**2)-R
    return ks+(kg-ks)*(np.tanh(sigma*d)+1)/2

    #return 2.0 + np.cos(2.0*np.pi*x)*np.cos(2.0*np.pi*y)

def f(x, y):
    return np.ones_like(x)
    # return -16.0*np.pi**2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + 1) * \
    #     np.sin(2*np.pi*x)*np.sin(2*np.pi*y)


g = patch.Grid2d(N, N, ng=1,xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
d = patch.CellCenterData2d(g)
bc_alpha = bnd.BC(xlb="dirichlet", xrb="dirichlet",
                  ylb="dirichlet", yrb="dirichlet")
d.register_var("alpha", bc_alpha)
d.create()
a = d.get_var("alpha")
a[:, :] = alpha(g.x2d, g.y2d)


rhs = f(g.x2d, g.y2d)

mg = MG.VarCoeffCCMG2d(N, N,
                       xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                       xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                       coeffs=a, coeffs_bc=bc_alpha,
                       verbose=1, vis=0, true_function=true)

mg.init_zeros()

rhs = f(mg.x2d, mg.y2d)
mg.init_RHS(rhs)

# solve
mg.solve(rtol=1.e-5)

# plotting
v = mg.get_solution()
fig, ax = plt.subplots()

im = ax.imshow(np.transpose(v.v()),
              interpolation="nearest", origin="lower",
              extent=[mg.xmin, mg.xmax, mg.ymin, mg.ymax])
fig.colorbar(im, ax=ax)

plt.show()
