

import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from ns_core.diff_matrices import Diff_mat_2D
from scipy.sparse.linalg import spsolve
from matplotlib import cm

Nx = 2**8
Ny = 2**8


x = np.linspace(0,1,Nx)
y = np.linspace(0,1,Ny)
X,Y = np.meshgrid(x,y) 
Xu = X.ravel() 
Yu = Y.ravel()

def index(X, val):
    return np.squeeze(np.where(X==val))

# f = np.zeros(Nx*Ny)
f=-np.exp(-(X-0.25)**2-(Y-0.6)**2)

B_ind = [
    index(Xu,x[0]),          # Left boundary
    index(Xu,x[Nx-1]),       # Right boundary
    index(Yu,y[0]),          # Bottom boundary
    index(Yu,y[Ny-1]),       # Top boundary
]
# B_type = [1,1,2,2]
B_type = [0,0,0,0]

N_B    = len(B_ind)

# Second partial derivatives
Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx,Ny)


I_sp = sp.eye(Nx*Ny).tocsr()


dx = x[1]-x[0]
dy = y[1]-y[0]

L_sys = D2x_2d/dx**2 + D2y_2d/dy**2     # system matrix without boundary conditions

# Boundary operators
BD = I_sp       # Dirichlet boundary operator
BNx = Dx_2d     # Neumann boundary operator for x component
BNy = Dy_2d     # Neumann boundary operator for y component

for m in range(N_B):
    if B_type[m] == 0:
        L_sys[B_ind[m],:] = BD[B_ind[m],:]
    if B_type[m] == 1:
        L_sys[B_ind[m],:] = BNx[B_ind[m],:]
    if B_type[m] == 2:
        L_sys[B_ind[m],:] = BNy[B_ind[m],:]

ind_unravel_L = np.squeeze(np.where(Xu==x[0]))          # Left boundary
ind_unravel_R = np.squeeze(np.where(Xu==x[Nx-1]))       # Right boundary
ind_unravel_B = np.squeeze(np.where(Yu==y[0]))          # Bottom boundary
ind_unravel_T = np.squeeze(np.where(Yu==y[Ny-1]))       # Top boundary

# Construction of right hand vector (function of x and y)
b = f[:].ravel()
b[ind_unravel_L] = 0
b[ind_unravel_R] = 0
b[ind_unravel_T] = 0
b[ind_unravel_B] = 0


# plt.figure()
# plt.imshow(L_sys.todense())


u = spsolve(L_sys,b).reshape(Ny,Nx)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, u, cmap=cm.coolwarm, 
                    linewidth=0, antialiased=False)

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# from IPython import embed; embed()


