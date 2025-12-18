import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from tqdm.notebook import tqdm

import sys
sys.path.insert(0, 'solving_pde_mooc/notebooks/modules')
# Function to compute an error in L2 norm
from norms import l2_diff

def rhs_2d(x,y):
    return np.sin(np.pi*x)*np.cos(np.pi*y)+np.sin(5*np.pi*x)*np.cos(5*np.pi*y)

def p_exact_2d(x,y):
    return -(1/(2*np.pi*np.pi))*np.sin(np.pi*x)*np.cos(np.pi*y)-(1/(50*np.pi*np.pi))*np.sin(5*np.pi*x)*np.cos(5*np.pi*y)

plt.style.use('solving_pde_mooc/notebooks/styles/mainstyle.use')

# Grid parameters.
nx = 1001                  # number of points in the x direction
ny = 1001                  # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx - 1)        # grid spacing in the x direction
dy = ly / (ny - 1)        # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Create the source term.
b = rhs_2d(X, Y)

# Compute the exact solution.
p_exact = p_exact_2d(X, Y)


def A(v, dx, dy):
    """
    Computes the action of (-) the Poisson operator on any
    vector v_{ij} for the interior grid nodes
    
    Parameters
    ----------
    v : numpy.ndarray
        input vector
    dx : float
         grid spacing in the x direction
    dy : float
        grid spacing in the y direction
        

    Returns
    -------
    Av : numpy.ndarray
        action of A on v
    """
    
    Av = -((v[:-2, 1:-1]-2.0*v[1:-1, 1:-1]+v[2:, 1:-1])/dx**2 
       + (v[1:-1, :-2]-2.0*v[1:-1,1:-1]+v[1:-1, 2:])/dy**2)
    
    return Av



# Initial guess
p0 = np.zeros((nx, ny))

# Place holders for the residual r and A(d)
r = np.zeros((nx, ny))
Ad = np.zeros((nx, ny))

tolerance = 1e-15
max_it = 100

pbar = tqdm(total=max_it)
pbar.set_description("iter / max_it");

it = 0 # iteration counter
diff = 1.0
tol_hist_jac = []

p = p0.copy()

# Initial residual r0 and initial search direction d0
r[1:-1, 1:-1] = -b[1:-1, 1:-1] - A(p, dx, dy)
d = r.copy()

while (diff > tolerance):
    if it > max_it:
        print('\nSolution did not converged within the maximum'
              ' number of iterations'
              f'\nLast l2_diff was: {diff:.5e}')
        break

    # Laplacian of the search direction.
    Ad[1:-1, 1:-1] = A(d, dx, dy)
    # Magnitude of jump.

    alpha = np.sum(r*r) / np.sum(d*Ad)
    # Iterated solution
    pnew = p + alpha*d
    # Intermediate computation
    beta_denom = np.sum(r*r)
    # Update the residual.
    r = r - alpha*Ad
    # Compute beta
    beta = np.sum(r*r) / beta_denom
    # Update the search direction.
    d = r + beta*d
    
    diff = l2_diff(pnew, p)
    tol_hist_jac.append(diff)
    
    # Get ready for next iteration
    it += 1
    np.copyto(p, pnew)
    
    # We update our progress bar
    pbar.update(1)

else:
    print(f'\nThe solution converged after {it} iterations')

# When the progress bar will not be used
# further, it has to be closed
del(pbar)






fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))
# We shall now use the
# matplotlib.pyplot.contourf function.
# As X and Y, we pass the mesh data.
#
# For more info
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html
#
CS1=ax_1.contourf(X, Y, p_exact, 20)
CS2=ax_2.contourf(X, Y, pnew, 20)
cbar = fig.colorbar(CS1)
cbar = fig.colorbar(CS2)

# plot along the line y=0:
jc = int(ly/(2*dy))
ax_3.plot(x, p_exact[:,jc], '*', color='red', markevery=2, label=r'$p_e$')
ax_3.plot(x, pnew[:,jc], label=r'$pnew$')

# add some labels and titles
ax_1.set_xlabel(r'$x$')
ax_1.set_ylabel(r'$y$')
ax_1.set_title('Exact solution')

ax_2.set_xlabel(r'$x$')
ax_2.set_ylabel(r'$y$')
ax_2.set_title('Numerical solution')

ax_3.set_xlabel(r'$x$')
ax_3.set_ylabel(r'$p$')
ax_3.set_title(r'$p(x,0)$')

ax_3.legend()

plt.show()