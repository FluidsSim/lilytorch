import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Setup
x_min = -np.pi # Left endpoint of x interval
x_max = +np.pi # Right endpoint of x interval
y_min = 0      # Left endpoint of y interval
y_max = 2      # Right endpoint of y interval

nx = 31                                    # Number of grid points in x
ny = 21                                    # Number of grid points in y
hx = (x_max - x_min)/(nx-1)                # Spacing in x direction
hy = (y_max - y_min)/(ny-1)                # Spacing in y direction
n  = nx*ny                                 # Dimension of system

A = np.zeros((n, n), dtype='float')
b = np.zeros(n, dtype='float')

# Define Lambda functions
index = lambda row, col: row*ny+col
f = lambda x: np.cos(x) if np.abs(x)<=np.pi/2 else 0

# Index of equation
row = 0

# Set 2D Laplace operator
for i in range(1, nx-1):
     for j in range(1, ny-1):
         A[row, index(i-1, j)] += 1/hx**2
         A[row, index(i, j)]   -= 2/hx**2+2/hy**2
         A[row, index(i+1, j)] += 1/hx**2
         A[row, index(i, j-1)] += 1/hy**2
         A[row, index(i, j+1)] += 1/hy**2
         b[row] = -f(x_min+hx*(i-1))
         row += 1

# Set Neumann boundary condition
for i in range(1, nx-1):
    if row >= n: break
    A[row, index(i, ny-1)] += 3 #/(2*hy)
    A[row, index(i, ny-2)] -= 4 #/(2*hy)
    A[row, index(i, ny-3)] += 1 #/(2*hy)

    A[row, index(i, 0)] += 3 #/(2*hy)
    A[row, index(i, 1)] -= 4 #/(2*hy)
    A[row, index(i, 2)] += 1 #/(2*hy)

    b[row] = 0
    row += 1

# Set Dirichlet boundary conditions
for i in range(0, nx):
    for j in range(0, ny):
        if i<=0 or j<=0 or i>=nx-1:
            if row >= n: break
            A[row, index(i, j)] = 1
            b[row] = 0
            row += 1

if row != n:
     raise Exception('Wrong number of equations')

plt.spy(A)

rank = np.linalg.matrix_rank(A)
print('rank=',rank,'row=', row)

if rank < row:
    raise Exception('Insufficient rank.')

# Solve the linear system
u = np.linalg.solve(A, b)
# np.savetxt('uuu.csv', u, delimiter=',')

error = np.matmul(A,u) - b
norm = np.linalg.norm(error)
print('error norm=', norm)

if norm > 1E-9:
    raise Exception('Error norm')

# Prepare visualization
x = np.linspace(x_min, x_max, nx)       # Grid points in x-direction
y = np.linspace(y_min, y_max, ny)       # Grid points in y-direction
xg, yg = np.meshgrid(x, y)
uu = u.reshape(nx, ny).transpose()

# Plot the solution
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xg, yg, uu)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u');
ax.azim = 45
plt.show()