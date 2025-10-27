from petsc4py import PETSc
import numpy as np

# Create a 2D DMDA (structured grid)
nx, ny = 32, 32
da = PETSc.DMDA().create([nx, ny], dof=1,
                          stencil_width=1,
                          boundary_type=(PETSc.DM.BoundaryType.MIRROR,
                                         PETSc.DM.BoundaryType.MIRROR))

A = da.createMatrix()
b = da.createGlobalVec()
x = da.createGlobalVec()

hx = 1.0 / (nx - 1)
hy = 1.0 / (ny - 1)
hx2 = hx * hx
hy2 = hy * hy

# Assemble the matrix and RHS
rows, cols = da.getRanges()

# Fill A and b manually
for j in range(rows[0], rows[1]):
    for i in range(cols[0], cols[1]):
        row = da.getGlobalIndices([i, j])[0]
        v_center = 0.0
        entries = []
        cols_ = []

        # f(x, y)
        xcoord, ycoord = i * hx, j * hy
        f_val = 1.0  # right-hand side

        # Center
        v_center = -2.0 * (1.0/hx2 + 1.0/hy2)

        # Neighbors
        for di, dj, coeff in [(-1,0,1.0/hx2), (1,0,1.0/hx2), (0,-1,1.0/hy2), (0,1,1.0/hy2)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                entries.append(coeff)
                cols_.append(da.getGlobalIndices([ni, nj])[0])
            else:
                # Neumann BC: du/dn = 0  → mirror the interior value
                v_center += coeff  # modifies the diagonal (acts as zero flux)

        entries.append(v_center)
        cols_.append(row)

        A.setValues([row], cols_, entries)
        b.setValue(row, f_val)

A.assemble()
b.assemble()

# Set up KSP solver
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
ksp.getPC().setType('hypre')  # or 'jacobi' / 'sor' etc.
ksp.setFromOptions()

# Solve
ksp.solve(b, x)

# Get solution as NumPy array
x_local = da.getVecArray(x)
u = np.zeros((ny, nx))
for j in range(ny):
    for i in range(nx):
        u[j, i] = x_local[i, j]

# Shift mean to zero since Neumann BC makes solution unique up to a constant
u -= np.mean(u)

# Print residual and info
print("Converged in", ksp.getIterationNumber(), "iterations.")
print("Final residual norm:", ksp.getResidualNorm())

# Optionally visualize
import matplotlib.pyplot as plt
plt.imshow(u, origin='lower', extent=(0,1,0,1))
plt.colorbar(label='u(x,y)')
plt.title("Poisson equation with Neumann BCs")
plt.show()