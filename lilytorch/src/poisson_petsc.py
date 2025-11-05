


from mpi4py import MPI
from petsc4py import PETSc
import torch

nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)

class PoissonPETSc:

    def __init__(self, nx, ny, x, y, device=None, dtype=None):

        self.nx = nx
        self.ny = ny
        self.device = device
        self.dtype = dtype

        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])

        print(self.dx, self.dy)
        assert (self.dx-self.dy)<1e-10, "Currently only square grids are supported for PETSc Poisson solver."
        self.h = self.dx

        # ==================== create Poisson matrix ====================
        self.A = PETSc.Mat()
        self.A.create(comm=PETSc.COMM_WORLD)
        self.A.setSizes((self.nx * self.ny, self.nx * self.ny))
        self.A.setType(PETSc.Mat.Type.AIJ)
        self.A.setFromOptions()
        self.A.setPreallocationNNZ(5)
        def index(i,j):
            return i * self.ny + j

        rstart, rend = self.A.getOwnershipRange()
        Neumann = True
        for i in range(self.nx):
            for j in range(self.ny):
                row = index(i,j)
                self.A.setValue(row, row, 4.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                if i > 0:
                    column = index(i-1,j)
                    self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    if Neumann:
                        column = index(i+1,j)
                        self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                if i < self.nx-1:
                    column = index(i+1,j)
                    self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    if Neumann:
                        column = index(i-1,j)
                        self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                if j > 0:
                    column = index(i,j-1)
                    self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    if Neumann:
                        column = index(i,j+1)
                        self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                if j < self.ny-1:
                    column = index(i,j+1)
                    self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    if Neumann:
                        column = index(i,j-1)
                        self.A.setValue(row, column, -1.0/self.h**2, addv=PETSc.InsertMode.ADD_VALUES)

        self.A.assemblyBegin()
        self.A.assemblyEnd()
        self.A.setNullSpace(nullspace)

        opts = PETSc.Options()

        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.A.getComm())
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.getPC().setType(PETSc.PC.Type.GAMG)
        self.ksp.setFromOptions()


        # petsc_options = {
        #     "ksp_error_if_not_converged": True,
        #     "ksp_type": "preonly",
        #     "pc_type": "lu",
        #     "pc_factor_mat_solver_type": "mumps",
        #     "ksp_monitor": None,
        # }
        # self.ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        # self.ksp.setOptionsPrefix("singular_direct")
        # opts.prefixPush(self.ksp.getOptionsPrefix())
        # for key, value in petsc_options.items():
        #     opts[key] = value
        # self.ksp.setFromOptions()
        # for key, value in petsc_options.items():
        #     del opts[key]
        # opts.prefixPop()


        # self.ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        # self.ksp.setOptionsPrefix("singular_iterative")
        # petsc_options_iterative = {
        #     "ksp_error_if_not_converged": True,
        #     "ksp_monitor": None,
        #     "ksp_type": "gmres",
        #     "pc_type": "hypre",
        #     "pc_hypre_type": "boomeramg",
        #     "pc_hypre_boomeramg_max_iter": 1,
        #     "pc_hypre_boomeramg_cycle_type": "v",
        #     "ksp_rtol": 1.0e-13,
        # }
        # opts.prefixPush(self.ksp.getOptionsPrefix())
        # for key, value in petsc_options_iterative.items():
        #     opts[key] = value
        # # self.ksp.setFromOptions()
        # for key, value in petsc_options_iterative.items():
        #     del opts[key]
        # opts.prefixPop()


        self.ksp.setOperators(self.A)


    def solve(self, f):
        f[0,:]=0.0
        f[-1,:]=0.0
        f[:,0]=0.0
        f[:,-1]=0.0
        sol, b = self.A.createVecs()
        b = PETSc.Vec().createWithArray(f.flatten())
        self.ksp.solve(b, sol)
        return sol.getArray()




if __name__ == "__main__":

    dtype = torch.float32
    device = "cpu"

    Nx = 302
    Ny = 102

    xmin = 0.0
    xmax = 3.0
    ymin = 0.0
    ymax = 1.0

    dx=(xmax-xmin)/(Nx-2)
    dy=(ymax-ymin)/(Ny-2)

    x = torch.arange(xmin-dx/2, xmax+dx, dx, dtype=dtype, device=device)
    y = torch.arange(ymin-dy/2, ymax+dy, dy, dtype=dtype, device=device)

    solver = PoissonPETSc(Nx, Ny, x, y, device=device, dtype=dtype)


    # f= torch.ones((Nx,Ny), dtype=dtype)
    f = torch.cos(2 * torch.pi * x[:, None]) * torch.ones((1, Ny), dtype=dtype, device=device)

    sol=solver.solve(f)

    # Compute exact solution with zero mean
    B = 0.0
    u_exact = torch.cos(2 * torch.pi * x[:, None]) / (4 * torch.pi ** 2)
    mean_u_exact = torch.mean(u_exact)
    u_exact = u_exact - mean_u_exact  # ensure zero mean


    import matplotlib.pyplot as plt

    # Plot both numerical and exact solutions side by side
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 5))

    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = sol.reshape(Nx, Ny)


    # Numerical solution
    axs[0].plot_surface(X, Y, Z, cmap='viridis')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_zlabel('Numerical Solution')
    axs[0].set_title('Poisson Equation Solution')

    # Exact solution
    axs[1].plot_surface(X, Y, u_exact, cmap='viridis')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_zlabel('Exact Solution')
    axs[1].set_title('Exact Solution (Zero Mean)')

    plt.tight_layout()
    plt.show()

