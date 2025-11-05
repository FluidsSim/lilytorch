
import jax
import numpy as np
from scipy.sparse.linalg import spsolve
from diff_matrices import Diff_mat_2D
import jax.experimental.sparse as jsparse


class Poisson2D_DIRECT:
    def __init__(self, Nx, Ny, x, y, device=None, dtype=None):
        self.Nx = Nx
        self.Ny = Ny
        x_np = x
        y_np = y
        self.device = device
        self.dtype = dtype

        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])

        X, Y = np.meshgrid(x_np, y_np, indexing='ij')
        Xu = X.ravel()
        Yu = Y.ravel()

        Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx,Ny)


        ind_unravel_L = np.squeeze(np.where(Xu==x[0])[0])
        ind_unravel_R = np.squeeze(np.where(Xu==x[Nx-1])[0])
        ind_unravel_B = np.squeeze(np.where(Yu==y[0])[0])
        ind_unravel_T = np.squeeze(np.where(Yu==y[Ny-1])[0])


        BNx = Dx_2d # Neumann boundary operator for x component
        BNy = Dy_2d # Neumann boundary operator for y component

        L_sys = D2x_2d/self.dx**2 + D2y_2d/self.dy**2
        L_sys[ind_unravel_T,:] = BNy[ind_unravel_T,:]
        L_sys[ind_unravel_B,:] = BNy[ind_unravel_B,:]
        L_sys[ind_unravel_L,:] = BNx[ind_unravel_L,:]
        L_sys[ind_unravel_R,:] = BNx[ind_unravel_R,:]

        L_sys_coo = L_sys.tocoo()

        L_jax = jsparse.BCOO.from_scipy_sparse(L_sys)

        from IPython import embed; embed()

        # self.L_sys_torch = np.sparse_csr_tensor(L_sys_coo.row, L_sys_coo.col, L_sys_coo.data, (self.Ny, self.Nx), dtype=self.dtype, device=self.device)


        jax.scipy.sparse.linalg.gmres(L_jax, jax.numpy.ones((Nx*Ny,), dtype=self.dtype))

        jax.scipy.sparse.linalg.bicgstab(L_jax, jax.numpy.ones((Nx*Ny,), dtype=self.dtype))


        jax.scipy.sparse.linalg.cg(L_jax, jax.numpy.ones((Nx*Ny,), dtype=self.dtype))

        # self.ind_unravel_L = torch.from_numpy(ind_unravel_L)
        # self.ind_unravel_R = torch.from_numpy(ind_unravel_R)
        # self.ind_unravel_T = torch.from_numpy(ind_unravel_T)
        # self.ind_unravel_B = torch.from_numpy(ind_unravel_B)



    def solve(self, b):
        b[self.ind_unravel_L] = 0
        b[self.ind_unravel_R] = 0
        b[self.ind_unravel_T] = 0
        b[self.ind_unravel_B] = 0
        return torch.sparse.spsolve(self.L_sys_torch,b).reshape(self.Ny,self.Nx)







if __name__ == "__main__":

    dtype = None
    device = "cpu"

    Nx = 80
    Ny = 180
    x = np.linspace(-6,6,Nx, dtype=dtype)
    y = np.linspace(-3,3,Ny, dtype=dtype)

    solver = Poisson2D_DIRECT(Nx, Ny, x, y, device=device, dtype=dtype)
    b = np.ones(Nx*Ny, dtype=dtype)

    b_jax= jax.numpy.array(b)

    u = solver.solve(b)

    # === plot solution ===
    import matplotlib.pylab as plt

    plt.figure(figsize=(8,5))
    cp = plt.contourf(solver.X, solver.Y, u.T, levels=50, cmap='viridis')
    plt.colorbar(cp, label='u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Poisson2D_DIRECT solution')
    plt.axis('equal')
    plt.tight_layout()

    A = solver.L_sys
    print("L_sys shape:", A.shape)
    print("nonzeros:", A.nnz if hasattr(A, 'nnz') else np.count_nonzero(A))

    plt.figure(figsize=(6,6))
    plt.spy(A, markersize=3)
    plt.title('spy(L_sys)')
    plt.tight_layout()

    plt.show()
