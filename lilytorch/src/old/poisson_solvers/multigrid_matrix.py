
import torch
import numpy
from ns_core.diff_matrices import Diff_mat_2D
import scipy.sparse as sp

class multigrid_matrix:
    """
    Multigrid solver
    """    
    def __init__(self, x, y):
        
        self.h2 = (x[1]-x[0])**2
        self.w=0.6 # weight by damping factor

        self.p1=30
        self.p2=30
        self.x=30

        self.x = x.cpu().numpy()
        self.y = y.cpu().numpy()
        self.Nx = len(x)
        self.Ny = len(y)

        X,Y = numpy.meshgrid(x,y) 
        self.Xu = X.ravel() 
        self.Yu = Y.ravel()

        L_sys = self.generate_matrix(self.x[1]-self.x[0], self.y[1]-self.y[0])

        self.A = torch.from_numpy(L_sys.todense()).to_sparse_csr()

    def generate_matrix(self, dx, dy):
        


        def index(X, val):
            return numpy.squeeze(numpy.where(X==val))

        B_ind = [
            index(self.Xu,x[0]),          # Left boundary
            index(self.Xu,x[self.Nx-1]),  # Right boundary
            index(self.Yu,y[0]),          # Bottom boundary
            index(self.Yu,y[self.Ny-1]),  # Top boundary
        ]
        # B_type = [1,1,2,2]
        B_type = [0,0,0,0]

        N_B    = len(B_ind)

        # Second partial derivatives
        Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(self.Nx,self.Ny)

        I_sp = sp.eye(self.Nx*self.Ny).tocsr()


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
        return L_sys
    
    def impose_rhs_bc(self, b):

        ind_unravel_L = numpy.squeeze(numpy.where(self.Xu==self.x[0]))          # Left boundary
        ind_unravel_R = numpy.squeeze(numpy.where(self.Xu==self.x[self.Nx-1]))  # Right boundary
        ind_unravel_B = numpy.squeeze(numpy.where(self.Yu==self.y[0]))          # Bottom boundary
        ind_unravel_T = numpy.squeeze(numpy.where(self.Yu==self.y[self.Ny-1]))  # Top boundary

        b[ind_unravel_L] = 0
        b[ind_unravel_R] = 0
        b[ind_unravel_T] = 0
        b[ind_unravel_B] = 0

    def general_relaxation(self, m):
        mnew = torch.zeros_like(m)
        mnew[1:-1,1:-1] = m[2:,1:-1]+m[:-2,1:-1]+m[1:-1,2:]+m[1:-1,:-2]
        return mnew

    def jacobi_relaxation(self, v, c, f):
        v_new = (self.general_relaxation(v)+self.h2*f)/4
        # v_new = (self.general_relaxation(c*v)+self.h2*f)/(4*c)
        return self.w*v_new+(1-self.w)*v

    def compute_residual(self, v, c, f):
        return f-(self.general_relaxation(v)-4*v)/self.h2
        # return f-(4*v*c-self.general_relaxation(c*v))/self.h2

    def prolong(self, e2h):
        """
        prolong e2h in eh
        """
        n, m = e2h.shape

        nh = int(2*(n-1))
        mh = int(2*(m-1))

        eh = torch.zeros(nh+1,mh+1)

        i1 = 2*numpy.arange(1,int(nh/2))
        i2 = 2*numpy.arange(0,int(nh/2))+1
        j1 = 2*numpy.arange(1,int(mh/2))
        j2 = 2*numpy.arange(0,int(mh/2))+1

        i1j1 = numpy.ix_(i1,j1)
        i2j1 = numpy.ix_(i2,j1)
        i1j2 = numpy.ix_(i1,j2)
        i2j2 = numpy.ix_(i2,j2)

        eh[i1j1] = e2h[1:-1,1:-1]
        eh[i2j1] = 0.5*(e2h[1:,1:-1]+e2h[:-1,1:-1])
        eh[i1j2] = 0.5*(e2h[1:-1,1:]+e2h[1:-1,:-1])
        eh[i2j2] = 0.25*(e2h[1:,1:]+e2h[1:,:-1]+e2h[:-1,1:]+e2h[:-1,:-1])

        return eh

    def restrict(self, r):
        """
        restrict residual from rh to r2h
        """
        n = r.shape[0]-1
        m = r.shape[1]-1

        n2h = int(n/2+1)
        m2h = int(m/2+1)
        r2h = torch.zeros(n2h,m2h)
        
        i  = numpy.arange(1,int(n/2))
        i1 = 2*i-1
        i2 = 2*i
        i3 = 2*i+1

        j  = numpy.arange(1,int(m/2))
        j1 = 2*j-1
        j2 = 2*j
        j3 = 2*j+1

        i1j1 = numpy.ix_(i1,j1)
        i1j3 = numpy.ix_(i1,j3)
        i3j1 = numpy.ix_(i3,j1)
        i3j3 = numpy.ix_(i3,j3)

        i2j1 = numpy.ix_(i2,j1)
        i2j3 = numpy.ix_(i2,j3)
        i1j2 = numpy.ix_(i1,j2)
        i3j2 = numpy.ix_(i3,j2)

        i2j2 = numpy.ix_(i2,j2)

        r2h[1:-1, 1:-1] = 0.0625*(r[i1j1]+r[i1j3]+r[i3j1]+r[i3j3])+\
                        0.125*(r[i2j1]+r[i2j3]+r[i1j2]+r[i3j2])+\
                        0.25*r[i2j2]

        return r2h



    def vcycle(self, v, c, f):
        
        # step 0 ============ Jacobi relaxation ============
        npoints_local = int(v.shape[0]-1)
        
        if (npoints_local == 2):
            for _ in range(self.x):
                v_new = self.jacobi_relaxation(v, c, f)
                v     = v_new
            return v

        else:
            # step 1 ============ Jacobi relaxation ============ 
            for _ in range(self.p1):
                v_new = self.jacobi_relaxation(v, c, f)
                v     = v_new

            # step 2 ============ Compute residual and restrict it to 2h ============ 
            r   = self.compute_residual(v, c, f)
            r2h = self.restrict(r)

            print("Steps: {}, Residual: {}".format(npoints_local, 1))

            # step 3 ============ Residual relaxation ============ 
            c2h = self.restrict(c) # compute coefficient matrix on the coarse grid
            e2h = torch.zeros_like(c2h) # initialize error to zero
            self.h2=self.h2*4

            # from IPython import embed; embed()
            e2h_new = self.vcycle(e2h, c2h, r2h) # relax to find the error of the residuals
            
            # step 4 ============ Prolongate error and correct v ============ 
            # print(torch.max(self.prolong(e2h_new)))
            v += self.prolong(e2h_new)

            # # step 5 ============ Relax more times ============ 
            # self.h2=self.h2/4
            # for _ in range(self.p1):
            #     v_new = self.jacobi_relaxation(v, c, f)
            #     v     = v_new
            return v


def test_restriction_prolongation():

    solver = multigrid_matrix(0.1)
    # original = torch.rand((2**2+1,2**2+1))
    original = torch.tensor(
        [
            [0,0,0,0,0],
            [0,1,2,3,0],
            [0,4,5,6,0],
            [0,7,8,9,0],
            [0,0,0,0,0]
        ]
    )
    print(original)
    restricted = solver.restrict(original)
    print(restricted)
    prolonged = solver.prolong(restricted)
    print(prolonged)


if __name__ == "__main__":



    # v=torch.rand((npoints,npoints))
    # f=torch.rand((npoints,npoints))
    # c=torch.rand((npoints,npoints))
    npoints=2**7+1
    x=torch.linspace(0,1,npoints)
    y=torch.linspace(0,1,npoints)
    h=x[1]-x[0]
    [X,Y]=torch.meshgrid(x,y, indexing="xy")
    c=torch.ones((npoints,npoints))
    f=-torch.exp(-(X-0.25)**2-(Y-0.6)**2)
    v=torch.rand((npoints,npoints))

    solver = multigrid_matrix(x, y)
    b = f[:].ravel()
    solver.impose_rhs_bc(b)

    Z = torch.linalg.solve(solver.A.to_dense(),b).reshape(solver.Ny,solver.Nx)
    
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



