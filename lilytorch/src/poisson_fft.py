
import torch
from scipy import special
from os.path import exists
import numpy
import torch_dct as dct
import os

class PoissonSolverFFT:
    """
    Solver class for the unbounded poisson equation
    """

    def __init__(self,
                x,
                y,
                bc_type = "free",
                overwrite = True,
                filename = "lilytorch/data/",
                ):
        """
        x : x-domain
        y : y-domain
        overwrite: if True compute the green function, otherwise searches for an already computed one (if found)
        """

        self.dtype = x.dtype
        if self.dtype == torch.float32:
            self.dtype_np = numpy.float32
        elif self.dtype == torch.float64:
            self.dtype_np = numpy.float64

        self.device = x.device

        self.bc_type = bc_type

        self.x = x.cpu().numpy()
        self.y = y.cpu().numpy()
        self.nx = len(x)
        self.ny = len(y)
        self.dx = (x[1]-x[0]).cpu().numpy()
        self.dy = (y[1]-y[0]).cpu().numpy()

        self.U = torch.zeros((2*self.nx,2*self.ny), dtype=self.dtype, device=self.device)

        self.name = "Gfft_"+bc_type+"_"+str(float(x[0]))+"_"+str(float(x[-1]))+str(float(y[0]))+"_"+str(float(y[-1]))+"_"+str(self.nx)+"_"+str(self.ny)

        if not os.path.exists(filename):
            os.makedirs(filename)

        self.save_filename = filename+self.name+".pt"
        # compute green function
        if exists(self.save_filename) and not overwrite:
            self.Gfft = torch.load(self.save_filename)
        else:
            self.Gfft = self.compute_solve_method()


    def compute_solve_method(self):

        if self.bc_type == "free":

            xshift = self.x-numpy.min(self.x)
            yshift = self.y-numpy.min(self.y)

            xshift_ext = numpy.concatenate((xshift,numpy.concatenate(([-xshift[-1]-self.dx],-xshift[::-1][:-1]))))
            yshift_ext = numpy.concatenate((yshift,numpy.concatenate(([-yshift[-1]-self.dx],-yshift[::-1][:-1]))))
            Xshift, Yshift = numpy.meshgrid(xshift_ext, yshift_ext, indexing='ij')

            R = numpy.sqrt(Xshift**2+Yshift**2)
            eps = 2*max(self.dx,self.dy)
            # G = torch.from_numpy(self.chatelain_fun(R, eps).astype(self.dtype_np)).to(self.device)
            G = torch.from_numpy(self.hej_fun(R, eps).astype(self.dtype_np)).to(self.device)
            Gfft = torch.fft.fftn(-self.dx*self.dy*G)

            torch.save(Gfft, self.save_filename)
            self.solve = self.solve_free_space
            return Gfft

        elif self.bc_type == "neumann":

            # kx = 1/(self.x[1]-self.x[0])
            # ky = 1/(self.y[1]-self.y[0])
            # K = torch.meshgrid(
            #     torch.pi*torch.linspace(-kx/2, kx/2, self.nx, dtype=self.dtype, device=self.device),
            #     torch.pi*torch.linspace(-ky/2, ky/2, self.ny, dtype=self.dtype, device=self.device),
            #     indexing='ij'
            # )

            # K = torch.meshgrid(
            #     torch.pi*torch.arange(-int(self.nx/2),int(self.nx/2), dtype=self.dtype, device=self.device)/(self.x[-1]-self.x[0]),
            #     torch.pi*torch.arange(-int(self.ny/2),int(self.ny/2), dtype=self.dtype, device=self.device)/(self.y[-1]-self.y[0]),
            #     indexing='ij'
            # )

            K = torch.meshgrid(
                torch.pi*torch.arange(self.nx, dtype=self.dtype, device=self.device)/(self.x[-1]-self.x[0]),
                torch.pi*torch.arange(self.ny, dtype=self.dtype, device=self.device)/(self.y[-1]-self.y[0]),
                indexing='ij'
            )
            Gfft = -1/(K[0]**2+K[1]**2)
            Gfft[0,0] = 0
            self.solve = self.solve_neumann
            return Gfft




    def solve_free_space(self,u):
        """
        Solve the inverse Fourier transform U_NEW=IFFT(FFT(U)*FFT(G))
        """
        self.U[:self.nx,:self.ny] = u
        return torch.real(
                torch.fft.ifftn( self.Gfft * torch.fft.fftn(self.U))
            )[:self.nx,:self.ny]


    def solve_neumann(self,u):
        """
        Homogeneous Neumann BC
        """
        out=dct.idct1( self.Gfft * dct.dct1(u))
        return out-out.mean()


    def chatelain_fun(self, r):
        """
        Green function singular approximation following Chatelain and Koumoutsakos, J Comp Phys, 2010
        """
        G = numpy.zeros((2*self.nx,2*self.ny))
        G = numpy.where(r>0,-(numpy.log(r))/(2*numpy.pi),0)
        req = numpy.sqrt(float(self.dx*self.dy)/numpy.pi)
        G[0,0] = -(req**2) * (numpy.log(req)-1/2)/2 # integral over spherical area equivalent to a rectangle of dimension [dx,dy]
        G[-1,-1] = G[0,0]
        G[0,-1] = G[0,0]
        G[-1,0] = G[0,0]
        return G


    def hej_fun(self, R, eps=0.01):
        """
        Green function approximation following Hejlesen et al, Applied Mathematics Letters, 2013
        """
        G = numpy.zeros((2*self.nx,2*self.ny))
        xeps=R/eps
        old = numpy.seterr(divide='ignore',invalid='ignore') # ignore invalid warnings
        logR = numpy.log(R)
        G = - (
            logR + \
            0.5*special.exp1(xeps**2/2) - \
            (
                137/120 - 163/240*xeps**2 + \
                137/960*xeps**4 - \
                7/640*xeps**6 + \
                1/3840*xeps**8
            )*numpy.exp(-xeps**2/2)
        )/(2*numpy.pi)
        singularity = (numpy.euler_gamma/2-numpy.log(numpy.sqrt(2)*eps) + 137/120)/(2*numpy.pi)
        G[0,0] = singularity
        numpy.seterr(**old) # reset warnings
        return G




if __name__ == "__main__":

    dtype=torch.float64
    device = "cuda"

    nx=512
    ny=256
    x=torch.linspace(0,5,nx,dtype=dtype, device=device)
    y=torch.linspace(-5,5,ny,dtype=dtype, device=device)
    ps = PoissonSolverFFT(x,y,overwrite=True)

    X,Y = torch.meshgrid(x,y, indexing='ij')
    Usmall = torch.zeros((nx,ny), dtype=torch.double)
    X0 = 2
    Y0 = 2
    XX = X-X0
    YY = Y-Y0
    RR2 = XX**2+YY**2
    c=6
    Usmall = 4*c*torch.exp(-c/(1-RR2)) * (c*RR2 + XX**4 + YY**4 + 2*(XX**2)*(YY**2) -1 ) * (-1 + RR2)**(-4)
    Usmall[RR2>=1] = 0

    sol = torch.exp(-c/(1-RR2))
    sol[RR2>=1] = 0

    ps.U[:ps.nx,:ps.ny] = Usmall # 0 padded extended initial function
    out_real = sol # exact solution in the smaller domain

    # compute approximate solution using the Green function
    out=ps.solve_free_space(Usmall)

    diff = torch.sqrt((out - out_real)**2)/torch.max(torch.max(abs(out_real)))

    # === compute errors between real and approximated solution ====
    print(torch.argmax(torch.abs(out-out_real)))
    print("The Linf error is "+str(torch.max(torch.max(abs(diff)))))
    print("The L2 error is "+str(torch.sqrt(torch.sum(torch.sum(diff**2))/(nx*ny))))


    # === plot solution ===
    import matplotlib.pylab as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X.cpu(), Y.cpu(), out_real.cpu())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.title("Solution real")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X.cpu(), Y.cpu(), out.cpu())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.title("Solution approx")

    plt.show()



