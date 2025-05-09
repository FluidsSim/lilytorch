
import torch
from scipy import special
from os.path import exists
import numpy

class PoissonSolverFFT:
    """
    Solver class for the unbounded poisson equation
    """

    def __init__(self,
                x,
                y,
                bc_type = "free-space",
                free_space_method = "hej",
                overwrite = True,
                filename = "lilytorch/data/",
                ):
        """
        x : x-domain
        y : y-domain
        overwrite: if True compute the green function, otherwise searches for an already computed one (if found)
        """

        self.dtype = x.dtype
        self.device = x.device

        self.bc_type = bc_type
        self.free_space_method = free_space_method

        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]

        self.U = torch.zeros((2*self.ny,2*self.nx), dtype=self.dtype, device=self.device)

        self.name = "Gfft_"+bc_type+"_"+str(float(x[0]))+"_"+str(float(x[-1]))+str(float(y[0]))+"_"+str(float(y[-1]))+"_"+str(self.nx)+"_"+str(self.ny)

        self.save_filename = filename+self.name+".pt"
        # compute green function
        if exists(self.save_filename) and not overwrite:
            self.Gfft = torch.load(self.save_filename)
        else:
            self.Gfft = self.compute_solve_method()


    def compute_solve_method(self):

        if self.bc_type == "free-space":

            # using Hejlesen (default) or Chatelain method for approximating the Green function
            xshift = self.x-self.x.min()
            yshift = self.y-self.y.min()

            # using Hockney-Eastwood padding method
            xshift_ext = torch.hstack((xshift, torch.hstack((torch.tensor([-xshift[-1]-self.dx],dtype=self.dtype,device=self.device),-torch.flip(xshift,dims=(0,))[:-1]))))
            yshift_ext = torch.hstack((yshift, torch.hstack((torch.tensor([-yshift[-1]-self.dy],dtype=self.dtype,device=self.device),-torch.flip(yshift,dims=(0,))[:-1]))))

            Xshift, Yshift = torch.meshgrid(xshift_ext, yshift_ext, indexing='ij')

            R = torch.sqrt(Xshift**2+Yshift**2).cpu().numpy()

            if self.free_space_method == "hej":
                eps = max(float(self.dx),float(self.dy))*2
                G = torch.from_numpy(self.hej_fun(R, eps=eps)).to(self.device)
            elif self.free_space_method == "chat":
                G = torch.from_numpy(self.chatelain_fun(R)).to(self.device)
            else:
                raise("Wrong method to compute the Green function!!")
            Gfft = torch.fft.fftn(-self.dx*self.dy*G)
            torch.save(Gfft, self.save_filename)
            self.solve = self.solve_free_space
            return Gfft


    def solve_free_space(self,u):
        """
        Solve the inverse Fourier transform U_NEW=IFFT(FFT(U)*FFT(G))
        """
        self.U[:self.ny,:self.nx] = u
        return torch.real(
                torch.fft.ifftn( self.Gfft * torch.fft.fftn(self.U))
            )[:self.ny,:self.nx]

    def chatelain_fun(self, x):
        """
        Green function singular approximation following Chatelain and Koumoutsakos, J Comp Phys, 2010
        """
        G = numpy.zeros((2*self.nx,2*self.ny))
        G = -(numpy.log(x))/(2*numpy.pi)
        print("Python will give an error of 0 division. Ignore it.")
        req = numpy.sqrt(float(self.dx*self.dy)/numpy.pi)
        G[0,0] = -(req**2) * (numpy.log(req)-1/2)/2 # integral over spherical area equivalent to a rectangle of dimension [dx,dy]
        G[-1,-1] = G[0,0]
        G[0,-1] = G[0,0]
        G[-1,0] = G[0,0]
        return G

    def hej_fun(self, x, eps=0.01):
        """
        Green function approximation following Hejlesen et al, Applied Mathematics Letters, 2013
        """
        G = numpy.zeros((2*self.nx,2*self.ny))
        xeps=x/eps
        G = -(
                numpy.log(x) + \
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
        G[-1,0] = singularity
        G[0,-1] = singularity
        G[-1,-1] = singularity
        return G




if __name__ == "__main__":

    dtype=torch.float32
    device = "cuda"

    nx=1360
    ny=1360
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
    out=ps.solve(Usmall)

    diff = torch.sqrt((out - out_real)**2)/torch.max(torch.max(abs(out_real)))

    # === compute errors between real and approximated solution ====
    print(torch.argmax(torch.abs(out-out_real)))
    print("The Linf error is "+str(torch.max(torch.max(abs(diff)))))
    print("The L2 error is "+str(torch.sqrt(torch.sum(torch.sum(diff**2))/(nx*ny))))


    # === plot solution ===
    import matplotlib.pylab as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X.cpu(), Y.cpu(), out_real.cpu().T)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.title("Solution real")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X.cpu(), Y.cpu(), out.cpu().T)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.title("Solution approx")

    plt.show()



