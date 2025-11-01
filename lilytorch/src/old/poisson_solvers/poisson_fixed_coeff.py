
import numpy
from scipy import special
from os.path import exists
import torch

class PoissonSolver:
    """
    Solver class for the unbounded poisson equation
    lapl(phi) = f
    """

    def __init__(self, 
                 x,
                 y,
                 method="gpu",
                 green_method = "hej",
                 overwrite = True
                 ):
        """
        x : x-domain
        y : y-domain
        method : gpu or cpu
        green_method: using Hejlesen or Chatelain method for approximating the Green function
        overwrite: if True compute the green function, otherwise searches for an already computed one (if found)
        """

        self.method = method
        self.green_method = green_method


        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]

        self.U = torch.zeros((2*self.ny,2*self.nx))

        self.name = "Gfft_"+green_method+"_"+str(x[0])+"_"+str(x[-1])+str(y[0])+"_"+str(y[-1])+"_"+str(self.nx)+"_"+str(self.ny)

        filename = "data/"+self.name+".npy"
        # compute green function
        if exists(filename) and not overwrite:
            self.Gfft = torch.from_numpy(torch.load(filename))
        else: 
            xshift = self.x-torch.min(self.x)
            yshift = self.y-torch.min(self.y)
            xshift_ext = torch.concatenate((
                xshift,
                torch.concatenate((
                    torch.tensor([-xshift[-1]-self.dx]),
                    torch.flip(-xshift,dims=(0,))[:-1]
                    ))
                ))
            yshift_ext = torch.concatenate((
                yshift,
                torch.concatenate((
                    torch.tensor([-yshift[-1]-self.dy]),
                    torch.flip(-yshift,dims=(0,))[:-1]
                    ))
                ))
            Xshift, Yshift = torch.meshgrid(xshift_ext, yshift_ext)
            R = torch.sqrt(Xshift**2+Yshift**2)
            if green_method == "hej":
                eps = max(self.dx,self.dy)*2
                self.G = self.hej_fun(R, eps=eps)
            elif green_method == "chat":
                self.G = self.chatelain_fun(R)
            else: 
                raise("Wrong method to compute the Green function!!")
            self.Gfft = torch.fft.fftn(self.dx*self.dy*self.G)
            numpy.save(filename, self.Gfft.cpu().numpy())


    def solve(self,u):
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
        G = torch.zeros((2*self.nx,2*self.ny))
        G = -(torch.log(x))/(2*torch.pi)
        print("Python will give an error of 0 division. Ignore it.")
        req = torch.sqrt(self.dx*self.dy/torch.pi)
        singularity = -(req**2) * (torch.log(req)-1/2)/2 # integral over spherical area equivalent to a rectangle of dimension [dx,dy]
        G[0,0]        = singularity
        G[-1,0]       = singularity
        G[0,-1]       = singularity
        G[-1,-1]      = singularity
        return G

    def hej_fun(self, x, eps=0.01):
        """
        Green function approximation following Hejlesen et al, Applied Mathematics Letters, 2013
        """
        G = torch.zeros((2*self.nx,2*self.ny))
        xeps=x/eps
        G = -(
                torch.log(x) + \
                0.5*special.exp1(xeps**2/2) - \
                (
                    137/120 - 163/240*xeps**2 + \
                    137/960*xeps**4 - \
                    7/640*xeps**6 + \
                    1/3840*xeps**8
                )*torch.exp(-xeps**2/2)
            )/(2*torch.pi)
        singularity   = (torch.tensor(numpy.euler_gamma)/2-torch.log(torch.sqrt(torch.tensor(2.0))*eps) + 137/120)/(2*torch.pi)
        G[0,0]        = singularity
        G[-1,0]       = singularity
        G[0,-1]       = singularity
        G[-1,-1]      = singularity
        return G

    def bump_u_sol(self):
        """
        Compute an example solution sol for a bump function u in 2d
        """
        X,Y = torch.meshgrid(self.x,self.y)
        Usmall = torch.zeros((self.nx,self.ny))
        X0 = 2
        Y0 = 2
        XX = X-X0
        YY = Y-Y0
        RR2 = XX**2+YY**2
        c=6
        Usmall = -4*c*torch.exp(-c/(1-RR2)) * (c*RR2 + XX**4 + YY**4 + 2*(XX**2)*(YY**2) -1 ) * (-1 + RR2)**(-4)
        Usmall[RR2>=1] = 0

        sol = torch.exp(-c/(1-RR2))
        sol[RR2>=1] = 0
        return (Usmall,sol)
    

    def test_accuracy(self):
        """
        test solver's accuracy using a known analytical solution taken from Saverin, 2023 arxiv
        """
        self.X,self.Y = torch.meshgrid(self.x,self.y)

        # compute example of an initial function and its solution
        (Usmall,sol) = self.bump_u_sol()
        self.U[:self.nx,:self.ny] = Usmall # 0 padded extended initial function
        out_real = sol # exact solution in the smaller domain

        # compute approximate solution using the Green function
        out=self.solve(Usmall)

        diff = torch.sqrt((out - out_real)**2)/torch.max(torch.max(abs(out_real)))

        # === compute errors between real and approximated solution ====
        print("The Linf error is "+str(torch.max(torch.max(abs(diff)))))
        print("The L2 error is "+str(torch.sqrt(torch.sum(torch.sum(diff**2))/(self.nx*self.ny))))


        # === plot solution ===
        import matplotlib.pylab as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.X, self.Y, torch.transpose(out_real,0,1))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.title("Solution real")

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.X, self.Y, torch.transpose(out,0,1))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.title("Solution approx")

        plt.show()



if __name__ == "__main__":

    ps = PoissonSolver( 
                 torch.linspace(0,5,1360),
                 torch.linspace(-5,5,1360),
                 method       = "cpu",
                 green_method = "hej",
                 overwrite    = True
                 )
    ps.test_accuracy()

