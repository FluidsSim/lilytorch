import torch
import poisson_cpp # for _C import loading poisson_cpp into torch

class PoissonSolver:
    ''' Solver class for the Poisson equation with variable coefficients '''
    def __init__(self, device, h, tol=1e-2, max_cycles=2, nsmoothing=5, verbose=True):
        # h2, tol, jcap_tol, max_cycles, nsmoothing, verbose, BC
        self.solver = torch.classes.poisson_cpp.PoissonSolver(
            float(h)**2*torch.ones((1, 1)),
            tol,
            1e-12, # jcap_tol
            max_cycles,
            nsmoothing,
            0 if device == torch.device("cpu") else 1,
            1, # 0: dirichlet, 1: neumann
            verbose
        )

    def CG(self, f, u, c, c_h, c_v , h2, maxit=100):
        # return torch.ops.poisson_cpp.CG.default(f, u.clone(), c, c_h, c_v, float(h2), maxit)
        return self.solver.CG(f, u.clone(), c, c_h, c_v, h2, maxit)

    def CG_jacobi_cond(self, f, u, c, c_h, c_v, h2, maxit=100):
        # return torch.ops.poisson_cpp.CG_jacobi_cond.default(f, u.clone(), c, c_h, c_v, float(h2), maxit)
        return self.solver.CG_jacobi_cond(f, u.clone(), c, c_h, c_v, h2, maxit)


    def solve_multigrid(self, f, u, c, c_h, c_v):
        # return torch.ops.poisson_cpp.solve_multigrid.default(f, u.clone(), c, c_h, c_v)
        return self.solver.solve_multigrid(f, u.clone(), c, c_h, c_v)

def test_solvers():
    use_gpu=False
    N=2**8+1

    import poisson_solvers.solutions2d
    from matplotlib import pyplot
    import time

    X, Y, u_exact, f, c, c_h, c_v = poisson_solvers.solutions2d.variable_coeff_c_hat(N)
    h = X[1,0]-X[0,0]
    print("Number of elements:{}, h={}".format(N, h))

    u0=torch.zeros((N,N))
    if torch.cuda.is_available() and use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("Using the CPU.")
        device = torch.device("cpu")
        torch.set_num_threads(8)

    solver = PoissonSolver(
        device,
        h,
        verbose=True,
        max_cycles=10,
        nsmoothing=100,
        tol=1e-4
    )

    # c_h = c[1:,:]
    # c_v = c[:,1:]
    c       = c.to(device)
    c_h     = c_h.to(device)
    c_v     = c_v.to(device)
    f       = f.to(device)
    u0      = u0.to(device)
    u_exact = u_exact.to(device)

    # ==== COMPUTE THE SOLUTION ======
    start = time.time()
    u_cg = solver.CG(f, u0, c, c_h, c_v , h**2, maxit=100).cpu()
    print("CG method took {}s".format(time.time()-start))
    start = time.time()
    u_cg_jac = solver.CG_jacobi_cond(f, u0, c, c_h, c_v , h**2, maxit=15000)[0].cpu()
    print("CG-PREC method took {}s".format(time.time()-start))
    start = time.time()
    u_multigrid = solver.solve_multigrid(f, u0, c, c_h, c_v).cpu()
    print("Multigrid method took {}s".format(time.time()-start))

    fig, (ax_1, ax_2, ax_3, ax_4) = pyplot.subplots(1, 4, figsize=(20,5))
    CS1=ax_1.imshow(u_cg.T,origin = "lower")
    ax_1.set_title("CG")
    fig.colorbar(CS1)
    CS2=ax_2.imshow(u_cg_jac.T,origin = "lower")
    ax_2.set_title("CG Jac")
    fig.colorbar(CS2)
    CS3=ax_3.imshow(u_multigrid.T,origin = "lower")
    ax_3.set_title("Multigrid")
    fig.colorbar(CS3)
    if u_exact is not None:
        CS4=ax_4.imshow(u_exact.cpu().T,origin = "lower")
        ax_4.set_title("Exact")
        fig.colorbar(CS4)
    pyplot.show()

if __name__ == "__main__":
    test_solvers()