import os, sys, time
import torch
import matplotlib
import matplotlib.pyplot as plt
import poisson_solvers.solutions2d
from testing_nonapprox_poisson import PoissonSolver
from testing_nonapprox_poisson_pytorch import PoissonSolver as PoissonSolverPt

def benchmark_multigrid():
    # solvers
    names = ["Python CPU", "C++ CPU"]
    devices = 2*["cpu"]
    limits = [4097, 2049]
    cuda = torch.cuda.is_available()
    if cuda:
        print("Torch GPU: ", torch.cuda.get_device_name(0))
        names.append("Python GPU")
        devices.append("cuda")
        limits.append(8193)
        names.append("C++ GPU")
        devices.append("cpu") # switch to cuda happens on C++ side
        limits.append(8193)

    # testing params
    max_cycles = 10
    nsmoothing = 100
    tol = 1e-4
    verbose = False
    profiling = False # trace.json can be opened with ui.perfetto.dev, best on chrome for better ram usage
    Ns = [2**i+1 for i in range(5, 14)]
    # Ns = [257]
    # threads = [2**i for i in range(0, 7)]
    threads = [os.cpu_count() // 2] # seems best with physical core count

    times = torch.zeros((len(names), len(threads), len(Ns))).cpu()
    if len(sys.argv) == 1:
        if profiling:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.start()
        for n, N in enumerate(Ns):
            print(f"{N = }")
            X, Y, u_exact, f, c, c_h, c_v = poisson_solvers.solutions2d.variable_coeff_c_hat(N, torch.device("cpu"))
            h = float(X[1,0]-X[0,0])
            u0=torch.zeros((N,N))

            # solvers continued
            solvers = []
            solvers.append(PoissonSolver(torch.device("cpu"), h, verbose=verbose, max_cycles=max_cycles, nsmoothing=nsmoothing, tol=tol))
            solvers.append(PoissonSolverPt(torch.device("cpu"), h, verbose=verbose, max_cycles=max_cycles, nsmoothing=nsmoothing, tol=tol))
            if cuda:
                solvers.append(PoissonSolver(torch.device("cuda"), h, verbose=verbose, max_cycles=max_cycles, nsmoothing=nsmoothing, tol=tol))
                solvers.append(PoissonSolverPt(torch.device("cuda"), h, verbose=verbose, max_cycles=max_cycles, nsmoothing=nsmoothing, tol=tol))

            s = 0
            for solver, name, device in zip(solvers, names, devices):
                print(4*" "+name)
                if N > limits[s]: # CPU gets very slow above N=2**11
                    print(8*" "+"skipping")
                    s += 1
                    continue
                f = f.to(torch.device(device))
                u0 = u0.to(torch.device(device))
                c = c.to(torch.device(device))
                c_h = c_h.to(torch.device(device))
                c_v = c_v.to(torch.device(device))
                if device == "cpu":
                    for t, thread in enumerate(threads):
                        torch.set_num_threads(thread)
                        start = time.time()
                        u_multigrid = solver.solve_multigrid(f, u0, c, c_h, c_v).cpu()
                        times[s, t, n] = time.time() - start
                        print(8*" "+f"{times[s, t, n]} ({thread} threads)")
                if device == "cuda":
                    start = time.time()
                    u_multigrid = solver.solve_multigrid(f, u0, c, c_h, c_v).cpu()
                    times[s, :, n] = time.time() - start
                    print(8*" "+f"{times[s, 0, n]}")
                s += 1
        if profiling:
            prof.stop()
            prof.export_chrome_trace("trace.json")
        print(times)
    else:
        times = torch.Tensor([
            [[0.5938, 0.7655, 0.9961, 1.5047, 1.9506, 2.9063,  8.6749, 88.6832,  0.0000]],
            [[0.2494, 0.3307, 0.4597, 0.7744, 1.0858, 2.0393, 21.8018,  0.0000,  0.0000]],
            [[1.5745, 1.7536, 2.1148, 2.4765, 2.8365, 3.2782,  5.8111, 16.4815, 58.1750]],
            [[1.0562, 1.3636, 1.6519, 1.9632, 2.2442, 2.5276,  4.7345, 15.2843, 56.0153]]])

    cmaps = ["Purples", "Blues", "Reds", "Oranges"]
    ends = torch.sum(times > 0., dim=2).squeeze()
    fig, ax = plt.subplots()
    for i in range(len(devices)):
        if devices[i] == "cpu":
            for j in range(len(threads)):
                ax.plot(Ns[:ends[i]], times[i, j][:ends[i]], c=matplotlib.colormaps[cmaps[i]]((j+1)/(len(threads)+1)), label=f"{names[i]}: {threads[j]}")
        if devices[i] == "cuda":
            ax.plot(Ns[:ends[i]], times[i, 0][:ends[i]], c=matplotlib.colormaps[cmaps[i]](0.7), label=names[i])
    ax.plot([Ns[0], Ns[-1]], [0, 0], c="0.7", linestyle="--")
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xticks(Ns, Ns)
    ax.set_xlabel("N")
    ax.set_ylabel("Time (s)")
    fig.legend()
    plt.show()


def benchmark_solvers():
    use_gpu = True
    N=2**8+1
    Ns = [2**i+1 for i in range(2, 11)]
    times = [[], [], []]
    names = ["CG", "CG-PREC", "Multigrid"]

    if len(sys.argv) == 1:
        for N in Ns:
            print(f"{N = }")
            X, Y, u_exact, f, c, c_h, c_v = poisson_solvers.solutions2d.variable_coeff_c_hat(N)
            h = float(X[1,0]-X[0,0])
            u0=torch.zeros((N,N))

            if torch.cuda.is_available() and use_gpu:
                print("Torch GPU: ", torch.cuda.get_device_name(0))
                device = torch.device("cuda")
            else:
                print("Using the CPU.")
                device = torch.device("cpu")
                torch.set_num_threads(8)

            solver = PoissonSolver(device, h, verbose=False, max_cycles=10, nsmoothing=100, tol=1e-4)
            # solver = PoissonSolverPt(device, h, verbose=False, max_cycles=10, nsmoothing=100, tol=1e-4)

            c       = c.to(device)
            c_h     = c_h.to(device)
            c_v     = c_v.to(device)
            f       = f.to(device)
            u0      = u0.to(device)
            u_exact = u_exact.to(device)

            print(4*" "+names[0])
            start = time.time()
            u_cg = solver.CG(f, u0, c, c_h, c_v , h**2, maxit=100).cpu()
            times[0].append(time.time() - start)
            print(8*" "+str(times[0][-1]))

            print(4*" "+names[1])
            start = time.time()
            u_cg_jac = solver.CG_jacobi_cond(f, u0, c, c_h, c_v , h**2, maxit=15000)[0].cpu()
            times[1].append(time.time() - start)
            print(8*" "+str(times[1][-1]))

            print(4*" "+names[2])
            start = time.time()
            u_multigrid = solver.solve_multigrid(f, u0, c, c_h, c_v).cpu()
            times[2].append(time.time() - start)
            print(8*" "+str(times[2][-1]))
    else:
        times = [[0.0010042190551757812, 0.0, 0.0, 0.0, 0.0, 0.0015041828155517578, 0.0009989738464355469, 0.0030040740966796875, 0.00701594352722168],
                 [0.0020020008087158203, 0.15325021743774414, 2.1715376377105713, 2.2712419033050537, 2.5542900562286377, 3.421254873275757, 6.376044273376465, 13.018293380737305, 41.07837224006653],
                 [0.1417829990386963, 0.28310155868530273, 0.4181642532348633, 0.572563886642456, 0.7414708137512207, 0.9435784816741943, 1.3870735168457031, 1.8244073390960693, 2.8636295795440674]]
    print(times)
    colours = ["b", "m", "r"]
    fig, ax = plt.subplots()
    for i in range(len(times)):
        if i == 0: continue # skip broken CG solver
        plt.plot(Ns, times[i], c=colours[i], label=names[i])
    ax.plot([Ns[0], Ns[-1]], [0, 0], c="0.7", linestyle="--")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(Ns, Ns)
    ax.set_xlabel("N")
    ax.set_ylabel("Time (s)")
    fig.legend()
    plt.show()

if __name__ == "__main__":
    benchmark_solvers()