

from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{"size":12})
plt.rcParams["figure.figsize"] = 15,15


pars = yaml2pyobject("lilytorch/scripts/flow_past_cylinder.yaml")


R = 0.1 # radius of the cylinder
final_conv_time = 7.0 # final convective time to simulate

# pars["solver"]["xmin"] = -1
# pars["solver"]["xmax"] = 1
# pars["solver"]["ymin"] = -1
# pars["solver"]["ymax"] = 1
# pars["solver"]["N"]    = 512

pars["solver"]["xmin"] = -0.5
pars["solver"]["xmax"] = 0.5
pars["solver"]["ymin"] = -0.5
pars["solver"]["ymax"] = 0.5
pars["solver"]["N"]    = 256
pars["body"]["sdf"]    = ["lambda x, y: circle(x,y,xt={},yt=0,r={})".format(0,R)]

u_inlet = pars["boundary_conditions"]["BC_values_u"]

pars["output"]["save_frames"] = True
pars["output"]["save_every"]=50
path=pars["output"]["save_path"] + pars["output"]["results_folder"]

dx=(pars["solver"]["xmax"]-pars["solver"]["xmin"])/pars["solver"]["N"]
t_clf=dx/u_inlet[0]


# pars["solver"]["dt"]                = t_clf
# pars["solver"]["convection_method"] = "implicit"
# pars["solver"]["nt"]                = 520

pars["solver"]["dt"]                = 0.1*t_clf
pars["solver"]["convection_method"] = "abdquickest"
pars["solver"]["nt"]                = int(final_conv_time*R/u_inlet[0]/pars["solver"]["dt"])

print("Number of iterations: ",pars["solver"]["nt"])

pars["solver"]["poisson_verbose"] = True


def run_sim():
    # =========== Run simulation ===========
    solver = FluidSolver(pars, dtype=torch.float32, compute_forces=True)
    solver.run_sim()


def analyze_results():
    # =========== Analyze results ===========

    viscous_forces = np.load(path+"/viscous_drags.npy")
    pressure_forces = np.load(path+"/pressure_drags.npy")


    time=np.arange(pars["solver"]["nt"])*pars["solver"]["dt"]

    U = u_inlet[0]
    D = 2*R
    nu = pars["solver"]["nu"]
    Re = U*D/nu
    print("Reynolds number: ",Re)

    viscous_drag_coeff = 2*viscous_forces[0,0,:]/(pars["solver"]["rho"]*D*U**2)
    pressure_drag_coeff = 2*pressure_forces[0,0,:]/(pars["solver"]["rho"]*D*U**2)
    total_drag_coeff = viscous_drag_coeff + pressure_drag_coeff

    convective_time = time*U/R


    data=np.genfromtxt('data_to_save/koumoutsatokos_keonard_1995.csv', delimiter=',')

    plt.figure(figsize=(8, 4))
    plt.title("Re = %.2f" % Re)
    plt.plot(convective_time,viscous_drag_coeff,'b')
    plt.plot(convective_time,pressure_drag_coeff,'r')
    plt.plot(convective_time,total_drag_coeff,'k', label="Our")
    plt.plot(data[:,0],data[:,1],'k--',label="Koumoutsakos & Leonard 1995")
    plt.xlim([convective_time[0],convective_time[-1]])
    plt.xlabel("Convective Time [t*U/R]")
    plt.ylim([0,2])
    plt.xlim([0,7])
    plt.ylabel("Drag Coefficient")
    plt.legend()
    plt.savefig(path+"/drag_force_flow_past_cylinder_"+pars["solver"]["convection_method"]+".png")




if __name__ == "__main__":
    run_sim()
    analyze_results()







