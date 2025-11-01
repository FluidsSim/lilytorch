

from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.rc('font', **{"size":20})
plt.rcParams["figure.figsize"] = 15,15
import os


# nxs=[4096, 2048, 1024, 512, 256, 128] # grid sizes to test
nxs=[2048, 1024, 512, 256, 128, 64] # grid sizes to test


for nx in nxs:

    pars = yaml2pyobject("lilytorch/scripts/flow_past_cylinder.yaml")

    R = 0.1 # radius of the cylinder
    time_stop=200

    U = pars["boundary_conditions"]["BC_values_u"][0]
    D = 2*R
    nu = pars["solver"]["nu"]
    Re = U*D/nu
    print("Reynolds number: ",Re)

    pars["solver"]["xmin"] = -1
    pars["solver"]["xmax"] = 1
    pars["solver"]["ymin"] = -1
    pars["solver"]["ymax"] = 1
    pars["solver"]["N"]    = nx
    pars["body"]["sdf"]    = ["lambda x, y: circle(x,y,xt={},yt=0,r={})".format(pars["solver"]["xmin"]+3*R,R)]

    dx=1/pars["solver"]["N"]
    t_clf=dx/pars["boundary_conditions"]["BC_values_u"][0]


    # pars["solver"]["dt"]                = 0.8*t_clf
    # pars["solver"]["convection_method"] = "implicit"
    # pars["solver"]["nt"]                = int(time_stop/pars["solver"]["dt"])+1

    pars["solver"]["dt"]                = 0.1*t_clf
    pars["solver"]["convection_method"] = "abdquickest"
    pars["solver"]["nt"]                = int(time_stop/pars["solver"]["dt"])+1

    pars["output"]["save_frames"] = True
    pars["output"]["save_every"]  = 200


    # =========== Run simulation ===========
    solver = FluidSolver(pars, dtype=torch.float32, compute_forces=True)
    solver.save_path = f'/data/andreaferrario/ns_data/flow_past_cylinder_error_tests/'+pars["solver"]["convection_method"]+'/{}'.format(nx)
    os.makedirs(solver.save_path, exist_ok=True)
    solver.set_initial_conditions()
    u=solver.u0
    v=solver.v0
    p=solver.p0

    for it in tqdm(range(0,solver.nt)):
        t=it*solver.dt
        (u,v,p,stop_sim) = solver.step_(u, v, p, it, t)

    # save last iteration
    uv_path = f'{solver.save_path}/uv_field'
    os.makedirs(uv_path, exist_ok=True)
    np.save(f'{uv_path}/u',u.cpu().numpy())
    np.save(f'{uv_path}/v',v.cpu().numpy())
    np.save(f'{uv_path}/p',p.cpu().numpy())
