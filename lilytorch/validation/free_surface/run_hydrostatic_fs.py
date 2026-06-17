"""Hydrostatic validation of the ONE-FLUID FreeSurfaceSolver.

Closed box, water below y=0.5, air (void) above, under gravity, at rest.
Expected steady state:
  * velocity ~ 0,
  * p = 0 in the air (alpha<0.5),
  * p = rho_w g (0.5 - y) in the water (p=0 at the free surface, increasing
    downward) -- i.e. buoyancy-producing hydrostatic pressure emerges purely
    from the p=0 free-surface BC, with NO density jump.
"""
import math, torch
torch.set_default_device("cuda")
from lilytorch.src.free_surface_solver import FreeSurfaceSolver

rho_w, g = 1000.0, 9.81
Nx = Ny = 48
h = 1.0/Nx
dt = 0.2*h/math.sqrt(g*0.5)
H = 0.5
pars = {"solver": {
    "use_gpu": True, "nthreads": 1, "Nx": Nx, "Ny": Ny,
    "xmin":0.0,"xmax":1.0,"ymin":0.0,"ymax":1.0,
    "nt": 800, "nu": 1e-3, "rho": rho_w, "dt": dt,
    "convection_method":"cds",
    "poisson_tol":1e-8,"jacobi_weight":1.0,"poisson_max_cycles":40,
    "poisson_max_mgcg_cycles":40,"poisson_nsmoothing":6,"poisson_verbose":False,
    "poisson_folder":"lilytorch/data/","poisson_method":"mgcg","poisson_smoother":"rbgs",
    "dtype":"float64","solver_method":"python","rho_body":rho_w,
    "gravity":[0.0,-g],
    "two_phase": {"alpha_init":"lambda X, Y: (Y < 0.5).double()",
                  "rho_water":rho_w,"rho_air":1.0,"nu_water":1e-3,"nu_air":1e-3,
                  "face_density":"harmonic"},
    "free_surface": {"extend_iters": 10},
    },
    "boundary_conditions":{"BC_type_u":["D","D","D","D"],"BC_values_u":[0.0]*4,
        "BC_type_v":["D","D","D","D"],"BC_values_v":[0.0]*4},
    "body":{"type":"composite_analytical","plotting":False,
        "sdf":["lambda x, y: circle(x,y,xt=0,yt=0,r=0.04)"],
        "update_maps":[{"rotation":"lambda t: torch.tensor(0.0)",
            "translation":["lambda t: torch.tensor(0.4)","lambda t: torch.tensor(0.85)"]}]},
    "output":{"save_path":"/tmp/fs_hydro/","save_frames":False,"save_every":10**9,
        "save":False,"save_drags":False,"vmin":-1,"vmax":1}}

solver = FreeSurfaceSolver(pars, dtype=torch.float64, compute_forces=False)
solver.inside = lambda *a, **k: True
solver.set_initial_conditions()
u, v, p = solver.u0, solver.v0, solver.p0
Y = torch.meshgrid(solver.x, solver.y, indexing="ij")[1]
alpha = solver.two_phase.alpha
int = torch.ones_like(Y, dtype=torch.bool)
int[0,:]=int[-1,:]=int[:,0]=int[:,-1]=False
water = (alpha>=0.5)&int; air=(alpha<0.5)&int
p_ref = torch.where(Y<H, rho_w*g*(H-Y), torch.zeros_like(Y))

for it in range(solver.nt):
    u,v,p,stop = solver.step_(u,v,p,it,it*dt)
    if stop: print("stop (explosion) at",it); break

umax = float(torch.maximum(u.abs().max(), v.abs().max()))
# gauge: p is pinned to 0 in air, so compare directly (no gauge fit needed)
p_air_max  = float(p[air].abs().max()) if air.any() else 0.0
perr_water = float((p[water]-p_ref[water]).abs().max())
pscale = rho_w*g*H
print(f"\nFreeSurfaceSolver hydrostatic (Nx={Nx}, nt={solver.nt}):")
print(f"  |u|max            = {umax:.3e}   (expect ~0)")
print(f"  max|p| in AIR     = {p_air_max:.3e}   (expect ~0, p=0 BC)")
print(f"  max|p-p_hydro| water = {perr_water:.3e}  rel={100*perr_water/pscale:.2f}%  (vs rho g H={pscale:.1f})")
ok = (umax<0.05*math.sqrt(g*H)) and (p_air_max<0.02*pscale) and (perr_water<0.05*pscale)
print(f"  ===> {'PASSED' if ok else 'FAILED'}")
