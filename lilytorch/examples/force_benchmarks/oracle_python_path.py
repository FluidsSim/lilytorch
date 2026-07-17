"""Physical-oracle comparison of the eulerian vs lagrangian force readouts.

Two analytic cases on a sphere of radius R where the exact force is known in
closed form (both exact for ANY closed body, via the divergence theorem):

  A) PRESSURE   p = -G*x, u = 0
     F_p = -oint p n dS = -int grad(p) dV = G * V  in +x.   (viscous = 0)

  B) VISCOUS    u_x = c*y^2, u_y = u_z = 0, p = 0   (divergence-free)
     sigma_ij = nu*rho*(d_i u_j + d_j u_i) => (div sigma)_x = nu*rho*lap(u_x) = 2*nu*rho*c
     F_v = oint (sigma.n)_x dS = int (div sigma)_x dV = 2*nu*rho*c*V  in +x.

The sphere SDF is an exact distance function (|grad phi| = 1), so this isolates
the delta convention from the force_delta_order question entirely.

The lagrangian readout is fed an EXACT world-frame uv-sphere triangulation
(body._build_surface_3d leaves tri_*_world in a bbox-centred LOCAL frame; only
BDIMhandler._refresh_lagrangian_tris_3d moves it to world, and there is no
handler here).  The triangulation is validated against oint x.n dS = 3V first.
"""
import math

import torch

from lilytorch.tests.test_two_phase import _parity_pars
from lilytorch.tests.test_lagrangian import _build_sphere_tris
from lilytorch.src.solver import FluidSolver

R = 0.2
CX = CY = CZ = 0.5
NU, RHO = 1.0e-2, 1000.0
G = 3.0        # pressure gradient magnitude for case A
CSH = 2.0      # shear coefficient for case B
V_TRUE = (4.0 / 3.0) * math.pi * R ** 3
LON, LAT = 160, 90     # uv-sphere resolution (independent of the fluid grid)


def build(N, eps_mult, force_method):
    body = [f'lambda x, y, z: sphere(x,y,z,xt={CX},yt={CY},zt={CZ},r={R})']
    pars = _parity_pars(3, N, 2.0e-3, NU, RHO, body)
    pars["solver"]["force_method"] = force_method
    pars["solver"]["eps_multiplier"] = eps_mult
    sp = FluidSolver(pars, dtype=torch.float64, compute_forces=True)
    c = sp.composite_body
    # forces_method2_3d full-grid branch reads comp.sdf_vals with no fallback
    # (pre-existing bug; same workaround as validation/.../run_drop_sphere_3d.py).
    if not hasattr(c, "sdf_vals") or c.sdf_vals is None:
        c.sdf_vals = torch.stack([b.sdf_val for b in c.bodies], dim=0)
    # Exact world-frame triangulation for the lagrangian readout.
    tc, tn, ta = _build_sphere_tris(CX, CY, CZ, R, LON, LAT)
    b = c.bodies[0]
    b.tri_centroid_world = tc
    b.tri_normal_world = tn
    b.tri_area = ta
    return sp


def check_triangulation():
    tc, tn, ta = _build_sphere_tris(CX, CY, CZ, R, LON, LAT)
    area = float(ta.sum())
    # oint x.n dS = 3V  (x measured from the sphere centre)
    xc = tc - torch.tensor([[CX], [CY], [CZ]], dtype=torch.float64)
    vol = float((xc * tn).sum(dim=0).mul(ta).sum()) / 3.0
    print(f"triangulation check: area {area:.6f} vs {4*math.pi*R**2:.6f} "
          f"({area/(4*math.pi*R**2):.4f}x) | enclosed vol {vol:.6e} vs "
          f"{V_TRUE:.6e} ({vol/V_TRUE:.4f}x)")
    return abs(vol / V_TRUE - 1.0) < 0.01


def readout(sp, method, u, v, w, p):
    if method == "eulerian":
        sp.forces_method2_3d(u, v, w, p, 0)
    else:
        sp.forces_lagrangian_3d(u, v, w, p, 0)
    return (float(sp.friction_force_lin_x.reshape(-1)[0]),
            float(sp.pressure_force_x.reshape(-1)[0]))


def run(N, eps_mult):
    out = {}
    for method in ("eulerian", "lagrangian"):
        sp = build(N, eps_mult, method)
        X, Y = sp.composite_body.X, sp.composite_body.Y
        z = torch.zeros_like(sp.p0)
        _, fp = readout(sp, method, z, z.clone(), z.clone(), -G * X)
        fv, _ = readout(sp, method, CSH * Y ** 2, z.clone(), z.clone(),
                        torch.zeros_like(sp.p0))
        out[method] = (fp, fv)
    return out


if __name__ == "__main__":
    print(f"sphere R={R}  V={V_TRUE:.6e}")
    assert check_triangulation(), "uv-sphere triangulation is not accurate"
    fp_true = G * V_TRUE
    fv_true = 2 * NU * RHO * CSH * V_TRUE
    print(f"exact pressure force F_p = G*V          = {fp_true:.6e}")
    print(f"exact viscous  force F_v = 2*nu*rho*c*V = {fv_true:.6e}")
    print()
    hdr = (f"{'N':>4} {'R/h':>5} {'eps_m':>6} | "
           f"{'eul F_p':>11} {'ratio':>6} | {'lag F_p':>11} {'ratio':>6} | "
           f"{'eul F_v':>11} {'ratio':>6} | {'lag F_v':>11} {'ratio':>6}")
    print(hdr)
    print("-" * len(hdr))
    for N in (24, 32, 48, 64):
        h = 1.0 / N
        for eps_mult in (1.0, 2.0):
            r = run(N, eps_mult)
            e_fp, e_fv = r["eulerian"]
            l_fp, l_fv = r["lagrangian"]
            print(f"{N:>4} {R/h:>5.1f} {eps_mult:>6.1f} | "
                  f"{e_fp:>11.4e} {e_fp/fp_true:>6.2f} | {l_fp:>11.4e} {l_fp/fp_true:>6.2f} | "
                  f"{e_fv:>11.4e} {e_fv/fv_true:>6.2f} | {l_fv:>11.4e} {l_fv/fv_true:>6.2f}")
