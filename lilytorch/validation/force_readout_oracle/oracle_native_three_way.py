"""Compare all THREE force readouts against an exact physical oracle.

  1. eulerian / ndelta   (force_submethod=0)
  2. eulerian / deltaH   (force_submethod=1)
  3. lagrangian

Same two analytic cases as oracle_forces.py, on the same sphere:
  A) p = -G*x, u = 0            -> F_p = G*V          (exact, divergence thm)
  B) u_x = c*y^2, p = 0         -> F_v = 2*nu*rho*c*V (exact, divergence thm)

deltaH lives ONLY in the native streaming op (the python fallback in
forces_method2_3d has no deltaH branch), so this drives
``streaming_sdf_forces_post_3d`` directly with a hand-built single-body scene.
CPU/float64 for accuracy; the CPU twin carries the same delta convention as the
CUDA kernel (verified: streaming_sdf_cpu.cpp:697 == streaming_sdf.cu:404).

Output channels: [fv_x,fv_y,fv_z, tv_x,tv_y,tv_z, fp_x,fp_y,fp_z, tp_x,tp_y,tp_z]
"""
import math

import torch

from lilytorch.src.native import streaming_sdf_forces_post_3d
from lilytorch.tests.test_lagrangian import _build_sphere_tris

R = 0.2
C = 0.5                    # sphere centre (all axes)
NU, RHO = 1.0e-2, 1000.0
G = 3.0
CSH = 2.0
V_TRUE = (4.0 / 3.0) * math.pi * R ** 3
DT = torch.float64


def build_scene(N, eps_mult):
    h = 1.0 / (N - 1)
    g = torch.linspace(0.0, (N - 1) * h, N, dtype=DT)

    # --- body SDF table (exact sphere, fine local grid) ---
    M = 161
    half = 0.4
    bl = torch.linspace(-half, half, M, dtype=DT)
    BX, BY, BZ = torch.meshgrid(bl, bl, bl, indexing="ij")
    F_flat = (torch.sqrt(BX**2 + BY**2 + BZ**2) - R).ravel().contiguous()
    F_offsets = torch.tensor([0], dtype=torch.int64)
    body_shapes = torch.tensor([[M, M, M]], dtype=torch.int64)
    inv_d = 1.0 / float(bl[1] - bl[0])
    body_meta = torch.tensor(
        [[float(bl[0])] * 3 + [float(bl[-1])] * 3 + [inv_d] * 3 + [inv_d**3]],
        dtype=DT)

    # kin: R_T(9) identity + bp(3) + cm(3) + lv(3) + av(3)
    kin = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1,
                         C, C, C,  C, C, C,  0, 0, 0,  0, 0, 0]], dtype=DT)

    aabb_lo = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    aabb_dim = torch.tensor([[N, N, N]], dtype=torch.int64)

    # --- union CC SDF on the fluid grid (exact sphere) ---
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    sdf_cc = (torch.sqrt((X - C)**2 + (Y - C)**2 + (Z - C)**2) - R).ravel().contiguous()

    return dict(h=h, g=g, X=X, Y=Y, Z=Z, F_flat=F_flat, F_offsets=F_offsets,
                body_shapes=body_shapes, body_meta=body_meta, kin=kin,
                aabb_lo=aabb_lo, aabb_dim=aabb_dim, sdf_cc=sdf_cc, N=N,
                eps=eps_mult * h)


def eulerian(sc, submethod, u, v, w, p):
    out = torch.zeros((1, 12), dtype=torch.float64)
    nu_rho = torch.tensor([NU * RHO], dtype=DT)
    streaming_sdf_forces_post_3d(
        sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
        sc["kin"], sc["aabb_lo"], sc["aabb_dim"],
        sc["g"], sc["g"], sc["g"], sc["h"], sc["N"] ** 3,
        sc["sdf_cc"], 0,
        u.ravel().contiguous(), v.ravel().contiguous(),
        w.ravel().contiguous(), p.ravel().contiguous(),
        nu_rho, sc["eps"], sc["eps"], sc["h"] ** 3, 1, out,
        submethod, 1.5 * sc["h"],
    )
    return float(out[0, 0]), float(out[0, 6])      # fv_x, fp_x


def lagrangian(sc, u, v, w, p):
    tc, tn, ta = _build_sphere_tris(C, C, C, R, 160, 90)
    out = torch.zeros((1, 12), dtype=torch.float64)
    from lilytorch.src.forces import _viscous_stress_tensor
    e = _viscous_stress_tensor((u, v, w), sc["h"])
    torch.ops.lilytorch_kernels.lagrangian_forces_3d.default(
        e[0][0].contiguous(), e[1][1].contiguous(), e[2][2].contiguous(),
        e[0][1].contiguous(), e[0][2].contiguous(), e[1][2].contiguous(),
        p.contiguous(), torch.tensor([NU * RHO], dtype=DT),
        tc, tn, ta, torch.tensor([0, tc.shape[1]], dtype=torch.int64),
        torch.tensor([[C, C, C]], dtype=DT),
        float(sc["g"][0]), float(sc["g"][0]), float(sc["g"][0]),
        1.0 / sc["h"], 1.0 / sc["h"], 1.0 / sc["h"],
        sc["N"], sc["N"], sc["N"], 0, 0.0, out,
    )
    return float(out[0, 0]), float(out[0, 6])


def run(N, eps_mult):
    sc = build_scene(N, eps_mult)
    X, Y = sc["X"], sc["Y"]
    z = torch.zeros_like(X)
    pA, uA = -G * X, z
    uB, pB = CSH * Y ** 2, z

    res = {}
    for name, sm in (("ndelta", 0), ("deltaH", 1)):
        _, fp = eulerian(sc, sm, uA, z.clone(), z.clone(), pA)
        fv, _ = eulerian(sc, sm, uB, z.clone(), z.clone(), pB)
        res[name] = (fp, fv)
    _, fp = lagrangian(sc, uA, z.clone(), z.clone(), pA)
    fv, _ = lagrangian(sc, uB, z.clone(), z.clone(), pB)
    res["lagrangian"] = (fp, fv)
    return res


if __name__ == "__main__":
    fp_true = G * V_TRUE
    fv_true = 2 * NU * RHO * CSH * V_TRUE
    print(f"sphere R={R}  V={V_TRUE:.6e}")
    print(f"exact F_p = {fp_true:.6e}   exact F_v = {fv_true:.6e}")
    print()
    hdr = (f"{'R/h':>5} {'eps_m':>6} | {'ndelta F_p':>11} {'r':>5} "
           f"{'deltaH F_p':>11} {'r':>5} {'lagr F_p':>11} {'r':>5} | "
           f"{'ndelta F_v':>11} {'r':>5} {'deltaH F_v':>11} {'r':>5} "
           f"{'lagr F_v':>11} {'r':>5}")
    print(hdr); print("-" * len(hdr))
    for N in (32, 48, 64, 96):
        h = 1.0 / (N - 1)
        for em in (1.0, 2.0):
            r = run(N, em)
            row = f"{R/h:>5.1f} {em:>6.1f} |"
            for key in ("ndelta", "deltaH", "lagrangian"):
                row += f" {r[key][0]:>11.4e} {r[key][0]/fp_true:>5.2f}"
            row += " |"
            for key in ("ndelta", "deltaH", "lagrangian"):
                row += f" {r[key][1]:>11.4e} {r[key][1]/fv_true:>5.2f}"
            print(row)
