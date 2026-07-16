#!/usr/bin/env python3
"""Force-level proof of the lagrangian marker fix, on the REAL boat hull SDF.

Loads the cached hull SDF, places it at a waterline on a fluid grid, lays down a
clean hydrostatic pressure field, and computes the vertical pressure force with:
  * eulerian band integral on the SDF        (reference, marker-independent)
  * lagrangian with BUGGY markers (tri_local, no local_center offset)
  * lagrangian with FIXED markers (tri_local + local_center)

The fix is correct iff FIXED ~ eulerian and BUGGY is far off (the bug).
"""
import math
import numpy as np
import torch
from skimage import measure
from lilytorch.src.native import RegularGridInterpolator

DT = torch.float64
D = "lilytorch/examples/boat/interp_data"
RHO_W, G = 1000.0, 9.81

xnp = np.load(f"{D}/xnp_hull.npy"); ynp = np.load(f"{D}/ynp_hull.npy")
znp = np.load(f"{D}/znp_hull.npy"); sdf_mesh = np.load(f"{D}/sdf_val_hull.npy")
xc = (xnp[0]+xnp[-1])/2; yc = (ynp[0]+ynp[-1])/2; zc = (znp[0]+znp[-1])/2
local_center = np.array([xc, yc, zc])

# marching cubes -> body-local (bbox-centred) triangulation (as BodyMesh does)
verts, faces, _, _ = measure.marching_cubes(sdf_mesh, level=0.0)
xa, ya, za = xnp-xc, ynp-yc, znp-zc
vx = xa[0]+verts[:,0]*(xa[1]-xa[0]); vy = ya[0]+verts[:,1]*(ya[1]-ya[0]); vz = za[0]+verts[:,2]*(za[1]-za[0])
tv = np.stack([vx,vy,vz],1)[faces]                 # (T,3,3)
tri_local = tv.mean(1)                              # (T,3) bbox-centred
e1 = tv[:,1]-tv[:,0]; e2 = tv[:,2]-tv[:,0]; cr = np.cross(e1,e2)
area = 0.5*np.linalg.norm(cr,axis=1); nrm = cr/np.maximum(np.linalg.norm(cr,axis=1,keepdims=True),1e-30)
ok = area > 1e-20; tri_local, nrm, area = tri_local[ok], nrm[ok], area[ok]

# --- place hull in world at identity pose; waterline cuts the hull in half ---
# hull mesh frame: surface points = tri_local + local_center.  Put the bbox
# centre at world origin; waterline at z=0 (so the hull straddles it).
def to_world(markers_meshframe):
    # markers given in mesh frame (= tri_local + local_center); world = same
    return markers_meshframe

surf_mesh = tri_local + local_center               # true surface (mesh frame)
# shift so bbox centre (=local_center) sits at z=0 waterline-ish: subtract zc only
# keep x,y; set waterline at the hull mid-height
WL = local_center[2]                               # waterline at hull mid-z

# --- fluid grid spanning the hull bbox + margin -----------------------------
pad = 0.2
lo = np.array([xnp[0],ynp[0],znp[0]]) - pad
hi = np.array([xnp[-1],ynp[-1],znp[-1]]) + pad
N = np.array([180, 60, 60])
ax = [torch.linspace(float(lo[i]), float(hi[i]), int(N[i]), dtype=DT) for i in range(3)]
h = float((ax[0][1]-ax[0][0]))
X, Y, Z = torch.meshgrid(ax[0], ax[1], ax[2], indexing="ij")

# hull SDF interpolated onto the fluid grid (mesh-frame coords == world here)
interp_sdf = RegularGridInterpolator(
    (torch.tensor(xnp,dtype=DT), torch.tensor(ynp,dtype=DT), torch.tensor(znp,dtype=DT)),
    torch.tensor(sdf_mesh,dtype=DT), method="linear")
sdf_grid = interp_sdf(X.reshape(-1), Y.reshape(-1), Z.reshape(-1)).reshape(X.shape)

# clean hydrostatic pressure (water below WL), uniform gauge in air
p = torch.where(Z < WL, RHO_W*G*(WL - Z), torch.zeros_like(Z))

# --- eulerian band integral on the SDF (reference) --------------------------
def grid_normals(s, h):
    gx = torch.gradient(s, spacing=h, dim=0, edge_order=2)[0]
    gy = torch.gradient(s, spacing=h, dim=1, edge_order=2)[0]
    gz = torch.gradient(s, spacing=h, dim=2, edge_order=2)[0]
    m = torch.sqrt(gx*gx+gy*gy+gz*gz).clamp(min=1e-12)
    return gz/m
nz = grid_normals(sdf_grid, h)
eps_body = 2*h
delta = torch.where(sdf_grid.abs()<eps_body, (1+torch.cos(math.pi*sdf_grid/eps_body))/(2*eps_body), torch.zeros_like(sdf_grid))
Fz_eul = float((-p*nz*delta).sum()*h**3)

# --- lagrangian: BUGGY markers (tri_local, no offset) vs FIXED (+local_center)
def lagr_Fz(markers, normals, area):
    ip = RegularGridInterpolator((ax[0],ax[1],ax[2]), p.contiguous(), method="linear")
    pm = ip(torch.tensor(markers[:,0],dtype=DT), torch.tensor(markers[:,1],dtype=DT), torch.tensor(markers[:,2],dtype=DT))
    return float((-pm*torch.tensor(normals[:,2],dtype=DT)*torch.tensor(area,dtype=DT)).sum())

def lagr_force_torque(markers, normals, area, com):
    """Full pressure force (3,) and torque (3,) about com, as forces.py does."""
    ip = RegularGridInterpolator((ax[0],ax[1],ax[2]), p.contiguous(), method="linear")
    m = torch.tensor(markers,dtype=DT); n = torch.tensor(normals,dtype=DT); a = torch.tensor(area,dtype=DT)
    pm = ip(m[:,0], m[:,1], m[:,2])
    f = (-pm)[:,None]*n*a[:,None]                  # (T,3) pressure force per triangle
    F = f.sum(0)
    r = m - torch.tensor(com,dtype=DT)
    tau = torch.cross(r, f, dim=1).sum(0)
    return F, tau

# buggy: BDIMhandler did R@tri_local + body_pos with body_pos = world bbox origin.
# At identity pose, body_pos anchors mesh-coord 0 at world; markers land at
# tri_local (missing local_center) -> off the surface.
Fz_lag_buggy = lagr_Fz(tri_local, nrm, area)
# fixed: tri_local + local_center = true surface
Fz_lag_fixed = lagr_Fz(tri_local + local_center, nrm, area)

# check marker on-surface-ness
def sdf_at(markers):
    s = interp_sdf(torch.tensor(markers[:,0],dtype=DT), torch.tensor(markers[:,1],dtype=DT), torch.tensor(markers[:,2],dtype=DT))
    return float(s.abs().mean())

# ground truth: Archimedes from SDF volume below the waterline
sub = (sdf_grid < 0) & (Z < WL)
Vsub = float(sub.sum()) * h**3
F_arch = RHO_W * G * Vsub
# watertightness of the triangulation
sumAn = (nrm * area[:,None]).sum(0)
print(f"hull: T={len(area)}  local_center={local_center.round(3)}  WL={WL:.4f}  h={h:.4f}")
print(f"  Sum(A*n) = {sumAn.round(5)}  (watertight if ~0)   Sum(A)={area.sum():.4f}")
print(f"  V_submerged (SDF<0 & z<WL) = {Vsub:.5f} m^3 -> Archimedes Fz = {F_arch:+.3f} N")
print(f"  |sdf| at BUGGY markers = {sdf_at(tri_local):.4e}   (off surface)")
print(f"  |sdf| at FIXED markers = {sdf_at(tri_local+local_center):.4e}   (on surface)")
print(f"\n  Fz eulerian band (reference) = {Fz_eul:+10.3f} N")
print(f"  Fz lagrangian BUGGY markers  = {Fz_lag_buggy:+10.3f} N   ({Fz_lag_buggy/Fz_eul:+.2f}x ref)")
print(f"  Fz lagrangian FIXED markers  = {Fz_lag_fixed:+10.3f} N   ({Fz_lag_fixed/Fz_eul:+.2f}x ref)")

# --- TORQUE about the body com (the documented "large pitch torque") --------
com = local_center  # world bbox centre at identity pose (~ hull com proxy)
F_b, tau_b = lagr_force_torque(tri_local, nrm, area, com)              # buggy
F_f, tau_f = lagr_force_torque(tri_local + local_center, nrm, area, com)  # fixed
print(f"\n  pitch torque Ty about com:")
print(f"    BUGGY markers Ty = {float(tau_b[1]):+12.1f} N*m   (spurious moment arm)")
print(f"    FIXED markers Ty = {float(tau_f[1]):+12.1f} N*m")
print(f"    full torque vec BUGGY = {tau_b.numpy().round(1)}")
print(f"    full torque vec FIXED = {tau_f.numpy().round(1)}")
