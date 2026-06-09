"""STANDALONE proof-of-concept: consistent mass-momentum transport for the
two-phase hydrostatic parasitic-current problem.  NO codebase edits.

A minimal 2-D MAC (staggered) variable-density solver.  Water below y=0.5, real
air above (1000:1), closed box, gravity.  The exact discrete equilibrium is u=0.
The codebase grows a parasitic current to ~0.2 once the interface moves; the
frozen-VOF test showed that is the only failure (the static balance + projection
are fine).  Hypothesis: the growth is mass-momentum advection INCONSISTENCY at
the density jump.  Two modes, identical otherwise:

  inconsistent : advect velocity u non-conservatively + advect alpha separately
                 (what the codebase effectively does).
  consistent   : advect MOMENTUM rho*u with the SAME face mass fluxes as the
                 mass/alpha, then recover u = (rho u)/rho  (Rudman 1998 idea).

If 'consistent' stays bounded while 'inconsistent' grows -> the fix is confirmed.
"""
import torch
torch.set_default_dtype(torch.float64)

N   = 48
h   = 1.0 / N
g   = -9.81
RW, RA = 1000.0, 1.0
dt  = 0.2 * h / (abs(g) * 0.5) ** 0.5
NT  = 6000

# ---- staggered fields (no ghosts; walls handled explicitly) --------------
#   alpha,p : (N,N) cell-centred       u : (N+1,N) x-faces      v : (N,N+1) y-faces
def cc_rho(alpha):
    return alpha * RW + (1.0 - alpha) * RA

def xface_avg(c):                       # cell -> x-face (N+1,N), one-sided at walls
    f = torch.zeros(N + 1, N)
    f[1:N] = 0.5 * (c[:-1] + c[1:])
    f[0], f[N] = c[0], c[-1]
    return f

def yface_avg(c):                       # cell -> y-face (N,N+1)
    f = torch.zeros(N, N + 1)
    f[:, 1:N] = 0.5 * (c[:, :-1] + c[:, 1:])
    f[:, 0], f[:, N] = c[:, 0], c[:, -1]
    return f

def divergence(u, v):
    return (u[1:] - u[:-1]) / h + (v[:, 1:] - v[:, :-1]) / h

# ---- variable-density pressure projection (CG, Neumann closed box) --------
def applyL(p, cxf, cyf):
    # L p = div( c grad p ), c on faces; zero flux (Neumann) at walls.
    gx = torch.zeros(N + 1, N); gy = torch.zeros(N, N + 1)
    gx[1:N] = cxf[1:N] * (p[1:] - p[:-1]) / h     # interior x-faces
    gy[:, 1:N] = cyf[:, 1:N] * (p[:, 1:] - p[:, :-1]) / h
    return (gx[1:] - gx[:-1]) / h + (gy[:, 1:] - gy[:, :-1]) / h

def project(u, v, rho):
    cxf = dt / xface_avg(rho); cyf = dt / yface_avg(rho)
    rhs = divergence(u, v)
    rhs = rhs - rhs.mean()                          # compatibility (Neumann)
    # A = -div(c grad)  is SPD; solve A p = -rhs.
    A = lambda p: -applyL(p, cxf, cyf)
    b = -rhs
    p = torch.zeros(N, N)
    r = b - A(p); d = r.clone(); rr = (r * r).sum()
    for _ in range(600):
        Ad = A(d)
        al = rr / (d * Ad).sum().clamp_min(1e-300)
        p += al * d; r -= al * Ad
        rr2 = (r * r).sum()
        if rr2 ** 0.5 < 1e-11: break
        d = r + (rr2 / rr) * d; rr = rr2
    u[1:N] -= cxf[1:N] * (p[1:] - p[:-1]) / h
    v[:, 1:N] -= cyf[:, 1:N] * (p[:, 1:] - p[:, :-1]) / h
    return u, v

# ---- first-order upwind helpers ------------------------------------------
def adv_alpha(alpha, u, v):
    # conservative upwind volume transport of alpha (cell-centred).
    Fx = torch.zeros(N + 1, N)
    aup = torch.where(u[1:N] >= 0, alpha[:-1], alpha[1:])
    Fx[1:N] = u[1:N] * aup
    Fy = torch.zeros(N, N + 1)
    aup = torch.where(v[:, 1:N] >= 0, alpha[:, :-1], alpha[:, 1:])
    Fy[:, 1:N] = v[:, 1:N] * aup
    return alpha - dt / h * ((Fx[1:] - Fx[:-1]) + (Fy[:, 1:] - Fy[:, :-1]))

def adv_cell_conservative(q, u, v):
    # conservative upwind transport of a cell-centred quantity q with face
    # velocities u,v (same fluxes used for mass and momentum -> consistency).
    Fx = torch.zeros(N + 1, N)
    qup = torch.where(u[1:N] >= 0, q[:-1], q[1:]); Fx[1:N] = u[1:N] * qup
    Fy = torch.zeros(N, N + 1)
    qup = torch.where(v[:, 1:N] >= 0, q[:, :-1], q[:, 1:]); Fy[:, 1:N] = v[:, 1:N] * qup
    return q - dt / h * ((Fx[1:] - Fx[:-1]) + (Fy[:, 1:] - Fy[:, :-1]))

def adv_u_nonconservative(u, v):
    # non-conservative velocity self-advection (u . grad)u, first-order upwind.
    un = u.clone()
    # x-convection on interior x-faces 1..N-1
    ui = u[1:N]
    dudx_back = (u[1:N] - u[0:N-1]) / h
    dudx_fwd  = (u[2:N+1] - u[1:N]) / h
    dudx = torch.where(ui >= 0, dudx_back, dudx_fwd)
    # v interpolated to x-faces (interior)
    v_at_u = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])  # (N-1,N)
    dudy = torch.zeros(N - 1, N)
    dudy[:, 1:-1] = torch.where(v_at_u[:, 1:-1] >= 0,
                                (u[1:N, 1:-1] - u[1:N, 0:-2]) / h,
                                (u[1:N, 2:] - u[1:N, 1:-1]) / h)
    un[1:N] = u[1:N] - dt * (ui * dudx + v_at_u * dudy)
    return un

def adv_v_nonconservative(u, v):
    vn = v.clone()
    vi = v[:, 1:N]
    dvdy_back = (v[:, 1:N] - v[:, 0:N-1]) / h
    dvdy_fwd  = (v[:, 2:N+1] - v[:, 1:N]) / h
    dvdy = torch.where(vi >= 0, dvdy_back, dvdy_fwd)
    u_at_v = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])  # (N,N-1)
    dvdx = torch.zeros(N, N - 1)
    dvdx[1:-1] = torch.where(u_at_v[1:-1] >= 0,
                             (v[1:-1, 1:N] - v[0:-2, 1:N]) / h,
                             (v[2:, 1:N] - v[1:-1, 1:N]) / h)
    vn[:, 1:N] = v[:, 1:N] - dt * (u_at_v * dvdx + vi * dvdy)
    return vn

# ---- interface-band velocity damping (pragmatic parasitic-current cure) ---
def interface_damp(u, v, alpha, nu):
    # artificial viscosity localized to the interface band via w = 4 a(1-a)
    # (peaks at alpha=0.5, ~0 in each bulk phase) -> damps the parasitic
    # currents at the density jump WITHOUT touching the bulk flow.
    w = (4.0 * alpha * (1.0 - alpha)).clamp(0.0, 1.0)
    wxf, wyf = xface_avg(w), yface_avg(w)
    lap = torch.zeros_like(u)
    lap[1:N, :] += (u[2:N+1, :] - 2*u[1:N, :] + u[0:N-1, :]) / h**2
    lap[:, 1:-1] += (u[:, 2:] - 2*u[:, 1:-1] + u[:, 0:-2]) / h**2
    u = u + dt * nu * wxf * lap
    lap = torch.zeros_like(v)
    lap[:, 1:N] += (v[:, 2:N+1] - 2*v[:, 1:N] + v[:, 0:N-1]) / h**2
    lap[1:-1, :] += (v[2:, :] - 2*v[1:-1, :] + v[0:-2, :]) / h**2
    v = v + dt * nu * wyf * lap
    return u, v

# ---- one step -------------------------------------------------------------
def step(alpha, u, v, mode):
    rho = cc_rho(alpha)
    v[:, 1:N] += dt * g                                   # gravity (per-mass)
    if mode == "static":                                  # gravity+projection only
        u, v = project(u, v, rho)
        u[0] = u[N] = 0.0; v[:, 0] = v[:, N] = 0.0
        return alpha, u, v
    if mode in ("inconsistent", "inc_damp"):
        u = adv_u_nonconservative(u, v)
        v = adv_v_nonconservative(u, v)
        alpha = adv_alpha(alpha, u, v)
        if mode == "inc_damp":
            u, v = interface_damp(u, v, alpha.clamp(0, 1), nu=0.02)
    elif mode == "semi":
        # CONTROL: identical cell<->face interpolation as 'consistent', but
        # advect VELOCITY (not rho*velocity) — no density weighting. If this
        # grows like 'inconsistent', the cure is the mass-momentum CONSISTENCY,
        # not the interpolation smoothing.
        ucc = 0.5 * (u[:-1] + u[1:]); vcc = 0.5 * (v[:, :-1] + v[:, 1:])
        ucc_n = adv_cell_conservative(ucc, u, v)
        vcc_n = adv_cell_conservative(vcc, u, v)
        u = u.clone(); v = v.clone()
        u[1:N] = 0.5 * (ucc_n[:-1] + ucc_n[1:])
        v[:, 1:N] = 0.5 * (vcc_n[:, :-1] + vcc_n[:, 1:])
        alpha = adv_alpha(alpha, u, v)
    else:  # consistent: advect rho*u, rho*v and alpha with the SAME face fluxes
        rxf, ryf = xface_avg(rho), yface_avg(rho)
        # momentum carried at faces as rho_face*vel; transport with cell-face
        # velocities u,v (conservative upwind), recover vel = mom / rho_face_new.
        ru = rxf * u; rv = ryf * v
        # transport ru on the x-face grid using a cell-conservative sweep is
        # awkward on staggered indices; use the cell-centred consistent route:
        # advect rho (mass) and the cell-centred momentum components, recover
        # face velocities by interpolation.  Cell-centred velocities:
        ucc = 0.5 * (u[:-1] + u[1:]); vcc = 0.5 * (v[:, :-1] + v[:, 1:])
        rho_n   = adv_cell_conservative(rho,        u, v)
        rhou_n  = adv_cell_conservative(rho * ucc,  u, v)
        rhov_n  = adv_cell_conservative(rho * vcc,  u, v)
        ucc_n = rhou_n / rho_n; vcc_n = vcc_to = rhov_n / rho_n
        # back to faces (interior); walls stay 0
        u = u.clone(); v = v.clone()
        u[1:N] = 0.5 * (ucc_n[:-1] + ucc_n[1:])
        v[:, 1:N] = 0.5 * (vcc_n[:, :-1] + vcc_n[:, 1:])
        alpha = adv_alpha(alpha, u, v)
    alpha = alpha.clamp(0.0, 1.0)
    rho = cc_rho(alpha)
    u, v = project(u, v, rho)
    u[0] = u[N] = 0.0; v[:, 0] = v[:, N] = 0.0
    return alpha, u, v

def run(mode):
    X = ((torch.arange(N) + 0.5) * h)[:, None]
    Y = ((torch.arange(N) + 0.5) * h)[None, :]
    # WAVY interface: water below 0.5 + A sin(2 pi x).  Under gravity it tries
    # to flatten / oscillate -> the interface MOVES, exercising the advection.
    iface = 0.5 + 0.05 * torch.sin(2 * torch.pi * X)
    alpha = (Y < iface).double()
    u = torch.zeros(N + 1, N); v = torch.zeros(N, N + 1)
    print(f"\n=== mode={mode}  N={N} dt={dt:.3e} ===")
    for it in range(NT):
        alpha, u, v = step(alpha, u, v, mode)
        if (it + 1) % 500 == 0:
            um = max(u.abs().max().item(), v.abs().max().item())
            print(f"  it={it+1:4d}  |u|max={um:.3e}")
            if um > 50 or not torch.isfinite(u).all():
                print("  DIVERGED"); break

if __name__ == "__main__":
    run("inconsistent")
    run("semi")
    run("consistent")
