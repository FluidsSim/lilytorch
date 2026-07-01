"""FAITHFUL go/no-go GATE for the Nangia et al. (2019) cure: does CONSISTENT
CONSERVATIVE mass/momentum transport stay bounded at 833:1 where the
NON-conservative (our current-style) scheme grows a parasitic current?

Both schemes: same MAC grid, same exact CG projection with c=dt/rho_face, same
first-order upwind (so the ONLY difference is consistency, not the limiter),
same gravity, density SYNCHRONIZED from alpha each step, same wavy interface
that should settle to flat with |u|->0.

  noncons : advect the VELOCITY  (u·∇u)         + rho from alpha   <- our solver
  cons    : carry rho*u, advect with ∇·(rho u u) using the SAME face mass flux
            F=rho_face*u that updates the density (Desjardins-Moureau/Nangia)

Decisive signal: cons settles (|u|->small, stays bounded) while noncons sustains
or grows the parasitic current. If so, the literature cure is worth implementing
in the core. Forward-Euler (the CONSISTENCY is the stabiliser, not the RK order).
"""
import torch
torch.set_default_dtype(torch.float64)

N = 48; h = 1.0 / N; g = -9.81; RW, RA = 1000.0, 1.2          # 833:1
dt = 0.12 * h / (abs(g) * 0.5) ** 0.5
NT = 24000

def rho_cc(a):  return a * RW + (1 - a) * RA

def xf(q):                                                    # cc(N,N)->xface(N+1,N)
    o = torch.zeros(N + 1, N); o[1:N] = 0.5 * (q[:-1] + q[1:]); o[0] = q[0]; o[N] = q[-1]; return o
def yf(q):                                                    # cc(N,N)->yface(N,N+1)
    o = torch.zeros(N, N + 1); o[:, 1:N] = 0.5 * (q[:, :-1] + q[:, 1:]); o[:, 0] = q[:, 0]; o[:, N] = q[:, -1]; return o

def divergence(u, v):  return (u[1:] - u[:-1]) / h + (v[:, 1:] - v[:, :-1]) / h

def project(u, v, r):
    rfx = xf(r); rfy = yf(r)
    cxf = dt / rfx; cyf = dt / rfy
    rhs = divergence(u, v); rhs = rhs - rhs.mean()
    def L(p):
        gx = torch.zeros(N + 1, N); gy = torch.zeros(N, N + 1)
        gx[1:N] = cxf[1:N] * (p[1:] - p[:-1]) / h
        gy[:, 1:N] = cyf[:, 1:N] * (p[:, 1:] - p[:, :-1]) / h
        return (gx[1:] - gx[:-1]) / h + (gy[:, 1:] - gy[:, :-1]) / h
    A = lambda p: -L(p); b = -rhs
    p = torch.zeros(N, N); r = b - A(p); d = r.clone(); rr = (r * r).sum()
    for _ in range(600):
        Ad = A(d); al = rr / (d * Ad).sum().clamp_min(1e-300)
        p += al * d; r -= al * Ad; rr2 = (r * r).sum()
        if rr2 ** 0.5 < 1e-11: break
        d = r + (rr2 / rr) * d; rr = rr2
    u[1:N] -= cxf[1:N] * (p[1:] - p[:-1]) / h
    v[:, 1:N] -= cyf[:, 1:N] * (p[:, 1:] - p[:, :-1]) / h
    return u, v

def adv_alpha(a, u, v):
    Fx = torch.zeros(N + 1, N); Fx[1:N] = u[1:N] * torch.where(u[1:N] >= 0, a[:-1], a[1:])
    Fy = torch.zeros(N, N + 1); Fy[:, 1:N] = v[:, 1:N] * torch.where(v[:, 1:N] >= 0, a[:, :-1], a[:, 1:])
    return (a - dt / h * ((Fx[1:] - Fx[:-1]) + (Fy[:, 1:] - Fy[:, :-1]))).clamp(0, 1)

# ----- NON-conservative velocity advection (our current style) -----
def adv_u_nc(u, v):
    un = u.clone(); ui = u[1:N]
    dudx = torch.where(ui >= 0, (u[1:N] - u[0:N-1]) / h, (u[2:N+1] - u[1:N]) / h)
    vu = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
    dudy = torch.zeros(N-1, N)
    dudy[:, 1:-1] = torch.where(vu[:, 1:-1] >= 0, (u[1:N, 1:-1] - u[1:N, 0:-2]) / h, (u[1:N, 2:] - u[1:N, 1:-1]) / h)
    un[1:N] = u[1:N] - dt * (ui * dudx + vu * dudy); return un
def adv_v_nc(u, v):
    vn = v.clone(); vi = v[:, 1:N]
    dvdy = torch.where(vi >= 0, (v[:, 1:N] - v[:, 0:N-1]) / h, (v[:, 2:N+1] - v[:, 1:N]) / h)
    uv = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    dvdx = torch.zeros(N, N-1)
    dvdx[1:-1] = torch.where(uv[1:-1] >= 0, (v[1:-1, 1:N] - v[0:-2, 1:N]) / h, (v[2:, 1:N] - v[1:-1, 1:N]) / h)
    vn[:, 1:N] = v[:, 1:N] - dt * (uv * dvdx + vi * dvdy); return vn

def step_noncons(u, v, alpha):
    v[:, 1:N] += dt * g                              # gravity on velocity
    u = adv_u_nc(u, v); v = adv_v_nc(u, v)
    alpha = adv_alpha(alpha, u, v)
    u, v = project(u, v, rho_cc(alpha))
    u[0] = u[N] = 0; v[:, 0] = v[:, N] = 0
    return u, v, alpha

# ----- CONSERVATIVE consistent momentum (Desjardins-Moureau / Nangia) -----
# State carried as rho_cc (cell density), NOT alpha. The SAME upwind mass flux
# F=u*rho_upwind drives BOTH the density update and the momentum convection, and
# u is recovered with the face density synced from the SAME flux-evolved cell
# density -> consistent -> bounded.
def step_cons(u, v, r):
    rfx = xf(r); rfy = yf(r)
    # upwind cell mass fluxes (positivity-preserving density transport)
    Fx = torch.zeros(N + 1, N); Fx[1:N] = u[1:N] * torch.where(u[1:N] >= 0, r[:-1], r[1:])
    Fy = torch.zeros(N, N + 1); Fy[:, 1:N] = v[:, 1:N] * torch.where(v[:, 1:N] >= 0, r[:, :-1], r[:, 1:])
    mu = rfx * u; mv = rfy * v                       # momenta on faces

    # momentum-CV mass fluxes = averages of the cell fluxes (discrete consistency)
    Mx_cc = 0.5 * (Fx[:-1] + Fx[1:])                 # (N,N) at cell centres
    My_cor = torch.zeros(N + 1, N + 1); My_cor[1:N] = 0.5 * (Fy[:-1] + Fy[1:])
    My_cc = 0.5 * (Fy[:, :-1] + Fy[:, 1:])           # (N,N) at cell centres
    Mx_cor = torch.zeros(N + 1, N + 1); Mx_cor[:, 1:N] = 0.5 * (Fx[:, :-1] + Fx[:, 1:])

    # x-momentum: d(mu)/dt = -div(rho u u)
    uE = torch.where(Mx_cc >= 0, u[:-1], u[1:]); phix = Mx_cc * uE          # (N,N)
    u_below = torch.cat([u[1:N, :1], u[1:N]], 1)
    u_above = torch.cat([u[1:N], u[1:N, -1:]], 1)
    uN = torch.where(My_cor[1:N] >= 0, u_below, u_above); psix = My_cor[1:N] * uN
    mu[1:N] = mu[1:N] - dt / h * ((phix[1:N] - phix[0:N-1]) + (psix[:, 1:] - psix[:, :-1]))

    # y-momentum + well-balanced gravity
    vN = torch.where(My_cc >= 0, v[:, :-1], v[:, 1:]); phiy = My_cc * vN    # (N,N)
    v_left  = torch.cat([v[:1, 1:N], v[:, 1:N]], 0)
    v_right = torch.cat([v[:, 1:N], v[-1:, 1:N]], 0)
    vE = torch.where(Mx_cor[:, 1:N] >= 0, v_left, v_right); psiy = Mx_cor[:, 1:N] * vE
    mv[:, 1:N] = mv[:, 1:N] - dt / h * ((phiy[:, 1:N] - phiy[:, 0:N-1]) + (psiy[1:, :] - psiy[:-1, :]))
    mv[:, 1:N] = mv[:, 1:N] + dt * g * rfy[:, 1:N]

    # evolve cell density by the SAME upwind flux, recover u with synced face density
    r_new = (r - dt / h * ((Fx[1:] - Fx[:-1]) + (Fy[:, 1:] - Fy[:, :-1]))).clamp_min(RA)
    u = mu / xf(r_new); v = mv / yf(r_new)
    u, v = project(u, v, r_new)
    u[0] = u[N] = 0; v[:, 0] = v[:, N] = 0
    return u, v, r_new

def run(scheme):
    X = ((torch.arange(N) + 0.5) * h)[:, None]; Y = ((torch.arange(N) + 0.5) * h)[None, :]
    alpha = (Y < 0.5 + 0.05 * torch.sin(2 * torch.pi * X)).double()
    u = torch.zeros(N + 1, N); v = torch.zeros(N, N + 1)
    if scheme == "noncons":
        step, state = step_noncons, alpha
    else:
        step, state = step_cons, rho_cc(alpha)       # carry cell density
    print(f"\n=== {scheme}  ratio={RW/RA:.0f}:1  N={N} ===")
    for it in range(NT):
        u, v, state = step(u, v, state)
        if (it + 1) % 500 == 0:
            um = max(u.abs().max().item(), v.abs().max().item())
            print(f"  it={it+1:4d}  |u|max={um:.3e}")
            if um > 50 or not torch.isfinite(u).all():
                print("  DIVERGED"); break

if __name__ == "__main__":
    run("noncons")
    run("cons")
