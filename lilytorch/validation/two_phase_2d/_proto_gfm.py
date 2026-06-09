"""GFM go/no-go GATE: does a sub-cell theta-weighted (ghost-fluid) face density
reduce the moving-interface parasitic current vs the current HARMONIC face
density?  Standalone 2D MAC, gravity, CG projection, upwind advection, wavy
interface (so the interface moves and sits OFF-grid -> sub-cell theta matters).

  harmonic : (1/rho)_face = 0.5(1/rho_i + 1/rho_j)         <- current solver
  gfm      : (1/rho)_face = 1/(theta*rho_i + (1-theta)*rho_j)  <- sub-cell GFM
             (theta = sub-cell interface location from alpha)

If gfm keeps |u| much smaller than harmonic on the wavy-interface settle, the
GFM density is worth implementing in the core. If not, GFM is not the cure.
"""
import torch
torch.set_default_dtype(torch.float64)

N = 48; h = 1.0 / N; g = -9.81; RW, RA = 1000.0, 1.0
dt = 0.2 * h / (abs(g) * 0.5) ** 0.5
NT = 6000

def rho_cc(a):       return a * RW + (1 - a) * RA
def beta_cc(a):      return 1.0 / rho_cc(a)            # 1/rho

def beta_face(a, d, mode):
    """Reciprocal-density on the d-face (N+1,N)/(N,N+1)."""
    nd = a.ndim
    lo = a.index_select(d, torch.arange(0, a.shape[d]-1))
    hi = a.index_select(d, torch.arange(1, a.shape[d]))
    rlo, rhi = rho_cc(lo), rho_cc(hi)
    if mode == "harmonic":
        bf = 0.5 * (1.0/rlo + 1.0/rhi)
    else:  # gfm: theta-weighted; theta = sub-cell interface fraction from lo->hi
        denom = (hi - lo)
        theta = torch.where(denom.abs() > 1e-12, (0.5 - lo) / denom,
                            torch.zeros_like(denom)).clamp(0.0, 1.0)
        # interface only where alpha crosses 0.5 between the two cells
        cross = ((lo - 0.5) * (hi - 0.5)) < 0
        rho_th = theta * rlo + (1.0 - theta) * rhi          # theta-weighted arithmetic
        bf = torch.where(cross, 1.0 / rho_th, 0.5 * (1.0/rlo + 1.0/rhi))
    # pad to face grid (boundary face = adjacent cell)
    out = torch.zeros([s + (1 if i == d else 0) for i, s in enumerate(a.shape)])
    sl_in = tuple(slice(1, -1) if i == d else slice(None) for i in range(nd))
    # place interior faces
    idx = [slice(None)] * nd; idx[d] = slice(1, a.shape[d]);
    out[tuple(idx)] = bf
    idx[d] = slice(0, 1); out[tuple(idx)] = (1.0/rho_cc(a.index_select(d, torch.tensor([0]))))
    return out

# --- staggered fields, walls explicit ---
def xface_avg_beta(a, mode): return beta_face(a, 0, mode)   # (N+1,N)
def yface_avg_beta(a, mode): return beta_face(a, 1, mode)   # (N,N+1)

def divergence(u, v):
    return (u[1:] - u[:-1]) / h + (v[:, 1:] - v[:, :-1]) / h

def project(u, v, alpha, mode):
    cxf = dt * xface_avg_beta(alpha, mode)     # dt/rho on x-faces
    cyf = dt * yface_avg_beta(alpha, mode)
    rhs = divergence(u, v); rhs = rhs - rhs.mean()
    def L(p):
        gx = torch.zeros(N+1, N); gy = torch.zeros(N, N+1)
        gx[1:N] = cxf[1:N] * (p[1:] - p[:-1]) / h
        gy[:, 1:N] = cyf[:, 1:N] * (p[:, 1:] - p[:, :-1]) / h
        return (gx[1:] - gx[:-1])/h + (gy[:, 1:] - gy[:, :-1])/h
    A = lambda p: -L(p); b = -rhs
    p = torch.zeros(N, N); r = b - A(p); d = r.clone(); rr = (r*r).sum()
    for _ in range(800):
        Ad = A(d); al = rr / (d*Ad).sum().clamp_min(1e-300)
        p += al*d; r -= al*Ad; rr2 = (r*r).sum()
        if rr2**0.5 < 1e-11: break
        d = r + (rr2/rr)*d; rr = rr2
    u[1:N] -= cxf[1:N] * (p[1:] - p[:-1]) / h
    v[:, 1:N] -= cyf[:, 1:N] * (p[:, 1:] - p[:, :-1]) / h
    return u, v

def adv_alpha(a, u, v):
    Fx = torch.zeros(N+1, N); Fx[1:N] = u[1:N]*torch.where(u[1:N] >= 0, a[:-1], a[1:])
    Fy = torch.zeros(N, N+1); Fy[:, 1:N] = v[:, 1:N]*torch.where(v[:, 1:N] >= 0, a[:, :-1], a[:, 1:])
    return (a - dt/h*((Fx[1:]-Fx[:-1]) + (Fy[:, 1:]-Fy[:, :-1]))).clamp(0, 1)

def adv_u(u, v):
    un = u.clone(); ui = u[1:N]
    dudx = torch.where(ui >= 0, (u[1:N]-u[0:N-1])/h, (u[2:N+1]-u[1:N])/h)
    vu = 0.25*(v[:-1, :-1]+v[1:, :-1]+v[:-1, 1:]+v[1:, 1:])
    dudy = torch.zeros(N-1, N)
    dudy[:, 1:-1] = torch.where(vu[:, 1:-1] >= 0, (u[1:N, 1:-1]-u[1:N, 0:-2])/h, (u[1:N, 2:]-u[1:N, 1:-1])/h)
    un[1:N] = u[1:N] - dt*(ui*dudx + vu*dudy); return un

def adv_v(u, v):
    vn = v.clone(); vi = v[:, 1:N]
    dvdy = torch.where(vi >= 0, (v[:, 1:N]-v[:, 0:N-1])/h, (v[:, 2:N+1]-v[:, 1:N])/h)
    uv = 0.25*(u[:-1, :-1]+u[:-1, 1:]+u[1:, :-1]+u[1:, 1:])
    dvdx = torch.zeros(N, N-1)
    dvdx[1:-1] = torch.where(uv[1:-1] >= 0, (v[1:-1, 1:N]-v[0:-2, 1:N])/h, (v[2:, 1:N]-v[1:-1, 1:N])/h)
    vn[:, 1:N] = v[:, 1:N] - dt*(uv*dvdx + vi*dvdy); return vn

def run(mode):
    X = ((torch.arange(N)+0.5)*h)[:, None]; Y = ((torch.arange(N)+0.5)*h)[None, :]
    alpha = (Y < 0.5 + 0.05*torch.sin(2*torch.pi*X)).double()   # wavy interface
    u = torch.zeros(N+1, N); v = torch.zeros(N, N+1)
    print(f"\n=== density={mode}  N={N} ===")
    for it in range(NT):
        v[:, 1:N] += dt*g
        u = adv_u(u, v); v = adv_v(u, v); alpha = adv_alpha(alpha, u, v)
        u, v = project(u, v, alpha, mode)
        u[0] = u[N] = 0; v[:, 0] = v[:, N] = 0
        if (it+1) % 500 == 0:
            um = max(u.abs().max().item(), v.abs().max().item())
            print(f"  it={it+1:4d}  |u|max={um:.3e}")
            if um > 50 or not torch.isfinite(u).all():
                print("  DIVERGED"); break

if __name__ == "__main__":
    run("harmonic")
    run("gfm")
