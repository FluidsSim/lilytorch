"""Convergence analyzer for any results dir. Usage:
    python analyze_dir.py <BASE_DIR> [ref_Nx]
Reports L2/Linf for |u| and p over fluid (band incl), interior (band excl),
far field; gauge-removed pressure; proper staggered/CC restriction; and the
location of the worst-cell errors."""
import numpy as np, os, sys

D, R = 1.0, 0.5
half_L = 5.0
xmin = ymin = -half_L
Lx = 2*half_L

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/bdimconsistent_abdquickest/"
if not BASE.endswith('/'): BASE += '/'

def make_grid(nx, dx):
    x = np.linspace(xmin + 0.5*dx, xmin + Lx - 0.5*dx, nx)
    return np.meshgrid(x, x, indexing='ij')
def sdf(X, Y): return np.sqrt(X**2 + Y**2) - R
def l2(e): return np.sqrt(np.mean(e**2))
def li(e): return np.max(np.abs(e))
def rate(a,b,ha,hb): return np.log(a/b)/np.log(ha/hb) if (a>0 and b>0) else float('nan')
def load(Nx,n): return np.load(os.path.join(BASE,f"Nx{Nx}","uv_field",f"{n}.npy"))[1:-1,1:-1]

grids = sorted(int(d[2:]) for d in os.listdir(BASE)
               if d.startswith('Nx') and d[2:].isdigit()
               and os.path.exists(os.path.join(BASE,d,"uv_field","p.npy")))
ref = int(sys.argv[2]) if len(sys.argv) > 2 else max(grids)
nxs = [n for n in grids if n != ref]
print(f"BASE={BASE}\nGrids={grids}  reference=Nx{ref}  compared={nxs}\n")

uR, vR, pR = load(ref,'u'), load(ref,'v'), load(ref,'p')
res = {}
for Nx in nxs:
    dx = Lx/Nx; di = ref//Nx
    u, v, p = load(Nx,'u'), load(Nx,'v'), load(Nx,'p')
    ur = uR[::di,:].reshape(Nx,Nx,di).mean(2)
    vr = vR.reshape(Nx,di,-1).mean(1)[:,::di]
    pr = pR.reshape(Nx,di,-1).mean(1).reshape(Nx,Nx,di).mean(2)
    X, Y = make_grid(Nx, dx); s = sdf(X, Y)
    fluid = s > 0; interior = s > 2*dx; far = s >= 2.5
    eu, ev = u-ur, v-vr
    emag = np.sqrt(eu**2+ev**2)
    ep = (p - p[fluid].mean()) - (pr - pr[fluid].mean())
    res[Nx] = dict(dx=dx, s=s, X=X, Y=Y, fluid=fluid, interior=interior, far=far,
                   emag=emag, ep=ep, eu=eu, ev=ev)

def tab(field, name):
    for region, lbl in [('fluid','fluid (band INCL)'),
                        ('interior','interior (band EXCL)'),
                        ('far','far (sdf>=2.5)')]:
        print(f"  {name} -- {lbl}")
        print(f"    {'Nx':>5s} {'dx':>9s} {'D/dx':>6s} {'L2':>11s} {'rate':>5s} {'Linf':>11s} {'rate':>5s}")
        pL2=pLi=None
        for Nx in nxs:
            m = res[Nx][region]; e = res[Nx][field]
            L2v=l2(e[m]); Liv=li(e[m]); dx=res[Nx]['dx']
            r2 = "---" if pL2 is None else f"{rate(pL2[0],L2v,pL2[1],dx):.2f}"
            ri = "---" if pLi is None else f"{rate(pLi[0],Liv,pLi[1],dx):.2f}"
            print(f"    {Nx:5d} {dx:9.5f} {D/dx:6.1f} {L2v:11.4e} {r2:>5s} {Liv:11.4e} {ri:>5s}")
            pL2=(L2v,dx); pLi=(Liv,dx)
        print()

print("="*70); print("VELOCITY |u| error"); print("="*70)
tab('emag', '|u|')
print("="*70); print("PRESSURE error (gauge-removed)"); print("="*70)
tab('ep', 'p')

print("Worst-cell locations (fluid, band incl):")
for fld in ['emag','ep']:
    for Nx in nxs:
        e = np.where(res[Nx]['fluid'], np.abs(res[Nx][fld]), 0.0)
        idx = np.unravel_index(np.argmax(e), e.shape)
        s=res[Nx]['s']; dx=res[Nx]['dx']
        print(f"  {fld} Nx={Nx}: max={e[idx]:.3e} at sdf={s[idx]:+.4f} (={s[idx]/dx:+.2f}dx) "
              f"x={res[Nx]['X'][idx]:+.2f} y={res[Nx]['Y'][idx]:+.2f}")
