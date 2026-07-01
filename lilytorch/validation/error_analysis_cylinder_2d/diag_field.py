"""Spatial structure of the velocity/pressure error near the body, and a
check of whether the worst error is a thin surface ring (BDIM artifact) or a
localized feature. Also reports max |div(u)| as a steadiness/quality check."""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, R = 1.0, 0.5
half_L = 5.0
xmin = ymin = -half_L
Lx = 2*half_L
BASE = sys.argv[1] if len(sys.argv)>1 else \
    "/data/andreaferrario/ns_data/flow_past_cylinder_error_tests_MW/mu0proj_abdquickest/"
if not BASE.endswith('/'): BASE+='/'
ref = 2048

def grid(nx, dx):
    x = np.linspace(xmin+0.5*dx, xmin+Lx-0.5*dx, nx); return np.meshgrid(x,x,indexing='ij')
def sdf(X,Y): return np.sqrt(X**2+Y**2)-R
def load(Nx,n): return np.load(os.path.join(BASE,f"Nx{Nx}","uv_field",f"{n}.npy"))[1:-1,1:-1]

uR,vR,pR = load(ref,'u'),load(ref,'v'),load(ref,'p')
nxs=[256,512]
fig,axes=plt.subplots(2,len(nxs),figsize=(6*len(nxs),11))
theta=np.linspace(0,2*np.pi,200)
for j,Nx in enumerate(nxs):
    dx=Lx/Nx; di=ref//Nx
    u,v,p=load(Nx,'u'),load(Nx,'v'),load(Nx,'p')
    ur=uR[::di,:].reshape(Nx,Nx,di).mean(2)
    vr=vR.reshape(Nx,di,-1).mean(1)[:,::di]
    pr=pR.reshape(Nx,di,-1).mean(1).reshape(Nx,Nx,di).mean(2)
    X,Y=grid(Nx,dx); s=sdf(X,Y); fluid=s>0
    emag=np.sqrt((u-ur)**2+(v-vr)**2)
    ep=(p-p[fluid].mean())-(pr-pr[fluid].mean())
    emag[~fluid]=np.nan; epm=ep.copy(); epm[~fluid]=np.nan
    # how much of L2^2 comes from the band vs interior?
    band=(s>0)&(s<2*dx); interior=s>2*dx
    e2=np.where(fluid,np.sqrt((u-ur)**2+(v-vr)**2),0)**2
    frac_band=e2[band].sum()/e2[fluid].sum()
    print(f"{os.path.basename(BASE.rstrip('/'))} Nx={Nx}: band cells={band.sum()}, "
          f"fraction of velocity-L2^2 energy in band = {frac_band:.1%}; "
          f"Linf|u| band={np.nanmax(np.abs(emag[band])):.3e} interior={np.nanmax(np.abs(emag[interior])):.3e}")
    for i,(f,ttl,cmap) in enumerate([(emag,"|u| error","hot"),(np.abs(epm),"|p| error","hot")]):
        ax=axes[i,j]
        im=ax.imshow(f.T,origin='lower',extent=[xmin,-xmin,ymin,-ymin],cmap=cmap,
                     norm=matplotlib.colors.LogNorm(vmin=1e-5,vmax=1e-1))
        ax.plot(R*np.cos(theta),R*np.sin(theta),'c-',lw=1)
        ax.plot((R+2*dx)*np.cos(theta),(R+2*dx)*np.sin(theta),'g--',lw=0.8)  # band edge
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.set_title(f"Nx={Nx} {ttl}")
        plt.colorbar(im,ax=ax,shrink=0.8)
out=os.path.join(BASE,"err_field_zoom.png")
fig.suptitle(f"Error fields near body — {os.path.basename(BASE.rstrip('/'))}\n"
             f"cyan=surface, green dashed=BDIM band edge (sdf=2dx)")
fig.tight_layout(); fig.savefig(out,dpi=140,bbox_inches='tight')
print("saved",out)
