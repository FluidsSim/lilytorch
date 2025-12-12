import torch

def compute_dpdx(p, dx):
    dpdx = torch.zeros_like(p)
    dpdx[1:-1,:] = (p[2:,:]-p[:-2,:])/(2*dx)
    dpdx[0,:] = (p[1,:]-p[0,:])/(dx)
    dpdx[-1,:] = (p[-1,:]-p[-2,:])/(dx)
    return dpdx

def compute_dpdy(p, dy):
    dpdy = torch.zeros_like(p)
    dpdy[:,1:-1] = (p[:,2:]-p[:,:-2])/(2*dy)
    dpdy[:,0] = (p[:,1]-p[:,0])/(dy)
    dpdy[:,-1] = (p[:,-1]-p[:,-2])/(dy)    
    return dpdy

def divergence(u, v, dx, dy):
    return compute_dpdx(u, dx)+compute_dpdy(v, dy)