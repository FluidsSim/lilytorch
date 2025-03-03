
import torch
import numpy

npoints=2**12+1
h=0.1
h2=h*h
w=0.6 # weight by damping factor

def general_relaxation(m):
    mnew = torch.zeros_like(m)
    mnew[1:-1,1:-1] = m[2:,1:-1]+m[:-2,1:-1]+m[1:-1,2:]+m[1:-1,:-2]
    return mnew

def jacobi_relaxation(v, c, f):
    v_new = general_relaxation(c*v)
    v_new += h2*f
    v_new /= w*(4*c)
    v_new += (1-w)*v
    return v_new

def compute_residual(v, c, f):
    v_new = f-(4*v*c-general_relaxation(c*v))/h2
    return v_new

def apply_neumann(m):
    m[0,1:-1]  = 2*m[1,1:-1]+m[0,2:]+m[0,:-2]
    m[-1,1:-1] = 2*m[-2,1:-1]+m[0,2:]+m[0,:-2]
    m[1:-1,0]  = m[2:,0]+m[:-2,0]+2*m[1:-1,1]
    m[1:-1,-1] = m[2:,-1]+m[:-2,-1]+2*m[1:-1,-2]
    m[0,0]     = 2*m[1,0]+2*m[0,1]
    m[0,-1]    = 2*m[1,-1]+2*m[0,-2]
    m[-1,0]    = 2*m[-2,0]+2*m[0,1]
    m[-1,-1]   = 2*m[-2,-1]+2*m[-1,-2]

def prolong(e2h):
    """
    prolong e2h in eh
    """
    n, m = e2h.shape

    nh = int(2*(n-1))
    mh = int(2*(m-1))

    eh = torch.zeros(nh+1,mh+1)

    i1 = 2*numpy.arange(1,int(nh/2))
    i2 = 2*numpy.arange(0,int(nh/2))+1
    j1 = 2*numpy.arange(1,int(mh/2))
    j2 = 2*numpy.arange(0,int(mh/2))+1

    i1j1 = numpy.ix_(i1,j1)
    i2j1 = numpy.ix_(i2,j1)
    i1j2 = numpy.ix_(i1,j2)
    i2j2 = numpy.ix_(i2,j2)

    eh[i1j1] = e2h[1:-1,1:-1]
    eh[i2j1] = 0.5*(e2h[1:,1:-1]+e2h[:-1,1:-1])
    eh[i1j2] = 0.5*(e2h[1:-1,1:]+e2h[1:-1,:-1])
    eh[i2j2] = 0.25*(e2h[1:,1:]+e2h[1:,:-1]+e2h[:-1,1:]+e2h[:-1,:-1])

    return eh

def restrict(r):
    """
    restrict residual from rh to r2h
    """
    n = r.shape[0]-1
    m = r.shape[1]-1

    n2h = int(n/2+1)
    m2h = int(m/2+1)
    r2h = torch.zeros(n2h,m2h)
    
    i  = numpy.arange(1,int(n/2))
    i1 = 2*i-1
    i2 = 2*i
    i3 = 2*i+1

    j  = numpy.arange(1,int(m/2))
    j1 = 2*j-1
    j2 = 2*j
    j3 = 2*j+1

    i1j1 = numpy.ix_(i1,j1)
    i1j3 = numpy.ix_(i1,j3)
    i3j1 = numpy.ix_(i3,j1)
    i3j3 = numpy.ix_(i3,j3)

    i2j1 = numpy.ix_(i2,j1)
    i2j3 = numpy.ix_(i2,j3)
    i1j2 = numpy.ix_(i1,j2)
    i3j2 = numpy.ix_(i3,j2)

    i2j2 = numpy.ix_(i2,j2)

    r2h[1:-1, 1:-1] = 0.0625*(r[i1j1]+r[i1j3]+r[i3j1]+r[i3j3])+\
                      0.125*(r[i2j1]+r[i2j3]+r[i1j2]+r[i3j2])+\
                      0.25*r[i2j2]

    return r2h

p1=30
p2=30
x=30

def vcycle(v, c, f):
    
    # step 0 ============ Jacobi relaxation ============
    npoints_local = int(v.shape[0]-1)
    
    if (npoints_local == 2):
        for _ in range(x):
            v_new = jacobi_relaxation(v, c, f)
            v     = v_new
        return v

    else:
        # step 1 ============ Jacobi relaxation ============ 
        for _ in range(p1):
            v_new = jacobi_relaxation(v, c, f)
            v     = v_new


        # step 2 ============ Compute residual and restrict it to 2h ============ 
        r   = compute_residual(v, c, f)

        print("Steps: {}, Residual: {}".format(npoints_local, r))

        # r2h = restrict(r)

        # # step 3 ============ Residual relaxation ============ 
        # c2h = restrict(c) # compute coefficient matrix on the coarse grid
        # e2h = torch.zeros_like(c2h) # initialize error to zero
        # e2h_new = vcycle(e2h, c2h, r2h) # relax to find the error of the residuals

        # # step 4 ============ Prolongare error and corret v ============ 
        # v += prolong(e2h_new)

        # step 5 ============ Relax more times ============ 
        for _ in range(p1):
            v_new = jacobi_relaxation(v, c, f)
            v     = v_new

        print(v)

        return v


def test_restriction_prolongation():

    # original = torch.rand((2**2+1,2**2+1))
    original = torch.tensor(
        [
            [0,0,0,0,0],
            [0,1,2,3,0],
            [0,4,5,6,0],
            [0,7,8,9,0],
            [0,0,0,0,0]
        ]
    )
    print(original)
    restricted = restrict(original)
    print(restricted)
    prolonged = prolong(restricted)
    print(prolonged)


if __name__ == "__main__":

    # v=torch.rand((npoints,npoints))
    # f=torch.rand((npoints,npoints))
    # c=torch.rand((npoints,npoints))

    x=torch.linspace(0,1,npoints)
    y=torch.linspace(0,1,npoints)
    [X,Y]=torch.meshgrid(x,y, indexing="xy")
    c=torch.ones((npoints,npoints))
    f=-torch.exp(-(X-0.25)**2-(Y-0.6)**2)
    v=torch.rand((npoints,npoints))

    vcycle(v, c, f)


