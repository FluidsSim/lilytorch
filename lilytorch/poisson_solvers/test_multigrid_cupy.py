



from numpy import arange, array
from pyamg import solve
from pyamg.gallery import poisson 
n = 100 
A = poisson((n,n),format='csr')
b = array(arange(A.shape[0]), dtype=float) 
x = solve(A,b,verb=True,tol=1e-8)