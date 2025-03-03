import time
import scipy.sparse.linalg as ssla
import scipy.sparse as ss
from scipy.linalg import solve_banded
import numpy as np

size = 3000
x = np.ones(size)
S = ss.diags([1, -2, 1], [-1, 0, 1], shape=[size, size])
A = S.toarray()
print('cond. nr.:{}'.format(np.linalg.cond(A)))
b = np.dot(A, x)

start_time = time.time()
sol = np.linalg.solve(A, b)
elapsed_time = time.time() - start_time
error = np.sum(np.abs(sol - x))
print('LU time, error:  {}  {}'.format(elapsed_time, error))

start_time = time.time()
sol, info = ssla.bicg(S, b, tol=1e-12)
elapsed_time = time.time() - start_time
error = np.sum(np.abs(sol - x))
print('CG time, ret code, error:  {} {} {}'.format(elapsed_time, info, error))
