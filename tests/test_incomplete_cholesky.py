import numpy as np
from scipy.sparse import csc_matrix
import nanoeigenpy

dim = 100
rng = np.random.default_rng(42)

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))
A = csc_matrix(A)

ichol = nanoeigenpy.IncompleteCholesky(A)
assert ichol.info() == nanoeigenpy.ComputationInfo.Success
assert ichol.rows() == dim
assert ichol.cols() == dim

X = rng.random((dim, 20))
B = A.dot(X)
X_est = ichol.solve(B)
assert isinstance(X_est, np.ndarray)
assert nanoeigenpy.is_approx(X, X_est)

x = rng.random(dim)
b = A.dot(x)
x_est = ichol.solve(b)
assert isinstance(x_est, np.ndarray)
assert nanoeigenpy.is_approx(x, x_est)

X_sparse = csc_matrix(rng.random((dim, 10)))
B_sparse = A.dot(X_sparse).tocsc()
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()
X_est_sparse = ichol.solve(B_sparse)
assert isinstance(X_est_sparse, csc_matrix)

ichol.analyzePattern(A)
ichol.factorize(A)
assert ichol.info() == nanoeigenpy.ComputationInfo.Success

ichol_shift = nanoeigenpy.IncompleteCholesky()
ichol_shift.setInitialShift(1e-2)
ichol_shift.compute(A)
assert ichol_shift.info() == nanoeigenpy.ComputationInfo.Success

L = ichol.matrixL()
scaling = ichol.scalingS()
perm = ichol.permutationP()
assert isinstance(L, csc_matrix)
assert isinstance(scaling, np.ndarray)
assert L.shape == (dim, dim)
assert scaling.shape == (dim,)
