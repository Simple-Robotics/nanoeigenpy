import nanoeigenpy
import numpy as np
import scipy.sparse as spa

dim = 100
rng = np.random.default_rng()

A_fac = spa.random(dim, dim, density=0.25, random_state=rng)
A = A_fac.T @ A_fac
A += spa.diags(10.0 * rng.standard_normal(dim) ** 2)
A = A.tocsc(True)
A.check_format()

llt = nanoeigenpy.SimplicialLLT(A)

assert llt.info() == nanoeigenpy.ComputationInfo.Success

L = llt.matrixL()
U = llt.matrixU()

LU = L @ U
perm = llt.permutationP().toDenseMatrix()
perm_inv = llt.permutationP().inverse().toDenseMatrix()
A_perm = perm @ A @ perm_inv
assert nanoeigenpy.is_approx(LU.toarray(), A_perm)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = llt.solve(B)
assert isinstance(X_est, np.ndarray)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

llt.analyzePattern(A)
llt.factorize(A)

X_sparse = spa.random(dim, 10, random_state=rng)
B_sparse = A.dot(X_sparse)
B_sparse: spa.csc_matrix = B_sparse.tocsc(True)
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()

X_est = llt.solve(B_sparse)
assert isinstance(X_est, spa.csc_matrix)
assert nanoeigenpy.is_approx(X_est.toarray(), X_sparse.toarray())
assert nanoeigenpy.is_approx(A.dot(X_est.toarray()), B_sparse.toarray())
