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

splu = nanoeigenpy.SparseLU(A)

assert splu.info() == nanoeigenpy.ComputationInfo.Success

X = rng.random((dim, 20))
B = A.dot(X)
X_est = splu.solve(B)
assert isinstance(X_est, np.ndarray)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

splu.analyzePattern(A)
splu.factorize(A)

X_sparse = spa.random(dim, 10, random_state=rng)
B_sparse = A.dot(X_sparse)
B_sparse: spa.csc_matrix = B_sparse.tocsc(True)
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()

X_est = splu.solve(B_sparse)
assert isinstance(X_est, spa.csc_matrix)
assert nanoeigenpy.is_approx(X_est.toarray(), X_sparse.toarray())
assert nanoeigenpy.is_approx(A.dot(X_est.toarray()), B_sparse.toarray())

assert splu.nnzL() > 0
assert splu.nnzU() > 0

L = splu.matrixL()
U = splu.matrixU()

assert L.rows() == dim
assert L.cols() == dim
assert U.rows() == dim
assert U.cols() == dim

x_true = rng.random(dim)
b_true = A.dot(x_true)
P_rows_indices = splu.rowsPermutation().indices()
P_cols_indices = splu.colsPermutation().indices()

b_permuted = b_true[P_rows_indices]
z = b_permuted.copy()
L.solveInPlace(z)
y = z.copy()
U.solveInPlace(y)
x_reconstructed = np.zeros(dim)
x_reconstructed[P_cols_indices] = y

assert nanoeigenpy.is_approx(x_reconstructed, x_true, 1e-6)

splu_transpose = splu.transpose()
splu_adjoint = splu.adjoint()

x_true = rng.random(dim)
b_transpose = A.T.dot(x_true)
x_est = splu_transpose.solve(b_transpose)
assert nanoeigenpy.is_approx(x_true, x_est)

x_est_adj = splu_adjoint.solve(b_transpose)
assert nanoeigenpy.is_approx(x_true, x_est_adj)
assert nanoeigenpy.is_approx(x_est, x_est_adj)
