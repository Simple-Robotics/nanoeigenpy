import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

llt = nanoeigenpy.LLT(A)

assert llt.info() == nanoeigenpy.ComputationInfo.Success

L = llt.matrixL()
assert nanoeigenpy.is_approx(L.dot(np.transpose(L)), A)

U = llt.matrixU()
LU = L @ U
assert nanoeigenpy.is_approx(LU, A)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = llt.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = llt.solve(b)
assert nanoeigenpy.is_approx(x, x_est)
assert nanoeigenpy.is_approx(A.dot(x_est), b)

LLT = llt.matrixLLT()
LLT_lower = np.tril(LLT)
assert nanoeigenpy.is_approx(LLT_lower, L)

A_upper = np.triu(A, k=1)
LLT_upper = np.triu(LLT, k=1)
assert nanoeigenpy.is_approx(A_upper, LLT_upper)

A_reconstructed = llt.reconstructedMatrix()
assert nanoeigenpy.is_approx(A_reconstructed, A)

adjoint = llt.adjoint()
assert adjoint is llt

A_cond = np.eye(dim)
llt_cond = nanoeigenpy.LLT(A_cond)
estimated_r_cond_num = llt_cond.rcond()
assert abs(estimated_r_cond_num - 1) <= 1e-9

sigma = 3
w = np.ones(dim)
llt.rankUpdate(w, sigma)
L = llt.matrixL()
U = llt.matrixU()
LU = L @ U
assert nanoeigenpy.is_approx(LU, A + sigma * w * np.transpose(w))

llt1 = nanoeigenpy.LLT()
llt2 = nanoeigenpy.LLT()

id1 = llt1.id()
id2 = llt2.id()

assert id1 != id2
assert id1 == llt1.id()
assert id2 == llt2.id()

dim_constructor = 3

llt3 = nanoeigenpy.LLT(dim_constructor)
llt4 = nanoeigenpy.LLT(dim_constructor)

id3 = llt3.id()
id4 = llt4.id()

assert id3 != id4
assert id3 == llt3.id()
assert id4 == llt4.id()
