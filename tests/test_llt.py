import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

llt = nanoeigenpy.LLT(A)


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
# assert nanoeigenpy.is_approx(x, x_est)             # not exposed yet for vectors
# assert nanoeigenpy.is_approx(A.dot(x_est), b)      # not exposed yet for vectors


ldlt1 = nanoeigenpy.LLT()
ldlt2 = nanoeigenpy.LLT()

id1 = ldlt1.id()
id2 = ldlt2.id()

assert id1 != id2
assert id1 == ldlt1.id()
assert id2 == ldlt2.id()
