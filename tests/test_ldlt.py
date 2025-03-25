import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()


A_neg = -np.eye(dim)
ldlt_neg = nanoeigenpy.LDLT(A_neg)
assert ldlt_neg.isNegative() == True
assert ldlt_neg.isPositive() == False

A_pos = np.eye(dim)
ldlt_pos = nanoeigenpy.LDLT(A_pos)
assert ldlt_pos.isPositive() == True
assert ldlt_pos.isNegative() == False


A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

ldlt = nanoeigenpy.LDLT(A)


L = ldlt.matrixL()
D = ldlt.vectorD()
P = ldlt.transpositionsP()
assert nanoeigenpy.is_approx(np.transpose(P).dot(L.dot(np.diag(D).dot(np.transpose(L).dot(P)))), A)


X = rng.random((dim, 20))
B = A.dot(X)
X_est = ldlt.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = ldlt.solve(b)
# assert nanoeigenpy.is_approx(x, x_est)             # not exposed yet for vectors
# assert nanoeigenpy.is_approx(A.dot(x_est), b)      # not exposed yet for vectors


ldlt1 = nanoeigenpy.LDLT()
ldlt2 = nanoeigenpy.LDLT()

id1 = ldlt1.id()
id2 = ldlt2.id()

assert id1 != id2
assert id1 == ldlt1.id()
assert id2 == ldlt2.id()
