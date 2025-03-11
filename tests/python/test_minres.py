import nanoeigenpy
import numpy as np

dim = 2
seed = 1
rng = np.random.default_rng(seed)

A = np.eye(dim)
minres = nanoeigenpy.MINRES(A)


X = rng.random((dim, 2))
B = A.dot(X)

# X_est = minres.solve(B)
# print("X :", X)
# print("X_est :", X_est)
# print("A.dot(X_est) :", A.dot(X_est))
# print("B :", B)
# # assert nanoeigenpy.is_approx(X, X_est, 1e-6)
# # assert nanoeigenpy.is_approx(A.dot(X_est), B, 1e-6)

# x = rng.random(dim)
# b = A.dot(x)
# x_est = minres.solve(b)
# # assert nanoeigenpy.is_approx(x, x_est)             # not exposed yet for vectors
# # assert nanoeigenpy.is_approx(A.dot(x_est), b)      # not exposed yet for vectors


# ldlt1 = nanoeigenpy.MINRES()
# ldlt2 = nanoeigenpy.MINRES()

# id1 = ldlt1.id()
# id2 = ldlt2.id()

# assert id1 != id2
# assert id1 == ldlt1.id()
# assert id2 == ldlt2.id()