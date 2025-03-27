import nanoeigenpy
import numpy as np

dim = 100
seed = 6
rng = np.random.default_rng(seed)

A = np.eye(dim)
minres = nanoeigenpy.MINRES(A)

# Vector rhs

x = rng.random(dim)
b = A.dot(x)
x_est = minres.solve(b)

assert nanoeigenpy.is_approx(x, x_est, 1e-6)
assert nanoeigenpy.is_approx(b, A.dot(x_est), 1e-6)

# Matrix rhs

X = rng.random((dim, 20))
B = A.dot(X)
X_est = minres.solve(B)

print("X_est :", X_est)

assert nanoeigenpy.is_approx(X, X_est, 1e-6)
assert nanoeigenpy.is_approx(B, A.dot(X_est), 1e-6)
