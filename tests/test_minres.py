import nanoeigenpy
import numpy as np

dim = 5
seed = 6
rng = np.random.default_rng(seed)

A = np.eye(dim)
minres = nanoeigenpy.MINRES(A)

X = rng.random((dim, 2))
print("X :", X)

B = A.dot(X)
X_est = minres.solve(B)

print("X_est :", X_est)

# assert nanoeigenpy.is_approx(X, X_est, 1e-6)
