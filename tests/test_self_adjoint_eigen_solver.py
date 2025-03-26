import nanoeigenpy
import numpy as np

dim = 5
seed = 1
rng = np.random.default_rng(seed)

A = rng.random((dim, dim))
A = (A + A.T) * 0.5
# ind = np.arange(dim) + np.ones(dim)
# A = np.diag(ind)
print("A :")
print(A)

es = nanoeigenpy.SelfAdjointEigenSolver(A)

assert es.info() == nanoeigenpy.ComputationInfo.Success

V = es.eigenvectors()
print("V")
print(V)
D = es.eigenvalues()
print("D")
print(D)

AdotV = A @ V
VdotD = V @ np.diag(D)

assert nanoeigenpy.is_approx(AdotV, VdotD, 1e-10)
