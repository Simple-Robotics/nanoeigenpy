import nanoeigenpy
import numpy as np

dim = 5
seed = 1
rng = np.random.default_rng(seed)

A = rng.random((dim, dim))
A = (A + A.T) * 0.5

es = nanoeigenpy.SelfAdjointEigenSolver(A)

assert es.info() == nanoeigenpy.ComputationInfo.Success

V = es.eigenvectors()
D = es.eigenvalues()

AdotV = A @ V
VdotD = V @ np.diag(D)

assert nanoeigenpy.is_approx(AdotV, VdotD, 1e-10)
