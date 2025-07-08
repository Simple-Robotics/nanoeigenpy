import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5

B = rng.random((dim, dim))
B = B @ B.T + 0.1 * np.eye(dim)

gsaes = nanoeigenpy.GeneralizedSelfAdjointEigenSolver(A, B)
assert gsaes.info() == nanoeigenpy.ComputationInfo.Success

V = gsaes.eigenvectors()
D = gsaes.eigenvalues()

for i in range(dim):
    v = V[:, i]
    lam = D[i]

    Av = A @ v
    lam_Bv = lam * (B @ v)

    assert nanoeigenpy.is_approx(Av, lam_Bv)
