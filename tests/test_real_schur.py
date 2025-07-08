import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

rs = nanoeigenpy.RealSchur(A)

U = rs.matrixU()
T = rs.matrixT()

assert nanoeigenpy.is_approx(A.real, (U @ T @ U.T).real)
