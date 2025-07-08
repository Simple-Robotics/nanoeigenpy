import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))

realqz = nanoeigenpy.RealQZ(A, B)

Q = realqz.matrixQ()
S = realqz.matrixS()
Z = realqz.matrixZ()
T = realqz.matrixT()

assert nanoeigenpy.is_approx(A, Q @ S @ Z)
assert nanoeigenpy.is_approx(B, Q @ T @ Z)
