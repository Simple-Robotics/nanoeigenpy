import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

es = nanoeigenpy.ComplexEigenSolver(A)
assert es.info() == nanoeigenpy.ComputationInfo.Success

V = es.eigenvectors()
D = es.eigenvalues()
assert V.shape == (dim, dim)
assert D.shape == (dim,)

AV = A @ V
VD = V @ np.diag(D)
assert nanoeigenpy.is_approx(AV.real, VD.real)
assert nanoeigenpy.is_approx(AV.imag, VD.imag)

trace_A = np.trace(A)
trace_D = np.sum(D)
assert abs(trace_A - trace_D.real) < 1e-10
assert abs(trace_D.imag) < 1e-10

ces5 = nanoeigenpy.ComplexEigenSolver(A)
ces6 = nanoeigenpy.ComplexEigenSolver(A)
id5 = ces5.id()
id6 = ces6.id()
assert id5 != id6
assert id5 == ces5.id()
assert id6 == ces6.id()
