import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

es = nanoeigenpy.EigenSolver(A)

V = es.eigenvectors()
D = es.eigenvalues()

assert nanoeigenpy.is_approx(A.dot(V).real, V.dot(np.diag(D)).real)
assert nanoeigenpy.is_approx(A.dot(V).imag, V.dot(np.diag(D)).imag)
