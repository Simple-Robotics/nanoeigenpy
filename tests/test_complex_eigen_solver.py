import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
# A_imag = rng.random((dim, dim))
# A = A_real + 1j * A_imag

# Tests init
ces = nanoeigenpy.ComplexEigenSolver()
es = nanoeigenpy.ComplexEigenSolver(dim)
es = nanoeigenpy.ComplexEigenSolver(A)
assert es.info() == nanoeigenpy.ComputationInfo.Success

# Test eigenvectors
# Test eigenvalues
V = es.eigenvectors()
D = es.eigenvalues()

assert nanoeigenpy.is_approx(A.dot(V).real, V.dot(np.diag(D)).real)
assert nanoeigenpy.is_approx(A.dot(V).imag, V.dot(np.diag(D)).imag)

# Test nb::init<>()
# Test id
ces1 = nanoeigenpy.ComplexEigenSolver()
ces2 = nanoeigenpy.ComplexEigenSolver()

id1 = ces1.id()
id2 = ces2.id()

assert id1 != id2
assert id1 == ces1.id()
assert id2 == ces2.id()

# Test nb::init<Eigen::DenseIndex>()
# Test id
dim_constructor = 3

ces3 = nanoeigenpy.ComplexEigenSolver(dim_constructor)
ces4 = nanoeigenpy.ComplexEigenSolver(dim_constructor)

id3 = ces3.id()
id4 = ces4.id()

assert id3 != id4
assert id3 == ces3.id()
assert id4 == ces4.id()
