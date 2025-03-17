import nanoeigenpy
import numpy as np

dim = 4
seed = 1
rng = np.random.default_rng(seed)

A = rng.random((dim, dim))
print("A :")
print(A)

# Tests init
es = nanoeigenpy.EigenSolver()
#assert es.info() == nanoeigenpy.ComputationInfo.Success
es = nanoeigenpy.EigenSolver(dim)
#assert es.info() == nanoeigenpy.ComputationInfo.Success
es = nanoeigenpy.EigenSolver(A)
#assert es.info() == nanoeigenpy.ComputationInfo.Success

# Test eigenvectors
# Test eigenvalues
V = es.eigenvectors()
D = es.eigenvalues()

assert nanoeigenpy.is_approx(A.dot(V).real, V.dot(np.diag(D)).real)
assert nanoeigenpy.is_approx(A.dot(V).imag, V.dot(np.diag(D)).imag)

# Test compute
# Done implicitly at init

# Test pseudoEigenvalueMatrix
print("es.pseudoEigenvalueMatrix()")
print(es.pseudoEigenvalueMatrix())

# Test nb::init<>()
# Test id
es1 = nanoeigenpy.EigenSolver()
es2 = nanoeigenpy.EigenSolver()

id1 = es1.id()
id2 = es2.id()

assert id1 != id2
assert id1 == es1.id()
assert id2 == es2.id()

# Test nb::init<Eigen::DenseIndex>()
# Test id
dim_constructor = 3

es3 = nanoeigenpy.EigenSolver(dim_constructor)
es4 = nanoeigenpy.EigenSolver(dim_constructor)

id3 = es3.id()
id4 = es4.id()

assert id3 != id4
assert id3 == es3.id()
assert id4 == es4.id()
