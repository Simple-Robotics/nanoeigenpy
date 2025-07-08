import nanoeigenpy
import numpy as np

dim = 100
seed = 1
rng = np.random.default_rng(seed)

A = rng.random((dim, dim))

# Tests init
cs = nanoeigenpy.ComplexSchur(dim)
cs = nanoeigenpy.ComplexSchur(A)
assert cs.info() == nanoeigenpy.ComputationInfo.Success

U = cs.matrixU()
T = cs.matrixT()
U_star = U.conj().T

assert nanoeigenpy.is_approx(A.real, (U @ T @ U_star).real)
assert np.allclose(A.imag, (U @ T @ U_star).imag)

# Test nb::init<Eigen::DenseIndex>()
# Test id
dim_constructor = 3

cs3 = nanoeigenpy.ComplexSchur(dim_constructor)
cs4 = nanoeigenpy.ComplexSchur(dim_constructor)

id3 = cs3.id()
id4 = cs4.id()

assert id3 != id4
assert id3 == cs3.id()
assert id4 == cs4.id()
