import nanoeigenpy
import numpy as np

dim = 5
rng = np.random.default_rng()
A = rng.random((dim, dim))

cs = nanoeigenpy.ComplexSchur(A)
assert cs.info() == nanoeigenpy.ComputationInfo.Success

U = cs.matrixU()
T = cs.matrixT()

A_complex = A.astype(complex)
assert nanoeigenpy.is_approx(A_complex, U @ T @ U.conj().T, 1e-10)
assert nanoeigenpy.is_approx(U @ U.conj().T, np.eye(dim), 1e-10)

for row in range(1, dim):
    for col in range(row):
        assert abs(T[row, col]) < 1e-12

A_test = rng.random((dim, dim))
cs1 = nanoeigenpy.ComplexSchur(dim)
cs1.compute(A_test)
cs2 = nanoeigenpy.ComplexSchur(A_test)

assert cs1.info() == nanoeigenpy.ComputationInfo.Success
assert cs2.info() == nanoeigenpy.ComputationInfo.Success

T1 = cs1.matrixT()
U1 = cs1.matrixU()
T2 = cs2.matrixT()
U2 = cs2.matrixU()

assert nanoeigenpy.is_approx(T1, T2, 1e-12)
assert nanoeigenpy.is_approx(U1, U2, 1e-12)

cs_no_u = nanoeigenpy.ComplexSchur(A, False)
assert cs_no_u.info() == nanoeigenpy.ComputationInfo.Success
T_no_u = cs_no_u.matrixT()

assert nanoeigenpy.is_approx(T, T_no_u, 1e-12)

cs_compute_no_u = nanoeigenpy.ComplexSchur(dim)
result_no_u = cs_compute_no_u.compute(A, False)
assert result_no_u.info() == nanoeigenpy.ComputationInfo.Success
T_compute_no_u = cs_compute_no_u.matrixT()
assert nanoeigenpy.is_approx(T, T_compute_no_u, 1e-12)

cs_iter = nanoeigenpy.ComplexSchur(dim)
cs_iter.setMaxIterations(30 * dim)  # m_maxIterationsPerRow * size
result_iter = cs_iter.compute(A)
assert result_iter.info() == nanoeigenpy.ComputationInfo.Success
assert cs_iter.getMaxIterations() == 30 * dim

T_iter = cs_iter.matrixT()
U_iter = cs_iter.matrixU()
assert nanoeigenpy.is_approx(T, T_iter, 1e-12)
assert nanoeigenpy.is_approx(U, U_iter, 1e-12)

cs_few_iter = nanoeigenpy.ComplexSchur(dim)
cs_few_iter.setMaxIterations(1)
result_few = cs_few_iter.compute(A)
assert cs_few_iter.getMaxIterations() == 1

A_triangular = np.triu(A)
cs_triangular = nanoeigenpy.ComplexSchur(dim)
cs_triangular.setMaxIterations(1)
result_triangular = cs_triangular.compute(A_triangular)
assert result_triangular.info() == nanoeigenpy.ComputationInfo.Success

T_triangular = cs_triangular.matrixT()
U_triangular = cs_triangular.matrixU()

A_triangular_complex = A_triangular.astype(complex)
assert nanoeigenpy.is_approx(T_triangular, A_triangular_complex, 1e-10)
assert nanoeigenpy.is_approx(U_triangular, np.eye(dim, dtype=complex), 1e-10)

hess = nanoeigenpy.HessenbergDecomposition(A)
H = hess.matrixH()
Q_hess = hess.matrixQ()

cs_from_hess = nanoeigenpy.ComplexSchur(dim)
result_from_hess = cs_from_hess.computeFromHessenberg(H, Q_hess, True)
assert result_from_hess.info() == nanoeigenpy.ComputationInfo.Success

T_from_hess = cs_from_hess.matrixT()
U_from_hess = cs_from_hess.matrixU()

A_complex = A.astype(complex)
assert nanoeigenpy.is_approx(
    A_complex, U_from_hess @ T_from_hess @ U_from_hess.conj().T, 1e-10
)

cs1_id = nanoeigenpy.ComplexSchur(dim)
cs2_id = nanoeigenpy.ComplexSchur(dim)
id1 = cs1_id.id()
id2 = cs2_id.id()
assert id1 != id2
assert id1 == cs1_id.id()
assert id2 == cs2_id.id()

cs3_id = nanoeigenpy.ComplexSchur(A)
cs4_id = nanoeigenpy.ComplexSchur(A)
id3 = cs3_id.id()
id4 = cs4_id.id()
assert id3 != id4
assert id3 == cs3_id.id()
assert id4 == cs4_id.id()

cs5_id = nanoeigenpy.ComplexSchur(A, True)
cs6_id = nanoeigenpy.ComplexSchur(A, False)
id5 = cs5_id.id()
id6 = cs6_id.id()
assert id5 != id6
assert id5 == cs5_id.id()
assert id6 == cs6_id.id()
