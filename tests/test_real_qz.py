import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))

realqz = nanoeigenpy.RealQZ(A, B)
assert realqz.info() == nanoeigenpy.ComputationInfo.Success

Q = realqz.matrixQ()
S = realqz.matrixS()
Z = realqz.matrixZ()
T = realqz.matrixT()

assert nanoeigenpy.is_approx(A, Q @ S @ Z, 1e-10)
assert nanoeigenpy.is_approx(B, Q @ T @ Z, 1e-10)

assert nanoeigenpy.is_approx(Q @ Q.T, np.eye(dim), 1e-10)
assert nanoeigenpy.is_approx(Z @ Z.T, np.eye(dim), 1e-10)

for i in range(dim):
    for j in range(i):
        assert abs(T[i, j]) < 1e-12

for i in range(dim):
    for j in range(i - 1):
        assert abs(S[i, j]) < 1e-12

realqz_no_qz = nanoeigenpy.RealQZ(A, B, False)
assert realqz_no_qz.info() == nanoeigenpy.ComputationInfo.Success
S_no_qz = realqz_no_qz.matrixS()
T_no_qz = realqz_no_qz.matrixT()

for i in range(dim):
    for j in range(i):
        assert abs(T_no_qz[i, j]) < 1e-12

for i in range(dim):
    for j in range(i - 1):
        assert abs(S_no_qz[i, j]) < 1e-12

realqz_compute_no_qz = nanoeigenpy.RealQZ(dim)
result_no_qz = realqz_compute_no_qz.compute(A, B, False)
assert result_no_qz.info() == nanoeigenpy.ComputationInfo.Success
S_compute_no_qz = realqz_compute_no_qz.matrixS()
T_compute_no_qz = realqz_compute_no_qz.matrixT()

realqz_with_qz = nanoeigenpy.RealQZ(dim)
realqz_without_qz = nanoeigenpy.RealQZ(dim)

result_with = realqz_with_qz.compute(A, B, True)
result_without = realqz_without_qz.compute(A, B, False)

assert result_with.info() == nanoeigenpy.ComputationInfo.Success
assert result_without.info() == nanoeigenpy.ComputationInfo.Success

S_with = realqz_with_qz.matrixS()
T_with = realqz_with_qz.matrixT()
S_without = realqz_without_qz.matrixS()
T_without = realqz_without_qz.matrixT()

assert nanoeigenpy.is_approx(S_with, S_without, 1e-12)
assert nanoeigenpy.is_approx(T_with, T_without, 1e-12)

iterations = realqz.iterations()
assert iterations >= 0

realqz_iter = nanoeigenpy.RealQZ(dim)
realqz_iter.setMaxIterations(100)
realqz_iter.setMaxIterations(500)
result_iter = realqz_iter.compute(A, B)
assert result_iter.info() == nanoeigenpy.ComputationInfo.Success

realqz1_id = nanoeigenpy.RealQZ(dim)
realqz2_id = nanoeigenpy.RealQZ(dim)
id1 = realqz1_id.id()
id2 = realqz2_id.id()
assert id1 != id2
assert id1 == realqz1_id.id()
assert id2 == realqz2_id.id()

realqz3_id = nanoeigenpy.RealQZ(A, B)
realqz4_id = nanoeigenpy.RealQZ(A, B)
id3 = realqz3_id.id()
id4 = realqz4_id.id()
assert id3 != id4
assert id3 == realqz3_id.id()
assert id4 == realqz4_id.id()

realqz5_id = nanoeigenpy.RealQZ(A, B, True)
realqz6_id = nanoeigenpy.RealQZ(A, B, False)
id5 = realqz5_id.id()
id6 = realqz6_id.id()
assert id5 != id6
assert id5 == realqz5_id.id()
assert id6 == realqz6_id.id()
