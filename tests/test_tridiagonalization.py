import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5

tri = nanoeigenpy.Tridiagonalization(A)

Q = tri.matrixQ()
T = tri.matrixT()

assert nanoeigenpy.is_approx(A, Q @ T @ Q.T)
assert nanoeigenpy.is_approx(Q @ Q.T, np.eye(dim))

for i in range(dim):
    for j in range(dim):
        if abs(i - j) > 1:
            assert abs(T[i, j]) < 1e-12

assert nanoeigenpy.is_approx(T, T.T)

diag = tri.diagonal()
sub_diag = tri.subDiagonal()

for i in range(dim):
    assert abs(diag[i] - T[i, i]) < 1e-12

for i in range(dim - 1):
    assert abs(sub_diag[i] - T[i + 1, i]) < 1e-12

A_test = rng.random((dim, dim))
A_test = (A_test + A_test.T) * 0.5

tri1 = nanoeigenpy.Tridiagonalization(dim)
tri1.compute(A_test)
tri2 = nanoeigenpy.Tridiagonalization(A_test)

Q1 = tri1.matrixQ()
T1 = tri1.matrixT()
Q2 = tri2.matrixQ()
T2 = tri2.matrixT()

assert nanoeigenpy.is_approx(Q1, Q2)
assert nanoeigenpy.is_approx(T1, T2)

h_coeffs = tri.householderCoefficients()
packed = tri.packedMatrix()

assert h_coeffs.shape == (dim - 1,)
assert packed.shape == (dim, dim)

for i in range(dim):
    for j in range(i + 1, dim):
        assert abs(packed[i, j] - A[i, j]) < 1e-12

for i in range(dim):
    assert abs(packed[i, i] - T[i, i]) < 1e-12
    if i < dim - 1:
        assert abs(packed[i + 1, i] - T[i + 1, i]) < 1e-12

A_diag = np.diag(rng.random(dim))
tri_diag = nanoeigenpy.Tridiagonalization(A_diag)
Q_diag = tri_diag.matrixQ()
T_diag = tri_diag.matrixT()

assert nanoeigenpy.is_approx(A_diag, Q_diag @ T_diag @ Q_diag.T)
for i in range(dim):
    for j in range(dim):
        if i != j:
            assert abs(T_diag[i, j]) < 1e-10

A_tridiag = np.zeros((dim, dim))
for i in range(dim):
    A_tridiag[i, i] = rng.random()
    if i < dim - 1:
        val = rng.random()
        A_tridiag[i, i + 1] = val
        A_tridiag[i + 1, i] = val

tri_tridiag = nanoeigenpy.Tridiagonalization(A_tridiag)
Q_tridiag = tri_tridiag.matrixQ()
T_tridiag = tri_tridiag.matrixT()

assert nanoeigenpy.is_approx(A_tridiag, Q_tridiag @ T_tridiag @ Q_tridiag.T)


tri1_id = nanoeigenpy.Tridiagonalization(dim)
tri2_id = nanoeigenpy.Tridiagonalization(dim)
id1 = tri1_id.id()
id2 = tri2_id.id()
assert id1 != id2
assert id1 == tri1_id.id()
assert id2 == tri2_id.id()

tri3_id = nanoeigenpy.Tridiagonalization(A)
tri4_id = nanoeigenpy.Tridiagonalization(A)
id3 = tri3_id.id()
id4 = tri4_id.id()
assert id3 != id4
assert id3 == tri3_id.id()
assert id4 == tri4_id.id()
