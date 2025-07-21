import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

hess = nanoeigenpy.HessenbergDecomposition(A)

Q = hess.matrixQ()
H = hess.matrixH()

if np.iscomplexobj(A):
    A_reconstructed = Q @ H @ Q.conj().T
else:
    A_reconstructed = Q @ H @ Q.T
assert nanoeigenpy.is_approx(A, A_reconstructed, 1e-10)

for row in range(2, dim):
    for col in range(row - 1):
        assert abs(H[row, col]) < 1e-12

if np.iscomplexobj(Q):
    QQ_conj = Q @ Q.conj().T
else:
    QQ_conj = Q @ Q.T
assert nanoeigenpy.is_approx(QQ_conj, np.eye(dim), 1e-10)

A_test = rng.random((dim, dim))
hess1 = nanoeigenpy.HessenbergDecomposition(dim)
hess1.compute(A_test)
hess2 = nanoeigenpy.HessenbergDecomposition(A_test)

H1 = hess1.matrixH()
H2 = hess2.matrixH()
Q1 = hess1.matrixQ()
Q2 = hess2.matrixQ()

assert nanoeigenpy.is_approx(H1, H2, 1e-12)
assert nanoeigenpy.is_approx(Q1, Q2, 1e-12)

hCoeffs = hess.householderCoefficients()
packed = hess.packedMatrix()

assert hCoeffs.shape == (dim - 1,)
assert packed.shape == (dim, dim)

for i in range(dim):
    for j in range(i - 1, dim):
        if j >= 0:
            assert abs(H[i, j] - packed[i, j]) < 1e-12

hess_default = nanoeigenpy.HessenbergDecomposition(dim)
hess_matrix = nanoeigenpy.HessenbergDecomposition(A)

hess1_id = nanoeigenpy.HessenbergDecomposition(dim)
hess2_id = nanoeigenpy.HessenbergDecomposition(dim)
id1 = hess1_id.id()
id2 = hess2_id.id()
assert id1 != id2
assert id1 == hess1_id.id()
assert id2 == hess2_id.id()

hess3_id = nanoeigenpy.HessenbergDecomposition(A)
hess4_id = nanoeigenpy.HessenbergDecomposition(A)
id3 = hess3_id.id()
id4 = hess4_id.id()
assert id3 != id4
assert id3 == hess3_id.id()
assert id4 == hess4_id.id()
