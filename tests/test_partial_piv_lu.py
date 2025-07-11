import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()

# Test nb::init<const MatrixType &>()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

partialpivlu = nanoeigenpy.PartialPivLU(A)

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = partialpivlu.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = partialpivlu.solve(b)
assert nanoeigenpy.is_approx(x, x_est)
assert nanoeigenpy.is_approx(A.dot(x_est), b)

# Others
rows = partialpivlu.rows()
cols = partialpivlu.cols()
assert cols == dim
assert rows == dim

A_reconstructed = partialpivlu.reconstructedMatrix()
assert nanoeigenpy.is_approx(A_reconstructed, A)

rcond = partialpivlu.rcond()
determinant = partialpivlu.determinant()
inverse = partialpivlu.inverse()
reconstructedmatrix = partialpivlu.reconstructedMatrix()

decomp1 = nanoeigenpy.PartialPivLU()
decomp2 = nanoeigenpy.PartialPivLU()

id1 = decomp1.id()
id2 = decomp2.id()

assert id1 != id2
assert id1 == decomp1.id()
assert id2 == decomp2.id()

dim_constructor = 3

decomp3 = nanoeigenpy.PartialPivLU(dim)
decomp4 = nanoeigenpy.PartialPivLU(dim)

id3 = decomp3.id()
id4 = decomp4.id()

assert id3 != id4
assert id3 == decomp3.id()
assert id4 == decomp4.id()
