import nanoeigenpy
import numpy as np

dim = 100
seed = 1
rng = np.random.default_rng(seed)

# Test nb::init<const MatrixType &>()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

fullpivlu = nanoeigenpy.FullPivLU(A)

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = fullpivlu.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = fullpivlu.solve(b)
assert nanoeigenpy.is_approx(x, x_est)
assert nanoeigenpy.is_approx(A.dot(x_est), b)

# Others
rows = fullpivlu.rows()
cols = fullpivlu.cols()
assert cols == dim
assert rows == dim

fullpivlu_compute = fullpivlu.compute(A)

A_reconstructed = fullpivlu.reconstructedMatrix()
assert nanoeigenpy.is_approx(A_reconstructed, A)

nonzeropivots = fullpivlu.nonzeroPivots()
maxpivot = fullpivlu.maxPivot()
kernel = fullpivlu.kernel()
image = fullpivlu.image(A)
rcond = fullpivlu.rcond()
determinant = fullpivlu.determinant()
rank = fullpivlu.rank()
dimkernel = fullpivlu.dimensionOfKernel()
injective = fullpivlu.isInjective()
surjective = fullpivlu.isSurjective()
invertible = fullpivlu.isInvertible()
inverse = fullpivlu.inverse()

fullpivlu.setThreshold(1e-8)
assert fullpivlu.threshold() == 1e-8

decomp1 = nanoeigenpy.FullPivLU()
decomp2 = nanoeigenpy.FullPivLU()

id1 = decomp1.id()
id2 = decomp2.id()

assert id1 != id2
assert id1 == decomp1.id()
assert id2 == decomp2.id()

dim_constructor = 3

decomp3 = nanoeigenpy.FullPivLU(rows, cols)
decomp4 = nanoeigenpy.FullPivLU(rows, cols)

id3 = decomp3.id()
id4 = decomp4.id()

assert id3 != id4
assert id3 == decomp3.id()
assert id4 == decomp4.id()
