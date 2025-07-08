import nanoeigenpy
import numpy as np

dim = 100
seed = 1
rng = np.random.default_rng(seed)

# Test nb::init<const MatrixType &>()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

opt_U = nanoeigenpy.DecompositionOptions.ComputeFullU.value
opt_V = nanoeigenpy.DecompositionOptions.ComputeFullV.value

jacobisvd = nanoeigenpy.JacobiSVD(A, opt_U | opt_V)
assert jacobisvd.info() == nanoeigenpy.ComputationInfo.Success

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = jacobisvd.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = jacobisvd.solve(b)
assert nanoeigenpy.is_approx(x, x_est)
assert nanoeigenpy.is_approx(A.dot(x_est), b)

# Others
rows = jacobisvd.rows()
cols = jacobisvd.cols()
assert cols == dim
assert rows == dim

bdcsvd_compute = jacobisvd.compute(A)
bdcsvd_compute_options = jacobisvd.compute(A, opt_U | opt_V)

rank = jacobisvd.rank()
singularvalues = jacobisvd.singularValues()
nonzerosingularvalues = jacobisvd.nonzeroSingularValues()

matrixU = jacobisvd.matrixU()
matrixV = jacobisvd.matrixV()

computeU = jacobisvd.computeU()
computeV = jacobisvd.computeV()

jacobisvd.setThreshold(1e-8)
assert jacobisvd.threshold() == 1e-8

decomp1 = nanoeigenpy.JacobiSVD()
decomp2 = nanoeigenpy.JacobiSVD()

id1 = decomp1.id()
id2 = decomp2.id()

assert id1 != id2
assert id1 == decomp1.id()
assert id2 == decomp2.id()

dim_constructor = 3

decomp3 = nanoeigenpy.JacobiSVD(rows, cols, opt_U | opt_V)
decomp4 = nanoeigenpy.JacobiSVD(rows, cols, opt_U | opt_V)

id3 = decomp3.id()
id4 = decomp4.id()

assert id3 != id4
assert id3 == decomp3.id()
assert id4 == decomp4.id()
