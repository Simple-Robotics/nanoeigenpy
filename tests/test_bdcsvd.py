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

bdcsvd = nanoeigenpy.BDCSVD(A, opt_U | opt_V)
assert bdcsvd.info() == nanoeigenpy.ComputationInfo.Success

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = bdcsvd.solve(B)
assert nanoeigenpy.is_approx(X, X_est)
assert nanoeigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = bdcsvd.solve(b)
assert nanoeigenpy.is_approx(x, x_est)
assert nanoeigenpy.is_approx(A.dot(x_est), b)

# Others
rows = bdcsvd.rows()
cols = bdcsvd.cols()
assert cols == dim
assert rows == dim

bdcsvd_compute = bdcsvd.compute(A)
bdcsvd_compute_options = bdcsvd.compute(A, opt_U | opt_V)

rank = bdcsvd.rank()
singularvalues = bdcsvd.singularValues()
nonzerosingularvalues = bdcsvd.nonzeroSingularValues()

matrixU = bdcsvd.matrixU()
matrixV = bdcsvd.matrixV()

computeU = bdcsvd.computeU()
computeV = bdcsvd.computeV()

bdcsvd.setSwitchSize(5)

bdcsvd.setThreshold(1e-8)
assert bdcsvd.threshold() == 1e-8

decomp1 = nanoeigenpy.BDCSVD()
decomp2 = nanoeigenpy.BDCSVD()

id1 = decomp1.id()
id2 = decomp2.id()

assert id1 != id2
assert id1 == decomp1.id()
assert id2 == decomp2.id()

dim_constructor = 3

decomp3 = nanoeigenpy.BDCSVD(rows, cols, opt_U | opt_V)
decomp4 = nanoeigenpy.BDCSVD(rows, cols, opt_U | opt_V)

id3 = decomp3.id()
id4 = decomp4.id()

assert id3 != id4
assert id3 == decomp3.id()
assert id4 == decomp4.id()
