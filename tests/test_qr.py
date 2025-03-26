import nanoeigenpy
import numpy as np

rows = 20
cols = 100
rng = np.random.default_rng()

A = rng.random((rows, cols))

# Test HouseholderQR decomposition
householder_qr = nanoeigenpy.HouseholderQR()
householder_qr = nanoeigenpy.HouseholderQR(rows, cols)
householder_qr = nanoeigenpy.HouseholderQR(A)

householder_qr_eye = nanoeigenpy.HouseholderQR(np.eye(rows, rows))
X = rng.random((rows, 20))
assert householder_qr_eye.absDeterminant() == 1.0
assert householder_qr_eye.logAbsDeterminant() == 0.0

Y = householder_qr_eye.solve(X)
assert (X == Y).all()

x = rng.random(rows)
y = householder_qr_eye.solve(x)
assert (x == y).all()

# Test FullPivHouseholderQR decomposition
fullpiv_householder_qr = nanoeigenpy.FullPivHouseholderQR()
fullpiv_householder_qr = nanoeigenpy.FullPivHouseholderQR(rows, cols)
fullpiv_householder_qr = nanoeigenpy.FullPivHouseholderQR(A)

fullpiv_householder_qr = nanoeigenpy.FullPivHouseholderQR(np.eye(rows, rows))
assert fullpiv_householder_qr.isSurjective()
assert fullpiv_householder_qr.isInjective()
fullpiv_householder_qr.isInvertible()

X = rng.random((rows, 20))
assert fullpiv_householder_qr.absDeterminant() == 1.0
assert fullpiv_householder_qr.logAbsDeterminant() == 0.0

Y = fullpiv_householder_qr.solve(X)
assert (X == Y).all()
assert fullpiv_householder_qr.rank() == rows

x = rng.random(rows)
y = fullpiv_householder_qr.solve(x)
assert (x == y).all()

fullpiv_householder_qr.setThreshold(1e-8)
assert fullpiv_householder_qr.threshold() == 1e-8
assert nanoeigenpy.is_approx(np.eye(rows, rows), fullpiv_householder_qr.inverse())

assert fullpiv_householder_qr.maxPivot() == 1.0
assert fullpiv_householder_qr.nonzeroPivots() == rows
assert fullpiv_householder_qr.dimensionOfKernel() == 0

# Test ColPivHouseholderQR decomposition
colpiv_householder_qr = nanoeigenpy.ColPivHouseholderQR(A)
assert colpiv_householder_qr.info() == nanoeigenpy.ComputationInfo.Success

colpiv_householder_qr = nanoeigenpy.ColPivHouseholderQR(np.eye(rows, rows))
X = rng.random((rows, 20))
assert colpiv_householder_qr.absDeterminant() == 1.0
assert colpiv_householder_qr.logAbsDeterminant() == 0.0

Y = colpiv_householder_qr.solve(X)
assert (X == Y).all()
assert colpiv_householder_qr.rank() == rows

colpiv_householder_qr.setThreshold(1e-8)
assert colpiv_householder_qr.threshold() == 1e-8
assert nanoeigenpy.is_approx(np.eye(rows, rows), colpiv_householder_qr.inverse())

assert colpiv_householder_qr.maxPivot() == 1.0
assert colpiv_householder_qr.nonzeroPivots() == rows
assert colpiv_householder_qr.dimensionOfKernel() == 0

# Test CompleteOrthogonalDecomposition
cod = nanoeigenpy.CompleteOrthogonalDecomposition(A)
assert cod.info() == nanoeigenpy.ComputationInfo.Success

cod = nanoeigenpy.CompleteOrthogonalDecomposition(np.eye(rows, rows))
X = rng.random((rows, 20))
assert cod.absDeterminant() == 1.0
assert cod.logAbsDeterminant() == 0.0

Y = cod.solve(X)
assert (X == Y).all()
assert cod.rank() == rows

x = rng.random(rows)
y = cod.solve(x)
assert (x == y).all()

cod.setThreshold(1e-8)
assert cod.threshold() == 1e-8
assert nanoeigenpy.is_approx(np.eye(rows, rows), cod.pseudoInverse())

assert cod.maxPivot() == 1.0
assert cod.nonzeroPivots() == rows
assert cod.dimensionOfKernel() == 0
