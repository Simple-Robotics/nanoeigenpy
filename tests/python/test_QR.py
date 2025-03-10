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