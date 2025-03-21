import numpy as np
from geometry import (
    AngleAxis
)
from numpy import cos

verbose = True


def isapprox(a, b, epsilon=1e-6):
    if issubclass(a.__class__, np.ndarray) and issubclass(b.__class__, np.ndarray):
        return np.allclose(a, b, epsilon)
    else:
        return abs(a - b) < epsilon

# --- Angle Vector ------------------------------------------------
r = AngleAxis(0.1, np.array([1, 0, 0], np.double))
if verbose:
    print("Rx(.1) = \n\n", r.matrix(), "\n")
assert isapprox(r.matrix()[2, 2], cos(r.angle))
assert isapprox(r.axis, np.array([1.0, 0, 0]))
assert isapprox(r.angle, 0.1)
assert r.isApprox(r)
assert r.isApprox(r, 1e-2)

r.axis = np.array([0, 1, 0], np.double).T
assert isapprox(r.matrix()[0, 0], cos(r.angle))

ri = r.inverse()
assert isapprox(ri.angle, -0.1)

R = r.matrix()
r2 = AngleAxis(np.dot(R, R))
assert isapprox(r2.angle, r.angle * 2)
