import nanoeigenpy
import numpy as np
from scipy.sparse import csc_matrix

rng = np.random.default_rng()

def test(SolverType: type):
    dim = 100
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    A = csc_matrix(A)

    llt = SolverType(A)

    assert llt.info() == nanoeigenpy.ComputationInfo.Success

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = llt.solve(B)
    #    import pdb; pdb.set_trace()
    assert nanoeigenpy.is_approx(X, X_est)
    assert nanoeigenpy.is_approx(A.dot(X_est), B)

    llt.analyzePattern(A)
    llt.factorize(A)

test(nanoeigenpy.AccelerateLLT)
test(nanoeigenpy.AccelerateLDLT)
test(nanoeigenpy.AccelerateLDLTUnpivoted)
test(nanoeigenpy.AccelerateLDLTSBK)
test(nanoeigenpy.AccelerateLDLTTPP)
test(nanoeigenpy.AccelerateQR)
