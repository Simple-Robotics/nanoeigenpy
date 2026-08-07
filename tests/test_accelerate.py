import nanoeigenpy
import numpy as np
from scipy.sparse import csc_matrix
import pytest

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "SolverType",
    [
        nanoeigenpy.AccelerateLLT,
        nanoeigenpy.AccelerateLDLT,
        nanoeigenpy.AccelerateLDLTUnpivoted,
        nanoeigenpy.AccelerateLDLTSBK,
        nanoeigenpy.AccelerateLDLTTPP,
        nanoeigenpy.AccelerateQR,
    ],
)
def test_accelerate_solver(SolverType):
    dim = 100
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    A = csc_matrix(A)

    llt = SolverType(A)

    assert llt.info() == nanoeigenpy.ComputationInfo.Success

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = llt.solve(B)

    assert nanoeigenpy.is_approx(X, X_est)
    assert nanoeigenpy.is_approx(A.dot(X_est), B)

    llt.analyzePattern(A)
    llt.factorize(A)
