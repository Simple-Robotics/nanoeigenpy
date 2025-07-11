import nanoeigenpy
import numpy as np
import pytest

_classes = [
    nanoeigenpy.ColPivHhJacobiSVD,
    nanoeigenpy.FullPivHhJacobiSVD,
    nanoeigenpy.HhJacobiSVD,
    nanoeigenpy.NoPrecondJacobiSVD,
]


@pytest.mark.parametrize("cls", _classes)
def test_jacobi(cls):
    dim = 100
    rng = np.random.default_rng()

    # Test nb::init<const MatrixType &>()
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    opt_U = nanoeigenpy.DecompositionOptions.ComputeFullU.value
    opt_V = nanoeigenpy.DecompositionOptions.ComputeFullV.value

    jacobisvd = cls(A, opt_U | opt_V)
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
    print("bdcsvd_compute: ", bdcsvd_compute)
    print("bdcsvd_compute_options: ", bdcsvd_compute_options)

    rank = jacobisvd.rank()
    singularvalues = jacobisvd.singularValues()
    nonzerosingularvalues = jacobisvd.nonzeroSingularValues()
    print("rank: ", rank)
    print("singularvalues: ", singularvalues)
    print("nonzerosingularvalues: ", nonzerosingularvalues)

    matrixU = jacobisvd.matrixU()
    matrixV = jacobisvd.matrixV()
    print("matrixU: ", matrixU)
    print("matrixV: ", matrixV)

    computeU = jacobisvd.computeU()
    computeV = jacobisvd.computeV()
    print("computeU: ", computeU)
    print("computeV: ", computeV)

    jacobisvd.setThreshold(1e-8)
    assert jacobisvd.threshold() == 1e-8

    decomp1 = cls()
    decomp2 = cls()

    id1 = decomp1.id()
    id2 = decomp2.id()

    assert id1 != id2
    assert id1 == decomp1.id()
    assert id2 == decomp2.id()

    dim_constructor = 3
    print("dim_constructor: ", dim_constructor)

    decomp3 = cls(rows, cols, opt_U | opt_V)
    decomp4 = cls(rows, cols, opt_U | opt_V)

    id3 = decomp3.id()
    id4 = decomp4.id()

    assert id3 != id4
    assert id3 == decomp3.id()
    assert id4 == decomp4.id()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
