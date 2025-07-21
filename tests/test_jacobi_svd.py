import nanoeigenpy
import numpy as np
import pytest

THIN_U = nanoeigenpy.DecompositionOptions.ComputeThinU.value
THIN_V = nanoeigenpy.DecompositionOptions.ComputeThinV.value
FULL_U = nanoeigenpy.DecompositionOptions.ComputeFullU.value
FULL_V = nanoeigenpy.DecompositionOptions.ComputeFullV.value

_options = [
    0,
    THIN_U,
    THIN_V,
    FULL_U,
    FULL_V,
    THIN_U | THIN_V,
    FULL_U | FULL_V,
    THIN_U | FULL_V,
    FULL_U | THIN_V,
]

_classes = [
    nanoeigenpy.ColPivHhJacobiSVD,
    nanoeigenpy.FullPivHhJacobiSVD,
    nanoeigenpy.HhJacobiSVD,
    nanoeigenpy.NoPrecondJacobiSVD,
]


def is_valid_combination(cls, options):
    if cls == nanoeigenpy.FullPivHhJacobiSVD:
        has_thin_u = bool(options & THIN_U)
        has_thin_v = bool(options & THIN_V)

        if has_thin_u or has_thin_v:
            return False

    return True


@pytest.mark.parametrize("cls", _classes)
@pytest.mark.parametrize("options", _options)
def test_jacobi(cls, options):
    if not is_valid_combination(cls, options):
        pytest.skip(f"Invalid combination: {cls.__name__} with options {options}")

    dim = 100
    rng = np.random.default_rng()
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    jacobisvd = cls(A, options)
    assert jacobisvd.info() == nanoeigenpy.ComputationInfo.Success

    has_u = options & (THIN_U | FULL_U)
    has_v = options & (THIN_V | FULL_V)

    if has_u and has_v:
        X = rng.random((dim, 20))
        B = A @ X
        X_est = jacobisvd.solve(B)
        assert nanoeigenpy.is_approx(X, X_est)
        assert nanoeigenpy.is_approx(A @ X_est, B)

        x = rng.random(dim)
        b = A @ x
        x_est = jacobisvd.solve(b)
        assert nanoeigenpy.is_approx(x, x_est)
        assert nanoeigenpy.is_approx(A @ x_est, b)

    assert jacobisvd.rows() == dim
    assert jacobisvd.cols() == dim

    _jacobisvd_compute = jacobisvd.compute(A)
    _jacobisvd_compute_options = jacobisvd.compute(A, options)

    rank = jacobisvd.rank()
    singularvalues = jacobisvd.singularValues()
    nonzerosingularvalues = jacobisvd.nonzeroSingularValues()
    assert rank == nonzerosingularvalues
    assert len(singularvalues) == dim
    assert all(
        singularvalues[i] >= singularvalues[i + 1]
        for i in range(len(singularvalues) - 1)
    )

    compute_u = jacobisvd.computeU()
    compute_v = jacobisvd.computeV()
    expected_compute_u = bool(has_u)
    expected_compute_v = bool(has_v)
    assert compute_u == expected_compute_u
    assert compute_v == expected_compute_v

    if compute_u:
        matrixU = jacobisvd.matrixU()
        assert matrixU.shape == (dim, dim)
        assert nanoeigenpy.is_approx(matrixU.T @ matrixU, np.eye(matrixU.shape[1]))

    if compute_v:
        matrixV = jacobisvd.matrixV()
        assert matrixV.shape == (dim, dim)
        assert nanoeigenpy.is_approx(matrixV.T @ matrixV, np.eye(matrixV.shape[1]))

    if compute_u and compute_v:
        U = jacobisvd.matrixU()
        V = jacobisvd.matrixV()
        S = jacobisvd.singularValues()
        S_matrix = np.diag(S)
        A_reconstructed = U @ S_matrix @ V.T
        assert nanoeigenpy.is_approx(A, A_reconstructed)

    jacobisvd.setThreshold()
    _default_threshold = jacobisvd.threshold()
    jacobisvd.setThreshold(1e-8)
    assert jacobisvd.threshold() == 1e-8

    decomp1 = cls()
    decomp2 = cls()
    id1 = decomp1.id()
    id2 = decomp2.id()
    assert id1 != id2
    assert id1 == decomp1.id()
    assert id2 == decomp2.id()

    decomp3 = cls(dim, dim, options)
    decomp4 = cls(dim, dim, options)
    id3 = decomp3.id()
    id4 = decomp4.id()
    assert id3 != id4
    assert id3 == decomp3.id()
    assert id4 == decomp4.id()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
