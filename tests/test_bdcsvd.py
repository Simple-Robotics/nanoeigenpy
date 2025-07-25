import nanoeigenpy
import numpy as np
import pytest

_options = [
    0,
    nanoeigenpy.DecompositionOptions.ComputeThinU.value,
    nanoeigenpy.DecompositionOptions.ComputeThinV.value,
    nanoeigenpy.DecompositionOptions.ComputeFullU.value,
    nanoeigenpy.DecompositionOptions.ComputeFullV.value,
    nanoeigenpy.DecompositionOptions.ComputeThinU.value
    | nanoeigenpy.DecompositionOptions.ComputeThinV.value,
    nanoeigenpy.DecompositionOptions.ComputeFullU.value
    | nanoeigenpy.DecompositionOptions.ComputeFullV.value,
    nanoeigenpy.DecompositionOptions.ComputeThinU.value
    | nanoeigenpy.DecompositionOptions.ComputeFullV.value,
    nanoeigenpy.DecompositionOptions.ComputeFullU.value
    | nanoeigenpy.DecompositionOptions.ComputeThinV.value,
]


@pytest.mark.parametrize("options", _options)
def test_bdcsvd(options):
    dim = 100
    rng = np.random.default_rng()
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    bdcsvd = nanoeigenpy.BDCSVD(A, options)
    assert bdcsvd.info() == nanoeigenpy.ComputationInfo.Success

    if options & (
        nanoeigenpy.DecompositionOptions.ComputeThinU.value
        | nanoeigenpy.DecompositionOptions.ComputeFullU.value
    ) and options & (
        nanoeigenpy.DecompositionOptions.ComputeThinV.value
        | nanoeigenpy.DecompositionOptions.ComputeFullV.value
    ):
        X = rng.random((dim, 20))
        B = A @ X
        X_est = bdcsvd.solve(B)
        assert nanoeigenpy.is_approx(X, X_est)
        assert nanoeigenpy.is_approx(A @ X_est, B)

        x = rng.random(dim)
        b = A @ x
        x_est = bdcsvd.solve(b)
        assert nanoeigenpy.is_approx(x, x_est)
        assert nanoeigenpy.is_approx(A @ x_est, b)

    rows = bdcsvd.rows()
    cols = bdcsvd.cols()
    assert cols == dim
    assert rows == dim

    _bdcsvd_compute = bdcsvd.compute(A)
    _bdcsvd_compute_options = bdcsvd.compute(A, options)

    rank = bdcsvd.rank()
    singularvalues = bdcsvd.singularValues()
    nonzerosingularvalues = bdcsvd.nonzeroSingularValues()
    assert rank == nonzerosingularvalues
    assert len(singularvalues) == dim
    assert all(
        singularvalues[i] >= singularvalues[i + 1]
        for i in range(len(singularvalues) - 1)
    )

    compute_u = bdcsvd.computeU()
    compute_v = bdcsvd.computeV()
    expected_compute_u = bool(
        options
        & (
            nanoeigenpy.DecompositionOptions.ComputeThinU.value
            | nanoeigenpy.DecompositionOptions.ComputeFullU.value
        )
    )
    expected_compute_v = bool(
        options
        & (
            nanoeigenpy.DecompositionOptions.ComputeThinV.value
            | nanoeigenpy.DecompositionOptions.ComputeFullV.value
        )
    )
    assert compute_u == expected_compute_u
    assert compute_v == expected_compute_v

    if compute_u:
        matrixU = bdcsvd.matrixU()
        if options & nanoeigenpy.DecompositionOptions.ComputeFullU.value:
            assert matrixU.shape == (dim, dim)
        elif options & nanoeigenpy.DecompositionOptions.ComputeThinU.value:
            assert matrixU.shape == (dim, dim)
        assert nanoeigenpy.is_approx(matrixU.T @ matrixU, np.eye(matrixU.shape[1]))

    if compute_v:
        matrixV = bdcsvd.matrixV()
        if options & nanoeigenpy.DecompositionOptions.ComputeFullV.value:
            assert matrixV.shape == (dim, dim)
        elif options & nanoeigenpy.DecompositionOptions.ComputeThinV.value:
            assert matrixV.shape == (dim, dim)
        assert nanoeigenpy.is_approx(matrixV.T @ matrixV, np.eye(matrixV.shape[1]))

    if compute_u and compute_v:
        U = bdcsvd.matrixU()
        V = bdcsvd.matrixV()
        S = bdcsvd.singularValues()
        S_matrix = np.diag(S)
        A_reconstructed = U @ S_matrix @ V.T
        assert nanoeigenpy.is_approx(A, A_reconstructed)

    bdcsvd.setSwitchSize(5)
    bdcsvd.setSwitchSize(16)
    bdcsvd.setSwitchSize(32)

    bdcsvd.setThreshold()
    _default_threshold = bdcsvd.threshold()
    bdcsvd.setThreshold(1e-8)
    assert bdcsvd.threshold() == 1e-8

    decomp1 = nanoeigenpy.BDCSVD()
    decomp2 = nanoeigenpy.BDCSVD()
    id1 = decomp1.id()
    id2 = decomp2.id()
    assert id1 != id2
    assert id1 == decomp1.id()
    assert id2 == decomp2.id()

    decomp3 = nanoeigenpy.BDCSVD(dim, dim, options)
    decomp4 = nanoeigenpy.BDCSVD(dim, dim, options)
    id3 = decomp3.id()
    id4 = decomp4.id()
    assert id3 != id4
    assert id3 == decomp3.id()
    assert id4 == decomp4.id()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
