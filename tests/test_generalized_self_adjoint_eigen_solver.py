import nanoeigenpy
import numpy as np
import pytest

_options = [
    nanoeigenpy.DecompositionOptions.ComputeEigenvectors.value
    | nanoeigenpy.DecompositionOptions.Ax_lBx.value,
    nanoeigenpy.DecompositionOptions.EigenvaluesOnly.value
    | nanoeigenpy.DecompositionOptions.Ax_lBx.value,
    nanoeigenpy.DecompositionOptions.ComputeEigenvectors.value
    | nanoeigenpy.DecompositionOptions.ABx_lx.value,
    nanoeigenpy.DecompositionOptions.EigenvaluesOnly.value
    | nanoeigenpy.DecompositionOptions.ABx_lx.value,
    nanoeigenpy.DecompositionOptions.ComputeEigenvectors.value
    | nanoeigenpy.DecompositionOptions.BAx_lx.value,
    nanoeigenpy.DecompositionOptions.EigenvaluesOnly.value
    | nanoeigenpy.DecompositionOptions.BAx_lx.value,
]


@pytest.mark.parametrize("options", _options)
def test_generalized_selfadjoint_eigensolver(options):
    dim = 100
    rng = np.random.default_rng()
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5
    B = rng.random((dim, dim))
    B = B @ B.T + 0.1 * np.eye(dim)

    gsaes = nanoeigenpy.GeneralizedSelfAdjointEigenSolver(A, B, options)
    assert gsaes.info() == nanoeigenpy.ComputationInfo.Success

    D = gsaes.eigenvalues()
    assert D.shape == (dim,)
    assert all(abs(D[i].imag) < 1e-12 for i in range(dim))
    assert all(D[i] <= D[i + 1] + 1e-12 for i in range(dim - 1))

    compute_eigenvectors = bool(
        options & nanoeigenpy.DecompositionOptions.ComputeEigenvectors.value
    )

    if compute_eigenvectors:
        V = gsaes.eigenvectors()
        assert V.shape == (dim, dim)

        if options & nanoeigenpy.DecompositionOptions.Ax_lBx.value:
            for i in range(dim):
                v = V[:, i]
                lam = D[i]
                Av = A @ v
                lam_Bv = lam * (B @ v)
                assert nanoeigenpy.is_approx(Av, lam_Bv, 1e-6)

            VT_B_V = V.T @ B @ V
            assert nanoeigenpy.is_approx(VT_B_V, np.eye(dim), 1e-6)

        elif options & nanoeigenpy.DecompositionOptions.ABx_lx.value:
            AB = A @ B
            for i in range(dim):
                v = V[:, i]
                lam = D[i]
                ABv = AB @ v
                lam_v = lam * v
                assert nanoeigenpy.is_approx(ABv, lam_v, 1e-6)

        elif options & nanoeigenpy.DecompositionOptions.BAx_lx.value:
            BA = B @ A
            for i in range(dim):
                v = V[:, i]
                lam = D[i]
                BAv = BA @ v
                lam_v = lam * v
                assert nanoeigenpy.is_approx(BAv, lam_v, 1e-6)

    _gsaes_compute = gsaes.compute(A, B)
    _gsaes_compute_options = gsaes.compute(A, B, options)

    rank = len([d for d in D if abs(d) > 1e-12])
    assert rank <= dim


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
