import nanoeigenpy
import numpy as np
import pytest

dim = 100
rng = np.random.default_rng()
MAX_ITER = 8000

_classes = [
    nanoeigenpy.solvers.ConjugateGradient,
    nanoeigenpy.solvers.IdentityConjugateGradient,
    nanoeigenpy.solvers.LeastSquaresConjugateGradient,
    nanoeigenpy.solvers.IdentityLeastSquaresConjugateGradient,
    nanoeigenpy.solvers.DiagonalLeastSquaresConjugateGradient,
]


@pytest.mark.parametrize("cls", _classes)
def test_solver(cls):
    Q = rng.standard_normal((dim, dim))
    A = 0.5 * (Q.T + Q)
    solver = cls(A)
    solver.setMaxIterations(MAX_ITER)

    x = rng.random(dim)
    b = A.dot(x)
    x_est = solver.solve(b)

    assert solver.info() == nanoeigenpy.ComputationInfo.Success
    assert nanoeigenpy.is_approx(b, A.dot(x_est), 1e-6)

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = solver.solve(B)

    assert nanoeigenpy.is_approx(B, A.dot(X_est), 1e-6)


@pytest.mark.parametrize("cls", _classes)
def test_solver_with_guess(cls):
    Q = rng.standard_normal((dim, dim))
    A = 0.5 * (Q.T + Q)
    solver = cls(A)
    solver.setMaxIterations(MAX_ITER)

    x = rng.random(dim)
    b = A.dot(x)
    x_est = solver.solveWithGuess(b, x + 0.01)

    assert solver.info() == nanoeigenpy.ComputationInfo.Success
    assert nanoeigenpy.is_approx(x, x_est, 1e-6)
    assert nanoeigenpy.is_approx(b, A.dot(x_est), 1e-6)

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = solver.solveWithGuess(B, X + 0.01)

    assert nanoeigenpy.is_approx(X, X_est, 1e-6)
    assert nanoeigenpy.is_approx(B, A.dot(X_est), 1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
