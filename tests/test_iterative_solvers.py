import nanoeigenpy
import numpy as np

dim = 100
seed = 6
rng = np.random.default_rng(seed)

clazzes = [
    nanoeigenpy.solvers.ConjugateGradient,
    nanoeigenpy.solvers.IdentityConjugateGradient,
    nanoeigenpy.solvers.LeastSquaresConjugateGradient,
    nanoeigenpy.MINRES,
]

for cls in clazzes:
    print("================")
    print(cls)

    A = np.eye(dim)
    solver = cls(A)

    # Vector rhs

    x = rng.random(dim)
    b = A.dot(x)
    x_est = solver.solve(b)
    print(x - x_est)

    assert nanoeigenpy.is_approx(x, x_est, 1e-6)
    assert nanoeigenpy.is_approx(b, A.dot(x_est), 1e-6)

    # Matrix rhs

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = solver.solve(B)

    print("X_est :", X_est)

    assert nanoeigenpy.is_approx(X, X_est, 1e-6)
    assert nanoeigenpy.is_approx(B, A.dot(X_est), 1e-6)

    # With guess
    # Vector rhs

    solver = cls(A)
    b = A.dot(x)
    x_est = solver.solveWithGuess(b, x)

    print("x_est :", x_est)

    assert nanoeigenpy.is_approx(x, x_est, 1e-6)
    assert nanoeigenpy.is_approx(b, A.dot(x_est), 1e-6)

    # Matrix rhs

    solver = cls(A)
    B = A.dot(X)
    X_est = solver.solveWithGuess(B, X)

    print("X_est :", X_est)

    assert nanoeigenpy.is_approx(X, X_est, 1e-6)
    assert nanoeigenpy.is_approx(B, A.dot(X_est), 1e-6)
