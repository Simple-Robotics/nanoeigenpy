import nanoeigenpy
import numpy as np


def test_ldlt():
    dim = 100
    rng = np.random.default_rng()

    A_neg = -np.eye(dim)
    ldlt_neg = nanoeigenpy.LDLT(A_neg)
    assert ldlt_neg.isNegative()
    assert not ldlt_neg.isPositive()

    A_pos = np.eye(dim)
    ldlt_pos = nanoeigenpy.LDLT(A_pos)
    assert ldlt_pos.isPositive()
    assert not ldlt_pos.isNegative()

    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    ldlt = nanoeigenpy.LDLT(A)
    assert ldlt.info() == nanoeigenpy.ComputationInfo.Success

    L = ldlt.matrixL()
    D = ldlt.vectorD()
    P = ldlt.transpositionsP()
    assert nanoeigenpy.is_approx(
        np.transpose(P).dot(L.dot(np.diag(D).dot(np.transpose(L).dot(P)))), A
    )

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = ldlt.solve(B)
    assert nanoeigenpy.is_approx(X, X_est)
    assert nanoeigenpy.is_approx(A.dot(X_est), B)

    x = rng.random(dim)
    b = A.dot(x)
    x_est = ldlt.solve(b)
    assert nanoeigenpy.is_approx(x, x_est)
    assert nanoeigenpy.is_approx(A.dot(x_est), b)

    A_reconstructed = ldlt.reconstructedMatrix()
    assert nanoeigenpy.is_approx(A_reconstructed, A)

    adjoint = ldlt.adjoint()
    assert adjoint is ldlt

    A_cond = np.eye(dim)
    ldlt_cond = nanoeigenpy.LDLT(A_cond)
    estimated_r_cond_num = ldlt_cond.rcond()
    assert abs(estimated_r_cond_num - 1) <= 1e-9

    ldlt_compute = ldlt.compute(A)  # noqa

    LDLT = ldlt.matrixLDLT()
    LDLT_lower_without_diag = np.tril(LDLT, k=-1)
    L_lower_without_diag = np.tril(L, k=-1)
    assert nanoeigenpy.is_approx(LDLT_lower_without_diag, L_lower_without_diag)

    A_upper_without_diag = np.triu(A, k=1)
    LLT_upper_without_diag = np.triu(LDLT, k=1)
    assert nanoeigenpy.is_approx(A_upper_without_diag, LLT_upper_without_diag)

    LDLT_diag = np.diagonal(LDLT)
    assert nanoeigenpy.is_approx(LDLT_diag, D)

    sigma = 3
    w = np.ones(dim)
    ldlt.rankUpdate(w, sigma)
    L = ldlt.matrixL()
    D = ldlt.vectorD()
    P = ldlt.transpositionsP()
    A_updated = np.transpose(P).dot(L.dot(np.diag(D).dot(np.transpose(L).dot(P))))
    assert nanoeigenpy.is_approx(A_updated, A + sigma * w * np.transpose(w))

    ldlt1 = nanoeigenpy.LDLT()
    ldlt2 = nanoeigenpy.LDLT()

    id1 = ldlt1.id()
    id2 = ldlt2.id()

    assert id1 != id2
    assert id1 == ldlt1.id()
    assert id2 == ldlt2.id()

    dim_constructor = 3

    ldlt3 = nanoeigenpy.LDLT(dim_constructor)
    ldlt4 = nanoeigenpy.LDLT(dim_constructor)

    id3 = ldlt3.id()
    id4 = ldlt4.id()

    assert id3 != id4
    assert id3 == ldlt3.id()
    assert id4 == ldlt4.id()
