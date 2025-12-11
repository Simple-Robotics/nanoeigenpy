import nanoeigenpy
import numpy as np
import scipy.sparse as spa


def test_sparse_qr():
    dim = 100
    rng = np.random.default_rng()

    A_fac = spa.random(dim, dim, density=0.25, random_state=rng)
    A = A_fac.T @ A_fac
    A += spa.diags(10.0 * rng.standard_normal(dim) ** 2)
    A = A.tocsc(True)
    A.check_format()

    spqr = nanoeigenpy.SparseQR(A)

    assert spqr.info() == nanoeigenpy.ComputationInfo.Success

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = spqr.solve(B)
    assert isinstance(X_est, np.ndarray)
    assert nanoeigenpy.is_approx(X, X_est)
    assert nanoeigenpy.is_approx(A.dot(X_est), B)

    spqr.analyzePattern(A)
    spqr.factorize(A)

    X_sparse = spa.random(dim, 10, random_state=rng)
    B_sparse = A.dot(X_sparse)
    B_sparse: spa.csc_matrix = B_sparse.tocsc(True)
    if not B_sparse.has_sorted_indices:
        B_sparse.sort_indices()

    X_est = spqr.solve(B_sparse)
    assert isinstance(X_est, spa.csc_matrix)
    assert nanoeigenpy.is_approx(X_est.toarray(), X_sparse.toarray())
    assert nanoeigenpy.is_approx(A.dot(X_est.toarray()), B_sparse.toarray())

    Q = spqr.matrixQ()
    R = spqr.matrixR()
    P = spqr.colsPermutation()

    assert spqr.matrixQ().rows() == dim
    assert spqr.matrixQ().cols() == dim
    assert R.shape[0] == dim
    assert R.shape[1] == dim
    assert P.indices().size == dim

    test_vec = rng.random(dim)
    test_matrix = rng.random((dim, 20))

    Qv = Q @ test_vec
    QM = Q @ test_matrix
    Qt = Q.transpose()
    QtV = Qt @ test_vec
    QtM = Qt @ test_matrix

    assert Qv.shape == (dim,)
    assert QM.shape == (dim, 20)
    assert QtV.shape == (dim,)
    assert QtM.shape == (dim, 20)

    Qa_real_mat = Q.adjoint()
    QaV = Qa_real_mat @ test_vec
    assert nanoeigenpy.is_approx(QtV, QaV)

    A_dense = A.toarray()
    P_indices = np.array([P.indices()[i] for i in range(dim)])
    A_permuted = A_dense[:, P_indices]

    QtAP = Qt @ A_permuted
    R_dense = spqr.matrixR().toarray()
    assert nanoeigenpy.is_approx(QtAP, R_dense)
