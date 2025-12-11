import nanoeigenpy
import numpy as np


def test_permutation_matrix():
    dim = 100
    rng = np.random.default_rng()
    indices = rng.permutation(dim)

    perm = nanoeigenpy.PermutationMatrix(dim)
    perm = nanoeigenpy.PermutationMatrix(indices)

    est_indices = perm.indices()
    assert est_indices.all() == indices.all()

    perm_left = perm.applyTranspositionOnTheLeft(0, 1)
    perm_left_right = perm_left.applyTranspositionOnTheRight(0, 1)
    assert perm_left_right.indices().all() == perm.indices().all()

    perm.setIdentity()
    assert perm.indices().all() == np.arange(dim).all()
    dim = dim + 1
    perm.setIdentity(dim)
    assert perm.indices().all() == np.arange(dim).all()

    perm.setIdentity()
    dense = perm.toDenseMatrix()
    assert dense.all() == np.eye(dim).all()

    perm = nanoeigenpy.PermutationMatrix(np.array([1, 0, 2]))
    perm_t = perm.transpose()
    dense = perm.toDenseMatrix()
    dense_t = perm_t.toDenseMatrix()
    assert dense_t.all() == dense.T.all()

    perm_inv = perm.inverse()
    result = perm * perm_inv
    identity = result.toDenseMatrix()
    assert identity.all() == np.eye(3).all()

    dim_constructor = 3

    perm1 = nanoeigenpy.PermutationMatrix(dim_constructor)
    perm2 = nanoeigenpy.PermutationMatrix(dim_constructor)

    id1 = perm1.id()
    id2 = perm2.id()

    assert id1 != id2
    assert id1 == perm1.id()
    assert id2 == perm2.id()

    es3 = nanoeigenpy.PermutationMatrix(indices)
    es4 = nanoeigenpy.PermutationMatrix(indices)

    id3 = es3.id()
    id4 = es4.id()

    assert id3 != id4
    assert id3 == es3.id()
    assert id4 == es4.id()
