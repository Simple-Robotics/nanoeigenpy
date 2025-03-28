import nanoeigenpy
import numpy as np

dim = 10
seed = 1
rng = np.random.default_rng(seed)
indices = rng.permutation(dim)

# Tests init
perm = nanoeigenpy.PermutationMatrix(dim)
perm = nanoeigenpy.PermutationMatrix(indices)

# Test indices
est_indices = perm.indices()
assert est_indices.all() == indices.all()

# Test applyTranspositionOnTheLeft
# Test applyTranspositionOnThtRight
perm_left = perm.applyTranspositionOnTheLeft(0, 1)
perm_left_right = perm_left.applyTranspositionOnTheRight(0, 1)
assert perm_left_right.indices().all() == perm.indices().all()

# Test setIdentity
perm.setIdentity()
assert perm.indices().all() == np.arange(dim).all()
dim = dim + 1
perm.setIdentity(dim)
assert perm.indices().all() == np.arange(dim).all()

# Test nb::init<Eigen::DenseIndex>()
# Test id
dim_constructor = 3

perm1 = nanoeigenpy.PermutationMatrix(dim_constructor)
perm2 = nanoeigenpy.PermutationMatrix(dim_constructor)

id1 = perm1.id()
id2 = perm2.id()

assert id1 != id2
assert id1 == perm1.id()
assert id2 == perm2.id()

# Test nb::init<Eigen::DenseIndex>()
# Test id
es3 = nanoeigenpy.PermutationMatrix(indices)
es4 = nanoeigenpy.PermutationMatrix(indices)

id3 = es3.id()
id4 = es4.id()

assert id3 != id4
assert id3 == es3.id()
assert id4 == es4.id()
