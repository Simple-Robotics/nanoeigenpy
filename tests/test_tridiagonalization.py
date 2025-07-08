import nanoeigenpy
import numpy as np

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

tri = nanoeigenpy.Tridiagonalization(A)
