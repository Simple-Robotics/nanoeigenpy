import nanoeigenpy
import numpy as np

m = 4
a = np.random.randn(m, m + 1)
a = a @ a.T

llt = nanoeigenpy.LLT(a)
L = llt.matrixL()
