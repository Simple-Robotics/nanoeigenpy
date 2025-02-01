import numpy as np
from nanoeigenpy import Quaternion

x = np.random.randn(3)
y = np.ones(3)

quat = Quaternion()
quat.setFromTwoVectors(x, y)
