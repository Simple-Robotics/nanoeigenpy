import numpy as np
from nanoeigenpy import Quaternion, AngleAxis

x = np.random.randn(3)
y = np.ones(3)

quat = Quaternion()
quat.setFromTwoVectors(x, y)

aa = AngleAxis(quat)
print(aa.angle)
print(aa.axis)
