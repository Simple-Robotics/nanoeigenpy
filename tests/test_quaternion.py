import numpy as np
import quaternion
from nanoeigenpy import Quaternion, AngleAxis

x = np.random.randn(3)
y = np.ones(3)

quat = Quaternion()
quat.setFromTwoVectors(x, y)

aa = AngleAxis(quat)
print(aa.angle)
print(aa.axis)

x = quaternion.X(quat)
# Implement and expose operator== first
assert x.a == quat
