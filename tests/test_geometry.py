import numpy as np
from numpy import cos
import nanoeigenpy
import quaternion


verbose = True


def isapprox(a, b, epsilon=1e-6):
    if issubclass(a.__class__, np.ndarray) and issubclass(b.__class__, np.ndarray):
        return np.allclose(a, b, epsilon)
    else:
        return abs(a - b) < epsilon


# --- Quaternion ---------------------------------------------------------------
verbose and print("[Quaternion] Coefficient initialisation")
q = nanoeigenpy.Quaternion(1, 2, 3, 4)
q.normalize()
assert isapprox(np.linalg.norm(q.coeffs()), q.norm())
assert isapprox(np.linalg.norm(q.coeffs()), 1)

verbose and print("[Quaternion] Coefficient-vector initialisation")
v = np.array([0.5, -0.5, 0.5, 0.5])
for k in range(10000):
    qv = nanoeigenpy.Quaternion(v)
assert isapprox(qv.coeffs(), v)

verbose and print("[Quaternion] AngleAxis initialisation")
r = nanoeigenpy.AngleAxis(q)
q2 = nanoeigenpy.Quaternion(r)
assert q == q
assert isapprox(q.coeffs(), q2.coeffs())
assert q2.isApprox(q2)
assert q2.isApprox(q2, 1e-2)

Rq = q.matrix()
Rr = r.matrix()
assert isapprox(Rq.dot(Rq.T), np.eye(3))
assert isapprox(Rr, Rq)

verbose and print("[Quaternion] Rotation Matrix initialisation")
qR = nanoeigenpy.Quaternion(Rr)
assert q.isApprox(qR)
assert isapprox(q.coeffs(), qR.coeffs())

assert isapprox(qR[3], 1.0 / np.sqrt(30))
try:
    qR[5]
    print("Error, this message should not appear.")
except IndexError as e:
    if verbose:
        print("As expected, caught exception: ", e)

x = quaternion.X(q)
assert x.a == q

# --- Angle Vector ------------------------------------------------
r = nanoeigenpy.AngleAxis(0.1, np.array([1, 0, 0], np.double))
if verbose:
    print("Rx(.1) = \n\n", r.matrix(), "\n")
assert isapprox(r.matrix()[2, 2], cos(r.angle))
assert isapprox(r.axis, np.array([1.0, 0, 0]))
assert isapprox(r.angle, 0.1)
assert r.isApprox(r)
assert r.isApprox(r, 1e-2)

r.axis = np.array([0, 1, 0], np.double).T
assert isapprox(r.matrix()[0, 0], cos(r.angle))

ri = r.inverse()
assert isapprox(ri.angle, -0.1)

R = r.matrix()
r2 = nanoeigenpy.AngleAxis(np.dot(R, R))
assert isapprox(r2.angle, r.angle * 2)

# --- Hyperplane ------------------------------------------------
verbose and print("[Hyperplane] Normal and point construction")
n = np.array([1.0, 0.0])
p = np.array([2.0, 3.0])
h = nanoeigenpy.Hyperplane(n, p)
assert isapprox(h.normal(), n)
assert isapprox(h.absDistance(p), 0.0)
assert h.dim() == 2

verbose and print("[Hyperplane] Normal and distance construction")
d = -np.dot(n, p)
h2 = nanoeigenpy.Hyperplane(n, d)
assert isapprox(h.coeffs(), h2.coeffs())
assert isapprox(h2.offset(), d)

verbose and print("[Hyperplane] Through two points")
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 1.0])
h3 = nanoeigenpy.Hyperplane.Through(p1, p2)
assert isapprox(h3.absDistance(p1), 0.0)
assert isapprox(h3.absDistance(p2), 0.0)
assert isapprox(np.linalg.norm(h3.normal()), 1.0)

verbose and print("[Hyperplane] Through three points")
p1_3d = np.array([1.0, 0.0, 0.0])
p2_3d = np.array([0.0, 1.0, 0.0])
p3_3d = np.array([0.0, 0.0, 1.0])
h4 = nanoeigenpy.Hyperplane.Through(p1_3d, p2_3d, p3_3d)
assert isapprox(h4.absDistance(p1_3d), 0.0)
assert isapprox(h4.absDistance(p2_3d), 0.0)
assert isapprox(h4.absDistance(p3_3d), 0.0)
assert isapprox(np.linalg.norm(h4.normal()), 1.0)
assert h4.dim() == 3

verbose and print("[Hyperplane] Distance calculations")
test_point = np.array([1.0, 0.0])
signed_dist = h3.signedDistance(test_point)
abs_dist = h3.absDistance(test_point)
assert isapprox(abs_dist, abs(signed_dist))

verbose and print("[Hyperplane] Projection")
proj = h3.projection(test_point)
assert isapprox(h3.absDistance(proj), 0.0)

verbose and print("[Hyperplane] Normalization")
h_copy = nanoeigenpy.Hyperplane(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
h_copy.normalize()
assert isapprox(np.linalg.norm(h_copy.normal()), 1.0)

verbose and print("[Hyperplane] Line intersection")
h_line1 = nanoeigenpy.Hyperplane(np.array([1.0, 0.0]), 0.0)
h_line2 = nanoeigenpy.Hyperplane(np.array([0.0, 1.0]), 0.0)
intersection = h_line1.intersection(h_line2)
assert isapprox(intersection, np.array([0.0, 0.0]))

verbose and print("[Hyperplane] isApprox")
h5 = nanoeigenpy.Hyperplane(h)
assert h.isApprox(h5)
assert h.isApprox(h5, 1e-12)

# --- ParametrizedLine ------------------------------------------------
verbose and print("[ParametrizedLine] Origin and direction construction")
origin = np.array([1.0, 2.0])
direction = np.array([1.0, 0.0])
line = nanoeigenpy.ParametrizedLine(origin, direction)
assert isapprox(line.origin(), origin)
assert isapprox(line.direction(), direction)
assert line.dim() == 2

verbose and print("[ParametrizedLine] Default constructor")
line_default = nanoeigenpy.ParametrizedLine()
assert line_default.dim() == 0

verbose and print("[ParametrizedLine] Dimension constructor")
line_3d = nanoeigenpy.ParametrizedLine(3)
assert line_3d.dim() == 3

verbose and print("[ParametrizedLine] Copy constructor")
line_copy = nanoeigenpy.ParametrizedLine(line)
assert isapprox(line_copy.origin(), line.origin())
assert isapprox(line_copy.direction(), line.direction())

verbose and print("[ParametrizedLine] Construction from 2D hyperplane")
h_2d = nanoeigenpy.Hyperplane(np.array([1.0, 0.0]), 0.0)
line_from_h = nanoeigenpy.ParametrizedLine(h_2d)
assert line_from_h.dim() == 2
assert isapprox(line_from_h.origin(), np.array([0.0, 0.0]))
assert isapprox(line_from_h.direction(), np.array([0.0, 1.0]))

verbose and print("[ParametrizedLine] 3D hyperplane should fail")
h_3d = nanoeigenpy.Hyperplane(np.array([1.0, 0.0, 0.0]), 0.0)
try:
    line_fail = nanoeigenpy.ParametrizedLine(h_3d)
    print("Error, this message should not appear.")
except ValueError as e:
    if verbose:
        print("As expected, caught exception:", e)

verbose and print("[ParametrizedLine] Distance calculations")
test_point = np.array([1.0, 0.0])
line_x_axis = nanoeigenpy.ParametrizedLine(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
distance = line_x_axis.distance(test_point)
squared_distance = line_x_axis.squaredDistance(test_point)
assert isapprox(distance, 0.0)
assert isapprox(squared_distance, 0.0)

off_line_point = np.array([1.0, 1.0])
distance_off = line_x_axis.distance(off_line_point)
squared_distance_off = line_x_axis.squaredDistance(off_line_point)
assert isapprox(distance_off, 1.0)
assert isapprox(squared_distance_off, 1.0)
assert isapprox(distance_off * distance_off, squared_distance_off)

verbose and print("[ParametrizedLine] Projection")
projection = line_x_axis.projection(off_line_point)
assert isapprox(projection, np.array([1.0, 0.0]))
assert isapprox(line_x_axis.distance(projection), 0.0)

verbose and print("[ParametrizedLine] Intersection with hyperplane")
line_diagonal = nanoeigenpy.ParametrizedLine(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
h_vertical = nanoeigenpy.Hyperplane(np.array([1.0, 0.0]), -1.0)

intersection_param = line_diagonal.intersectionParameter(h_vertical)
assert isapprox(intersection_param, 1.0)

intersection_param_old = line_diagonal.intersection(h_vertical)
assert isapprox(intersection_param_old, intersection_param)

intersection_point = line_diagonal.intersectionPoint(h_vertical)
expected_intersection = np.array([1.0, 1.0])
assert isapprox(intersection_point, expected_intersection)
assert isapprox(h_vertical.absDistance(intersection_point), 0.0)

verbose and print("[ParametrizedLine] isApprox")
line1 = nanoeigenpy.ParametrizedLine(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
line2 = nanoeigenpy.ParametrizedLine(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
line3 = nanoeigenpy.ParametrizedLine(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
assert line1.isApprox(line2)
assert line1.isApprox(line2, 1e-12)
assert not line1.isApprox(line3)

verbose and print("[ParametrizedLine] Parallel lines")
line_parallel1 = nanoeigenpy.ParametrizedLine(
    np.array([0.0, 0.0]), np.array([1.0, 0.0])
)
line_parallel2 = nanoeigenpy.ParametrizedLine(
    np.array([0.0, 1.0]), np.array([1.0, 0.0])
)
assert not line_parallel1.isApprox(line_parallel2)

test_points = [np.array([i, 0.0]) for i in range(5)]
distances = [line_parallel2.distance(p) for p in test_points]
for d in distances:
    assert isapprox(d, 1.0)

verbose and print("[ParametrizedLine] Through two points")
p0 = np.array([0.0, 0.0])
p1 = np.array([1.0, 1.0])
line_through = nanoeigenpy.ParametrizedLine.Through(p0, p1)
direction = line_through.direction()
expected_dir = (p1 - p0) / np.linalg.norm(p1 - p0)
assert isapprox(line_through.origin(), p0)
assert isapprox(np.linalg.norm(direction), 1.0)
assert isapprox(direction, expected_dir)

# --- Rotation2D ------------------------------------------------
verbose and print("[Rotation2D] Default constructor")
r_default = nanoeigenpy.Rotation2D()
assert isapprox(r_default.angle, 0.0)

verbose and print("[Rotation2D] Angle constructor")
angle = np.pi / 4
r_angle = nanoeigenpy.Rotation2D(angle)
assert isapprox(r_angle.angle, angle)

verbose and print("[Rotation2D] Copy constructor")
r_copy = nanoeigenpy.Rotation2D(r_angle)
assert isapprox(r_copy.angle, r_angle.angle)
assert r_copy == r_angle

verbose and print("[Rotation2D] Matrix constructor")
theta = np.pi / 6
rotation_matrix = np.array(
    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
)
r_matrix = nanoeigenpy.Rotation2D(rotation_matrix)
assert isapprox(r_matrix.angle, theta)

verbose and print("[Rotation2D] Angle property")
r_prop = nanoeigenpy.Rotation2D()
new_angle = np.pi / 3
r_prop.angle = new_angle
assert isapprox(r_prop.angle, new_angle)

verbose and print("[Rotation2D] smallestPositiveAngle")
r_negative = nanoeigenpy.Rotation2D(-np.pi / 4)
positive_angle = r_negative.smallestPositiveAngle()
assert positive_angle >= 0.0
assert positive_angle < 2 * np.pi
assert isapprox(positive_angle, 7 * np.pi / 4)

verbose and print("[Rotation2D] smallestAngle")
r_large = nanoeigenpy.Rotation2D(3 * np.pi)
smallest_angle = r_large.smallestAngle()
assert smallest_angle >= -np.pi
assert smallest_angle <= np.pi
assert isapprox(smallest_angle, np.pi)

verbose and print("[Rotation2D] Identity")
r_identity = nanoeigenpy.Rotation2D.Identity()
assert isapprox(r_identity.angle, 0.0)

verbose and print("[Rotation2D] fromRotationMatrix")
r_from_matrix = nanoeigenpy.Rotation2D()
theta2 = np.pi / 2
matrix2 = np.array(
    [[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]]
)
r_from_matrix.fromRotationMatrix(matrix2)
assert isapprox(r_from_matrix.angle, theta2)

verbose and print("[Rotation2D] Rotation composition")
r1 = nanoeigenpy.Rotation2D(np.pi / 4)
r2 = nanoeigenpy.Rotation2D(np.pi / 6)
r_composed = r1 * r2
expected_angle = np.pi / 4 + np.pi / 6
assert isapprox(r_composed.angle, expected_angle)

verbose and print("[Rotation2D] In-place multiplication")
r_inplace = nanoeigenpy.Rotation2D(np.pi / 4)
original_angle = r_inplace.angle
r_inplace *= nanoeigenpy.Rotation2D(np.pi / 6)
assert isapprox(r_inplace.angle, original_angle + np.pi / 6)

verbose and print("[Rotation2D] Vector rotation")
r_90 = nanoeigenpy.Rotation2D(np.pi / 2)
vec = np.array([1.0, 0.0])
rotated_vec = r_90 * vec
expected_vec = np.array([0.0, 1.0])
assert isapprox(rotated_vec, expected_vec)

vec2 = np.array([1.0, 1.0])
r_45 = nanoeigenpy.Rotation2D(np.pi / 4)
rotated_vec2 = r_45 * vec2
expected_vec2 = np.array([0.0, np.sqrt(2)])
assert isapprox(rotated_vec2, expected_vec2)

verbose and print("[Rotation2D] Equality operators")
r_eq1 = nanoeigenpy.Rotation2D(np.pi / 3)
r_eq2 = nanoeigenpy.Rotation2D(np.pi / 3)
r_eq3 = nanoeigenpy.Rotation2D(np.pi / 4)

assert r_eq1 == r_eq2
assert not (r_eq1 == r_eq3)
assert r_eq1 != r_eq3
assert not (r_eq1 != r_eq2)

verbose and print("[Rotation2D] Periodic angles")
r_period1 = nanoeigenpy.Rotation2D(0.0)
r_period2 = nanoeigenpy.Rotation2D(2 * np.pi)
verbose and print("[Rotation2D] isApprox")
r_approx1 = nanoeigenpy.Rotation2D(np.pi / 4)
r_approx2 = nanoeigenpy.Rotation2D(np.pi / 4 + 1e-15)
r_approx3 = nanoeigenpy.Rotation2D(np.pi / 3)

assert r_approx1.isApprox(r_approx2)
assert r_approx1.isApprox(r_approx2, 1e-12)
assert not r_approx1.isApprox(r_approx3)

verbose and print("[Rotation2D] slerp")
r_start = nanoeigenpy.Rotation2D(0.0)
r_end = nanoeigenpy.Rotation2D(np.pi / 2)
r_middle = r_start.slerp(0.5, r_end)
assert isapprox(r_middle.angle, np.pi / 4)

r_slerp_0 = r_start.slerp(0.0, r_end)
r_slerp_1 = r_start.slerp(1.0, r_end)
assert isapprox(r_slerp_0.angle, r_start.angle)
assert isapprox(r_slerp_1.angle, r_end.angle)

verbose and print("[Rotation2D] Inverse rotation")
try:
    r_original = nanoeigenpy.Rotation2D(np.pi / 3)
    r_inverse = r_original.inverse()
    assert isapprox(r_inverse.angle, -np.pi / 3)

    r_identity_test = r_original * r_inverse
    assert isapprox(r_identity_test.angle, 0.0, 1e-12)
except AttributeError:
    if verbose:
        print("inverse() method not exposed or not available")

verbose and print("[Rotation2D] Matrix conversion")
try:
    r_matrix_test = nanoeigenpy.Rotation2D(np.pi / 6)
    matrix = r_matrix_test.matrix()

    assert matrix.shape == (2, 2)
    assert isapprox(matrix @ matrix.T, np.eye(2))
    assert isapprox(np.linalg.det(matrix), 1.0)

    expected_matrix = np.array(
        [
            [np.cos(np.pi / 6), -np.sin(np.pi / 6)],
            [np.sin(np.pi / 6), np.cos(np.pi / 6)],
        ]
    )
    assert isapprox(matrix, expected_matrix)

except AttributeError:
    if verbose:
        print("matrix() method not exposed or not available")

verbose and print("[Rotation2D] Angle normalization")
r_large_angle = nanoeigenpy.Rotation2D(3 * np.pi)
vec_test = np.array([1.0, 0.0])
rotated_large = r_large_angle * vec_test
expected_large = np.array([-1.0, 0.0])
assert isapprox(rotated_large, expected_large)

# --- UniformScaling ------------------------------------------------
verbose and print("[UniformScaling] Default constructor")
s_default = nanoeigenpy.UniformScaling()

verbose and print("[UniformScaling] Factor constructor")
factor = 2.5
s_factor = nanoeigenpy.UniformScaling(factor)
assert isapprox(s_factor.factor(), factor)

verbose and print("[UniformScaling] Copy constructor")
s_copy = nanoeigenpy.UniformScaling(s_factor)
assert isapprox(s_copy.factor(), s_factor.factor())

verbose and print("[UniformScaling] Factor getter")
s_test = nanoeigenpy.UniformScaling(3.0)
assert isapprox(s_test.factor(), 3.0)

verbose and print("[UniformScaling] Inverse scaling")
s_original = nanoeigenpy.UniformScaling(4.0)
s_inverse = s_original.inverse()
assert isapprox(s_inverse.factor(), 1.0 / 4.0)

s_identity_test = s_original * s_inverse
assert isapprox(s_identity_test.factor(), 1.0)

verbose and print("[UniformScaling] Concatenation of scalings")
s1 = nanoeigenpy.UniformScaling(2.0)
s2 = nanoeigenpy.UniformScaling(3.0)
s_combined = s1 * s2
assert isapprox(s_combined.factor(), 6.0)

verbose and print("[UniformScaling] Multiplication with matrix")
s_scale = nanoeigenpy.UniformScaling(2.0)
matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
scaled_matrix = s_scale * matrix
expected_matrix = matrix * 2.0
assert isapprox(scaled_matrix, expected_matrix)

identity = np.eye(3)
s_identity_scale = nanoeigenpy.UniformScaling(5.0)
scaled_identity = s_identity_scale * identity
expected_identity = identity * 5.0
assert isapprox(scaled_identity, expected_identity)

verbose and print("[UniformScaling] Multiplication with AngleAxis")
try:
    angle_axis = nanoeigenpy.AngleAxis(np.pi / 4, np.array([0.0, 0.0, 1.0]))
    s_with_rotation = nanoeigenpy.UniformScaling(2.0)
    result_rotation = s_with_rotation * angle_axis

    assert result_rotation.shape == (3, 3)
    det = np.linalg.det(result_rotation)
    assert isapprox(det, 2.0**3)

except (AttributeError, NameError):
    if verbose:
        print("AngleAxis class not available or not exposed")

verbose and print("[UniformScaling] Multiplication with Quaternion")
try:
    quat = nanoeigenpy.Quaternion(1, 0, 0, 0)
    s_with_quat = nanoeigenpy.UniformScaling(3.0)
    result_quat = s_with_quat * quat

    assert result_quat.shape == (3, 3)
    expected_scaled_identity = np.eye(3) * 3.0
    assert isapprox(result_quat, expected_scaled_identity)

except (AttributeError, NameError):
    if verbose:
        print("Quaternion class not available or not exposed")

verbose and print("[UniformScaling] Multiplication with Rotation2D")
try:
    rotation_2d = nanoeigenpy.Rotation2D(np.pi / 4)
    s_with_rot2d = nanoeigenpy.UniformScaling(2.0)
    result_rot2d = s_with_rot2d * rotation_2d

    assert result_rot2d.shape == (2, 2)
    det_2d = np.linalg.det(result_rot2d)
    assert isapprox(det_2d, 2.0**2)

except (AttributeError, NameError):
    if verbose:
        print("Rotation2D class not available or not exposed")

verbose and print("[UniformScaling] isApprox")
s_approx1 = nanoeigenpy.UniformScaling(2.0)
s_approx2 = nanoeigenpy.UniformScaling(2.0 + 1e-15)
s_approx3 = nanoeigenpy.UniformScaling(3.0)

assert s_approx1.isApprox(s_approx2)
assert s_approx1.isApprox(s_approx2, 1e-12)
assert not s_approx1.isApprox(s_approx3)

verbose and print("[UniformScaling] Edge cases")
s_zero = nanoeigenpy.UniformScaling(0.0)
assert isapprox(s_zero.factor(), 0.0)
try:
    s_zero_inverse = s_zero.inverse()
    if verbose:
        print("Zero scaling inverse:", s_zero_inverse.factor())
except Exception as e:
    if verbose:
        print("Zero scaling inverse threw exception (expected):", type(e).__name__)

s_negative = nanoeigenpy.UniformScaling(-2.0)
assert isapprox(s_negative.factor(), -2.0)
s_negative_inverse = s_negative.inverse()
assert isapprox(s_negative_inverse.factor(), -0.5)

s_small = nanoeigenpy.UniformScaling(1e-10)
s_small_inverse = s_small.inverse()
assert isapprox(s_small_inverse.factor(), 1e10)

verbose and print("[UniformScaling] Chain operations")
s_chain1 = nanoeigenpy.UniformScaling(2.0)
s_chain2 = nanoeigenpy.UniformScaling(3.0)
s_chain3 = nanoeigenpy.UniformScaling(4.0)

left_assoc = (s_chain1 * s_chain2) * s_chain3
right_assoc = s_chain1 * (s_chain2 * s_chain3)
assert isapprox(left_assoc.factor(), right_assoc.factor())
assert isapprox(left_assoc.factor(), 24.0)

verbose and print("[UniformScaling] Vector scaling")
s_vector = nanoeigenpy.UniformScaling(2.0)
vector = np.array([[1.0], [2.0], [3.0]])
identity_3x3 = np.eye(3)
scaled_identity = s_vector * identity_3x3
scaled_vector = scaled_identity @ vector
expected_vector = vector * 2.0
assert isapprox(scaled_vector, expected_vector)

# --- Translation ------------------------------------------------
verbose and print("[Translation] Default constructor")
t_default = nanoeigenpy.Translation()

verbose and print("[Translation] 2D constructor with vector")
t_2d = nanoeigenpy.Translation(np.array([1.0, 2.0]))
assert isapprox(t_2d.x, 1.0)
assert isapprox(t_2d.y, 2.0)

verbose and print("[Translation] 3D constructor with vector")
t_3d = nanoeigenpy.Translation(np.array([1.0, 2.0, 3.0]))
assert isapprox(t_3d.x, 1.0)
assert isapprox(t_3d.y, 2.0)
assert isapprox(t_3d.z, 3.0)

verbose and print("[Translation] Vector constructor")
vector = np.array([1.5, 2.5, 3.5])
t_vector = nanoeigenpy.Translation(vector)
assert isapprox(t_vector.x, 1.5)
assert isapprox(t_vector.y, 2.5)
assert isapprox(t_vector.z, 3.5)

verbose and print("[Translation] Copy constructor")
t_copy = nanoeigenpy.Translation(t_3d)
assert isapprox(t_copy.x, t_3d.x)
assert isapprox(t_copy.y, t_3d.y)
assert isapprox(t_copy.z, t_3d.z)

verbose and print("[Translation] Property setters")
t_test = nanoeigenpy.Translation(np.array([1.0, 2.0, 3.0]))
t_test.x = 10.0
t_test.y = 20.0
t_test.z = 30.0
assert isapprox(t_test.x, 10.0)
assert isapprox(t_test.y, 20.0)
assert isapprox(t_test.z, 30.0)

verbose and print("[Translation] Vector and translation getters")
vector_result = t_test.vector()
translation_result = t_test.translation()
assert isapprox(vector_result[0], 10.0)
assert isapprox(translation_result[0], 10.0)

verbose and print("[Translation] Inverse")
t_original = nanoeigenpy.Translation(np.array([2.0, 3.0, 4.0]))
t_inverse = t_original.inverse()
assert isapprox(t_inverse.x, -2.0)
assert isapprox(t_inverse.y, -3.0)
assert isapprox(t_inverse.z, -4.0)

verbose and print("[Translation] Concatenation")
t1 = nanoeigenpy.Translation(np.array([1.0, 2.0, 3.0]))
t2 = nanoeigenpy.Translation(np.array([4.0, 5.0, 6.0]))
t_combined = t1 * t2
assert isapprox(t_combined.x, 5.0)
assert isapprox(t_combined.y, 7.0)
assert isapprox(t_combined.z, 9.0)

verbose and print("[Translation] isApprox")
t_approx1 = nanoeigenpy.Translation(np.array([1.0, 2.0, 3.0]))
t_approx2 = nanoeigenpy.Translation(np.array([1.0 + 1e-15, 2.0 + 1e-15, 3.0 + 1e-15]))
t_approx3 = nanoeigenpy.Translation(np.array([1.1, 2.1, 3.1]))
assert t_approx1.isApprox(t_approx2)
assert t_approx1.isApprox(t_approx2, 1e-12)
assert not t_approx1.isApprox(t_approx3)

# --- JacobiRotation ---------------------------------------------------------------
verbose and print("[JacobiRotation] Default constructor")
j = nanoeigenpy.JacobiRotation()
assert hasattr(j, "c")
assert hasattr(j, "s")

verbose and print("[JacobiRotation] Cosine-sine constructor")
c_val = 0.8
s_val = 0.6
j = nanoeigenpy.JacobiRotation(c_val, s_val)
assert isapprox(j.c, c_val)
assert isapprox(j.s, s_val)

verbose and print("[JacobiRotation] Property access")
j.c = 0.8
j.s = 0.6
assert isapprox(j.c, 0.8)
assert isapprox(j.s, 0.6)
norm_squared = j.c**2 + j.s**2
assert isapprox(norm_squared, 1.0, 1e-12)

verbose and print("[JacobiRotation] Multiplication operator")
j1 = nanoeigenpy.JacobiRotation(0.8, 0.6)
j2 = nanoeigenpy.JacobiRotation(0.6, 0.8)
j_mult = j1 * j2
assert hasattr(j_mult, "c")
assert hasattr(j_mult, "s")
norm_mult = j_mult.c**2 + j_mult.s**2
assert isapprox(norm_mult, 1.0, 1e-12)

verbose and print("[JacobiRotation] Transpose")
j = nanoeigenpy.JacobiRotation(0.8, 0.6)
j_t = j.transpose()
assert isapprox(j_t.c, j.c)
assert isapprox(j_t.s, -j.s)

verbose and print("[JacobiRotation] Adjoint")
j = nanoeigenpy.JacobiRotation(0.8, 0.6)
j_adj = j.adjoint()
assert isapprox(j_adj.c, j.c)
assert isapprox(j_adj.s, -j.s)

verbose and print("[JacobiRotation] Identity property")
j = nanoeigenpy.JacobiRotation(0.8, 0.6)
j_t = j.transpose()
identity = j * j_t
assert isapprox(identity.c, 1.0, 1e-12)
assert isapprox(identity.s, 0.0, 1e-12)

verbose and print("[JacobiRotation] makeJacobi from scalars")
j = nanoeigenpy.JacobiRotation()
x, z = 4.0, 1.0
y = 2.0
result = j.makeJacobi(x, y, z)
assert isinstance(result, bool)
norm_after = j.c**2 + j.s**2
assert isapprox(norm_after, 1.0, 1e-12)

verbose and print("[JacobiRotation] makeJacobi from matrix")
M = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 1.0]])
j = nanoeigenpy.JacobiRotation()
result = j.makeJacobi(M, 0, 1)
assert isinstance(result, bool)
norm_matrix = j.c**2 + j.s**2
assert isapprox(norm_matrix, 1.0, 1e-12)

verbose and print("[JacobiRotation] makeGivens basic")
j = nanoeigenpy.JacobiRotation()
p_val = 3.0
q_val = 4.0
j.makeGivens(p_val, q_val)
norm_givens = j.c**2 + j.s**2
assert isapprox(norm_givens, 1.0, 1e-12)

verbose and print("[JacobiRotation] makeGivens with r parameter")
j = nanoeigenpy.JacobiRotation()
p_val = 3.0
q_val = 4.0
r_container = np.array([0.0])
j.makeGivens(p_val, q_val, r_container.ctypes.data)
expected_r = np.sqrt(p_val**2 + q_val**2)

verbose and print("[JacobiRotation] Edge cases")
j_zero = nanoeigenpy.JacobiRotation(1.0, 0.0)
assert isapprox(j_zero.c, 1.0)
assert isapprox(j_zero.s, 0.0)

j_90 = nanoeigenpy.JacobiRotation(0.0, 1.0)
assert isapprox(j_90.c, 0.0)
assert isapprox(j_90.s, 1.0)

j = nanoeigenpy.JacobiRotation()
j.makeGivens(5.0, 0.0)
assert isapprox(abs(j.c), 1.0)
assert isapprox(j.s, 0.0)

j.makeGivens(0.0, 5.0)
assert isapprox(j.c, 0.0)
assert isapprox(abs(j.s), 1.0)

verbose and print("[JacobiRotation] makeJacobi small off-diagonal")
j = nanoeigenpy.JacobiRotation()
result = j.makeJacobi(1.0, 1e-15, 2.0)
assert isinstance(result, bool)
if not result:
    assert isapprox(j.c, 1.0)
    assert isapprox(j.s, 0.0)
