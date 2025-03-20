#pragma once

#include "detail/rotation-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(
    const Eigen::AngleAxis<Scalar> &aa, const Eigen::AngleAxis<Scalar> &other,
    const Scalar &prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return aa.isApprox(other, prec);
}

template <typename Scalar>
void exposeAngleAxis(nb::module_ m, const char *name) {
  using namespace nb::literals;
  using AngleAxis = Eigen::AngleAxis<Scalar>;
  using Quaternion = typename AngleAxis::QuaternionType;
  using Vector3 = typename AngleAxis::Vector3;
  using Matrix3 = typename AngleAxis::Matrix3;

  nb::class_<AngleAxis>(m, name, "Angle-axis representation of a 3D rotation.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Scalar &, const Vector3 &>(), "angle"_a, "axis"_a,
           "Construct from angle and axis.")
      .def(nb::init<const Matrix3 &>(), "R"_a,
           "Construct from a rotation matrix.")
      .def(nb::init<const Quaternion &>(), "quaternion"_a,
           "Construct from a quaternion.")
      .def(nb::init<const AngleAxis &>(), "copy"_a, "Copy constructor.")

      .def_prop_rw(
          "angle", [](const AngleAxis &aa) { return aa.angle(); },
          [](AngleAxis &aa, Scalar a) { aa.angle() = a; },
          "The rotation angle.")
      .def_prop_rw(
          "axis", [](const AngleAxis &aa) -> const auto & { return aa.axis(); },
          [](AngleAxis &aa, const Vector3 &v) { aa.axis() = v; },
          "The rotation axis.", nb::rv_policy::reference_internal)

      .def(RotationBaseVisitor<AngleAxis, 3>())
      .def(
          "fromRotationMatrix",
          [](AngleAxis &aa, const Matrix3 &mat) -> auto & {
            return aa.fromRotationMatrix(mat);
          },
          "mat"_a, "Sets *this from a 3x3 rotation matrix",
          nb::rv_policy::reference_internal)
      .def(
          "isApprox",
          [](const AngleAxis &aa, const AngleAxis &other,
             const Scalar &prec) -> bool { return isApprox(aa, other, prec); },
          "other"_a, "rec"_a,
          "Returns true if *this is approximately equal to other, "
          "within the precision determined by prec.")

      .def(nb::self * nb::self)
      .def(nb::self * Quaternion());
}

}  // namespace nanoeigenpy

// TODO

// Differences between angle-axis and decompositions files:
// Remove the isApprox and make it clean
// bp::implicitly_convertible -> Let it for later (eventually Justin work)
// Manage the overload of isApprox (cf the macro in eigenpy) (try with
// quaternion implemented too, add if needed (probably)) angle and axis
// functions: check if correct from eigenpy (probably not) See if clean to
// factorize with RotationBaseVisitor How to define the operators
