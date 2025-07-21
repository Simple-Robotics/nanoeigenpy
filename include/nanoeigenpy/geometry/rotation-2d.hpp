/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "detail/rotation-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(
    const Eigen::Rotation2D<Scalar> &r, const Eigen::Rotation2D<Scalar> &other,
    const Scalar &prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return r.isApprox(other, prec);
}

template <typename Scalar>
void exposeRotation2D(nb::module_ m, const char *name) {
  using namespace nb::literals;
  using Rotation2D = Eigen::Rotation2D<Scalar>;
  using Vector2 = typename Rotation2D::Vector2;
  using Matrix2 = typename Rotation2D::Matrix2;

  if (check_registration_alias<Rotation2D>(m)) {
    return;
  }
  nb::class_<Rotation2D>(
      m, name, "Represents a rotation/orientation in a 2 dimensional space.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Scalar &>(), "a"_a,
           "Construct a 2D counter clock wise rotation from the angle \a a in "
           "radian.")
      .def(nb::init<const Matrix2 &>(), "mat"_a,
           "Construct a 2D rotation from a 2x2 rotation matrix \a mat.")
      .def(nb::init<const Rotation2D &>(), "copy"_a, "Copy constructor.")

      .def_prop_rw(
          "angle", [](const Rotation2D &r) -> Scalar { return r.angle(); },
          [](Rotation2D &r, Scalar a) { r.angle() = a; }, "The rotation angle.")

      .def("smallestPositiveAngle", &Rotation2D::smallestPositiveAngle,
           "Returns the rotation angle in [0,2pi]")
      .def("smallestAngle", &Rotation2D::smallestAngle,
           "Returns the rotation angle in [-pi,pi]")

      .def(RotationBaseVisitor<Rotation2D, 2>())

      .def(
          "fromRotationMatrix",
          [](Rotation2D &r, const Matrix2 &mat) -> Rotation2D & {
            return r.fromRotationMatrix(mat);
          },
          "mat"_a, "Sets *this from a 2x2 rotation matrix",
          nb::rv_policy::reference_internal)
      .def("toRotationMatrix", &Rotation2D::toRotationMatrix)

      .def(
          "slerp",
          [](const Rotation2D &self, const Scalar t, const Rotation2D &other)
              -> Rotation2D { return self.slerp(t, other); },
          "t"_a, "other"_a,
          "Returns the spherical interpolation between *this and \a other using"
          "parameter \a t. It is in fact equivalent to a linear interpolation.")

      .def_static("Identity", &Rotation2D::Identity)

      .def(
          "isApprox",
          [](const Rotation2D &r, const Rotation2D &other,
             const Scalar &prec) -> bool { return isApprox(r, other, prec); },
          "other"_a, "prec"_a,
          "Returns true if *this is approximately equal to other, "
          "within the precision determined by prec.")
      .def(
          "isApprox",
          [](const Rotation2D &r, const Rotation2D &other) -> bool {
            return isApprox(r, other);
          },
          "other"_a,
          "Returns true if *this is approximately equal to other, "
          "within the default precision.")

      .def(
          "__mul__",
          [](const Rotation2D &self, const Rotation2D &other) -> Rotation2D {
            return self * other;
          },
          "other"_a, "Concatenates two rotations")
      .def(
          "__imul__",
          [](Rotation2D &self, const Rotation2D &other) -> Rotation2D & {
            return self *= other;
          },
          "other"_a, "Concatenates two rotations in-place",
          nb::rv_policy::reference_internal)
      .def(
          "__mul__",
          [](const Rotation2D &self, const Vector2 &vec) -> Vector2 {
            return self * vec;
          },
          "vec"_a, "Applies the rotation to a 2D vector")

      .def(
          "__eq__",
          [](const Rotation2D &self, const Rotation2D &other) -> bool {
            return std::abs(self.angle() - other.angle()) < 1e-12;
          },
          "other"_a, "Tests equality")
      .def(
          "__ne__",
          [](const Rotation2D &self, const Rotation2D &other) -> bool {
            return std::abs(self.angle() - other.angle()) >= 1e-12;
          },
          "other"_a, "Tests inequality")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
