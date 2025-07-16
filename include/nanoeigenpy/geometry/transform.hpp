/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar, typename Transform>
bool isApprox(
    const Transform& r, const Transform& other,
    const Scalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return r.isApprox(other, prec);
}

template <typename Transform>
void exposeTransform(nb::module_ m, const char* name) {
  using namespace nb::literals;
  using Scalar = typename Transform::Scalar;
  using MatrixType = typename Transform::MatrixType;
  using TranslationType = Eigen::Translation<Scalar, Eigen::Dynamic>;
  using UniformScaling = Eigen::UniformScaling<Scalar>;
  using AngleAxis = Eigen::AngleAxis<Scalar>;
  using Quaternion = Eigen::Quaternion<Scalar>;
  using Rotation2D = Eigen::Rotation2D<Scalar>;

  if (check_registration_alias<Transform>(m)) {
    return;
  }
  nb::class_<Transform>(
      m, name,
      "Represents an homogeneous transformation in a N dimensional space.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Transform&>(), "t"_a, "Copy constructor.")
      // .def(nb::init<const TranslationType&>(), "t"_a) // Fail
      .def(nb::init<const UniformScaling&>(), "s"_a)
      // .def(nb::init<const Eigen::RotationBase<AngleAxis, 3>&>(), "aa"_a)
      // .def(nb::init<const Eigen::RotationBase<Quaternion, 3>&>(), "q"_a)
      // .def(nb::init<const Eigen::RotationBase<Rotation2D, 2>&>(), "r"_a)
      .def(
          "__init__",
          [](Transform* self, const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
            new (self) Transform(matrix);
          },
          "matrix"_a,
          "Initialize from a matrix. Accepts Dim x Dim or (Dim+1) x (Dim+1) "
          "matrices.")

      // .def("rows", &Transform::rows)
      // .def("cols", &Transform::cols)

      // .def(
      //     "matrix",
      //     [](const Transform &self) -> const MatrixType & {
      //       return self.matrix();
      //     },
      //     "Returns the transformation matrix.",
      //     nb::rv_policy::reference_internal)
      // .def(
      //     "linear",
      //     [](const Transform &self) { return self.linear(); },
      //     "Returns the linear part of the transformation.")
      // .def(
      //     "affine",
      //     [](const Transform &self) { return self.affine(); },
      //     "Returns the affine part of the transformation.")
      // .def(
      //     "translation",
      //     [](const Transform &self) { return self.translation(); },
      //     "Returns the translation vector of the transformation.")

      // .def("setIdentity", &Transform::setIdentity)
      // .def_static("Identity", &Transform::Identity)

      // .def(
      //     "scale",
      //     [](Transform& self, const Scalar& s) -> Transform & {
      //         return self.scale(s);
      //     },
      //     "s"_a,
      //     nb::rv_policy::reference)
      // .def(
      //     "scale",
      //     [](Transform& self, const MatrixType& other) -> Transform & {
      //         return self.scale(other);
      //     },
      //     "factors"_a,
      //     nb::rv_policy::reference)

      // .def(
      //     "prescale",
      //     [](Transform& self, const Scalar& s) -> Transform & {
      //         return self.prescale(s);
      //     },
      //     "s"_a,
      //     nb::rv_policy::reference)
      // .def(
      //     "prescale",
      //     [](Transform& self, const MatrixType& other) -> Transform & {
      //         return self.prescale(other);
      //     },
      //     "factors"_a,
      //     nb::rv_policy::reference)

      // .def("translate", &Transform::translate, "other"_a,
      //      nb::rv_policy::reference)
      // .def("pretranslate", &Transform::pretranslate, "other"_a,
      //      nb::rv_policy::reference)
      // .def("rotate", &Transform::rotate, "rotation"_a,
      //      nb::rv_policy::reference)
      // .def("prerotate", &Transform::prerotate, "rotation"_a,
      //      nb::rv_policy::reference)

      // .def("shear", &Transform::shear, "sx"_a, "sy"_a,
      //      nb::rv_policy::reference)
      // .def("preshear", &Transform::preshear, "sx"_a, "sy"_a,
      //      nb::rv_policy::reference)

      // .def("rotation", &Transform::rotation)

      // .def("computeRotationScaling", &Transform::computeRotationScaling,
      //      "rotation"_a, "scaling"_a)
      // .def("computeScalingRotation", &Transform::computeScalingRotation,
      //      "scaling"_a, "rotation"_a)
      // .def("fromPositionOrientationScale",
      // &Transform::fromPositionOrientationScale,
      //      "position"_a, "orientation"_a, "scale"_a,
      //      nb::rv_policy::reference)

      // .def("inverse", &Transform::inverse, "traits"_a)
      // .def("data", &Transform::data)

      // .def("makeAffine", &Transform::makeAffine)

      // // Operators

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
