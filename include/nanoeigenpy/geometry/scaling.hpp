/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "detail/rotation-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(
    const Eigen::UniformScaling<Scalar>& r,
    const Eigen::UniformScaling<Scalar>& other,
    const Scalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return r.isApprox(other, prec);
}

template <typename Scalar>
void exposeUniformScaling(nb::module_ m, const char* name) {
  using namespace nb::literals;
  using UniformScaling = Eigen::UniformScaling<Scalar>;

  if (check_registration_alias<UniformScaling>(m)) {
    return;
  }
  nb::class_<UniformScaling>(
      m, name, "Represents a generic uniform scaling transformation.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Scalar&>(), "s"_a,
           "Constructs and initialize a uniform scaling transformation")
      .def(nb::init<const UniformScaling&>(), "copy"_a, "Copy constructor.")

      .def(
          "factor",
          [](const UniformScaling& self) -> const Scalar& {
            return self.factor();
          },
          nb::rv_policy::reference_internal, "Returns the scaling factor")
      .def("inverse", &UniformScaling::inverse, "Returns the inverse scaling")

      .def(
          "isApprox",
          [](const UniformScaling& r, const UniformScaling& other,
             const Scalar& prec) -> bool { return isApprox(r, other, prec); },
          "other"_a, "prec"_a,
          "Returns true if *this is approximately equal to other, "
          "within the precision determined by prec.")
      .def(
          "isApprox",
          [](const UniformScaling& r, const UniformScaling& other) -> bool {
            return isApprox(r, other);
          },
          "other"_a,
          "Returns true if *this is approximately equal to other, "
          "within the default precision.")

      .def(
          "__mul__",
          [](const UniformScaling& self, const UniformScaling& other)
              -> UniformScaling { return self * other; },
          "other"_a, "Concatenates two uniform scalings")

      .def(
          "__mul__",
          [](const UniformScaling& self, const Eigen::MatrixXd& matrix)
              -> Eigen::MatrixXd { return self * matrix; },
          "matrix"_a, "Multiplies uniform scaling with a matrix")

      .def(
          "__mul__",
          [](const UniformScaling& self, const Eigen::AngleAxis<Scalar>& r)
              -> Eigen::Matrix<Scalar, 3, 3> { return self * r; },
          "rotation"_a, "Multiplies uniform scaling with AngleAxis rotation")

      .def(
          "__mul__",
          [](const UniformScaling& self, const Eigen::Quaternion<Scalar>& q)
              -> Eigen::Matrix<Scalar, 3, 3> { return self * q; },
          "quaternion"_a, "Multiplies uniform scaling with quaternion")

      .def(
          "__mul__",
          [](const UniformScaling& self, const Eigen::Rotation2D<Scalar>& r)
              -> Eigen::Matrix<Scalar, 2, 2> { return self * r; },
          "rotation2d"_a, "Multiplies uniform scaling with 2D rotation")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
