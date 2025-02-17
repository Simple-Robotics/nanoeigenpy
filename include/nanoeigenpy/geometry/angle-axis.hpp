#pragma once

#include "detail/rotation-base.hpp"

namespace nanoeigenpy {

template <typename Scalar>
void exposeAngleAxis(nb::module_ &m, const char *name) {
  using namespace nb::literals;
  using AngleAxis = Eigen::AngleAxis<Scalar>;
  using Quat = typename AngleAxis::QuaternionType;
  using Vector3 = typename AngleAxis::Vector3;
  using Matrix3 = typename AngleAxis::Matrix3;

  nb::class_<AngleAxis>(m, name)
      .def(nb::init<>())
      .def(nb::init<const Scalar &, Vector3>(), "angle"_a, "axis"_a)
      .def(nb::init<Quat>(), "q"_a)
      .def(nb::init<Matrix3>(), "R"_a)
      .def_prop_rw(
          "angle", [](const AngleAxis &aa) { return aa.angle(); },
          [](AngleAxis &aa, Scalar a) { aa.angle() = a; })
      .def_prop_rw(
          "axis", [](const AngleAxis &aa) { return aa.axis(); },
          [](AngleAxis &aa, const Vector3 &v) { aa.axis() = v; })
      .def(RotationBaseVisitor<AngleAxis, 3>());
}

}  // namespace nanoeigenpy
