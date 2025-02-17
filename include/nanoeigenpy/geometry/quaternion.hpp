/// Copyright 2025 INRIA
#pragma once

#include "detail/rotation-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

/// Visitor for Eigen Quaternion types.
template <typename Quaternion>
struct QuaternionVisitor : nb::def_visitor<QuaternionVisitor<Quaternion>> {
  using Class = Quaternion;
  using QuaternionBase = Eigen::QuaternionBase<Quaternion>;
  static_assert(std::is_base_of_v<QuaternionBase, Quaternion>);
  using Scalar = typename QuaternionBase::Scalar;
  using Vector3 = typename QuaternionBase::Vector3;
  using Matrix3 = typename QuaternionBase::Coefficients;
  using AngleAxisType = typename QuaternionBase::AngleAxisType;

  static Quaternion& setFromTwoVectors(Quaternion& self,
                                       Eigen::Ref<const Vector3> a,
                                       Eigen::Ref<const Vector3> b) {
    return self.setFromTwoVectors(a, b);
  }

  template <typename... Ts>
  void execute(nb::class_<Class, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(nb::init<>())
        .def(nb::init<Scalar, Scalar, Scalar, Scalar>(), "x"_a, "y"_a, "z"_a,
             "w"_a)
        .def(nb::init<AngleAxisType>(), "aa"_a)
        .def("setFromTwoVectors", setFromTwoVectors, "a"_a, "b"_a)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self, nb::rv_policy::none)
        .def(nb::self * Vector3())
        .def("__repr__",
             [](const Quaternion& self) {
               std::stringstream ss;
               ss << "[x,y,z,w] = " << self.coeffs().transpose();
               return ss.str();
             })
        .def(RotationBaseVisitor<Quaternion, 3>());
  }

  static void expose(nb::module_& m, const char* name) {
    nb::class_<Quaternion>(m, name).def(QuaternionVisitor());
  }
};

template <typename Scalar>
void exposeQuaternion(nb::module_& m, const char* name) {
  using Quaternion = Eigen::Quaternion<Scalar>;
  QuaternionVisitor<Quaternion>::expose(m, name);
}

}  // namespace nanoeigenpy
