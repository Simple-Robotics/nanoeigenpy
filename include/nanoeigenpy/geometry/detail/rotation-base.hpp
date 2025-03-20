/// Copyright 2025 INRIA

#pragma once

#include <nanobind/eigen/dense.h>
#include <Eigen/Geometry>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Rotation, int Dim>
struct RotationBaseVisitor
    : nb::def_visitor<RotationBaseVisitor<Rotation, Dim>> {
  using RotationBase = Eigen::RotationBase<Rotation, Dim>;
  static_assert(std::is_base_of_v<RotationBase, Rotation>);

  template <typename... Ts>
  void execute(nb::class_<Rotation, Ts...>& cl) {
    cl.def("toRotationMatrix", &Rotation::toRotationMatrix)
        .def("matrix", &Rotation::matrix)
        .def("inverse", &Rotation::inverse);
  }
};

}  // namespace nanoeigenpy
