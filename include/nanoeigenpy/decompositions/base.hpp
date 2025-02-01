/// Copyright 2025 INRIA
#pragma once
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

namespace nanoeigenpy {
namespace nb = nanobind;

struct EigenBaseVisitor : nb::def_visitor<EigenBaseVisitor> {
  template <typename Derived, typename... Ts>
  void execute(nb::class_<Derived, Ts...> &cl) {
    using EigenBase = Eigen::EigenBase<Derived>;
    static_assert(std::is_base_of_v<EigenBase, Derived>);
    cl.def_prop_ro("cols", &Derived::cols)
        .def_prop_ro("rows", &Derived::rows)
        .def_prop_ro("size", &Derived::size);
  }
};

}  // namespace nanoeigenpy
