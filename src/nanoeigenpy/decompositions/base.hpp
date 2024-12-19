#pragma once
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

namespace nanoeigenpy {

template <typename T, typename... Ts>
void addEigenBaseFeatures(nanobind::class_<T, Ts...> &cl) {
  cl.def_prop_ro("cols", &T::cols)
      .def_prop_ro("rows", &T::rows)
      .def_prop_ro("size", &T::size);
}

} // namespace nanoeigenpy
