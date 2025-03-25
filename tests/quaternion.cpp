/// Copyright 2025 INRIA
#include <nanoeigenpy/geometry/angle-axis.hpp>
#include <nanoeigenpy/geometry/quaternion.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanoeigenpy;

struct X {
  Eigen::Quaterniond a;
};

NB_MODULE(quaternion, m) {
  nb::class_<X>(m, "X")
      .def(nb::init<Eigen::Quaterniond>(), "a"_a)
      .def_rw("a", &X::a);
}
