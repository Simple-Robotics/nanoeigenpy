/// Copyright 2025 INRIA
#include <Eigen/Geometry>
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

struct X {
  Eigen::Quaterniond a;
};

NB_MODULE(quaternion, m) {
  nb::class_<X>(m, "X")
      .def(nb::init<Eigen::Quaterniond>(), "a"_a)
      .def_rw("a", &X::a);
}
