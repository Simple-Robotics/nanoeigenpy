/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
void exposeJacobiRotation(nb::module_ m, const char* name) {
  using namespace nb::literals;
  using JacobiRotation = Eigen::JacobiRotation<Scalar>;
  using RealScalar = Eigen::NumTraits<Scalar>::Real;

  if (check_registration_alias<JacobiRotation>(m)) {
    return;
  }
  nb::class_<JacobiRotation>(
      m, name, "This class represents a Jacobi or Givens rotation.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Scalar&, const Scalar&>(), "c"_a, "s"_a,
           "Construct a planar rotation from a cosine-sine pair (c, s).")

      .def_prop_rw(
          "c", [](const JacobiRotation& self) { return self.c(); },
          [](JacobiRotation& self, const Scalar& value) { self.c() = value; })
      .def_prop_rw(
          "s", [](const JacobiRotation& self) { return self.s(); },
          [](JacobiRotation& self, const Scalar& value) { self.s() = value; })

      .def("__mul__", &JacobiRotation::operator*, "other"_a,
           "Concatenates two planar rotations")

      .def("transpose", &JacobiRotation::transpose)
      .def("adjoint", &JacobiRotation::adjoint)

      .def(
          "makeJacobi",
          [](JacobiRotation& self, const RealScalar& x, const Scalar& y,
             const RealScalar& z) { return self.makeJacobi(x, y, z); },
          "x"_a, "y"_a, "z"_a)

      .def(
          "makeJacobi",
          [](JacobiRotation& self, const Eigen::MatrixXd& m, Eigen::Index p,
             Eigen::Index q) { return self.makeJacobi(m, p, q); },
          "matrix"_a, "p"_a, "q"_a)

      .def(
          "makeGivens",
          [](JacobiRotation& self, const Scalar& p, const Scalar& q) {
            self.makeGivens(p, q, nullptr);
          },
          "p"_a, "q"_a)

      .def(
          "makeGivens",
          [](JacobiRotation& self, const Scalar& p, const Scalar& q,
             Scalar* r = 0) { self.makeGivens(p, q, r); },
          "p"_a, "q"_a, "r"_a)

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
