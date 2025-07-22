/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeRealQZ(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::RealQZ<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Real QZ decomposition")

      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, const MatrixType &, bool>(), "A"_a,
           "B"_a, "computeQZ"_a = true,
           "Constructor; computes real QZ decomposition of given matrices.")

      .def("matrixQ", &Solver::matrixQ,
           "Returns matrix Q in the QZ decomposition.",
           nb::rv_policy::reference_internal)
      .def("matrixZ", &Solver::matrixZ,
           "Returns matrix Z in the QZ decomposition.",
           nb::rv_policy::reference_internal)
      .def("matrixS", &Solver::matrixS,
           "Returns matrix S in the QZ decomposition.",
           nb::rv_policy::reference_internal)
      .def("matrixT", &Solver::matrixT,
           "Returns matrix T in the QZ decomposition.",
           nb::rv_policy::reference_internal)

      .def(
          "compute",
          [](Solver &c, const MatrixType &A, const MatrixType &B) -> Solver & {
            return c.compute(A, B);
          },
          "A"_a, "B"_a, "Computes QZ decomposition of given matrix. ",
          nb::rv_policy::reference)

      .def(
          "compute",
          [](Solver &c, const MatrixType &A, const MatrixType &B,
             bool computeQZ) -> Solver & { return c.compute(A, B, computeQZ); },
          "A"_a, "B"_a, "computeQZ"_a,
          "Computes QZ decomposition of given matrix. ",
          nb::rv_policy::reference)

      .def("info", &Solver::info,
           "Reports whether previous computation was successful.")

      .def("iterations", &Solver::iterations,
           "Returns number of performed QR-like iterations.")
      .def("setMaxIterations", &Solver::setMaxIterations,
           "Sets the maximum number of iterations allowed.",
           nb::rv_policy::reference)

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
