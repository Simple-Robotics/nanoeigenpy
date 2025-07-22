/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeHessenbergDecomposition(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::HessenbergDecomposition<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Hessenberg decomposition")

      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructor; computes Hessenberg decomposition of given matrix.")

      .def(
          "compute",
          [](Solver &c, const MatrixType &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes Hessenberg decomposition of given matrix.",
          nb::rv_policy::reference)

      .def("householderCoefficients", &Solver::householderCoefficients,
           "Returns the Householder coefficients.",
           nb::rv_policy::reference_internal)

      .def("packedMatrix", &Solver::packedMatrix,
           "Returns the internal representation of the decomposition.",
           nb::rv_policy::reference_internal)

      .def(
          "matrixQ", [](const Solver &c) -> MatrixType { return c.matrixQ(); },
          "Reconstructs the orthogonal matrix Q in the decomposition.")
      .def(
          "matrixH", [](const Solver &c) -> MatrixType { return c.matrixH(); },
          "Constructs the Hessenberg matrix H in the decomposition.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
