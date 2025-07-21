/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeTridiagonalization(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::Tridiagonalization<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Tridiagonalization")

      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructor; computes tridiagonal decomposition of given matrix.")

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes tridiagonal decomposition of given matrix.",
          nb::rv_policy::reference)

      .def("householderCoefficients", &Solver::householderCoefficients,
           "Returns the Householder coefficients.")

      .def("packedMatrix", &Solver::packedMatrix,
           "Returns the internal representation of the decomposition.",
           nb::rv_policy::reference_internal)

      .def(
          "matrixQ", [](const Solver &c) -> MatrixType { return c.matrixQ(); },
          "Returns the unitary matrix Q in the decomposition.")
      .def(
          "matrixT", [](Solver &c) -> MatrixType { return c.matrixT(); },
          "Returns an expression of the tridiagonal matrix T in the "
          "decomposition.")

      .def("diagonal", &Solver::diagonal,
           "Returns the diagonal of the tridiagonal matrix T in the "
           "decomposition.")
      .def("subDiagonal", &Solver::subDiagonal,
           "Returns the subdiagonal of the tridiagonal matrix T in the "
           "decomposition.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
