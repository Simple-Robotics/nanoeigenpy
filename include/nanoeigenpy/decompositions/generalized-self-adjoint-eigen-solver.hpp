/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeGeneralizedSelfAdjointEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Generalized self adjoint Eigen Solver")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, const MatrixType &>(), "matA"_a,
           "matB"_a,
           "Computes the generalized eigendecomposition of given matrix pencil")
      .def(nb::init<const MatrixType &, const MatrixType &, int>(), "matA"_a,
           "matB"_a, "options"_a,
           "Computes the generalized eigendecomposition of given matrix pencil")

      .def(
          "compute",
          [](Solver &c, MatrixType const &matA, MatrixType const &matB)
              -> Solver & { return c.compute(matA, matB); },
          "matA"_a, "matB"_a,
          "Computes the generalized eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matA, MatrixType const &matB,
             int options) -> Solver & {
            return c.compute(matA, matB, options);
          },
          "matA"_a, "matB"_a, "options"_a,
          "Computes the generalized eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def("eigenvalues", &Solver::eigenvalues,
           "Returns the eigenvalues of given matrix.",
           nb::rv_policy::reference_internal)
      .def("eigenvectors", &Solver::eigenvectors,
           "Returns the eigenvectors of given matrix.",
           nb::rv_policy::reference_internal)

      .def(
          "computeDirect",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return static_cast<Solver &>(c.computeDirect(matrix));
          },
          nb::arg("matrix"),
          "Computes eigendecomposition of given matrix using a closed-form "
          "algorithm.",
          nb::rv_policy::reference)
      .def(
          "computeDirect",
          [](Solver &c, MatrixType const &matrix, int options) -> Solver & {
            return static_cast<Solver &>(c.computeDirect(matrix, options));
          },
          nb::arg("matrix"), nb::arg("options"),
          "Computes eigendecomposition of given matrix using a closed-form "
          "algorithm.",
          nb::rv_policy::reference)

      .def("operatorInverseSqrt", &Solver::operatorInverseSqrt,
           "Computes the inverse square root of the matrix.")
      .def("operatorSqrt", &Solver::operatorSqrt,
           "Computes the square root of the matrix.")

      .def("info", &Solver::info,
           "NumericalIssue if the input contains INF or NaN values or "
           "overflow occured. Returns Success otherwise.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
