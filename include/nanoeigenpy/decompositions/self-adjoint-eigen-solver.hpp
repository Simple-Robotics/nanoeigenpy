/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename _MatrixType>
void exposeSelfAdjointEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::SelfAdjointEigenSolver<MatrixType>;

  nb::class_<Solver>(m, name, "Self adjoint Eigen Solver")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, int>(), nb::arg("matrix"),
           nb::arg("options") = 0,
           "Computes eigendecomposition of given matrix")

      .def("eigenvalues", &Solver::eigenvalues,
           "Returns the eigenvalues of given matrix.",
           nb::rv_policy::reference_internal)
      .def("eigenvectors", &Solver::eigenvectors,
           "Returns the eigenvectors of given matrix.",
           nb::rv_policy::reference_internal)

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          nb::arg("matrix"), "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix, int options) -> Solver & {
            return c.compute(matrix, options);
          },
          nb::arg("matrix"), nb::arg("options"),
          "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def(
          "computeDirect",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.computeDirect(matrix);
          },
          nb::arg("matrix"),
          "Computes eigendecomposition of given matrix using a closed-form "
          "algorithm.",
          nb::rv_policy::reference)
      .def(
          "computeDirect",
          [](Solver &c, MatrixType const &matrix, int options) -> Solver & {
            return c.computeDirect(matrix, options);
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
