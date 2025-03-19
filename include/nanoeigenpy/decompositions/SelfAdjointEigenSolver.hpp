/// Copyright 2025 INRIA
#pragma once

#include "base.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType>
void exposeSelfAdjointEigenSolver(nb::module_ m, const char *name) {
  using Solver = Eigen::SelfAdjointEigenSolver<MatrixType>;
  auto cl =
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
              [](Solver &c, Eigen::EigenBase<MatrixType> const &matrix) {
                return c.compute(matrix);
              },
              nb::arg("matrix"),
              "Computes the eigendecomposition of given matrix.",
              nb::rv_policy::reference)  // Check here
          .def(
              "compute",
              [](Solver &c, Eigen::EigenBase<MatrixType> const &matrix,
                 int options) { return c.compute(matrix, options); },
              nb::arg("matrix"), nb::arg("options"),
              "Computes the eigendecomposition of given matrix.",
              nb::rv_policy::reference)

          .def(
              "computeDirect",
              [](Solver &c, MatrixType const &matrix) {
                return c.computeDirect(matrix);
              },
              nb::arg("matrix"),
              "Computes eigendecomposition of given matrix using a closed-form "
              "algorithm.",
              nb::rv_policy::reference)
          .def(
              "computeDirect",
              [](Solver &c, MatrixType const &matrix, int options) {
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

          .def(
              "id",
              [](Solver const &c) -> int64_t {
                return reinterpret_cast<int64_t>(&c);
              },
              "Returns the unique identity of an object.\n"
              "For objects held in C++, it corresponds to its memory address.");
}

}  // namespace nanoeigenpy

// TODO

// See if Options = 0 is correct
// "Check here" to see if I chose the right rv_policy
