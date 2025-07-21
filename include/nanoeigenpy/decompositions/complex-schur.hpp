/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeComplexSchur(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::ComplexSchur<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Complex Schur decomposition")

      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructor; computes Schur decomposition of given matrix.")
      .def(nb::init<const MatrixType &, bool>(), "matrix"_a,
           "computeU"_a = true,
           "Constructor; computes Schur decomposition of given matrix.")

      .def("matrixU", &Solver::matrixU,
           "Returns the unitary matrix in the Schur decomposition.",
           nb::rv_policy::reference_internal)
      .def("matrixT", &Solver::matrixT,
           "Returns the triangular matrix in the Schur decomposition. ",
           nb::rv_policy::reference_internal)

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes Schur decomposition of given matrix. ",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix, bool computeU) -> Solver & {
            return c.compute(matrix, computeU);
          },
          "matrix"_a, "computeU"_a,
          "Computes Schur decomposition of given matrix. ",
          nb::rv_policy::reference)

      .def(
          "computeFromHessenberg",
          [](Solver &c, MatrixType const &matrixH, MatrixType const &matrixQ,
             bool computeU) -> Solver & {
            return c.computeFromHessenberg(matrixH, matrixQ, computeU);
          },
          "matrixH"_a, "matrixQ"_a, "computeU"_a,
          "Compute Schur decomposition from a given Hessenberg matrix. ",
          nb::rv_policy::reference)

      .def("info", &Solver::info,
           "NumericalIssue if the input contains INF or NaN values or "
           "overflow occured. Returns Success otherwise.")

      .def("getMaxIterations", &Solver::getMaxIterations,
           "Returns the maximum number of iterations.")
      .def("setMaxIterations", &Solver::setMaxIterations,
           "Sets the maximum number of iterations allowed.",
           nb::rv_policy::reference)

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
