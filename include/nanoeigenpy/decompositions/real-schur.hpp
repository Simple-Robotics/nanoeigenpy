/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeRealSchur(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::RealSchur<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Real Schur decomposition")

      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, bool>(), "matrix"_a,
           "computeU"_a = true,
           "Constructor; computes real Schur decomposition of given matrices.")

      .def("matrixU", &Solver::matrixU,
           "Returns the orthogonal matrix in the Schur decomposition. ",
           nb::rv_policy::reference_internal)
      .def("matrixT", &Solver::matrixT,
           "Returns the quasi-triangular matrix in the Schur decomposition.",
           nb::rv_policy::reference_internal)

      .def(
          "compute",
          [](Solver &c, const MatrixType &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes Schur decomposition of given matrix.",
          nb::rv_policy::reference)

      .def(
          "compute",
          [](Solver &c, const MatrixType &matrix, bool computeU) -> Solver & {
            return c.compute(matrix, computeU);
          },
          "matrix"_a, "computeU"_a,
          "Computes Schur decomposition of given matrix.",
          nb::rv_policy::reference)

      .def(
          "computeFromHessenberg",
          [](Solver &c, const MatrixType &matrixH, const MatrixType &matrixQ,
             bool computeU) -> Solver & {
            return c.computeFromHessenberg(matrixH, matrixQ, computeU);
          },
          "matrixH"_a, "matrixQ"_a, "computeU"_a,
          "Computes Schur decomposition of a Hessenberg matrix H = Z T Z^T",
          nb::rv_policy::reference)

      .def("info", &Solver::info,
           "Reports whether previous computation was successful.")

      .def("setMaxIterations", &Solver::setMaxIterations,
           "Sets the maximum number of iterations allowed.",
           nb::rv_policy::reference)
      .def("getMaxIterations", &Solver::getMaxIterations,
           "Returns the maximum number of iterations.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
