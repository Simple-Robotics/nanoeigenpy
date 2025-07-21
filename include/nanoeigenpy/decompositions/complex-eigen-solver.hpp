/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeComplexEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::ComplexEigenSolver<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Complex Eigen Solver")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Computes eigendecomposition of given matrix")
      .def(nb::init<const MatrixType &, bool>(), "matrix"_a,
           "computeEigenvectors"_a,
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
          "matrix"_a, "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix, bool computeEigenvectors)
              -> Solver & { return c.compute(matrix, computeEigenvectors); },
          "matrix"_a, "computeEigenvectors"_a,
          "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def("info", &Solver::info,
           "NumericalIssue if the input contains INF or NaN values or "
           "overflow occured. Returns Success otherwise.")

      .def("setMaxIterations", &Solver::setMaxIterations,
           "Sets the maximum number of iterations allowed.",
           nb::rv_policy::reference)
      .def("getMaxIterations", &Solver::getMaxIterations,
           "Returns the maximum number of iterations.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
