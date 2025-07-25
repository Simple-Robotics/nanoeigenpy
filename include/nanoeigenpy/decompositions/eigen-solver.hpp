/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::EigenSolver<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Eigen solver.")
      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, bool>(), "matrix"_a,
           "compute_eigen_vectors"_a = true,
           "Computes eigendecomposition of given matrix")

      .def("eigenvalues", &Solver::eigenvalues,
           "Returns the eigenvalues of the matrix.",
           nb::rv_policy::reference_internal)
      .def("eigenvectors", &Solver::eigenvectors,
           "Returns the eigenvectors of the matrix.")

      .def(
          "compute",
          [](Solver &c, const MatrixType &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, const MatrixType &matrix, bool compute_eigen_vectors)
              -> Solver & { return c.compute(matrix, compute_eigen_vectors); },
          "matrix"_a, "compute_eigen_vectors"_a,
          "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def("getMaxIterations", &Solver::getMaxIterations,
           "Returns the maximum number of iterations.")
      .def("setMaxIterations", &Solver::setMaxIterations,
           "Sets the maximum number of iterations allowed.",
           nb::rv_policy::reference)

      .def("pseudoEigenvalueMatrix", &Solver::pseudoEigenvalueMatrix,
           "Returns the block-diagonal matrix in the "
           "pseudo-eigendecomposition.")
      .def("pseudoEigenvectors", &Solver::pseudoEigenvectors,
           "Returns the pseudo-eigenvectors of given matrix.",
           nb::rv_policy::reference_internal)

      .def("info", &Solver::info,
           "NumericalIssue if the input contains INF or NaN values or "
           "overflow occured. Returns Success otherwise.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
