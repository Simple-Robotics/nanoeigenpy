/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename _MatrixType>
void exposeEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::EigenSolver<MatrixType>;

  nb::class_<Solver>(m, name, "Eigen solver.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, bool>(), nb::arg("matrix"),
           nb::arg("compute_eigen_vectors") = true,
           "Computes eigendecomposition of given matrix")

      .def("eigenvalues", &Solver::eigenvalues,
           "Returns the eigenvalues of the matrix.",
           nb::rv_policy::reference_internal)
      .def("eigenvectors", &Solver::eigenvectors,
           "Returns the eigenvectors of the matrix.")

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          nb::arg("matrix"), "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix, bool compute_eigen_vectors)
              -> Solver & { return c.compute(matrix, compute_eigen_vectors); },
          nb::arg("matrix"), nb::arg("compute_eigen_vectors"),
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
