/// Copyright 2025 INRIA
#pragma once
#include "nanoeigenpy/eigen-base.hpp"
// #include "nanoeigenpy/eigen-to-python.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <optional>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType>
Eigen::EigenSolver<MatrixType> &compute_proxy(
    Eigen::EigenSolver<MatrixType> &c,
    const Eigen::EigenBase<MatrixType> &matrix) {
  return c.compute(matrix);
}

template <typename MatrixType>
void exposeEigenSolver(nb::module_ m, const char *name) {
  using Solver = Eigen::EigenSolver<MatrixType>;
  auto cl =
      nb::class_<Solver>(m, name, "Eigen solver.")

          .def(nb::init<>(), "Default constructor.")
          .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),
               "Default constructor with memory preallocation.")
          .def(nb::init<const MatrixType &, bool>(), nb::arg("matrix"),
               nb::arg("compute_eigen_vectors") = true,
               "Computes eigendecomposition of given matrix")

          .def("eigenvalues", &Solver::eigenvalues,
               "Returns the eigenvalues of given matrix.",
               nb::rv_policy::reference_internal)
          .def("eigenvectors", &Solver::eigenvectors,
               "Returns the eigenvectors of given matrix.")

          .def(
              "compute",
              [](Solver &c, Eigen::EigenBase<MatrixType> const &matrix)
                  -> Solver & { return compute_proxy(c, matrix); },
              nb::arg("matrix"),
              "Computes the eigendecomposition of given matrix.",
              nb::rv_policy::reference)
          .def(
              "compute",
              [](Solver &c, Eigen::EigenBase<MatrixType> const &matrix,
                 bool compute_eigen_vectors) {
                return c.compute(matrix, compute_eigen_vectors);
              },
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

          .def(
              "id",
              [](Solver const &c) -> int64_t {
                return reinterpret_cast<int64_t>(&c);
              },
              "Returns the unique identity of an object.\n"
              "For objects held in C++, it corresponds to its memory address.");
}

}  // namespace nanoeigenpy
