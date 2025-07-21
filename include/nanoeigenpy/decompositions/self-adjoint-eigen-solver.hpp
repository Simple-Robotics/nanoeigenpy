/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeSelfAdjointEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::SelfAdjointEigenSolver<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Self adjoint Eigen Solver")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, Eigen::DecompositionOptions>(),
           "matrix"_a, "options"_a = Eigen::ComputeEigenvectors,
           "Computes eigendecomposition of given matrix")

      .def(
          "eigenvalues",
          [](Solver &c) -> const VectorType & { return c.eigenvalues(); },
          "Returns the eigenvalues of given matrix.",
          nb::rv_policy::reference_internal)
      .def(
          "eigenvectors",
          [](Solver &c) -> const MatrixType & { return c.eigenvectors(); },
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
          [](Solver &c, MatrixType const &matrix, int options) -> Solver & {
            return c.compute(matrix, options);
          },
          "matrix"_a, "options"_a,
          "Computes the eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def(
          "computeDirect",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.computeDirect(matrix);
          },
          "matrix"_a,
          "Computes eigendecomposition of given matrix using a closed-form "
          "algorithm.",
          nb::rv_policy::reference)
      .def(
          "computeDirect",
          [](Solver &c, MatrixType const &matrix, int options) -> Solver & {
            return c.computeDirect(matrix, options);
          },
          "matrix"_a, "options"_a,
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
