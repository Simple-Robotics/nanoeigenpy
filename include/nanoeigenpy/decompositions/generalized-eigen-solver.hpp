/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeGeneralizedEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::GeneralizedEigenSolver<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name, "Generalized Eigen solver")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, const MatrixType &, bool>(), "A"_a,
           "B"_a, "computeEigenvectors"_a = true,
           "Constructor; computes the generalized eigendecomposition of given "
           "matrix pair.")

      .def("eigenvectors", &Solver::eigenvectors,
           "Returns the computed generalized eigenvectors.")
      .def(
          "eigenvalues", [](const Solver &c) { return c.eigenvalues().eval(); },
          "Returns the computed generalized eigenvalues.")

      .def("alphas", &Solver::alphas,
           "Returns the vectors containing the alpha values.")
      .def("betas", &Solver::betas,
           "Returns tthe vectors containing the beta values.")

      .def(
          "compute",
          [](Solver &c, const MatrixType &A, const MatrixType &B) -> Solver & {
            return c.compute(A, B);
          },
          "A"_a, "B"_a,
          "Computes generalized eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, const MatrixType &A, const MatrixType &B,
             bool computeEigenvectors) -> Solver & {
            return c.compute(A, B, computeEigenvectors);
          },
          "A"_a, "B"_a, "computeEigenvectors"_a,
          "Computes generalized eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def("info", &Solver::info,
           "NumericalIssue if the input contains INF or NaN values or "
           "overflow occured. Returns Success otherwise.")

      .def("setMaxIterations", &Solver::setMaxIterations,
           "Sets the maximum number of iterations allowed.",
           nb::rv_policy::reference)

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
