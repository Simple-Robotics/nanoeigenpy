/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeGeneralizedSelfAdjointEigenSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType>;
  using Base = Eigen::SelfAdjointEigenSolver<MatrixType>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver, Base>(m, name, "Generalized self adjoint Eigen Solver")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &, const MatrixType &>(), "matA"_a,
           "matB"_a,
           "Computes the generalized eigendecomposition of given matrix pencil")
      .def(nb::init<const MatrixType &, const MatrixType &, int>(), "matA"_a,
           "matB"_a, "options"_a,
           "Computes the generalized eigendecomposition of given matrix pencil")

      .def(
          "compute",
          [](Solver &c, MatrixType const &matA, MatrixType const &matB)
              -> Solver & { return c.compute(matA, matB); },
          "matA"_a, "matB"_a,
          "Computes the generalized eigendecomposition of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matA, MatrixType const &matB,
             int options) -> Solver & {
            return c.compute(matA, matB, options);
          },
          "matA"_a, "matB"_a, "options"_a,
          "Computes the generalized eigendecomposition of given matrix.",
          nb::rv_policy::reference)

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
