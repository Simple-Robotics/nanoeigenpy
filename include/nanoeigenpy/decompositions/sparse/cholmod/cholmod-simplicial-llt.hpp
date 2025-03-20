/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-decomposition.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType, int UpLo = Eigen::Lower>
void exposeCholmodSimplicialLLT(nb::module_ m, const char *name) {
  using Solver = Eigen::CholmodSimplicialLLT<MatrixType, UpLo>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  auto cl =
      nb::class_<Solver>(
          m, name,
          "A simplicial direct Cholesky (LLT) factorization and solver based "
          "on "
          "Cholmod.\n\n"
          "This class allows to solve for A.X = B sparse linear problems via a "
          "simplicial LL^T Cholesky factorization using the Cholmod library."
          "This simplicial variant is equivalent to Eigen's built-in "
          "SimplicialLLT class."
          "Therefore, it has little practical interest. The sparse matrix A "
          "must "
          "be selfadjoint and positive definite."
          "The vectors or matrices X and B can be either dense or sparse.")

          .def(nb::init<>(), "Default constructor.")
          .def(nb::init<const MatrixType &>(), nb::arg("matrix"),
               "Constructs a LDLT factorization from a given matrix.")

          .def(CholmodDecompositionVisitor())

      ;
}

}  // namespace nanoeigenpy

// TODO:

// Wilson used to define the inetrmediate structures (eg EigenBaseVisitor) ->
// check if it works well here as I defined, to be sure that ir is the
// appropriate solver that is called

// Add the option noncopyable in
