/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-decomposition.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename _MatrixType, int _UpLo = Eigen::Lower>
void exposeCholmodSupernodalLLT(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::CholmodSupernodalLLT<_MatrixType, _UpLo>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;

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

      .def(CholmodDecompositionVisitor());
}

}  // namespace nanoeigenpy
