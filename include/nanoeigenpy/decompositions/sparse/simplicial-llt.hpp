/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/decompositions/sparse/simplicial-cholesky.hpp"

namespace nanoeigenpy {

template <
    typename _MatrixType, int _UpLo = Eigen::Lower,
    typename _Ordering = Eigen::AMDOrdering<typename _MatrixType::StorageIndex>>
void exposeSimplicialLLT(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::SimplicialLLT<MatrixType>;

  if (register_symbolic_link_to_registered_type<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "A direct sparse LLT Cholesky factorizations.\n\n"
      "This class provides a LL^T Cholesky factorizations of sparse "
      "matrices "
      "that are selfadjoint and positive definite."
      "The factorization allows for solving A.X = B where X and B can be "
      "either dense or sparse.\n\n"
      "In order to reduce the fill-in, a symmetric permutation P is "
      "applied "
      "prior to the factorization such that the factorized matrix is P A "
      "P^-1.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<const MatrixType &>(), nb::arg("matrix"),
           "Constructs a LLT factorization from a given matrix.")

      .def(SimplicialCholeskyVisitor())

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
