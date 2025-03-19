/// Copyright 2025 INRIA
#pragma once

#include "nanoeigenpy/decompositions/sparse/SimplicialCholesky.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;

template <
    typename MatrixType, int UpLo = Eigen::Lower,
    typename Ordering = Eigen::AMDOrdering<typename MatrixType::StorageIndex>>
void exposeSimplicialLDLT(nb::module_ m, const char *name) {
  using Solver = Eigen::SimplicialLDLT<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using DenseVectorXs =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>;
  //   using DenseMatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic,
  //   Eigen::Dynamic,
  //                         MatrixType::Options>;

  auto cl =
      nb::class_<Solver>(
          m, name,
          "A direct sparse LDLT Cholesky factorizations.\n\n"
          "This class provides a LDL^T Cholesky factorizations of sparse "
          "matrices that are selfadjoint and positive definite."
          "The factorization allows for solving A.X = B where X and B can be "
          "either dense or sparse.\n\n"
          "In order to reduce the fill-in, a symmetric permutation P is "
          "applied "
          "prior to the factorization such that the factorized matrix is P A "
          "P^-1.")

          .def(nb::init<>(), "Default constructor.")
          .def(nb::init<const MatrixType &>(), nb::arg("matrix"),
               "Constructs a LDLT factorization from a given matrix.")

          .def(
              "vectorD",
              [](Solver const &self) -> DenseVectorXs {
                return self.vectorD();
              },
              "Returns the diagonal vector D.")

          .def(SimplicialCholeskyVisitor())

          .def(
              "id",
              [](Solver const &self) -> int64_t {
                return reinterpret_cast<int64_t>(&self);
              },
              "Returns the unique identity of an object.\n"
              "For objects held in C++, it corresponds to its memory address.");
}

}  // namespace nanoeigenpy
