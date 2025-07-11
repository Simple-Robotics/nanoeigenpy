/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <Eigen/LU>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::PartialPivLU<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename _MatrixType>
void exposePartialPivLU(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::PartialPivLU<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "LU decomposition of a matrix with partial pivoting, and "
      "related features. \n\n"
      "This class represents a LU decomposition of a square "
      "invertible matrix, with partial pivoting: the matrix "
      "A is decomposed as A = PLU where L is unit-lower-triangular, "
      "U is upper-triangular, and P is a permutation matrix.\n\n"
      "Typically, partial pivoting LU decomposition is only considered "
      "numerically stable for square invertible matrices. Thus LAPACK's "
      "dgesv and dgesvx require the matrix to be square and invertible. "
      "The present class does the same. It will assert that the matrix is "
      "square, but it won't (actually it can't) check that the matrix is "
      "invertible: it is your task to check that you only use this "
      "decomposition on invertible matrices.\n\n"
      "The guaranteed safe alternative, working for all matrices, is the "
      "full pivoting LU decomposition, provided by class FullPivLU.\n\n"
      "This is not a rank-revealing LU decomposition. Many features are "
      "intentionally absent from this class, such as rank computation. If "
      "you need these features, use class FullPivLU.\n\n"
      "This LU decomposition is suitable to invert invertible matrices. "
      "It is what MatrixBase::inverse() uses in the general case. On the "
      "other hand, it is not suitable to determine whether a given matrix "
      "is invertible.\n\n"
      "The data of the LU decomposition can be directly accessed through "
      "the methods matrixLU(), permutationP().")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructs a LU factorization from a given matrix.")

      .def("rows", &Solver::rows, "Returns the number of rows of the matrix.")
      .def("cols", &Solver::cols, "Returns the number of cols of the matrix.")

      .def("matrixLU", &Solver::matrixLU,
           "Returns the LU decomposition matrix: the upper-triangular part is "
           "U, the "
           "unit-lower-triangular part is L (at least for square matrices; in "
           "the non-square "
           "case, special care is needed, see the documentation of class "
           "FullPivLU).",
           nb::rv_policy::reference_internal)

      // TODO: Expose so that the return type are convertible to np arrays
      // permutationP

      .def("rcond", &Solver::rcond,
           "Returns an estimate of the reciprocal condition number of the "
           "matrix.")
      .def(
          "inverse", [](Solver const &c) -> MatrixType { return c.inverse(); },
          "Returns the inverse of the matrix associated with the LU "
          "decomposition.")
      .def("determinant", &Solver::determinant,
           "Returns the determinant of the underlying matrix from the "
           "current factorization.")
      .def("reconstructedMatrix", &Solver::reconstructedMatrix,
           "Returns the matrix represented by the decomposition,"
           "i.e., it returns the product: P^{-1} L U."
           "This function is provided for debug purpose.")

      .def(
          "solve",
          [](Solver const &c, VectorType const &b) -> VectorType {
            return solve(c, b);
          },
          nb::arg("b"),
          "Returns the solution x of A x = b using the current "
          "decomposition of A.")
      .def(
          "solve",
          [](Solver const &c, MatrixType const &B) -> MatrixType {
            return solve(c, B);
          },
          nb::arg("B"),
          "Returns the solution X of A X = B using the current "
          "decomposition of A where B is a right hand side matrix.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
