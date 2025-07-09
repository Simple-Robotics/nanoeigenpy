/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/svd-base.hpp"
#include <Eigen/SVD>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::JacobiSVD<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename _MatrixType>
void exposeJacobiSVD(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::JacobiSVD<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "Two-sided Jacobi SVD decomposition of a rectangular matrix.  \n\n"
      "SVD decomposition consists in decomposing any n-by-p matrix A as "
      "a product A=USV^* where U is a n-by-n unitary, V is a p-by-p unitary, "
      "and S is a n-by-p real positive matrix which is zero outside of its "
      "main diagonal; the diagonal entries of S are known as the singular "
      "values "
      "of A and the columns of U and V are known as the left and right "
      "singular "
      "vectors of A respectively.\n\n"
      "Singular values are always sorted in decreasing order.\n\n"
      "This JacobiSVD decomposition computes only the singular values by "
      "default. "
      "If you want U or V, you need to ask for them explicitly.\n\n"
      "You can ask for only thin U or V to be computed, meaning the following. "
      "In case of a rectangular n-by-p matrix, letting m be the smaller value "
      "among "
      "n and p, there are only m singular vectors; the remaining columns of U "
      "and V "
      "do not correspond to actual singular vectors. Asking for thin U or V "
      "means asking "
      "for only their m first columns to be formed. So U is then a n-by-m "
      "matrix, and V "
      "is then a p-by-m matrix. Notice that thin U and V are all you need for "
      "(least "
      "squares) solving.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex, unsigned int>(),
           "rows"_a, "cols"_a, "computationOptions"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex>(), "rows"_a, "cols"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructs a SVD factorization from a given matrix.")
      .def(nb::init<const MatrixType &, unsigned int>(), "matrix"_a,
           "computationOptions"_a,
           "Constructs a SVD factorization from a given matrix.")

      .def(SVDBaseVisitor())

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes the SVD of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix, unsigned int) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "computationOptions"_a,
          "Computes the SVD of given matrix.", nb::rv_policy::reference)

      .def(
          "solve",
          [](Solver const &c, VectorType const &b) -> VectorType {
            return solve(c, b);
          },
          "b"_a,
          "Returns the solution x of A x = b using the current "
          "decomposition of A.")
      .def(
          "solve",
          [](Solver const &c, MatrixType const &B) -> MatrixType {
            return solve(c, B);
          },
          "B"_a,
          "Returns the solution X of A X = B using the current "
          "decomposition of A where B is a right hand side matrix.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
