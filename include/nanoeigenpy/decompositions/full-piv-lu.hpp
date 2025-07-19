/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/LU>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::FullPivLU<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename _MatrixType>
void exposeFullPivLU(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::FullPivLU<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name,
                     "LU decomposition of a matrix with complete pivoting, and "
                     "related features.\n\n"
                     "This class represents a LU decomposition of any matrix, "
                     "with complete pivoting: "
                     "the matrix A is decomposed as A=P−1LUQ−1 where L is "
                     "unit-lower-triangular, U is "
                     "upper-triangular, and P and Q are permutation matrices. "
                     "This is a rank-revealing "
                     "LU decomposition. The eigenvalues (diagonal "
                     "coefficients) of U are sorted in such "
                     "a way that any zeros are at the end.\n\n"
                     "This decomposition provides the generic approach to "
                     "solving systems of linear "
                     "equations, computing the rank, invertibility, inverse, "
                     "kernel, and determinant.\n\n"
                     "This LU decomposition is very stable and well tested "
                     "with large matrices. However "
                     "there are use cases where the SVD decomposition is "
                     "inherently more stable and/or "
                     "flexible. For example, when computing the kernel of a "
                     "matrix, working with the SVD "
                     "allows to select the smallest singular values of the "
                     "matrix, something that the LU "
                     "decomposition doesn't see.\n\n"
                     "The data of the LU decomposition can be directly "
                     "accessed through the methods "
                     "matrixLU(), permutationP(), permutationQ().")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex>(), "rows"_a, "cols"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructs a LU factorization from a given matrix.")

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes the LU of given matrix.",
          nb::rv_policy::reference)

      .def("matrixLU", &Solver::matrixLU,
           "Returns the LU decomposition matrix: the upper-triangular part is "
           "U, the "
           "unit-lower-triangular part is L (at least for square matrices; in "
           "the non-square "
           "case, special care is needed, see the documentation of class "
           "FullPivLU).",
           nb::rv_policy::reference_internal)

      .def("nonzeroPivots", &Solver::nonzeroPivots,
           "Returns the number of nonzero pivots in the LU decomposition.")
      .def("maxPivot", &Solver::maxPivot,
           "Returns the absolute value of the biggest pivot, i.e. the biggest"
           "diagonal coefficient of U.")

      .def("permutationP", &Solver::permutationP,
           "Returns the permutation matrix P in the decomposition A = P^{-1} L "
           "U Q^{-1}.",
           nb::rv_policy::reference_internal)
      .def("permutationQ", &Solver::permutationQ,
           "Returns the permutation matrix Q in the decomposition A = P^{-1} L "
           "U Q^{-1}.",
           nb::rv_policy::reference_internal)

      .def(
          "kernel", [](Solver &c) -> MatrixType { return c.kernel(); },
          "Computes the LU of given matrix.")
      .def(
          "image",
          [](Solver &c, MatrixType const &originalMatrix) -> MatrixType {
            return c.image(originalMatrix);
          },
          "Computes the LU of given matrix.")

      .def("rcond", &Solver::rcond,
           "Returns an estimate of the reciprocal condition number of the "
           "matrix.")
      .def("determinant", &Solver::determinant,
           "Returns the determinant of the underlying matrix from the "
           "current factorization.")

      .def(
          "setThreshold",
          [](Solver &c, RealScalar const &threshold) {
            return c.setThreshold(threshold);
          },
          "threshold"_a,
          "Allows to prescribe a threshold to be used by certain methods, "
          "such as rank(), who need to determine when pivots are to be "
          "considered nonzero. This is not used for the LU decomposition "
          "itself.\n"
          "\n"
          "When it needs to get the threshold value, Eigen calls "
          "threshold(). By default, this uses a formula to automatically "
          "determine a reasonable threshold. Once you have called the "
          "present method setThreshold(const RealScalar&), your value is "
          "used instead.\n"
          "\n"
          "Note: A pivot will be considered nonzero if its absolute value "
          "is strictly greater than |pivot| ⩽ threshold×|maxpivot| where "
          "maxpivot is the biggest pivot.",
          nb::rv_policy::reference)
      .def(
          "setThreshold",
          [](Solver &c) { return c.setThreshold(Eigen::Default); },
          "Allows to come back to the default behavior, letting Eigen use "
          "its default formula for determining the threshold.",
          nb::rv_policy::reference)
      .def("threshold", &Solver::threshold,
           "Returns the threshold that will be used by certain methods such "
           "as rank().")

      .def("rank", &Solver::rank,
           "Returns the rank of the matrix associated with the LU "
           "decomposition.\n"
           "\n"
           "Note: This method has to determine which pivots should be "
           "considered nonzero. For that, it uses the threshold value that "
           "you can control by calling setThreshold(threshold).")
      .def("dimensionOfKernel", &Solver::dimensionOfKernel,
           "Returns the dimension of the kernel of the matrix of which "
           "*this is the LU decomposition.")

      .def("isInjective", &Solver::isInjective,
           "Returns true if the matrix of which *this is the LU decomposition "
           "represents an injective linear map, i.e. has trivial kernel; "
           "false otherwise.\n\n"
           "Note: This method has to determine which pivots should be "
           "considered nonzero. For that, it uses the threshold value that "
           "you can control by calling setThreshold(threshold).")
      .def("isSurjective", &Solver::isSurjective,
           "Returns true if the matrix of which *this is the LU decomposition "
           "represents a surjective linear map; false otherwise.\n\n"
           "Note: This method has to determine which pivots should be "
           "considered nonzero. For that, it uses the threshold value that "
           "you can control by calling setThreshold(threshold).")
      .def("isInvertible", &Solver::isInvertible,
           "Returns true if the matrix of which *this is the LU decomposition "
           "is invertible.\n\n"
           "Note: This method has to determine which pivots should be "
           "considered nonzero. For that, it uses the threshold value that "
           "you can control by calling setThreshold(threshold).")

      .def(
          "inverse", [](Solver const &c) -> MatrixType { return c.inverse(); },
          "Returns the inverse of the matrix associated with the LU "
          "decomposition.")
      .def("reconstructedMatrix", &Solver::reconstructedMatrix,
           "Returns the matrix represented by the decomposition,"
           "i.e., it returns the product: \f$ P^{-1} L U Q^{-1} \f$."
           "This function is provided for debug purposes.")

      .def("rows", &Solver::rows, "Returns the number of rows of the matrix.")
      .def("cols", &Solver::cols, "Returns the number of cols of the matrix.")

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
