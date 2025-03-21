/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <Eigen/QR>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::ColPivHouseholderQR<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename MatrixType>
MatrixType inverse(const Eigen::ColPivHouseholderQR<MatrixType> &c) {
  return c.inverse();
}

template <typename _MatrixType>
void exposeColPivHouseholderQRSolver(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::ColPivHouseholderQR<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;
  auto cl =
      nb::class_<Solver>(
          m, name,
          "This class performs a rank-revealing QR decomposition of a matrix A "
          "into matrices P, Q and R such that:\n"
          "AP=QR\n"
          "by using Householder transformations. Here, P is a permutation "
          "matrix, Q a unitary matrix and R an upper triangular matrix.\n"
          "\n"
          "This decomposition performs column pivoting in order to be "
          "rank-revealing and improve numerical stability. It is slower than "
          "HouseholderQR, and faster than FullPivHouseholderQR.")

          .def(nb::init<>(),
               "Default constructor.\n"
               "The default constructor is useful in cases in which the "
               "user intends to perform decompositions via "
               "HouseholderQR.compute(matrix).")
          .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex>(),
               nb::arg("rows"), nb::arg("cols"),
               "Default constructor with memory preallocation.\n"
               "Like the default constructor but with preallocation of the "
               "internal data according to the specified problem size. ")
          .def(nb::init<const MatrixType &>(), nb::arg("matrix"),
               "Constructs a QR factorization from a given matrix.\n"
               "This constructor computes the QR factorization of the matrix "
               "matrix by calling the method compute().")

          .def("info", &Solver::info,
               "Reports whether the QR factorization was successful.\n"
               "Note: This function always returns Success. It is provided for "
               "compatibility with other factorization routines.")

          .def("absDeterminant", &Solver::absDeterminant,
               "Returns the absolute value of the determinant of the matrix of "
               "which *this is the QR decomposition.\n"
               "It has only linear complexity (that is, O(n) where n is the "
               "dimension of the square matrix) as the QR decomposition has "
               "already been computed.\n"
               "Note: This is only for square matrices.")
          .def("logAbsDeterminant", &Solver::logAbsDeterminant,
               "Returns the natural log of the absolute value of the "
               "determinant "
               "of the matrix of which *this is the QR decomposition.\n"
               "It has only linear complexity (that is, O(n) where n is the "
               "dimension of the square matrix) as the QR decomposition has "
               "already been computed.\n"
               "Note: This is only for square matrices. This method is useful "
               "to "
               "work around the risk of overflow/underflow that's inherent to "
               "determinant computation.")
          .def("dimensionOfKernel", &Solver::dimensionOfKernel,
               "Returns the dimension of the kernel of the matrix of which "
               "*this "
               "is the QR decomposition.")
          .def("isInjective", &Solver::isInjective,
               "Returns true if the matrix associated with this QR "
               "decomposition "
               "represents an injective linear map, i.e. has trivial kernel; "
               "false otherwise.\n"
               "\n"
               "Note: This method has to determine which pivots should be "
               "considered nonzero. For that, it uses the threshold value that "
               "you can control by calling setThreshold(threshold).")
          .def(
              "isInvertible", &Solver::isInvertible,
              "Returns true if the matrix associated with the QR decomposition "
              "is invertible.\n"
              "\n"
              "Note: This method has to determine which pivots should be "
              "considered nonzero. For that, it uses the threshold value that "
              "you can control by calling setThreshold(threshold).")
          .def("isSurjective", &Solver::isSurjective,
               "Returns true if the matrix associated with this QR "
               "decomposition "
               "represents a surjective linear map; false otherwise.\n"
               "\n"
               "Note: This method has to determine which pivots should be "
               "considered nonzero. For that, it uses the threshold value that "
               "you can control by calling setThreshold(threshold).")
          .def("maxPivot", &Solver::maxPivot,
               "Returns the absolute value of the biggest pivot, i.e. the "
               "biggest diagonal coefficient of U.")
          .def(
              "nonzeroPivots", &Solver::nonzeroPivots,
              "Returns the number of nonzero pivots in the QR decomposition. "
              "Here nonzero is meant in the exact sense, not in a fuzzy sense. "
              "So that notion isn't really intrinsically interesting, but it "
              "is "
              "still useful when implementing algorithms.")
          .def("rank", &Solver::rank,
               "Returns the rank of the matrix associated with the QR "
               "decomposition.\n"
               "\n"
               "Note: This method has to determine which pivots should be "
               "considered nonzero. For that, it uses the threshold value that "
               "you can control by calling setThreshold(threshold).")

          .def(
              "setThreshold",
              [](Solver &c, RealScalar const &threshold) {
                return c.setThreshold(threshold);
              },
              nb::arg("threshold"),
              "Allows to prescribe a threshold to be used by certain methods, "
              "such as rank(), who need to determine when pivots are to be "
              "considered nonzero. This is not used for the QR decomposition "
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
              "threshold", &Solver::threshold,
              "Returns the threshold that will be used by certain methods such "
              "as rank().")

          .def("matrixQR", &Solver::matrixQR,
               "Returns the matrix where the Householder QR decomposition is "
               "stored in a LAPACK-compatible way.",
               nb::rv_policy::copy)
          .def("matrixR", &Solver::matrixR,
               "Returns the matrix where the result Householder QR is stored.",
               nb::rv_policy::copy)

          .def(
              "compute",
              [](Solver &c, MatrixType const &matrix) -> Solver & {
                return c.compute(matrix);
              },
              nb::arg("matrix"),
              "Computes the QR factorization of given matrix.",
              nb::rv_policy::reference)

          .def(
              "inverse",
              [](Solver const &c) -> MatrixType { return inverse(c); },
              "Returns the inverse of the matrix associated with the QR "
              "decomposition.")

          .def(
              "solve",
              [](Solver const &c, VectorType const &b) -> VectorType {
                return solve(c, b);
              },
              nb::arg("b"),
              "Returns the solution x of A x = B using the current "
              "decomposition of A where b is a right hand side vector.")
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
