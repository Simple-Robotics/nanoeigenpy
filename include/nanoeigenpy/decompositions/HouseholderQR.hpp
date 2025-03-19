/// Copyright 2025 INRIA
#pragma once
#include "base.hpp"
#include <Eigen/QR>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::HouseholderQR<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename MatrixType>
void exposeHouseholderQRSolver(nb::module_ m, const char *name) {
  using Solver = Eigen::HouseholderQR<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;
  auto cl =
      nb::class_<Solver>(
          m, name,
          "This class performs a QR decomposition of a matrix A into matrices "
          "Q "
          "and R such that A=QR by using Householder transformations.\n"
          "Here, Q a unitary matrix and R an upper triangular matrix. The "
          "result "
          "is stored in a compact way compatible with LAPACK.\n"
          "\n"
          "Note that no pivoting is performed. This is not a rank-revealing "
          "decomposition. If you want that feature, use FullPivHouseholderQR "
          "or "
          "ColPivHouseholderQR instead.\n"
          "\n"
          "This Householder QR decomposition is faster, but less numerically "
          "stable and less feature-full than FullPivHouseholderQR or "
          "ColPivHouseholderQR.")

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

          .def("householderQ", &Solver::matrixQR,
               "Returns the matrix where the Householder QR decomposition is "
               "stored in a LAPACK-compatible way.",
               nb::rv_policy::copy)

          .def("matrixQR", &Solver::matrixQR,
               "Returns the matrix where the Householder QR decomposition is "
               "stored in a LAPACK-compatible way.",
               nb::rv_policy::copy)

          .def(
              "compute",
              [](Solver &c, VectorType const &matrix) {
                return c.compute(matrix);
              },
              nb::arg("matrix"),
              "Computes the QR factorization of given matrix.",
              nb::rv_policy::reference)

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

          .def(
              "id",
              [](Solver const &c) -> int64_t {
                return reinterpret_cast<int64_t>(&c);
              },
              "Returns the unique identity of an object.\n"
              "For objects held in C++, it corresponds to its memory address.");
}

}  // namespace nanoeigenpy
