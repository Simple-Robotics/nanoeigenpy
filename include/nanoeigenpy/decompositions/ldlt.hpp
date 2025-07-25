/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <Eigen/Cholesky>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::LDLT<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename _MatrixType>
void exposeLDLT(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::LDLT<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "Robust Cholesky decomposition of a matrix with pivoting.\n\n"
      "Perform a robust Cholesky decomposition of a positive semidefinite "
      "or "
      "negative semidefinite matrix $ A $ such that $ A = P^TLDL^*P $, "
      "where "
      "P is a permutation matrix, L is lower triangular with a unit "
      "diagonal "
      "and D is a diagonal matrix.\n\n"
      "The decomposition uses pivoting to ensure stability, so that L will "
      "have zeros in the bottom right rank(A) - n submatrix. Avoiding the "
      "square root on D also stabilizes the computation.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructs a LLT factorization from a given matrix.")

      .def(EigenBaseVisitor())

      .def("isNegative", &Solver::isNegative,
           "Returns true if the matrix is negative (semidefinite).")
      .def("isPositive", &Solver::isPositive,
           "Returns true if the matrix is positive (semidefinite).")

      .def(
          "matrixL", [](const Solver &c) -> MatrixType { return c.matrixL(); },
          "Returns the lower triangular matrix L.")
      .def(
          "matrixU", [](const Solver &c) -> MatrixType { return c.matrixU(); },
          "Returns the upper triangular matrix U.")
      .def(
          "vectorD", [](const Solver &c) -> VectorType { return c.vectorD(); },
          "Returns the coefficients of the diagonal matrix D.")
      .def("matrixLDLT", &Solver::matrixLDLT,
           "Returns the LDLT decomposition matrix made of the lower matrix "
           "L, the diagonal D, then the remaining part that corresponds to "
           "A.",
           nb::rv_policy::reference_internal)

      .def(
          "transpositionsP",
          [](const Solver &c) -> MatrixType {
            return c.transpositionsP() *
                   MatrixType::Identity(c.matrixL().rows(), c.matrixL().rows());
          },
          "Returns the permutation matrix P.")

      .def(
          "rankUpdate",
          [](Solver &c, const VectorType &w, Scalar sigma) -> Solver & {
            return c.rankUpdate(w, sigma);
          },
          "If LDL^* = A, then it becomes A + sigma * v v^*", "w"_a, "sigma"_a)

      .def("adjoint", &Solver::adjoint,
           "Returns the adjoint, that is, a reference to the decomposition "
           "itself as if the underlying matrix is self-adjoint.",
           nb::rv_policy::reference)

      .def(
          "compute",
          [](Solver &c, const MatrixType &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes the LDLT of given matrix.",
          nb::rv_policy::reference)
      .def("info", &Solver::info,
           "NumericalIssue if the input contains INF or NaN values or "
           "overflow occured. Returns Success otherwise.")

      .def("rcond", &Solver::rcond,
           "Returns an estimate of the reciprocal condition number of the "
           "matrix.")

      .def("reconstructedMatrix", &Solver::reconstructedMatrix,
           "Returns the matrix represented by the decomposition, i.e., it "
           "returns the product: L L^*. This function is provided for debug "
           "purpose.")

      .def(
          "solve",
          [](const Solver &c, const VectorType &b) -> VectorType {
            return solve(c, b);
          },
          "b"_a,
          "Returns the solution x of A x = b using the current "
          "decomposition of A.")
      .def(
          "solve",
          [](const Solver &c, const MatrixType &B) -> MatrixType {
            return solve(c, B);
          },
          "B"_a,
          "Returns the solution X of A X = B using the current "
          "decomposition of A where B is a right hand side matrix.")

      .def("setZero", &Solver::setZero, "Clear any existing decomposition.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
