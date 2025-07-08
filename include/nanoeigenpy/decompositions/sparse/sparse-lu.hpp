/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"
#include <Eigen/SparseLU>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType, typename _Ordering = Eigen::COLAMDOrdering<
                                    typename _MatrixType::StorageIndex>>
void exposeSparseLU(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::SparseLU<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using DenseVectorXs =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "Sparse supernodal LU factorization for general matrices.\n\n"
      "This class implements the supernodal LU factorization for general "
      "matrices. "
      "It uses the main techniques from the sequential SuperLU package "
      "(http://crd-legacy.lbl.gov/~xiaoye/SuperLU/). It handles transparently "
      "real "
      "and complex arithmetic with single and double precision, depending on "
      "the "
      "scalar type of your input matrix. The code has been optimized to "
      "provide BLAS-3 "
      "operations during supernode-panel updates. It benefits directly from "
      "the built-in "
      "high-performant Eigen BLAS routines. Moreover, when the size of a "
      "supernode is very "
      "small, the BLAS calls are avoided to enable a better optimization from "
      "the compiler. "
      "For best performance, you should compile it with NDEBUG flag to avoid "
      "the numerous "
      "bounds checking on vectors.\n\n"
      "An important parameter of this class is the ordering method. It is used "
      "to reorder the "
      "columns (and eventually the rows) of the matrix to reduce the number of "
      "new elements that "
      "are created during numerical factorization. The cheapest method "
      "available is COLAMD. See "
      "the OrderingMethods module for the list of built-in and external "
      "ordering methods.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructs a LU factorization from a given matrix.")

      .def(SparseSolverBaseVisitor())

      .def("analyzePattern", &Solver::analyzePattern,
           "Performs a symbolic decomposition on the sparcity of matrix.\n"
           "This function is particularly useful when solving for several "
           "problems having the same structure.")
      .def("factorize", &Solver::factorize,
           "Performs a numeric decomposition of a given matrix.\n"
           "The given matrix must has the same sparcity than the matrix on "
           "which the symbolic decomposition has been performed.\n"
           "See also analyzePattern().")
      .def("compute", &Solver::compute,
           "Compute the symbolic and numeric factorization of the input sparse "
           "matrix.\n\n"
           "The input matrix should be in column-major storage.")

      // TODO: Expose so that the return type are convertible to np arrays
      // transpose()
      // adjoint()
      // matrixU()
      // matrixL()

      .def("rows", &Solver::rows, "Returns the number of rows of the matrix.")
      .def("cols", &Solver::cols, "Returns the number of cols of the matrix.")

      .def("rowsPermutation", &Solver::rowsPermutation,
           "Returns a reference to the row matrix permutation "
           "\f$ P_r \f$ such that \f$P_r A P_c^T = L U\f$.",
           nb::rv_policy::reference_internal)
      .def("colsPermutation", &Solver::colsPermutation,
           "Returns a reference to the column matrix permutation"
           "\f$ P_c^T \f$ such that \f$P_r A P_c^T = L U\f$.",
           nb::rv_policy::reference_internal)

      .def(
          "setPivotThreshold",
          [](Solver &self, const RealScalar &thresh) -> void {
            return self.setPivotThreshold(thresh);
          },
          "Set the threshold used for a diagonal entry to be an acceptable "
          "pivot.")

      .def("info", &Solver::info,
           "Reports whether previous computation was successful.")
      .def("lastErrorMessage", &Solver::lastErrorMessage,
           "A string describing the type of error")

      .def(
          "absDeterminant", &Solver::absDeterminant,
          "Returns the absolute value of the determinant of the matrix of which"
          "*this is the QR decomposition.")
      .def("logAbsDeterminant", &Solver::logAbsDeterminant,
           "Returns the natural log of the absolute value of the determinant "
           "of the matrix of which **this is the QR decomposition")
      .def("signDeterminant", &Solver::signDeterminant,
           "A number representing the sign of the determinant")
      .def("determinant", &Solver::determinant,
           "The determinant of the matrix.")

      .def("nnzL", &Solver::nnzL, "The number of non zero elements in L")
      .def("nnzU", &Solver::nnzU, "The number of non zero elements in L")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
