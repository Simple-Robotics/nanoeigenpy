/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"
#include <Eigen/SparseQR>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename SparseQRType>
void exposeMatrixQ(nb::module_ m) {
  using Scalar = typename SparseQRType::Scalar;
  using QType = Eigen::SparseQRMatrixQReturnType<SparseQRType>;
  using QTransposeType =
      Eigen::SparseQRMatrixQTransposeReturnType<SparseQRType>;
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using QRMatrixType = typename SparseQRType::QRMatrixType;

  if (!check_registration_alias<QTransposeType>(m)) {
    nb::class_<QTransposeType>(m, "SparseQRMatrixQTransposeReturnType")
        .def(nb::init<const SparseQRType&>(), "qr"_a)

        .def(
            "__matmul__",
            [](QTransposeType& self, const MatrixXd& other) -> MatrixXd {
              return MatrixXd(self * other);
            },
            "other"_a)

        .def(
            "__matmul__",
            [](QTransposeType& self, const VectorXd& other) -> VectorXd {
              return VectorXd(self * other);
            },
            "other"_a);
  }

  if (!check_registration_alias<QType>(m)) {
    nb::class_<QType>(m, "SparseQRMatrixQReturnType")
        .def(nb::init<const SparseQRType&>(), "qr"_a)

        .def_prop_ro("rows", &QType::rows)
        .def_prop_ro("cols", &QType::cols)

        .def(
            "__matmul__",
            [](QType& self, const MatrixXd& other) -> MatrixXd {
              return MatrixXd(self * other);
            },
            "other"_a)

        .def(
            "__matmul__",
            [](QType& self, const VectorXd& other) -> VectorXd {
              return VectorXd(self * other);
            },
            "other"_a)

        .def("adjoint",
             [](const QType& self) -> QTransposeType { return self.adjoint(); })

        .def("transpose", [](const QType& self) -> QTransposeType {
          return self.transpose();
        });
  }
}

template <typename _MatrixType, typename _Ordering = Eigen::COLAMDOrdering<
                                    typename _MatrixType::StorageIndex>>
void exposeSparseQR(nb::module_ m, const char* name) {
  using MatrixType = _MatrixType;
  using Ordering = _Ordering;
  using Solver = Eigen::SparseQR<MatrixType, Ordering>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using QRMatrixType = typename Solver::QRMatrixType;
  using QType = Eigen::SparseQRMatrixQReturnType<Solver>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }

  exposeMatrixQ<Solver>(m);

  nb::class_<Solver>(
      m, name,
      "Sparse left-looking QR factorization with numerical column pivoting. "
      "\n\n"
      "This class implements a left-looking QR decomposition of sparse "
      "matrices with "
      "numerical column pivoting. When a column has a norm less than a given "
      "tolerance "
      "it is implicitly permuted to the end. The QR factorization thus "
      "obtained is given "
      "by A*P = Q*R where R is upper triangular or trapezoidal.\n\n"
      "P is the column permutation which is the product of the fill-reducing "
      "and the "
      "numerical permutations. Use colsPermutation() to get it.\n\n"
      "Q is the orthogonal matrix represented as products of Householder "
      "reflectors. "
      "Use matrixQ() to get an expression and matrixQ().adjoint() to get the "
      "adjoint. Y"
      "ou can then apply it to a vector.\n\n"
      "R is the sparse triangular or trapezoidal matrix. The later occurs when "
      "A is "
      "rank-deficient. matrixR().topLeftCorner(rank(), rank()) always returns "
      "a triangular "
      "factor of full rank.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<const MatrixType&>(), "matrix"_a,
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
           "The input matrix should be in compressed mode "
           "(see SparseMatrix::makeCompressed()).")

      .def(
          "matrixQ", [](const Solver& self) -> QType { return self.matrixQ(); },
          "Returns an expression of the matrix Q as products of sparse "
          "Householder reflectors.")
      .def(
          "matrixR",
          [](const Solver& self) -> const QRMatrixType& {
            return self.matrixR();
          },
          "Returns a const reference to the \b sparse upper triangular matrix "
          "R of the QR factorization.",
          nb::rv_policy::reference_internal)

      .def("rows", &Solver::rows, "Returns the number of rows of the matrix.")
      .def("cols", &Solver::cols, "Returns the number of cols of the matrix.")

      .def("rank", &Solver::rank,
           "Returns the number of non linearly dependent columns as determined "
           "by the pivoting threshold.")

      .def("colsPermutation", &Solver::colsPermutation,
           "Returns a reference to the column matrix permutation"
           "\f$ P_c^T \f$ such that \f$P_r A P_c^T = L U\f$.",
           nb::rv_policy::reference_internal)

      .def("info", &Solver::info,
           "Reports whether previous computation was successful.")
      .def("lastErrorMessage", &Solver::lastErrorMessage,
           "A string describing the type of error")

      .def(
          "setPivotThreshold",
          [](Solver& self, const RealScalar& thresh) -> void {
            return self.setPivotThreshold(thresh);
          },
          "Set the threshold used for a diagonal entry to be an acceptable "
          "pivot.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
