/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"
#include <Eigen/SparseQR>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

// template<typename SparseQRType>
// class SparseQRMatrixQReturnTypeWrapper {
// private:
//     Eigen::SparseQRMatrixQReturnType<SparseQRType> m_q_expr;

// public:
//     explicit SparseQRMatrixQReturnTypeWrapper(const SparseQRType& qr)
//         : m_q_expr(qr) {}

//     Eigen::Index rows() const { return m_q_expr.rows(); }
//     Eigen::Index cols() const { return m_q_expr.cols(); }

//     Eigen::VectorXd multiply_vec(const Eigen::VectorXd& vec) {
//      return Eigen::VectorXd(m_q_expr * vec);
//     }

//     Eigen::MatrixXd multiply_mat(const Eigen::MatrixXd& mat) {
//      return Eigen::MatrixXd(m_q_expr * mat);
//     }

//     auto adjoint() const {
//         return
//         SparseQRMatrixQTransposeReturnTypeWrapper<SparseQRType>(m_q_expr);
//     }

//     auto transpose() const {
//         return
//         SparseQRMatrixQTransposeReturnTypeWrapper<SparseQRType>(m_q_expr);
//     }
// };

// template<typename SparseQRType>
// class SparseQRMatrixQTransposeReturnTypeWrapper {
// private:
//     Eigen::SparseQRMatrixQTransposeReturnType<SparseQRType> m_qt_expr;

// public:
//     explicit SparseQRMatrixQTransposeReturnTypeWrapper(const SparseQRType&
//     qt)
//         : m_qt_expr(qt) {}

//     Eigen::VectorXd multiply_vec(const Eigen::VectorXd& vec) {
//      return Eigen::VectorXd(m_qt_expr * vec);
//     }

//     Eigen::MatrixXd multiply_mat(const Eigen::MatrixXd& mat) {
//      return Eigen::MatrixXd(m_qt_expr * mat);
//     }
// };

template <typename _MatrixType, typename _Ordering = Eigen::COLAMDOrdering<
                                    typename _MatrixType::StorageIndex>>
void exposeSparseQR(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Ordering = _Ordering;
  using Solver = Eigen::SparseQR<MatrixType, Ordering>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using QRMatrixType = Eigen::SparseMatrix<Scalar, Eigen::ColMajor,
                                           typename MatrixType::StorageIndex>;
  //   using QWrapper = SparseQRMatrixQReturnTypeWrapper<Solver>;
  //   using QTWrapper = SparseQRMatrixQTransposeReturnTypeWrapper<Solver>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }

  //   nb::class_<QWrapper>(m, "SparseQRMatrixQ")
  //       .def("rows", &QWrapper::rows)
  //       .def("cols", &QWrapper::cols)
  //       .def("__mul__", [](const QWrapper& self, const Eigen::Ref<const
  //       Eigen::VectorXd>& vec) {
  //           return self.multiply_vec(vec);
  //       }, "vec"_a)
  //       .def("__mul__", [](const QWrapper& self, const Eigen::Ref<const
  //       Eigen::MatrixXd>& mat) {
  //           return self.multiply_mat(mat);
  //       }, "mat"_a)
  //       .def("adjoint", &QWrapper::adjoint)
  //       .def("transpose", &QWrapper::transpose);

  //   nb::class_<QTWrapper>(m, "SparseQRMatrixQTranspose")
  //       .def("__mul__", [](const QTWrapper& self, const Eigen::Ref<const
  //       Eigen::VectorXd>& vec) {
  //           return self.multiply_vec(vec);
  //       }, "vec"_a)
  //       .def("__mul__", [](const QTWrapper& self, const Eigen::Ref<const
  //       Eigen::MatrixXd>& mat) {
  //           return self.multiply_mat(mat);
  //       }, "mat"_a);

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
           "The input matrix should be in compressed mode "
           "(see SparseMatrix::makeCompressed()).")

      //  .def("matrixQ", [](const Solver& self) -> QWrapper {
      //      return QWrapper(self);
      //      }, "Returns an expression of the matrix Q")

      .def(
          "matrixR",
          [](Solver &self) -> const QRMatrixType & { return self.matrixR(); },
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
          [](Solver &self, const RealScalar &thresh) -> void {
            return self.setPivotThreshold(thresh);
          },
          "Set the threshold used for a diagonal entry to be an acceptable "
          "pivot.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
