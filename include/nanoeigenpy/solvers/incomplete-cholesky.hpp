/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeIncompleteCholesky(nb::module_ m, const char* name) {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Solver = Eigen::IncompleteCholesky<Scalar>;
  using Factortype = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
  using VectorRx = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
  using PermutationType =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

  static constexpr int Options =
      MatrixType::Options;  // Options = Eigen::ColMajor
  using DenseVectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
  using DenseMatrixXs =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(m, name,
                     "Modified Incomplete Cholesky with dual threshold.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<const MatrixType&>(), "matrix"_a,
           "Constructs an incomplete cholesky factorization from a given "
           "matrix.")

      .def("rows", &Solver::rows, "Returns the number of rows of the matrix.")
      .def("cols", &Solver::cols, "Returns the number of cols of the matrix.")

      .def("info", &Solver::info,
           "Reports whether previous computation was successful.")

      .def("setInitialShift", &Solver::setInitialShift, "shift"_a,
           "Set the initial shift parameter.")

      .def(
          "analyzePattern",
          [](Solver& self, const MatrixType& amat) {
            self.analyzePattern(amat);
          },
          "matrix"_a)
      .def(
          "factorize",
          [](Solver& self, const MatrixType& amat) { self.factorize(amat); },
          "matrix"_a)
      .def(
          "compute",
          [](Solver& self, const MatrixType& amat) { self.compute(amat); },
          "matrix"_a)

      .def(
          "matrixL",
          [](Solver& self) -> const Factortype& { return self.matrixL(); },
          nb::rv_policy::reference_internal)
      .def(
          "scalingS",
          [](Solver& self) -> const VectorRx& { return self.scalingS(); },
          nb::rv_policy::reference_internal)
      .def(
          "permutationP",
          [](Solver& self) -> const PermutationType& {
            return self.permutationP();
          },
          nb::rv_policy::reference_internal)

      .def(
          "solve",
          [](Solver const& self, const Eigen::Ref<DenseVectorXs const>& b)
              -> DenseVectorXs { return self.solve(b); },
          "b"_a,
          "Returns the solution x of A x = b using the current decomposition "
          "of A, where b is a right hand side vector.")
      .def(
          "solve",
          [](Solver const& self, const Eigen::Ref<DenseMatrixXs const>& B)
              -> DenseMatrixXs { return self.solve(B); },
          "B"_a,
          "Returns the solution X of A X = B using the current decomposition "
          "of A where B is a right hand side matrix.")
      .def(
          "solve",
          [](Solver const& self, const MatrixType& B) -> MatrixType {
            DenseMatrixXs B_dense = DenseMatrixXs(B);
            DenseMatrixXs X_dense = self.solve(B_dense);
            return MatrixType(X_dense.sparseView());
          },
          "B"_a,
          "Returns the solution X of A X = B using the current decomposition "
          "of A where B is a right hand side matrix.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
