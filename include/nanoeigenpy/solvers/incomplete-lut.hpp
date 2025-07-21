/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename _MatrixType>
void exposeIncompleteLUT(nb::module_ m, const char* name) {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Solver = Eigen::IncompleteLUT<Scalar>;

  static constexpr int Options =
      MatrixType::Options;  // Options = Eigen::ColMajor
  using DenseVectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
  using DenseMatrixXs =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "Incomplete LU factorization with dual-threshold strategy. "
      "\n\n"
      "This class follows the sparse solver concept .\n\n"
      "During the numerical factorization, two dropping rules are used : "
      "1) any element whose magnitude is less than some tolerance is dropped. "
      "This tolerance is obtained by multiplying the input tolerance droptol "
      "by the average magnitude of all the original elements in the current "
      "row. 2) After the elimination of the row, only the fill largest "
      "elements "
      "in the L part and the fill largest elements in the U part are kept (in "
      "addition to the diagonal element ). Note that fill is computed from the "
      "input parameter fillfactor which is used the ratio to control the "
      "fill_in "
      "relatively to the initial number of nonzero elements. The two extreme "
      "cases are when droptol=0 (to keep all the fill*2 largest elements) and "
      "when fill=n/2 with droptol being different to zero.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<const MatrixType&>(), "matrix"_a,
           "Constructs an incomplete LU factorization from a given matrix.")
      .def(
          "__init__",
          [](Solver* self, const MatrixType& mat,
             RealScalar droptol = Eigen::NumTraits<Scalar>::dummy_precision(),
             int fillfactor = 10) {
            new (self) Solver(mat, droptol, fillfactor);
          },
          "matrix"_a, "droptol"_a = Eigen::NumTraits<Scalar>::dummy_precision(),
          "fillfactor"_a = 10)

      .def("rows", &Solver::rows, "Returns the number of rows of the matrix.")
      .def("cols", &Solver::cols, "Returns the number of cols of the matrix.")

      .def("info", &Solver::info,
           "Reports whether previous computation was successful.")

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
          [](Solver& self, const MatrixType& amat) -> Solver& {
            return self.compute(amat);
          },
          "matrix"_a, nb::rv_policy::reference)

      .def("setDroptol", &Solver::setDroptol)
      .def("setFillfactor", &Solver::setFillfactor)

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
