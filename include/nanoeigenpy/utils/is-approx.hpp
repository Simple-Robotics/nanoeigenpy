/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/eigen-typedef.hpp"
#include "nanoeigenpy/fwd.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <nanobind/nanobind.h>

namespace nanoeigenpy {

template <typename MatrixOrVectorType1, typename MatrixOrVectorType2>
EIGEN_DONT_INLINE bool is_approx(
    const Eigen::MatrixBase<MatrixOrVectorType1>& mat1,
    const Eigen::MatrixBase<MatrixOrVectorType2>& mat2,
    const typename MatrixOrVectorType1::RealScalar& prec) {
  return mat1.isApprox(mat2, prec);
}

template <typename MatrixOrVectorType1, typename MatrixOrVectorType2>
EIGEN_DONT_INLINE bool is_approx(
    const Eigen::MatrixBase<MatrixOrVectorType1>& mat1,
    const Eigen::MatrixBase<MatrixOrVectorType2>& mat2) {
  return is_approx(
      mat1, mat2,
      Eigen::NumTraits<
          typename MatrixOrVectorType1::RealScalar>::dummy_precision());
}

template <typename MatrixOrVectorType1, typename MatrixOrVectorType2>
EIGEN_DONT_INLINE bool is_approx(
    const Eigen::SparseMatrixBase<MatrixOrVectorType1>& mat1,
    const Eigen::SparseMatrixBase<MatrixOrVectorType2>& mat2,
    const typename MatrixOrVectorType1::RealScalar& prec) {
  return mat1.isApprox(mat2, prec);
}

template <typename MatrixOrVectorType1, typename MatrixOrVectorType2>
EIGEN_DONT_INLINE bool is_approx(
    const Eigen::SparseMatrixBase<MatrixOrVectorType1>& mat1,
    const Eigen::SparseMatrixBase<MatrixOrVectorType2>& mat2) {
  return is_approx(
      mat1, mat2,
      Eigen::NumTraits<
          typename MatrixOrVectorType1::RealScalar>::dummy_precision());
}

namespace nb = nanobind;

template <typename Scalar>
void exposeIsApprox(nb::module_ m) {
  enum { Options = 0 };
  NANOEIGENPY_MAKE_TYPEDEFS(Scalar, Options, s, Eigen::Dynamic, X);
  NANOEIGENPY_UNUSED_TYPE(RowVectorXs);
  typedef typename MatrixXs::RealScalar RealScalar;

  using namespace Eigen;
  const RealScalar dummy_precision =
      Eigen::NumTraits<RealScalar>::dummy_precision();

  // is_approx for dense matrices
  m.def(
      "is_approx",
      [](const MatrixXs& mat1, const MatrixXs& mat2, RealScalar precision) {
        return is_approx(mat1, mat2, precision);
      },
      nb::arg("mat1"), nb::arg("mat2"), nb::arg("precision") = dummy_precision,
      "Check if two dense matrices are approximately equal.");

  // is_approx for dense vectors
  m.def(
      "is_approx",
      [](const VectorXs& vec1, const VectorXs& vec2, RealScalar precision) {
        return is_approx(vec1, vec2, precision);
      },
      nb::arg("vec1"), nb::arg("vec2"), nb::arg("precision") = dummy_precision,
      "Check if two dense vectors are approximately equal.");
}

}  // namespace nanoeigenpy
