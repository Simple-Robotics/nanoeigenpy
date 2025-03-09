/// Copyright 2025 INRIA
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <nanobind/nanobind.h>

#include "nanoeigenpy/eigen-typedef.hpp"
#include "nanoeigenpy/fwd.hpp"

namespace nanoeigenpy {

template <typename MatrixType1, typename MatrixType2>
EIGEN_DONT_INLINE bool is_approx(const Eigen::MatrixBase<MatrixType1>& mat1,
                                 const Eigen::MatrixBase<MatrixType2>& mat2,
                                 const typename MatrixType1::RealScalar& prec) {
  return mat1.isApprox(mat2, prec);
}

template <typename MatrixType1, typename MatrixType2>
EIGEN_DONT_INLINE bool is_approx(const Eigen::MatrixBase<MatrixType1>& mat1,
                                 const Eigen::MatrixBase<MatrixType2>& mat2) {
  return is_approx(
      mat1, mat2,
      Eigen::NumTraits<typename MatrixType1::RealScalar>::dummy_precision());
}

template <typename MatrixType1, typename MatrixType2>
EIGEN_DONT_INLINE bool is_approx(
    const Eigen::SparseMatrixBase<MatrixType1>& mat1,
    const Eigen::SparseMatrixBase<MatrixType2>& mat2,
    const typename MatrixType1::RealScalar& prec) {
  return mat1.isApprox(mat2, prec);
}

template <typename MatrixType1, typename MatrixType2>
EIGEN_DONT_INLINE bool is_approx(
    const Eigen::SparseMatrixBase<MatrixType1>& mat1,
    const Eigen::SparseMatrixBase<MatrixType2>& mat2) {
  return is_approx(
      mat1, mat2,
      Eigen::NumTraits<typename MatrixType1::RealScalar>::dummy_precision());
}

namespace nb = nanobind;

template <typename Scalar>
void exposeIsApprox(nb::module_ m) {
    enum { Options = 0 };
    NANOEIGENPY_MAKE_TYPEDEFS(Scalar, Options, s, Eigen::Dynamic, X);
    NANOEIGENPY_UNUSED_TYPE(VectorXs);
    NANOEIGENPY_UNUSED_TYPE(RowVectorXs);
    typedef typename MatrixXs::RealScalar RealScalar;

    using namespace Eigen;
    const RealScalar dummy_precision =
        Eigen::NumTraits<RealScalar>::dummy_precision();

    // Exposition de la fonction is_approx pour matrices denses
    m.def("is_approx",
          [](const MatrixXs& mat1, const MatrixXs& mat2, RealScalar precision) {
              return is_approx(mat1, mat2, precision);
          },
          nb::arg("mat1"), nb::arg("mat2"),
          nb::arg("precision") = dummy_precision,
          "Check if two dense matrices are approximately equal.");
}

}  // namespace nanoeigenpy

