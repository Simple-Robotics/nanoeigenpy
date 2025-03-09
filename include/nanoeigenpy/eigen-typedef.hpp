/// Copyright 2025 INRIA
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#define NANOEIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, Size, SizeSuffix) \
  /** \ingroup matrixtypedefs */                                                \
  typedef Eigen::Matrix<Type, Size, Size, Options>                              \
      Matrix##SizeSuffix##TypeSuffix;                                           \
  /** \ingroup matrixtypedefs */                                                \
  typedef Eigen::Matrix<Type, Size, 1> Vector##SizeSuffix##TypeSuffix;          \
  /** \ingroup matrixtypedefs */                                                \
  typedef Eigen::Matrix<Type, 1, Size> RowVector##SizeSuffix##TypeSuffix;
