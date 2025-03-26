/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/config.hpp"
#include "nanoeigenpy/id.hpp"
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

#if defined(__clang__)
#define NANOEIGENPY_CLANG_COMPILER
#elif defined(__GNUC__)
#define NANOEIGENPY_GCC_COMPILER
#elif defined(_MSC_VER)
#define NANOEIGENPY_MSVC_COMPILER
#endif

#if (__cplusplus >= 202002L || (defined(_MSVC_LAG) && _MSVC_LANG >= 202002L))
#define NANOEIGENPY_WITH_CXX20_SUPPORT
#endif

#define NANOEIGENPY_UNUSED_TYPE(Type) (void)(Type*)(NULL)

#define NANOEIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, Size, SizeSuffix) \
  /** \ingroup matrixtypedefs */                                               \
  typedef Eigen::Matrix<Type, Size, Size, Options>                             \
      Matrix##SizeSuffix##TypeSuffix;                                          \
  /** \ingroup matrixtypedefs */                                               \
  typedef Eigen::Matrix<Type, Size, 1> Vector##SizeSuffix##TypeSuffix;         \
  /** \ingroup matrixtypedefs */                                               \
  typedef Eigen::Matrix<Type, 1, Size> RowVector##SizeSuffix##TypeSuffix;

namespace nanoeigenpy {
namespace nb = nanobind;
}
