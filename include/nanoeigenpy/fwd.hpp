/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/config.hpp"
#include "nanoeigenpy/id.hpp"
#include <nanobind/nanobind.h>

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

#define NANOEIGENPY_UNUSED_TYPE(Type) (Type*)(NULL)

namespace nanoeigenpy {
namespace nb = nanobind;
}
