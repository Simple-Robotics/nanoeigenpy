/// Copyright 2025 INRIA
#pragma once

#define NANOEIGENPY_UNUSED_VARIABLE(var) (void)(var)
#define NANOEIGENPY_UNUSED_TYPE(type) \
  NANOEIGENPY_UNUSED_VARIABLE((type *)(NULL))
