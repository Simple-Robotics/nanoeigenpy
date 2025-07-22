/// Copyright 2025 INRIA

#pragma once

#include <nanobind/nanobind.h>

namespace nanoeigenpy {
namespace nb = nanobind;

struct IdVisitor : nb::def_visitor<IdVisitor> {
  template <typename C, typename... Ts>
  void execute(nb::class_<C, Ts...> &cl) {
    cl.def(
        "id",
        [](const C &self) -> int64_t {
          return reinterpret_cast<int64_t>(&self);
        },
        "Returns the unique identity of an object.\n"
        "For object held in C++, it corresponds to its memory address.");
  }
};

}  // namespace nanoeigenpy
