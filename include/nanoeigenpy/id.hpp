/// Copyright 2025 INRIA
#pragma once
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

namespace nanoeigenpy {
namespace nb = nanobind;

struct IdVisitor : nb::def_visitor<IdVisitor> {
  template <typename C, typename... Ts>
  void execute(nb::class_<C, Ts...> &cl) {
    using EigenBase = Eigen::EigenBase<C>;
    static_assert(std::is_base_of_v<EigenBase, C>);  // Good here ? because of
                                                     // SimplicialCholesky, etc
    cl.def(
        "id",
        [](C const &self) -> int64_t {
          return reinterpret_cast<int64_t>(&self);
        },
        "Returns the unique identity of an object.\n"
        "For object held in C++, it corresponds to its memory address.");
  }

  //   private:
  //   template <typename C>
  //   static int64_t id(const C& self) {
  //     return reinterpret_cast<int64_t>(&self);
  //   }
};

}  // namespace nanoeigenpy
