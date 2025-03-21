/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename SparseSolver>
struct SparseSolverVisitor
    : nb::def_visitor<SparseSolverVisitor<SparseSolver>> {
  using VectorType = Eigen::VectorXd;

  template <typename... Ts>
  void execute(nb::class_<SparseSolver, Ts...> &cl) {
    using namespace nb::literals;
    cl.def("solve", &solve, "b"_a,
           "Returns the solution x of Ax = b using the current decomposition "
           "of A.");
  }

 private:
  static VectorType solve(const SparseSolver &self, const VectorType &vec) {
    return self.solve(vec);
  }
};

}  // namespace nanoeigenpy
