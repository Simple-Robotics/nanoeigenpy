/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-base.hpp"

namespace nanoeigenpy {

struct CholmodDecompositionVisitor
    : nb::def_visitor<CholmodDecompositionVisitor> {
  template <typename CholdmodDerived, typename... Ts>
  void execute(nb::class_<CholdmodDerived, Ts...> &cl) {
    using Solver = CholdmodDerived;
    static_assert(nb::is_base_of_template_v<Solver, Eigen::SparseSolverBase>,
                  "Template type parameter Solver must inherit from "
                  "Eigen::SparseSolverBase");

    cl.def(CholmodBaseVisitor())
        .def("setMode", &Solver::setMode, "mode"_a,
             "Set the mode for the Cholesky decomposition.");
  }
};

}  // namespace nanoeigenpy
