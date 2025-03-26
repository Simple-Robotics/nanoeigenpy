/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-base.hpp"

namespace nanoeigenpy {

struct CholmodDecompositionVisitor
    : nb::def_visitor<CholmodDecompositionVisitor> {
  template <typename SimplicialDerived, typename... Ts>
  void execute(nb::class_<SimplicialDerived, Ts...> &cl) {
    using Solver = SimplicialDerived;

    cl.def(CholmodBaseVisitor())

        .def("setMode", &Solver::setMode, nb::arg("mode"),
             "Set the mode for the Cholesky decomposition.");
  }
};

}  // namespace nanoeigenpy
