/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include <nanobind/eigen/dense.h>
#include <Eigen/IterativeLinearSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename ConjugateGradient>
struct ConjugateGradientVisitor
    : nb::def_visitor<ConjugateGradientVisitor<ConjugateGradient>> {
  using MatrixType = typename ConjugateGradient::MatrixType;
  using CtorArg = nb::DMap<const MatrixType>;

  template <typename... Ts>
  void execute(nb::class_<ConjugateGradient, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<CtorArg>(), "A"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")
        .def(IterativeSolverVisitor<ConjugateGradient>());
  }

  static void expose(nb::module_& m, const char* name) {
    if (!register_symbolic_link_to_registered_type<ConjugateGradient>(m)) {
      nb::class_<ConjugateGradient>(m, name)
          .def(ConjugateGradientVisitor<ConjugateGradient>())
          .def(IdVisitor());
    }
  }
};

template <typename ConjugateGradient>
void exposeConjugateGradient(nb::module_& m, const char* name) {
  ConjugateGradientVisitor<ConjugateGradient>::expose(m, name);
}

}  // namespace nanoeigenpy
