/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include <nanobind/eigen/dense.h>
#include <unsupported/Eigen/IterativeSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MINRES>
struct MINRESVisitor : nb::def_visitor<MINRESVisitor<MINRES>> {
  using MatrixType = typename MINRES::MatrixType;
  using CtorArg = nb::DMap<const MatrixType>;

  template <typename... Ts>
  void execute(nb::class_<MINRES, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<CtorArg>(), "A"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")
        .def(IterativeSolverVisitor<MINRES>());
  }

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<MINRES>(m)) {
      return;
    }
    nb::class_<MINRES>(m, name).def(MINRESVisitor<MINRES>()).def(IdVisitor());
  }
};

template <typename MINRES>
void exposeMINRES(nb::module_& m, const char* name) {
  MINRESVisitor<MINRES>::expose(m, name);
}

}  // namespace nanoeigenpy
