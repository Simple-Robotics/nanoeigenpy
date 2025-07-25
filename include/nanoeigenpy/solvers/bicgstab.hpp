/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include <nanobind/eigen/dense.h>
#include <Eigen/IterativeLinearSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename BiCGSTAB>
struct BiCGSTABVisitor : nb::def_visitor<BiCGSTABVisitor<BiCGSTAB>> {
  using MatrixType = typename BiCGSTAB::MatrixType;
  using CtorArg = nb::DMap<const MatrixType>;

  template <typename... Ts>
  void execute(nb::class_<BiCGSTAB, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<CtorArg>(), "A"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")
        .def(IterativeSolverVisitor<BiCGSTAB>());
  }

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<BiCGSTAB>(m)) {
      return;
    }
    nb::class_<BiCGSTAB>(m, name)
        .def(BiCGSTABVisitor<BiCGSTAB>())
        .def(IdVisitor());
  }
};

template <typename BiCGSTAB>
void exposeBiCGSTAB(nb::module_& m, const char* name) {
  BiCGSTABVisitor<BiCGSTAB>::expose(m, name);
}

}  // namespace nanoeigenpy
