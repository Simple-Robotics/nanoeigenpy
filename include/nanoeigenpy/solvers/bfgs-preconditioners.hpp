/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/solvers/basic-preconditioners.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Preconditioner>
struct BFGSPreconditionerBaseVisitor
    : nb::def_visitor<BFGSPreconditionerBaseVisitor<Preconditioner>> {
  using VectorType = Eigen::VectorXd;

  template <typename... Ts>
  void execute(nb::class_<Preconditioner, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def("rows", &Preconditioner::rows,
             "Returns the number of rows in the preconditioner.")
        .def("cols", &Preconditioner::rows,
             "Returns the number of cols in the preconditioner.")
        .def("dim", &Preconditioner::dim,
             "Returns the dimension of the BFGS preconditioner.")
        .def("update", &Preconditioner::update, "s"_a, "y"_a,
             "Update the BFGS estimate of the matrix A.",
             nb::rv_policy::reference)
        .def("reset", &Preconditioner::reset, "Reset the BFGS estimate.");
  }

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<Preconditioner>(m)) {
      return;
    }
    nb::class_<Preconditioner>(m, name)
        .def(BFGSPreconditionerBaseVisitor<Preconditioner>())
        .def(IdVisitor());
  }
};

template <typename Preconditioner>
void exposeBFGSPreconditionerBase(nb::module_& m, const char* name) {
  BFGSPreconditionerBaseVisitor<Preconditioner>::expose(m, name);
}

template <typename Preconditioner>
struct LimitedBFGSPreconditionerBaseVisitor
    : nb::def_visitor<LimitedBFGSPreconditionerBaseVisitor<Preconditioner>> {
  template <typename... Ts>
  void execute(nb::class_<Preconditioner, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def(BFGSPreconditionerBaseVisitor<Preconditioner>())
        .def("resize", &Preconditioner::resize, "dim"_a,
             "Resizes the preconditionner with size dim.",
             nb::rv_policy::reference);
  }

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<Preconditioner>(m)) {
      return;
    }
    nb::class_<Preconditioner>(m, name)
        .def(LimitedBFGSPreconditionerBaseVisitor<Preconditioner>())
        .def(IdVisitor());
  }
};

template <typename Preconditioner>
void exposeLimitedBFGSPreconditionerBase(nb::module_& m, const char* name) {
  LimitedBFGSPreconditionerBaseVisitor<Preconditioner>::expose(m, name);
}

}  // namespace nanoeigenpy
