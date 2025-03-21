/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename LeastSquaresConjugateGradient>
struct LeastSquaresConjugateGradientVisitor
    : nb::def_visitor<
          LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient>> {
  using MatrixType = Eigen::MatrixXd;

  template <typename... Ts>
  void execute(nb::class_<LeastSquaresConjugateGradient, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<MatrixType>(), "A"_a,
             "Initialize the solver with matrix A for further || Ax - b || "
             "solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().");
  }

  static void expose(nb::module_& m, const char* name) {
    nb::class_<LeastSquaresConjugateGradient>(
        m, name, nb::is_final())  // is_final good to replace bp::noncopyable ?
        .def(IterativeSolverVisitor<LeastSquaresConjugateGradient>())
        .def(LeastSquaresConjugateGradientVisitor<
             LeastSquaresConjugateGradient>())
        .def(IdVisitor());
  }
};

template <typename LeastSquaresConjugateGradient>
void exposeLeastSquaresConjugateGradient(nb::module_& m, const char* name) {
  LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient>::expose(
      m, name);
}

}  // namespace nanoeigenpy
