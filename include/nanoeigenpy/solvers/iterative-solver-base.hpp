/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/solvers/sparse-solver-base.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename IterativeSolver>
struct IterativeSolverVisitor
    : nb::def_visitor<IterativeSolverVisitor<IterativeSolver>> {
  using MatrixType = typename IterativeSolver::MatrixType;
  using Preconditioner = typename IterativeSolver::Preconditioner;
  using VectorType = Eigen::VectorXd;

  template <typename... Ts>
  void execute(nb::class_<IterativeSolver, Ts...>& cl) {
    using IS = IterativeSolver;
    using namespace nb::literals;
    cl.def(SparseSolverVisitor<IS>())
        .def("error", &IS::error,
             "Returns the tolerance error reached during the last solve.\n"
             "It is a close approximation of the true relative residual error "
             "|Ax-b|/|b|.")
        .def("info", &IS::info,
             "Returns success if the iterations converged, and NoConvergence "
             "otherwise.")
        .def(
            "iterations", &IS::iterations,
            "Returns the number of iterations performed during the last solve.")
        .def("maxIterations", &IS::maxIterations,
             "Returns the max number of iterations.\n"
             "It is either the value setted by setMaxIterations or, by "
             "default, twice the number of columns of the matrix.")
        .def("setMaxIterations", &IS::setMaxIterations,
             "Sets the max number of iterations.\n"
             "Default is twice the number of columns of the matrix.",
             nb::rv_policy::reference)
        .def("tolerance", &IS::tolerance,
             "Returns he tolerance threshold used by the stopping criteria.")
        .def("setTolerance", &IS::setTolerance,
             "Sets the tolerance threshold used by the stopping criteria.\n"
             "This value is used as an upper bound to the relative residual "
             "error: |Ax-b|/|b|. The default value is the machine precision.",
             nb::rv_policy::reference)
        .def("analyzePattern", &analyzePattern, "A"_a,
             "Initializes the iterative solver for the sparsity pattern of the "
             "matrix A for further solving Ax=b problems.\n"
             "Currently, this function mostly calls analyzePattern on the "
             "preconditioner.\n"
             "In the future we might, for instance, implement column "
             "reordering for faster matrix vector products.",
             nb::rv_policy::reference)
        .def("factorize", &factorize, "A"_a,
             "Initializes the iterative solver with the numerical values of "
             "the matrix A for further solving Ax=b problems.\n"
             "Currently, this function mostly calls factorize on the "
             "preconditioner.",
             nb::rv_policy::reference)
        .def("compute", &compute, "A"_a,
             "Initializes the iterative solver with the numerical values of "
             "the matrix A for further solving Ax=b problems.\n"
             "Currently, this function mostly calls factorize on the "
             "preconditioner.\n"
             "In the future we might, for instance, implement column "
             "reordering for faster matrix vector products.",
             nb::rv_policy::reference)
        .def("solveWithGuess", &solveWithGuess, "b"_a, "x_0"_a,
             "Returns the solution x of Ax = b using the current decomposition "
             "of A and x0 as an initial solution.")
        .def(
            "preconditioner",
            [](IterativeSolver& self) -> Preconditioner& {
              return const_cast<Preconditioner&>(self.preconditioner());
            },
            nb::rv_policy::reference_internal,
            "Returns a read-write reference to the preconditioner for custom "
            "configuration.");
  }

 private:
  static IterativeSolver& factorize(IterativeSolver& self,
                                    const MatrixType& m) {
    return self.factorize(m);
  }

  static IterativeSolver& compute(IterativeSolver& self, const MatrixType& m) {
    return self.compute(m);
  }

  static IterativeSolver& analyzePattern(IterativeSolver& self,
                                         const MatrixType& m) {
    return self.analyzePattern(m);
  }

  static VectorType solveWithGuess(IterativeSolver& self,
                                   const Eigen::VectorXd& b,
                                   const Eigen::VectorXd& x0) {
    return self.solveWithGuess(b, x0);
  }
};

}  // namespace nanoeigenpy
