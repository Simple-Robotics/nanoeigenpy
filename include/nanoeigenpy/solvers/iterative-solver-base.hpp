/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"

namespace nanoeigenpy {

template <typename IterativeSolver>
struct IterativeSolverVisitor
    : nb::def_visitor<IterativeSolverVisitor<IterativeSolver>> {
  using MatrixType = typename IterativeSolver::MatrixType;
  using Scalar = typename MatrixType::Scalar;
  static constexpr int Options = MatrixType::Options;
  using Preconditioner = typename IterativeSolver::Preconditioner;
  using DenseMatrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
  static_assert(
      nb::is_base_of_template_v<IterativeSolver, Eigen::IterativeSolverBase>,
      "IterativeSolver template type parameter must inherit from "
      "IterativeSolverBase!");

  template <typename... Ts>
  void execute(nb::class_<IterativeSolver, Ts...>& cl) {
    using IS = IterativeSolver;
    using namespace nb::literals;
    cl.def(
          "solve",
          [](const IS& self, Eigen::Ref<const VectorType> vec) -> VectorType {
            return self.solve(vec);
          },
          "Returns the solution x of Ax = b using the current decomposition of "
          "A.")
        .def(
            "solve",
            [](const IS& self, Eigen::Ref<const DenseMatrix> vec)
                -> DenseMatrix { return self.solve(vec); },
            "Returns the solution x of Ax = b using the current decomposition "
            "of A.")
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
        .def("solveWithGuess", &solveWithGuess<VectorType>, "b"_a, "x_0"_a,
             "Returns the solution x of Ax = b using the current decomposition "
             "of A and x0 as an initial solution.")
        .def("solveWithGuess", &solveWithGuess<DenseMatrix>, "b"_a, "x_0"_a,
             "Returns the solution x of Ax = b using the current decomposition "
             "of A and x0 as an initial solution.")
        .def(
            "preconditioner",
            [](IterativeSolver& self) -> Preconditioner& {
              return self.preconditioner();
            },
            nb::rv_policy::reference_internal,
            "Returns a read-write reference to the preconditioner for custom "
            "configuration.");
  }

 private:
  static IterativeSolver& factorize(IterativeSolver& self,
                                    Eigen::Ref<const MatrixType> m) {
    return self.factorize(m);
  }

  static IterativeSolver& compute(IterativeSolver& self,
                                  Eigen::Ref<const MatrixType> m) {
    return self.compute(m);
  }

  static IterativeSolver& analyzePattern(IterativeSolver& self,
                                         Eigen::Ref<const MatrixType> m) {
    return self.analyzePattern(m);
  }

  template <typename T>
  static T solveWithGuess(IterativeSolver& self, Eigen::Ref<const T> b,
                          Eigen::Ref<const T> x0) {
    return self.solveWithGuess(b, x0);
  }
};

}  // namespace nanoeigenpy
