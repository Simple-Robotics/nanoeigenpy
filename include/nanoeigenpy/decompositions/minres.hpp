/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <unsupported/Eigen/IterativeSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename _Solver>
struct IterativeSolverBaseVisitor
    : nb::def_visitor<IterativeSolverBaseVisitor<_Solver>> {
  using Solver = _Solver;
  using MatrixType = typename Solver::MatrixType;
  using Preconditioner = typename Solver::Preconditioner;
  using Scalar = typename Solver::Scalar;
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                 MatrixType::Options>;

 public:
  template <typename... Ts>
  void execute(nb::class_<Solver, Ts...>& cl) {
    using namespace nb::literals;

    cl.def(
          "analyzePattern",
          [](Solver& c, Eigen::EigenBase<MatrixType> const& matrix) -> Solver& {
            return c.analyzePattern(matrix);
          },
          "matrix"_a,
          "Initializes the iterative solver for the sparsity pattern of "
          "the matrix A for further solving Ax=b problems.",
          nb::rv_policy::reference)
        .def(
            "factorize",
            [](Solver& c, Eigen::EigenBase<MatrixType> const& matrix)
                -> Solver& { return c.factorize(matrix); },
            nb::arg("matrix"),
            "Initializes the iterative solver with the numerical values of "
            "the "
            "matrix A for further solving Ax=b problems.",
            nb::rv_policy::reference)
        .def(
            "compute",
            [](Solver& c, Eigen::EigenBase<MatrixType> const& matrix)
                -> Solver& { return c.compute(matrix); },
            nb::arg("matrix"),
            "Initializes the iterative solver with the matrix A for further "
            "solving Ax=b problems.",
            nb::rv_policy::reference)

        .def("rows", &Solver::rows, "Returns the number of rows.")
        .def("cols", &Solver::cols, "Returns the number of columns.")

        .def("tolerance", &Solver::tolerance,
             "Returns the tolerance threshold used by the stopping criteria.")
        .def("setTolerance", &Solver::setTolerance, "tolerance"_a,
             "Sets the tolerance threshold used by the stopping criteria.\n"
             "This value is used as an upper bound to the relative residual "
             "error: |Ax-b|/|b|.\n"
             "The default value is the machine precision given by "
             "NumTraits<Scalar>::epsilon().",
             nb::rv_policy::reference)
        .def(
            "preconditioner",
            [](Solver& c) -> Preconditioner& { return c.preconditioner(); },
            "Returns a read-write reference to the preconditioner for custom "
            "configuration.",
            nb::rv_policy::reference_internal)

        .def("maxIterations", &Solver::maxIterations,
             "Returns the max number of iterations.\n"
             "It is either the value setted by setMaxIterations or, by "
             "default, twice the number of columns of the matrix.")
        .def("setMaxIterations", &Solver::setMaxIterations, "max_iterations"_a,
             "Sets the max number of iterations.\n"
             "Default is twice the number of columns of the matrix.",
             nb::rv_policy::reference)

        .def("iterations", &Solver::iterations,
             "Returns the number of iterations performed during the last "
             "solve.")
        .def("error", &Solver::error,
             "Returns the tolerance error reached during the last solve.\n"
             "It is a close approximation of the true relative residual error "
             "|Ax-b|/|b|.")
        .def("info", &Solver::info,
             "Returns Success if the iterations converged, and NoConvergence "
             "otherwise.")

        .def("solveWithGuess", &solveWithGuess<MatrixXs, MatrixXs>, "b"_a,
             "x0"_a,
             "Returns the solution x of A x = b using the current "
             "decomposition of A and x0 as an initial solution.")

        .def(
            "solve", &solve<MatrixXs>, "b"_a,
            "Returns the solution x of A x = b using the current decomposition "
            "of A where b is a right hand side matrix or vector.");
  }

 private:
  template <typename MatrixOrVector1, typename MatrixOrVector2>
  static MatrixOrVector1 solveWithGuess(const Solver& self,
                                        const MatrixOrVector1& b,
                                        const MatrixOrVector2& guess) {
    return self.solveWithGuess(b, guess);
  }

  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver& self,
                              const MatrixOrVector& mat_or_vec) {
    MatrixOrVector res = self.solve(mat_or_vec);
    return res;
  }
};

template <typename _MatrixType>
struct MINRESSolverVisitor : nb::def_visitor<MINRESSolverVisitor<_MatrixType>> {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using VectorXs =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>;
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                 MatrixType::Options>;
  using Solver = Eigen::MINRES<MatrixType>;

 public:
  template <typename... Ts>
  void execute(nb::class_<Solver, Ts...>& cl) {
    using namespace nb::literals;

    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<const MatrixType&>(), "matrix"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")

        .def(IterativeSolverBaseVisitor<Solver>());
  }

  static void expose(nb::module_& m, const char* name) {
    nb::class_<Solver>(
        m, name,
        "A minimal residual solver for sparse symmetric problems.\n"
        "This class allows to solve for A.x = b sparse linear problems using "
        "the MINRES algorithm of Paige and Saunders (1975). The sparse "
        "matrix "
        "A must be symmetric (possibly indefinite). The vectors x and b can "
        "be "
        "either dense or sparse.\n"
        "The maximal number of iterations and tolerance value can be "
        "controlled via the setMaxIterations() and setTolerance() methods. "
        "The "
        "defaults are the size of the problem for the maximal number of "
        "iterations and NumTraits<Scalar>::epsilon() for the tolerance.\n")
        .def(MINRESSolverVisitor())
        .def(IdVisitor());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver& self, const MatrixOrVector& vec) {
    return self.solve(vec);
  }
};

template <typename _MatrixType>
void exposeMINRESSolver(nb::module_& m, const char* name) {
  MINRESSolverVisitor<_MatrixType>::expose(m, name);
}

}  // namespace nanoeigenpy
