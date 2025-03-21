/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <unsupported/Eigen/IterativeSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

struct IterativeSolverBaseVisitor
    : nb::def_visitor<IterativeSolverBaseVisitor> {
  template <typename _Solver, typename... Ts>
  void execute(nb::class_<_Solver, Ts...> &cl) {
    using Solver = _Solver;
    using MatrixType = typename Solver::MatrixType;
    using Preconditioner = typename Solver::Preconditioner;
    using Scalar = typename Solver::Scalar;
    using VectorXs =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>;

    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<const MatrixType &>(), nb::arg("matrix"),
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")

        .def(
            "analyzePattern",
            [](Solver &c, MatrixType const &matrix) -> Solver & {
              return c.analyzePattern(matrix);
            },
            nb::arg("matrix"),
            "Initializes the iterative solver for the sparsity pattern of "
            "the "
            "matrix A for further solving Ax=b problems.",
            nb::rv_policy::reference)
        .def(
            "factorize",
            [](Solver &c, MatrixType const &matrix) -> Solver & {
              return c.factorize(matrix);
            },
            nb::arg("matrix"),
            "Initializes the iterative solver with the numerical values of "
            "the "
            "matrix A for further solving Ax=b problems.",
            nb::rv_policy::reference)
        .def(
            "compute",
            [](Solver &c, MatrixType const &matrix) -> Solver & {
              return c.compute(matrix);
            },
            nb::arg("matrix"),
            "Initializes the iterative solver with the matrix A for further "
            "solving Ax=b problems.",
            nb::rv_policy::reference)

        .def("rows", &Solver::rows, "Returns the number of rows.")
        .def("cols", &Solver::cols, "Returns the number of columns.")

        .def("tolerance", &Solver::tolerance,
             "Returns the tolerance threshold used by the stopping criteria.")
        .def("setTolerance", &Solver::setTolerance, nb::arg("tolerance"),
             "Sets the tolerance threshold used by the stopping criteria.\n"
             "This value is used as an upper bound to the relative residual "
             "error: |Ax-b|/|b|.\n"
             "The default value is the machine precision given by "
             "NumTraits<Scalar>::epsilon().",
             nb::rv_policy::reference)
        .def(
            "preconditioner",
            [](Solver &c) -> Preconditioner & { return c.preconditioner(); },
            "Returns a read-write reference to the preconditioner for custom "
            "configuration.",
            nb::rv_policy::reference_internal)

        .def("maxIterations", &Solver::maxIterations,
             "Returns the max number of iterations.\n"
             "It is either the value setted by setMaxIterations or, by "
             "default, twice the number of columns of the matrix.")
        .def("setMaxIterations", &Solver::setMaxIterations,
             nb::arg("max_iterations"),
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

        .def(
            "solve",
            [](Solver const &c, VectorXs const &b) -> VectorXs {
              return solve(c, b);
            },
            nb::arg("b"),
            "Returns the solution x of A x = b using the current "
            "decomposition of A.")
        .def(
            "solve",
            [](Solver const &c, MatrixType const &B) -> MatrixType {
              return solve(c, B);
            },
            nb::arg("B"),
            "Returns the solution X of A X = B using the current "
            "decomposition of A where B is a right hand side matrix.")

        .def(
            "solveWithGuess",
            [](Solver const &c, VectorXs const &b, VectorXs const &x_0)
                -> VectorXs { return solveWithGuess(c, b, x_0); },
            nb::arg("b"), nb::arg("x_0"),
            "Returns the solution x of A x = b using the current "
            "decomposition of A.")
        .def(
            "solveWithGuess",
            [](Solver const &c, MatrixType const &B, MatrixType const &X_0)
                -> MatrixType { return solveWithGuess(c, B, X_0); },
            nb::arg("B"), nb::arg("X_0"),
            "Returns the solution X of A X = B using the current "
            "decomposition of A where B is a right hand side matrix.");
  }

 private:
  template <typename MatrixType, typename MatrixOrVector>
  static MatrixOrVector solve(const Eigen::MINRES<MatrixType> &c,
                              const MatrixOrVector &vec) {
    return c.solve(vec);
  }

  template <typename MatrixType, typename MatrixOrVector1,
            typename MatrixOrVector2>
  static MatrixOrVector1 solveWithGuess(const Eigen::MINRES<MatrixType> &c,
                                        const MatrixOrVector1 &b,
                                        const MatrixOrVector2 &guess) {
    return c.solveWithGuess(b, guess);
  }
};

template <typename MatrixType>
void exposeMINRESSolver(nb::module_ m, const char *name) {
  using Solver = Eigen::MINRES<MatrixType>;
  auto cl =
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

          .def(IterativeSolverBaseVisitor())
          .def(IdVisitor());
}

}  // namespace nanoeigenpy
