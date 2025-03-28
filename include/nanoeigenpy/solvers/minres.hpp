/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include <nanobind/eigen/dense.h>
#include <unsupported/Eigen/IterativeSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

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
  using CtorArg = nb::DMap<const MatrixXs>;

 public:
  template <typename... Ts>
  void execute(nb::class_<Solver, Ts...>& cl) {
    using namespace nb::literals;

    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<CtorArg>(), "A"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")
        .def(IterativeSolverVisitor<Solver>());
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
};

template <typename _MatrixType>
void exposeMINRESSolver(nb::module_& m, const char* name) {
  MINRESSolverVisitor<_MatrixType>::expose(m, name);
}

}  // namespace nanoeigenpy
