/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/id.hpp"
#include "nanoeigenpy/nanoeigenpy.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"

#include <Eigen/AccelerateSupport>

namespace nanoeigenpy {

template <typename AccelerateDerived>
struct AccelerateImplVisitor
    : nb::def_visitor<AccelerateImplVisitor<AccelerateDerived>> {
  using Solver = AccelerateDerived;
  using MatrixType = typename AccelerateDerived::MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using CholMatrixType = MatrixType;
  using StorageIndex = typename MatrixType::StorageIndex;

  template <typename... Ts>
  void execute(nb::class_<Solver, Ts...>& cl) {
    using namespace nb::literals;

    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<const MatrixType&>(), "matrix"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")

        .def("analyzePattern", &Solver::analyzePattern,
             "Performs a symbolic decomposition on the sparcity of matrix.\n"
             "This function is particularly useful when solving for several "
             "problems having the same structure.")

        .def(SparseSolverBaseVisitor())

        .def(
            "compute",
            [](Solver& c, MatrixType const& matrix) -> Solver& {
              return c.compute(matrix);
            },
            nb::arg("matrix"),
            "Computes the sparse Cholesky decomposition of a given matrix.",
            nb::rv_policy::reference)

        .def("factorize", &Solver::factorize, "matrix"_a,
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.\n"
             "See also analyzePattern().")

        .def("info", &Solver::info,
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setOrder", &Solver::setOrder, "Set order");
  }

  static void expose(nb::module_& m, const char* name) {
    nb::class_<Solver>(m, name, "Apple accelerate")
        .def(AccelerateImplVisitor())
        .def(IdVisitor());
  }
};

}  // namespace nanoeigenpy
