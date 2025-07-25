/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"
#include <Eigen/SparseCholesky>

namespace nanoeigenpy {
using namespace nb::literals;

struct SimplicialCholeskyVisitor : nb::def_visitor<SimplicialCholeskyVisitor> {
  template <typename SimplicialDerived, typename... Ts>
  void execute(nb::class_<SimplicialDerived, Ts...> &cl) {
    using Solver = SimplicialDerived;
    using Base = Eigen::SimplicialCholeskyBase<Solver>;
    static_assert(std::is_base_of_v<Base, Solver>);
    using MatrixType = typename SimplicialDerived::MatrixType;
    using RealScalar = typename MatrixType::RealScalar;

    cl.def("analyzePattern", &Solver::analyzePattern,
           "Performs a symbolic decomposition on the sparcity of matrix.\n"
           "This function is particularly useful when solving for several "
           "problems having the same structure.")

        .def(SparseSolverBaseVisitor())

        .def(
            "matrixL",
            [](const Solver &self) -> MatrixType { return self.matrixL(); },
            "Returns the lower triangular matrix L.")
        .def(
            "matrixU",
            [](const Solver &self) -> MatrixType { return self.matrixU(); },
            "Returns the upper triangular matrix U.")

        .def(
            "compute",
            [](Solver &self, const MatrixType &matrix) -> decltype(auto) {
              return self.compute(matrix);
            },
            "matrix"_a,
            "Computes the sparse Cholesky decomposition of a given matrix.",
            nb::rv_policy::reference)

        .def("determinant", &Solver::determinant,
             "Returns the determinant of the underlying matrix from the "
             "current factorization.")

        .def("factorize", &Solver::factorize, "matrix"_a,
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.\n"
             "See also analyzePattern().")

        .def("rows", &Solver::rows)
        .def("cols", &Solver::cols)
        .def("info", &Solver::info,
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setShift", &Solver::setShift, "offset"_a,
             "scale"_a = RealScalar(1),
             "Sets the shift parameters that will be used to adjust the "
             "diagonal coefficients during the numerical factorization.\n"
             "During the numerical factorization, the diagonal coefficients "
             "are transformed by the following linear model: d_ii = offset + "
             "scale * d_ii.\n"
             "The default is the identity transformation with offset=0, and "
             "scale=1.",
             nb::rv_policy::reference)

        .def("permutationP", &Solver::permutationP,
             "Returns the permutation P.", nb::rv_policy::copy)
        .def("permutationPinv", &Solver::permutationPinv,
             "Returns the inverse P^-1 of the permutation P.",
             nb::rv_policy::copy)

        ;
  }
};

}  // namespace nanoeigenpy
