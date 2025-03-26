/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <nanobind/eigen/sparse.h>
#include <Eigen/SparseCholesky>

namespace nanoeigenpy {

/// \addtogroup sparse_solvers
///
/// \brief Base visitor for all sparse matrix solvers.
/// \note The use of `Eigen::Ref` in the first two overloads of \c solve helps
/// disambiguate the dense matrix type and the sparse matrix type.
struct SparseSolverBaseVisitor : nb::def_visitor<SparseSolverBaseVisitor> {
  template <typename SimplicialDerived, typename... Ts>
  void execute(nb::class_<SimplicialDerived, Ts...> &cl) {
    using namespace nb::literals;
    using Solver = SimplicialDerived;
    static_assert(nb::is_base_of_template_v<Solver, Eigen::SparseSolverBase>,
                  "Template type parameter Solver must inherit from "
                  "Eigen::SparseSolverBase!");
    using SparseMatrixType = typename SimplicialDerived::MatrixType;
    using Scalar = typename SparseMatrixType::Scalar;
    static constexpr int Options = SparseMatrixType::Options;
    using DenseVectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
    using DenseMatrixXs =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;

    cl.def(
          "solve",
          [](Solver const &self, const Eigen::Ref<DenseVectorXs const> &b)
              -> DenseVectorXs { return self.solve(b); },
          "b"_a,
          "Returns the solution x of A x = b using the current decomposition "
          "of A, where b is a right hand side vector.")
        .def(
            "solve",
            [](Solver const &self, const Eigen::Ref<DenseMatrixXs const> &B)
                -> DenseMatrixXs { return self.solve(B); },
            "B"_a,
            "Returns the solution X of A X = B using the current decomposition "
            "of A where B is a right hand side matrix.")
        .def(
            "solve",
            [](Solver const &self, const SparseMatrixType &B)
                -> SparseMatrixType { return self.solve(B); },
            "B"_a,
            "Returns the solution X of A X = B using the current decomposition "
            "of A where B is a right hand side matrix.");
  }
};

}  // namespace nanoeigenpy
