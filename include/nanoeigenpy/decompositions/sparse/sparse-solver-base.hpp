/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/SparseCholesky>

namespace nanoeigenpy {

struct SparseSolverBaseVisitor : nb::def_visitor<SparseSolverBaseVisitor> {
  template <typename SimplicialDerived, typename... Ts>
  void execute(nb::class_<SimplicialDerived, Ts...> &cl) {
    using Solver = SimplicialDerived;
    static_assert(std::is_base_of_v<Eigen::SparseSolverBase<Solver>, Solver>);
    using MatrixType = typename SimplicialDerived::MatrixType;
    using Scalar = typename MatrixType::Scalar;
    using DenseVectorXs =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>;
    using DenseMatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                        MatrixType::Options>;

    cl.def(
          "solve",
          [](Solver const &self, DenseVectorXs const &b) -> DenseVectorXs {
            return self.solve(b);
          },
          nb::arg("b"),
          "Returns the solution x of A x = b using the current "
          "decomposition of A.")
        .def(
            "solve",
            [](Solver const &self, DenseMatrixXs const &b) -> DenseMatrixXs {
              return self.solve(b);
            },
            nb::arg("b"),
            "Returns the solution X of A X = B using the current "
            "decomposition of A where B is a right hand side matrix.")
        .def(
            "solve",
            [](Solver const &self, MatrixType const &B) -> MatrixType {
              return self.solve(B);
            },
            nb::arg("B"),
            "Returns the solution X of A X = B using the current "
            "decomposition of A where B is a right hand side matrix.");
  }
};

}  // namespace nanoeigenpy
