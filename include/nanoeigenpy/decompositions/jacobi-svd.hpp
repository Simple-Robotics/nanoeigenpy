/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/svd-base.hpp"
#include <Eigen/SVD>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename MatrixOrVector, typename JacobiSVD>
MatrixOrVector solve(const JacobiSVD &c, const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename JacobiSVD>
struct JacobiSVDVisitor : nb::def_visitor<JacobiSVDVisitor<JacobiSVD>> {
  using MatrixType = typename JacobiSVD::MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;

  template <typename... Ts>
  void execute(nb::class_<JacobiSVD, Ts...> &cl) {
    using namespace nb::literals;
    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex, unsigned int>(),
             "rows"_a, "cols"_a, "computationOptions"_a = 0,
             "Default constructor with memory preallocation.")
        .def(nb::init<const MatrixType &, unsigned int>(), "matrix"_a,
             "computationOptions"_a = 0,
             "Constructs a SVD factorization from a given matrix.")

        .def(SVDBaseVisitor())

        .def(
            "compute",
            [](JacobiSVD &c, MatrixType const &matrix) -> JacobiSVD & {
              return c.compute(matrix);
            },
            "matrix"_a, "Computes the SVD of given matrix.",
            nb::rv_policy::reference)
        .def(
            "compute",
            [](JacobiSVD &c, MatrixType const &matrix,
               unsigned int) -> JacobiSVD & { return c.compute(matrix); },
            "matrix"_a, "computationOptions"_a,
            "Computes the SVD of given matrix.", nb::rv_policy::reference)

        .def(
            "solve",
            [](JacobiSVD const &c, VectorType const &b) -> VectorType {
              return solve(c, b);
            },
            "b"_a,
            "Returns the solution x of A x = b using the current "
            "decomposition of A.")
        .def(
            "solve",
            [](JacobiSVD const &c, MatrixType const &B) -> MatrixType {
              return solve(c, B);
            },
            "B"_a,
            "Returns the solution X of A X = B using the current "
            "decomposition of A where B is a right hand side matrix.");
  }

  static void expose(nb::module_ &m, const char *name) {
    if (check_registration_alias<JacobiSVD>(m)) {
      return;
    }
    nb::class_<JacobiSVD>(m, name)
        .def(JacobiSVDVisitor<JacobiSVD>())
        .def(IdVisitor());
  }
};

template <typename JacobiSVD>
void exposeJacobiSVD(nb::module_ &m, const char *name) {
  JacobiSVDVisitor<JacobiSVD>::expose(m, name);
}

}  // namespace nanoeigenpy
