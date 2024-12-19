#pragma once
#include "base.hpp"
#include <Eigen/Cholesky>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType>
void exposeLDLTSolver(nb::module_ m, const char *name) {
  using Solver = Eigen::LDLT<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;
  auto cl = nb::class_<Solver>(m, name)
                .def(nb::init<const MatrixType &>(), nb::arg("matrix"))
                .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"))
                .def("matrixL",
                     [](Solver const &c) -> MatrixType { return c.matrixL(); })
                .def("matrixU",
                     [](Solver const &c) -> MatrixType { return c.matrixU(); })
                .def("vectorD",
                     [](Solver const &c) -> VectorType { return c.vectorD(); })
                .def("matrixLDLT", &Solver::matrixLDLT,
                     nb::rv_policy::reference_internal);
  addEigenBaseFeatures(cl);
}

} // namespace nanoeigenpy
