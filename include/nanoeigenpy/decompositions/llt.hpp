/// Copyright 2025 INRIA
#pragma once
#include "base.hpp"
#include <Eigen/Cholesky>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType>
void exposeLLTSolver(nb::module_ m, const char *name) {
  using Chol = Eigen::LLT<MatrixType>;
  auto cl = nb::class_<Chol>(m, name)
                .def(nb::init<const MatrixType &>(), nb::arg("matrix"))
                .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"))
                .def("matrixL",
                     [](Chol const &c) -> MatrixType { return c.matrixL(); })
                .def("matrixU",
                     [](Chol const &c) -> MatrixType { return c.matrixU(); })
                .def("matrixLLT", &Chol::matrixLLT,
                     nb::rv_policy::reference_internal);
  addEigenBaseFeatures(cl);
}

}  // namespace nanoeigenpy
