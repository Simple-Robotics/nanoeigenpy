/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/svd-base.hpp"
#include <Eigen/SVD>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::BDCSVD<MatrixType> &c,
                     const MatrixOrVector &vec) {
  return c.solve(vec);
}

template <typename _MatrixType>
void exposeBDCSVD(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::BDCSVD<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "Bidiagonal Divide and Conquer SVD. \n\n"
      "This class first reduces the input matrix "
      "to bi-diagonal form using class UpperBidiagonalization, "
      "and then performs a divide-and-conquer diagonalization. "
      "Small blocks are diagonalized using class JacobiSVD. You "
      "can control the switching size with the setSwitchSize() "
      "method, default is 16. For small matrice (<16), it is thus "
      "preferable to directly use JacobiSVD. For larger ones, BDCSVD "
      "is highly recommended and can several order of magnitude faster.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex, unsigned int>(),
           "rows"_a, "cols"_a, "computationOptions"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<Eigen::DenseIndex, Eigen::DenseIndex>(), "rows"_a, "cols"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<const MatrixType &>(), "matrix"_a,
           "Constructs a SVD factorization from a given matrix.")
      .def(nb::init<const MatrixType &, unsigned int>(), "matrix"_a,
           "computationOptions"_a,
           "Constructs a SVD factorization from a given matrix.")

      .def(SVDBaseVisitor())

      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "Computes the SVD of given matrix.",
          nb::rv_policy::reference)
      .def(
          "compute",
          [](Solver &c, MatrixType const &matrix, unsigned int) -> Solver & {
            return c.compute(matrix);
          },
          "matrix"_a, "computationOptions"_a,
          "Computes the SVD of given matrix.", nb::rv_policy::reference)

      .def("setSwitchSize", &Solver::setSwitchSize, "s"_a)

      .def(
          "solve",
          [](Solver const &c, VectorType const &b) -> VectorType {
            return solve(c, b);
          },
          "b"_a,
          "Returns the solution x of A x = b using the current "
          "decomposition of A.")
      .def(
          "solve",
          [](Solver const &c, MatrixType const &B) -> MatrixType {
            return solve(c, B);
          },
          "B"_a,
          "Returns the solution X of A X = B using the current "
          "decomposition of A where B is a right hand side matrix.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
